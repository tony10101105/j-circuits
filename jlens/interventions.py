# Copyright 2026 Tung-Yu (Tony) Wu
# SPDX-License-Identifier: Apache-2.0
# Added by Tung-Yu (Tony) Wu and his Claude.
"""Causal interventions in J-lens coordinates: steer, ablate, and swap.

The lens *reads* the residual stream (:mod:`jlens.lens`); this module *writes*
it. The paper ("Technical details of J-lens use cases", Writing) defines three
interventions on a residual ``h`` at layer ``l``, all built from J-lens
vectors — the rows ``v_t = (W_U @ J_l)[t]`` of the transported unembedding:

  steer     ``h <- h + alpha * v_t``
  ablate    ``h <- h - (<v_t, h> / ||v_t||^2) * v_t``   (project it out)
  swap      ``h <- h + alpha * V (sigma(c) - c)``, with ``V = [v_s  v_t]``
            and lens coordinates ``c = V^+ h`` (``V^+`` the pseudoinverse of
            ``V``, ``sigma`` exchanges the two entries of ``c``)

Steering with negative ``alpha`` is the paper's other ablation form; the
per-layer ``alpha`` is absolute (no normalisation by ``||h||`` — implementations
that rescale by the residual norm are diverging from the paper's formula).

Two implementation notes, both exact:

- Ablation is applied as ``h - <v_hat, h> * v_hat`` with ``v_hat = v / ||v||``,
  which is identical to the ``/||v||^2`` form above.
- The swap uses an algebraic collapse instead of a runtime pseudoinverse.
  Because ``sigma`` merely exchanges the two coordinates,
  ``sigma(c) - c = (c_t - c_s) * (1, -1)``, hence
  ``V (sigma(c) - c) = (c_t - c_s) * (v_s - v_t)``. With Gram entries
  ``g_ss = <v_s, v_s>``, ``g_st = <v_s, v_t>``, ``g_tt = <v_t, v_t>`` and
  ``det = g_ss * g_tt - g_st^2``, the coordinate difference is a single dot
  product ``c_t - c_s = <u, h>`` where

    ``u = ( -(g_tt + g_st) * v_s + (g_ss + g_st) * v_t ) / det``

  For a full-column-rank ``V`` (the guarded case; near-collinear pairs raise)
  this is ``(V^T V)^{-1} V^T`` written out, so it equals the paper's formula to
  machine precision while costing one dot product and one rank-1 update per
  position. The properties of the paper's operator carry over: the component
  of ``h`` orthogonal to ``span{v_s, v_t}`` is untouched, and at ``alpha = 1``
  the swap is an involution (swapping back restores ``h``).

Interventions are forward hooks on the same blocks
:class:`~jlens.hooks.ActivationRecorder` reads, so a :meth:`JacobianLens.apply
<jlens.lens.JacobianLens.apply>` inside an :class:`Intervention` context reads
the *edited* stream — read and write compose naturally.

The paper's fourth intervention — top-``k`` J-space ablation, suppressing at
each position the ``k`` most active lens vectors while sparing the tokens the
clean model actually outputs there — is :class:`TopKAblation`. It needs a
clean reference pass (for the per-position exclusion sets) and a per-position
vocabulary ranking at hook time, so it binds to one tokenised prompt at
construction rather than being an :class:`Intervention` edit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from jlens.protocol import LensModel

#: Reject a swap pair whose directions are near-collinear: the lens
#: coordinates scale as ``1 / sin^2(angle)``, so below this the swap mostly
#: amplifies noise. ``2e-3`` corresponds to ``|cos| > ~0.999``.
_SWAP_MIN_SIN2 = 2e-3

_ZERO_NORM_EPS = 1e-8


@dataclass(frozen=True)
class Steer:
    """``h <- h + alpha * v_token``. Negative ``alpha`` suppresses.

    ``alpha`` is the paper's absolute scalar strength; a useful magnitude is a
    fraction of the residual norm at the hooked layers (``||h||`` is typically
    10–100x ``||v_t||`` at mid layers, so expect alphas well above 1).
    """

    token: int | str
    alpha: float


@dataclass(frozen=True)
class Ablate:
    """``h <- h - (<v_token, h> / ||v_token||^2) * v_token`` (full removal)."""

    token: int | str


@dataclass(frozen=True)
class Swap:
    """Exchange the ``source`` and ``target`` lens coordinates of ``h``.

    ``h <- h + alpha * V (sigma(c) - c)`` with ``V = [v_source  v_target]``,
    ``c = V^+ h``. ``alpha=2.0`` is the paper's "double strength" swap.
    """

    source: int | str
    target: int | str
    alpha: float = 1.0


Edit = Steer | Ablate | Swap


def _unembed_weight(model: LensModel) -> torch.Tensor:
    weight = getattr(model, "unembed_weight", None)
    if weight is None:
        raise AttributeError(
            f"{type(model).__name__} does not expose unembed_weight "
            "([vocab_size, d_model]); interventions need the raw unembedding "
            "rows to build J-lens vectors"
        )
    return weight


def resolve_token_id(model: LensModel, token: int | str) -> int:
    """Map ``token`` to a vocabulary id.

    An ``int`` is validated against the unembedding's vocab size and returned.
    A ``str`` must encode to exactly one token (HF-style
    ``tokenizer.encode(..., add_special_tokens=False)``); a leading space is
    usually significant (``" spider"`` vs ``"spider"`` are different tokens).
    """
    vocab_size = _unembed_weight(model).shape[0]
    if isinstance(token, int):
        if not 0 <= token < vocab_size:
            raise ValueError(f"token id {token} out of range for vocab {vocab_size}")
        return token
    tokenizer = model.tokenizer
    if not hasattr(tokenizer, "encode"):
        raise TypeError(
            "tokenizer has no encode(); pass token ids as ints instead of strings"
        )
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        pieces = [tokenizer.decode([i]) for i in ids]
        raise ValueError(
            f"{token!r} tokenizes to {len(ids)} pieces {pieces}; interventions "
            "need a single token (a leading space often fixes this)"
        )
    return int(ids[0])


def lens_vector(
    lens: JacobianLens,
    model: LensModel,
    token: int | str,
    layer: int,
    *,
    use_jacobian: bool = True,
) -> torch.Tensor:
    """The J-lens vector ``v_t`` at ``layer``: row ``t`` of ``W_U @ J_l``.

    This is the layer-``l`` residual direction whose lens readout is ``token``
    (``(W_U J_l)[t] = J_l^T w_t``, with ``w_t`` the raw unembedding row; the
    final norm is not folded in, matching the paper's definition). With
    ``use_jacobian=False`` returns the plain unembedding row ``w_t``
    (logit-lens baseline, ``J_l = I``).

    Returns:
        A float32 CPU tensor of shape ``[d_model]``.

    Raises:
        ValueError: If ``use_jacobian`` and ``layer`` is not fitted.
    """
    token_id = resolve_token_id(model, token)
    w = _unembed_weight(model)[token_id].detach().float().cpu()
    if not use_jacobian:
        return w
    J = lens.jacobians.get(layer)
    if J is None:
        raise ValueError(
            f"layer {layer} not in source_layers; fitted layers are "
            f"{lens.source_layers}"
        )
    return w @ J  # row of W_U @ J_l, i.e. J_l^T w_t


def _validate_layers(
    lens: JacobianLens,
    model: LensModel,
    layers: Sequence[int] | None,
    use_jacobian: bool,
) -> list[int]:
    """Layers to hook, validated against the model and (if used) the lens."""
    if layers is None:
        layers = lens.source_layers
    out_of_range = sorted(l for l in set(layers) if not 0 <= l < model.n_layers)
    if out_of_range:
        raise ValueError(
            f"layers {out_of_range} out of range for a {model.n_layers}-layer model"
        )
    unknown = set(layers) - set(lens.source_layers)
    if use_jacobian and unknown:
        raise ValueError(
            f"layers {sorted(unknown)} not in source_layers; "
            f"fitted layers are {lens.source_layers}"
        )
    return sorted(set(layers))


def _steer_entry(v: torch.Tensor, alpha: float) -> tuple:
    return ("steer", alpha * v)


def _ablate_entry(v: torch.Tensor) -> tuple:
    return ("ablate", v / v.norm())


def _swap_entry(
    v_s: torch.Tensor, v_t: torch.Tensor, alpha: float, label: str
) -> tuple:
    g_ss, g_tt, g_st = v_s @ v_s, v_t @ v_t, v_s @ v_t
    det = g_ss * g_tt - g_st * g_st
    sin2 = det / (g_ss * g_tt)
    if sin2 < _SWAP_MIN_SIN2:
        cos = (g_st / torch.sqrt(g_ss * g_tt)).item()
        raise ValueError(
            f"swap directions for {label} are near-collinear (cos = {cos:.5f}); "
            "the lens coordinates are ill-conditioned there"
        )
    u = (-(g_tt + g_st) * v_s + (g_ss + g_st) * v_t) / det
    return ("swap", u, alpha * (v_s - v_t))


class _ResidualHooks:
    """Shared hook plumbing for residual-stream rewrites.

    Subclasses fill ``self._entries`` (``{layer: payload}``) and implement
    :meth:`_edit_positions`; this class registers one forward hook per layer on
    ``__enter__``, routes the block's output residual through the edit at the
    selected positions (edit math in float32 regardless of model dtype), and
    removes the hooks on ``__exit__``.
    """

    def __init__(
        self,
        model: LensModel,
        positions: Sequence[int] | None,
        skip_bos: bool,
    ) -> None:
        self._model = model
        self._positions = None if positions is None else [int(p) for p in positions]
        self._skip_bos = skip_bos
        self._entries: dict[int, object] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _edit_positions(
        self,
        work: torch.Tensor,
        index: torch.Tensor | None,
        seq_len: int,
        layer: int,
        payload,
    ) -> torch.Tensor:
        """Edit ``work`` (``[..., n_positions, d_model]``, float32) in place of
        the selected positions. ``index`` holds their absolute positions
        (``None`` means all of ``0..seq_len-1``)."""
        raise NotImplementedError

    def _position_index(self, seq_len: int, device) -> torch.Tensor | None:
        """Positions to edit as an index tensor; ``None`` means all."""
        if self._positions is None:
            if not self._skip_bos:
                return None
            return torch.arange(1, seq_len, device=device)
        resolved = []
        for p in self._positions:
            q = p + seq_len if p < 0 else p
            if not 0 <= q < seq_len:
                raise IndexError(f"position {p} out of range for seq_len {seq_len}")
            resolved.append(q)
        return torch.tensor(resolved, dtype=torch.long, device=device)

    def _edit_tensor(self, tensor: torch.Tensor, layer: int, payload) -> torch.Tensor:
        seq_len = tensor.shape[-2]
        index = self._position_index(seq_len, tensor.device)
        sub = tensor if index is None else tensor.index_select(-2, index)
        edited = self._edit_positions(sub.float(), index, seq_len, layer, payload)
        edited = edited.to(tensor.dtype)
        if index is None:
            return edited
        out = tensor.clone()
        out.index_copy_(-2, index, edited)
        return out

    def _make_hook(self, layer: int, payload):
        def hook(module, inputs, output):
            # Some HF blocks return a tuple (hidden, present_kv, ...).
            tensor = output if torch.is_tensor(output) else output[0]
            edited = self._edit_tensor(tensor, layer, payload)
            if torch.is_tensor(output):
                return edited
            return (edited, *tuple(output[1:]))

        return hook

    def __enter__(self):
        try:
            for layer, payload in self._entries.items():
                self._handles.append(
                    self._model.layers[layer].register_forward_hook(
                        self._make_hook(layer, payload)
                    )
                )
        except Exception:
            for handle in self._handles:
                handle.remove()
            self._handles = []
            raise
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


class Intervention(_ResidualHooks):
    """Applies J-lens edits to the residual stream while the context is open.

    Registers a forward hook on each requested block on ``__enter__`` (the same
    blocks :class:`~jlens.hooks.ActivationRecorder` reads) and removes them on
    ``__exit__``. Each hook rewrites the block's output residual with the edits
    in list order; everything downstream — later blocks, the model's own
    output, and any lens readout taken inside the context — sees the edited
    stream.

    Args:
        lens: The fitted lens providing ``J_l`` for the edit directions.
        model: The model to intervene on.
        edits: :class:`Steer` / :class:`Ablate` / :class:`Swap` instances,
            applied in order at every hooked layer.
        layers: Layers to hook. Defaults to all of ``lens.source_layers``.
            Must be a subset of ``source_layers`` when ``use_jacobian``.
        positions: Token positions to edit (Python indexing; negative indices
            count from the end of the *current* sequence, so ``-1`` tracks the
            newest token during generation). ``None`` edits every position.
        use_jacobian: If ``False``, use plain unembedding rows as directions
            (logit-lens baseline, ``J_l = I``).
        skip_bos: When ``positions`` is ``None``, leave position 0 unedited
            (default). The attention-sink BOS residual has an outsized norm, so
            ``h``-dependent edits (ablate/swap) there inject spuriously large
            deltas. Set ``False`` for the literal all-positions form.

    Raises:
        ValueError: On out-of-range/unfitted layers, unresolvable or
            multi-piece tokens, a zero-norm direction (the token's unembedding
            row vanishes under ``J_l``), or a near-collinear swap pair.
    """

    def __init__(
        self,
        lens: JacobianLens,
        model: LensModel,
        edits: Sequence[Edit],
        *,
        layers: Sequence[int] | None = None,
        positions: Sequence[int] | None = None,
        use_jacobian: bool = True,
        skip_bos: bool = True,
    ) -> None:
        if not edits:
            raise ValueError("no edits given")
        layers = _validate_layers(lens, model, layers, use_jacobian)
        super().__init__(model, positions, skip_bos)

        def direction(token: int | str, layer: int) -> torch.Tensor:
            v = lens_vector(lens, model, token, layer, use_jacobian=use_jacobian)
            if v.norm() < _ZERO_NORM_EPS:
                raise ValueError(
                    f"direction for token {token!r} vanishes at layer {layer}"
                )
            return v

        # Per-layer edit entries, precomputed once (float32, CPU): resolution
        # and conditioning errors surface here, not mid-forward.
        for layer in layers:
            entries: list[tuple] = []
            for edit in edits:
                if isinstance(edit, Steer):
                    entries.append(
                        _steer_entry(direction(edit.token, layer), edit.alpha)
                    )
                elif isinstance(edit, Ablate):
                    entries.append(_ablate_entry(direction(edit.token, layer)))
                elif isinstance(edit, Swap):
                    entries.append(
                        _swap_entry(
                            direction(edit.source, layer),
                            direction(edit.target, layer),
                            edit.alpha,
                            label=f"{edit.source!r} -> {edit.target!r} at layer {layer}",
                        )
                    )
                else:
                    raise TypeError(f"unknown edit type {type(edit).__name__}")
            self._entries[layer] = entries

    def _edit_positions(self, work, index, seq_len, layer, entries) -> torch.Tensor:
        for entry in entries:
            kind, vec = entry[0], entry[1].to(work.device)
            if kind == "steer":
                work = work + vec
            else:  # ablate / swap: coefficient <vec, h> per position
                coeff = (work * vec).sum(-1, keepdim=True)
                if kind == "ablate":
                    work = work - coeff * vec
                else:
                    work = work + coeff * entry[2].to(work.device)
        return work


class TopKAblation(_ResidualHooks):
    """The paper's top-``k`` J-space ablation, bound to one tokenised prompt.

    At every hooked layer and selected position, project the residual onto the
    orthogonal complement of the span of the ``k`` currently most active J-lens
    vectors — excluding the vectors of the tokens the *clean* model actually
    outputs at that position (its top-``exclude_top`` next-token predictions),
    so the edit suppresses active-but-unspoken readouts rather than the model's
    own answer.

    Construction runs the clean reference pass on ``prompt`` (no hooks
    registered yet) and stores the per-position exclusion sets. At hook time,
    per position, the vocabulary is ranked by the lens readout of the *current*
    residual — ``unembed(J_l @ h)``, the same quantity :meth:`JacobianLens.apply
    <jlens.lens.JacobianLens.apply>` reports — the top ``k`` non-excluded
    tokens are selected, and their lens vectors ``v_t = (W_U @ J_l)[t]`` (as in
    :func:`lens_vector`) are all projected out at once, so every selected
    ``<v_t, h>`` is zeroed even when the vectors are far from orthogonal.
    Near-dependent selections are handled by an SVD rank cutoff instead of
    raising: only the numerically independent directions of the span are
    removed.

    Because the exclusion sets are per absolute position of the clean pass,
    the hooks refuse (``RuntimeError``) any forward whose sequence length
    differs from the reference — construct a new instance per prompt. In
    particular :func:`greedy_generate`, which grows the sequence each step,
    cannot run inside this context.

    Attributes:
        excluded: ``[seq_len, exclude_top]`` clean top-``exclude_top`` output
            token ids per position (CPU).
        selected: After a forward pass, maps each hooked layer to the token ids
            whose lens vectors were suppressed, ``[..., n_positions, k]``
            (CPU); overwritten on every pass.

    Args:
        lens: The fitted lens providing ``J_l``.
        model: The model to intervene on.
        prompt: The prompt the clean reference pass is bound to — text
            (encoded via ``model.encode``) or pre-tokenised ``input_ids`` of
            shape ``[1, seq_len]``.
        k: How many lens vectors to suppress per position (the paper sweeps
            this).
        exclude_top: Size of the per-position exclusion set, the clean model's
            top-``exclude_top`` output tokens (paper value: 10; ``0`` disables
            the exclusion).
        layers: Layers to hook. Defaults to all of ``lens.source_layers``.
        positions: Token positions to edit (Python indexing). ``None`` edits
            every position.
        use_jacobian: If ``False``, rank and ablate with plain unembedding
            rows (logit-lens baseline, ``J_l = I``).
        skip_bos: When ``positions`` is ``None``, leave position 0 unedited
            (default), as in :class:`Intervention`.
        max_seq_len: Truncation length when ``prompt`` is text.

    Raises:
        ValueError: On out-of-range/unfitted layers, ``k < 1``,
            ``exclude_top < 0``, or ``k + exclude_top`` exceeding the vocab.
    """

    #: Relative singular-value cutoff below which a direction of the selected
    #: span is treated as numerically dependent and not projected out.
    _RANK_RCOND = 1e-5

    def __init__(
        self,
        lens: JacobianLens,
        model: LensModel,
        prompt: str | torch.Tensor,
        *,
        k: int,
        exclude_top: int = 10,
        layers: Sequence[int] | None = None,
        positions: Sequence[int] | None = None,
        use_jacobian: bool = True,
        skip_bos: bool = True,
        max_seq_len: int = 512,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if exclude_top < 0:
            raise ValueError(f"exclude_top must be >= 0, got {exclude_top}")
        vocab_size = _unembed_weight(model).shape[0]
        if k + exclude_top > vocab_size:
            raise ValueError(
                f"k={k} + exclude_top={exclude_top} exceeds vocab size {vocab_size}"
            )
        layers = _validate_layers(lens, model, layers, use_jacobian)
        super().__init__(model, positions, skip_bos)
        self._k = k
        self._exclude_top = exclude_top
        for layer in layers:
            self._entries[layer] = lens.jacobians[layer] if use_jacobian else None
        self.selected: dict[int, torch.Tensor] = {}

        # Clean reference pass: the model's own top-exclude_top output tokens
        # at every position. Runs before any hooks are registered.
        if isinstance(prompt, str):
            input_ids = model.encode(prompt, max_length=max_seq_len)
        else:
            input_ids = prompt
        final_layer = model.n_layers - 1
        with torch.no_grad():
            with ActivationRecorder(model.layers, at=[final_layer]) as recorder:
                model.forward(input_ids)
                final = recorder.activations[final_layer][0].detach().float()
            clean_logits = model.unembed(final).float()  # [seq_len, vocab]
        self.excluded: torch.Tensor = clean_logits.topk(
            exclude_top, dim=-1
        ).indices.cpu()

    def _edit_positions(self, work, index, seq_len, layer, J) -> torch.Tensor:
        if seq_len != self.excluded.shape[0]:
            raise RuntimeError(
                f"forward pass has seq_len {seq_len} but the clean reference "
                f"was {self.excluded.shape[0]} tokens; TopKAblation is bound "
                "to one tokenised prompt — construct a new instance per input "
                "(generation grows the sequence, so it cannot run here)"
            )
        device = work.device
        d_model = work.shape[-1]
        flat = work.reshape(-1, d_model)  # [rows, d]; leading dims collapsed
        if J is not None:
            J = J.to(device)

        # Rank the vocabulary by the lens readout of the current residual,
        # mask each position's clean output tokens, and take the top k.
        transported = flat if J is None else flat @ J.T
        lens_logits = self._model.unembed(transported).float().to(device)
        if self._exclude_top:
            pos = torch.arange(seq_len, device=device) if index is None else index
            pos = pos.repeat(flat.shape[0] // pos.shape[0])  # tile over batch
            excluded = self.excluded.to(device)[pos]  # [rows, exclude_top]
            lens_logits.scatter_(-1, excluded, float("-inf"))
        idx = lens_logits.topk(self._k, dim=-1).indices  # [rows, k]

        # Gather the selected lens vectors and project out their span:
        # h <- h - U U^T h with U an orthonormal basis of span{v_t}.
        weight = _unembed_weight(self._model)
        V = weight[idx.to(weight.device)].to(device).float()  # [rows, k, d]
        if J is not None:
            V = V @ J  # rows (W_U @ J_l)[t], as in lens_vector
        basis, sv, _ = torch.linalg.svd(V.transpose(-2, -1), full_matrices=False)
        keep = sv > sv[..., :1] * self._RANK_RCOND  # drop dependent directions
        basis = basis * keep.unsqueeze(-2)  # [rows, d, k]
        coords = torch.einsum("rdk,rd->rk", basis, flat)
        flat = flat - torch.einsum("rdk,rk->rd", basis, coords)

        self.selected[layer] = idx.reshape(*work.shape[:-1], self._k).cpu()
        return flat.reshape(work.shape)


@torch.no_grad()
def greedy_generate(
    model: LensModel,
    prompt: str,
    *,
    max_new_tokens: int = 32,
    max_seq_len: int = 512,
) -> str:
    """Greedy continuation of ``prompt``; returns the generated text only.

    Re-runs the full sequence each step (the :class:`~jlens.protocol.LensModel`
    protocol has no KV cache) and reads next-token logits from the final block
    via :class:`~jlens.hooks.ActivationRecorder` + ``unembed``. Inside an
    :class:`Intervention` context this re-applies the edits at *every* position
    on every step — the "clamped at every position" regime of the paper's swap
    experiments (a KV-cached decoder could not re-edit already-computed
    positions). Stops early at the tokenizer's ``eos_token_id``, if any.
    """
    input_ids = model.encode(prompt, max_length=max_seq_len)
    eos = getattr(model.tokenizer, "eos_token_id", None)
    final_layer = model.n_layers - 1
    generated: list[int] = []
    for _ in range(max_new_tokens):
        with ActivationRecorder(model.layers, at=[final_layer]) as recorder:
            model.forward(input_ids)
            last = recorder.activations[final_layer][:, -1]
        next_id = int(model.unembed(last.float())[0].argmax())
        if eos is not None and next_id == eos:
            break
        generated.append(next_id)
        step = torch.tensor([[next_id]], device=input_ids.device)
        input_ids = torch.cat([input_ids, step], dim=1)
    return model.tokenizer.decode(generated)
