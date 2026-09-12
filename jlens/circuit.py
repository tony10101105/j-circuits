# Copyright 2026 Tung-Yu (Tony) Wu
# SPDX-License-Identifier: Apache-2.0
# Added by Tung-Yu (Tony) Wu and his Claude.
"""J-circuit: Tracing Chains of Internal Thought in Language Models.

A J-circuit is a layered DAG whose nodes are *concepts at token positions* —
the top-``k`` J-lens readouts of one position at one layer, plus an optional
error node standing for what those concepts cannot express — and whose edges
score how much a source concept supports a concept one level above it. Scores
are first-order attribution (EAP, or path-integrated EAP-IG) against the
rank-1 residual step that ablating a source concept would produce, computed a
level pair at a time with one batched VJP, then pruned top-down from its
roots to the strongest ``prune_percent`` of each level. :func:`build_jcircuit`
is the entry point and
:class:`JCircuit` is the result, carrying the nodes, edges, and the readout
and diagnostic helpers.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field

import torch

from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from jlens.protocol import LensModel
from jlens.pursuit import pursue_lens

logger = logging.getLogger(__name__)

#: Perturbed forward rows batched at once under ``estimator="ig"``. Each row is
#: one ``(source concept, path sample)`` pair and carries a full sequence, so
#: this is the memory knob; it is capped by ``vjp_chunk``.
IG_ROW_CHUNK = 8

_ZERO_NORM_EPS = 1e-8

#: Token id carried by an error node. Negative so it can never collide with a
#: real vocabulary entry.
ERROR_TOKEN_ID = -1

#: Display text for an error node.
ERROR_TOKEN = "<error>"

#: Singular values below this multiple of the largest are treated as outside the
#: span when projecting onto a block's concept directions.
_SPAN_RANK_TOL = 1e-6


@dataclass(frozen=True)
class Node:
    """One concept: the lens readout ``token`` at ``(layer, position)``.

    Attributes:
        layer: Residual-block index.
        position: Token position the concept was read at.
        token_id: Vocabulary id of the concept.
        token: Its decoded string.
        activation: The J-space coordinate ``a = <v_hat, h>`` (clean pass).
            This is what an ablation of the concept removes, so it — not
            :attr:`coefficient` — is what edge scores are built from.
        lens_rank: Order in which selection picked this concept (0 = first):
            the readout rank under ``"topk"``, the pursuit iteration under
            ``"pursuit"``. Under ``"topk"`` this is not quite the order of
            :attr:`activation`, because the readout applies the model's final
            norm and the lens vectors do not.
        coefficient: Weight this concept carries in the sparse reconstruction
            of the residual, under ``"pursuit"`` selection; ``None`` under
            ``"topk"``. It differs from :attr:`activation`: the activation
            counts overlap with every other concept present, the coefficient
            only what this concept explains that the others do not.
    """

    layer: int
    position: int
    token_id: int
    token: str
    activation: float
    lens_rank: int
    coefficient: float | None = None

    @property
    def is_error(self) -> bool:
        """Whether this is the block's error node rather than a named concept.

        An error node carries what the block's ``k`` concepts cannot express —
        the component of the residual orthogonal to all of them. It has no
        vocabulary entry, so it cannot be steered, swapped, or looked up; it can
        only be read as "this much of the residual has no name here".
        """
        return self.token_id == ERROR_TOKEN_ID

    def __str__(self) -> str:
        return f"{self.token!r}@L{self.layer}:{self.position}"


@dataclass(frozen=True)
class Edge:
    """A scored source -> target concept edge between consecutive levels.

    Attributes:
        source: Concept at ``target.layer - stride``, at a position ``<=`` the
            target's (the residual stream for equal positions, attention for
            earlier ones).
        target: Concept at ``source.layer + stride``.
        score: EAP attribution ``A`` — the estimated drop in the target's
            readout if the source's coordinate were ablated.
        identity: The share of ``score`` explained by the residual stream
            carrying the source forward unchanged, ``a_s * <v_s, v_t>``. Zero
            for cross-position edges, which have no residual path.
    """

    source: Node
    target: Node
    score: float
    identity: float

    @property
    def computed(self) -> float:
        """``score - identity``: the part attributable to the block's compute."""
        return self.score - self.identity

    @property
    def cross_position(self) -> bool:
        """Whether this edge crosses positions, i.e. rides attention."""
        return self.source.position != self.target.position

    def __str__(self) -> str:
        return f"{self.source} -> {self.target}  A={self.score:+.4f}"


def _level_budget(n_dense_edges: int, percent: float) -> int:
    """How many edges one level pair may keep.

    ``percent`` of the level's *dense* edge count — the level as it was built,
    counting every named target — not of the subset that is still reachable
    part-way down the descent. Restricting ``roots``, or losing targets higher
    up, therefore spends the same allowance on a narrower cone rather than
    compounding the reduction at every level.

    At least one edge, so a level pair never vanishes outright.
    """
    return max(1, round(n_dense_edges * percent / 100.0))


def _keep_top_n(edges: list[Edge], n_keep: int) -> list[Edge]:
    """The ``n_keep`` strongest of ``edges`` by ``|score|``, or all of them.

    Shared by :func:`build_jcircuit` and :meth:`JCircuit.prune`. Survivors keep their input
    order, so the result is deterministic and ties break by position in
    ``edges``. Fewer than ``n_keep`` edges means all of them survive.
    """
    if not edges or n_keep <= 0:
        return []
    order = sorted(range(len(edges)), key=lambda i: -abs(edges[i].score))
    return [edges[i] for i in sorted(order[:n_keep])]


def _root_positions(
    roots: str | Sequence[int] | None, positions: Sequence[int]
) -> list[int] | None:
    """The top-layer positions the circuit is read out at.

    ``"all"`` (or ``None``) imposes no restriction and returns ``None``;
    ``"last"`` selects the final position; a sequence names them explicitly.
    ``positions`` is the set the circuit actually carries concepts at.
    """
    if roots is None or roots == "all":
        return None
    available = list(positions)
    if roots == "last":
        return [available[-1]]
    if isinstance(roots, str):
        raise ValueError(
            f"roots must be 'all', 'last', or a sequence of positions, got {roots!r}"
        )
    try:
        wanted = sorted({int(p) for p in roots})
    except (TypeError, ValueError):
        raise ValueError(
            f"root positions must be integers here, got {list(roots)!r}; token text "
            "is only resolvable in build_jcircuit, which has the tokenizer"
        ) from None
    if not wanted:
        raise ValueError("roots is empty")
    unknown = [p for p in wanted if p not in set(available)]
    if unknown:
        raise ValueError(
            f"root positions {unknown} carry no concepts; available: {available}"
        )
    return wanted


def _prune_levels(
    nodes: dict[int, list[Node]],
    edges_by_layer: dict[int, list[Edge]],
    layer_top: int,
    layer_bottom: int,
    percent: float,
    roots: list[int] = None,
    stride: int = 1,
) -> tuple[dict[int, list[Node]], list[Edge]]:
    """Top-down prune: keep ``percent`` of each dense level, drop dead sources.

    ``edges_by_layer[l]`` holds the edges into layer ``l``. ``roots``, when
    given, restricts the layer-``layer_top`` nodes the descent starts from, so
    the budget is spent on the cone feeding those positions. Returns the
    surviving ``(nodes, edges)``; iteration stops early if a level pair leaves
    no source alive.
    """
    all_live = [n for n in nodes[layer_top] if not n.is_error]
    live = (
        all_live if roots is None else [n for n in all_live if n.position in set(roots)]
    )
    if not live:
        raise ValueError(
            f"no concept at layer {layer_top} sits at a root position "
            f"{sorted(set(roots))}"
        )
    kept_nodes = {layer_top: live}
    kept_edges: list[Edge] = []
    for layer in range(layer_top, layer_bottom, -stride):
        live_set = set(live)
        all_level = edges_by_layer.get(layer, [])
        budget = _level_budget(len(all_level), percent)
        level = [e for e in all_level if e.target in live_set]
        keep = _keep_top_n(level, budget)
        if not keep:
            break
        kept_edges.extend(keep)
        survivors = {e.source for e in keep}
        kept = [n for n in nodes[layer - stride] if n in survivors]
        if not kept:
            break
        kept_nodes[layer - stride] = kept
        # Error nodes stay in the graph as leaves but never become targets, so
        # they cannot seed the next descent — the same rule the build
        # follows, or the two paths would disagree.
        live = [n for n in kept if not n.is_error]
        if not live:
            break
    return kept_nodes, kept_edges


@dataclass
class JCircuit:
    """A built J-circuit: concepts per layer, scored edges, and the settings used.

    Attributes:
        nodes: ``{layer: [Node, ...]}``, ordered by position then lens rank.
            Only concepts that survived pruning are present.
        edges: All scored edges, ordered top layer pair first.
        hparams: The settings the circuit was built with (see
            :func:`build_jcircuit`), plus ``n_levels`` and ``positions_resolved``.
    """

    nodes: dict[int, list[Node]]
    edges: list[Edge]
    hparams: dict = field(default_factory=dict)
    error_directions: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)

    def direction(
        self, node: Node, lens: JacobianLens, model: LensModel
    ) -> torch.Tensor:
        """The unit residual direction of the node."""
        if not node.is_error:
            from jlens.interventions import lens_vector

            v = lens_vector(lens, model, node.token_id, node.layer)
            return v / v.norm()
        return self.error_directions[(node.layer, node.position)]

    @property
    def layers(self) -> list[int]:
        """Layers carrying at least one concept, deepest first."""
        return sorted(self.nodes, reverse=True)

    @property
    def layer_top(self) -> int:
        return int(self.hparams["layer_top"])

    @property
    def layer_bottom(self) -> int:
        return int(self.hparams["layer_bottom"])

    @property
    def stride(self) -> int:
        """Layer gap between consecutive levels."""
        return int(self.hparams.get("stride", 1))

    @property
    def positions(self) -> list[int]:
        """Positions carrying at least one concept, in order."""
        return sorted({n.position for v in self.nodes.values() for n in v})

    def edges_into(self, layer: int) -> list[Edge]:
        """Edges whose target sits at ``layer``."""
        return [e for e in self.edges if e.target.layer == layer]

    def nodes_at(self, layer: int, position: int) -> list[Node]:
        """The concepts of one ``(layer, position)`` block, by lens rank."""
        return [n for n in self.nodes.get(layer, []) if n.position == position]

    def __repr__(self) -> str:
        return (
            f"JCircuit(layers={self.layer_top}..{self.layer_bottom}, "
            f"k={self.hparams.get('k')}, positions={len(self.positions)}, "
            f"nodes={sum(len(v) for v in self.nodes.values())}, edges={len(self.edges)})"
        )

    def prune(
        self, percent: float = 20.0, *, roots: str | Sequence[int] | None = "last"
    ) -> JCircuit:
        """Return the circuit obtained by pruning this one further.

        Applies the same top-down rule :func:`build_jcircuit` uses, so
        ``build(prune_percent=100, roots="all").prune(p, roots=r)`` and
        ``build(prune_percent=p, roots=r)`` agree.

        Args:
            percent: Percentage of each layer pair's *dense* edge count to
                keep, by ``|score|``. The budget ignores how much of the level
                is still reachable, so a narrow cone keeps whichever is smaller
                — its allowance, or every edge it has. At least one edge per
                pair always survives.
            roots: Which top-layer concepts count as the circuit's output, and
                therefore where the descent starts. ``"last"`` keeps
                only the final position, ``"all"`` every position, or name
                positions explicitly. Token text is not accepted here — this
                method has no tokenizer; pass it to :func:`build_jcircuit`
                instead.

        Raises:
            ValueError: If ``percent`` is outside ``(0, 100]``, or ``roots``
                names a position the circuit does not carry.
        """
        if not 0 < percent <= 100:
            raise ValueError(f"prune_percent must be in (0, 100], got {percent}")
        chosen = _root_positions(roots, self.positions)
        by_layer: dict[int, list[Edge]] = {}
        for edge in self.edges:
            by_layer.setdefault(edge.target.layer, []).append(edge)
        kept_nodes, kept_edges = _prune_levels(
            self.nodes,
            by_layer,
            self.layer_top,
            self.layer_bottom,
            percent,
            chosen,
            self.stride,
        )
        hparams = dict(self.hparams)
        hparams["prune_percent"] = percent
        hparams["roots"] = chosen
        return JCircuit(
            nodes=kept_nodes,
            edges=kept_edges,
            hparams=hparams,
            error_directions=dict(self.error_directions),
        )

    def tokens(self) -> list[tuple[int, str]]:
        """Every distinct named ``(token_id, token)`` the circuit contains, sorted.

        The input to a token filter: one entry per concept regardless of how
        many ``(layer, position)`` blocks it appears in.

        Error nodes are excluded. They are structural, not lexical: their id is
        :data:`ERROR_TOKEN_ID` and their text is the literal ``"<error>"``, which
        a semantic filter reads as markup and removes on sight — deleting every
        error node in the graph at once, whatever the caller asked
        ``error_nodes`` for. Whether the graph carries them is a build-time
        choice, so it is not one a filter over vocabulary gets to revisit.
        """
        seen = {
            (n.token_id, n.token)
            for v in self.nodes.values()
            for n in v
            if not n.is_error
        }
        return sorted(seen, key=lambda pair: pair[1])

    def drop_tokens(self, token_ids: Sequence[int]) -> JCircuit:
        """Return this circuit with every concept in ``token_ids`` removed.

        Removes the named concepts at every layer and position, the edges
        touching them, and then — repeatedly — anything left dangling. The
        circuit's own top and bottom levels are the boundary: everything below
        the top needs a surviving incoming edge (otherwise it is supported only
        by what was dropped), and everything above the bottom needs a surviving
        outgoing edge (otherwise it can no longer influence anything). Those two
        levels are exempt from the requirement they cannot meet — the top layer
        never has outgoing edges, the bottom never has incoming ones.

        The boundary is fixed at the levels this circuit started with, so
        emptying the top level cascades the whole graph away rather than
        promoting the next level down to be the new root.

        Args:
            token_ids: Vocabulary ids to remove. Ids the circuit does not carry
                are ignored.

        Returns:
            A new :class:`JCircuit`; this one is unchanged. Its ``hparams``
            record the ids under ``dropped_token_ids``.
        """
        drop = {int(t) for t in token_ids}
        present = self.layers  # deepest first; the graph's own boundary
        top, bottom = present[0], present[-1]
        nodes = {
            layer: [n for n in v if n.token_id not in drop]
            for layer, v in self.nodes.items()
        }
        alive = {n for v in nodes.values() for n in v}
        edges = [e for e in self.edges if e.source in alive and e.target in alive]

        # Fixpoint: dropping a node can orphan its neighbours, which can orphan
        # theirs. Recompute degrees each round rather than assuming one pass.
        while True:
            incoming = {e.target for e in edges}
            outgoing = {e.source for e in edges}
            keep = {
                n
                for n in alive
                # Error nodes are sources only and legitimately have no incoming
                # edges at any layer, so the incoming test would orphan every one
                # of them on the first pass.
                if (n.layer == bottom or n.is_error or n in incoming)
                and (n.layer == top or n in outgoing)
            }
            if keep == alive:
                break
            alive = keep
            edges = [e for e in edges if e.source in alive and e.target in alive]

        hparams = dict(self.hparams)
        hparams["dropped_token_ids"] = sorted(drop)
        return JCircuit(
            nodes={
                layer: [n for n in v if n in alive]
                for layer, v in nodes.items()
                if any(n in alive for n in v)
            },
            edges=edges,
            hparams=hparams,
            error_directions=dict(self.error_directions),
        )

    def format(self, *, max_edges_per_level: int = 8) -> str:
        """A readable layer-by-layer rendering of the circuit."""
        lines = [repr(self)]
        for layer in self.layers:
            lines.append(f"\nL{layer}")
            for position in sorted({n.position for n in self.nodes[layer]}):
                concepts = "  ".join(
                    f"{n.token!r}(a={n.activation:+.1f})"
                    for n in self.nodes_at(layer, position)
                )
                lines.append(f"  pos {position:>3}: {concepts}")
            level = sorted(self.edges_into(layer), key=lambda e: -abs(e.score))
            for edge in level[:max_edges_per_level]:
                kind = "attn" if edge.cross_position else "resid"
                lines.append(
                    f"    {edge.source.token!r:>12}@{edge.source.position:<3}"
                    f" -> {edge.target.token!r:<12}@{edge.target.position:<3}"
                    f" A={edge.score:+9.4f} compute={edge.computed:+8.4f} [{kind}]"
                )
            if len(level) > max_edges_per_level:
                lines.append(f"    ... {len(level) - max_edges_per_level} more")
        return "\n".join(lines)


def _percentile_layer(source_layers: list[int], percentile: float, name: str) -> int:
    """The fitted layer at ``percentile`` of the fitted-layer list."""
    if not 0 <= percentile <= 100:
        raise ValueError(f"{name} must be in [0, 100], got {percentile}")
    index = round(percentile / 100.0 * (len(source_layers) - 1))
    return source_layers[index]


def _resolve_positions(
    positions: Sequence[int | str] | None,
    input_ids: torch.Tensor,
    skip_bos: bool,
    tokenizer: object,
) -> list[int]:
    """Resolve numeric indices or decoded token text to sorted positions."""
    seq_len = int(input_ids.shape[1])
    if positions is None:
        resolved = list(range(1 if skip_bos else 0, seq_len))
        if not resolved:
            raise ValueError(f"nothing left to build on: seq_len={seq_len}, skip_bos")
        return resolved
    out = set()
    token_ids = input_ids[0].tolist()
    for position in positions:
        if isinstance(position, str):
            try:
                p = int(position)
            except ValueError:
                query = position.strip()
                matches = []
                for start in range(seq_len):
                    decoded = ""
                    for stop in range(start + 1, seq_len + 1):
                        decoded += tokenizer.decode([token_ids[stop - 1]])
                        if decoded.strip() == query:
                            matches.extend(range(start, stop))
                            break
                if not matches:
                    raise ValueError(
                        f"position {position!r} did not match a decoded input token"
                    ) from None
                out.update(matches)
                continue
        else:
            p = position
        q = p + seq_len if p < 0 else p
        if not 0 <= q < seq_len:
            raise ValueError(f"position {p} out of range for seq_len {seq_len}")
        out.add(q)
    if not out:
        raise ValueError("positions is empty")
    return sorted(out)


def _select_concepts(
    lens: JacobianLens,
    model: LensModel,
    layer: int,
    residuals: torch.Tensor,
    k: int,
    selection: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Choose ``k`` concepts per position and return ids, directions, weights.

    Shapes are ``[n_positions, k]``, ``[n_positions, k, d_model]`` and
    ``[n_positions, k]``. A token id of ``-1`` marks an empty slot. The third
    return is the pursuit coefficient under ``"pursuit"`` and ``NaN`` under
    ``"topk"``, which has no such quantity.
    """
    device = residuals.device
    J = lens.jacobians[layer].to(device)
    if selection == "pursuit":
        ids, coefficients, directions = pursue_lens(
            model.unembed_weight, J, residuals, k
        )
        return ids, directions, coefficients

    readout = model.unembed(residuals @ J.T).float()  # [n_pos, vocab]
    ids = readout.topk(k, dim=-1).indices  # [n_pos, k]
    # Rows of W_U @ J_l, i.e. lens_vector(...) for each id, batched on device.
    # Detached: directions are fixed geometry, never differentiated through.
    vectors = model.unembed_weight[ids].detach().to(device).float() @ J
    norms = vectors.norm(dim=-1)  # [n_pos, k]
    directions = vectors / norms.clamp_min(_ZERO_NORM_EPS).unsqueeze(-1)
    # A vanishing direction cannot be normalised, so drop that slot.
    ids = ids.masked_fill(norms < _ZERO_NORM_EPS, -1)
    return ids, directions, torch.full_like(norms, float("nan"))


def _unexplained(
    residual: torch.Tensor, directions: torch.Tensor
) -> tuple[torch.Tensor, float] | None:
    """The part of ``residual`` no direction in ``directions`` can express.

    Returns ``(unit direction, magnitude)`` of ``r = h - P h``, with ``P`` the
    orthogonal projector onto the span of the block's concepts, or ``None`` when
    nothing measurable is left (the concepts already span ``h``).

    Why the *orthogonal* residual and not ``h - sum(c_i * v_i)``, the leftover
    the pursuit loop itself stops at:

    - Pursuit's coefficients are clamped non-negative, so its residual is not
      orthogonal to the atoms it selected. An error node built from it would
      re-carry influence the named edges already claim, and the two would double
      count. The orthogonal residual is the unique choice that carries *only*
      what no named concept can.
    - It is defined for ``selection="topk"`` too, which has no coefficients.
    - It makes the split exact: ``<h, g> = <P h, g> + <r, g>``, so the error
      node's edge is precisely the influence the named span cannot reach.

    The two agree whenever pursuit's non-negativity is not binding; the
    orthogonal one is never larger.
    """
    basis = _span_basis(directions)
    leftover = residual - basis @ (basis.T @ residual) if basis.shape[1] else residual
    magnitude = float(leftover.norm())
    if magnitude <= _ZERO_NORM_EPS:
        return None
    return leftover / magnitude, magnitude


def _layer_concepts(
    lens: JacobianLens,
    model: LensModel,
    layer: int,
    residuals: torch.Tensor,
    positions: list[int],
    k: int,
    selection: str,
    error_nodes: bool,
) -> tuple[list[Node], torch.Tensor, torch.Tensor]:
    """The concepts of one layer, flattened over ``positions``.

    ``residuals`` is ``[n_positions, d_model]``, the clean activations at those
    positions. Returns ``(nodes, directions, node_positions)`` where the node
    list is position-major then selection order, and ``directions`` is the
    matching ``[n_nodes, d_model]`` of unit J-lens vectors. Empty slots are
    dropped, so a block may hold fewer than ``k`` nodes and its
    :attr:`Node.lens_rank` values may have gaps — a rank is always the true
    selection order.

    With ``error_nodes``, each block gets one extra node past its named ranks
    carrying the residual those concepts leave unexplained (:func:`_unexplained`).
    """
    ids, directions, weights = _select_concepts(
        lens, model, layer, residuals, k, selection
    )
    activations = torch.einsum("pkd,pd->pk", directions, residuals)

    nodes: list[Node] = []
    kept: list[torch.Tensor] = []
    node_positions: list[int] = []
    for i, position in enumerate(positions):
        for rank in range(ids.shape[1]):
            token_id = int(ids[i, rank])
            if token_id < 0:
                continue
            coefficient = float(weights[i, rank])
            nodes.append(
                Node(
                    layer=layer,
                    position=position,
                    token_id=token_id,
                    token=model.tokenizer.decode([token_id]),
                    activation=float(activations[i, rank]),
                    lens_rank=rank,
                    coefficient=None if coefficient != coefficient else coefficient,
                )
            )
            kept.append(directions[i, rank])
            node_positions.append(position)
        if not error_nodes:
            continue
        block = [d for d, p in zip(kept, node_positions, strict=True) if p == position]
        leftover = _unexplained(residuals[i], torch.stack(block)) if block else None
        if leftover is None:
            continue
        direction, magnitude = leftover
        nodes.append(
            Node(
                layer=layer,
                position=position,
                token_id=ERROR_TOKEN_ID,
                token=ERROR_TOKEN,
                # <v_hat, h> for this direction is exactly ||r||, because r is
                # orthogonal to everything the named concepts explain.
                activation=magnitude,
                # After every named rank, so a block still reads in order.
                lens_rank=ids.shape[1],
            )
        )
        kept.append(direction)
        node_positions.append(position)
    if not nodes:
        raise ValueError(f"no usable lens concept at layer {layer}")
    return (
        nodes,
        torch.stack(kept),
        torch.tensor(node_positions, device=residuals.device),
    )


def _span_basis(directions: torch.Tensor) -> torch.Tensor:
    """An orthonormal basis ``[d, r]`` of the row space of ``directions``.

    Via SVD rather than QR: the concept directions of a block are not
    orthogonal and can be rank-deficient, and QR would hand back columns
    outside the span for the deficient part.
    """
    u, s, _ = torch.linalg.svd(directions.T, full_matrices=False)
    if s.numel() == 0 or float(s[0]) <= _ZERO_NORM_EPS:
        return u[:, :0]
    rank = int((s > float(s[0]) * _SPAN_RANK_TOL).sum())
    return u[:, :rank]


def _score_layer_pair(
    activations: dict[int, torch.Tensor],
    target_layer: int,
    target_dirs: torch.Tensor,
    target_pos: torch.Tensor,
    source_dirs: torch.Tensor,
    source_pos: torch.Tensor,
    source_acts: torch.Tensor,
    stride: int,
    chunk: int,
) -> torch.Tensor:
    """EAP scores ``[n_source, n_target]`` for one level pair ``stride`` apart.

    One batched VJP per chunk of targets. A single backward already carries the
    gradient at *every* source position, so cross-position edges cost nothing
    beyond the bookkeeping: the cotangent for a target places its lens vector at
    that target's position and zero elsewhere, and the result is read off at
    each source position in turn.

    Nothing here depends on the two layers being adjacent: the VJP differentiates
    through however many blocks separate them, giving the total effect over the
    span.

    Returns the ``[n_source, n_target]`` score matrix.
    """
    source_layer = target_layer - stride
    out = activations[target_layer]
    n_target = target_dirs.shape[0]
    scores = torch.zeros(source_dirs.shape[0], n_target, device=source_dirs.device)
    unique_positions = source_pos.unique()

    for start in range(0, n_target, chunk):
        stop = min(start + chunk, n_target)
        size = stop - start
        cotangents = torch.zeros(size, *out.shape, device=out.device, dtype=out.dtype)
        cotangents[torch.arange(size), 0, target_pos[start:stop]] = target_dirs[
            start:stop
        ].to(out.dtype)
        grads = torch.autograd.grad(
            out,
            activations[source_layer],
            grad_outputs=cotangents,
            is_grads_batched=True,
            retain_graph=True,
        )[0][:, 0].float()  # [size, seq, d]
        for position_value in unique_positions:
            mask = source_pos == position_value
            scores[mask, start:stop] = source_dirs[mask] @ grads[:, position_value].T
    return scores * source_acts.unsqueeze(1)


def _score_layer_pair_ig(
    model: LensModel,
    input_ids: torch.Tensor,
    target_layer: int,
    target_dirs: torch.Tensor,
    target_pos: torch.Tensor,
    source_dirs: torch.Tensor,
    source_pos: torch.Tensor,
    source_acts: torch.Tensor,
    stride: int,
    steps: int,
    chunk: int,
) -> torch.Tensor:
    """EAP-IG scores ``[n_source, n_target]`` for one level pair.

    Averages the integrand over ``steps`` midpoints of the ablation path
    ``h - alpha * a_s * v_s``. Unlike :func:`_score_layer_pair` this cannot
    share one backward across sources — the path point differs per source — so
    it runs one perturbed forward per ``(source, alpha)`` pair. Targets are
    still shared: a single batched VJP off each perturbed forward scores every
    target at once, which is what keeps the whole thing within a small multiple
    of EAP rather than a large one.
    """
    source_layer = target_layer - stride
    device = source_dirs.device
    n_source, n_target = source_dirs.shape[0], target_dirs.shape[0]
    alphas = (torch.arange(steps, device=device, dtype=torch.float32) + 0.5) / steps
    scores = torch.zeros(n_source, n_target, device=device)
    positions = source_pos.tolist()
    jobs = [(i, a) for i in range(n_source) for a in range(steps)]

    for start in range(0, len(jobs), chunk):
        batch = jobs[start : start + chunk]
        captured: dict[str, torch.Tensor] = {}

        # batch/captured bound as defaults: the hook must see *this* iteration's
        # rows and write into *this* iteration's slot, not the last loop value.
        def perturb(module, inputs, output, batch=batch, captured=captured):
            tensor = output if torch.is_tensor(output) else output[0]
            work = tensor.float().clone()
            for row, (i, a) in enumerate(batch):
                place = positions[i]
                work[row, place] = (
                    work[row, place] - alphas[a] * source_acts[i] * source_dirs[i]
                )
            # Root the graph at the path point, so the gradient is taken there
            # and not at the clean activation.
            rooted = work.detach().requires_grad_(True)
            captured["h"] = rooted
            edited = rooted.to(tensor.dtype)
            return edited if torch.is_tensor(output) else (edited, *tuple(output[1:]))

        handle = model.layers[source_layer].register_forward_hook(perturb)
        try:
            with (
                torch.enable_grad(),
                ActivationRecorder(model.layers, at=[target_layer]) as rec,
            ):
                model.forward(input_ids.expand(len(batch), -1))
                out = rec.activations[target_layer]
            cotangents = torch.zeros(
                n_target, *out.shape, device=out.device, dtype=out.dtype
            )
            cotangents[torch.arange(n_target), :, target_pos] = target_dirs.to(
                out.dtype
            ).unsqueeze(1)
            grads = torch.autograd.grad(
                out, captured["h"], grad_outputs=cotangents, is_grads_batched=True
            )[0]  # [n_target, len(batch), seq, d]
            for row, (i, _) in enumerate(batch):
                scores[i] += (
                    grads[:, row, positions[i]].float() @ source_dirs[i]
                ) / steps
        finally:
            handle.remove()
    return scores * source_acts.unsqueeze(1)


def build_jcircuit(
    lens: JacobianLens,
    model: LensModel,
    prompt: str | torch.Tensor,
    *,
    k: int = 5,
    selection: str = "pursuit",
    error_nodes: bool = True,
    estimator: str = "eap",
    ig_steps: int = 2,
    stride: int = 1,
    layer_top: int | None = None,
    layer_bottom: int | None = None,
    layer_top_percentile: float = 95.0,
    layer_bottom_percentile: float = 20.0,
    prune_percent: float = 20.0,
    roots: str | Sequence[int | str] | None = "last",
    positions: Sequence[int | str] | None = None,
    skip_bos: bool = True,
    vjp_chunk: int = 32,
    max_seq_len: int = 512,
) -> JCircuit:
    """Build a J-circuit for ``prompt``, top-down from ``layer_top``.

    One forward pass records the residual stream; each level pair then costs one
    batched VJP per ``vjp_chunk`` of target nodes, from which the edge matrix
    falls out as inner products.

    Args:
        lens: The fitted lens supplying ``J_l``.
        model: The model to analyse.
        prompt: Text (encoded via ``model.encode``) or ``input_ids`` of shape
            ``[1, seq_len]``.
        k: Number of concepts per ``(layer, position)`` block.
        selection: How those concepts are chosen. ``"pursuit"`` decomposes the
            residual into a sparse non-negative combination of lens vectors
            (:mod:`jlens.pursuit`), so near-duplicate spellings of one concept
            do not all get picked; ``"topk"`` takes the plain ranked readout.
        estimator: ``"eap"`` (default) evaluates the integrand at the clean
            point; ``"ig"`` averages it over the ablation path. ``"ig"`` costs
            roughly 3x and is markedly more faithful on prompts with instruction
            conflict — see the module docstring for the measurements.
        ig_steps: Path samples under ``estimator="ig"``. 2 is the default and
            measurably as good as 5.
        stride: Layer gap between consecutive levels, ``>= 1``. Only the level
            layers need a fitted Jacobian, and only they carry concepts; a
            larger stride costs proportionally fewer VJPs and hides whatever
            mediates an edge inside its span (see the module docstring).
        layer_top: Layer to start from. Defaults to the fitted layer at
            ``layer_top_percentile``. Lowered to the last level reachable from
            ``layer_bottom`` in whole strides when the band does not divide
            evenly; :attr:`JCircuit.hparams` records what was asked for.
        layer_bottom: Layer to stop at (inclusive), and the layer the stride is
            anchored at. Defaults to the fitted layer at
            ``layer_bottom_percentile``.
        layer_top_percentile: Where in the fitted-layer list the band ends,
            used when ``layer_top`` is not given. The default keeps the top of
            the band just below the last layers, whose readouts collapse onto
            the model's own output.
        layer_bottom_percentile: Where the band starts, used when
            ``layer_bottom`` is not given. The default skips the earliest
            layers, where the lens mostly reads formatting and punctuation
            rather than concepts.
        prune_percent: Percentage of each layer pair's *dense* edge count to
            keep, by ``|score|``. The allowance is set by the full level, not
            by the part still reachable, so it does not compound as the descent
            narrows. ``100`` keeps every edge, which is the dense graph.
        roots: Which top-layer concepts count as the circuit's output, and
            so where pruning starts. ``"last"`` (the default) takes only the
            final position's, the ones whose readout is the model's next token.
            ``"all"`` (or ``None``) keeps every position's, or name positions
            explicitly, as indices or decoded token text like ``"has"``.
            Restricting the roots spends the whole per-layer percentage on the
            cone that reaches them, and needs only ``k`` cotangents in the top
            layer pair's VJP rather than ``p*k``.
        positions: Token positions to build over, or decoded
            token text such as ``"legs"``. Each text value selects every exact
            matching decoded token sequence, including words split across
            multiple tokens. ``None`` takes every position, minus BOS when
            ``skip_bos``.
        skip_bos: Drop position 0 when ``positions`` is ``None``. Its
            attention-sink residual has an outsized norm, which makes its
            coordinates and edges spuriously large.
        vjp_chunk: Target nodes per batched VJP. Lower it if a long prompt runs
            the accelerator out of memory.
        max_seq_len: Truncation length when ``prompt`` is text.

    Returns:
        The built :class:`JCircuit`.

    Raises:
        ValueError: If ``stride < 1`` or is wider than the layer range,
            ``k < 1``, ``prune_percent`` is outside ``(0, 100]``, the layer
            range is empty or a level is not fitted, or a position or root is
            out of range.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if selection not in ("pursuit", "topk"):
        raise ValueError(
            f"selection must be one of ('pursuit', 'topk'), got {selection!r}"
        )
    if estimator not in ("eap", "ig"):
        raise ValueError(f"estimator must be one of ('eap', 'ig'), got {estimator!r}")
    if ig_steps < 1:
        raise ValueError(f"ig_steps must be >= 1, got {ig_steps}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if vjp_chunk < 1:
        raise ValueError(f"vjp_chunk must be >= 1, got {vjp_chunk}")
    if not 0 < prune_percent <= 100:
        raise ValueError(f"prune_percent must be in (0, 100], got {prune_percent}")

    fitted = lens.source_layers
    layer_top_explicit = layer_top is not None
    layer_bottom_explicit = layer_bottom is not None
    if layer_top is None:
        layer_top = _percentile_layer(
            fitted, layer_top_percentile, "layer_top_percentile"
        )
    if layer_bottom is None:
        layer_bottom = _percentile_layer(
            fitted, layer_bottom_percentile, "layer_bottom_percentile"
        )
    if layer_top <= layer_bottom:
        source = (
            f"percentiles {layer_bottom_percentile}..{layer_top_percentile} over "
            f"{len(fitted)} fitted layers"
            if not (layer_top_explicit and layer_bottom_explicit)
            else "the layers given"
        )
        raise ValueError(
            f"layer_top ({layer_top}) must be above layer_bottom ({layer_bottom}); "
            f"they came from {source}"
        )
    # The band is anchored at layer_bottom and stepped up, so a leftover shorter
    # than one stride is dropped from the top rather than scored as a ragged
    # final level of a different width.
    layer_top_requested = layer_top
    n_levels = (layer_top - layer_bottom) // stride
    if n_levels < 1:
        raise ValueError(
            f"stride {stride} is wider than the layer range "
            f"{layer_bottom}..{layer_top}, leaving no level to score"
        )
    layer_top = layer_bottom + n_levels * stride
    if layer_top != layer_top_requested:
        logger.info(
            "stride %d does not divide %d..%d; lowering layer_top to %d",
            stride,
            layer_bottom,
            layer_top_requested,
            layer_top,
        )
    span = list(range(layer_bottom, layer_top + 1, stride))
    missing = [l for l in span if l not in set(fitted)]
    if missing:
        raise ValueError(
            f"every level of {layer_bottom}..{layer_top} step {stride} needs a "
            f"fitted Jacobian; missing {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''}"
        )
    if any(l >= model.n_layers for l in span):
        raise ValueError(f"layer range exceeds the model's {model.n_layers} layers")

    input_ids = (
        model.encode(prompt, max_length=max_seq_len)
        if isinstance(prompt, str)
        else prompt
    )
    seq_len = int(input_ids.shape[1])
    chosen = _resolve_positions(positions, input_ids, skip_bos, model.tokenizer)

    # Token text ("has") is resolvable here but not in JCircuit.prune, which
    # has no tokenizer, so turn it into indices before the shared root rule.
    if roots is not None and not isinstance(roots, str):
        roots = _resolve_positions(list(roots), input_ids, False, model.tokenizer)
    root_positions = _root_positions(roots, chosen)

    # The recorder's hooks must be gone before IG runs its own forwards: they
    # would re-root the graph at layer_bottom on every perturbed pass, retaining
    # a graph IG never uses. ExitStack lets us drop them early without splitting
    # the level loop in two.
    with ExitStack() as stack:
        rec = stack.enter_context(
            ActivationRecorder(model.layers, at=span, start_graph_at=layer_bottom)
        )
        with torch.enable_grad():
            model.forward(input_ids)
        acts = {layer: rec.activations[layer] for layer in span}

        index = torch.tensor(chosen, device=acts[layer_top].device)
        concepts: dict[int, tuple[list[Node], torch.Tensor, torch.Tensor]] = {}
        error_dirs: dict[tuple[int, int], torch.Tensor] = {}
        for layer in span:
            residuals = acts[layer][0].detach().float().index_select(0, index)
            concepts[layer] = _layer_concepts(
                lens, model, layer, residuals, chosen, k, selection, error_nodes
            )
            for node, direction in zip(*concepts[layer][:2], strict=True):
                if node.is_error:
                    error_dirs[(layer, node.position)] = direction.detach()
        if estimator == "ig":
            stack.close()  # concepts are read off; the clean graph is dead weight
            acts = {}

        # Error nodes are sources only, as in Circuit Tracing: they stand for
        # what no concept explains, so "which concept below produced it" is not
        # a question the graph should answer — and an unnamed root is not
        # something anyone reads a circuit out of.
        live = [n for n in concepts[layer_top][0] if not n.is_error]
        if root_positions is not None:
            keep = set(root_positions)
            live = [n for n in live if n.position in keep]
            if not live:
                raise ValueError(
                    f"no concept at layer {layer_top} sits at a root position "
                    f"{sorted(keep)}"
                )
        kept_nodes: dict[int, list[Node]] = {layer_top: live}
        edges: list[Edge] = []
        for layer in range(layer_top, layer_bottom, -stride):
            all_targets, all_dirs, all_pos = concepts[layer]
            sources, source_dirs, source_pos = concepts[layer - stride]
            order = {n: i for i, n in enumerate(all_targets)}
            wanted = live
            pick = torch.tensor([order[n] for n in wanted], device=all_dirs.device)
            targets, target_dirs, target_pos = wanted, all_dirs[pick], all_pos[pick]

            source_acts = torch.tensor(
                [n.activation for n in sources], device=source_dirs.device
            )
            if estimator == "ig":
                scores = _score_layer_pair_ig(
                    model,
                    input_ids,
                    layer,
                    target_dirs,
                    target_pos,
                    source_dirs,
                    source_pos,
                    source_acts,
                    stride,
                    ig_steps,
                    max(1, min(vjp_chunk, IG_ROW_CHUNK)),
                )
            else:
                scores = _score_layer_pair(
                    acts,
                    layer,
                    target_dirs,
                    target_pos,
                    source_dirs,
                    source_pos,
                    source_acts,
                    stride,
                    vjp_chunk,
                )
            # The residual stream only carries a concept forward within its own
            # position; across positions there is no identity path.
            same = source_pos.unsqueeze(1) == target_pos.unsqueeze(0)
            identity = source_acts.unsqueeze(1) * (source_dirs @ target_dirs.T) * same
            causal = source_pos.unsqueeze(1) <= target_pos.unsqueeze(0)

            # The budget comes from the level as the dense graph would have
            # built it — every named target, not only the live ones — so
            # `_prune_levels` and this path allow the same count.
            named_mask = torch.tensor(
                [not n.is_error for n in all_targets], device=all_pos.device
            )
            dense_pos = all_pos[named_mask]
            n_dense = int((source_pos.unsqueeze(1) <= dense_pos.unsqueeze(0)).sum())
            flat = scores.abs().masked_fill(~causal, float("-inf")).flatten()
            n_eligible = int(causal.sum())
            n_keep = _level_budget(n_dense, prune_percent)
            # A stable sort, not topk: equal scores must then be broken by flat
            # index, the same rule `_keep_top_n` follows, or the two pruning
            # paths disagree wherever the cutoff lands in a tie.
            order = torch.argsort(flat, descending=True, stable=True)
            chosen_flat = order[: min(n_keep, n_eligible)].sort().values
            picks = [divmod(int(f), len(targets)) for f in chosen_flat]

            level = [
                Edge(
                    source=sources[i],
                    target=targets[j],
                    score=float(scores[i, j]),
                    identity=float(identity[i, j]),
                )
                for i, j in picks
            ]
            if not level:
                break
            edges.extend(level)
            survivors = {e.source for e in level}
            kept = [n for n in sources if n in survivors]
            if not kept:
                break
            kept_nodes[layer - stride] = kept
            # An error node that survived stays in the graph as a leaf, but it
            # is never a target, so it does not seed the next descent.
            live = [n for n in kept if not n.is_error]
            if not live:
                break

    hparams = {
        "k": k,
        "selection": selection,
        "error_nodes": error_nodes,
        "estimator": estimator,
        "ig_steps": ig_steps if estimator == "ig" else None,
        "stride": stride,
        "layer_top": layer_top,
        # What was asked for before the stride truncated the band, so a circuit
        # says whether its top is the one requested.
        "layer_top_requested": layer_top_requested,
        "layer_bottom": layer_bottom,
        # None when the layer was given explicitly, so the record says which
        # of the two actually selected each end of the band.
        "layer_top_percentile": None if layer_top_explicit else layer_top_percentile,
        "layer_bottom_percentile": (
            None if layer_bottom_explicit else layer_bottom_percentile
        ),
        "prune_percent": prune_percent,
        # None means "every position is a root", the dense reading.
        "roots": root_positions,
        "positions": None if positions is None else list(positions),
        "positions_resolved": chosen,
        "skip_bos": skip_bos,
        "n_levels": n_levels,
        "seq_len": seq_len,
        "vjp_chunk": vjp_chunk,
    }
    return JCircuit(
        nodes=kept_nodes,
        edges=edges,
        hparams=hparams,
        error_directions=error_dirs,
    )
