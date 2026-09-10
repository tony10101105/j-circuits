# Copyright 2026 Tung-Yu (Tony) Wu
# SPDX-License-Identifier: Apache-2.0
# Added by Tung-Yu (Tony) Wu and his Claude.
"""Non-negative gradient pursuit over J-lens vectors.

Taking the top-``k`` lens readouts of an activation gives a *ranked* list, not a
*decomposition*: the J-lens vectors are overcomplete and strongly
non-orthogonal, so the top of that list is mostly spelling variants of one
concept — ``" spider"``, ``" spiders"``, ``" Spider"``, ``"蜘蛛"`` all score
highly on the same underlying content. Sparse decomposition asks the different
question the paper's occupancy analyses ask: which ``k`` lens vectors, combined
with non-negative weights, best *reconstruct* the activation? Once one variant
is in the support it explains that content, and the next atom is chosen to
explain what is left.

The solver is Gradient Pursuit (Blumensath & Davies 2008). It minimises
``0.5 * ||h - V^T c||^2`` over a support that grows by one atom per step; at
each step it moves along the gradient restricted to the current support rather
than re-solving least squares there::

    g     = V r                     gradient on the support
    alpha = ||g||^2 / ||V^T g||^2   exact line search, since <r, V^T g> = ||g||^2
    c     = max(c + alpha * g, 0)   step, then project onto the non-negative orthant

Two matrix-vector products per step and no linear solve, which is what makes it
cheap enough to run at every layer and token position.

Nothing here materialises the dictionary. The atoms are the rows of
``W_U @ J_l`` — 150k x 2560 per layer for a mid-size model, tens of gigabytes
across a layer band — but a pursuit step only needs correlations against every
atom, and those factor as ``W_U @ (J_l @ r)``: one matrix-vector product, the
same one the lens readout already computes. Callers therefore supply
``correlate`` and ``gather_atoms`` closures instead of a dictionary array, which
also keeps the solver testable against a small explicit dictionary.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

#: Guards the line-search denominator when the gradient on the support vanishes.
_STEP_EPS = 1e-12

#: Rows of the unembedding processed at once when measuring atom norms.
_NORM_CHUNK = 16384


def gradient_pursuit(
    targets: torch.Tensor,
    k: int,
    *,
    correlate: Callable[[torch.Tensor], torch.Tensor],
    gather_atoms: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose each row of ``targets`` into at most ``k`` non-negative atoms.

    Args:
        targets: ``[n, d]`` vectors to reconstruct, one per row.
        k: Maximum atoms per target.
        correlate: Maps a ``[n, d]`` residual to ``[n, n_atoms]`` correlations
            against every (unit-norm) atom. Called once per step.
        gather_atoms: Maps ``[n]`` atom indices to the ``[n, d]`` unit atoms.

    Returns:
        ``(indices, coefficients, atoms)`` of shapes ``[n, k]``, ``[n, k]`` and
        ``[n, k, d]``. A slot holds ``-1`` when that target ran out of atoms
        with positive correlation — no non-negative coefficient on any remaining
        atom could shrink its residual — and its coefficient and atom are zero.

    Raises:
        ValueError: If ``k < 1`` or ``targets`` is not 2-D.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if targets.ndim != 2:
        raise ValueError(f"targets must be [n, d], got shape {tuple(targets.shape)}")

    n, d = targets.shape
    device = targets.device
    residual = targets.clone()
    indices = torch.full((n, k), -1, dtype=torch.long, device=device)
    coefficients = torch.zeros(n, k, device=device, dtype=targets.dtype)
    atoms = torch.zeros(n, k, d, device=device, dtype=targets.dtype)
    rows = torch.arange(n, device=device)

    for step in range(k):
        scores = correlate(residual)
        if step:
            # Never reselect: mask the atoms already on the support, leaving
            # rows that stopped early (index -1) untouched.
            taken = indices[:, :step]
            safe = taken.clamp_min(0)
            scores.scatter_(
                1,
                safe,
                torch.where(taken >= 0, float("-inf"), scores.gather(1, safe)),
            )
        best = scores.argmax(dim=1)
        # Non-negativity: only a positively correlated atom can reduce the
        # residual, so a target whose best correlation is <= 0 is finished.
        admits = scores[rows, best] > 0
        if not admits.any():
            break
        indices[admits, step] = best[admits]
        atoms[admits, step] = gather_atoms(best)[admits].to(atoms.dtype)

        support = atoms[:, : step + 1]  # [n, s, d]
        live = (indices[:, : step + 1] >= 0).to(targets.dtype)  # [n, s]
        gradient = torch.einsum("nsd,nd->ns", support, residual) * live
        projected = torch.einsum("nsd,ns->nd", support, gradient)
        alpha = (gradient * gradient).sum(1) / (projected * projected).sum(1).clamp_min(
            _STEP_EPS
        )
        updated = coefficients[:, : step + 1] + alpha.unsqueeze(1) * gradient
        coefficients[:, : step + 1] = updated.clamp_min(0) * live
        residual = targets - torch.einsum(
            "nsd,ns->nd", support, coefficients[:, : step + 1]
        )
    return indices, coefficients, atoms


def lens_atom_norms(
    unembed_weight: torch.Tensor,
    jacobian: torch.Tensor,
    *,
    chunk: int = _NORM_CHUNK,
) -> torch.Tensor:
    """Norms of every J-lens vector at one layer: ``||W_U[t] @ J_l||`` per token.

    Matching pursuit needs unit-norm atoms, or selection just prefers whichever
    token happens to have the longest unembedding row. Computed in chunks so the
    ``[vocab, d_model]`` product is never held whole.
    """
    vocab = unembed_weight.shape[0]
    out = torch.empty(vocab, device=jacobian.device, dtype=torch.float32)
    for start in range(0, vocab, chunk):
        rows = unembed_weight[start : start + chunk].detach().to(jacobian.device)
        out[start : start + chunk] = (rows.float() @ jacobian).norm(dim=-1)
    return out


def pursue_lens(
    unembed_weight: torch.Tensor,
    jacobian: torch.Tensor,
    residuals: torch.Tensor,
    k: int,
    *,
    atom_norms: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run :func:`gradient_pursuit` over a layer's J-lens vectors, matrix-free.

    Args:
        unembed_weight: ``[vocab, d_final]`` raw unembedding matrix.
        jacobian: ``J_l``, the layer's fitted Jacobian.
        residuals: ``[n_positions, d_model]`` activations to decompose.
        k: Atoms per position.
        atom_norms: Precomputed :func:`lens_atom_norms`; measured here when
            omitted. Pass it in when decomposing the same layer repeatedly.

    Returns:
        ``(token_ids, coefficients, directions)`` as returned by
        :func:`gradient_pursuit`, with ``directions`` the unit J-lens vectors.
    """
    weight = unembed_weight.detach().to(jacobian.device)
    norms = (
        lens_atom_norms(weight, jacobian) if atom_norms is None else atom_norms
    ).clamp_min(_STEP_EPS)

    def correlate(residual: torch.Tensor) -> torch.Tensor:
        # <v_t, r> = w_t . (J r), so one matvec covers the whole vocabulary
        # without ever forming W_U @ J.
        return ((residual @ jacobian.T) @ weight.float().T) / norms

    def gather_atoms(token_ids: torch.Tensor) -> torch.Tensor:
        return (weight[token_ids].float() @ jacobian) / norms[token_ids].unsqueeze(1)

    return gradient_pursuit(
        residuals.float(), k, correlate=correlate, gather_atoms=gather_atoms
    )
