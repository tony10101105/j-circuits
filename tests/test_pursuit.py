# Copyright 2026 Tung-Yu (Tony) Wu
# SPDX-License-Identifier: Apache-2.0
# Added by Tung-Yu (Tony) Wu and his Claude.
"""Non-negative gradient pursuit: recovery, invariants, and the redundancy it fixes."""

import pytest
import torch

from jlens.interventions import lens_vector
from jlens.lens import JacobianLens
from jlens.pursuit import gradient_pursuit, lens_atom_norms, pursue_lens
from tests.tiny import TinyDecoder


def unit_rows(n: int, d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    D = torch.randn(n, d, generator=g)
    return D / D.norm(dim=-1, keepdim=True)


def solver(D: torch.Tensor):
    """`correlate` / `gather_atoms` closures over an explicit dictionary."""
    return {"correlate": lambda r: r @ D.T, "gather_atoms": lambda i: D[i]}


def orthonormal_rows(n: int, d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, n, generator=g))
    return Q.T[:n]


# --------------------------------------------------------------------------- #
# correctness
# --------------------------------------------------------------------------- #


def test_orthonormal_dictionary_is_recovered_exactly():
    """With orthonormal atoms the line search gives alpha = 1, so one gradient
    step per atom lands on the exact projection — pursuit must be exact."""
    D = orthonormal_rows(6, 12, seed=0)
    picked, weights = [1, 3, 4], torch.tensor([2.0, 0.5, 1.25])
    target = (weights.unsqueeze(1) * D[picked]).sum(0, keepdim=True)

    ids, coefficients, _ = gradient_pursuit(target, 3, **solver(D))
    order = ids[0].argsort()
    assert ids[0][order].tolist() == picked
    torch.testing.assert_close(coefficients[0][order], weights, atol=1e-5, rtol=1e-4)


def test_coefficients_are_non_negative_and_atoms_unique():
    D = unit_rows(40, 16, seed=1)
    targets = torch.randn(8, 16, generator=torch.Generator().manual_seed(2))
    ids, coefficients, _ = gradient_pursuit(targets, 5, **solver(D))
    assert (coefficients >= 0).all()
    for row in ids:
        chosen = [int(i) for i in row if i >= 0]
        assert len(chosen) == len(set(chosen))


def test_residual_never_grows():
    D = unit_rows(30, 16, seed=3)
    targets = torch.randn(4, 16, generator=torch.Generator().manual_seed(4))
    previous = targets.norm(dim=-1)
    for k in range(1, 6):
        ids, coefficients, atoms = gradient_pursuit(targets, k, **solver(D))
        residual = targets - torch.einsum("nkd,nk->nd", atoms, coefficients)
        current = residual.norm(dim=-1)
        assert (current <= previous + 1e-5).all()
        previous = current


def test_atoms_without_positive_correlation_are_never_selected():
    D = orthonormal_rows(2, 8, seed=5)
    target = -D[:1]  # correlates -1 with atom 0 and 0 with atom 1
    ids, coefficients, _ = gradient_pursuit(target, 2, **solver(D))
    assert ids.tolist() == [[-1, -1]]
    assert coefficients.abs().max() == 0.0


def test_empty_slots_are_marked_and_zeroed():
    D = orthonormal_rows(2, 8, seed=6)
    target = D[:1] * 3.0  # only one atom can help; the other is orthogonal
    ids, coefficients, atoms = gradient_pursuit(target, 4, **solver(D))
    assert ids[0, 0] == 0 and (ids[0, 1:] == -1).all()
    assert coefficients[0, 0] == pytest.approx(3.0, abs=1e-5)
    assert coefficients[0, 1:].abs().max() == 0.0 and atoms[0, 1:].abs().max() == 0.0


def test_invalid_arguments_raise():
    D = unit_rows(5, 4, seed=7)
    with pytest.raises(ValueError, match="k must be"):
        gradient_pursuit(torch.zeros(1, 4), 0, **solver(D))
    with pytest.raises(ValueError, match=r"targets must be \[n, d\]"):
        gradient_pursuit(torch.zeros(4), 2, **solver(D))


# --------------------------------------------------------------------------- #
# the redundancy this replaces top-k to fix
# --------------------------------------------------------------------------- #


def test_pursuit_skips_near_duplicates_that_topk_wastes_slots_on():
    """A dictionary holding three spellings of one concept plus a second
    concept. Ranking by correlation spends every slot on the duplicates;
    pursuit takes one of them and then goes after what is left."""
    g = torch.Generator().manual_seed(8)
    base = torch.randn(16, generator=g)
    base = base / base.norm()
    other = torch.randn(16, generator=g)
    other = other - (other @ base) * base
    other = other / other.norm()

    variants = base + 0.05 * torch.randn(3, 16, generator=g)
    D = torch.cat([variants / variants.norm(dim=-1, keepdim=True), other[None]])
    target = (base + 0.6 * other).unsqueeze(0)

    by_correlation = (target @ D.T)[0].topk(2).indices.tolist()
    assert by_correlation == [0, 1] or set(by_correlation) <= {0, 1, 2}, (
        "top-k should spend both slots on duplicates"
    )
    assert 3 not in by_correlation

    ids, coefficients, atoms = gradient_pursuit(target, 2, **solver(D))
    assert 3 in ids[0].tolist(), "pursuit must reach the second concept"
    residual = target - torch.einsum("nkd,nk->nd", atoms, coefficients)
    topk_atoms = D[by_correlation]
    topk_fit = torch.linalg.lstsq(topk_atoms.T, target[0]).solution
    topk_residual = target[0] - topk_atoms.T @ topk_fit
    assert residual.norm() < topk_residual.norm()


# --------------------------------------------------------------------------- #
# the lens wrapper
# --------------------------------------------------------------------------- #


@pytest.fixture()
def model():
    return TinyDecoder(n_layers=4, d_model=8, vocab_size=32, seed=0)


@pytest.fixture()
def lens():
    g = torch.Generator().manual_seed(7)
    return JacobianLens(
        jacobians={
            l: torch.randn(8, 8, generator=g) * 0.5 + torch.eye(8) for l in range(4)
        },
        n_prompts=1,
        d_model=8,
    )


def test_atom_norms_match_lens_vector(lens, model):
    norms = lens_atom_norms(model.unembed_weight, lens.jacobians[2])
    assert norms.shape == (32,)
    for token_id in (0, 5, 17, 31):
        expected = lens_vector(lens, model, token_id, 2).norm()
        assert norms[token_id] == pytest.approx(float(expected), abs=1e-5)


def test_pursue_lens_returns_unit_lens_directions(lens, model):
    residuals = torch.randn(3, 8, generator=torch.Generator().manual_seed(9))
    ids, coefficients, directions = pursue_lens(
        model.unembed_weight, lens.jacobians[1], residuals, 3
    )
    assert ids.shape == coefficients.shape == (3, 3)
    assert directions.shape == (3, 3, 8)
    for row, slots in enumerate(ids):
        for slot, token_id in enumerate(slots):
            if token_id < 0:
                continue
            expected = lens_vector(lens, model, int(token_id), 1)
            expected = expected / expected.norm()
            torch.testing.assert_close(
                directions[row, slot], expected, atol=1e-5, rtol=1e-4
            )
    assert (coefficients >= 0).all()


def test_precomputed_atom_norms_give_the_same_answer(lens, model):
    residuals = torch.randn(2, 8, generator=torch.Generator().manual_seed(10))
    J = lens.jacobians[3]
    norms = lens_atom_norms(model.unembed_weight, J)
    a = pursue_lens(model.unembed_weight, J, residuals, 3)
    b = pursue_lens(model.unembed_weight, J, residuals, 3, atom_norms=norms)
    assert a[0].tolist() == b[0].tolist()
    torch.testing.assert_close(a[1], b[1])
