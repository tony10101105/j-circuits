# Copyright 2026 Tung-Yu (Tony) Wu
# SPDX-License-Identifier: Apache-2.0
# Added by Tung-Yu (Tony) Wu and his Claude.
"""Interventions match the paper's formulas on the tiny model.

The swap is checked against a literal pseudoinverse reference
(``h + alpha * V (sigma(c) - c)`` with ``c = pinv(V) @ h``), since the
implementation uses the rank-1 collapse documented in
:mod:`jlens.interventions`.
"""

import pytest
import torch

from jlens.hooks import ActivationRecorder
from jlens.interventions import (
    Ablate,
    Intervention,
    Steer,
    Swap,
    TopKAblation,
    greedy_generate,
    lens_vector,
)
from jlens.lens import JacobianLens
from tests.tiny import TinyDecoder

PROMPT = "spin spin web"
LAYER = 1


@pytest.fixture()
def model():
    return TinyDecoder(n_layers=4, d_model=8, vocab_size=32, seed=0)


@pytest.fixture()
def lens(model):
    g = torch.Generator().manual_seed(7)
    jacobians = {
        layer: torch.randn(8, 8, generator=g) * 0.5 + torch.eye(8) for layer in (1, 2)
    }
    return JacobianLens(jacobians=jacobians, n_prompts=1, d_model=8)


def residual_at(model, layer, prompt=PROMPT):
    with ActivationRecorder(model.layers, at=[layer]) as recorder:
        model.forward(model.encode(prompt))
        return recorder.activations[layer].detach().clone()


def test_lens_vector_is_row_of_wu_j(lens, model):
    v = lens_vector(lens, model, 5, LAYER)
    expected = model.unembed_weight[5].float() @ lens.jacobians[LAYER]
    torch.testing.assert_close(v, expected)
    # logit-lens baseline: J = I
    torch.testing.assert_close(
        lens_vector(lens, model, 5, LAYER, use_jacobian=False),
        model.unembed_weight[5].float(),
    )


def test_steer_adds_alpha_v(lens, model):
    clean = residual_at(model, LAYER)
    alpha, token = 3.0, 5
    v = lens_vector(lens, model, token, LAYER)
    with Intervention(
        lens, model, [Steer(token, alpha)], layers=[LAYER], skip_bos=False
    ):
        steered = residual_at(model, LAYER)
    torch.testing.assert_close(steered, clean + alpha * v)


def test_ablate_zeroes_projection_and_is_idempotent(lens, model):
    token = 9
    v = lens_vector(lens, model, token, LAYER)
    with Intervention(lens, model, [Ablate(token)], layers=[LAYER], skip_bos=False):
        once = residual_at(model, LAYER)
    assert (once @ v).abs().max() < 1e-5
    # paper form h - (<v,h>/||v||^2) v against the clean residual
    clean = residual_at(model, LAYER)
    coeff = (clean @ v) / (v @ v)
    torch.testing.assert_close(once, clean - coeff.unsqueeze(-1) * v)
    with Intervention(
        lens, model, [Ablate(token), Ablate(token)], layers=[LAYER], skip_bos=False
    ):
        twice = residual_at(model, LAYER)
    torch.testing.assert_close(twice, once)


@pytest.mark.parametrize("alpha", [1.0, 2.0])
def test_swap_matches_pinv_reference(lens, model, alpha):
    source, target = 5, 9
    clean = residual_at(model, LAYER)
    v_s = lens_vector(lens, model, source, LAYER)
    v_t = lens_vector(lens, model, target, LAYER)
    V = torch.stack([v_s, v_t], dim=1)  # [d, 2]
    c = clean @ torch.linalg.pinv(V).T  # [..., 2]
    sigma = c.flip(-1)
    expected = clean + alpha * (sigma - c) @ V.T
    with Intervention(
        lens, model, [Swap(source, target, alpha=alpha)], layers=[LAYER], skip_bos=False
    ):
        swapped = residual_at(model, LAYER)
    torch.testing.assert_close(swapped, expected, atol=1e-5, rtol=1e-4)


def test_swap_delta_lies_in_span(lens, model):
    source, target = 5, 9
    clean = residual_at(model, LAYER)
    v_s = lens_vector(lens, model, source, LAYER)
    v_t = lens_vector(lens, model, target, LAYER)
    with Intervention(
        lens, model, [Swap(source, target)], layers=[LAYER], skip_bos=False
    ):
        swapped = residual_at(model, LAYER)
    delta = (swapped - clean).reshape(-1, 8)
    Q, _ = torch.linalg.qr(torch.stack([v_s, v_t], dim=1))
    outside = delta - (delta @ Q) @ Q.T
    assert outside.abs().max() < 1e-5


def test_swap_roundtrip_is_identity(lens, model):
    clean = residual_at(model, LAYER)
    with Intervention(
        lens, model, [Swap(5, 9), Swap(9, 5)], layers=[LAYER], skip_bos=False
    ):
        back = residual_at(model, LAYER)
    torch.testing.assert_close(back, clean, atol=1e-5, rtol=1e-4)


def test_positions_and_skip_bos(lens, model):
    clean = residual_at(model, LAYER)
    with Intervention(
        lens, model, [Steer(5, 10.0)], layers=[LAYER], positions=[2], skip_bos=False
    ):
        edited = residual_at(model, LAYER)
    changed = (edited != clean).any(-1)[0]
    assert (
        changed[2] and not changed[[i for i in range(clean.shape[1]) if i != 2]].any()
    )
    # positions=None with skip_bos leaves position 0 (BOS) alone
    with Intervention(lens, model, [Steer(5, 10.0)], layers=[LAYER]):
        edited = residual_at(model, LAYER)
    changed = (edited != clean).any(-1)[0]
    assert not changed[0] and changed[1:].all()


def test_hooks_removed_on_exit_and_layers_validated(lens, model):
    clean = residual_at(model, LAYER)
    with Intervention(lens, model, [Steer(5, 10.0)], layers=[LAYER]):
        pass
    torch.testing.assert_close(residual_at(model, LAYER), clean)
    with pytest.raises(ValueError, match="source_layers"):
        Intervention(lens, model, [Steer(5, 1.0)], layers=[3])  # unfitted
    with pytest.raises(ValueError, match="out of range"):
        Intervention(lens, model, [Steer(5, 1.0)], layers=[99])
    with pytest.raises(ValueError, match="collinear"):
        Intervention(lens, model, [Swap(5, 5)], layers=[LAYER])


def topk_reference(lens, model, k, exclude_top, layer, prompt=PROMPT):
    """Literal reference: rank by lens readout, mask the clean top-n output
    tokens, gather the top-k lens vectors, and project them out via pinv."""
    clean = residual_at(model, layer, prompt)[0].float()  # [seq, d]
    final = residual_at(model, model.n_layers - 1, prompt)[0].float()
    excluded = model.unembed(final).topk(exclude_top, dim=-1).indices
    J = lens.jacobians[layer]
    lens_logits = model.unembed(clean @ J.T).float()
    lens_logits.scatter_(-1, excluded, float("-inf"))
    idx = lens_logits.topk(k, dim=-1).indices  # [seq, k]
    vectors = model.unembed_weight[idx].float() @ J  # [seq, k, d]
    expected = clean.clone()
    for p in range(clean.shape[0]):
        V = vectors[p].T  # [d, k]
        expected[p] = clean[p] - V @ (torch.linalg.pinv(V) @ clean[p])
    return expected, idx, excluded


@pytest.mark.parametrize("k,exclude_top", [(1, 0), (3, 5), (4, 10)])
def test_topk_matches_pinv_reference(lens, model, k, exclude_top):
    expected, idx, excluded = topk_reference(lens, model, k, exclude_top, LAYER)
    ablation = TopKAblation(
        lens,
        model,
        PROMPT,
        k=k,
        exclude_top=exclude_top,
        layers=[LAYER],
        skip_bos=False,
    )
    with ablation:
        edited = residual_at(model, LAYER)[0].float()
    torch.testing.assert_close(edited, expected, atol=1e-5, rtol=1e-4)
    # the selected vectors' activations are zeroed
    vectors = model.unembed_weight[idx].float() @ lens.jacobians[LAYER]
    activity = torch.einsum("skd,sd->sk", vectors, edited)
    assert activity.abs().max() < 1e-4
    # bookkeeping: excluded ids recorded, selection disjoint from them
    torch.testing.assert_close(ablation.excluded, excluded.cpu())
    torch.testing.assert_close(ablation.selected[LAYER][0], idx.cpu())
    for p in range(idx.shape[0]):
        assert not set(idx[p].tolist()) & set(excluded[p].tolist())


def test_topk_positions_and_skip_bos(lens, model):
    clean = residual_at(model, LAYER)
    with TopKAblation(
        lens, model, PROMPT, k=2, layers=[LAYER], positions=[2], skip_bos=False
    ):
        edited = residual_at(model, LAYER)
    changed = (edited != clean).any(-1)[0]
    assert (
        changed[2] and not changed[[i for i in range(clean.shape[1]) if i != 2]].any()
    )
    # positions=None with skip_bos leaves position 0 (BOS) alone
    with TopKAblation(lens, model, PROMPT, k=2, layers=[LAYER]):
        edited = residual_at(model, LAYER)
    changed = (edited != clean).any(-1)[0]
    assert not changed[0] and changed[1:].all()


def test_topk_validates_and_binds_to_one_prompt(lens, model):
    with pytest.raises(ValueError, match="k must be"):
        TopKAblation(lens, model, PROMPT, k=0, layers=[LAYER])
    with pytest.raises(ValueError, match="exclude_top"):
        TopKAblation(lens, model, PROMPT, k=1, exclude_top=-1, layers=[LAYER])
    with pytest.raises(ValueError, match="vocab"):
        TopKAblation(lens, model, PROMPT, k=30, exclude_top=10, layers=[LAYER])
    with pytest.raises(ValueError, match="source_layers"):
        TopKAblation(lens, model, PROMPT, k=1, layers=[3])  # unfitted
    # a forward with a different sequence length is refused at hook time
    with TopKAblation(lens, model, PROMPT, k=2, layers=[LAYER]):
        with pytest.raises(RuntimeError, match="seq_len"):
            residual_at(model, LAYER, prompt=PROMPT + " extra")
        with pytest.raises(RuntimeError, match="seq_len"):
            greedy_generate(model, PROMPT, max_new_tokens=2)
    # hooks removed after exit
    clean = residual_at(model, LAYER)
    with TopKAblation(lens, model, PROMPT, k=2, layers=[LAYER]):
        pass
    torch.testing.assert_close(residual_at(model, LAYER), clean)


def test_reads_compose_and_generation_runs(lens, model):
    lens_clean, _, _ = lens.apply(model, PROMPT, layers=[LAYER], positions=[-1])
    with Intervention(lens, model, [Steer(5, 25.0)], layers=[LAYER]):
        lens_steered, _, _ = lens.apply(model, PROMPT, layers=[LAYER], positions=[-1])
        text = greedy_generate(model, PROMPT, max_new_tokens=4)
    assert not torch.allclose(lens_clean[LAYER], lens_steered[LAYER])
    assert isinstance(text, str) and len(text) == 4  # byte tokenizer: 1 char/token
    # deterministic and clean again after exit
    assert greedy_generate(model, PROMPT, max_new_tokens=4) == greedy_generate(
        model, PROMPT, max_new_tokens=4
    )
