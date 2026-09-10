# Copyright 2026 Tung-Yu (Tony) Wu
# SPDX-License-Identifier: Apache-2.0
# Added by Tung-Yu (Tony) Wu and his Claude.
"""J-circuit construction, pruning, and edge-score correctness.

The tiny model's blocks are ``h + 0.1*linear(h)`` — linear — so a readout
``<v_t, h_l>`` is a linear function of ``h_{l-1}`` and the first-order EAP score
is *exact*. Edges are therefore checked against real ablations at tight
tolerance.

The tiny decoder has no attention, so a cross-position edge there is genuinely
zero; that fact is itself used as a test.
"""

import contextlib

import pytest
import torch

from jlens.circuit import JCircuit, Node, _resolve_positions, build_jcircuit
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from tests.tiny import TinyDecoder

PROMPT = "spin spin web"


@contextlib.contextmanager
def ablate(circuit, node, lens, model):
    """Project ``node``'s direction out of its own ``(layer, position)``.

    Not ``Intervention([Ablate(token_id)])``: that resolves a vocabulary entry,
    and an error node has none. Going through :meth:`JCircuit.direction` lets
    the same ground-truth check cover both kinds of source.
    """
    v = circuit.direction(node, lens, model)

    def hook(module, inputs, output):
        tensor = output if torch.is_tensor(output) else output[0]
        work = tensor.float().clone()
        coeff = work[:, node.position] @ v
        work[:, node.position] = work[:, node.position] - coeff.unsqueeze(-1) * v
        edited = work.to(tensor.dtype)
        return edited if torch.is_tensor(output) else (edited, *tuple(output[1:]))

    handle = model.layers[node.layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@pytest.fixture()
def model():
    return TinyDecoder(n_layers=4, d_model=8, vocab_size=32, seed=0)


@pytest.fixture()
def lens():
    g = torch.Generator().manual_seed(7)
    jacobians = {
        layer: torch.randn(8, 8, generator=g) * 0.5 + torch.eye(8) for layer in range(4)
    }
    return JacobianLens(jacobians=jacobians, n_prompts=1, d_model=8)


def build(lens, model, **kw):
    kw.setdefault("layer_top", 3)
    kw.setdefault("layer_bottom", 0)
    kw.setdefault("k", 3)
    return build_jcircuit(lens, model, PROMPT, **kw)


def seq_len(model):
    return int(model.encode(PROMPT).shape[1])


# --------------------------------------------------------------------------- #
# structure
# --------------------------------------------------------------------------- #


def test_each_layer_is_sliced_into_position_blocks_of_k(lens, model):
    circuit = build(lens, model, k=3, error_nodes=False)
    expected = list(range(1, seq_len(model)))  # skip_bos drops position 0
    assert circuit.positions == expected
    for layer in range(4):
        assert len(circuit.nodes[layer]) == len(expected) * 3
        for position in expected:
            block = circuit.nodes_at(layer, position)
            assert [n.lens_rank for n in block] == [0, 1, 2]
            assert all(n.position == position for n in block)


def test_error_nodes_add_one_leaf_per_block_below_the_top(lens, model):
    """One extra node per (layer, position), ranked after every named concept.

    Not at ``layer_top``: error nodes are sources only, so the root level never
    carries one."""
    circuit = build(lens, model, k=3)
    positions = list(range(1, seq_len(model)))
    for layer in range(4):
        blocks = [circuit.nodes_at(layer, p) for p in positions]
        if layer == circuit.layer_top:
            assert all(not n.is_error for b in blocks for n in b)
            assert len(circuit.nodes[layer]) == len(positions) * 3
            continue
        assert len(circuit.nodes[layer]) == len(positions) * 4
        for block in blocks:
            assert [n.is_error for n in block] == [False, False, False, True]
            assert block[-1].lens_rank == 3
            assert block[-1].token_id == -1
            assert block[-1].coefficient is None


def test_dense_edge_count_is_causal_pairs_times_k_squared(lens, model):
    for k in (1, 2, 3):
        circuit = build(lens, model, k=k, error_nodes=False)
        p = len(circuit.positions)
        pairs = p * (p + 1) // 2  # source position <= target position
        assert len(circuit.edges) == 3 * pairs * k * k
        assert all(e.target.layer - e.source.layer == 1 for e in circuit.edges)


def test_error_nodes_add_a_source_row_but_never_a_target_column(lens, model):
    for k in (1, 2, 3):
        circuit = build(lens, model, k=k)
        p = len(circuit.positions)
        pairs = p * (p + 1) // 2
        # k+1 sources, still only k targets
        assert len(circuit.edges) == 3 * pairs * (k + 1) * k
        assert any(e.source.is_error for e in circuit.edges)
        assert not any(e.target.is_error for e in circuit.edges)
        assert all(e.target.layer - e.source.layer == 1 for e in circuit.edges)


def test_edges_obey_the_causal_position_order(lens, model):
    circuit = build(lens, model, k=2)
    assert all(e.source.position <= e.target.position for e in circuit.edges)
    assert any(e.cross_position for e in circuit.edges)
    # every allowed pair appears exactly once
    seen = {(e.source, e.target) for e in circuit.edges}
    assert len(seen) == len(circuit.edges)


def test_positions_argument_restricts_the_blocks(lens, model):
    n = seq_len(model)
    circuit = build(lens, model, k=2, positions=[1, 3])
    assert circuit.positions == [1, 3]
    assert circuit.hparams["positions_resolved"] == [1, 3]
    # negative indices resolve against the sequence
    assert build(lens, model, k=2, positions=[-1]).positions == [n - 1]
    # a single position reproduces the same-position-only circuit
    single = build(lens, model, k=2, positions=[-1])
    assert all(not e.cross_position for e in single.edges)
    assert len(single.edges) == 3 * 3 * 2  # (k+1) sources x k targets


def test_decoded_token_text_resolves_to_all_matching_positions(lens, model):
    input_ids = model.encode(PROMPT)[0].tolist()
    first = model.tokenizer.decode([input_ids[1]])
    last = model.tokenizer.decode([input_ids[-1]])
    circuit = build(lens, model, k=2, positions=[first, last])
    expected = [
        index
        for index, token_id in enumerate(input_ids)
        if model.tokenizer.decode([token_id]) in {first, last}
    ]
    assert circuit.positions == expected
    assert circuit.hparams["positions_resolved"] == expected


def test_decoded_token_text_can_span_multiple_tokens(model):
    input_ids = model.encode(PROMPT)[:, :4]
    word = "".join(model.tokenizer.decode([token_id]) for token_id in input_ids[0, 1:])
    assert _resolve_positions([word], input_ids, True, model.tokenizer) == [1, 2, 3]


def test_skip_bos_controls_position_zero(lens, model):
    assert 0 not in build(lens, model, k=2).positions
    assert build(lens, model, k=2, skip_bos=False).positions[0] == 0


def test_node_activation_matches_lens_vector_at_its_position(lens, model):
    circuit = build(lens, model, k=2)
    with ActivationRecorder(model.layers, at=[1]) as rec:
        model.forward(model.encode(PROMPT))
        h1 = rec.activations[1][0].detach().float()
    for node in circuit.nodes_at(1, 2):
        v = circuit.direction(node, lens, model)
        v = v / v.norm()
        assert node.activation == pytest.approx(float(h1[2] @ v), abs=1e-5)


def test_identity_is_the_residual_carry_and_only_within_a_position(lens, model):
    circuit = build(lens, model, k=2)
    same = [e for e in circuit.edges if not e.cross_position]
    cross = [e for e in circuit.edges if e.cross_position]
    assert all(e.identity == 0.0 for e in cross), "no residual path across positions"
    edge = same[0]
    vs = circuit.direction(edge.source, lens, model)
    vt = circuit.direction(edge.target, lens, model)
    expected = edge.source.activation * float(vs @ vt)
    assert edge.identity == pytest.approx(expected, abs=1e-5)
    assert edge.computed == pytest.approx(edge.score - edge.identity, abs=1e-9)


# --------------------------------------------------------------------------- #
# edge scores against ground truth
# --------------------------------------------------------------------------- #


def test_edge_scores_match_true_ablation(lens, model):
    circuit = build(lens, model, k=2, positions=[1, 2])

    def readout(node):
        v = circuit.direction(node, lens, model)
        with torch.no_grad(), ActivationRecorder(model.layers, at=[node.layer]) as rec:
            model.forward(model.encode(PROMPT))
            return float(rec.activations[node.layer][0, node.position].float() @ v)

    checked = 0
    for edge in circuit.edges:
        if edge.target.layer != 2:  # one layer pair is enough, and cheap
            continue
        clean = readout(edge.target)
        with ablate(circuit, edge.source, lens, model):
            ablated = readout(edge.target)
        assert edge.score == pytest.approx(clean - ablated, abs=1e-4)
        checked += 1
    assert checked == 3 * 3 * 2  # 3 position pairs x (k+1) sources x k targets


def test_cross_position_edges_vanish_without_attention(lens, model):
    """The tiny decoder is position-wise, so attention edges must be zero."""
    circuit = build(lens, model, k=3)
    cross = [e for e in circuit.edges if e.cross_position]
    assert cross, "fixture should produce cross-position pairs"
    assert max(abs(e.score) for e in cross) < 1e-6
    same = [e for e in circuit.edges if not e.cross_position]
    assert max(abs(e.score) for e in same) > 1e-3


# --------------------------------------------------------------------------- #
# estimator: eap vs ig
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("steps", [1, 2, 5])
def test_ig_equals_eap_on_a_linear_model(lens, model, steps):
    """The integrand is constant in alpha when the blocks are linear, so every
    path sample sees the same gradient and IG must reproduce EAP exactly. Any
    drift here means the perturbation, the path, or the graph root is wrong."""
    common = dict(k=3, positions=[1, 2, 3])
    eap = build(lens, model, estimator="eap", **common)
    ig = build(lens, model, estimator="ig", ig_steps=steps, **common)
    assert ig.nodes == eap.nodes
    assert ig.hparams["estimator"] == "ig" and ig.hparams["ig_steps"] == steps
    assert eap.hparams["estimator"] == "eap" and eap.hparams["ig_steps"] is None
    for a, b in zip(eap.edges, ig.edges, strict=True):
        assert a.source == b.source and a.target == b.target
        assert a.identity == pytest.approx(b.identity, abs=1e-9)
        assert a.score == pytest.approx(b.score, abs=1e-4)


def test_ig_edge_scores_match_true_ablation(lens, model):
    """The same ground-truth check the EAP scorer gets, through the IG path."""
    circuit = build(lens, model, estimator="ig", k=2, positions=[1, 2])

    def readout(node):
        v = circuit.direction(node, lens, model)
        with torch.no_grad(), ActivationRecorder(model.layers, at=[node.layer]) as rec:
            model.forward(model.encode(PROMPT))
            return float(rec.activations[node.layer][0, node.position].float() @ v)

    checked = 0
    for edge in circuit.edges:
        if edge.target.layer != 2:
            continue
        clean = readout(edge.target)
        with ablate(circuit, edge.source, lens, model):
            ablated = readout(edge.target)
        assert edge.score == pytest.approx(clean - ablated, abs=1e-4)
        checked += 1
    assert checked == 3 * 3 * 2  # (k+1) sources x k targets


def test_ig_works_with_stride_mode2_and_roots(lens, model):
    """The estimator is orthogonal to the rest of the knobs."""
    common = dict(k=3, positions=[1, 2, 3], stride=2, layer_bottom=0, layer_top=3)
    dense = build_jcircuit(lens, model, PROMPT, mode=1, estimator="ig", **common)
    assert dense.layers == [2, 0]
    pruned = dense.prune(40.0, roots="last")
    direct = build_jcircuit(
        lens, model, PROMPT, mode=2, estimator="ig", prune_percent=40.0, **common
    )
    assert pruned.nodes == direct.nodes
    for a, b in zip(pruned.edges, direct.edges, strict=True):
        assert a.source == b.source and a.target == b.target
        assert a.score == pytest.approx(b.score, abs=1e-4)


def test_ig_leaves_no_hooks_behind(lens, model):
    """IG registers its own forward hooks; a leak would corrupt later passes."""
    before = {id(h) for layer in model.layers for h in layer._forward_hooks.values()}
    build(lens, model, estimator="ig", k=2, positions=[1, 2])
    after = {id(h) for layer in model.layers for h in layer._forward_hooks.values()}
    assert after == before
    # and a clean readout afterwards is unperturbed
    with torch.no_grad(), ActivationRecorder(model.layers, at=[2]) as rec:
        model.forward(model.encode(PROMPT))
        again = rec.activations[2].clone()
    with torch.no_grad(), ActivationRecorder(model.layers, at=[2]) as rec:
        model.forward(model.encode(PROMPT))
        torch.testing.assert_close(rec.activations[2], again)


def test_invalid_estimator_arguments_raise(lens, model):
    with pytest.raises(ValueError, match="estimator must be one of"):
        build(lens, model, estimator="integrated")
    with pytest.raises(ValueError, match="ig_steps must be >= 1"):
        build(lens, model, estimator="ig", ig_steps=0)


# --------------------------------------------------------------------------- #
# stride
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("stride", [1, 2, 3])
def test_levels_step_by_stride_from_the_bottom(lens, model, stride):
    circuit = build(lens, model, k=2, positions=[1, 2], stride=stride)
    expected = list(range(0, 3 + 1, stride))
    assert circuit.stride == stride
    assert circuit.layers == sorted(expected, reverse=True)
    assert circuit.hparams["n_levels"] == len(expected) - 1
    assert circuit.layer_top == expected[-1]
    for edge in circuit.edges:
        assert edge.target.layer - edge.source.layer == stride


def test_a_ragged_top_is_dropped_not_scored(lens, model):
    """20..23 at stride 2 means 20 -> 22; the leftover 22 -> 23 is not a level."""
    circuit = build_jcircuit(
        lens,
        model,
        PROMPT,
        k=2,
        positions=[1, 2],
        layer_bottom=0,
        layer_top=3,
        stride=2,
    )
    assert (circuit.layer_bottom, circuit.layer_top) == (0, 2)
    assert circuit.hparams["layer_top_requested"] == 3
    assert circuit.hparams["n_levels"] == 1
    assert 3 not in circuit.nodes
    # an exact fit records the same top it was asked for
    exact = build(lens, model, k=2, positions=[1, 2], stride=3)
    assert exact.layer_top == exact.hparams["layer_top_requested"] == 3


def test_strided_edge_scores_match_true_ablation(lens, model):
    """The tiny blocks are linear, so first-order EAP is exact however many of
    them an edge spans — a strided score must still equal a real ablation."""
    circuit = build_jcircuit(
        lens,
        model,
        PROMPT,
        k=2,
        positions=[1, 2],
        layer_bottom=0,
        layer_top=3,
        stride=3,
    )

    def readout(node):
        v = circuit.direction(node, lens, model)
        with torch.no_grad(), ActivationRecorder(model.layers, at=[node.layer]) as rec:
            model.forward(model.encode(PROMPT))
            return float(rec.activations[node.layer][0, node.position].float() @ v)

    assert circuit.edges
    for edge in circuit.edges:
        clean = readout(edge.target)
        with ablate(circuit, edge.source, lens, model):
            ablated = readout(edge.target)
        assert edge.score == pytest.approx(clean - ablated, abs=1e-4)


def test_stride_keeps_the_identity_share_exact(lens, model):
    """The residual path across several blocks is still the identity, so the
    carry term is the same inner product it is at stride 1."""
    circuit = build(lens, model, k=2, positions=[1, 2], stride=3)
    same = [e for e in circuit.edges if not e.cross_position]
    assert same
    for edge in same:
        vs = circuit.direction(edge.source, lens, model)
        vt = circuit.direction(edge.target, lens, model)
        expected = edge.source.activation * float((vs / vs.norm()) @ (vt / vt.norm()))
        assert edge.identity == pytest.approx(expected, abs=1e-5)


def test_stride_survives_pruning_in_both_paths(lens, model):
    common = dict(k=3, positions=[1, 2, 3], stride=2, layer_bottom=0, layer_top=3)
    dense = build_jcircuit(lens, model, PROMPT, mode=1, **common)
    pruned = dense.prune(40.0)
    direct = build_jcircuit(lens, model, PROMPT, mode=2, prune_percent=40.0, **common)
    assert pruned.layers == direct.layers == [2, 0]
    assert pruned.nodes == direct.nodes
    for a, b in zip(pruned.edges, direct.edges, strict=True):
        assert a.source == b.source and a.target == b.target
        assert a.score == pytest.approx(b.score, abs=1e-9)


# --------------------------------------------------------------------------- #
# mode 2 / pruning
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("roots", ["last", "all"])
@pytest.mark.parametrize("percent", [20.0, 50.0, 100.0])
def test_mode2_equals_mode1_then_prune(lens, model, percent, roots):
    """Including where the cutoff falls inside a run of tied scores: this model
    has no attention, so its cross-position edges are all exactly zero."""
    common = dict(k=3, positions=[1, 2, 3])
    dense = build(lens, model, mode=1, **common)
    pruned = dense.prune(percent, roots=roots)
    direct = build(lens, model, mode=2, prune_percent=percent, roots=roots, **common)
    assert pruned.mode == direct.mode == 2
    assert pruned.nodes == direct.nodes
    assert len(pruned.edges) == len(direct.edges)
    for a, b in zip(pruned.edges, direct.edges, strict=True):
        assert a.source == b.source and a.target == b.target
        assert a.score == pytest.approx(b.score, abs=1e-9)


def test_prune_keeps_requested_fraction_per_level(lens, model):
    dense = build(lens, model, k=2, positions=[1, 2, 3])
    top_level = dense.edges_into(dense.layer_top)
    pruned = dense.prune(25.0, roots="all")
    kept = pruned.edges_into(pruned.layer_top)
    assert len(kept) == max(1, round(len(top_level) * 0.25))
    strongest = sorted(top_level, key=lambda e: -abs(e.score))[: len(kept)]
    assert {(e.source, e.target) for e in kept} == {
        (e.source, e.target) for e in strongest
    }


def test_prune_always_keeps_at_least_one_edge(lens, model):
    pruned = build(lens, model, k=2, positions=[1, 2]).prune(0.5)
    assert all(
        len(pruned.edges_into(l)) >= 1
        for l in pruned.layers
        # A level whose only survivors are error nodes has no incoming edges by
        # construction: error nodes are leaves.
        if l > pruned.layer_bottom and any(not n.is_error for n in pruned.nodes[l])
    )


def test_pruned_sources_disappear_from_deeper_levels(lens, model):
    dense = build(lens, model, k=3, positions=[1, 2, 3])
    pruned = dense.prune(20.0)
    for layer in pruned.layers:
        if layer == pruned.layer_top:
            continue
        survivors = set(pruned.nodes[layer])
        assert survivors == {e.source for e in pruned.edges_into(layer + 1)}
        assert {e.target for e in pruned.edges_into(layer)} <= survivors
    assert len(pruned.edges) < len(dense.edges)


def test_prune_is_idempotent_at_100_percent(lens, model):
    dense = build(lens, model, k=2, positions=[1, 2])
    assert len(dense.prune(100.0, roots="all").edges) == len(dense.edges)


# --------------------------------------------------------------------------- #
# roots: where mode 2 starts its descent
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("roots", ["last", [2]])
def test_roots_restrict_the_top_layer_to_the_output_position(lens, model, roots):
    common = dict(k=3, positions=[1, 2])
    dense = build(lens, model, mode=1, **common)
    pruned = dense.prune(20.0, roots=roots)
    assert {n.position for n in pruned.nodes[pruned.layer_top]} == {2}
    assert pruned.hparams["roots"] == [2]
    # and every edge into the top layer now lands on a root
    assert all(e.target.position == 2 for e in pruned.edges_into(pruned.layer_top))
    # only "all" keeps both blocks on top
    unrestricted = dense.prune(20.0, roots="all")
    assert {n.position for n in unrestricted.nodes[dense.layer_top]} == {1, 2}
    assert unrestricted.hparams["roots"] is None


def test_roots_spend_the_budget_on_the_cone_that_reaches_them(lens, model):
    """Same percentage, but none of it wasted on positions nobody reads."""
    common = dict(k=3, positions=[1, 2, 3])
    dense = build(lens, model, mode=1, **common)
    focused = dense.prune(20.0, roots="last")
    spread = dense.prune(20.0, roots="all")
    top = dense.layer_top
    assert len(focused.edges_into(top)) < len(spread.edges_into(top))
    # what survives is a subset of the dense graph's edges into the root block
    into_root = {
        (e.source, e.target)
        for e in dense.edges_into(top)
        if e.target.position == dense.positions[-1]
    }
    assert {(e.source, e.target) for e in focused.edges_into(top)} <= into_root


@pytest.mark.parametrize("roots", ["last", [13], ["i"]])
def test_mode2_with_roots_equals_dense_then_prune_with_roots(lens, model, roots):
    # positions 11..13 are the only ones whose characters occur once, so token
    # text names exactly one of them (the tiny tokenizer is character-level)
    common = dict(k=3, positions=[11, 12, 13])
    dense = build(lens, model, mode=1, **common)
    resolved = (
        roots
        if isinstance(roots, str)
        else _resolve_positions(roots, model.encode(PROMPT), False, model.tokenizer)
    )
    pruned = dense.prune(20.0, roots=resolved)
    direct = build(lens, model, mode=2, prune_percent=20.0, roots=roots, **common)
    assert pruned.nodes == direct.nodes
    assert len(pruned.edges) == len(direct.edges)
    for a, b in zip(pruned.edges, direct.edges, strict=True):
        assert a.source == b.source and a.target == b.target
        assert a.score == pytest.approx(b.score, abs=1e-9)


def test_roots_default_to_the_last_position(lens, model):
    common = dict(k=2, positions=[1, 2])
    default = build(lens, model, mode=2, prune_percent=30.0, **common)
    explicit = build(lens, model, mode=2, prune_percent=30.0, roots="last", **common)
    assert default.hparams["roots"] == [2]
    assert default.nodes == explicit.nodes
    assert len(default.edges) == len(explicit.edges)
    # mode 1 takes the same default and stays dense, keeping every position
    dense = build(lens, model, mode=1, **common)
    assert dense.hparams["roots"] is None
    assert {n.position for n in dense.nodes[dense.layer_top]} == {1, 2}


def test_roots_are_rejected_for_mode_1_and_when_unknown(lens, model):
    for value in ("last", "all", [1]):
        with pytest.raises(ValueError, match="roots applies to mode=2 only"):
            build(lens, model, mode=1, roots=value)
    with pytest.raises(ValueError, match="carry no concepts"):
        build(lens, model, mode=2, positions=[1, 2], roots=[3])
    with pytest.raises(ValueError, match="must be 'all', 'last'"):
        build(lens, model, mode=2, roots="final")
    with pytest.raises(ValueError, match="must be integers here"):
        build(lens, model, mode=1, k=2).prune(20.0, roots=["i"])


# --------------------------------------------------------------------------- #
# hyperparameters and validation
# --------------------------------------------------------------------------- #


def test_band_defaults_to_percentiles_of_the_fitted_layers(lens, model):
    kw = dict(k=2, positions=[-1])
    # fitted layers are [0,1,2,3]; the defaults are 20% and 95%
    circuit = build_jcircuit(lens, model, PROMPT, **kw)
    assert circuit.layer_bottom == 1  # round(0.20 * 3) == 1
    assert circuit.layer_top == 3  # round(0.95 * 3) == 3
    assert circuit.hparams["layer_bottom_percentile"] == 20.0
    assert circuit.hparams["layer_top_percentile"] == 95.0
    assert circuit.hparams["stride"] == 1

    narrow = build_jcircuit(
        lens,
        model,
        PROMPT,
        layer_top_percentile=67.0,
        layer_bottom_percentile=0.0,
        **kw,
    )
    assert (narrow.layer_bottom, narrow.layer_top) == (0, 2)

    # an explicit layer records that the percentile was not what chose that end
    explicit = build_jcircuit(lens, model, PROMPT, layer_bottom=0, **kw)
    assert explicit.layer_bottom == 0
    assert explicit.hparams["layer_bottom_percentile"] is None
    assert explicit.hparams["layer_top_percentile"] == 95.0
    assert build(lens, model, k=2).hparams["layer_top_percentile"] is None


def test_percentile_band_that_collapses_raises(lens, model):
    with pytest.raises(ValueError, match="percentiles 90.0..10.0"):
        build_jcircuit(
            lens, model, PROMPT, layer_top_percentile=10.0, layer_bottom_percentile=90.0
        )
    for bad in ({"layer_top_percentile": 101.0}, {"layer_bottom_percentile": -1.0}):
        name = next(iter(bad))
        with pytest.raises(ValueError, match=f"{name} must be in"):
            build_jcircuit(lens, model, PROMPT, **bad)


def test_invalid_stride_raises(lens, model):
    for stride in (0, -1):
        with pytest.raises(ValueError, match="stride must be >= 1"):
            build(lens, model, stride=stride)
    # 0..3 with stride 4 leaves no level at all
    with pytest.raises(ValueError, match="wider than the layer range"):
        build(lens, model, stride=4)


def test_dense_edge_budget_is_enforced(lens, model):
    with pytest.raises(ValueError, match="max_edges"):
        build(lens, model, k=3, max_edges=10)
    # mode 2 only materialises survivors, so the budget does not apply
    assert build(lens, model, k=3, mode=2, max_edges=10).edges


def test_invalid_arguments_raise(lens, model):
    with pytest.raises(ValueError, match="mode must be"):
        build(lens, model, mode=3)
    with pytest.raises(ValueError, match="k must be"):
        build(lens, model, k=0)
    with pytest.raises(ValueError, match="vjp_chunk"):
        build(lens, model, vjp_chunk=0)
    with pytest.raises(ValueError, match="prune_percent"):
        build(lens, model, mode=2, prune_percent=0.0)
    with pytest.raises(ValueError, match="prune_percent"):
        build(lens, model, k=2).prune(101.0)
    with pytest.raises(ValueError, match="must be above"):
        build_jcircuit(lens, model, PROMPT, layer_top=1, layer_bottom=1)
    with pytest.raises(ValueError, match="position .* out of range"):
        build(lens, model, positions=[999])
    with pytest.raises(ValueError, match="positions is empty"):
        build(lens, model, positions=[])


def test_unfitted_level_raises_but_a_stride_can_step_over_the_gap(model):
    partial = JacobianLens(
        jacobians={l: torch.eye(8) for l in (0, 2, 3)}, n_prompts=1, d_model=8
    )
    kw = dict(layer_top=3, layer_bottom=0, k=2, positions=[1, 2])
    with pytest.raises(ValueError, match="needs a fitted Jacobian; missing \\[1\\]"):
        build_jcircuit(partial, model, PROMPT, **kw)
    # only the levels need fitting, and stride 2 lands on 0 and 2, skipping 1
    stepped = build_jcircuit(partial, model, PROMPT, stride=2, **kw)
    assert stepped.layers == [2, 0]


def test_vjp_chunking_does_not_change_scores(lens, model):
    common = dict(k=3, positions=[1, 2, 3])
    whole = build(lens, model, vjp_chunk=64, **common)
    split = build(lens, model, vjp_chunk=1, **common)
    for a, b in zip(whole.edges, split.edges, strict=True):
        assert a.source == b.source and a.target == b.target
        assert a.score == pytest.approx(b.score, abs=1e-6)


def test_format_and_repr_render(lens, model):
    circuit = build(lens, model, k=2, positions=[1, 2])
    text = circuit.format()
    assert "JCircuit(mode=1" in text and "L3" in text and "pos" in text
    assert "attn" in text or "resid" in text
    assert isinstance(str(Node(1, 2, 3, "x", 0.5, 0)), str)
    assert isinstance(JCircuit({}, [], {"layer_top": 1, "layer_bottom": 0}), JCircuit)


# --------------------------------------------------------------------------- #
# coverage / error node
# --------------------------------------------------------------------------- #


def test_coverage_named_part_is_the_lone_projection_when_k_is_one(lens, model):
    """With one *named* concept per block the span is a line, so ``||P_J g||``
    must be exactly the magnitude of that concept's own edge divided by ``a_s``.

    The block's error node is a source too, but coverage deliberately excludes
    it — it measures what can be given a name.
    """
    circuit = build(lens, model, k=1)
    assert circuit.coverage
    for row in circuit.coverage:
        edges = [
            e
            for e in circuit.edges
            if e.target == row.target
            and e.source.layer == row.source_layer
            and e.source.position == row.source_position
            and not e.source.is_error
        ]
        assert len(edges) == 1
        edge = edges[0]
        if abs(edge.source.activation) < 1e-3:
            continue
        expected = abs(edge.score / edge.source.activation)
        assert row.named == pytest.approx(expected, rel=1e-4, abs=1e-6)


def test_coverage_is_a_bounded_share_of_the_gradient(lens, model):
    circuit = build(lens, model, k=3)
    assert circuit.coverage
    for row in circuit.coverage:
        assert 0.0 <= row.named <= row.total * (1 + 1e-5)
        if row.total == 0.0:
            continue
        assert 0.0 <= row.fraction <= 1.0 + 1e-5
        assert row.error == pytest.approx(1.0 - row.fraction)


def test_a_gradient_of_zero_reports_nan_rather_than_total_leakage(lens, model):
    """The tiny decoder has no attention, so a cross-position gradient is
    exactly zero. That is *no influence*, not influence the concepts missed."""
    circuit = build(lens, model, k=3)
    blind = [r for r in circuit.coverage if r.source_position != r.target.position]
    assert blind
    assert all(r.total == 0.0 and r.named == 0.0 for r in blind)
    assert all(r.fraction != r.fraction and r.error != r.error for r in blind)
    # ...and they must not poison the aggregate.
    assert 0.0 <= circuit.error_mass() <= 1.0


def test_a_full_rank_concept_set_names_the_whole_gradient(lens, model):
    """``k`` equal to ``d_model`` spans the residual space, so nothing leaks."""
    circuit = build(lens, model, k=8, selection="topk")
    live = [row for row in circuit.coverage if row.total > 0]
    assert live
    assert all(row.fraction == pytest.approx(1.0, abs=1e-4) for row in live)
    assert circuit.error_mass() == pytest.approx(0.0, abs=1e-4)


def test_fewer_concepts_name_less_of_the_gradient(lens, model):
    masses = [build(lens, model, k=k, selection="topk").error_mass() for k in (1, 2, 4)]
    assert masses == sorted(masses, reverse=True)
    assert all(0.0 <= m <= 1.0 for m in masses)


def test_coverage_rows_exist_exactly_for_causal_source_positions(lens, model):
    circuit = build(lens, model, k=2)
    stride = circuit.stride
    seen = {(r.target, r.source_layer, r.source_position) for r in circuit.coverage}
    expected = {
        (target, layer - stride, position)
        for layer in circuit.nodes
        if layer - stride in circuit.nodes
        for target in circuit.nodes[layer]
        # error nodes are never targets, so they have no incoming gradient
        if not target.is_error
        for position in circuit.positions
        if position <= target.position
    }
    assert seen == expected


def test_error_mass_restricts_to_one_source_layer(lens, model):
    circuit = build(lens, model, k=2)
    pooled = circuit.error_mass()
    per_layer = [circuit.error_mass(layer=l) for l in (0, 1, 2)]
    assert all(0.0 <= m <= 1.0 for m in per_layer)
    assert min(per_layer) <= pooled <= max(per_layer)


def test_pruning_keeps_coverage_for_surviving_targets_only(lens, model):
    dense = build(lens, model, k=3)
    pruned = dense.prune(20.0, roots="last")
    alive = {n for v in pruned.nodes.values() for n in v}
    assert pruned.coverage
    assert len(pruned.coverage) < len(dense.coverage)
    assert all(row.target in alive for row in pruned.coverage)
    # Coverage is a property of the target's own gradient, so the rows that
    # survive must be unchanged by the pruning, not recomputed.
    assert set(pruned.coverage) <= set(dense.coverage)


def test_dropping_tokens_drops_their_coverage(lens, model):
    circuit = build(lens, model, k=3).prune(50.0, roots="all")
    doomed = circuit.nodes[circuit.layer_top][0].token_id
    smaller = circuit.drop_tokens([doomed])
    assert all(row.target.token_id != doomed for row in smaller.coverage)


def test_ig_builds_carry_no_coverage(lens, model):
    circuit = build(lens, model, k=2, estimator="ig", ig_steps=2)
    assert circuit.coverage == ()
    assert circuit.error_mass() != circuit.error_mass()  # nan


def test_span_basis_ignores_duplicate_directions():
    from jlens.circuit import _span_basis

    v = torch.randn(8)
    v = v / v.norm()
    w = torch.randn(8)
    w = w - (w @ v) * v
    w = w / w.norm()
    assert _span_basis(torch.stack([v, v, v])).shape == (8, 1)
    assert _span_basis(torch.stack([v, w])).shape == (8, 2)
    assert _span_basis(torch.zeros(3, 8)).shape == (8, 0)


# --------------------------------------------------------------------------- #
# error nodes
# --------------------------------------------------------------------------- #


def test_error_direction_is_orthogonal_to_every_named_concept(lens, model):
    """That orthogonality is the whole point: the error node must carry only
    what no named concept can, or its edge double counts with theirs."""
    circuit = build(lens, model, k=3)
    for layer in circuit.nodes:
        for position in circuit.positions:
            block = circuit.nodes_at(layer, position)
            errors = [n for n in block if n.is_error]
            if not errors:
                continue
            r = circuit.direction(errors[0], lens, model)
            assert float(r.norm()) == pytest.approx(1.0, abs=1e-5)
            for node in block:
                if node.is_error:
                    continue
                v = circuit.direction(node, lens, model)
                assert float(r @ v) == pytest.approx(0.0, abs=1e-5)


def test_error_activation_is_the_length_of_what_is_left_over(lens, model):
    """``a = <v_hat, h>`` holds for an error node too, and equals ``||r||``."""
    circuit = build(lens, model, k=2)
    with ActivationRecorder(model.layers, at=[1]) as rec:
        model.forward(model.encode(PROMPT))
        h1 = rec.activations[1][0].detach().float()
    for node in circuit.nodes_at(1, 2):
        if not node.is_error:
            continue
        r = circuit.direction(node, lens, model)
        assert node.activation == pytest.approx(float(h1[2] @ r), abs=1e-5)
        assert node.activation > 0


def test_error_node_plus_named_span_accounts_for_the_whole_cell(lens, model):
    """``<h, g> = <P h, g> + <r, g>``: ablating the block's whole residual is
    the named span's effect plus the error node's, with nothing unaccounted."""
    circuit = build(lens, model, k=2, positions=[1, 2])
    layer, position = 1, 2
    block = circuit.nodes_at(layer, position)
    error = next(n for n in block if n.is_error)
    named = [n for n in block if not n.is_error]
    target = next(n for n in circuit.nodes_at(2, position) if not n.is_error)

    def readout():
        v = circuit.direction(target, lens, model)
        with torch.no_grad(), ActivationRecorder(model.layers, at=[2]) as rec:
            model.forward(model.encode(PROMPT))
            return float(rec.activations[2][0, position].float() @ v)

    basis = torch.stack([circuit.direction(n, lens, model) for n in named])

    @contextlib.contextmanager
    def project_out(directions):
        def hook(module, inputs, output):
            tensor = output if torch.is_tensor(output) else output[0]
            work = tensor.float().clone()
            h = work[:, position]
            work[:, position] = h - (h @ directions.T) @ directions
            edited = work.to(tensor.dtype)
            return edited if torch.is_tensor(output) else (edited, *tuple(output[1:]))

        handle = model.layers[layer].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    clean = readout()
    with project_out(basis):
        without_named = readout()
    error_edge = next(
        e for e in circuit.edges if e.source == error and e.target == target
    )
    # What the named concepts jointly explain, plus the error node's own edge,
    # is the effect of wiping the cell along all of them at once.
    with project_out(torch.cat([basis, circuit.direction(error, lens, model)[None]])):
        without_either = readout()
    assert (clean - without_either) == pytest.approx(
        (clean - without_named) + error_edge.score, abs=1e-4
    )


def test_error_nodes_can_be_pruned_away_like_any_other_node(lens, model):
    dense = build(lens, model, k=3)
    n_dense = sum(1 for v in dense.nodes.values() for n in v if n.is_error)
    pruned = dense.prune(10.0, roots="last")
    n_pruned = sum(1 for v in pruned.nodes.values() for n in v if n.is_error)
    assert n_dense > 0
    assert n_pruned < n_dense, "pruning must be able to drop an error node"
    # ...and whichever survive still carry edges, like any surviving source.
    survivors = {e.source for e in pruned.edges}
    assert all(
        n in survivors
        for v in pruned.nodes.values()
        for n in v
        if n.is_error and n.layer != pruned.layer_bottom
    )


def test_error_nodes_can_be_turned_off(lens, model):
    circuit = build(lens, model, k=3, error_nodes=False)
    assert not any(n.is_error for v in circuit.nodes.values() for n in v)
    assert circuit.error_directions == {}
    assert circuit.hparams["error_nodes"] is False


def test_direction_recovers_both_kinds_of_node(lens, model):
    circuit = build(lens, model, k=2)
    for nodes in circuit.nodes.values():
        for node in nodes:
            v = circuit.direction(node, lens, model)
            assert v.shape == (8,)
            assert float(v.norm()) == pytest.approx(1.0, abs=1e-5)
            assert (node.token_id == -1) == node.is_error
    missing = Node(
        layer=99, position=0, token_id=-1, token="<error>", activation=0.0, lens_rank=0
    )
    with pytest.raises(KeyError):
        circuit.direction(missing, lens, model)
