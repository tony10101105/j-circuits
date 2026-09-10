# Copyright 2026 Tung-Yu (Tony) Wu
# SPDX-License-Identifier: Apache-2.0
# Added by Tung-Yu (Tony) Wu and his Claude.
"""LLM token filtering: the graph surgery, and the request/reply handling.

The Claude call is exercised against a fake client, so these tests need no
network and no credentials — what they pin down is that a reply is parsed
strictly, that a hallucinated id cannot delete a concept, and that removing a
token takes its orphans with it.
"""

import json
import types

import pytest
import torch

from jlens.circuit import ERROR_TOKEN, ERROR_TOKEN_ID, build_jcircuit
from jlens.lens import JacobianLens
from jlens.token_filter import MAX_TOKENS_PER_CALL, select_noise_tokens
from tests.tiny import TinyDecoder

PROMPT = "spin spin web"


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


@pytest.fixture()
def circuit(lens, model):
    return build_jcircuit(
        lens, model, PROMPT, k=3, layer_top=3, layer_bottom=0, positions=[1, 2]
    )


def fake_client(payload, *, stop_reason="end_turn", record=None):
    """A stand-in for `anthropic.Anthropic` returning one canned reply."""

    def create(**kwargs):
        if record is not None:
            record.update(kwargs)
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return types.SimpleNamespace(
            stop_reason=stop_reason,
            content=[types.SimpleNamespace(type="text", text=text)],
        )

    return types.SimpleNamespace(messages=types.SimpleNamespace(create=create))


# --------------------------------------------------------------------------- #
# choosing what to drop
# --------------------------------------------------------------------------- #


def test_request_carries_the_prompt_and_every_offered_token(circuit):
    record = {}
    tokens = circuit.tokens()
    select_noise_tokens(
        PROMPT, tokens, client=fake_client({"remove": []}, record=record)
    )

    sent = record["messages"][0]["content"]
    assert PROMPT in sent
    for token_id, text in tokens:
        assert f"{token_id}\t{text!r}" in sent
    # a schema, so the reply cannot come back unparseable
    assert record["output_config"]["format"]["type"] == "json_schema"
    assert record["model"] and record["max_tokens"] > 0


def test_only_offered_ids_can_be_dropped(circuit):
    """A hallucinated id must not delete a concept that was never offered."""
    tokens = circuit.tokens()
    real = tokens[0][0]
    payload = {
        "remove": [
            {"id": real, "reason": "irrelevant"},
            {"id": 99999, "reason": "non_semantic"},
        ]
    }
    chosen = select_noise_tokens(PROMPT, tokens, client=fake_client(payload))
    assert chosen == {real: "irrelevant"}


def test_empty_token_list_makes_no_request():
    def explode(**kwargs):
        raise AssertionError("should not call the API for an empty list")

    client = types.SimpleNamespace(messages=types.SimpleNamespace(create=explode))
    assert select_noise_tokens(PROMPT, [], client=client) == {}


def test_bad_replies_raise_rather_than_silently_dropping_nothing(circuit):
    tokens = circuit.tokens()
    with pytest.raises(ValueError, match="not JSON"):
        select_noise_tokens(PROMPT, tokens, client=fake_client("sorry, no"))
    with pytest.raises(ValueError, match="declined"):
        select_noise_tokens(
            PROMPT, tokens, client=fake_client({"remove": []}, stop_reason="refusal")
        )
    with pytest.raises(ValueError, match="max_tokens"):
        select_noise_tokens(
            PROMPT, tokens, client=fake_client({"remove": []}, stop_reason="max_tokens")
        )


def test_oversized_token_list_raises_before_spending_a_call():
    tokens = [(i, f"tok{i}") for i in range(MAX_TOKENS_PER_CALL + 1)]

    def explode(**kwargs):
        raise AssertionError("should not call the API")

    client = types.SimpleNamespace(messages=types.SimpleNamespace(create=explode))
    with pytest.raises(ValueError, match="MAX_TOKENS_PER_CALL"):
        select_noise_tokens(PROMPT, tokens, client=client)


# --------------------------------------------------------------------------- #
# the graph surgery
# --------------------------------------------------------------------------- #


def test_tokens_lists_each_named_concept_once(circuit):
    tokens = circuit.tokens()
    ids = [i for i, _ in tokens]
    assert len(ids) == len(set(ids))
    assert set(ids) == {
        n.token_id for v in circuit.nodes.values() for n in v if not n.is_error
    }
    assert tokens == sorted(tokens, key=lambda pair: pair[1])


def test_error_nodes_are_never_offered_to_the_filter(circuit):
    """'<error>' reads as markup, so a filter that sees it deletes every one.

    Whether the graph carries error nodes is ``build_jcircuit(error_nodes=)``,
    a build-time choice; a filter over vocabulary does not get to revisit it.
    """
    assert any(n.is_error for v in circuit.nodes.values() for n in v)
    tokens = circuit.tokens()
    assert ERROR_TOKEN_ID not in {i for i, _ in tokens}
    assert ERROR_TOKEN not in {t for _, t in tokens}

    record = {}
    select_noise_tokens(
        PROMPT, tokens, client=fake_client({"remove": []}, record=record)
    )
    assert ERROR_TOKEN not in record["messages"][0]["content"]


def test_a_hallucinated_error_id_cannot_delete_the_error_nodes(circuit):
    """Second line of defence: unoffered ids are already ignored."""
    payload = {"remove": [{"id": ERROR_TOKEN_ID, "reason": "non_semantic"}]}
    chosen = select_noise_tokens(PROMPT, circuit.tokens(), client=fake_client(payload))
    assert chosen == {}
    kept = circuit.drop_tokens(chosen)
    assert sum(1 for v in kept.nodes.values() for n in v if n.is_error) == sum(
        1 for v in circuit.nodes.values() for n in v if n.is_error
    )


def test_drop_tokens_removes_the_concept_everywhere(circuit):
    victim = circuit.nodes[circuit.layer_top][0].token_id
    filtered = circuit.drop_tokens([victim])
    assert not any(n.token_id == victim for v in filtered.nodes.values() for n in v)
    assert not any(
        e.source.token_id == victim or e.target.token_id == victim
        for e in filtered.edges
    )
    assert filtered.hparams["dropped_token_ids"] == [victim]
    # the original is untouched
    assert any(n.token_id == victim for v in circuit.nodes.values() for n in v)


def test_drop_tokens_leaves_no_dangling_nodes_or_edges(circuit):
    filtered = circuit.drop_tokens([n.token_id for n in circuit.nodes[2][:2]])
    alive = {n for v in filtered.nodes.values() for n in v}
    for edge in filtered.edges:
        assert edge.source in alive and edge.target in alive
    incoming = {e.target for e in filtered.edges}
    outgoing = {e.source for e in filtered.edges}
    for layer, nodes in filtered.nodes.items():
        for node in nodes:
            # Error nodes are sources only — being incoming-free is what they
            # are, not a dangling edge.
            assert layer == filtered.layers[-1] or node.is_error or node in incoming, (
                f"{node} unsupported"
            )
            assert layer == filtered.layers[0] or node in outgoing, f"{node} unused"


def test_dropping_cascades_past_the_first_layer(lens, model):
    """Removing a token can orphan a node, which can orphan its neighbour."""
    dense = build_jcircuit(
        lens, model, PROMPT, k=3, layer_top=3, layer_bottom=0, positions=[2]
    )
    circuit = dense.prune(20.0, roots="all")
    top = circuit.layer_top
    victims = [n.token_id for n in circuit.nodes[top]]
    filtered = circuit.drop_tokens(victims)
    # with every root gone, nothing below it can still reach an output
    assert sum(len(v) for v in filtered.nodes.values()) == 0
    assert filtered.edges == []


def test_dropping_unknown_ids_is_a_no_op(circuit):
    filtered = circuit.drop_tokens([99999])
    assert filtered.nodes == circuit.nodes
    assert filtered.edges == circuit.edges
