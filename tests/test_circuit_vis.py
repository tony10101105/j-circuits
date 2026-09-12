# Copyright 2026 Tung-Yu (Tony) Wu
# SPDX-License-Identifier: Apache-2.0
# Added by Tung-Yu (Tony) Wu and his Claude.
"""SVG rendering of a J-circuit: well-formedness, encoding, and file output."""

import re
import xml.etree.ElementTree as ET
from dataclasses import replace

import pytest
import torch

from jlens.circuit import JCircuit, build_jcircuit
from jlens.circuit_vis import render_svg, save_svg
from jlens.lens import JacobianLens
from tests.tiny import TinyDecoder

PROMPT = "spin spin web"


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


@pytest.fixture()
def circuit(lens, model):
    """Two position blocks, so cross-position edges exist in the figure.

    Dense: these tests are about what the renderer draws, so the graph handed
    to it should still carry every edge.
    """
    return build_jcircuit(
        lens,
        model,
        PROMPT,
        k=3,
        layer_top=3,
        layer_bottom=0,
        positions=[1, 2],
        prune_percent=100.0,
        roots="all",
    )


def _layer_labels(svg: str) -> set[str]:
    """The ``L<n>`` row labels actually rendered as text (not path data)."""
    ns = "{http://www.w3.org/2000/svg}"
    texts = (t.text or "" for t in ET.fromstring(svg).findall(f"{ns}text"))
    return {t for t in texts if t.startswith("L") and t[1:].isdigit()}


def test_output_is_well_formed_and_self_contained(circuit):
    svg = render_svg(circuit)
    root = ET.fromstring(svg)  # raises on malformed markup
    assert root.tag.endswith("svg")
    assert root.get("viewBox") and root.get("role") == "img" and root.get("aria-label")
    # an intrinsic size too: viewBox alone is enough for a browser, but native
    # viewers refuse to open an SVG that does not say how big it is
    box = [float(v) for v in root.get("viewBox").split()]
    assert (float(root.get("width")), float(root.get("height"))) == (box[2], box[3])
    # a strict CSP forbids scripts and anything fetched from elsewhere; the
    # xmlns declaration is the one URL that legitimately appears
    for banned in ("<script", "<style", "<image", "<foreignObject", "href=", "src="):
        assert banned not in svg


def test_every_node_and_edge_is_drawn(circuit):
    svg = render_svg(circuit, min_score_fraction=0.0)
    root = ET.fromstring(svg)
    ns = "{http://www.w3.org/2000/svg}"
    n_nodes = sum(len(v) for v in circuit.nodes.values())
    assert len(root.findall(f"{ns}rect")) == n_nodes
    # one path per edge, plus the two arrowhead markers inside <defs>
    assert len(root.findall(f"{ns}path")) == len(circuit.edges)


def test_the_four_edge_kinds_are_styled_apart(circuit):
    svg = render_svg(
        circuit,
        min_score_fraction=0.0,
        carry_color="#111111",
        compute_color="#222222",
        attention_color="#333333",
        error_color="#444444",
    )
    # An edge out of an error node is its own kind whatever it rode, so it is
    # excluded before the carry/compute/attention split.
    named = [e for e in circuit.edges if not e.source.is_error]
    error = [e for e in circuit.edges if e.source.is_error]
    same = [e for e in named if not e.cross_position]
    attention = [e for e in named if e.cross_position]
    compute = [e for e in same if abs(e.computed) > 0.15 * abs(e.score)]
    carry = [e for e in same if e not in compute]
    assert carry and compute and attention and error, "need all four kinds"
    # error_color paints the edges *and* the error node boxes, so count paths
    n_error_nodes = sum(1 for v in circuit.nodes.values() for n in v if n.is_error)
    assert svg.count('stroke="#444444"') == len(error) + n_error_nodes
    assert svg.count('stroke="#222222"') == len(compute)
    assert svg.count('stroke="#333333"') == len(attention)
    # error node boxes are dashed too, so they join the carry edges in the count
    assert svg.count("stroke-dasharray") == len(carry) + n_error_nodes


def test_ink_and_background_make_a_file_readable_on_any_canvas(circuit):
    """Embedding wants to inherit the page; a standalone file cannot afford to."""
    ns = "{http://www.w3.org/2000/svg}"
    embedded = render_svg(circuit)
    assert 'fill="currentColor"' in embedded and 'stroke="currentColor"' in embedded
    assert "<rect" in embedded  # node boxes, but no backdrop
    n_nodes = sum(len(v) for v in circuit.nodes.values())

    standalone = render_svg(circuit, ink="#1a1a1a", background="#ffffff")
    assert "currentColor" not in standalone
    rects = ET.fromstring(standalone).findall(f"{ns}rect")
    assert len(rects) == n_nodes + 1
    backdrop = rects[0]  # painted first, so nothing hides behind it
    box = [float(v) for v in ET.fromstring(standalone).get("viewBox").split()]
    assert backdrop.get("fill") == "#ffffff"
    assert (float(backdrop.get("width")), float(backdrop.get("height"))) == (
        box[2],
        box[3],
    )


def test_labels_are_capped_and_staggered(circuit):
    ns = "{http://www.w3.org/2000/svg}"

    def n_texts(svg):
        return len(ET.fromstring(svg).findall(f"{ns}text"))

    # fixed chrome: one label per layer, per node, and per position block
    chrome = (
        len(circuit.layers)
        + sum(len(v) for v in circuit.nodes.values())
        + len(circuit.positions)
    )
    assert n_texts(render_svg(circuit, max_labels_per_level=0)) == chrome
    for cap in (1, 2, 3):
        extra = n_texts(render_svg(circuit, max_labels_per_level=cap)) - chrome
        assert 0 < extra <= cap * (len(circuit.layers) - 1)


def test_positions_argument_crops_the_figure(circuit):
    one = render_svg(circuit, positions=[2])
    ns = "{http://www.w3.org/2000/svg}"
    n_boxes = len(ET.fromstring(one).findall(f"{ns}rect"))
    assert n_boxes == sum(len(circuit.nodes_at(l, 2)) for l in circuit.layers)
    assert "pos 2" in one and "pos 1" not in one
    with pytest.raises(ValueError, match="no concepts left"):
        render_svg(circuit, positions=[99])


def test_position_labels_caption_the_blocks(circuit):
    svg = render_svg(circuit, position_labels={1: " web", 2: " spins"})
    assert "␣web" in svg and "␣spins" in svg


def test_too_many_columns_raises(lens, model, monkeypatch):
    monkeypatch.setattr("jlens.circuit_vis.MAX_COLUMNS", 48)
    wide = build_jcircuit(
        lens,
        model,
        PROMPT,
        k=8,
        layer_top=3,
        layer_bottom=0,
        prune_percent=100.0,
        roots="all",
    )
    with pytest.raises(ValueError, match="MAX_COLUMNS"):
        render_svg(wide)


def test_layers_argument_crops_the_figure(circuit):
    cropped = render_svg(circuit, layers=[3, 2])
    ns = "{http://www.w3.org/2000/svg}"
    root = ET.fromstring(cropped)
    assert len(root.findall(f"{ns}rect")) == len(circuit.nodes[3]) + len(
        circuit.nodes[2]
    )
    assert _layer_labels(cropped) == {"L3", "L2"}
    assert _layer_labels(render_svg(circuit)) == {"L3", "L2", "L1", "L0"}


@pytest.mark.parametrize(
    "token,expected",
    [
        (" ____", "␣_×4"),  # filler tokens draw as a rule without this
        ("____", "_×4"),
        ("  ", "␣×2"),
        ('."\n\n', ".&quot;↵×2"),
        (" spider", "␣spider"),  # letters are never collapsed
        ("aaa", "aaa"),
        ("蜘蛛", "蜘蛛"),
        ('!"', "!&quot;"),  # different characters are not a run
    ],
)
def test_labels_are_readable(token, expected):
    from jlens.circuit_vis import _label

    assert _label(token) == expected


def test_two_ids_spelled_the_same_get_their_own_columns(circuit):
    """Vocabularies contain distinct ids that decode identically (Qwen3 has 11,
    all undecodable-byte tokens). Columns key on the id, so they do not stack."""
    ns = "{http://www.w3.org/2000/svg}"
    layer, position = circuit.layer_top, circuit.positions[-1]
    block = circuit.nodes_at(layer, position)
    assert len(block) >= 2
    # respell one node like its neighbour, keeping its own id
    layer_nodes = list(circuit.nodes[layer])
    layer_nodes[layer_nodes.index(block[1])] = replace(block[1], token=block[0].token)
    clashing = JCircuit(
        nodes={**circuit.nodes, layer: layer_nodes},
        edges=[],
        hparams=circuit.hparams,
    )
    svg = render_svg(clashing, layers=[layer])
    xs = [float(r.get("x")) for r in ET.fromstring(svg).findall(f"{ns}rect")]
    assert len(xs) == len(set(xs)) == len(circuit.nodes[layer])


def test_whitespace_in_tokens_is_made_visible(lens, model):
    circuit = build_jcircuit(
        lens,
        model,
        PROMPT,
        k=2,
        layer_top=2,
        layer_bottom=0,
        positions=[2],
        prune_percent=100.0,
        roots="all",
    )
    svg = render_svg(circuit)
    for node in circuit.nodes_at(2, 2):
        if " " in node.token:
            assert "␣" in svg
            break


def test_empty_selection_raises(circuit):
    with pytest.raises(ValueError, match="no layers to draw"):
        render_svg(circuit, layers=[])


def test_save_svg_creates_missing_parent_directories(circuit, tmp_path):
    target = tmp_path / "exp" / "demo" / "test.svg"
    markup = save_svg(circuit, target)
    assert target.read_text(encoding="utf-8") == markup
    # and a bare filename in the cwd still works (parent is ".")
    plain = tmp_path / "plain.svg"
    save_svg(circuit, str(plain))
    assert plain.exists()


def test_weak_edges_are_dropped_and_faded(circuit):
    ns = "{http://www.w3.org/2000/svg}"

    def n_paths(svg):
        # two arrowhead markers live in <defs> and also use <path>
        return len(ET.fromstring(svg).findall(f"{ns}path"))

    assert n_paths(render_svg(circuit, min_score_fraction=0.0)) == len(circuit.edges)
    trimmed = render_svg(circuit, min_score_fraction=0.5)
    # thinning is per kind, so each kind keeps its own strongest edges
    from jlens.circuit_vis import _kind

    groups: dict[str, list] = {}
    for edge in circuit.edges:
        groups.setdefault(_kind(edge, 0.15), []).append(edge)
    expected = sum(
        1
        for g in groups.values()
        for e in g
        if abs(e.score) >= 0.5 * max(abs(x.score) for x in g)
    )
    assert n_paths(trimmed) == expected < len(circuit.edges)
    # every kind survives the cut, including the far weaker attention edges
    assert len(groups) == 4
    for colour in ("#8a94a6", "#0e7a72", "#b26308", "#9a4fbf"):
        assert f'stroke="{colour}"' in trimmed
    # the strongest edge is drawn more opaquely than the weakest kept one
    opacities = sorted(
        float(m) for m in re.findall(r'opacity="([0-9.]+)" marker-end', trimmed)
    )
    assert opacities[0] < opacities[-1]


def test_numbers_are_recorded_not_printed(circuit):
    """Width and opacity carry the magnitude; the exact values ride a <title>."""
    ns = "{http://www.w3.org/2000/svg}"
    svg = render_svg(circuit, min_score_fraction=0.0)
    root = ET.fromstring(svg)
    printed = [
        (t.text or "")
        for t in root.findall(f"{ns}text")
        if (t.text or "").startswith(("+", "-"))
    ]
    assert not printed, f"no numeric edge labels by default, got {printed[:3]}"

    titles = [t.text or "" for t in root.findall(f"{ns}path/{ns}title")]
    assert len(titles) == len(circuit.edges)
    assert all("A=" in t and "computed=" in t for t in titles)
    # and a title carries a real edge's numbers verbatim
    strongest = max(circuit.edges, key=lambda e: abs(e.score))
    assert any(f"A={strongest.score:+.4f}" in t for t in titles)


def test_edge_labels_show_computed_with_significant_figures(circuit):
    """Labels carry `computed`, and a weak edge must not round to "+0.0"."""
    ns = "{http://www.w3.org/2000/svg}"
    svg = render_svg(circuit, min_score_fraction=0.0, max_labels_per_level=3)
    printed = {
        (t.text or "")
        for t in ET.fromstring(svg).findall(f"{ns}text")
        if (t.text or "").startswith(("+", "-"))
    }
    assert printed, "expected some edge labels"
    assert not {p for p in printed if float(p) == 0.0}, "a label rounded away"
    # each printed value is some edge's computed, not its score
    from jlens.circuit_vis import _format_value

    computed = {_format_value(e.computed) for e in circuit.edges}
    assert printed <= computed
