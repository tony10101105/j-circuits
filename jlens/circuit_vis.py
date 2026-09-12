# Copyright 2026 Tung-Yu (Tony) Wu
# SPDX-License-Identifier: Apache-2.0
# Added by Tung-Yu (Tony) Wu and his Claude.
"""Render a :class:`~jlens.circuit.JCircuit` as a self-contained SVG.

Layout follows the computation: the earliest layer is at the bottom, the latest
at the top, and every edge points upward. Each layer is sliced left to right
into its token-position blocks, and within a block a concept keeps a fixed
column across layers — so the residual stream carrying a concept forward draws
as a straight vertical line, and anything slanted is the model doing something.

Edges come in the four kinds the mechanism actually has:

- **carry** — same position, dominated by :attr:`~jlens.circuit.Edge.identity`:
  the residual stream moving a concept along unchanged. Dashed, muted.
- **compute** — same position, with real
  :attr:`~jlens.circuit.Edge.computed` mass: the intervening block derived the
  target from the source. Solid, accent.
- **attention** — crossing positions. These have no residual path at all
  (``identity`` is 0), so every bit of their score is information moved between
  positions. Solid, second accent.
- **error** — out of a block's error node, whatever it rode. Its own colour,
  because the point is that the *source* has no name: the influence is real and
  measured, but the graph cannot say what it is.

Stroke width scales with ``|score|`` in all four cases.

Error nodes themselves draw dashed and coloured rather than as plain boxes, so
a reader never mistakes one for one more thing the model is thinking about.
They are leaves — sources only — so nothing ever points into them.

The output is plain SVG markup — no script, no external references — so it can
be written to a file, embedded in a notebook, or pasted inline into a page.
Colours default to hex literals; pass CSS ``var(--token)`` strings instead when
embedding in a themed document.
"""

from __future__ import annotations

import re
from pathlib import Path

from jlens.circuit import Edge, JCircuit, Node

#: Refuse to draw a figure wider than this many concept columns. A dense
#: multi-position circuit blows past it easily; the error names the arguments
#: that crop it back down.
MAX_COLUMNS = 200

_CHAR_WIDTH = 7.0
_ESCAPES = (("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"), ('"', "&quot;"))


def _escape(text: str) -> str:
    for raw, ref in _ESCAPES:
        text = text.replace(raw, ref)
    return text


def _collapse_runs(text: str) -> str:
    """Rewrite a run of the same non-alphanumeric character as ``c×n``.

    Vocabularies carry filler tokens like ``"____"`` and ``"------"``. In a
    monospace face those draw as a rule sitting on the baseline, which is
    indistinguishable from an empty box — ``_×4`` says what the token is.
    Letters and digits are left alone, so ``"aaa"`` stays ``"aaa"``.
    """

    def replace(match: re.Match[str]) -> str:
        char = match.group(1)
        return match.group(0) if char.isalnum() else f"{char}×{len(match.group(0))}"

    # DOTALL so a run of newlines counts too — "." skips them by default.
    return re.sub(r"(.)\1+", replace, text, flags=re.DOTALL)


def _label(token: str, *, max_chars: int = 13) -> str:
    """Printable node label: repeats collapsed, whitespace shown, long tokens cut."""
    shown = _collapse_runs(token)
    shown = shown.replace(" ", "␣").replace("\n", "↵").replace("\t", "⇥")
    if len(shown) > max_chars:
        shown = shown[: max_chars - 1] + "…"
    return _escape(shown)


def _stroke_width(score: float, largest: float) -> float:
    """Edge thickness from ``|score|``, square-rooted so weak edges stay visible."""
    if largest <= 0:
        return 1.0
    return 0.7 + 3.3 * (abs(score) / largest) ** 0.5


def _format_value(value: float) -> str:
    """Two significant figures. Attention edges are tens of times weaker than
    same-position ones, so fixed decimals would render them all as ``+0.0``."""
    return f"{value:+.2g}"


def _kind(edge: Edge, compute_threshold: float) -> str:
    """``"error"``, ``"attention"``, ``"compute"``, or ``"carry"`` for one edge."""
    # Checked first: what matters about an edge out of an error node is that its
    # source has no name, not whether it rode the residual or attention.
    if edge.source.is_error:
        return "error"
    if edge.cross_position:
        return "attention"
    if abs(edge.computed) > compute_threshold * max(abs(edge.score), 1e-12):
        return "compute"
    return "carry"


def render_svg(
    circuit: JCircuit,
    *,
    layers: list[int] | None = None,
    positions: list[int] | None = None,
    position_labels: dict[int, str] | None = None,
    row_height: float = 66.0,
    node_height: float = 24.0,
    compute_threshold: float = 0.15,
    min_score_fraction: float = 0.02,
    carry_color: str = "#8a94a6",
    compute_color: str = "#0e7a72",
    attention_color: str = "#b26308",
    error_color: str = "#9a4fbf",
    ink: str = "currentColor",
    background: str | None = None,
    max_labels_per_level: int = 0,
    aria_label: str | None = None,
) -> str:
    """Return SVG markup drawing ``circuit``.

    Args:
        circuit: The circuit to draw. Pruned circuits read best; a dense graph
            over many positions produces a very busy figure.
        layers: Layers to include, default all the circuit has. Pass a slice of
            :attr:`~jlens.circuit.JCircuit.layers` to crop a deep circuit.
        positions: Position blocks to include, default all the circuit has.
            The main lever for cropping a wide circuit.
        position_labels: Optional caption per position — the prompt's token
            there reads far better than a bare index.
        row_height: Vertical distance between layers.
        node_height: Height of a concept box.
        compute_threshold: ``|computed| / |score|`` above which a same-position
            edge counts as compute rather than carry.
        min_score_fraction: Drop edges weaker than this fraction of the
            strongest edge *of their own kind*. A dense circuit is mostly
            negligible edges, and drawing them buries the ones that matter;
            thinning per kind keeps the attention structure visible even though
            it is far weaker than the residual stream. Set to 0 to keep all.
        carry_color: Colour for residual-carry edges.
        compute_color: Colour for edges the intervening block computed.
        attention_color: Colour for edges crossing positions.
        error_color: Colour for edges out of a block's error node.
        ink: Colour of the boxes, node names, and axis labels. The default
            ``"currentColor"`` inherits the surrounding page's text colour, so
            an embedded figure follows a light or dark theme by itself — but a
            standalone file opened on a dark canvas then draws dark on dark and
            looks empty. Pass a literal colour when writing a file to view on
            its own.
        background: Fill painted behind the figure, default none (transparent),
            which lets an embedded figure sit on the page's own background. A
            standalone file is safer with one: the canvas an image viewer paints
            behind a transparent SVG is not something the file controls, and a
            dark one hides everything drawn in ``ink``.
        max_labels_per_level: How many edges per layer pair get their
            ``computed`` value printed as text, strongest first. Defaults to 0:
            stroke width and opacity already encode magnitude, so printed
            numbers mostly add clutter. Every edge carries its exact ``score``,
            ``identity`` and ``computed`` in an SVG ``<title>`` regardless, so
            the values stay in the file and appear on hover.
        aria_label: Accessible description; a summary is generated when omitted.

    Raises:
        ValueError: If nothing is left to draw, a named root position is not
            among those drawn, or the figure would exceed :data:`MAX_COLUMNS`
            concept columns.
    """
    drawn = sorted(layers if layers is not None else circuit.layers, reverse=True)
    drawn = [l for l in drawn if circuit.nodes.get(l)]
    if not drawn:
        raise ValueError("circuit has no layers to draw")
    blocks = sorted(positions if positions is not None else circuit.positions)
    kept_nodes = {
        layer: [n for n in circuit.nodes[layer] if n.position in set(blocks)]
        for layer in drawn
    }
    drawn = [l for l in drawn if kept_nodes[l]]
    if not drawn:
        raise ValueError("no concepts left after cropping to the requested positions")

    # One column per (position, concept), grouped so each position is a block.
    # Keyed on token *id*: a vocabulary can spell two ids the same way (Qwen3
    # has 11 such strings, all undecodable-byte tokens, and they do turn up in
    # real circuits), and keying on the string would stack them in one column.
    column: dict[tuple[int, int], int] = {}
    block_span: dict[int, tuple[int, int]] = {}
    for position in blocks:
        start = len(column)
        for layer in drawn:
            for node in kept_nodes[layer]:
                if node.position == position:
                    column.setdefault((position, node.token_id), len(column))
        if len(column) > start:
            block_span[position] = (start, len(column) - 1)
    if not column:
        raise ValueError("no concepts left after cropping to the requested positions")
    if len(column) > MAX_COLUMNS:
        raise ValueError(
            f"this figure needs {len(column)} concept columns (> MAX_COLUMNS="
            f"{MAX_COLUMNS}). Crop it with positions=, layers=, a smaller k, or "
            "prune the circuit first."
        )

    labels = {n.token_id: _label(n.token) for layer in drawn for n in kept_nodes[layer]}
    label_chars = max((len(v) for v in labels.values()), default=6)
    box_w = max(64.0, label_chars * _CHAR_WIDTH + 18.0)
    col_gap = box_w + 16.0
    block_gap = 26.0

    # Blocks are separated by extra space, so a column's x depends on how many
    # block boundaries precede it.
    boundaries = sorted(start for start, _ in block_span.values())

    def column_x(index: int) -> float:
        preceding = sum(1 for b in boundaries if b <= index) - 1
        return 58.0 + index * col_gap + preceding * block_gap + box_w / 2

    header = 34.0 if block_span else 8.0
    width = column_x(len(column) - 1) + box_w / 2 + 18.0
    height = header + (len(drawn) - 1) * row_height + node_height + 34.0

    row_of = {layer: i for i, layer in enumerate(drawn)}
    pos_of: dict[Node, tuple[float, float]] = {}
    for layer in drawn:
        for node in kept_nodes[layer]:
            x = column_x(column[(node.position, node.token_id)])
            y = header + row_of[layer] * row_height + node_height / 2
            pos_of[node] = (x, y)

    edges = [e for e in circuit.edges if e.source in pos_of and e.target in pos_of]
    largest = max((abs(e.score) for e in edges), default=1.0)
    # Thin each kind against its own strongest edge, not the figure's. A
    # cross-position edge is typically tens of times weaker than a
    # same-position one, so a global cut would silently delete every attention
    # edge in the circuit. Width and opacity still come from the global scale,
    # so a surviving weak edge still *looks* weak.
    if min_score_fraction > 0:
        grouped: dict[str, list[Edge]] = {}
        for edge in edges:
            grouped.setdefault(_kind(edge, compute_threshold), []).append(edge)
        edges = [
            edge
            for group in grouped.values()
            for edge in group
            if abs(edge.score) >= min_score_fraction * max(abs(e.score) for e in group)
        ]
    colours = {
        "carry": carry_color,
        "compute": compute_color,
        "attention": attention_color,
        "error": error_color,
    }

    out: list[str] = []
    out.append(
        # width/height as well as viewBox: a viewBox alone scales correctly in
        # a browser, but several native viewers (macOS Preview, Windows Photos,
        # some editors) treat an SVG with no intrinsic size as unopenable.
        f'<svg width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
        f'aria-label="{_escape(aria_label or _auto_label(drawn, blocks))}" '
        'xmlns="http://www.w3.org/2000/svg">'
    )
    out.append("<defs>")
    for name, colour in colours.items():
        out.append(
            # userSpaceOnUse keeps arrowheads a constant size; the default
            # scales them by stroke-width, so the heaviest edges would sprout
            # arrowheads several times larger than the boxes they point at.
            f'<marker id="jc-{name}" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" '
            'orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{colour}"/></marker>'
        )
    out.append("</defs>")
    if background is not None:
        out.append(
            f'<rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" '
            f'fill="{background}"/>'
        )

    # Position block headers.
    for position, (start, end) in block_span.items():
        left = column_x(start) - box_w / 2
        right = column_x(end) + box_w / 2
        caption = position_labels.get(position) if position_labels else None
        text = f"pos {position}" + (f"  {_label(caption)}" if caption else "")
        out.append(
            f'<text x="{(left + right) / 2:.1f}" y="{header - 18:.1f}" '
            'text-anchor="middle" font-family="monospace" font-size="11" '
            f'fill="{ink}" opacity="0.62">{text}</text>'
        )
        out.append(
            f'<line x1="{left:.1f}" y1="{header - 12:.1f}" x2="{right:.1f}" '
            f'y2="{header - 12:.1f}" stroke="{ink}" stroke-width="1" '
            'opacity="0.22"/>'
        )

    # Label only the strongest non-carry edges of each layer pair; a dense
    # level has far too many to annotate without collisions.
    labelled: dict[Edge, int] = {}
    if max_labels_per_level > 0:
        by_level: dict[int, list[Edge]] = {}
        for edge in edges:
            # An edge whose computed part renders as zero has nothing to say —
            # in a model without attention every cross-position edge is exactly
            # that, and labelling them would bury the real annotations.
            if _kind(edge, compute_threshold) != "carry" and float(
                _format_value(edge.computed)
            ):
                by_level.setdefault(edge.target.layer, []).append(edge)
        for level in by_level.values():
            level.sort(key=lambda e: -abs(e.computed))
            for slot, edge in enumerate(level[:max_labels_per_level]):
                labelled[edge] = slot

    # Edges first so the concept boxes sit on top of them.
    for edge in sorted(edges, key=lambda e: abs(e.score)):
        x1, y1 = pos_of[edge.source]
        x2, y2 = pos_of[edge.target]
        y1 -= node_height / 2  # leave the top of the source box
        y2 += node_height / 2  # arrive at the bottom of the target box
        kind = _kind(edge, compute_threshold)
        mid_y = (y1 + y2) / 2
        path = (
            f"M {x1:.1f} {y1:.1f} C {x1:.1f} {mid_y:.1f}, "
            f"{x2:.1f} {mid_y:.1f}, {x2:.1f} {y2:.1f}"
        )
        dash = ' stroke-dasharray="4 3"' if kind == "carry" else ""
        # Opacity tracks magnitude as well as width. A cross-position edge is
        # typically tens of times weaker than a same-position one, and drawing
        # it at full strength lets the weakest edges dominate the figure.
        relative = (abs(edge.score) / largest) ** 0.5 if largest > 0 else 1.0
        opacity = (0.18 + 0.72 * relative) * (0.65 if kind == "carry" else 1.0)
        # The numbers are recorded, not printed: stroke width and opacity carry
        # the magnitude visually, and a <title> keeps the exact values one hover
        # away without adding ink to the figure.
        tip = _escape(
            f"{edge.source.token!r}@L{edge.source.layer}:{edge.source.position}"
            f" -> {edge.target.token!r}@L{edge.target.layer}:{edge.target.position}"
            f"  A={edge.score:+.4f}  identity={edge.identity:+.4f}"
            f"  computed={edge.computed:+.4f}  [{kind}]"
        )
        out.append(
            f'<path d="{path}" fill="none" stroke="{colours[kind]}" '
            f'stroke-width="{_stroke_width(edge.score, largest):.2f}"{dash} '
            f'opacity="{opacity:.2f}" marker-end="url(#jc-{kind})">'
            f"<title>{tip}</title></path>"
        )
        if edge in labelled:
            # Stagger by rank so several labels on one level do not overlap.
            offset = 4 + labelled[edge] * 11
            out.append(
                f'<text x="{(x1 + x2) / 2:.1f}" y="{mid_y - offset:.1f}" '
                'text-anchor="middle" font-family="monospace" font-size="10" '
                f'fill="{colours[kind]}">{_format_value(edge.computed)}</text>'
            )

    # Layer labels and concept boxes.
    for layer in drawn:
        y = header + row_of[layer] * row_height
        out.append(
            f'<text x="46" y="{y + node_height * 0.72:.1f}" text-anchor="end" '
            f'font-family="monospace" font-size="11" fill="{ink}" '
            f'opacity="0.55">L{layer}</text>'
        )
        for node in kept_nodes[layer]:
            cx, cy = pos_of[node]
            # An error node is not a concept: dashed and coloured, so it never
            # reads as one more thing the model is thinking about.
            stroke = error_color if node.is_error else ink
            dash = ' stroke-dasharray="3 2"' if node.is_error else ""
            out.append(
                f'<rect x="{cx - box_w / 2:.1f}" y="{y:.1f}" width="{box_w:.1f}" '
                f'height="{node_height:.1f}" rx="6" fill="none" stroke="{stroke}" '
                f'stroke-width="1.1"{dash} opacity="0.8"/>'
            )
            out.append(
                f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" '
                f'font-family="monospace" font-size="11" fill="{stroke}">'
                f"{labels[node.token_id]}</text>"
            )
    out.append("</svg>")
    return "\n".join(out)


def _auto_label(drawn: list[int], blocks: list[int]) -> str:
    return (
        f"J-circuit over layers {drawn[-1]} to {drawn[0]} at {len(blocks)} token "
        "position(s): concept nodes per layer, with dashed edges where the "
        "residual stream carries a concept forward, solid edges where a block "
        "computed the target, and a third colour where attention moved "
        "information between positions"
    )


def save_svg(circuit: JCircuit, path: str | Path, **kwargs) -> str:
    """Write :func:`render_svg` output to ``path`` and return the markup.

    Missing parent directories are created, so ``exp/run1/circuit.svg`` works
    without a prior ``mkdir``. Keyword arguments are forwarded to
    :func:`render_svg`.
    """
    markup = render_svg(circuit, **kwargs)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(markup, encoding="utf-8")
    return markup
