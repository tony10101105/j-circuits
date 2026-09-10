# Copyright 2026 Tung-Yu (Tony) Wu
# SPDX-License-Identifier: Apache-2.0
# Added by Tung-Yu (Tony) Wu and his Claude.
"""J-circuit: layered concept-attribution graphs in J-lens coordinates.

A J-circuit is a layered DAG whose nodes are *concepts at token positions* —
the top-``k`` J-lens readouts of one position of one layer — and whose edges
score how much a concept at layer ``l-s`` supports a concept at layer ``l``,
where ``s`` is the ``stride`` between consecutive levels. Each level is
therefore sliced into ``p`` position blocks of ``k`` concepts.

Edge score (EAP, first order). Ablating a source concept moves the residual by
the analytically known rank-1 step ``dh = -a_s * v_s`` at that position, so the
effect on a target readout ``M = <v_t, h_l[q]>`` is::

    A(s@l-s,p -> t@l,q) = a_s * <v_s, grad_{h_{l-s}[p]} M>

with ``a_s = <v_s, h_{l-s}[p]>`` the source's clean coordinate and ``v`` the
unit-normalised J-lens vectors of :func:`jlens.interventions.lens_vector`.

Edges run between consecutive *levels* and obey attention's causal order,
``p <= q``: a source may feed its own position (the residual stream) or any
later one (attention). One batched VJP per layer pair — cotangents for every
target node at once — yields the whole matrix, because a single backward
already produces the gradient at *every* source position.

Estimators. Because the corruption is a straight rank-1 path, the exact effect
is a line integral, ``M_clean - M_ablated = int_0^1 a_s <v_s, grad M(h(a))> da``
with ``h(a) = h - a*a_s*v_s`` — an identity, not an approximation.
``estimator="eap"`` evaluates the integrand at ``a=0`` only;
``estimator="ig"`` averages ``ig_steps`` midpoints of the path (EAP-IG).

Which to use depends on what the number is for. Measured against real ablations
on Qwen3-4B, over 540 pruned-surviving edges from six prompts with instruction
conflict (sycophancy, deception, refusal, evaluation-awareness):

===========  ========  =======  =======  ============
estimator      median      p90    worst   >10% off
===========  ========  =======  =======  ============
``"eap"``       3.24%   10.55%   56.14%   61 of 540
``"ig"``        0.54%    1.98%    7.69%    0 of 540
===========  ========  =======  =======  ============

EAP's *ranking* is fine either way — per-prompt Spearman 0.988–0.994 against
the true ordering, and top-40 overlap 39–40 of 40 — so pruning and graph
structure barely move. What EAP gets wrong is the magnitude of individual
edges, and it errs in one direction: it systematically under-reports. On the
sycophancy-math prompt the edge ``' incorrect' -> ' correct'`` is really 34.29
and EAP calls it 19.81. Short factual prompts are much kinder to it (median
1.03%), so an estimator validated on those does not transfer.

Cost is a small multiple, not a large one. IG cannot share one backward across
sources — the path point differs per source — but it still shares one perturbed
forward across every target of a level. On a 30-level stride-1 band with 5
positions and ``k=5`` (11,250 edges), building took 3.6 s with ``"eap"`` and
11.4 s with ``"ig"``: **3.2x**. Refining individual edges after the fact is
*worse* than rebuilding once past roughly 200 edges, because per-edge work
throws away the target sharing too.

Default is ``"eap"``: it is what the pruning and the figures need. Reach for
``"ig"`` when a specific edge weight is going into a claim.

Levels and ``stride``. Construction runs top-down from ``layer_top`` in steps of
``stride``, so the levels are ``layer_bottom, layer_bottom + stride, ...`` up to
``layer_top``. Each level is a complete cut of the residual stream and every
path through the circuit crosses each cut exactly once, so no edge double-counts
influence another edge already carries — true for any single stride, since the
layers *between* two levels are simply never cut. (Double counting is what
happens when several strides are mixed into one graph, not what a coarse stride
does.)

What a larger stride buys is cost: ``(layer_top - layer_bottom) / stride`` VJP
levels instead of one per layer, and only the level layers need a fitted
Jacobian. On Qwen3-4B over layers 12..30, 5 positions and ``k=5``, stride 1
takes 2.6 s for 6750 edges and stride 6 takes 0.4 s for 1125.

What it costs is resolution, in two ways. An edge becomes the *total* effect
over ``stride`` blocks, so whichever concepts mediate it inside that span are
not in the graph at all. And the first-order EAP score drifts further from a
true ablation the more blocks it linearises through. Against real ablations of
the strongest same-position edge into layer 30, the relative error runs
``0.5%, 0.2%, 0.3%, 1.2%, 5.4%, 10.0%`` at strides ``1, 2, 3, 4, 6, 8`` — so up
to about three blocks a score is still worth reading as a number, and past that
it is a ranking.

``layer_top - layer_bottom`` need not be a multiple of ``stride``. The band is
anchored at ``layer_bottom`` and stepped upward, so a leftover shorter than one
stride is dropped from the top: ``layer_bottom=20, layer_top=23, stride=2``
builds 20 -> 22 and lowers ``layer_top`` to 22 rather than scoring a ragged
final level of 22 -> 23.

Two modes:

``mode=1`` — dense. Every causal pair is kept: ``k**2 * p*(p+1)/2`` edges per
level pair when all positions are in play.
``mode=2`` — dense, then Circuit-Tracing-style top-down pruning. At each layer
pair only the strongest ``prune_percent`` of edges (ranked by ``|score|``)
survive; a source node left with no surviving edge is dropped and does not
appear as a target when the next pair down is scored. There is no global edge
budget, only this per-layer percentage.

Pruning starts from the circuit's *roots*: the top-layer concepts whose readout
counts as the output. By default that is the last token, whose readout is the
model's next token — the one place a circuit is normally read. The per-layer
percentage is then spent entirely on the cone feeding that position, rather than
shared with concepts nobody is going to look at. ``roots="all"`` restores the
descent from every position.

``mode=2`` is ``mode=1`` followed by :meth:`JCircuit.prune`: both call the same
selection rule on the same restricted edge sets, so in exact arithmetic they
give the same circuit. Running ``mode=2`` directly is cheaper — dead targets
are never scored, and only surviving edges are ever materialised.

The two paths are not bit-identical in reduced precision, though: ``mode=2``
batches fewer cotangents into each VJP, and a different batch size changes the
bf16 reduction order. Measured on Qwen3-4B, scores agree to ~1e-2 on magnitudes
reaching 45. Because pruning ranks by ``|score|``, two edges tied to within that
margin can come out ordered differently: on a 1300-edge circuit built with
``roots="last"``, one edge of the 1300 differed. Build ``mode=1`` once and
:meth:`~JCircuit.prune` it when you need a circuit reproducible across runs of
different ``prune_percent`` or ``roots``.

Two properties of concept graphs are worth knowing before reading one (both
measured on Qwen3-4B at stride 1):

- **Persistence dominates.** 95–98% of a same-position stride-1 edge is the
  residual stream carrying a concept forward unchanged; only the remainder is
  computation. That share is analytic (``a_s * <v_s^{l-1}, v_t^l>``), so
  subtract it to see the compute — :attr:`Edge.identity` / :attr:`Edge.computed`.
  Cross-position edges have no residual path at all, so their identity is 0 and
  every bit of their score is attention moving information.
- **Named concepts leak.** The ``k`` lens directions of a block span only part
  of the gradient arriving at a target, so path sums over circuit edges
  under-report the total effect. :attr:`JCircuit.coverage` and
  :meth:`JCircuit.error_mass` measure the gap — this codebase's analogue of
  Circuit Tracing's error node. Over L22..L28 at stride 1 with ``k=5``, the
  named directions carry **0.44** of the incoming gradient norm. Raising ``k``
  to 20 only reaches 0.50, so the remainder is not simply an under-sized
  concept set. Attention is where it goes: same-position gradients are 0.49
  named, cross-position ones **0.09**. For scale, a random 5-dimensional
  subspace of 2560 would capture 0.04, so 0.44 is ten times alignment, not
  noise — the leak and the signal are both real. Treat scores as a ranking over
  concepts, not as an exhaustive decomposition.

Error nodes. Because the leak is real, each ``(layer, position)`` block also
carries one node past its named ranks standing for what those concepts cannot
express: ``r = h - P h``, with ``P`` the orthogonal projector onto their span.
It is a node like any other — it has an activation (``||r||``, which is exactly
``<r_hat, h>``), scored edges into every causal target, and it is pruned by the
same rule, so a block whose leftover does nothing simply loses it. Turn the
whole thing off with ``error_nodes=False``.

What it shows, measured on Qwen3-4B over L22..L28 at stride 1 with ``k=5``: the
error node holds **97.5%** of the residual's norm and its activation runs 6.6x
the strongest named concept's, yet its edges carry only **20%** of the graph's
total ``|A|``. The named concepts are 2.5% of the length and 80% of the
influence. That enormous enrichment is the paper's "J-space is a small share of
variance but causally privileged" claim, measured from both sides at once.
Raising ``k`` to 20 pulls the error share down to 7.9%.

The error node is not a formality: at ``k=5``, 18 of 30 survive a 20% prune and
they source 56 of the 275 surviving edges. Read their ``computed``, not their
``score`` — because the activation is so large, even a slight overlap between
the leftover direction and a target's lens vector produces a big ``identity``,
and most error edges are nearly pure carry.

Three things follow from the definition and are worth knowing:

- **Sources only.** An error node is what nothing explains, so asking what
  explains it is not a question the graph should answer, and an unnamed root is
  not something a circuit is read out of. They are leaves: no incoming edges at
  any layer, and ``layer_top`` carries none at all.
- **Orthogonal, not the pursuit residual.** Pursuit's own leftover
  ``h - sum(c_i v_i)`` is not orthogonal to the atoms it picked (its
  coefficients are clamped non-negative), so an error node built from it would
  re-carry influence the named edges already claim. The orthogonal residual is
  the unique choice that double counts nothing, and it is defined under
  ``selection="topk"`` too, which has no coefficients.
- **No vocabulary entry.** ``token_id`` is :data:`ERROR_TOKEN_ID` and the
  direction depends on the prompt, so it is stored on the circuit rather than
  recovered from the id: use :meth:`JCircuit.direction`, which handles both
  kinds. It also means an error node cannot be steered or swapped — the
  :class:`~jlens.interventions.Intervention` API resolves token ids.
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

#: Layer gap between consecutive levels when ``stride`` is not given. 1 cuts the
#: residual stream at every block, the finest resolution the lens allows.
DEFAULT_STRIDE = 1

#: How an edge score is estimated. ``"eap"`` takes the gradient at the clean
#: point; ``"ig"`` averages it along the ablation path (see the module
#: docstring).
ESTIMATORS = ("eap", "ig")

#: Default path samples for ``estimator="ig"``. Measured on Qwen3-4B, 2 and 5
#: are indistinguishable (median relative error 0.54% vs 0.51% over 540 edges),
#: so the cheaper one is the default.
DEFAULT_IG_STEPS = 2

#: Perturbed forward rows batched at once under ``estimator="ig"``. Each row is
#: one ``(source concept, path sample)`` pair and carries a full sequence, so
#: this is the memory knob; it is capped by ``vjp_chunk``.
IG_ROW_CHUNK = 8

#: How a ``(layer, position)`` block picks its concepts. ``"pursuit"`` is the
#: sparse non-negative decomposition of :mod:`jlens.pursuit`; ``"topk"`` is the
#: plain ranked readout, which is more redundant but needs no solver.
SELECTIONS = ("pursuit", "topk")


class _RootsDefault:
    """Sentinel for an unset ``roots``: ``"last"`` under ``mode=2``, and no
    restriction under ``mode=1``, which is dense by definition. Passing
    ``roots`` explicitly to a ``mode=1`` build is an error rather than a
    silently ignored argument, which is what this distinguishes."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic, shows in help()
        return "'last' (mode 2) / unrestricted (mode 1)"


ROOTS_DEFAULT = _RootsDefault()

#: Dense circuits refuse to materialise more edges than this. Cross-position
#: edges grow as ``p**2``, so a long prompt overruns it quickly; the error names
#: the knobs (``positions``, ``k``, ``mode=2``) that bring it back down.
MAX_DENSE_EDGES = 500_000

_ZERO_NORM_EPS = 1e-8

#: Token id carried by an error node. Negative so it can never collide with a
#: real vocabulary entry, and shared across layers so the renderer stacks every
#: error node of a position into one column, as it does for a real concept.
ERROR_TOKEN_ID = -1

#: Display text for an error node.
ERROR_TOKEN = "<error>"

#: Singular values below this multiple of the largest are treated as outside the
#: span when projecting onto a block's concept directions. The ``k`` lens vectors
#: are not orthogonal — ``"topk"`` selection in particular returns near-duplicate
#: directions — so the span can be rank-deficient and a plain basis would
#: overstate what the concepts cover.
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


@dataclass(frozen=True)
class Coverage:
    """How much of a target's incoming influence the named concepts can carry.

    The gradient ``g = grad_{h_{l,p}} M_t`` is the whole of what one source
    block can do to this target's readout. Only the component of ``g`` lying in
    the span of that block's ``k`` selected lens directions can ever surface as
    an edge; the rest flows through directions the circuit never names. That
    remainder is this codebase's analogue of Circuit Tracing's *error node* —
    real influence, with no node to attach it to.

    One row per ``(target, source position)``: every edge from that block into
    that target shares the same gradient, so the split is a property of the
    pair, not of the individual edges.

    Attributes:
        target: The concept whose readout the gradient was taken of.
        source_layer: Layer the gradient was read at, ``target.layer - stride``.
        source_position: Position within that layer.
        named: ``||P_J g||``, the part inside the span of the block's concepts.
        total: ``||g||``.
    """

    target: Node
    source_layer: int
    source_position: int
    named: float
    total: float

    @property
    def fraction(self) -> float:
        """``named / total`` in ``[0, 1]``: the share the edges can express.

        ``nan`` when the gradient itself is zero — the target does not depend on
        that block at all, so there is no influence to have missed. Reporting 0
        there would read as "everything leaked", the opposite of the truth.
        """
        if self.total <= _ZERO_NORM_EPS:
            return float("nan")
        return self.named / self.total

    @property
    def error(self) -> float:
        """``1 - fraction``: the error-node share (``nan`` for a zero gradient)."""
        return 1.0 - self.fraction

    def __str__(self) -> str:
        return (
            f"{self.target} <- L{self.source_layer}:{self.source_position}  "
            f"named={self.fraction:.3f}"
        )


def _restrict_coverage(
    coverage: Sequence[Coverage], nodes: dict[int, list[Node]]
) -> tuple[Coverage, ...]:
    """The coverage rows whose target survives in ``nodes``.

    Coverage measures what a target's *own* gradient reaches, so it is unchanged
    by pruning edges — dropping a weak edge does not make the influence it
    carried disappear. Only rows for vanished targets are removed.
    """
    alive = {n for v in nodes.values() for n in v}
    return tuple(row for row in coverage if row.target in alive)


def _keep_top_percent(edges: list[Edge], percent: float) -> list[Edge]:
    """The pruning rule, shared by ``mode=2`` and :meth:`JCircuit.prune`.

    Keeps the strongest ``percent`` of ``edges`` by ``|score|`` (at least one),
    preserving the input order among survivors so the result is deterministic.
    """
    if not edges:
        return []
    n_keep = max(1, round(len(edges) * percent / 100.0))
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
    roots: list[int] | None = None,
    stride: int = DEFAULT_STRIDE,
) -> tuple[dict[int, list[Node]], list[Edge]]:
    """Top-down prune: keep ``percent`` of each level pair, drop dead sources.

    ``edges_by_layer[l]`` holds the edges into layer ``l``. ``roots``, when
    given, restricts the layer-``layer_top`` nodes the descent starts from, so
    the budget is spent on the cone feeding those positions. Returns the
    surviving ``(nodes, edges)``; iteration stops early if a level pair leaves
    no source alive.
    """
    live = [n for n in nodes[layer_top] if not n.is_error]
    if roots is not None:
        keep = set(roots)
        live = [n for n in live if n.position in keep]
        if not live:
            raise ValueError(
                f"no concept at layer {layer_top} sits at a root position {sorted(keep)}"
            )
    kept_nodes = {layer_top: live}
    kept_edges: list[Edge] = []
    for layer in range(layer_top, layer_bottom, -stride):
        live_set = set(live)
        level = [e for e in edges_by_layer.get(layer, []) if e.target in live_set]
        keep = _keep_top_percent(level, percent)
        if not keep:
            break
        kept_edges.extend(keep)
        survivors = {e.source for e in keep}
        kept = [n for n in nodes[layer - stride] if n in survivors]
        if not kept:
            break
        kept_nodes[layer - stride] = kept
        # Error nodes stay in the graph as leaves but never become targets, so
        # they cannot seed the next descent — the same rule the mode=2 build
        # follows, or the two paths would disagree.
        live = [n for n in kept if not n.is_error]
        if not live:
            break
    return kept_nodes, kept_edges


@dataclass
class JCircuit:
    """A built J-circuit: concepts per layer, scored edges, and the settings used.

    Attributes:
        nodes: ``{layer: [Node, ...]}``, ordered by position then lens rank. In
            ``mode=2`` only surviving concepts are present.
        edges: All scored edges, ordered top layer pair first.
        hparams: The settings the circuit was built with (see
            :func:`build_jcircuit`), plus ``n_levels`` and ``positions_resolved``.
        mode: ``1`` (dense) or ``2`` (pruned).
        coverage: One :class:`Coverage` row per ``(target, source position)``,
            measuring how much of each target's incoming gradient the named
            concepts span. Empty under ``estimator="ig"``, which never takes a
            gradient at the clean point; see :meth:`error_mass`.
    """

    nodes: dict[int, list[Node]]
    edges: list[Edge]
    hparams: dict = field(default_factory=dict)
    mode: int = 1
    coverage: tuple[Coverage, ...] = ()
    error_directions: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)

    def direction(
        self, node: Node, lens: JacobianLens, model: LensModel
    ) -> torch.Tensor:
        """The unit residual direction ``node`` stands for.

        A named concept's direction is fixed geometry — its J-lens vector, which
        :func:`jlens.interventions.lens_vector` recovers from the token id. An
        error node has no token, and its direction depends on the prompt, so the
        build stores it on the circuit; this method hides the difference.

        Raises:
            KeyError: If ``node`` is an error node from a circuit built without
                ``error_nodes``, or one this circuit does not carry.
        """
        if not node.is_error:
            from jlens.interventions import lens_vector

            v = lens_vector(lens, model, node.token_id, node.layer)
            return v / v.norm()
        return self.error_directions[(node.layer, node.position)]

    def error_mass(self, layer: int | None = None) -> float:
        """Share of incoming influence that no concept in this circuit names.

        The gradient-weighted mean of :attr:`Coverage.error` — weighted by
        ``total``, so a target whose readout barely depends on the layer below
        cannot drag the average around. This is the aggregate form of Circuit
        Tracing's error node: 0.0 would mean the ``k`` concepts per block span
        every direction that matters.

        Args:
            layer: Restrict to gradients read at this source layer. ``None``
                (default) pools every level.

        Returns:
            A share in ``[0, 1]``, or ``nan`` when there is nothing to measure
            (an ``estimator="ig"`` build, or no gradient of any size).
        """
        rows = [
            row
            for row in self.coverage
            # A zero gradient carries no influence to have missed, and its
            # ``error`` is nan, which would poison the sum rather than weigh 0.
            if row.total > _ZERO_NORM_EPS
            and (layer is None or row.source_layer == layer)
        ]
        weight = sum(row.total for row in rows)
        if not rows or weight <= _ZERO_NORM_EPS:
            return float("nan")
        return sum(row.error * row.total for row in rows) / weight

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
        return int(self.hparams.get("stride", DEFAULT_STRIDE))

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
            f"JCircuit(mode={self.mode}, layers={self.layer_top}..{self.layer_bottom}, "
            f"k={self.hparams.get('k')}, positions={len(self.positions)}, "
            f"nodes={sum(len(v) for v in self.nodes.values())}, edges={len(self.edges)})"
        )

    def prune(
        self, percent: float = 20.0, *, roots: str | Sequence[int] | None = "last"
    ) -> JCircuit:
        """Return the ``mode=2`` circuit obtained by pruning this one.

        Applies the same top-down rule ``build_jcircuit(mode=2)`` uses, so
        ``build(mode=1).prune(p, roots=r)`` and
        ``build(mode=2, prune_percent=p, roots=r)`` agree (exactly in exact
        arithmetic; see the module docstring for the reduced precision caveat).

        Args:
            percent: Percentage of each layer pair's edges to keep, by
                ``|score|``. At least one edge per pair always survives.
            roots: Which top-layer concepts count as the circuit's output, and
                therefore where the descent starts. ``"last"`` (default) keeps
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
        hparams["mode"] = 2
        return JCircuit(
            nodes=kept_nodes,
            edges=kept_edges,
            hparams=hparams,
            mode=2,
            coverage=_restrict_coverage(self.coverage, kept_nodes),
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
            mode=self.mode,
            coverage=tuple(r for r in self.coverage if r.target in alive),
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


def _coverage_split(
    grads: torch.Tensor,
    source_dirs: torch.Tensor,
    source_pos: torch.Tensor,
    unique_positions: torch.Tensor,
    named: torch.Tensor,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """``(||P_J g||, ||g||)`` per target, for each source position.

    ``grads`` is ``[n_target, seq, d]``: the gradient of each target's readout
    at every source position, which is what one batched VJP already produces.
    Projecting it onto the span of that position's concept directions costs a
    ``[k, k]``-sized SVD per position, so this rides along for free.

    ``named`` masks out error-node directions. The question this answers is how
    much of the influence can be given a *name*; the error node exists precisely
    to account for the rest, so counting it here would define the gap away.
    """
    out: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for position_value in unique_positions:
        place = int(position_value)
        basis = _span_basis(source_dirs[(source_pos == position_value) & named])
        at_position = grads[:, place]  # [n_target, d]
        inside = (
            (at_position @ basis).norm(dim=1)
            if basis.shape[1]
            else torch.zeros_like(at_position[:, 0])
        )
        out[place] = (inside, at_position.norm(dim=1))
    return out


def _score_layer_pair(
    activations: dict[int, torch.Tensor],
    target_layer: int,
    target_dirs: torch.Tensor,
    target_pos: torch.Tensor,
    source_dirs: torch.Tensor,
    source_pos: torch.Tensor,
    source_acts: torch.Tensor,
    source_named: torch.Tensor,
    stride: int,
    chunk: int,
) -> tuple[torch.Tensor, dict[int, tuple[torch.Tensor, torch.Tensor]]]:
    """EAP scores ``[n_source, n_target]`` for one level pair ``stride`` apart.

    One batched VJP per chunk of targets. A single backward already carries the
    gradient at *every* source position, so cross-position edges cost nothing
    beyond the bookkeeping: the cotangent for a target places its lens vector at
    that target's position and zero elsewhere, and the result is read off at
    each source position in turn.

    Nothing here depends on the two layers being adjacent: the VJP differentiates
    through however many blocks separate them, giving the total effect over the
    span.

    Returns the score matrix and, from the same gradients, the
    :class:`Coverage` split per source position (see :func:`_coverage_split`).
    """
    source_layer = target_layer - stride
    out = activations[target_layer]
    n_target = target_dirs.shape[0]
    scores = torch.zeros(source_dirs.shape[0], n_target, device=source_dirs.device)
    unique_positions = source_pos.unique()
    named = torch.zeros(n_target, len(unique_positions), device=source_dirs.device)
    total = torch.zeros_like(named)

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
        split = _coverage_split(
            grads, source_dirs, source_pos, unique_positions, source_named
        )
        for slot, position_value in enumerate(unique_positions):
            named[start:stop, slot], total[start:stop, slot] = split[
                int(position_value)
            ]

    coverage = {
        int(position_value): (named[:, slot], total[:, slot])
        for slot, position_value in enumerate(unique_positions)
    }
    return scores * source_acts.unsqueeze(1), coverage


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
    mode: int = 1,
    k: int = 5,
    selection: str = "pursuit",
    error_nodes: bool = True,
    estimator: str = "eap",
    ig_steps: int = DEFAULT_IG_STEPS,
    stride: int = DEFAULT_STRIDE,
    layer_top: int | None = None,
    layer_bottom: int | None = None,
    layer_top_percentile: float = 95.0,
    layer_bottom_percentile: float = 20.0,
    prune_percent: float = 20.0,
    roots: str | Sequence[int | str] | None = ROOTS_DEFAULT,
    positions: Sequence[int | str] | None = None,
    skip_bos: bool = True,
    max_edges: int = MAX_DENSE_EDGES,
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
        mode: ``1`` for the dense graph, ``2`` to additionally prune (see the
            module docstring).
        k: Concepts per ``(layer, position)`` block.
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
        prune_percent: ``mode=2`` only — percentage of each layer pair's edges
            to keep, by ``|score|``.
        roots: ``mode=2`` only — which top-layer concepts count as the
            circuit's output, and so where pruning starts. Defaults to
            ``"last"``: only the final position's top-layer concepts, the ones
            whose readout is the model's next token. ``"all"`` keeps every
            position's, or name positions explicitly, as indices or decoded
            token text like ``"has"``. Restricting the roots spends the whole
            per-layer percentage on the cone that reaches them, and needs only
            ``k`` cotangents in the top layer pair's VJP rather than ``p*k``.
            Passing it to a ``mode=1`` build raises, since a dense graph keeps
            every top-layer concept by definition.
        positions: Token positions to build over (Python indexing), or decoded
            token text such as ``"legs"``. Each text value selects every exact
            matching decoded token sequence, including words split across
            multiple tokens. ``None`` takes every position, minus BOS when
            ``skip_bos``.
        skip_bos: Drop position 0 when ``positions`` is ``None``. Its
            attention-sink residual has an outsized norm, which makes its
            coordinates and edges spuriously large.
        max_edges: Refuse to build a dense circuit larger than this. Ignored
            for ``mode=2``, which only materialises survivors.
        vjp_chunk: Target nodes per batched VJP. Lower it if a long prompt runs
            the accelerator out of memory.
        max_seq_len: Truncation length when ``prompt`` is text.

    Returns:
        The built :class:`JCircuit`.

    Raises:
        ValueError: If ``stride < 1`` or is wider than the layer range, ``mode``
            is not 1 or 2, ``k < 1``, the layer range is empty or a level is not
            fitted, a position or root is out of range, ``roots`` restricts a
            ``mode=1`` build, or a dense circuit would exceed ``max_edges``.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if mode not in (1, 2):
        raise ValueError(f"mode must be 1 (dense) or 2 (pruned), got {mode}")
    if selection not in SELECTIONS:
        raise ValueError(f"selection must be one of {SELECTIONS}, got {selection!r}")
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}, got {estimator!r}")
    if ig_steps < 1:
        raise ValueError(f"ig_steps must be >= 1, got {ig_steps}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if vjp_chunk < 1:
        raise ValueError(f"vjp_chunk must be >= 1, got {vjp_chunk}")
    if mode == 2 and not 0 < prune_percent <= 100:
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

    given_roots = roots is not ROOTS_DEFAULT
    if mode == 1:
        if given_roots:
            raise ValueError(
                "roots applies to mode=2 only: mode 1 is the dense graph, which "
                "keeps every top-layer concept by definition. Build it, then call "
                "JCircuit.prune(percent, roots=...)"
            )
        root_positions = None
    else:
        # Token text ("has") is resolvable here but not in JCircuit.prune, which
        # has no tokenizer, so turn it into indices before the shared root rule.
        if not given_roots:
            roots = "last"
        elif roots is not None and not isinstance(roots, str):
            roots = _resolve_positions(list(roots), input_ids, False, model.tokenizer)
        root_positions = _root_positions(roots, chosen)

    if mode == 1:
        # Causal pairs per level pair: for each target position, every source
        # position at or before it, times k^2 concept pairs.
        pairs = sum(sum(1 for p in chosen if p <= q) for q in chosen)
        projected = pairs * k * k * n_levels
        if projected > max_edges:
            raise ValueError(
                f"a dense circuit here would hold ~{projected:,} edges "
                f"(> max_edges={max_edges:,}): {len(chosen)} positions x {k} "
                f"concepts over {n_levels} level pairs. Narrow positions=, lower "
                "k, shrink the layer range, raise stride, use mode=2, or raise "
                "max_edges."
            )

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
        coverage: list[Coverage] = []
        for layer in range(layer_top, layer_bottom, -stride):
            all_targets, all_dirs, all_pos = concepts[layer]
            sources, source_dirs, source_pos = concepts[layer - stride]
            order = {n: i for i, n in enumerate(all_targets)}
            wanted = live if mode == 2 else [n for n in all_targets if not n.is_error]
            pick = torch.tensor([order[n] for n in wanted], device=all_dirs.device)
            targets, target_dirs, target_pos = wanted, all_dirs[pick], all_pos[pick]

            source_acts = torch.tensor(
                [n.activation for n in sources], device=source_dirs.device
            )
            source_named = torch.tensor(
                [not n.is_error for n in sources], device=source_dirs.device
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
                scores, split = _score_layer_pair(
                    acts,
                    layer,
                    target_dirs,
                    target_pos,
                    source_dirs,
                    source_pos,
                    source_acts,
                    source_named,
                    stride,
                    vjp_chunk,
                )
                coverage.extend(
                    Coverage(
                        target=target,
                        source_layer=layer - stride,
                        source_position=position,
                        named=float(named[j]),
                        total=float(total[j]),
                    )
                    for position, (named, total) in sorted(split.items())
                    for j, target in enumerate(targets)
                    # A source block at or after the target carries no edge into
                    # it, so its gradient is not influence the circuit dropped.
                    if position <= target.position
                )
            # The residual stream only carries a concept forward within its own
            # position; across positions there is no identity path.
            same = source_pos.unsqueeze(1) == target_pos.unsqueeze(0)
            identity = source_acts.unsqueeze(1) * (source_dirs @ target_dirs.T) * same
            causal = source_pos.unsqueeze(1) <= target_pos.unsqueeze(0)

            if mode == 2:
                flat = scores.abs().masked_fill(~causal, float("-inf")).flatten()
                n_eligible = int(causal.sum())
                n_keep = max(1, round(prune_percent / 100.0 * n_eligible))
                # A stable sort, not topk: equal scores must then be broken by
                # flat index, the same rule `_keep_top_percent` follows, or the
                # two mode-2 paths disagree wherever the cutoff lands in a tie.
                order = torch.argsort(flat, descending=True, stable=True)
                chosen_flat = order[: min(n_keep, n_eligible)].sort().values
                picks = [divmod(int(f), len(targets)) for f in chosen_flat]
            else:
                picks = [
                    (i, j)
                    for i in range(len(sources))
                    for j in range(len(targets))
                    if causal[i, j]
                ]

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
            if mode == 2:
                survivors = {e.source for e in level}
                kept = [n for n in sources if n in survivors]
                if not kept:
                    break
                kept_nodes[layer - stride] = kept
                # An error node that survived stays in the graph as a leaf, but
                # it is never a target, so it does not seed the next descent.
                live = [n for n in kept if not n.is_error]
                if not live:
                    break
            else:
                kept_nodes[layer - stride] = sources

    hparams = {
        "mode": mode,
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
        "prune_percent": prune_percent if mode == 2 else None,
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
        mode=mode,
        coverage=tuple(coverage),
        error_directions=error_dirs,
    )
