# Copyright 2026 Tung-Yu (Tony) Wu
# SPDX-License-Identifier: Apache-2.0
# Added by Tung-Yu (Tony) Wu and his Claude.
"""Build a J-circuit: a layered concept-attribution graph over J-lens readouts.

Runs the dense graph (mode 1), then prunes it (mode 2) and verifies that
pruning the stored dense graph gives the same circuit as building mode 2 from
scratch. Examples:

  python jcircuit_demo.py
  python jcircuit_demo.py --mode 2 --k 8 --prune-percent 15
  python jcircuit_demo.py --mode 2 --roots all     # prune from every position
    python jcircuit_demo.py --prompt "Fact: the capital of France is" --layer-top-percentile 95 --layer-bottom-percentile 20
    python jcircuit_demo.py --svg circuit.svg --svg-layer-top-percentile 95 --svg-layer-bottom-percentile 20
"""

from __future__ import annotations

import argparse
import time

import torch

import jlens
from jlens.circuit import _resolve_positions, build_jcircuit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--lens", default="neuronpedia/jacobian-lens")
    p.add_argument(
        "--lens-file",
        default="qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--prompt",
        default="Q: How many legs does the animal that barks have?\nA: It has",
    )
    p.add_argument(
        "--mode",
        type=int,
        default=1,
        choices=(1, 2),
        help="1 = dense graph, 2 = dense + per-layer pruning",
    )
    p.add_argument("--k", type=int, default=5, help="concepts per layer")
    p.add_argument(
        "--no-error-nodes",
        dest="error_nodes",
        action="store_false",
        help="drop the per-block node carrying the residual the k concepts "
        "cannot express; the graph then accounts only for named influence",
    )
    p.add_argument(
        "--selection",
        default="pursuit",
        choices=("pursuit", "topk"),
        help="pursuit = sparse non-negative decomposition; topk = ranked readout",
    )
    p.add_argument(
        "--estimator",
        default="eap",
        choices=("eap", "ig"),
        help="eap = gradient at the clean point (default); ig = averaged along "
        "the ablation path — ~3x slower, far more faithful per-edge",
    )
    p.add_argument(
        "--ig-steps", type=int, default=2, help="path samples for --estimator ig"
    )
    p.add_argument(
        "--stride",
        type=int,
        default=2,
        help="layer gap between levels; >1 is cheaper and coarser, and the band "
        "is truncated at the top if it does not divide evenly (default: 2)",
    )
    p.add_argument(
        "--layer-top-percentile",
        type=float,
        default=100.0,
        help="upper bound percentile for the selected layer band (default: 100)",
    )
    p.add_argument(
        "--layer-bottom-percentile",
        type=float,
        default=20.0,
        help="lower bound percentile for the selected layer band (default: 20)",
    )
    p.add_argument(
        "--prune-percent",
        type=float,
        default=20.0,
        help="mode 2: percentage of each layer pair's edges to keep",
    )
    p.add_argument(
        "--roots",
        default="last",
        help="mode 2: whose top-layer concepts pruning descends from — 'last' "
        "(default), 'all', or comma-separated token indices/text",
    )
    p.add_argument(
        "--llm-token-filter-pruning",
        action="store_true",
        help="mode 2: send the pruned graph's tokens and the prompt to Claude, "
        "and drop the ones it judges non-semantic or irrelevant to this "
        "scenario (needs the anthropic SDK and credentials)",
    )
    p.add_argument(
        "--llm-filter-model",
        default=None,
        help="model for --llm-token-filter-pruning (default: claude-sonnet-5)",
    )
    p.add_argument(
        "--positions",
        nargs="+",
        default=None,
        help="comma-separated token indices or text (default: all but BOS)",
    )
    p.add_argument("--svg", default=None, help="write an SVG of the circuit here")
    p.add_argument(
        "--svg-roots",
        default="last",
        help="whose top-layer concepts count as the circuit output: 'last' "
        "(default), 'all', or comma-separated token indices/text",
    )
    p.add_argument(
        "--svg-ink",
        default="#1a1a1a",
        help="colour of boxes and labels; 'currentColor' to inherit the "
        "surrounding page's text colour when embedding (default: #1a1a1a)",
    )
    p.add_argument(
        "--svg-background",
        default="#ffffff",
        help="fill behind the figure, or 'none' for transparent when embedding "
        "(default: #ffffff, so the file is readable in any viewer)",
    )
    p.add_argument(
        "--svg-layer-top-percentile",
        type=float,
        default=100.0,
        help="upper bound percentile for SVG layers (default: 100)",
    )
    p.add_argument(
        "--svg-layer-bottom-percentile",
        type=float,
        default=20.0,
        help="lower bound percentile for SVG layers (default: 20)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # HF_TOKEN for the gated downloads, ANTHROPIC_API_KEY for --llm-token-filter
    # -pruning. Anything already exported wins over the file.
    loaded = jlens.load_dotenv()
    if loaded:
        print(f"loaded from .env: {', '.join(loaded)}")
    print(f"loading {args.model} on {args.device} ...")
    import transformers

    hf = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16 if args.device.startswith("cuda") else None
    ).to(args.device)
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(args.lens, filename=args.lens_file)
    print(f"{lens!r}\nprompt: {args.prompt!r}\n")

    positions = (
        None
        if args.positions is None
        else [
            item.strip()
            for group in args.positions
            for item in group.split(",")
            if item.strip()
        ]
    )

    def split_roots(value: str) -> str | list[str]:
        if value in ("last", "all"):
            return value
        return [item.strip() for item in value.split(",") if item.strip()]

    common = dict(
        k=args.k,
        error_nodes=args.error_nodes,
        selection=args.selection,
        estimator=args.estimator,
        ig_steps=args.ig_steps,
        stride=args.stride,
        layer_top_percentile=args.layer_top_percentile,
        layer_bottom_percentile=args.layer_bottom_percentile,
        positions=positions,
    )
    token_strings = {
        i: tok.decode([t]) for i, t in enumerate(model.encode(args.prompt)[0].tolist())
    }

    t0 = time.perf_counter()
    dense = build_jcircuit(lens, model, args.prompt, mode=1, **common)
    t_dense = time.perf_counter() - t0
    # causal pairs = p(p+1)/2 source/target position combinations
    n_pos = len(dense.positions)
    expected = dense.hparams["n_levels"] * n_pos * (n_pos + 1) // 2 * args.k**2
    asked = dense.hparams["layer_top_requested"]
    truncated = (
        f" (asked for {asked}, not a whole number of strides)"
        if (asked != dense.layer_top)
        else ""
    )
    estimator = args.estimator + (
        f"(steps={args.ig_steps})" if args.estimator == "ig" else ""
    )
    print(f"=== MODE 1 (dense, {estimator})   {dense!r}")
    print(
        f"    layers {dense.layer_top}..{dense.layer_bottom} step {args.stride}"
        f"{truncated}"
        f"  levels={dense.hparams['n_levels']}  positions={n_pos}"
        f"  expected edges={expected:,}"
        f"  built in {t_dense:.2f}s"
    )
    mass = dense.error_mass()
    if mass == mass:  # not nan; estimator="ig" records no coverage
        print(
            f"    error mass={mass:.3f} — the share of incoming gradient norm no "
            f"concept in this graph names (raise k to shrink it)"
        )
    print()

    def write_svg(circuit) -> None:
        if args.svg is None:
            return
        from jlens.circuit_vis import save_svg

        if not 0 <= args.svg_layer_bottom_percentile <= 100:
            raise ValueError("svg_layer_bottom_percentile must be in [0, 100]")
        if not 0 <= args.svg_layer_top_percentile <= 100:
            raise ValueError("svg_layer_top_percentile must be in [0, 100]")
        source_layers = list(reversed(circuit.layers))
        top = source_layers[
            round(args.svg_layer_top_percentile / 100 * (len(source_layers) - 1))
        ]
        bottom = source_layers[
            round(args.svg_layer_bottom_percentile / 100 * (len(source_layers) - 1))
        ]
        shown = [layer for layer in circuit.layers if bottom <= layer <= top]
        blocks = circuit.positions
        roots = args.svg_roots
        if roots not in ("last", "all"):
            wanted = [item.strip() for item in roots.split(",") if item.strip()]
            roots = _resolve_positions(wanted, model.encode(args.prompt), False, tok)
        save_svg(
            circuit,
            args.svg,
            layers=shown,
            positions=blocks,
            roots=roots,
            position_labels=token_strings,
            ink=args.svg_ink,
            background=None if args.svg_background == "none" else args.svg_background,
        )
        print(
            f"\nSVG -> {args.svg}  (layers {shown[0]}..{shown[-1]}, positions {blocks})"
        )

    if args.mode == 1:
        print(dense.format(max_edges_per_level=6))
        write_svg(dense)
        return

    # mode 2 two ways: prune the stored dense graph, and build from scratch.
    # `roots` reaches prune() as indices, since only build has the tokenizer.
    build_roots = split_roots(args.roots)
    prune_roots = (
        build_roots
        if isinstance(build_roots, str)
        else _resolve_positions(build_roots, model.encode(args.prompt), False, tok)
    )
    t0 = time.perf_counter()
    pruned = dense.prune(args.prune_percent, roots=prune_roots)
    t_prune = time.perf_counter() - t0
    t0 = time.perf_counter()
    scratch = build_jcircuit(
        lens,
        model,
        args.prompt,
        mode=2,
        prune_percent=args.prune_percent,
        roots=build_roots,
        **common,
    )
    t_scratch = time.perf_counter() - t0

    # The two paths agree exactly in exact arithmetic. In bf16 they differ a
    # little, because mode 2 batches fewer cotangents per VJP and the batch size
    # sets the reduction order; where that noise straddles the pruning cutoff,
    # the two can keep different edges. So compare the edge sets, and the scores
    # only over the edges both kept.
    a_edges = {(e.source, e.target): e.score for e in pruned.edges}
    b_edges = {(e.source, e.target): e.score for e in scratch.edges}
    shared = a_edges.keys() & b_edges.keys()
    differing = len(a_edges) + len(b_edges) - 2 * len(shared)
    worst = max((abs(a_edges[p] - b_edges[p]) for p in shared), default=0.0)
    print(
        f"=== MODE 2 (prune {args.prune_percent}% per layer pair, "
        f"roots={pruned.hparams['roots'] or 'all positions'})"
    )
    print(f"    dense.prune()      -> {pruned!r}   ({t_prune * 1000:.1f} ms)")
    print(f"    build(mode=2)      -> {scratch!r}   ({t_scratch:.2f} s)")
    print(
        f"    same nodes {pruned.nodes == scratch.nodes}, "
        f"{len(shared)}/{len(a_edges)} edges in common ({differing} differ), "
        f"max |score| diff {worst:.1e}\n"
    )

    if args.llm_token_filter_pruning:
        from jlens.token_filter import DEFAULT_MODEL, select_noise_tokens

        model_id = args.llm_filter_model or DEFAULT_MODEL
        tokens = pruned.tokens()
        print(f"=== LLM TOKEN FILTER ({model_id}, {len(tokens)} concepts)")
        t0 = time.perf_counter()
        drop = select_noise_tokens(args.prompt, tokens, model=model_id)
        t_filter = time.perf_counter() - t0
        by_id = dict(tokens)
        for reason in ("non_semantic", "irrelevant"):
            named = [by_id[i] for i, r in sorted(drop.items()) if r == reason]
            if named:
                print(f"    {reason:>13}: " + "  ".join(repr(t) for t in named)[:600])
        before = pruned
        filtered = before.drop_tokens(drop)
        n_before = sum(len(v) for v in before.nodes.values())
        n_after = sum(len(v) for v in filtered.nodes.values())
        print(
            f"    dropped {len(drop)}/{len(tokens)} concepts in {t_filter:.1f}s -> "
            f"{filtered!r}"
        )
        print(
            f"    nodes {n_before} -> {n_after} "
            f"({n_before - n_after - _direct_hits(before, drop)} more removed as "
            f"orphans), edges {len(before.edges)} -> {len(filtered.edges)}"
        )
        # Removing a mid-graph concept orphans whatever depended only on it, so
        # an over-eager filter cascades. Say so instead of drawing an empty page.
        if n_after == 0:
            print(
                "    !! the filter emptied the graph — it judged too much of the "
                "trace irrelevant. Keeping the unfiltered circuit; try a stronger "
                "--llm-filter-model, or widen --prune-percent first.\n"
            )
        else:
            pruned = filtered
            print()

    print(pruned.format(max_edges_per_level=6))

    surviving = {l: len(pruned.nodes[l]) for l in pruned.layers}
    print("\nsurvived concepts per layer (top -> bottom):")
    print("   " + "  ".join(f"L{l}:{n}" for l, n in surviving.items())[:400])
    write_svg(pruned)


def _direct_hits(circuit, drop) -> int:
    """Nodes removed because their own token was filtered, not as orphans."""
    return sum(1 for v in circuit.nodes.values() for n in v if n.token_id in drop)


if __name__ == "__main__":
    main()
