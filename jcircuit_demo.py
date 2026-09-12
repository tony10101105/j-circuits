# Copyright 2026 Tung-Yu (Tony) Wu
# SPDX-License-Identifier: Apache-2.0
# Added by Tung-Yu (Tony) Wu and his Claude.
"""Build a J-circuit: a layered concept-attribution graph over J-lens readouts.

Builds a pruned J-circuit in one pass, optionally filters its concepts with an
LLM, prints it, and writes an SVG. Examples:

  python jcircuit_demo.py
  python jcircuit_demo.py --k 8 --prune-percent 15
  python jcircuit_demo.py --roots all     # prune from every position
    python jcircuit_demo.py --prompt "Fact: the capital of France is" --layer-top-percentile 95 --layer-bottom-percentile 20
    python jcircuit_demo.py --svg circuit.svg    # draws exactly the circuit that was built
"""

from __future__ import annotations

import argparse
import time

import torch

import jlens
from jlens.circuit import build_jcircuit


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
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--prompt",
        default="Q: How many legs does the animal that barks have?\nA: It has",
    )
    p.add_argument(
        "--k", type=int, default=5, help="concepts per ``(layer, position)`` block"
    )
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
        help="method to extract concepts in each block. pursuit = sparse non-negative decomposition; topk = ranked readout",
    )
    p.add_argument(
        "--estimator",
        default="eap",
        choices=("eap", "ig"),
        help="attribution calculation method. eap = gradient at the clean point; ig = averaged along the ablation path—slower but more precise",
    )
    p.add_argument(
        "--ig-steps",
        type=int,
        default=2,
        help="number of path samples for eap-ig if `--estimator ig`",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=2,
        help="layer gap between levels; >1 is cheaper and coarser, and the band is truncated at the top if it does not divide evenly",
    )
    p.add_argument(
        "--layer-top-percentile",
        type=float,
        default=100.0,
        help="which fitted layer the circuit's top level sits at, as a "
        "percentile of the lens's fitted-layer list (not of the model's "
        "depth): 0 = its lowest fitted layer, 100 = its highest. These are "
        "the concepts read closest to the output, and where pruning starts",
    )
    p.add_argument(
        "--layer-bottom-percentile",
        type=float,
        default=20.0,
        help="which fitted layer the circuit's bottom level sits at, on the "
        "same scale. The circuit spans bottom..top in steps of --stride, so "
        "lowering this adds levels, edges and build time; the default skips "
        "the earliest layers, whose readouts are mostly formatting",
    )
    p.add_argument(
        "--prune-percent",
        type=float,
        default=20.0,
        help="percentage of each layer pair's dense edge count to keep",
    )
    p.add_argument(
        "--roots",
        default="last",
        help="whose top-layer concepts pruning descends from — 'last', 'all', or comma-separated token indices/text",
    )
    p.add_argument(
        "--llm-token-filter-pruning",
        action="store_true",
        help="send the pruned graph's tokens and the prompt to an external LLM, "
        "and drop the ones it judges non-semantic or irrelevant to this scenario",
    )
    p.add_argument(
        "--llm-filter-model",
        default="claude-opus-5",
        help="model for --llm-token-filter-pruning",
    )
    p.add_argument(
        "--positions",
        nargs="+",
        default=None,
        help="comma-separated token indices or text (default: all but BOS)",
    )
    p.add_argument("--svg", default=None, help="write an SVG of the circuit here")
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    loaded = jlens.load_dotenv()
    if loaded:
        print(f"loaded from .env: {', '.join(loaded)}")
    print(f"loading {args.model} on {args.device} ...")
    import transformers

    hf = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16
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
    circuit = build_jcircuit(
        lens,
        model,
        args.prompt,
        prune_percent=args.prune_percent,
        roots=split_roots(args.roots),
        **common,
    )
    t_build = time.perf_counter() - t0

    # causal pairs = p(p+1)/2 source/target position combinations, times k^2.
    # prune_percent is a share of this, so it is the number to read the result
    # against — not a prediction of how many edges were built.
    n_pos = len(circuit.positions)
    dense_edges = circuit.hparams["n_levels"] * n_pos * (n_pos + 1) // 2 * args.k**2
    asked = circuit.hparams["layer_top_requested"]
    truncated = (
        f" (asked for {asked}, not a whole number of strides)"
        if (asked != circuit.layer_top)
        else ""
    )
    estimator = args.estimator + (
        f"(steps={args.ig_steps})" if args.estimator == "ig" else ""
    )
    print(
        f"=== J-CIRCUIT ({estimator}, prune {args.prune_percent}% per level pair, "
        f"roots={circuit.hparams['roots'] or 'all positions'})"
    )
    print(f"    {circuit!r}")
    print(
        f"    layers {circuit.layer_top}..{circuit.layer_bottom} step {args.stride}"
        f"{truncated}"
        f"  levels={circuit.hparams['n_levels']}  positions={n_pos}"
        f"  dense would hold {dense_edges:,}"
        f"  built in {t_build:.2f}s"
    )
    print()

    def write_svg(circuit) -> None:
        if args.svg is None:
            return
        from jlens.circuit_vis import save_svg

        save_svg(
            circuit,
            args.svg,
            position_labels=token_strings,
            ink=args.svg_ink,
            background=None if args.svg_background == "none" else args.svg_background,
        )
        shown = sorted(circuit.layers)
        print(
            f"\nSVG -> {args.svg}  (layers {shown[-1]}..{shown[0]}, "
            f"positions {circuit.positions})"
        )

    if args.llm_token_filter_pruning:
        from jlens.token_filter import select_noise_tokens

        model_id = args.llm_filter_model
        tokens = circuit.tokens()
        print(f"=== LLM TOKEN FILTER ({model_id}, {len(tokens)} concepts)")
        t0 = time.perf_counter()
        drop = select_noise_tokens(args.prompt, tokens, model=model_id)
        t_filter = time.perf_counter() - t0
        by_id = dict(tokens)
        for reason in ("non_semantic", "irrelevant"):
            named = [by_id[i] for i, r in sorted(drop.items()) if r == reason]
            if named:
                print(f"    {reason:>13}: " + "  ".join(repr(t) for t in named)[:600])
        before = circuit
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
            circuit = filtered
            print()

    print(circuit.format(max_edges_per_level=6))

    surviving = {l: len(circuit.nodes[l]) for l in circuit.layers}
    print("\nsurvived concepts per layer (top -> bottom):")
    print("   " + "  ".join(f"L{l}:{n}" for l, n in surviving.items())[:400])
    write_svg(circuit)


def _direct_hits(circuit, drop) -> int:
    """Nodes removed because their own token was filtered, not as orphans."""
    return sum(1 for v in circuit.nodes.values() for n in v if n.token_id in drop)


if __name__ == "__main__":
    main()
