# Copyright 2026 Tung-Yu (Tony) Wu
# SPDX-License-Identifier: Apache-2.0
# Added by Tung-Yu (Tony) Wu and his Claude.
"""What the error node looks like on a real circuit.

How big is the unnamed leftover next to the named concepts, how much influence
does it carry, and does it survive pruning — i.e. is it a node worth drawing or
a formality.
"""

import argparse

import torch
import transformers

import jlens
from jlens.circuit import build_jcircuit

DEVICE = "cuda"
PROMPT = "Q: How many legs does the animal that spins webs have?\nA: It has"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--layer-top", type=int, default=28)
    p.add_argument("--layer-bottom", type=int, default=22)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--ks", type=int, nargs="+", default=[1, 5, 20])
    p.add_argument("--prune", type=float, default=20.0)
    p.add_argument("--n-positions", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    jlens.load_dotenv()
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B", dtype=torch.bfloat16
    ).to(DEVICE)
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(
        "neuronpedia/jacobian-lens",
        filename="qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt",
    )

    seq = int(model.encode(PROMPT).shape[1])
    positions = list(range(max(1, seq - args.n_positions), seq))

    print(f"{'k':>4} {'||r||/||h||':>12} {'err a / max named a':>21} "
          f"{'err edge share':>15} {'survives prune':>15}")
    for k in args.ks:
        circuit = build_jcircuit(
            lens,
            model,
            PROMPT,
            prune_percent=100.0, roots="all",
            k=k,
            stride=args.stride,
            layer_top=args.layer_top,
            layer_bottom=args.layer_bottom,
            positions=positions,
        )
        ratios, rel = [], []
        for layer, nodes in circuit.nodes.items():
            for position in positions:
                block = [n for n in nodes if n.position == position]
                errors = [n for n in block if n.is_error]
                named = [n for n in block if not n.is_error]
                if not errors or not named:
                    continue
                h = errors[0].activation  # ||r||
                strongest = max(abs(n.activation) for n in named)
                rel.append(h / strongest)
                ratios.append((layer, position, h))
        err_edges = [e for e in circuit.edges if e.source.is_error]
        share = sum(abs(e.score) for e in err_edges) / sum(
            abs(e.score) for e in circuit.edges
        )
        pruned = circuit.prune(args.prune, roots="last")
        kept = sum(1 for v in pruned.nodes.values() for n in v if n.is_error)
        total = sum(1 for v in circuit.nodes.values() for n in v if n.is_error)
        mean_rel = sum(rel) / len(rel)
        print(
            f"{k:>4} {'':>12} {mean_rel:>21.2f} {share:>14.1%} "
            f"{kept:>7}/{total:<7}"
        )

    print("\n=== k=5, pruned graph: where the error nodes end up")
    circuit = build_jcircuit(
        lens,
        model,
        PROMPT,
        prune_percent=100.0, roots="all",
        k=5,
        stride=args.stride,
        layer_top=args.layer_top,
        layer_bottom=args.layer_bottom,
        positions=positions,
    ).prune(args.prune, roots="last")
    err = [n for v in circuit.nodes.values() for n in v if n.is_error]
    print(f"{len(err)} error nodes survive of "
          f"{sum(len(v) for v in circuit.nodes.values())} nodes total")
    out = [e for e in circuit.edges if e.source.is_error]
    print(f"{len(out)} of {len(circuit.edges)} surviving edges leave an error node")
    if out:
        print(f"\n{'edge':>52} {'A':>9} {'identity':>9}")
        for e in sorted(out, key=lambda e: -abs(e.score))[:8]:
            label = (
                f"{e.source.token}@L{e.source.layer}:{e.source.position}"
                f" -> {e.target.token!r}@L{e.target.layer}:{e.target.position}"
            )
            print(f"{label:>52} {e.score:>9.3f} {e.identity:>9.3f}")


if __name__ == "__main__":
    main()
