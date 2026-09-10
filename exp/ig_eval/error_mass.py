# Copyright 2026 Tung-Yu (Tony) Wu
# SPDX-License-Identifier: Apache-2.0
# Added by Tung-Yu (Tony) Wu and his Claude.
"""How much of a target's incoming influence the named concepts miss.

Sweeps ``k`` on a real prompt and reports :meth:`JCircuit.error_mass` per level.
The question it answers: is the leftover *automatic* processing (raising k does
not recover it) or just an under-sized concept set (it does)?
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
    p.add_argument("--layer-bottom", type=int, default=18)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 20])
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
    levels = list(range(args.layer_bottom, args.layer_top, args.stride))

    print(f"prompt has {seq} tokens; scoring positions {positions}")
    print(f"\n{'k':>4} {'error mass':>11}   " + " ".join(f"L{l:<5}" for l in levels))
    for k in args.ks:
        circuit = build_jcircuit(
            lens,
            model,
            PROMPT,
            mode=1,
            k=k,
            stride=args.stride,
            layer_top=args.layer_top,
            layer_bottom=args.layer_bottom,
            positions=positions,
        )
        per_level = " ".join(f"{circuit.error_mass(layer=l):<6.3f}" for l in levels)
        print(f"{k:>4} {circuit.error_mass():>10.3f}   {per_level}")

    print("\n=== same vs cross position, k=5")
    circuit = build_jcircuit(
        lens,
        model,
        PROMPT,
        mode=1,
        k=5,
        stride=args.stride,
        layer_top=args.layer_top,
        layer_bottom=args.layer_bottom,
        positions=positions,
    )
    for label, flag in (("same position", True), ("cross position", False)):
        rows = [
            r
            for r in circuit.coverage
            if (r.source_position == r.target.position) is flag and r.total > 1e-8
        ]
        weight = sum(r.total for r in rows)
        mass = sum(r.error * r.total for r in rows) / weight
        share = weight / sum(r.total for r in circuit.coverage if r.total > 1e-8)
        print(
            f"{label:>16}  {len(rows):>5} rows  error mass {mass:.3f}  "
            f"({share:.1%} of all gradient norm)"
        )

    worst = sorted(
        (r for r in circuit.coverage if r.total > 1e-8),
        key=lambda r: -r.error * r.total,
    )[:8]
    print("\n=== targets losing the most influence (k=5)")
    print(f"{'target':>24} {'from':>9} {'||g||':>9} {'named':>7} {'error':>7}")
    for r in worst:
        print(
            f"{str(r.target):>24} {'L%d:%d' % (r.source_layer, r.source_position):>9} "
            f"{r.total:>9.2f} {r.fraction:>7.3f} {r.error:>7.3f}"
        )


if __name__ == "__main__":
    main()
