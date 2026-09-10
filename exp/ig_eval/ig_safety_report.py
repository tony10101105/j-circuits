# Copyright 2026 Tung-Yu (Tony) Wu
# SPDX-License-Identifier: Apache-2.0
# Added by Tung-Yu (Tony) Wu and his Claude.
"""Does EAP degrade on safety-relevant prompts, and does IG rescue it?"""

import json
import statistics as st

rows = json.load(open("exp/ig_eval/ig_safety_rows.json"))
EST = ["eap", "ig2", "ig5"]


def rel(r, k):
    return abs(r[k] - r["truth"]) / max(abs(r["truth"]), 1e-9)


def med(vals):
    vals = sorted(vals)
    return vals[len(vals) // 2] if vals else float("nan")


def pct(vals, q):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(q * len(vals)))] if vals else float("nan")


def pearson(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return num / den if den else float("nan")


print(
    f"{len(rows)} pruned-surviving edges over {len({r['prompt'] for r in rows})} "
    f"safety prompts\n"
)

print("=== overall relative error")
print(
    f"{'estimator':>10} {'median':>9} {'p90':>9} {'p99':>9} {'worst':>9} {'>10% off':>10}"
)
for k in EST:
    e = [rel(r, k) for r in rows]
    bad = sum(1 for x in e if x > 0.10)
    print(
        f"{k:>10} {med(e):>8.2%} {pct(e, 0.9):>8.2%} {pct(e, 0.99):>8.2%} "
        f"{max(e):>8.2%} {bad:>7}/{len(e)}"
    )

print("\n=== per prompt (median relative error)")
print(f"{'prompt':>18} {'n':>4} " + " ".join(f"{k:>9}" for k in EST) + "   IG closer")
for name in sorted({r["prompt"] for r in rows}):
    sub = [r for r in rows if r["prompt"] == name]
    cells = " ".join(f"{med([rel(r, k) for r in sub]):>8.2%}" for k in EST)
    wins = sum(abs(r["ig2"] - r["truth"]) < abs(r["eap"] - r["truth"]) for r in sub)
    print(f"{name:>18} {len(sub):>4} {cells}   {wins:>3}/{len(sub)}")

print("\n=== same-position vs cross-position (attention) edges")
print(f"{'kind':>16} {'n':>4} " + " ".join(f"{k:>9}" for k in EST))
for label, flag in (("same position", False), ("cross position", True)):
    sub = [r for r in rows if r["cross_position"] is flag]
    if not sub:
        continue
    cells = " ".join(f"{med([rel(r, k) for r in sub]):>8.2%}" for k in EST)
    print(f"{label:>16} {len(sub):>4} {cells}")

print("\n=== by edge magnitude")
print(f"{'|true|':>12} {'n':>4} " + " ".join(f"{k:>9}" for k in EST) + "   IG closer")
for lo, hi, lbl in [
    (0, 0.25, "< 0.25"),
    (0.25, 1, "0.25-1"),
    (1, 5, "1-5"),
    (5, 20, "5-20"),
    (20, 1e9, ">= 20"),
]:
    sub = [r for r in rows if lo <= abs(r["truth"]) < hi]
    if not sub:
        continue
    cells = " ".join(f"{med([rel(r, k) for r in sub]):>8.2%}" for k in EST)
    wins = sum(abs(r["ig2"] - r["truth"]) < abs(r["eap"] - r["truth"]) for r in sub)
    print(f"{lbl:>12} {len(sub):>4} {cells}   {wins:>3}/{len(sub)}")

print("\n=== does the sign ever flip? (an edge read as support vs suppression)")
for k in EST:
    flips = [r for r in rows if r[k] * r["truth"] < 0 and abs(r["truth"]) > 0.1]
    print(f"{k:>10}: {len(flips):>3} sign errors on edges with |true| > 0.1")

print("\n=== correlation with truth")
truth = [r["truth"] for r in rows]
for k in EST:
    print(f"{k:>10}: pearson {pearson(truth, [r[k] for r in rows]):.5f}")

print("\n=== worst EAP errors overall (|true| > 0.5)")
big = [r for r in rows if abs(r["truth"]) > 0.5]
print(
    f"{'prompt':>17} {'true':>8} {'eap':>8} {'ig2':>8} {'e_eap':>7} {'e_ig2':>7}  edge"
)
for r in sorted(big, key=lambda r: -rel(r, "eap"))[:12]:
    tag = "x" if r["cross_position"] else " "
    print(
        f"{r['prompt']:>17} {r['truth']:>8.3f} {r['eap']:>8.3f} {r['ig2']:>8.3f} "
        f"{rel(r, 'eap'):>6.1%} {rel(r, 'ig2'):>6.1%} {tag} "
        f"{r['src']!r}->{r['tgt']!r}"
    )

print("\n=== is m=5 better than m=2?")
w = sum(abs(r["ig5"] - r["truth"]) < abs(r["ig2"] - r["truth"]) for r in rows)
print(
    f"  ig5 closer than ig2 on {w}/{len(rows)} edges "
    f"(median {med([rel(r, 'ig2') for r in rows]):.2%} vs "
    f"{med([rel(r, 'ig5') for r in rows]):.2%})"
)
