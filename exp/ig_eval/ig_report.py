# Copyright 2026 Tung-Yu (Tony) Wu
# SPDX-License-Identifier: Apache-2.0
# Added by Tung-Yu (Tony) Wu and his Claude.
"""Read ig_rows.json and answer: is EAP-IG worth it here?"""

import json
import statistics as st

D = "/tmp/claude-1000/-home-ubuntu/22d07dcd-b0c9-4b9d-8e70-ac71e3876cd6/scratchpad/"
rows = json.load(open(D + "ig_rows.json"))
MS = [2, 5, 10, 20]
EST = ["eap"] + [f"ig{m}" for m in MS]


def rel(row, key):
    return abs(row[key] - row["truth"]) / max(abs(row["truth"]), 1e-9)


def pearson(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return num / den if den else float("nan")


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for place, i in enumerate(order):
            r[i] = place
        return r

    return pearson(rank(xs), rank(ys))


strong = [r for r in rows if abs(r["truth"]) >= 1.0]
print(f"{len(rows)} edges over 5 prompts; {len(strong)} with |true| >= 1.0\n")

print("=== relative error vs a real ablation (all edges)")
print(f"{'estimator':>10} {'median':>9} {'p90':>9} {'worst':>9} {'>10% off':>9}")
for key in EST:
    errs = sorted(rel(r, key) for r in rows)
    n = len(errs)
    bad = sum(1 for e in errs if e > 0.10)
    print(
        f"{key:>10} {errs[n // 2]:>8.1%} {errs[int(0.9 * n)]:>8.1%} "
        f"{errs[-1]:>8.1%} {bad:>6}/{n}"
    )

print("\n=== relative error by layer distance (median)")
deltas = sorted({r["delta"] for r in rows})
print(f"{'delta':>6} {'n':>4} " + " ".join(f"{k:>8}" for k in EST))
for d in deltas:
    sub = [r for r in rows if r["delta"] == d]
    cells = " ".join(
        f"{sorted(rel(r, k) for r in sub)[len(sub) // 2]:>7.1%}" for k in EST
    )
    print(f"{d:>6} {len(sub):>4} {cells}")

print("\n=== does it change the ranking? (what a circuit actually uses)")
truth = [r["truth"] for r in rows]
print(f"{'estimator':>10} {'pearson':>9} {'spearman':>9}")
for key in EST:
    est = [r[key] for r in rows]
    print(f"{key:>10} {pearson(truth, est):>9.5f} {spearman(truth, est):>9.5f}")

print("\n=== per-prompt top-1 and top-3 agreement with the true ranking")
prompts = sorted({r["prompt"] for r in rows})
for key in EST:
    top1 = top3 = 0
    for p in prompts:
        for d in deltas:
            sub = [r for r in rows if r["prompt"] == p and r["delta"] == d]
            if len(sub) < 3:
                continue
            by_true = sorted(sub, key=lambda r: -abs(r["truth"]))
            by_est = sorted(sub, key=lambda r: -abs(r[key]))
            top1 += by_true[0]["src"] == by_est[0]["src"]
            top3 += (
                len({r["src"] for r in by_true[:3]} & {r["src"] for r in by_est[:3]})
                == 3
            )
    print(
        f"{key:>10}  top-1 {top1:>2}/{len(prompts) * len(deltas)}  "
        f"top-3 set {top3:>2}/{len(prompts) * len(deltas)}"
    )

print("\n=== cost, measured (per edge, one target)")
base = st.median(r["t_eap"] for r in rows)
print(f"{'estimator':>10} {'median s':>10} {'vs EAP':>8}")
print(f"{'eap':>10} {base:>10.4f} {1.0:>7.1f}x")
for m in MS:
    t = st.median(r[f"t_ig{m}"] for r in rows)
    print(f"{'ig' + str(m):>10} {t:>10.4f} {t / base:>7.1f}x")

print("\n=== where EAP is worst (largest relative error)")
print(
    f"{'delta':>5} {'a_src':>8} {'true':>9} {'eap':>9} {'ig20':>9} {'err_eap':>8} {'err_ig20':>9}  edge"
)
for r in sorted(rows, key=lambda r: -rel(r, "eap"))[:10]:
    print(
        f"{r['delta']:>5} {r['a_src']:>8.1f} {r['truth']:>9.3f} {r['eap']:>9.3f} "
        f"{r['ig20']:>9.3f} {rel(r, 'eap'):>7.0%} {rel(r, 'ig20'):>8.0%}  "
        f"{r['src']!r}->{r['tgt']!r}"
    )

print("\n=== does error track the perturbation size |a_src|?")
buckets = [(0, 10), (10, 20), (20, 40), (40, 1e9)]
print(f"{'|a_src|':>12} {'n':>4} {'eap':>8} {'ig5':>8} {'ig20':>8}")
for lo, hi in buckets:
    sub = [r for r in rows if lo <= abs(r["a_src"]) < hi]
    if not sub:
        continue
    cells = " ".join(
        f"{sorted(rel(r, k) for r in sub)[len(sub) // 2]:>7.1%}"
        for k in ("eap", "ig5", "ig20")
    )
    label = f"{lo}-{hi if hi < 1e9 else '+'}"
    print(f"{label:>12} {len(sub):>4} {cells}")
