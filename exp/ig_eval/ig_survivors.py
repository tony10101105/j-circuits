# Copyright 2026 Tung-Yu (Tony) Wu
# SPDX-License-Identifier: Apache-2.0
# Added by Tung-Yu (Tony) Wu and his Claude.
"""Ground truth for the edges that actually survive pruning.

ig_eval.py scores a spread of edges chosen to stress the estimators. This asks
the narrower, more decision-relevant question: on the edges a 10% prune keeps —
the ones that end up in the figure — is EAP-IG's number closer to a real
ablation than EAP's?

Every surviving edge gets its own ablation, including the cross-position ones
(ablate at the source's position, read the target at its own).
"""

import time

import torch
import transformers

import jlens
from jlens.circuit import _score_layer_pair, build_jcircuit
from jlens.hooks import ActivationRecorder
from jlens.interventions import lens_vector

DEVICE = "cuda"
PROMPT = "Q: How many legs does the animal that spins webs have?\nA: It has"
POSITIONS = ["legs", "animal", "spins", "webs", "has"]
SRC, TGT = 27, 28
PRUNE_PERCENTS = (10, 20)

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
ids = model.encode(PROMPT).to(DEVICE)


def unit(token_id, layer):
    v = lens_vector(lens, model, token_id, layer)
    return (v / v.norm()).to(DEVICE).float()


circuit = build_jcircuit(
    lens,
    model,
    PROMPT,
    prune_percent=100.0, roots="all",
    k=5,
    stride=1,
    layer_top=TGT,
    layer_bottom=SRC,
    positions=POSITIONS,
)
sources, targets = circuit.nodes[SRC], circuit.nodes[TGT]
sdirs = torch.stack([unit(n.token_id, SRC) for n in sources])
tdirs = torch.stack([unit(n.token_id, TGT) for n in targets])
spos = torch.tensor([n.position for n in sources], device=DEVICE)
tpos = torch.tensor([n.position for n in targets], device=DEVICE)
sacts = torch.tensor([n.activation for n in sources], device=DEVICE)
causal = spos.unsqueeze(1) <= tpos.unsqueeze(0)


def readout(layer, position, direction, batch=ids):
    with torch.no_grad(), ActivationRecorder(model.layers, at=[layer]) as rec:
        model.forward(batch)
        return float(rec.activations[layer][0, position].float() @ direction)


def ablate_hook(position, v_src):
    def hook(module, inputs, output):
        tensor = output if torch.is_tensor(output) else output[0]
        work = tensor.float().clone()
        coeff = work[:, position] @ v_src
        work[:, position] = work[:, position] - coeff.unsqueeze(-1) * v_src
        edited = work.to(tensor.dtype)
        return edited if torch.is_tensor(output) else (edited, *tuple(output[1:]))

    return hook


def true_edge(i, j):
    """M_clean - M_ablated for source i -> target j, positions respected."""
    p, q = int(spos[i]), int(tpos[j])
    clean = readout(TGT, q, tdirs[j])
    handle = model.layers[SRC].register_forward_hook(ablate_hook(p, sdirs[i]))
    try:
        ablated = readout(TGT, q, tdirs[j])
    finally:
        handle.remove()
    return clean - ablated


def ig_pair(i, j, m):
    """IG score for one edge: m midpoint alphas in a single batched pass."""
    alphas = torch.tensor(
        [(x + 0.5) / m for x in range(m)], device=DEVICE, dtype=torch.float32
    )
    p, q = int(spos[i]), int(tpos[j])
    captured = {}

    def perturb(module, inputs, output):
        tensor = output if torch.is_tensor(output) else output[0]
        work = tensor.float().clone()
        work[:, p] = work[:, p] - (alphas * sacts[i]).unsqueeze(-1) * sdirs[i]
        rooted = work.detach().requires_grad_(True)
        captured["h"] = rooted
        edited = rooted.to(tensor.dtype)
        return edited if torch.is_tensor(output) else (edited, *tuple(output[1:]))

    handle = model.layers[SRC].register_forward_hook(perturb)
    try:
        with torch.enable_grad(), ActivationRecorder(model.layers, at=[TGT]) as rec:
            model.forward(ids.expand(m, -1))
            out = rec.activations[TGT]
        vals = out[:, q].float() @ tdirs[j]
        g = torch.autograd.grad(vals.sum(), captured["h"])[0]
    finally:
        handle.remove()
    return float(sacts[i]) * float((g[:, p].float() @ sdirs[i]).mean())


with ActivationRecorder(model.layers, at=[SRC, TGT], start_graph_at=SRC) as rec:
    with torch.enable_grad():
        model.forward(ids)
    acts = {l: rec.activations[l] for l in (SRC, TGT)}
    eap = _score_layer_pair(acts, TGT, tdirs, tpos, sdirs, spos, sacts, 1, 32)

n_elig = int(causal.sum())
flat = eap.abs().masked_fill(~causal, -1e9).flatten()

for percent in PRUNE_PERCENTS:
    keep = max(1, round(percent / 100 * n_elig))
    picks = [divmod(int(f), len(targets)) for f in flat.topk(keep).indices]
    print(
        f"\n{'=' * 76}\n{percent}% prune: {keep} surviving edges of {n_elig}\n{'=' * 76}"
    )

    t0 = time.perf_counter()
    truth = [true_edge(i, j) for i, j in picks]
    t_truth = time.perf_counter() - t0
    est = {"eap": [float(eap[i, j]) for i, j in picks]}
    for m in (2, 5):
        t0 = time.perf_counter()
        est[f"ig{m}"] = [ig_pair(i, j, m) for i, j in picks]
        print(f"    IG(m={m}) on {keep} edges took {time.perf_counter() - t0:.1f}s")
    print(f"    ground truth ({keep} ablations) took {t_truth:.1f}s")

    print(
        f"\n{'estimator':>10} {'median rel':>11} {'p90 rel':>9} {'worst rel':>10} "
        f"{'median abs':>11} {'max abs':>9} {'closer':>8}"
    )
    for key, values in est.items():
        r = sorted(abs(v - t) / max(abs(t), 1e-9) for v, t in zip(values, truth))
        a = sorted(abs(v - t) for v, t in zip(values, truth))
        n = len(r)
        wins = sum(
            abs(v - t) < abs(e - t) for v, e, t in zip(values, est["eap"], truth)
        )
        won = "-" if key == "eap" else f"{wins}/{n}"
        print(
            f"{key:>10} {r[n // 2]:>10.2%} {r[int(0.9 * n)]:>8.2%} {r[-1]:>9.2%} "
            f"{a[n // 2]:>11.4f} {a[-1]:>9.4f} {won:>8}"
        )

    print(f"\n  worst EAP errors among the {keep} survivors:")
    order = sorted(
        range(keep),
        key=lambda x: -abs(est["eap"][x] - truth[x]) / max(abs(truth[x]), 1e-9),
    )
    print(f"  {'true':>9} {'eap':>9} {'ig5':>9} {'err_eap':>8} {'err_ig5':>8}  edge")
    for x in order[:8]:
        i, j = picks[x]
        t = truth[x]
        print(
            f"  {t:>9.3f} {est['eap'][x]:>9.3f} {est['ig5'][x]:>9.3f} "
            f"{abs(est['eap'][x] - t) / max(abs(t), 1e-9):>7.1%} "
            f"{abs(est['ig5'][x] - t) / max(abs(t), 1e-9):>7.1%}  "
            f"{sources[i].token!r}@{int(spos[i])} -> {targets[j].token!r}@{int(tpos[j])}"
        )
