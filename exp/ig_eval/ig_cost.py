# Copyright 2026 Tung-Yu (Tony) Wu
# SPDX-License-Identifier: Apache-2.0
# Added by Tung-Yu (Tony) Wu and his Claude.
"""The decision-relevant cost: EAP vs EAP-IG *inside a level pair*.

`build_jcircuit` scores a whole level pair with ONE batched VJP — the gradient
at the clean point lands at every source position at once, so n_sources costs
nothing extra. IG breaks that sharing: the gradient must be evaluated at
h - alpha*a_s*v_hat_s, which is a different point for every source. So IG needs
n_sources * m perturbed forward+backward passes where EAP needs 1.

This measures both on a real level pair, and checks whether the two disagree
about which edges survive pruning.
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


# One real level pair from a real circuit.
circuit = build_jcircuit(
    lens,
    model,
    PROMPT,
    mode=1,
    k=5,
    stride=1,
    layer_top=28,
    layer_bottom=27,
    positions=POSITIONS,
)
TGT, SRC = 28, 27
sources = circuit.nodes[SRC]
targets = circuit.nodes[TGT]
print(f"level pair L{SRC} -> L{TGT}: {len(sources)} sources x {len(targets)} targets")

# ---------------------------------------------------------------- EAP timing
with ActivationRecorder(model.layers, at=[SRC, TGT], start_graph_at=SRC) as rec:
    with torch.enable_grad():
        model.forward(ids)
    acts = {l: rec.activations[l] for l in (SRC, TGT)}
    sdirs = torch.stack([unit(n.token_id, SRC) for n in sources])
    tdirs = torch.stack([unit(n.token_id, TGT) for n in targets])
    spos = torch.tensor([n.position for n in sources], device=DEVICE)
    tpos = torch.tensor([n.position for n in targets], device=DEVICE)
    sacts = torch.tensor([n.activation for n in sources], device=DEVICE)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(3):
        _score_layer_pair(acts, TGT, tdirs, tpos, sdirs, spos, sacts, 1, 32)
    torch.cuda.synchronize()
    t_eap = (time.perf_counter() - t0) / 3
print(f"EAP  (1 batched VJP, all sources at once): {t_eap * 1000:7.1f} ms")


# ------------------------------------------------------- IG timing, batched
def ig_scores(m, chunk=16):
    """IG for every (source, target) pair. One perturbed pass per (source, alpha)."""
    alphas = torch.tensor(
        [(j + 0.5) / m for j in range(m)], device=DEVICE, dtype=torch.float32
    )
    out = torch.zeros(len(sources), len(targets), device=DEVICE)
    jobs = [(i, a) for i in range(len(sources)) for a in range(m)]
    for start in range(0, len(jobs), chunk):
        batch = jobs[start : start + chunk]
        rows = ids.expand(len(batch), -1)
        captured = {}

        def perturb(module, inputs, output, batch=batch):
            tensor = output if torch.is_tensor(output) else output[0]
            work = tensor.float().clone()
            for row, (i, a) in enumerate(batch):
                p = int(spos[i])
                work[row, p] = work[row, p] - alphas[a] * sacts[i] * sdirs[i]
            rooted = work.detach().requires_grad_(True)
            captured["h"] = rooted
            edited = rooted.to(tensor.dtype)
            return edited if torch.is_tensor(output) else (edited, *tuple(output[1:]))

        handle = model.layers[SRC].register_forward_hook(perturb)
        try:
            with (
                torch.enable_grad(),
                ActivationRecorder(model.layers, at=[TGT]) as rec2,
            ):
                model.forward(rows)
                tgt_act = rec2.activations[TGT]
            # one cotangent set per target, batched
            for j, tdir in enumerate(tdirs):
                q = int(tpos[j])
                m_val = tgt_act[:, q].float() @ tdir
                g = torch.autograd.grad(
                    m_val.sum(), captured["h"], retain_graph=(j < len(tdirs) - 1)
                )[0]
                for row, (i, _) in enumerate(batch):
                    p = int(spos[i])
                    out[i, j] += float(g[row, p].float() @ sdirs[i]) / m
        finally:
            handle.remove()
    return out * sacts.unsqueeze(1)


for m in (2, 5):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ig = ig_scores(m)
    torch.cuda.synchronize()
    t_ig = time.perf_counter() - t0
    print(
        f"IG(m={m:>2}) ({len(sources)} sources x {m} alphas = "
        f"{len(sources) * m:>3} perturbed passes): {t_ig * 1000:7.1f} ms  "
        f"= {t_ig / t_eap:6.1f}x EAP"
    )

# --------------------------------------------- do they disagree on pruning?
eap = _score_layer_pair(acts, TGT, tdirs, tpos, sdirs, spos, sacts, 1, 32)
ig5 = ig_scores(5)
causal = spos.unsqueeze(1) <= tpos.unsqueeze(0)
n_elig = int(causal.sum())
for percent in (10, 20, 50):
    keep = max(1, round(percent / 100 * n_elig))
    a = set(map(int, eap.abs().masked_fill(~causal, -1e9).flatten().topk(keep).indices))
    b = set(map(int, ig5.abs().masked_fill(~causal, -1e9).flatten().topk(keep).indices))
    print(
        f"prune {percent:>2}%: keep {keep:>3} of {n_elig} -> "
        f"{len(a & b):>3} edges in common ({len(a & b) / keep:.0%})"
    )
