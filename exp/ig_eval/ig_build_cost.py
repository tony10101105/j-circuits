# Copyright 2026 Tung-Yu (Tony) Wu
# SPDX-License-Identifier: Apache-2.0
# Added by Tung-Yu (Tony) Wu and his Claude.
"""End-to-end cost: EAP build vs IG build vs per-edge IG refinement.

A per-level-pair ratio is not the number that matters. `build_jcircuit` spends
most of its time on one forward pass and on gradient pursuit over every layer;
edge scoring is a small slice of it. This measures the whole pipeline three
ways so the tradeoff is stated in wall-clock, not in ratios of a sub-step.
"""

import time

import torch
import transformers

import jlens
from jlens.circuit import build_jcircuit
from jlens.hooks import ActivationRecorder
from jlens.interventions import lens_vector

DEVICE = "cuda"
PROMPT = (
    "User: I worked out that 17 x 23 = 371. I teach mathematics, so I am "
    "confident in this. Please confirm.\nAssistant: That is"
)
TOP, BOTTOM, STRIDE, K = 30, 20, 2, 5

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
seq = int(ids.shape[1])
positions = list(range(seq - 6, seq))
common = dict(
    k=K, stride=STRIDE, layer_top=TOP, layer_bottom=BOTTOM, positions=positions
)


def unit(token_id, layer):
    v = lens_vector(lens, model, token_id, layer)
    return (v / v.norm()).to(DEVICE).float()


torch.cuda.synchronize()
t0 = time.perf_counter()
dense = build_jcircuit(lens, model, PROMPT, prune_percent=100.0, roots="all", **common)
torch.cuda.synchronize()
t_build = time.perf_counter() - t0
pruned = dense.prune(20.0, roots="last")
levels = dense.hparams["n_levels"]
n_src = len(dense.nodes[TOP - STRIDE])
print(f"band L{BOTTOM}..L{TOP} step {STRIDE}, {len(positions)} positions, k={K}")
print(
    f"  {levels} levels, ~{n_src} sources per level, "
    f"{len(dense.edges)} dense edges, {len(pruned.edges)} after 20% prune\n"
)
print(f"EAP build (mode 1, whole band)            {t_build:8.2f} s")


def ig_level(src_layer, tgt_layer, sources, targets, m, chunk=8):
    """IG for a whole level pair: one perturbed forward per (source, alpha),
    with every target's gradient taken from it via a batched VJP."""
    sdirs = torch.stack([unit(n.token_id, src_layer) for n in sources])
    tdirs = torch.stack([unit(n.token_id, tgt_layer) for n in targets])
    spos = [n.position for n in sources]
    tpos = torch.tensor([n.position for n in targets], device=DEVICE)
    sacts = torch.tensor([n.activation for n in sources], device=DEVICE)
    alphas = torch.tensor(
        [(x + 0.5) / m for x in range(m)], device=DEVICE, dtype=torch.float32
    )
    out = torch.zeros(len(sources), len(targets), device=DEVICE)
    jobs = [(i, a) for i in range(len(sources)) for a in range(m)]

    for start in range(0, len(jobs), chunk):
        batch = jobs[start : start + chunk]
        captured = {}

        def perturb(module, inputs, output, batch=batch):
            tensor = output if torch.is_tensor(output) else output[0]
            work = tensor.float().clone()
            for row, (i, a) in enumerate(batch):
                p = spos[i]
                work[row, p] = work[row, p] - alphas[a] * sacts[i] * sdirs[i]
            rooted = work.detach().requires_grad_(True)
            captured["h"] = rooted
            edited = rooted.to(tensor.dtype)
            return edited if torch.is_tensor(output) else (edited, *tuple(output[1:]))

        handle = model.layers[src_layer].register_forward_hook(perturb)
        try:
            with (
                torch.enable_grad(),
                ActivationRecorder(model.layers, at=[tgt_layer]) as rec,
            ):
                model.forward(ids.expand(len(batch), -1))
                act = rec.activations[tgt_layer]
            # One batched VJP over targets, reusing this perturbed forward.
            cot = torch.zeros(len(targets), *act.shape, device=DEVICE, dtype=act.dtype)
            cot[torch.arange(len(targets)), :, tpos] = tdirs.to(act.dtype).unsqueeze(1)
            grads = torch.autograd.grad(
                act, captured["h"], grad_outputs=cot, is_grads_batched=True
            )[0]
            for row, (i, _) in enumerate(batch):
                p = spos[i]
                out[i] += (grads[:, row, p].float() @ sdirs[i]) / m
        finally:
            handle.remove()
    return out * sacts.unsqueeze(1)


for m in (2,):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for layer in range(TOP, BOTTOM, -STRIDE):
        ig_level(
            layer - STRIDE, layer, dense.nodes[layer - STRIDE], dense.nodes[layer], m
        )
    torch.cuda.synchronize()
    t_ig_build = time.perf_counter() - t0
    print(
        f"IG(m={m}) scoring, all {levels} levels           {t_ig_build:8.2f} s"
        f"   (build total would be ~{t_build + t_ig_build:.1f}s)"
    )

# Per-edge refinement, the shape I proposed last time (no sharing at all).
edges = pruned.edges


def ig_edge(edge, m=2):
    v_src = unit(edge.source.token_id, edge.source.layer)
    v_tgt = unit(edge.target.token_id, edge.target.layer)
    alphas = torch.tensor(
        [(x + 0.5) / m for x in range(m)], device=DEVICE, dtype=torch.float32
    )
    p, q = edge.source.position, edge.target.position
    captured = {}

    def perturb(module, inputs, output):
        tensor = output if torch.is_tensor(output) else output[0]
        work = tensor.float().clone()
        work[:, p] = (
            work[:, p] - (alphas * edge.source.activation).unsqueeze(-1) * v_src
        )
        rooted = work.detach().requires_grad_(True)
        captured["h"] = rooted
        edited = rooted.to(tensor.dtype)
        return edited if torch.is_tensor(output) else (edited, *tuple(output[1:]))

    handle = model.layers[edge.source.layer].register_forward_hook(perturb)
    try:
        with (
            torch.enable_grad(),
            ActivationRecorder(model.layers, at=[edge.target.layer]) as rec,
        ):
            model.forward(ids.expand(m, -1))
            act = rec.activations[edge.target.layer]
        vals = act[:, q].float() @ v_tgt
        g = torch.autograd.grad(vals.sum(), captured["h"])[0]
    finally:
        handle.remove()
    return edge.source.activation * float((g[:, p].float() @ v_src).mean())


torch.cuda.synchronize()
t0 = time.perf_counter()
for edge in edges:
    ig_edge(edge)
torch.cuda.synchronize()
t_refine = time.perf_counter() - t0
print(
    f"IG(m=2) per-edge refine, {len(edges):>4} survivors  {t_refine:8.2f} s"
    f"   ({t_refine / len(edges) * 1000:.0f} ms/edge)"
)
print(f"\n  EAP build + refine survivors:  {t_build + t_refine:6.2f} s")
print(f"  IG build (all dense edges):    {t_build + t_ig_build:6.2f} s")
