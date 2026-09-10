# Copyright 2026 Tung-Yu (Tony) Wu
# SPDX-License-Identifier: Apache-2.0
# Added by Tung-Yu (Tony) Wu and his Claude.
"""EAP vs EAP-IG on J-lens edges, scored against real ablations.

The corruption is analytic here — ablating source s moves the residual along the
straight line h(a) = h - a*a_s*v_hat_s, a in [0,1] — so

    M_clean - M_ablated = integral_0^1 a_s*<v_hat_s, grad M(h(a))> da

is an *identity*, not an approximation. EAP evaluates the integrand at a=0 only;
EAP-IG averages m midpoint samples. Ground truth is a real ablation, so we can
score both.

Cost note: EAP shares one backward across every source (the gradient at the
clean point lands at all positions at once). IG cannot — each source needs its
own perturbed forward — so IG costs m * n_sources passes where EAP costs 1.
"""

import argparse
import json
import time

import torch
import transformers

import jlens
from jlens.hooks import ActivationRecorder
from jlens.interventions import lens_vector
from jlens.pursuit import pursue_lens

DEVICE = "cuda"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=4, help="source concepts per case")
    p.add_argument("--ms", type=int, nargs="+", default=[2, 5, 10, 20])
    p.add_argument("--out", default=None)
    return p.parse_args()


# (prompt, target layer, source-layer distances) — spread over strides and
# subject matter so the comparison is not one prompt's quirk.
CASES = [
    (
        "Q: How many legs does the animal that spins webs have?\nA: It has",
        28,
        [1, 2, 4, 8],
    ),
    ("Fact: the capital of France is", 24, [1, 2, 4, 8]),
    ("The Eiffel Tower is located in the city of", 26, [1, 2, 4, 8]),
    ("2 + 3 = 5, and 4 + 6 =", 22, [1, 2, 4, 8]),
    ("Water freezes at zero degrees", 20, [1, 2, 4, 8]),
]


def load():
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
    return model, lens


def unit(lens, model, token_id, layer):
    v = lens_vector(lens, model, token_id, layer)
    return (v / v.norm()).to(DEVICE).float()


def readout(model, layer, position, direction, ids):
    """Clean lens coordinate <v_hat, h_layer[position]>."""
    with torch.no_grad(), ActivationRecorder(model.layers, at=[layer]) as rec:
        model.forward(ids)
        return float(rec.activations[layer][0, position].float() @ direction)


def true_ablation(model, src_layer, position, v_src, tgt_layer, v_tgt, ids):
    """M_clean - M_ablated, projecting v_src out of h[src_layer, position]."""
    clean = readout(model, tgt_layer, position, v_tgt, ids)
    handle = model.layers[src_layer].register_forward_hook(_projector(position, v_src))
    try:
        ablated = readout(model, tgt_layer, position, v_tgt, ids)
    finally:
        handle.remove()
    return clean - ablated


def _projector(position, v_src):
    def hook(module, inputs, output):
        tensor = output if torch.is_tensor(output) else output[0]
        work = tensor.float()
        coeff = work[:, position] @ v_src
        work = work.clone()
        work[:, position] = work[:, position] - coeff.unsqueeze(-1) * v_src
        edited = work.to(tensor.dtype)
        return edited if torch.is_tensor(output) else (edited, *tuple(output[1:]))

    return hook


def path_gradients(
    model, src_layer, position, v_src, a_src, alphas, tgt_layer, v_tgt, ids
):
    """<v_hat_s, grad M> at each h - alpha*a_s*v_hat_s, batched over alphas.

    One forward/backward for the whole batch: row i carries alpha_i.
    """
    n = len(alphas)
    batch = ids.expand(n, -1)
    steps = torch.tensor(alphas, device=DEVICE, dtype=torch.float32)
    captured = {}

    def perturb(module, inputs, output):
        tensor = output if torch.is_tensor(output) else output[0]
        work = tensor.float().clone()
        delta = (steps * a_src).unsqueeze(-1) * v_src  # [n, d]
        work[:, position] = work[:, position] - delta
        # Graph root at the path point, so grad is evaluated exactly there.
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
            model.forward(batch)
            out = rec.activations[tgt_layer]
        m_values = out[:, position].float() @ v_tgt  # [n]
        grads = torch.autograd.grad(m_values.sum(), captured["h"])[0]
    finally:
        handle.remove()
    return (grads[:, position].float() @ v_src).detach()  # [n]


def main():
    args = parse_args()
    model, lens = load()
    rows = []

    for prompt, tgt_layer, deltas in CASES:
        ids = model.encode(prompt).to(DEVICE)
        position = int(ids.shape[1]) - 1  # last token

        with torch.no_grad(), ActivationRecorder(model.layers, at=[tgt_layer]) as rec:
            model.forward(ids)
            h_t = rec.activations[tgt_layer][0, position].detach().float()
        tgt_ids, _, _ = pursue_lens(
            model.unembed_weight,
            lens.jacobians[tgt_layer].to(DEVICE),
            h_t.unsqueeze(0),
            1,
        )
        tgt_id = int(tgt_ids[0, 0])
        v_tgt = unit(lens, model, tgt_id, tgt_layer)
        tgt_text = model.tokenizer.decode([tgt_id])

        for delta in deltas:
            src_layer = tgt_layer - delta
            with (
                torch.no_grad(),
                ActivationRecorder(model.layers, at=[src_layer]) as rec,
            ):
                model.forward(ids)
                h_s = rec.activations[src_layer][0, position].detach().float()
            src_ids, _, _ = pursue_lens(
                model.unembed_weight,
                lens.jacobians[src_layer].to(DEVICE),
                h_s.unsqueeze(0),
                args.k,
            )
            for src_id in [int(i) for i in src_ids[0] if i >= 0]:
                v_src = unit(lens, model, src_id, src_layer)
                a_src = float(h_s @ v_src)

                truth = true_ablation(
                    model, src_layer, position, v_src, tgt_layer, v_tgt, ids
                )
                # EAP: the integrand at alpha = 0
                t0 = time.perf_counter()
                eap = a_src * float(
                    path_gradients(
                        model,
                        src_layer,
                        position,
                        v_src,
                        a_src,
                        [0.0],
                        tgt_layer,
                        v_tgt,
                        ids,
                    )[0]
                )
                t_eap = time.perf_counter() - t0

                row = {
                    "prompt": prompt[:28],
                    "delta": delta,
                    "src": model.tokenizer.decode([src_id]),
                    "tgt": tgt_text,
                    "a_src": a_src,
                    "truth": truth,
                    "eap": eap,
                    "t_eap": t_eap,
                }
                for m in args.ms:
                    alphas = [(j + 0.5) / m for j in range(m)]
                    t0 = time.perf_counter()
                    g = path_gradients(
                        model,
                        src_layer,
                        position,
                        v_src,
                        a_src,
                        alphas,
                        tgt_layer,
                        v_tgt,
                        ids,
                    )
                    row[f"ig{m}"] = a_src * float(g.mean())
                    row[f"t_ig{m}"] = time.perf_counter() - t0
                rows.append(row)
        print(f"  ... {prompt[:40]!r} done ({len(rows)} edges so far)")

    out = (
        args.out
        or "/tmp/claude-1000/-home-ubuntu/22d07dcd-b0c9-4b9d-8e70-ac71e3876cd6/scratchpad/ig_rows.json"
    )
    with open(out, "w") as f:
        json.dump(rows, f)
    print(f"\n{len(rows)} edges -> {out}")


if __name__ == "__main__":
    main()
