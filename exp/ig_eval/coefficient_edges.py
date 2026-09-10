# Copyright 2026 Tung-Yu (Tony) Wu
# SPDX-License-Identifier: Apache-2.0
# Added by Tung-Yu (Tony) Wu and his Claude.
"""Activation edges vs coefficient edges: the paper's A against ours.

Ours differentiates ``a_t = <v_t, h>``; the workspace paper differentiates the
pursuit coefficient ``c_t``. With the support held fixed the coefficient is
linear too, ``c = (V V^T)^-1 V h``, so both edges are ``scalar * <v_s, J^T
cotangent>`` -- they differ only in which scalar and which cotangent. This
script builds both blocks for one real (source layer, target layer) pair and
reports how far apart they land.
"""

import argparse

import torch
import transformers

import jlens
from jlens.circuit import _layer_concepts, _span_basis
from jlens.hooks import ActivationRecorder

DEVICE = "cuda"
PROMPT = "Q: How many legs does the animal that spins webs have?\nA: It has"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target-layer", type=int, default=26)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n-positions", type=int, default=4)
    return p.parse_args()


def block(nodes, dirs, layer, position):
    """Named rows of one (layer, position) block: (indices, directions, nodes)."""
    idx = [
        i
        for i, n in enumerate(nodes)
        if n.position == position and n.layer == layer and not n.is_error
    ]
    return torch.tensor(idx, device=dirs.device), dirs[idx], [nodes[i] for i in idx]


def duals(V):
    """Rows of (V V^T)^-1 V: the biorthogonal frame, <d_i, v_j> = delta_ij."""
    return torch.linalg.solve(V @ V.T, V)


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

    target_layer = args.target_layer
    source_layer = target_layer - args.stride
    ids = model.encode(PROMPT)
    seq = int(ids.shape[1])
    positions = list(range(seq - args.n_positions, seq))
    last = positions[-1]

    with ActivationRecorder(
        model.layers, at=[source_layer, target_layer], start_graph_at=source_layer
    ) as rec:
        with torch.enable_grad():
            model.forward(ids)
        acts = {l: rec.activations[l] for l in (source_layer, target_layer)}
        index = torch.tensor(positions, device=acts[target_layer].device)

        concepts = {}
        for layer in (source_layer, target_layer):
            residuals = acts[layer][0].detach().float().index_select(0, index)
            concepts[layer] = _layer_concepts(
                lens, model, layer, residuals, positions, args.k, "pursuit", True
            )

        s_nodes, s_dirs, _ = concepts[source_layer]
        t_nodes, t_dirs, _ = concepts[target_layer]
        _, V, s_named = block(s_nodes, s_dirs, source_layer, last)
        _, U, t_named = block(t_nodes, t_dirs, target_layer, last)

        h_src = acts[source_layer][0, last].detach().float()
        h_tgt = acts[target_layer][0, last].detach().float()
        a_src = V @ h_src
        a_tgt = U @ h_tgt
        c_src = torch.tensor([n.coefficient for n in s_named], device=V.device)
        c_tgt = torch.tensor([n.coefficient for n in t_named], device=U.device)

        G_src, G_tgt = V @ V.T, U @ U.T
        D_tgt = duals(U)

        # One VJP per target cotangent, for both conventions at once.
        out = acts[target_layer]
        cot = torch.zeros(
            2 * len(t_named), *out.shape, device=out.device, dtype=out.dtype
        )
        for i in range(len(t_named)):
            cot[i, 0, last] = U[i].to(out.dtype)
            cot[len(t_named) + i, 0, last] = D_tgt[i].to(out.dtype)
        grads = torch.autograd.grad(
            out,
            acts[source_layer],
            grad_outputs=cot,
            is_grads_batched=True,
            retain_graph=True,
        )[0][:, 0].float()[:, last]  # [2n, d]
        g_hat, g_dual = grads[: len(t_named)], grads[len(t_named) :]

    n = len(t_named)
    print(
        f"prompt {seq} tokens; site (L{source_layer},{last}) -> (L{target_layer},{last})"
    )
    print(f"source concepts: {[n_.token for n_ in s_named]}")
    print(f"target concepts: {[n_.token for n_ in t_named]}\n")

    print("=== 1. the Gram matrix is far from identity")
    off = G_src - torch.eye(len(s_named), device=G_src.device)
    print(
        f"source G: max |off-diag| {off.abs().max():.3f}  mean {off.abs().mean():.3f}"
    )
    print(f"          cond(G) = {torch.linalg.cond(G_src):.1f}")
    off = G_tgt - torch.eye(n, device=G_tgt.device)
    print(
        f"target G: max |off-diag| {off.abs().max():.3f}  mean {off.abs().mean():.3f}"
    )
    print(f"          cond(G) = {torch.linalg.cond(G_tgt):.1f}\n")

    print("=== 2. c = G^-1 a?  and how far the dual is from the atom")
    lstsq = torch.linalg.solve(G_tgt, a_tgt)
    print(
        f"{'concept':>12} {'a':>8} {'c':>8} {'G^-1 a':>8} {'||d||':>7} {'cos(d,v)':>9}"
    )
    for i, node in enumerate(t_named):
        cos = float(D_tgt[i] @ U[i] / D_tgt[i].norm())
        print(
            f"{node.token!r:>12} {a_tgt[i]:>8.2f} {c_tgt[i]:>8.2f} {lstsq[i]:>8.2f} "
            f"{D_tgt[i].norm():>7.2f} {cos:>9.3f}"
        )
    print()

    print("=== 3. the two edge blocks  [source, target]")
    A_mine = torch.diag(a_src) @ (V @ g_hat.T)
    A_paper = torch.diag(c_src) @ (V @ g_dual.T)
    flat_m, flat_p = A_mine.flatten(), A_paper.flatten()
    corr = torch.corrcoef(torch.stack([flat_m, flat_p]))[0, 1]
    print(f"pearson r over the {A_mine.numel()} edges: {corr:.4f}")
    print(f"|A| sum   ours {flat_m.abs().sum():.2f}   paper {flat_p.abs().sum():.2f}")
    order_m = flat_m.abs().argsort(descending=True)
    order_p = flat_p.abs().argsort(descending=True)
    print(f"top edge  ours #{int(order_m[0])}   paper #{int(order_p[0])}")
    overlap = len(set(order_m[:5].tolist()) & set(order_p[:5].tolist()))
    print(f"top-5 edge overlap: {overlap}/5\n")

    print(f"{'edge':>26} {'ours':>10} {'paper':>10} {'ratio':>8}")
    for i in order_m[:8]:
        s, t = int(i) // n, int(i) % n
        label = f"{s_named[s].token!r} -> {t_named[t].token!r}"
        ratio = float(flat_p[i] / flat_m[i]) if abs(flat_m[i]) > 1e-6 else float("nan")
        print(f"{label:>26} {flat_m[i]:>10.3f} {flat_p[i]:>10.3f} {ratio:>8.3f}")
    print()

    print("=== 3b. the closed-form conversion, A_paper = diag(c/a) A_ours G^-1")
    converted = torch.diag(c_src / a_src) @ A_mine @ torch.linalg.inv(G_tgt)
    print(f"max |A_paper - converted| = {(A_paper - converted).abs().max():.2e}")
    print(f"                (|A| scale {A_paper.abs().max():.2f})\n")

    print("=== 4. does the block conserve the total linear influence?")
    basis = _span_basis(torch.stack(list(V)))
    r_orth = h_src - basis @ (basis.T @ h_src)
    r_pursuit = h_src - c_src @ V
    print(
        f"{'target':>12} {'<h,g>':>10} {'paper sum':>10} {'ours sum':>10} {'ours gap':>9}"
    )
    for i, node in enumerate(t_named):
        truth_p = float(h_src @ g_dual[i])
        got_p = float(A_paper[:, i].sum() + r_pursuit @ g_dual[i])
        truth_m = float(h_src @ g_hat[i])
        got_m = float(A_mine[:, i].sum() + r_orth @ g_hat[i])
        print(
            f"{node.token!r:>12} {truth_m:>10.3f} "
            f"{got_p - truth_p:>+10.2e} {got_m:>10.3f} {got_m - truth_m:>+9.3f}"
        )
    print()

    # The sentence pins the source scalar (c) but not the target cotangent.
    # "mixed" keeps our cotangent v_t and only swaps the scalar, which is the
    # cheapest reading of the paper; compare all three.
    A_mixed = torch.diag(c_src) @ (V @ g_hat.T)
    print("=== 5. conservation needs only the SOURCE scalar")
    print(f"{'target':>12} {'<h,g_v>':>10} {'mixed sum':>10} {'err':>10}")
    for i, node in enumerate(t_named):
        truth = float(h_src @ g_hat[i])
        got = float(A_mixed[:, i].sum() + r_pursuit @ g_hat[i])
        print(f"{node.token!r:>12} {truth:>10.3f} {got:>10.3f} {got - truth:>+10.2e}")
    print()

    print("=== 6. how much does the UNPINNED cotangent choice move the graph?")
    for name, X, Y in (
        ("ours   vs mixed (scalar only)", A_mine, A_mixed),
        ("mixed  vs paper (cotangent only)", A_mixed, A_paper),
        ("ours   vs paper (both)", A_mine, A_paper),
    ):
        x, y = X.flatten(), Y.flatten()
        r = torch.corrcoef(torch.stack([x, y]))[0, 1]
        top = len(
            set(x.abs().argsort(descending=True)[:5].tolist())
            & set(y.abs().argsort(descending=True)[:5].tolist())
        )
        print(f"{name:>34}  r={r:.4f}  top-5 overlap {top}/5")
    print()

    print("=== 7. do incoming edges sum to the target node's own label?")
    print(f"{'target':>12} {'c_t':>8} {'<h,J^T d>':>10} {'a_t':>8} {'<h,J^T v>':>10}")
    for i, node in enumerate(t_named):
        print(
            f"{node.token!r:>12} {c_tgt[i]:>8.2f} {h_src @ g_dual[i]:>10.2f} "
            f"{a_tgt[i]:>8.2f} {h_src @ g_hat[i]:>10.2f}"
        )


if __name__ == "__main__":
    main()
