# Copyright 2026 Tung-Yu (Tony) Wu
# SPDX-License-Identifier: Apache-2.0
# Added by Tung-Yu (Tony) Wu and his Claude.
"""Play with the Jacobian lens: read the workspace, then steer / ablate / swap it.

Defaults reproduce the paper's "broadcast representation" swap (Figure 18) on
an open-weights model: the lens READS the argument France out of the residual
stream on "Fact: the capital of France is"; SWAPping its lens coordinate for
China at mid layers flips the model's answer Paris -> Beijing, and the *same*
swap redirects other functions of the same argument (continent Europe -> Asia).
STEER injects a concept at increasing strength (the paper's introspection
protocol); ABLATE projects a concept out of the band and shows whether later
layers re-derive it.

Defaults download Qwen3-4B and its pre-fitted lens from the HuggingFace Hub
(~8.5 GB on first run). Examples:

  python interventions_demo.py
  python interventions_demo.py --swap-target " Italy" --steer-token " dragons"
  # the paper's two-hop example (works ~half the time on models this small):
  python interventions_demo.py \\
      --prompt "Q: How many legs does the animal that spins webs have?\\nA: It has" \\
      --swap-source " spider" --swap-target " ant" \\
      --swap-prompts "Q: How many legs does the animal that spins webs have?\\nA: It has"
"""

from __future__ import annotations

import argparse

import torch

import jlens
from jlens import Ablate, Intervention, Steer, Swap, greedy_generate
from jlens.hooks import ActivationRecorder
from jlens.interventions import resolve_token_id

SWAP_PROMPTS = (
    "Fact: the capital of France is;"
    "Fact: most people in France speak;"
    "Fact: France is located on the continent of"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument(
        "--lens", default="neuronpedia/jacobian-lens", help="repo id, dir, or file"
    )
    p.add_argument(
        "--lens-file",
        default="qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt",
        help="path inside the lens repo/dir",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--band",
        default=None,
        metavar="LO-HI",
        help="layer band to intervene on, inclusive (default: ~25-60%% of the "
        "fitted layers)",
    )
    p.add_argument("--prompt", default="Fact: the capital of France is")
    p.add_argument("--swap-source", default=" France")
    p.add_argument("--swap-target", default=" China")
    p.add_argument(
        "--swap-alpha", type=float, default=1.0, help="2.0 = paper's double-strength"
    )
    p.add_argument(
        "--swap-prompts",
        default=SWAP_PROMPTS,
        help="';'-separated prompts the same swap is applied to (the paper's "
        "function templates)",
    )
    p.add_argument("--ablate-token", default=" France")
    p.add_argument("--steer-prompt", default="I've been thinking a lot about")
    p.add_argument("--steer-token", default=" Paris")
    p.add_argument(
        "--steer-alphas",
        default="0,2,4,8,16",
        help="comma-separated strengths to sweep",
    )
    p.add_argument("--max-new-tokens", type=int, default=12)
    return p.parse_args()


def next_logits(model: jlens.LensModel, prompt: str) -> torch.Tensor:
    """Model's next-token logits at the last position (one forward, no cache)."""
    final = model.n_layers - 1
    with torch.no_grad(), ActivationRecorder(model.layers, at=[final]) as recorder:
        model.forward(model.encode(prompt))
        return model.unembed(recorder.activations[final][:, -1].float())[0].cpu()


def top5(tokenizer, logits: torch.Tensor) -> str:
    values, indices = logits.topk(5)
    return "  ".join(
        f"{tokenizer.decode([i])!r}({v:.1f})"
        for i, v in zip(indices, values, strict=True)
    )


def rank_of(logits: torch.Tensor, token_id: int) -> int:
    return int((logits > logits[token_id]).sum())


def main() -> None:
    args = parse_args()
    jlens.load_dotenv()  # HF_TOKEN for the gated model and lens downloads
    print(f"loading {args.model} on {args.device} ...")
    import transformers

    hf = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16 if args.device.startswith("cuda") else None
    ).to(args.device)
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    model = jlens.from_hf(hf, tok)

    print(f"loading lens {args.lens} :: {args.lens_file} ...")
    lens = jlens.JacobianLens.from_pretrained(args.lens, filename=args.lens_file)
    sls = lens.source_layers
    if args.band:
        lo, hi = (int(x) for x in args.band.split("-"))
        band = [l for l in sls if lo <= l <= hi]
    else:
        band = sls[int(0.25 * len(sls)) : int(0.6 * len(sls))]
    if not band:
        raise SystemExit(f"empty layer band; fitted layers are {sls}")
    print(f"{lens!r}\nintervention band: layers {band[0]}..{band[-1]}\n")
    gen = lambda prompt: greedy_generate(  # noqa: E731
        model, prompt, max_new_tokens=args.max_new_tokens
    )
    # a few layers to read out at: spread over the band, plus one before/after
    show = sorted(
        {
            sls[max(0, sls.index(band[0]) - 4)],
            *band[:: len(band) // 2],
            band[-1],
            sls[min(len(sls) - 1, sls.index(band[-1]) + 4)],
        }
    )

    def lens_ranks(prompt: str, token_id: int) -> dict[int, int]:
        lens_logits, _, _ = lens.apply(model, prompt, layers=show, positions=[-1])
        return {layer: rank_of(lens_logits[layer][0], token_id) for layer in show}

    # ---- READ: the argument is in the lens at mid layers ------------------- #
    print(f"=== READ  {args.prompt!r}")
    source_id = resolve_token_id(model, args.swap_source)
    lens_logits, model_logits, _ = lens.apply(
        model, args.prompt, layers=show, positions=[-1]
    )
    for layer in show:
        row = lens_logits[layer][0]
        marker = " <- band" if band[0] <= layer <= band[-1] else ""
        print(
            f"  L{layer:>3}  {args.swap_source!r} rank {rank_of(row, source_id):>5}"
            f"  | top: {top5(tok, row)}{marker}"
        )
    print(f"  model next token: {top5(tok, model_logits[0])}")
    print(f"  continuation: {gen(args.prompt)!r}\n")

    # ---- STEER: h <- h + alpha * v_t, strength sweep ----------------------- #
    print(f"=== STEER {args.steer_token!r} into {args.steer_prompt!r}")
    steer_id = resolve_token_id(model, args.steer_token)
    for alpha in (float(a) for a in args.steer_alphas.split(",")):
        edits = [Steer(args.steer_token, alpha)]
        with Intervention(lens, model, edits, layers=band):
            logits = next_logits(model, args.steer_prompt)
            print(
                f"  alpha {alpha:>5.1f}  token rank {rank_of(logits, steer_id):>5}"
                f"  continuation: {gen(args.steer_prompt)!r}"
            )
    print()

    # ---- ABLATE: project the direction out; do later layers re-derive it? -- #
    print(f"=== ABLATE {args.ablate_token!r} on {args.prompt!r}")
    ablate_id = resolve_token_id(model, args.ablate_token)
    clean_ranks = lens_ranks(args.prompt, ablate_id)
    with Intervention(lens, model, [Ablate(args.ablate_token)], layers=band):
        ablated_ranks = lens_ranks(args.prompt, ablate_id)
        continuation = gen(args.prompt)
    for layer in show:
        marker = " <- band" if band[0] <= layer <= band[-1] else ""
        print(
            f"  L{layer:>3}  lens rank {clean_ranks[layer]:>5} -> "
            f"{ablated_ranks[layer]:>5}{marker}"
        )
    print(f"  continuation: {gen(args.prompt)!r}  (clean)")
    print(f"  continuation: {continuation!r}  (ablated)\n")

    # ---- SWAP: one coordinate exchange redirects several functions --------- #
    print(
        f"=== SWAP  {args.swap_source!r} -> {args.swap_target!r}"
        f" (alpha={args.swap_alpha}) applied identically to each prompt"
    )
    swap = Swap(args.swap_source, args.swap_target, alpha=args.swap_alpha)
    for prompt in args.swap_prompts.split(";"):
        clean = gen(prompt)
        with Intervention(lens, model, [swap], layers=band):
            swapped = gen(prompt)
        print(f"  {prompt!r}\n    clean  : {clean!r}\n    swapped: {swapped!r}")


if __name__ == "__main__":
    main()
