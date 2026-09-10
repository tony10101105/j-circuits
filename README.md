# j-circuits

Concept-level attribution graphs over the **Jacobian lens**: which concept, at
which layer and token position, caused the model to be thinking of which other
concept two layers up — measured causally, one backward pass for the whole
graph.

> **Built on [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)**,
> the companion code for [*Verbalizable Representations Form a Global Workspace
> in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html)
> (Anthropic PBC, Apache-2.0). That repo fits and applies the lens; this one adds
> the circuit layer on top. See [NOTICE](NOTICE) for the file-by-file split.
> This project is not affiliated with or endorsed by Anthropic.

The lens reads out what an internal activation is disposed to make the model
say. It linearly transports a residual-stream vector at any layer and position
into the final-layer basis, then decodes it with the model's own unembedding
into a ranked list of vocabulary tokens:

```
lens_l(h) = unembed( J_l @ h ), J_l = E[d h_final / d h_l]
```

A **J-circuit** turns those readouts into a graph. Each node is a concept at a
`(layer, position)`; each edge is the causal effect of projecting the source
concept out of the residual on the target concept's readout, estimated by
attribution patching so that one backward pass scores every candidate edge at
that level at once.

![Slice visualisation: ASCII-face example](assets/slice_vis.png)

## What this repo adds

| | |
|---|---|
| `jlens/circuit.py` | J-circuit construction: concept selection, batched-VJP edge scoring (EAP and EAP-IG), error nodes, identity/computed edge split, top-down pruning |
| `jlens/pursuit.py` | Non-negative gradient pursuit over lens vectors, so a block's `k` concepts are a decomposition rather than `k` spellings of one concept |
| `jlens/circuit_vis.py` | SVG rendering: concepts fixed to columns, edges typed as carry / compute / attention / error |
| `jlens/interventions.py` | Steer, Ablate, Swap and TopKAblation as forward hooks in lens coordinates |
| `jlens/token_filter.py` | Asks Claude which surviving concepts are lens noise, and drops them with their orphans |
| `jlens/_env.py` | `.env` loading for the demos |
| `jcircuit_demo.py`, `interventions_demo.py`, `exp/` | Runnable demos and the measurement scripts behind the numbers |

Upstream's `fitting.py`, `lens.py`, `hf.py`, `hooks.py`, `protocol.py`,
`vis.py` and `examples.py` do the fitting, the readout and the slice
visualisation; they are used as-is apart from small additive changes, each
marked in-file.

## Install

```bash
pip install -e .
```

## Usage

### Circuits

[`jcircuit_demo.py`](jcircuit_demo.py) builds a J-circuit — a layered
attribution graph over lens concepts — and writes it as an SVG. `--mode 2`
prunes it; `--llm-token-filter-pruning` then asks Claude which of the surviving
concepts are noise (byte fragments, punctuation, words unrelated to the prompt)
and drops them along with anything left dangling. On the spider prompt that
takes a pruned graph from 160 concept nodes to 70 without losing the reasoning:
competing hypotheses like `' wings'`, `'worm'`, `'蜜蜂'` (bee) are kept, while
`'/sp'`, `'˘'`, `'超市'` (supermarket) go.

```sh
python jcircuit_demo.py --mode 2 --llm-token-filter-pruning --svg circuit.svg
```

### Apply

To apply a pre-fitted lens:

```python
import transformers, jlens

hf = transformers.AutoModelForCausalLM.from_pretrained("org/model").cuda()
tok = transformers.AutoTokenizer.from_pretrained("org/model")
model = jlens.from_hf(hf, tok)

lens = jlens.JacobianLens.from_pretrained("org/lens-repo", filename="model/lens.pt")
lens_logits, model_logits, _ = lens.apply(
    model, "Fact: The currency used in the country shaped like a boot is",
    positions=[-2])
for layer, logits in sorted(lens_logits.items()):
    print(layer, [tok.decode([t]) for t in logits[0].topk(5).indices])
```

### Intervene

The paper's three causal interventions ("Writing", § Technical details of
J-lens use cases) are forward hooks over the same residual stream the lens
reads: `Steer(token, alpha)` adds `alpha * v_t` along a J-lens vector,
`Ablate(token)` projects the component out, and `Swap(source, target,
alpha=1.0)` exchanges the two lens coordinates (`h + alpha * V(sigma(c) - c)`
with `c = V^+ h`, computed via an exact rank-1 identity — see the
[`jlens.interventions`](jlens/interventions.py) module docstring).

```python
from jlens import Swap, greedy_generate

prompt = "Fact: the capital of France is"
band = lens.source_layers[8:21]

with lens.intervene(model, [Swap(" France", " China")], layers=band):
    print(greedy_generate(model, prompt, max_new_tokens=2))  # " Beijing" — was " Paris"
    lens_logits, _, _ = lens.apply(model, prompt, positions=[-1])  # reads the edited stream
```

The paper's top-`k` J-space ablation is `TopKAblation`: per position it
suppresses the `k` most active lens vectors (ranked at hook time from the
current residual) while sparing the clean model's own top-`exclude_top` output
tokens there (paper value 10; both are tunable), using a clean reference pass
bound to one prompt at construction:

```python
from jlens import TopKAblation

with TopKAblation(lens, model, prompt, k=8, exclude_top=10, layers=band):
    lens_logits, model_logits, _ = lens.apply(model, prompt)
```

[`interventions_demo.py`](interventions_demo.py) is a runnable tour of all
three, alongside the read.

### Credentials

The demos read a gitignored `.env` beside the code on startup — `HF_TOKEN` for
the gated model and lens downloads, `ANTHROPIC_API_KEY` for the token filter.
Anything already exported wins over the file. `jlens.load_dotenv()` does this
on its own if you want it in a script or notebook.

```
HF_TOKEN=hf-...
ANTHROPIC_API_KEY=sk-ant-...
```

### Fit

To fit a lens on your own model:

```python
lens = jlens.fit(model, prompts=my_prompts, checkpoint_path="out/ckpt.pt")
lens.save("out/jacobian_lens.pt")
```

The paper's lenses use 1000 sequences of 128 tokens from a pretraining-like
corpus. Quality saturates quickly (§9.3); ~100 prompts is usable. This is a
reference implementation and is not optimized; fitting time is dominated by
the model's own backward pass. Parallelize by running `fit()` on disjoint
slices and combining with `JacobianLens.merge()`.

## Walkthrough

[`walkthrough.ipynb`](walkthrough.ipynb) is the end-to-end notebook: load a
model, load (or fit) a lens, apply it at a few layers, and render a slice page
like the one above.

Reading a slice page:

- Each cell shows the lens top-1 word at that (position, layer); the
  superscript is its rank over the full vocabulary.
- Click a cell to select a (position, layer) and pin its top-1 token; pinned
  tokens get rank-tracking charts and a rank heatmap.
- The bottom row (`L = n_layers − 1`) is the model's actual output.

## License and attribution

Code is released under the Apache License 2.0 — see [LICENSE](LICENSE).

This repository is a derivative work of
[anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)
(Copyright 2026 Anthropic PBC, Apache-2.0). Files taken from upstream keep
their original copyright; files modified from upstream carry a change notice,
as Apache-2.0 section 4(b) requires; files added here are Copyright 2026
Tung-Yu (Tony) Wu. [NOTICE](NOTICE) lists which is which.

Neither the Apache License nor this notice grants any trademark rights, and
this project is not affiliated with or endorsed by Anthropic PBC.

## Upstream data and dependencies

The replication and lens-eval prompt sets in [`data/`](data/) are synthetic,
authored by Anthropic, and released under the same Apache License 2.0 as the
code. See the READMEs in [`data/experiments/`](data/experiments/) and
[`data/evaluations/`](data/evaluations/) for what each set contains.

The slice-vis pages use [d3](https://github.com/d3/d3) (ISC license), loaded
from the jsDelivr CDN with subresource integrity or inlined into
self-contained pages.

No model weights or text corpora are bundled; models and datasets downloaded
at run time are subject to their own licenses.
