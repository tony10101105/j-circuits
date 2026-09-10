# Copyright 2026 Tung-Yu (Tony) Wu
# SPDX-License-Identifier: Apache-2.0
# Added by Tung-Yu (Tony) Wu and his Claude.
"""Ask Claude which of a circuit's concepts are noise, and drop them.

A J-circuit's nodes come from a lens readout over the whole vocabulary, so a
pruned graph still carries two kinds of concept nobody wants to read:

- **non-semantic** tokens — ``"\\n"``, ``"˘"``, ``"/sp"``, byte fragments,
  markup and code punctuation that name no concept at all;
- **off-topic** tokens — real words with no bearing on this prompt. On the
  spider prompt, ``"约定"`` (agreement) and ``"好事"`` (good thing) are of this
  kind: perfectly good words, unrelated to how many legs a spider has.

Both are plausibly lens noise rather than mechanism, and both crowd out the
concepts that matter. Deciding which is which needs to read the prompt, which
is what makes it a job for a language model rather than a heuristic.

This module only *chooses*; :meth:`jlens.circuit.JCircuit.drop_tokens` does the
graph surgery, so the filtering rule is testable without a network call.

The classification is one request against the Messages API with a JSON schema
(``output_config.format``), so the reply is guaranteed to parse. Ids that are
not in the circuit are ignored, which makes a hallucinated id harmless.

Needs the ``anthropic`` extra (``pip install 'jlens[llm]'``) and credentials.
``ANTHROPIC_API_KEY`` is read from the environment, which
:func:`jlens.load_dotenv` will populate from a ``.env`` file; a profile from
``ant auth login`` also works, since a bare ``anthropic.Anthropic()`` resolves
either on its own.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence

from jlens._env import load_dotenv

logger = logging.getLogger(__name__)

#: Claude model used for the classification. Sonnet is the right tier here: the
#: task is per-token classification against a short prompt, not reasoning.
DEFAULT_MODEL = "claude-sonnet-5"

#: Cap on the reply. Thinking is on by default on this model and shares the
#: budget with the answer, so leave room for both.
DEFAULT_MAX_TOKENS = 8192

#: Refuse to classify a list longer than this — a dense circuit's vocabulary
#: runs to thousands of tokens, which is a sign the caller meant to prune first.
MAX_TOKENS_PER_CALL = 400

_SYSTEM = """\
You audit concept lists read out of a language model's internal activations by \
a readout lens. The lens scores the whole vocabulary, so its output mixes real \
concepts with noise.

These concepts are a trace of the model's own intermediate reasoning, not a \
summary of the answer. Hypotheses it raised and rejected, near-misses, related \
entities, and competing candidates are all part of that reasoning and are \
exactly what the trace exists to show. Removing them destroys the thing being \
studied.

Mark a token for removal ONLY when it is:

1. NON-SEMANTIC: it names no concept a person would recognise — whitespace, \
punctuation, diacritics, byte or subword fragments, orphaned affixes, markup, \
code operators, filler like "____", replacement characters.
2. UNRELATED: a real word with NO plausible connection to the prompt at all — \
not to its subject, not to any category the subject belongs to, not to any \
answer or wrong answer, not to anything the model might have passed through on \
the way. The bar is "no connection I can construct", not "not the final answer".

Keep everything else. In particular KEEP:
- alternative and incorrect answers, and quantities near the right one;
- other members of the subject's category (other animals, other tools, other \
countries), even when the prompt is not about them;
- parts, properties, and attributes of the subject, including ones it does not \
have;
- translations and morphological variants of any kept concept — content in \
another language is not noise by itself;
- broader or narrower categories of the subject.

When you are unsure, keep the token. A wrongly removed concept is invisible \
downstream and silently changes the conclusion; a wrongly kept one is merely \
clutter. Expect to remove a minority of the list.\
"""

_SCHEMA = {
    "type": "object",
    "properties": {
        "remove": {
            "type": "array",
            "description": "Vocabulary ids to remove from the graph.",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "The token's id."},
                    "reason": {
                        "type": "string",
                        "enum": ["non_semantic", "irrelevant"],
                        "description": "Which rule the token failed.",
                    },
                },
                "required": ["id", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["remove"],
    "additionalProperties": False,
}


def _client(api_key: str | None):
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "llm_token_filter_pruning needs the anthropic SDK: "
            "pip install 'jlens[llm]' (or pip install anthropic)"
        ) from exc
    if api_key:
        return anthropic.Anthropic(api_key=api_key)
    # Pull ANTHROPIC_API_KEY out of a .env if it is not already exported; a bare
    # client then resolves it (or ANTHROPIC_AUTH_TOKEN, or an `ant auth login`
    # profile) on its own.
    load_dotenv()
    return anthropic.Anthropic()


def select_noise_tokens(
    prompt: str,
    tokens: Sequence[tuple[int, str]],
    *,
    model: str = DEFAULT_MODEL,
    effort: str = "medium",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    api_key: str | None = None,
    client=None,
) -> dict[int, str]:
    """Ask Claude which of ``tokens`` are noise for ``prompt``.

    Args:
        prompt: The prompt the circuit was built on — the context the judgement
            of "irrelevant" is made against.
        tokens: ``(token_id, decoded_text)`` pairs, as returned by
            :meth:`jlens.circuit.JCircuit.tokens`.
        model: Claude model id.
        effort: ``"low"``/``"medium"``/``"high"``/``"xhigh"``/``"max"``. This is
            a classification, so the default is deliberately not the API's own
            ``"high"``.
        max_tokens: Cap on the reply (thinking and answer share it).
        api_key: Explicit key; omit to resolve from the environment.
        client: A pre-built ``anthropic.Anthropic``. Mainly for tests.

    Returns:
        ``{token_id: reason}`` for the ids to remove, restricted to ids that
        were actually offered — a hallucinated id is dropped, not raised on.
        Empty when ``tokens`` is empty.

    Raises:
        ValueError: If more than :data:`MAX_TOKENS_PER_CALL` tokens are given,
            or the reply does not parse.
        ImportError: If the ``anthropic`` SDK is not installed.
    """
    if not tokens:
        return {}
    if len(tokens) > MAX_TOKENS_PER_CALL:
        raise ValueError(
            f"{len(tokens)} tokens is more than MAX_TOKENS_PER_CALL="
            f"{MAX_TOKENS_PER_CALL}; prune the circuit further, narrow "
            "positions=, or lower k before filtering"
        )

    offered = {int(i) for i, _ in tokens}
    listing = "\n".join(f"{i}\t{text!r}" for i, text in tokens)
    user = (
        f"Prompt the model was running:\n<prompt>\n{prompt}\n</prompt>\n\n"
        f"Concepts read out of its activations, as id and decoded text:\n"
        f"<tokens>\n{listing}\n</tokens>\n\n"
        "Return the ids to remove."
    )

    client = client or _client(api_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=_SYSTEM,
        output_config={
            "effort": effort,
            "format": {"type": "json_schema", "schema": _SCHEMA},
        },
        messages=[{"role": "user", "content": user}],
    )
    if response.stop_reason == "refusal":
        raise ValueError("the model declined to classify these tokens")
    if response.stop_reason == "max_tokens":
        raise ValueError(
            f"reply hit max_tokens={max_tokens} before finishing; raise it or "
            "lower effort"
        )

    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise ValueError(
            f"no text block in the reply (stop_reason={response.stop_reason})"
        )
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"reply was not JSON: {text[:200]!r}") from exc

    chosen: dict[int, str] = {}
    for entry in parsed.get("remove", []):
        token_id = int(entry["id"])
        if token_id in offered:
            chosen[token_id] = str(entry.get("reason", ""))
        else:
            logger.warning("ignoring token id %d, which was not offered", token_id)
    return chosen
