# Copyright 2026 Tung-Yu (Tony) Wu
# SPDX-License-Identifier: Apache-2.0
# Added by Tung-Yu (Tony) Wu and his Claude.
"""Load a ``.env`` file into ``os.environ``.

The scripts in this repo need credentials that are awkward to keep in the
shell: ``HF_TOKEN`` for the gated model and lens downloads, and
``ANTHROPIC_API_KEY`` for :mod:`jlens.token_filter`. Keeping them in a
gitignored ``.env`` beside the code is the least error-prone option, so the
demos read it on startup.

Deliberately tiny and dependency-free — the format handled is
``KEY=value`` lines with optional ``export``, ``#`` comments, and quoted
values. Anything stranger belongs in a real secrets manager, not here.

**A value already in the environment always wins.** A file cannot silently
override a key you exported for this one run, which is what makes it safe to
leave a stale ``.env`` lying around.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

#: Name of the file searched for.
ENV_FILENAME = ".env"


def find_dotenv(start: str | Path | None = None) -> Path | None:
    """The nearest ``.env`` at or above ``start`` (default: the caller's cwd)."""
    here = Path(start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        candidate = directory / ENV_FILENAME
        if candidate.is_file():
            return candidate
    return None


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    # An unquoted trailing comment is not part of the value.
    return value.split(" #", 1)[0].strip()


def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> list[str]:
    """Read ``path`` (or the nearest ``.env``) into :data:`os.environ`.

    Args:
        path: The file to read. ``None`` searches upward from the cwd.
        override: Replace variables that are already set. Off by default, so
            the real environment beats the file.

    Returns:
        The names — never the values — of the variables that were set, in file
        order. Empty when no file was found, which is not an error: the
        environment may already carry everything needed.
    """
    dotenv = Path(path) if path else find_dotenv()
    if dotenv is None or not dotenv.is_file():
        return []

    applied: list[str] = []
    for number, raw in enumerate(
        dotenv.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or not name:
            logger.warning("%s:%d is not KEY=value; skipped", dotenv, number)
            continue
        if not override and name in os.environ:
            continue
        os.environ[name] = _unquote(value)
        applied.append(name)

    if applied:
        # Names only. Never log a value.
        logger.info("loaded %s from %s", ", ".join(applied), dotenv)
    return applied
