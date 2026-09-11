# Copyright 2026 Tung-Yu (Tony) Wu
# SPDX-License-Identifier: Apache-2.0
# Added by Tung-Yu (Tony) Wu and his Claude.
"""Load a ``.env`` file into ``os.environ``.

The scripts in this repo need credentials that are awkward to keep in the
shell: ``HF_TOKEN`` for the gated model and lens downloads, and
``ANTHROPIC_API_KEY`` for :mod:`jlens.token_filter`. Keeping them in a
gitignored ``.env`` beside the code is the least error-prone option, so the
demos read it on startup. See ``.env.template`` for the expected keys.

Parsing is :mod:`dotenv`'s; this module only adds the precedence rule and a
list of what it set.

**A value already in the environment always wins.** A file cannot silently
override a key you exported for this one run, which is what makes it safe to
leave a stale ``.env`` lying around.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import dotenv_values, find_dotenv

logger = logging.getLogger(__name__)


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
    # usecwd: search up from the caller's cwd, not from this file's directory.
    dotenv = Path(path) if path else Path(find_dotenv(usecwd=True) or ".")
    if not dotenv.is_file():
        return []

    applied: list[str] = []
    for name, value in dotenv_values(dotenv, encoding="utf-8").items():
        if value is None or (not override and name in os.environ):
            continue
        os.environ[name] = value
        applied.append(name)

    if applied:
        # Names only. Never log a value.
        logger.info("loaded %s from %s", ", ".join(applied), dotenv)
    return applied
