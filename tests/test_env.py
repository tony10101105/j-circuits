# Copyright 2026 Tung-Yu (Tony) Wu
# SPDX-License-Identifier: Apache-2.0
# Added by Tung-Yu (Tony) Wu and his Claude.
""".env loading: parsing, precedence, and not leaking secrets into logs."""

import logging

import pytest

from jlens._env import find_dotenv, load_dotenv


@pytest.fixture()
def env_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / ".env"
    return path


def test_parses_the_shapes_a_real_file_uses(env_file, monkeypatch):
    env_file.write_text(
        "\n".join(
            [
                "# a comment",
                "",
                "OPENAI_API_KEY=sk-plain",
                'ANTHROPIC_API_KEY="sk-double"',
                "HF_TOKEN='hf-single'",
                "export EXPORTED=yes",
                "  SPACED = padded  ",
                "WITH_COMMENT=value # trailing",
                "URL=https://example.com/#anchor",
                "EMPTY=",
            ]
        ),
        encoding="utf-8",
    )
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "HF_TOKEN",
        "EXPORTED",
        "SPACED",
        "WITH_COMMENT",
        "URL",
        "EMPTY",
    ):
        monkeypatch.delenv(name, raising=False)

    import os

    applied = load_dotenv()
    assert applied[:3] == ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN"]
    assert os.environ["OPENAI_API_KEY"] == "sk-plain"
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-double"  # quotes stripped
    assert os.environ["HF_TOKEN"] == "hf-single"
    assert os.environ["EXPORTED"] == "yes"
    assert os.environ["SPACED"] == "padded"
    assert os.environ["WITH_COMMENT"] == "value"  # trailing comment dropped
    assert (
        os.environ["URL"] == "https://example.com/#anchor"
    )  # but a '#' in a URL is not
    assert os.environ["EMPTY"] == ""


def test_the_environment_beats_the_file(env_file, monkeypatch):
    import os

    env_file.write_text("ANTHROPIC_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-shell")

    assert load_dotenv() == []
    assert os.environ["ANTHROPIC_API_KEY"] == "from-shell"
    # ...unless the caller insists
    assert load_dotenv(override=True) == ["ANTHROPIC_API_KEY"]
    assert os.environ["ANTHROPIC_API_KEY"] == "from-file"


def test_missing_file_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert find_dotenv() is None
    assert load_dotenv() == []
    assert load_dotenv(tmp_path / "nope.env") == []


def test_found_by_walking_up_from_a_subdirectory(env_file, monkeypatch):
    import os

    env_file.write_text("HF_TOKEN=hf-x\n", encoding="utf-8")
    nested = env_file.parent / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    assert find_dotenv() == env_file
    assert load_dotenv() == ["HF_TOKEN"]
    assert os.environ["HF_TOKEN"] == "hf-x"


def test_malformed_lines_are_skipped_with_a_warning(env_file, monkeypatch, caplog):
    import os

    env_file.write_text("GOOD=1\nthis is not a pair\n=novalue\n", encoding="utf-8")
    monkeypatch.delenv("GOOD", raising=False)
    with caplog.at_level(logging.WARNING, logger="jlens._env"):
        assert load_dotenv() == ["GOOD"]
    assert os.environ["GOOD"] == "1"
    assert sum("skipped" in r.message for r in caplog.records) == 2


def test_logs_names_but_never_values(env_file, monkeypatch, caplog):
    secret = "sk-do-not-log-me"
    env_file.write_text(f"ANTHROPIC_API_KEY={secret}\n", encoding="utf-8")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with caplog.at_level(logging.DEBUG, logger="jlens._env"):
        load_dotenv()
    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert "ANTHROPIC_API_KEY" in blob
    assert secret not in blob
