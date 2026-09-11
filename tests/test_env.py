# Copyright 2026 Tung-Yu (Tony) Wu
# SPDX-License-Identifier: Apache-2.0
# Added by Tung-Yu (Tony) Wu and his Claude.
""".env loading: precedence, discovery, and not leaking secrets into logs."""

import logging
import os

import pytest

from jlens._env import load_dotenv


@pytest.fixture()
def env_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / ".env"
    return path


def test_reads_the_template_keys(env_file, monkeypatch):
    env_file.write_text(
        "ANTHROPIC_API_KEY=sk-ant\nHF_TOKEN=hf-x\n",
        encoding="utf-8",
    )
    for name in ("ANTHROPIC_API_KEY", "HF_TOKEN"):
        monkeypatch.delenv(name, raising=False)

    assert load_dotenv() == ["ANTHROPIC_API_KEY", "HF_TOKEN"]
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant"
    assert os.environ["HF_TOKEN"] == "hf-x"


def test_blank_values_are_left_alone(env_file, monkeypatch):
    env_file.write_text("HF_TOKEN=hf-x\nANTHROPIC_API_KEY=\n", encoding="utf-8")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert load_dotenv() == ["HF_TOKEN", "ANTHROPIC_API_KEY"]
    assert os.environ["ANTHROPIC_API_KEY"] == ""


def test_the_environment_beats_the_file(env_file, monkeypatch):
    env_file.write_text("ANTHROPIC_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-shell")

    assert load_dotenv() == []
    assert os.environ["ANTHROPIC_API_KEY"] == "from-shell"
    # ...unless the caller insists
    assert load_dotenv(override=True) == ["ANTHROPIC_API_KEY"]
    assert os.environ["ANTHROPIC_API_KEY"] == "from-file"


def test_missing_file_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_dotenv() == []
    assert load_dotenv(tmp_path / "nope.env") == []


def test_found_by_walking_up_from_a_subdirectory(env_file, monkeypatch):
    env_file.write_text("HF_TOKEN=hf-x\n", encoding="utf-8")
    nested = env_file.parent / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    assert load_dotenv() == ["HF_TOKEN"]
    assert os.environ["HF_TOKEN"] == "hf-x"


def test_malformed_lines_are_skipped(env_file, monkeypatch):
    env_file.write_text("HF_TOKEN=hf-x\nthis is not a pair\n", encoding="utf-8")
    monkeypatch.delenv("HF_TOKEN", raising=False)

    assert load_dotenv() == ["HF_TOKEN"]
    assert os.environ["HF_TOKEN"] == "hf-x"


def test_logs_names_but_never_values(env_file, monkeypatch, caplog):
    secret = "sk-do-not-log-me"
    env_file.write_text(f"ANTHROPIC_API_KEY={secret}\n", encoding="utf-8")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with caplog.at_level(logging.DEBUG, logger="jlens._env"):
        load_dotenv()
    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert "ANTHROPIC_API_KEY" in blob
    assert secret not in blob
