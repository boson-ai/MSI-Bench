"""CLI environment loading behavior."""

from __future__ import annotations

import os

from ib.cli.environment import load_cli_environment


def test_loads_nearest_dotenv_from_current_working_tree(tmp_path, monkeypatch) -> None:
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=dotenv-key\nOPENROUTER_API_KEY=openrouter-dotenv-key\n",
        encoding="utf-8",
    )
    nested = tmp_path / "nested"
    nested.mkdir()
    monkeypatch.chdir(nested)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    load_cli_environment()

    assert os.environ["OPENAI_API_KEY"] == "dotenv-key"
    assert os.environ["OPENROUTER_API_KEY"] == "openrouter-dotenv-key"


def test_exported_environment_takes_precedence_over_dotenv(tmp_path, monkeypatch) -> None:
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=dotenv-key\nOPENROUTER_API_KEY=openrouter-dotenv-key\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "exported-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-exported-key")

    load_cli_environment()

    assert os.environ["OPENAI_API_KEY"] == "exported-key"
    assert os.environ["OPENROUTER_API_KEY"] == "openrouter-exported-key"
