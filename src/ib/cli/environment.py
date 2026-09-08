"""environment — load CLI secrets from the nearest repository-style `.env`.

Calling spec:
    load_cli_environment() -> None

Searches from the current working directory upward and loads the first `.env`.
Existing process environment variables take precedence.

Side effects: adds missing variables to os.environ.
"""

from __future__ import annotations

from dotenv import find_dotenv, load_dotenv


def load_cli_environment() -> None:
    """Load the nearest `.env` without overriding explicitly exported variables."""
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)
