"""templates — Jinja2 prompt rendering for unified LLM calls.

Calling spec:
    text = render_prompt("dialogue/director.j2", {"cell": cell})

Templates are package-local under ``ib/llm/templates``. Rendering is deterministic
for a given context and has no side effects.
"""

from __future__ import annotations

from importlib.resources import files
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined


def _environment() -> Environment:
    template_dir = files("ib.llm").joinpath("templates")
    return Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=False,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


_ENV = _environment()


def render_prompt(template_name: str, context: dict[str, Any]) -> str:
    """Render a package prompt template with strict undefined-variable checks."""
    return _ENV.get_template(template_name).render(**context)
