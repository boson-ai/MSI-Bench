"""prompt_version — which evaluation setup prompt a manifest row is rendered with.

Calling spec:
    prompt_version_for(row: dict) -> "v1" | "v2" | None

A release suite stamps row["prompt_version"]: "v1" (MSI-Bench v1 paper prompt) or "v2"
(MSI-Bench v2 published-run prompt). Any other value raises pydantic.ValidationError
(a ValueError). A row without the field returns None, the current dev prompt, except a
row whose cell_id is an MSI-Bench v1 release id (msi-<lang>-<pattern>-<scene>-NN, with
the s/h probe suffix): suites built by the v1 adapter before the field existed are v1.

Deterministic; no side effects.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from ib.models.base import IbBaseModel

PromptVersion = Literal["v1", "v2"]

_V1_RELEASE_CELL_ID = re.compile(
    r"^msi-(?:en|zh)-(?:auth|disc|prior|retr|scope|seq)-"
    r"(?:commerce|domestic|edu|health|leisure|mobility|public|workprof)-\d{2}[sh]?$"
)


class PromptVersionField(IbBaseModel):
    prompt_version: PromptVersion | None = None


def prompt_version_for(row: dict[str, Any]) -> PromptVersion | None:
    version = PromptVersionField(prompt_version=row.get("prompt_version")).prompt_version
    if version is None and _V1_RELEASE_CELL_ID.match(str(row.get("cell_id") or "")):
        return "v1"
    return version
