"""base — strict Pydantic base for all MSI-Bench schema models.

Calling spec:
    class Foo(IbBaseModel): ...

IbBaseModel sets extra='forbid' so unknown keys / typos fail loud at validation
time rather than being silently dropped. Every config and schema model inherits
from it.

Side effects: none.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class IbBaseModel(BaseModel):
    """Base for all IB schema models. Rejects unknown fields."""

    model_config = ConfigDict(extra="forbid")
