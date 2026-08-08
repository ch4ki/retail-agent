"""Reading a value out of a MaskedFrame, for assertions.

Lived on `MaskedFrame` itself until nothing in production turned out to call it
— a convenience for tests does not belong on a domain type.
"""

from __future__ import annotations

from typing import Any

from retail_agent.safety.frame import MaskedFrame


def value(frame: MaskedFrame, column: str, row: int = 0) -> Any:
    """One cell, by column name."""
    index = frame.columns.index(column)
    return frame.rows[row][index]
