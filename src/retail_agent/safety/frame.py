"""A query result after masking: the only shape rows travel in.

Lives beside the policy that produces it rather than with the agent, because
ownership is the point. `mask_dataframe` decides what a row may contain and
this is what comes out; anything downstream — a tool message, a trace, an eval
— reads this and never the raw DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pandas as pd

# A frame is carried in checkpointed conversation state, so results are plain
# values rather than a DataFrame. Only a slice is kept: prompts show at most 20
# rows, and a checkpoint per turn should not carry the whole result set.
MAX_STORED_ROWS = 100


@dataclass
class MaskedFrame:
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    row_count: int
    redactions: int
    dropped_columns: tuple[str, ...] = ()
    # The guard caps every query so a result stays printable and affordable,
    # and that cap is invisible in the result: 500 rows returned looks the same
    # whether there were 500 or 5,823. Measured on the real warehouse, the query
    # written for "how many loyal customers" matched 5,823 and the agent saw
    # 500. This is what stops a sample being read as the whole answer.
    truncated: bool = False

    @classmethod
    def from_dataframe(
        cls,
        frame: pd.DataFrame,
        *,
        row_count: int,
        redactions: int,
        dropped_columns: tuple[str, ...] = (),
        truncated: bool = False,
    ) -> "MaskedFrame":
        head = frame.head(MAX_STORED_ROWS)
        return cls(
            columns=tuple(str(column) for column in frame.columns),
            rows=tuple(
                tuple(_cell(value) for value in row)
                for row in head.itertuples(index=False, name=None)
            ),
            row_count=row_count,
            redactions=redactions,
            dropped_columns=tuple(dropped_columns),
            truncated=truncated,
        )

    @property
    def is_empty(self) -> bool:
        """No rows, or the aggregate spelling of no rows.

        `SUM(x) WHERE brand = 'Levis'` against a column holding `Levi's` returns
        one row containing NULL, not zero rows. Checking `row_count == 0` alone
        misses the case this exists for — confirmed against live BigQuery. The
        graph spent a model call to notice this; `run_sql` now says it in the
        tool result and the loop reads it directly.
        """
        if self.row_count == 0:
            return True
        if self.row_count != 1 or len(self.rows) != 1:
            return False
        return all(value is None for value in self.rows[0])

    def to_markdown(self, max_rows: int = 20) -> str:
        if not self.columns:
            return "_(no rows)_"

        shown = self.rows[:max_rows]
        lines = [
            "| " + " | ".join(self.columns) + " |",
            "| " + " | ".join("---" for _ in self.columns) + " |",
            *("| " + " | ".join(_fmt(v) for v in row) + " |" for row in shown),
        ]
        table = "\n".join(lines)

        hidden = self.row_count - len(shown)
        if hidden > 0:
            table += f"\n\n_({hidden} more rows not shown)_"
        return table


def _cell(value: Any) -> Any:
    """Coerce a pandas/numpy value into something a checkpointer can store."""
    try:
        if value is None or pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass  # arrays and other non-scalars land here

    if isinstance(value, (bool, int, float, str)):
        return value

    unwrap = getattr(value, "item", None)  # numpy scalars
    if callable(unwrap):
        try:
            return _cell(unwrap())
        except (AttributeError, TypeError, ValueError):
            pass

    if isinstance(value, Decimal):
        return float(value)
    return str(value)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
