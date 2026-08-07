"""Which values a column actually holds, for the model to read.

Two live eval failures came from guessing a literal. The model wrote
`WHERE gender = 'female'` against a column holding 'F', and
`WHERE status = 'cancelled'` against one holding 'Cancelled'. Both were valid
SQL, passed the guard, ran without error and returned 0 rows — the worst failure
shape available, because no layer reports a problem and the agent narrates the
zero as a finding.

No prompt rule fixes that; the model cannot know a value it has never seen. A
schema line that reads `gender STRING -- one of: 'F', 'M'` does.

This puts *data* into the prompt, which is what the PII policy exists to stop,
so the exclusion is by policy and not by cardinality: a restricted column is
never enumerated even if it holds three values. Relying on "there are too many
last names to list" would be luck rather than a rule.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import replace

from retail_agent.datasources.base import TableSchema

# Measured on theLook, where the gap is clean: the columns worth listing top out
# at 26 distinct values (product category, order status, traffic source, country,
# gender, department) and the next one up is state at 230. Listing 230 states
# would cost more prompt than the mistakes it prevents.
MAX_DISTINCT = 30

# Only these can hold a misspelled literal. Enumerating an id or a timestamp is
# noise, and enumerating a float is meaningless.
_ENUMERABLE_TYPES = frozenset({"STRING"})


def enumerable_columns(
    schema: TableSchema, *, restricted: Collection[str]
) -> tuple[str, ...]:
    """Columns of this table that may have their values read into a prompt.

    `restricted` is the PII policy's set. Nothing in it is ever enumerated.
    """
    blocked = {name.lower() for name in restricted}
    return tuple(
        column.name
        for column in schema.columns
        if column.type.upper() in _ENUMERABLE_TYPES
        and column.name.lower() not in blocked
    )


def build_discovery_query(
    table: str, columns: Sequence[str], *, dataset: str
) -> str:
    """One query per table rather than one per column.

    `APPROX_COUNT_DISTINCT` says whether the column is an enumeration at all,
    and `APPROX_TOP_COUNT` brings back the values in the same pass — so a table
    costs a single scan instead of one per candidate column.
    """
    if not columns:
        return ""

    selections = []
    for column in columns:
        selections.append(f"APPROX_COUNT_DISTINCT(`{column}`) AS `{column}__n`")
        # One more than the ceiling, so a column sitting exactly at the limit is
        # distinguishable from one just over it rather than silently truncated
        # into looking small enough.
        selections.append(
            f"APPROX_TOP_COUNT(`{column}`, {MAX_DISTINCT + 1}) AS `{column}__v`"
        )

    return f"SELECT {', '.join(selections)} FROM `{dataset}.{table}`"


def read_discovery_row(
    row: Mapping[str, object], columns: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    """Turn one discovery row into {column: values}, dropping what does not fit."""
    found: dict[str, tuple[str, ...]] = {}

    for column in columns:
        count = row.get(f"{column}__n")
        if not isinstance(count, int) or count > MAX_DISTINCT:
            continue

        entries = row.get(f"{column}__v") or []
        values = []
        for entry in entries:
            value = entry.get("value") if isinstance(entry, Mapping) else getattr(entry, "value", None)
            # NULL is not a literal worth offering: `WHERE status = 'None'`
            # matches nothing and reads as a real filter.
            if value is None:
                continue
            values.append(str(value))

        if values:
            # Sorted, not frequency-ordered: this text goes into every prompt,
            # and an order that shifts with the data would invalidate prompt
            # caches and make two runs hard to diff.
            found[column] = tuple(sorted(values))

    return found


def with_values(
    schema: TableSchema, values: Mapping[str, Sequence[str]]
) -> TableSchema:
    """A copy of the schema with discovered values folded into the descriptions.

    Discovery is best-effort — a column with nothing found is left exactly as it
    was, rather than rendered with an empty list, which would read as "this
    column is empty".
    """
    columns = []
    for column in schema.columns:
        found = values.get(column.name)
        if not found:
            columns.append(column)
            continue
        listed = ", ".join(f"'{value}'" for value in found)
        note = f"one of: {listed}"
        description = f"{column.description}. {note}" if column.description else note
        columns.append(replace(column, description=description))
    return replace(schema, columns=tuple(columns))
