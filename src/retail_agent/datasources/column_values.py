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

What lives here is the part every warehouse shares: which columns may be read,
and how the answer is rendered. Actually fetching the values takes
dialect-specific SQL, so it belongs to the adapter — see `bigquery.py`.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import replace

from retail_agent.datasources.base import TableSchema

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


def with_values(
    schema: TableSchema,
    values: Mapping[str, Sequence[str]],
    notes: Mapping[str, str] | None = None,
) -> TableSchema:
    """A copy of the schema with values and conventions folded into the
    descriptions.

    Discovery is best-effort — a column with nothing found is left exactly as it
    was, rather than rendered with an empty list, which would read as "this
    column is empty".

    `notes` says what the values *mean*, which the values themselves cannot.
    Seeing that 'Complete' is a valid status made the model filter to it alone,
    where a completed order is everything except Cancelled and Returned; the
    note sits beside the list to say so. A note applies whether or not the
    column had values discovered.
    """
    notes = notes or {}
    columns = []
    for column in schema.columns:
        parts = []
        found = values.get(column.name)
        if found:
            parts.append("one of: " + ", ".join(f"'{value}'" for value in found))
        note = notes.get(column.name)
        if note:
            parts.append(note)
        if not parts:
            columns.append(column)
            continue
        described = ". ".join([column.description, *parts] if column.description else parts)
        columns.append(replace(column, description=described))
    return replace(schema, columns=tuple(columns))
