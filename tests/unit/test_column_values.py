"""Telling the model which values a column actually holds.

Two live eval failures came from the model guessing a literal: it wrote
`WHERE gender = 'female'` when the column holds 'F', and
`WHERE status = 'cancelled'` when it holds 'Cancelled'. Both queries were valid,
passed the guard, ran without error and returned 0 — the worst failure shape
there is, because nothing anywhere reports a problem.

A schema that says `gender STRING` cannot prevent that. One that says
`gender STRING -- one of: 'F', 'M'` can.

The hard constraint is that this reads *data* into the prompt, which is exactly
what the PII policy exists to stop. A restricted column is never enumerated, by
policy rather than by cardinality — relying on "there are too many last names
to list" would be luck, not a rule.
"""

from __future__ import annotations

from retail_agent.datasources.base import ColumnSchema, TableSchema
from retail_agent.datasources.column_values import (
    MAX_DISTINCT,
    enumerable_columns,
    with_values,
)


def table(*columns: ColumnSchema) -> TableSchema:
    return TableSchema(name="users", columns=columns)


def string(name: str) -> ColumnSchema:
    return ColumnSchema(name=name, type="STRING")


def test_string_columns_are_candidates():
    schema = table(string("gender"), string("status"))

    assert enumerable_columns(schema, restricted=set()) == ("gender", "status")


def test_a_restricted_column_is_never_a_candidate():
    """The whole safety argument. `email` has 83,777 distinct values so the
    cardinality rule would exclude it anyway — but that is luck, and a column
    with three sensitive values would sail through."""
    schema = table(string("gender"), string("email"), string("last_name"))

    candidates = enumerable_columns(schema, restricted={"email", "last_name"})

    assert candidates == ("gender",)


def test_restriction_matching_ignores_case():
    schema = table(string("Email"))

    assert enumerable_columns(schema, restricted={"email"}) == ()


def test_non_string_columns_are_not_candidates():
    """Enumerating an id or a timestamp is noise at best. The failure this
    solves is a misspelled string literal."""
    schema = table(
        string("gender"),
        ColumnSchema(name="id", type="INTEGER"),
        ColumnSchema(name="created_at", type="TIMESTAMP"),
    )

    assert enumerable_columns(schema, restricted=set()) == ("gender",)


def test_values_are_rendered_into_the_schema_the_model_reads():
    schema = table(string("gender"))

    enriched = with_values(schema, {"gender": ("F", "M")})

    assert "'F', 'M'" in enriched.to_ddl()


def test_a_column_with_no_discovered_values_is_unchanged():
    """Discovery is best-effort: a failed or skipped lookup must leave the
    schema exactly as it was rather than render an empty list."""
    schema = table(string("gender"), string("city"))

    ddl = with_values(schema, {"gender": ("F", "M")}).to_ddl()

    assert "city STRING," in ddl
    assert "one of" in ddl.split("city")[0]


def test_an_existing_description_survives_alongside_the_values():
    schema = table(ColumnSchema(name="gender", type="STRING", description="shopper sex"))

    ddl = with_values(schema, {"gender": ("F", "M")}).to_ddl()

    assert "shopper sex" in ddl
    assert "'F'" in ddl


def test_the_cardinality_ceiling_sits_above_category_and_below_state():
    """Measured on theLook, where there is a clean gap: the useful columns top
    out at 26 distinct values (product category) and the next one up is 230
    (state). Listing 230 states would cost more prompt than it saves."""
    assert 26 <= MAX_DISTINCT < 230


# --- discovering the values ---


def test_the_discovery_query_asks_for_count_and_values_per_column():
    from retail_agent.datasources.column_values import build_discovery_query

    sql = build_discovery_query("users", ("gender", "country"), dataset="ds")

    assert "APPROX_COUNT_DISTINCT(`gender`)" in sql
    assert "APPROX_TOP_COUNT(`gender`, 31)" in sql
    assert "APPROX_COUNT_DISTINCT(`country`)" in sql
    assert "`ds.users`" in sql


def test_the_discovery_query_asks_for_one_more_than_the_ceiling():
    """So a column sitting exactly at the ceiling is distinguishable from one
    just over it, rather than silently truncated to look small."""
    from retail_agent.datasources.column_values import build_discovery_query

    assert f"APPROX_TOP_COUNT(`gender`, {MAX_DISTINCT + 1})" in build_discovery_query(
        "users", ("gender",), dataset="ds"
    )


def test_no_columns_means_no_query():
    """A table of nothing but ids and timestamps must not be queried at all."""
    from retail_agent.datasources.column_values import build_discovery_query

    assert build_discovery_query("users", (), dataset="ds") == ""


def test_a_low_cardinality_column_yields_its_values():
    from retail_agent.datasources.column_values import read_discovery_row

    row = {"gender__n": 2, "gender__v": [{"value": "F", "count": 9}, {"value": "M", "count": 8}]}

    assert read_discovery_row(row, ("gender",)) == {"gender": ("F", "M")}


def test_a_high_cardinality_column_is_dropped():
    """230 states is more prompt than the mistakes it would prevent."""
    from retail_agent.datasources.column_values import read_discovery_row

    row = {"state__n": 230, "state__v": [{"value": "Texas", "count": 1}]}

    assert read_discovery_row(row, ("state",)) == {}


def test_values_come_back_in_a_stable_order():
    """The schema goes into every prompt. Reordering it between runs would
    invalidate prompt caches and make two runs hard to diff."""
    from retail_agent.datasources.column_values import read_discovery_row

    row = {
        "s__n": 3,
        "s__v": [
            {"value": "Shipped", "count": 5},
            {"value": "Complete", "count": 9},
            {"value": "Cancelled", "count": 7},
        ],
    }

    assert read_discovery_row(row, ("s",)) == {"s": ("Cancelled", "Complete", "Shipped")}


def test_nulls_are_not_offered_as_a_literal():
    """`WHERE status = 'None'` matches nothing. A NULL is not a value the model
    should be told to compare against."""
    from retail_agent.datasources.column_values import read_discovery_row

    row = {"s__n": 2, "s__v": [{"value": "F", "count": 5}, {"value": None, "count": 2}]}

    assert read_discovery_row(row, ("s",)) == {"s": ("F",)}


def test_a_column_missing_from_the_row_is_skipped_not_an_error():
    from retail_agent.datasources.column_values import read_discovery_row

    assert read_discovery_row({}, ("gender",)) == {}
