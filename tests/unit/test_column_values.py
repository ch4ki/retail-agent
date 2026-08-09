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
from retail_agent.datasources.column_values import enumerable_columns, with_values


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


# Reading the values out of the warehouse takes BigQuery SQL and is tested
# beside the adapter that writes it, in `test_bigquery_source.py`.


# --- business conventions, beside the values they qualify ---


def test_a_note_is_rendered_with_the_values():
    """Live: the agent wrote `WHERE status = 'Complete'` — one of five statuses
    — where the convention is NOT IN ('Cancelled','Returned'), undercounting
    93,893 orders as 31,303. Seeing the values made 'Complete' look like the
    answer; the convention has to sit next to them."""
    from retail_agent.datasources.column_values import with_values

    schema = table(string("status"))

    ddl = with_values(
        schema, {"status": ("Cancelled", "Complete")}, notes={"status": "Completed means not Cancelled and not Returned"}
    ).to_ddl()

    assert "'Cancelled', 'Complete'" in ddl
    assert "not Cancelled and not Returned" in ddl


def test_a_note_is_rendered_even_without_discovered_values():
    """A convention on a numeric or high-cardinality column still matters."""
    from retail_agent.datasources.column_values import with_values

    ddl = with_values(table(string("state")), {}, notes={"state": "two-letter code"}).to_ddl()

    assert "two-letter code" in ddl


def test_a_column_with_no_note_is_unchanged():
    from retail_agent.datasources.column_values import with_values

    ddl = with_values(table(string("gender")), {"gender": ("F", "M")}, notes={}).to_ddl()

    assert "'F', 'M'" in ddl
    assert ddl.count("--") == 1


def test_notes_are_optional():
    """The existing call sites pass no notes and must keep working."""
    from retail_agent.datasources.column_values import with_values

    assert "'F', 'M'" in with_values(table(string("gender")), {"gender": ("F", "M")}).to_ddl()


def test_theLook_defines_completed_for_both_status_columns():
    """orders and order_items each carry a status, and the same convention
    governs both — the agent joins them freely."""
    from retail_agent.knowledge.conventions import notes_for

    assert "Cancelled" in notes_for("orders")["status"]
    assert "Cancelled" in notes_for("order_items")["status"]


def test_a_table_with_no_conventions_yields_none():
    from retail_agent.knowledge.conventions import notes_for

    assert notes_for("distribution_centers") == {}
