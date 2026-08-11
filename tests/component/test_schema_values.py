"""The schema the model reads carries the values a column can hold.

Guessing a literal was a live failure twice — `gender = 'female'` against a
column holding 'F', `status = 'cancelled'` against one holding 'Cancelled'.
Both ran cleanly and returned 0, which no other layer reports as a problem.
"""

from __future__ import annotations

from retail_agent.agent.schema import render_schema_for_sql
from retail_agent.datasources.base import ColumnSchema, TableSchema


class ValueSource:
    """A warehouse that answers the discovery query and records being asked."""

    dialect = "bigquery"

    def __init__(self, values=None, fail=False):
        self.values = values or {"gender": [{"value": "F"}, {"value": "M"}]}
        self.fail = fail
        self.queries: list[str] = []

    def list_tables(self):
        return ["orders"]

    def describe(self, table):
        return TableSchema(
            name="orders",
            columns=(
                ColumnSchema(name="id", type="INTEGER"),
                ColumnSchema(name="gender", type="STRING"),
                ColumnSchema(name="status", type="STRING"),
                ColumnSchema(name="email", type="STRING"),
            ),
        )

    def describe_all(self):
        return [self.describe("users")]

    def column_values(self, table, columns):
        from retail_agent.datasources.bigquery import (
            build_discovery_query,
            read_discovery_row,
        )

        self.queries.append(build_discovery_query(table, columns, dataset="ds"))
        if self.fail:
            raise RuntimeError("discovery is not permitted")
        return read_discovery_row(self.values, columns)

    def execute(self, sql):
        raise AssertionError("value discovery must not go through execute")

    def dry_run(self, sql):
        raise NotImplementedError

    def assert_within_budget(self, sql):
        raise NotImplementedError


def test_the_rendered_schema_lists_the_values(make_deps):
    deps = make_deps([], src=ValueSource())

    assert "'F', 'M'" in render_schema_for_sql(deps)


def test_a_pii_column_is_never_enumerated(make_deps):
    """The safety constraint. This reads data into the prompt, which is what
    the PII policy exists to stop, so `email` must not even be asked about."""
    source = ValueSource()
    deps = make_deps([], src=source)

    render_schema_for_sql(deps)

    assert "email" not in source.queries[0]
    assert "`gender`" in source.queries[0]


def test_discovery_failure_degrades_to_a_plain_schema(make_deps):
    """A warehouse that refuses the query must cost the hint, not the turn."""
    deps = make_deps([], src=ValueSource(fail=True))

    rendered = render_schema_for_sql(deps)

    assert "gender STRING" in rendered
    # "one of:" with the colon is what the value renderer emits; a business
    # note may legitimately contain the words "one of".
    assert "one of:" not in rendered


def test_the_business_convention_reaches_the_sql_schema(make_deps):
    """Values alone made the agent filter to status = 'Complete'. The note
    saying what "completed" means has to travel with them."""
    deps = make_deps([], src=ValueSource())

    rendered = render_schema_for_sql(deps)

    assert "NOT IN ('Cancelled', 'Returned')" in rendered


def test_a_restricted_column_is_never_annotated(make_deps, monkeypatch):
    """Same rule as the values: nothing is said about a PII column, so a note
    added carelessly cannot describe one."""
    from retail_agent.knowledge import conventions

    monkeypatch.setattr(
        conventions, "COLUMN_NOTES", {("users", "email"): "primary contact"}
    )
    deps = make_deps([], src=ValueSource())

    assert "primary contact" not in render_schema_for_sql(deps)


def test_the_structural_schema_carries_no_conventions(make_deps):
    """`describe_schema` answers "what data do you have" from table shape.
    Business conventions belong to the query writer, and that path pays
    nothing."""
    from retail_agent.agent.capture import TurnCapture
    from retail_agent.agent.schema import build_schema_tool

    describe = build_schema_tool(make_deps(src=ValueSource()), TurnCapture())[0].func

    assert "NOT IN" not in describe()
