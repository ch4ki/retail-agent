"""Static validation of model-authored SQL.

Parses to an AST rather than pattern-matching text. A regex guard can be
defeated by comments, casing, or unicode; a parser sees the same tree the
database will.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import sqlglot
from sqlglot import exp

DIALECT = "bigquery"

# Built defensively: node class names vary slightly across sqlglot releases.
_FORBIDDEN_NODE_NAMES = (
    "Insert",
    "Update",
    "Delete",
    "Drop",
    "Create",
    "Alter",
    "TruncateTable",
    "Merge",
    "Command",
    "Grant",
    "Use",
    "Set",
    "Copy",
    "Export",
)
FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = tuple(
    getattr(exp, name) for name in _FORBIDDEN_NODE_NAMES if hasattr(exp, name)
)

# Aggregates that collapse identity. COUNT(email) discloses nothing about any
# individual. MAX(first_name) grouped by user id returns that person's actual
# name, so it is not disclosure-safe despite being an aggregate.
_COUNTING_AGG_NAMES = ("Count", "ApproxDistinct", "CountIf")
COUNTING_AGGS: tuple[type[exp.Expression], ...] = tuple(
    getattr(exp, name) for name in _COUNTING_AGG_NAMES if hasattr(exp, name)
)


@dataclass(frozen=True)
class GuardResult:
    ok: bool
    sql: str
    violations: tuple[str, ...] = ()


def check_sql(
    sql: str,
    *,
    allowed_tables: Collection[str],
    restricted_columns: Collection[str],
    default_limit: int = 500,
    max_limit: int = 5_000,
    qualify_with: str | None = None,
) -> GuardResult:
    """Validate and normalise a query. Never raises on bad input.

    `qualify_with` is a dataset like "project.dataset". Allowed tables that
    arrive unqualified are rewritten to use it, because BigQuery rejects bare
    table names and that is a mechanical fix, not one worth a repair attempt.
    """
    if not sql or not sql.strip():
        return GuardResult(False, sql, ("Query is empty.",))

    try:
        statements = [s for s in sqlglot.parse(sql, dialect=DIALECT) if s is not None]
    except Exception as err:  # sqlglot raises several parse error types
        return GuardResult(False, sql, (f"Could not parse SQL: {err}",))

    if len(statements) != 1:
        return GuardResult(
            False, sql, (f"Expected exactly one statement, found {len(statements)}.",)
        )

    tree = statements[0]
    cte_names = {
        cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE) if cte.alias_or_name
    }

    violations: list[str] = []
    violations += _check_read_only(tree)
    violations += _check_tables(tree, allowed_tables, cte_names)
    violations += _check_projections(tree, restricted_columns)

    if violations:
        return GuardResult(False, sql, tuple(violations))

    if qualify_with:
        _qualify_tables(tree, qualify_with, cte_names)

    return GuardResult(True, _apply_limit(tree, default_limit, max_limit), ())


def _check_read_only(tree: exp.Expression) -> list[str]:
    found = {
        type(node).__name__ for node in tree.walk() if isinstance(node, FORBIDDEN_NODES)
    }
    if found:
        return [f"Only read queries are allowed; found {', '.join(sorted(found))}."]
    if not isinstance(tree, (exp.Select, exp.Union, exp.Subquery, exp.With)):
        return [f"Only SELECT queries are allowed; found {type(tree).__name__}."]
    return []


def _check_tables(
    tree: exp.Expression, allowed: Collection[str], cte_names: set[str]
) -> list[str]:
    allowed_lower = {t.lower() for t in allowed}

    violations = []
    for table in tree.find_all(exp.Table):
        name = (table.name or "").lower()
        if not name or name in cte_names or name in allowed_lower:
            continue
        violations.append(
            f"Table '{table.name}' is not available. "
            f"Allowed tables: {', '.join(sorted(allowed))}."
        )
    return violations


def _check_projections(tree: exp.Expression, restricted: Collection[str]) -> list[str]:
    restricted_lower = {c.lower() for c in restricted}
    violations: list[str] = []

    for select in tree.find_all(exp.Select):
        for projection in select.expressions:
            if isinstance(projection, exp.Star) or projection.find(exp.Star):
                violations.append(
                    "SELECT * is not allowed. List the columns you need explicitly."
                )
                continue

            aliased = isinstance(projection, exp.Alias)
            inner = projection.this if aliased else projection

            # A bare, unaliased column keeps its name in the result set, which
            # is what lets the masking policy find and mask it. Renaming it, or
            # burying it in an expression, defeats that.
            if not aliased and isinstance(inner, exp.Column):
                continue

            for column in inner.find_all(exp.Column):
                if column.name.lower() not in restricted_lower:
                    continue
                if _inside_counting_aggregate(column, inner):
                    continue
                violations.append(
                    f"Column '{column.name}' is personal data. Select it on its "
                    f"own with no alias (it is masked automatically), wrap it in "
                    f"COUNT(...), or use 'id' to identify a customer. It cannot "
                    f"appear inside another expression."
                )

    return _dedupe(violations)


def _inside_counting_aggregate(column: exp.Expression, root: exp.Expression) -> bool:
    """True when the nearest aggregate enclosing `column` only counts rows."""
    node = column.parent
    while node is not None:
        if isinstance(node, COUNTING_AGGS):
            return True
        if isinstance(node, exp.AggFunc):
            return False  # MAX / MIN / ANY_VALUE / STRING_AGG all return the value
        if node is root:
            return False
        node = node.parent
    return False


def _qualify_tables(
    tree: exp.Expression, dataset: str, cte_names: set[str]
) -> None:
    """Prefix bare table references with the configured project and dataset."""
    parts = dataset.split(".")
    catalog, db = (parts[0], parts[1]) if len(parts) == 2 else ("", parts[-1])

    for table in tree.find_all(exp.Table):
        name = (table.name or "").lower()
        if not name or name in cte_names or table.args.get("db"):
            continue
        table.set("db", exp.to_identifier(db))
        if catalog:
            table.set("catalog", exp.to_identifier(catalog))


def _apply_limit(tree: exp.Expression, default_limit: int, max_limit: int) -> str:
    existing = tree.args.get("limit")
    target = default_limit

    if existing is not None:
        try:
            current = int(existing.expression.this)
            target = min(current, max_limit)
        except (AttributeError, TypeError, ValueError):
            target = default_limit

    try:
        limited = tree.limit(target)
    except Exception:
        return tree.sql(dialect=DIALECT)
    return limited.sql(dialect=DIALECT)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
