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
    violations += _check_no_parameters(tree)
    violations += _check_timestamp_intervals(tree)
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


def _check_no_parameters(tree: exp.Expression) -> list[str]:
    """Reject `@name`, `?` and `:name`.

    Nothing binds them, so the query cannot run — BigQuery returns a 400. The
    model reaches for them when it has no agreed value to use, which is exactly
    the case where an undefined business term is in play. Catching it here
    costs a guard rejection with instructions; letting it through costs a
    billed round trip and, three times over, the whole repair budget.
    """
    kinds = (exp.Parameter, exp.Placeholder, exp.SessionParameter)
    if any(isinstance(node, kinds) for node in tree.walk()):
        return [
            "Query parameters are not supported — nothing binds them. Choose a "
            "concrete value and write it as a literal (for example `>= 3` "
            "rather than `>= @threshold`). If the value is a judgement call, "
            "pick a defensible one; it will be stated in the answer."
        ]
    return []


# BigQuery accepts only MICROSECOND..DAY on TIMESTAMP_SUB / TIMESTAMP_ADD.
# MONTH, QUARTER and YEAR are a 400, and the raw error does not say what to do
# instead — observed live, the same rejected query written three times.
_CALENDAR_PARTS = frozenset({"MONTH", "QUARTER", "YEAR", "WEEK", "ISOYEAR"})
_TIMESTAMP_MATH = ("TimestampSub", "TimestampAdd")
TIMESTAMP_MATH_NODES: tuple[type[exp.Expression], ...] = tuple(
    getattr(exp, name) for name in _TIMESTAMP_MATH if hasattr(exp, name)
)


def _check_timestamp_intervals(tree: exp.Expression) -> list[str]:
    """Catch a calendar interval on TIMESTAMP arithmetic before BigQuery does."""
    if not TIMESTAMP_MATH_NODES:
        return []

    for node in tree.walk():
        if not isinstance(node, TIMESTAMP_MATH_NODES):
            continue
        # `unit` is a Var node, not a string, so read its name rather than
        # coercing the node.
        unit = node.args.get("unit")
        name = (getattr(unit, "name", "") or str(unit or "")).upper()
        if name in _CALENDAR_PARTS:
            return [
                "TIMESTAMP_SUB and TIMESTAMP_ADD accept only MICROSECOND "
                "through DAY. Express the period in DAY (90 DAY rather than 3 "
                "MONTH), or use DATE_SUB on a DATE column."
            ]
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
            aliased = isinstance(projection, exp.Alias)
            inner = projection.this if aliased else projection

            # `COUNT(*)` is not `SELECT *`. It returns a row count and discloses
            # nothing about any individual, and it is the most common aggregate
            # there is — rejecting it makes every "how many" question fail the
            # guard, spend the repair budget, and degrade.
            star = projection.find(exp.Star)
            if isinstance(projection, exp.Star) or (
                star is not None and not _inside_counting_aggregate(star, inner)
            ):
                violations.append(
                    "SELECT * is not allowed. List the columns you need explicitly."
                )
                continue

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


def _parse(sql: str) -> exp.Expression | None:
    """Parse, or None. Never raises.

    Both readers below run on queries the guard has already ruled on, so a
    parse failure here is not a safety decision — it must cost the enrichment,
    never the turn.
    """
    try:
        return sqlglot.parse_one(sql, read="bigquery")
    except Exception:
        return None


