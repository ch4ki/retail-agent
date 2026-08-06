import pytest

from retail_agent.safety.sql_guard import check_sql

TABLES = {"orders", "order_items", "products", "users"}
PII = {"email", "first_name", "last_name", "street_address", "latitude", "longitude"}


def guard(sql: str, **kwargs):
    return check_sql(
        sql,
        allowed_tables=TABLES,
        restricted_columns=PII,
        default_limit=500,
        max_limit=5000,
        **kwargs,
    )


# --- queries that must pass ---


def test_simple_select_passes():
    assert guard("SELECT id, age FROM users LIMIT 10").ok


def test_aggregate_over_pii_column_is_allowed():
    result = guard("SELECT COUNT(DISTINCT email) AS c FROM users")
    assert result.ok, result.violations


def test_cte_name_is_not_treated_as_a_table():
    sql = """
        WITH recent AS (SELECT user_id, sale_price FROM order_items)
        SELECT user_id, SUM(sale_price) AS total FROM recent GROUP BY user_id
    """
    assert guard(sql).ok


def test_pii_column_in_where_clause_is_allowed():
    assert guard("SELECT id FROM users WHERE email IS NOT NULL").ok


def test_comment_containing_drop_is_harmless():
    assert guard("SELECT id FROM users -- ; DROP TABLE users").ok


def test_fully_qualified_backticked_table_passes():
    # This is the form the SQL prompt instructs the model to produce. If it
    # were rejected, every generated query would fail the guard.
    result = guard(
        "SELECT id FROM `bigquery-public-data.thelook_ecommerce.users` LIMIT 5"
    )
    assert result.ok, result.violations


def test_fully_qualified_unquoted_table_passes():
    assert guard(
        "SELECT id FROM bigquery-public-data.thelook_ecommerce.users LIMIT 5"
    ).ok


def test_realistic_join_survives_the_rewrite():
    sql = (
        "SELECT u.id, SUM(oi.sale_price) AS spend "
        "FROM `bigquery-public-data.thelook_ecommerce.order_items` oi "
        "JOIN `bigquery-public-data.thelook_ecommerce.users` u ON u.id = oi.user_id "
        "WHERE oi.status NOT IN ('Cancelled', 'Returned') "
        "GROUP BY u.id ORDER BY spend DESC LIMIT 10"
    )
    result = guard(sql)

    assert result.ok, result.violations
    assert "`bigquery-public-data.thelook_ecommerce.users`" in result.sql
    assert "ORDER BY spend DESC" in result.sql
    assert "LIMIT 10" in result.sql


def test_constant_select_with_no_table_passes():
    assert guard("SELECT 1 AS x").ok


# --- queries that must be rejected ---


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM users",
        "UPDATE users SET email = 'x'",
        "INSERT INTO users (id) VALUES (1)",
        "DROP TABLE users",
        "CREATE TABLE evil AS SELECT 1",
        "TRUNCATE TABLE users",
    ],
)
def test_write_statements_are_rejected(sql):
    assert not guard(sql).ok


def test_stacked_statements_are_rejected():
    result = guard("SELECT id FROM users; DROP TABLE users")
    assert not result.ok
    assert any("one statement" in v for v in result.violations)


# A restricted column may be projected only in a form the masking layer can
# still find by output column name — i.e. bare — or inside an aggregate that
# genuinely collapses identity. Anything else is rejected.


def test_bare_pii_column_is_allowed_because_masking_catches_it():
    # Output column stays `email`, so the policy hashes it before the model sees it.
    assert guard("SELECT id, email FROM users").ok


def test_bare_qualified_pii_column_is_allowed():
    assert guard("SELECT u.last_name FROM users AS u").ok


def test_aliased_pii_projection_is_rejected():
    # An alias renames the output column, which defeats name-based masking.
    assert not guard("SELECT email AS contact FROM users").ok


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(email) AS c FROM users",
        "SELECT COUNT(DISTINCT email) AS c FROM users",
        "SELECT APPROX_COUNT_DISTINCT(email) AS c FROM users",
    ],
)
def test_counting_aggregates_over_pii_are_allowed(sql):
    assert guard(sql).ok, guard(sql).violations


@pytest.mark.parametrize(
    ("label", "sql"),
    [
        ("max leaks the value", "SELECT MAX(first_name) AS n FROM users"),
        ("min leaks the value", "SELECT MIN(email) AS e FROM users"),
        ("any_value leaks", "SELECT ANY_VALUE(last_name) AS n FROM users"),
        ("string_agg leaks", "SELECT STRING_AGG(email) AS e FROM users"),
        ("array_agg leaks", "SELECT ARRAY_AGG(last_name) AS n FROM users"),
    ],
)
def test_non_counting_aggregates_over_pii_are_rejected(label, sql):
    assert not guard(sql).ok, label


def test_concat_of_max_names_is_rejected():
    # The exact evasion a live model produced after the guard blocked the
    # bare-but-aliased form: MAX() over a GROUP BY returns the real name.
    sql = (
        "SELECT u.id AS user_id, "
        "CONCAT(MAX(u.first_name), ' ', MAX(u.last_name)) AS user_name, "
        "SUM(oi.sale_price) AS total_spend "
        "FROM order_items AS oi JOIN users AS u ON oi.user_id = u.id "
        "GROUP BY user_id ORDER BY total_spend DESC LIMIT 10"
    )
    result = guard(sql)

    assert not result.ok
    assert any("first_name" in v for v in result.violations)


def test_violation_message_does_not_suggest_a_plain_aggregate():
    # The original message said "use an aggregate", which is what taught a live
    # model to reach for MAX(). The advice must not re-open the hole.
    violations = " ".join(guard("SELECT email AS contact FROM users").violations)

    assert "COUNT" in violations
    assert "use an aggregate" not in violations.lower()


def test_select_star_is_rejected():
    result = guard("SELECT * FROM users")
    assert not result.ok
    assert any("*" in v for v in result.violations)


def test_unknown_table_is_rejected():
    result = guard("SELECT id FROM billing_secrets")
    assert not result.ok
    assert any("billing_secrets" in v for v in result.violations)


def test_unparseable_sql_is_rejected_not_raised():
    result = guard("SELCT id FRM users")
    assert not result.ok
    assert result.violations


def test_empty_sql_is_rejected():
    assert not guard("   ").ok


@pytest.mark.parametrize(
    ("label", "sql"),
    [
        ("scalar subquery", "SELECT (SELECT email FROM users LIMIT 1) AS x FROM orders"),
        ("subquery in FROM", "SELECT e FROM (SELECT email AS e FROM users)"),
        ("concat is not an aggregate", "SELECT CONCAT(first_name, last_name) AS n FROM users"),
        ("qualified star", "SELECT u.* FROM users u"),
    ],
)
def test_pii_cannot_be_smuggled_through_nesting(label, sql):
    assert not guard(sql).ok, label


@pytest.mark.parametrize(
    ("label", "sql"),
    [
        ("union to a hidden table", "SELECT id FROM users UNION ALL SELECT id FROM secrets"),
        ("delete after a cte", "WITH x AS (SELECT id FROM users) DELETE FROM users"),
        ("dynamic sql", "EXECUTE IMMEDIATE 'SELECT 1'"),
        ("data exfiltration", "EXPORT DATA OPTIONS(uri='gs://x') AS SELECT id FROM users"),
        ("view creation", "CREATE OR REPLACE VIEW v AS SELECT id FROM users"),
    ],
)
def test_evasion_attempts_are_rejected(label, sql):
    assert not guard(sql).ok, label


# --- LIMIT enforcement ---


def test_missing_limit_is_injected():
    result = guard("SELECT id FROM users")
    assert result.ok
    assert "LIMIT 500" in result.sql.upper()


def test_oversized_limit_is_capped():
    result = guard("SELECT id FROM users LIMIT 100000")
    assert result.ok
    assert "LIMIT 5000" in result.sql.upper()


def test_reasonable_limit_is_preserved():
    result = guard("SELECT id FROM users LIMIT 10")
    assert result.ok
    assert "LIMIT 10" in result.sql.upper()


# The prompt asks the model to fully qualify tables. Models forget, and
# BigQuery then rejects the query. Qualification is mechanical, so the guard
# does it rather than spending repair budget on it.

DATASET = "bigquery-public-data.thelook_ecommerce"


def qualified(sql: str):
    return check_sql(
        sql,
        allowed_tables=TABLES,
        restricted_columns=PII,
        default_limit=500,
        max_limit=5000,
        qualify_with=DATASET,
    )


def test_bare_table_is_qualified():
    # sqlglot backticks only the hyphenated project part, which BigQuery
    # accepts, so assert on the parts rather than one exact rendering.
    result = qualified("SELECT id FROM users LIMIT 5")

    assert result.ok, result.violations
    assert "bigquery-public-data" in result.sql
    assert "thelook_ecommerce.users" in result.sql


def test_the_exact_query_that_failed_against_bigquery():
    sql = (
        "SELECT p.name, COUNT(oi.id) AS sales_count "
        "FROM order_items AS oi JOIN products AS p ON oi.product_id = p.id "
        "WHERE EXTRACT(MONTH FROM oi.created_at) = 3 "
        "GROUP BY p.name ORDER BY sales_count DESC LIMIT 1"
    )
    result = qualified(sql)

    assert result.ok, result.violations
    assert "thelook_ecommerce.order_items" in result.sql
    assert "thelook_ecommerce.products" in result.sql
    assert "LIMIT 1" in result.sql


def test_already_qualified_tables_are_untouched():
    result = qualified(
        "SELECT id FROM `bigquery-public-data.thelook_ecommerce.users` LIMIT 5"
    )
    assert result.ok
    assert result.sql.count("thelook_ecommerce") == 1


def test_cte_names_are_never_qualified():
    sql = (
        "WITH recent AS (SELECT user_id FROM order_items) "
        "SELECT user_id FROM recent LIMIT 10"
    )
    result = qualified(sql)

    assert result.ok, result.violations
    assert "thelook_ecommerce.recent" not in result.sql
    assert "thelook_ecommerce.order_items" in result.sql


def test_qualification_is_optional():
    result = check_sql(
        "SELECT id FROM users",
        allowed_tables=TABLES,
        restricted_columns=PII,
    )
    assert result.ok
    assert "thelook_ecommerce" not in result.sql


# COUNT(*) is not SELECT *. Found by the seed corpus: a hand-written analyst
# query counting churned customers was rejected by the guard.


def test_count_star_is_allowed():
    """`COUNT(*)` discloses a row count and nothing about any individual. It is
    also the single most common aggregate there is — rejecting it means every
    "how many" question burns the repair budget and then degrades."""
    result = guard("SELECT COUNT(*) AS n FROM users")

    assert result.ok, result.violations


def test_count_star_inside_a_cte_is_allowed():
    sql = """
        WITH recent AS (SELECT user_id FROM orders)
        SELECT COUNT(*) AS n FROM recent
    """
    assert guard(sql).ok


def test_bare_select_star_is_still_rejected():
    assert not guard("SELECT * FROM users").ok


def test_qualified_star_is_still_rejected():
    assert not guard("SELECT u.* FROM users AS u").ok


def test_star_inside_a_value_returning_aggregate_is_still_rejected():
    """ARRAY_AGG(*) would hand back whole rows, PII included."""
    assert not guard("SELECT ARRAY_AGG(t) AS rows FROM (SELECT * FROM users) AS t").ok


# Bind parameters. Found in a live session: asked "how many loyal customers do
# we have?", the model had no agreed threshold, reached for `@threshold` as a
# placeholder, and BigQuery rejected it three times — burning the whole repair
# budget on the same 400 because nothing told it to inline a value.


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT user_id FROM orders GROUP BY user_id HAVING COUNT(id) > @threshold",
        "SELECT id FROM users WHERE age > ?",
        "SELECT id FROM users WHERE age > :min_age",
    ],
)
def test_bind_parameters_are_rejected(sql):
    """Nothing binds them, so the query cannot run. Catching it here costs a
    guard rejection; letting it through costs a billed round trip."""
    result = guard(sql)

    assert not result.ok
    assert any("parameter" in v.lower() for v in result.violations)


def test_the_rejection_says_what_to_do_instead():
    """The repair prompt shows the model this text. "Invalid query" would have
    it try the same thing again."""
    result = guard("SELECT id FROM users WHERE age > @min_age")

    assert any("literal" in v.lower() for v in result.violations)


def test_an_email_address_in_a_string_is_not_a_parameter():
    """`@` is common in ordinary values; only a bound parameter is a problem."""
    assert guard("SELECT id FROM users WHERE email = 'ada@example.com'").ok


# TIMESTAMP_SUB with a calendar part. Found live: asked "how many customers are
# at risk", the model wrote `TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 4
# MONTH)`, BigQuery rejected it, and the model wrote the identical query twice
# more — the raw 400 was not enough to tell it what to do instead.


@pytest.mark.parametrize("part", ["MONTH", "QUARTER", "YEAR"])
def test_timestamp_arithmetic_with_a_calendar_part_is_rejected(part):
    """BigQuery allows only MICROSECOND..DAY on TIMESTAMP_SUB/ADD. MONTH and
    friends need DATE_SUB on a DATE."""
    result = guard(
        f"SELECT id FROM users "
        f"WHERE created_at > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 {part})"
    )

    assert not result.ok
    assert any("DAY" in v for v in result.violations), result.violations


@pytest.mark.parametrize("part", ["DAY", "HOUR", "MINUTE", "SECOND"])
def test_timestamp_arithmetic_with_a_time_part_is_allowed(part):
    sql = (
        f"SELECT id FROM users "
        f"WHERE created_at > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 {part})"
    )
    assert guard(sql).ok, guard(sql).violations


def test_date_sub_with_a_month_is_fine():
    """The rule is about TIMESTAMP arithmetic, not about months."""
    assert guard(
        "SELECT id FROM users WHERE created_at > DATE_SUB(CURRENT_DATE(), INTERVAL 3 MONTH)"
    ).ok
