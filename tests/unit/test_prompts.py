"""The SQL rules are load-bearing, so the ones that produced live failures are
pinned here. Assertions run against whitespace-normalised text; the prompts are
wrapped for readability and rewrapping them should not fail a test."""

from retail_agent.agent.prompts import PLANNER_PROMPT, REPAIR_PROMPT, SQL_PROMPT


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_planner_prompt_describes_no_bespoke_reply_format():
    """The reply shape is the `Plan` schema now. A prompt that also describes a
    text format reintroduces the two-artifacts problem the schema removed —
    they can disagree, and the disagreement is silent."""
    rules = _flat(PLANNER_PROMPT)

    assert "STEP:" not in rules
    assert "Reply with one line per step" not in rules


def test_planner_prompt_rejects_non_retrieval_steps():
    """A live planner emitted "Compare the results of the current period to
    April" as a step. Steps become SQL, so a comparison step spends an attempt
    on something synthesis does for free."""
    rules = _flat(PLANNER_PROMPT)

    assert "retrieval" in rules
    assert "comparing, ranking and explaining happen after" in rules


def test_planner_prompt_does_not_ask_for_sql():
    """draft_sql writes the SQL. A planner told to emit SQL returns a query,
    which contains no step markers at all."""
    rules = _flat(PLANNER_PROMPT)

    assert "SQL:" not in rules
    assert "SQL generator" not in rules


def test_clamp_rule_bounds_timestamps_with_a_timestamp():
    """`created_at <= CURRENT_DATE()` is a BigQuery type error: the theLook
    timestamp columns are TIMESTAMP and there is no implicit coercion."""
    rules = _flat(SQL_PROMPT)

    assert "CURRENT_TIMESTAMP()" in rules
    assert "Never compare a TIMESTAMP column to CURRENT_DATE()" in rules


def test_clamp_rule_excludes_named_complete_periods():
    """"in March" is not a to-date question. Clamping it anyway produced the
    failure that motivated this rule."""
    assert "complete period" in _flat(SQL_PROMPT)


def test_repair_prompt_carries_the_sql_rules():
    """A repair that has never seen the dataset or the type rule reintroduces
    the bare table names and the DATE bound the first draft was told to avoid."""
    rules = _flat(REPAIR_PROMPT)

    assert "{dataset}" in rules
    assert "CURRENT_TIMESTAMP()" in rules
