"""Second look at a query that ran fine and returned nothing.

An empty result is not an error. `WHERE brand = 'Levis'` against a column
holding `Levi's` is valid SQL, costs nothing, raises nothing, and returns zero
rows — and the agent will faithfully report "no revenue for that brand", which
is a plausible-sounding lie with a real business consequence.

So emptiness gets one dedicated retry, on its own budget. It cannot come out of
the repair budget: that exists for SQL that is actually broken, and sometimes
"no orders matched" really is the answer.
"""

from __future__ import annotations

from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.state import SqlAttempt, TurnState

EMPTY_RESULT_DIAGNOSIS = (
    "This query ran successfully but returned no data. That is usually a filter "
    "matching nothing rather than a broken query. Rewrite it to match more "
    "loosely, in this order of suspicion:\n"
    "1. String literals the user typed. Punctuation inside the stored value "
    "breaks a whole-word match: the brand \"Levi's\" does NOT match "
    "LIKE '%levis%', because of the apostrophe. Match on a distinctive "
    "FRAGMENT instead — LOWER(col) LIKE '%levi%' — and keep the fragment short "
    "enough to survive punctuation, plurals and spacing.\n"
    "2. Date ranges. Check the period contains data at all before narrowing.\n"
    "3. Status filters, which may be excluding every row.\n"
    "Change exactly one thing and make it the most likely one. If you believe "
    "no data really is the answer, return the same query unchanged."
)


def diagnose_node(state: TurnState, deps: AgentDeps) -> dict:
    """Rewind one step and hand the emptiness back as the problem to solve.

    Recorded as a failed `SqlAttempt` so `draft_sql` picks its repair prompt
    without needing to know that diagnosis exists — the two paths differ in
    which budget pays, not in machinery.
    """
    attempts = list(state.get("sql_attempts", []))
    if not attempts:
        return {}

    last = attempts[-1]
    attempts.append(
        SqlAttempt(
            step_id=last.step_id,
            sql=last.sql,
            executed_sql=last.executed_sql,
            error=EMPTY_RESULT_DIAGNOSIS,
        )
    )

    # `execute` advanced past this step and stored an empty frame for it. Both
    # have to come back, or the redraft answers the following step and the
    # empty frame stays in the synthesis prompt.
    frames = {k: v for k, v in state.get("frames", {}).items() if k != last.step_id}

    return {
        "sql_attempts": attempts,
        "frames": frames,
        "step_index": max(0, state.get("step_index", 1) - 1),
        "diagnose_budget": state.get("diagnose_budget", 0) - 1,
    }
