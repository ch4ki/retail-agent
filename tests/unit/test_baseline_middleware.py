"""The middleware stack, and the parity it is supposed to hold.

These are configuration tests, and they earn their place because the
configuration is the comparison. A ReAct arm allowed twice the SQL calls, or one
whose PII redaction never looks at tool results, produces a number that says
something other than what the report will claim it says.
"""

from __future__ import annotations

from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    PIIMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
)

from retail_agent.baseline.react import build_middleware
from retail_agent.config import Settings


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, google_cloud_project="test", **overrides)


def only(stack, kind):
    found = [m for m in stack if isinstance(m, kind)]
    assert found, f"no {kind.__name__} in the stack"
    return found


def test_the_sql_budget_tracks_the_graphs_ceiling():
    """The graph can spend `max_analysis_steps` executions plus `repair_budget`
    redrafts plus one diagnosis. Hard-coding 14 here would let the two arms
    drift apart the first time someone tunes a budget in config."""
    stack = build_middleware(settings(repair_budget=3, max_analysis_steps=10))

    limit = only(stack, ToolCallLimitMiddleware)[0]

    assert limit.tool_name == "run_sql"
    assert limit.run_limit == 14


def test_a_tuned_repair_budget_moves_the_react_ceiling_with_it():
    stack = build_middleware(settings(repair_budget=5, max_analysis_steps=4))

    assert only(stack, ToolCallLimitMiddleware)[0].run_limit == 10


def test_redaction_looks_at_tool_results_not_just_the_conversation():
    """PII arrives from BigQuery inside a `ToolMessage`. Middleware that only
    inspected user input and final output would never see it — which is the
    single easiest way to build a baseline that leaks and still looks clean."""
    stack = build_middleware(settings())

    for rule in only(stack, PIIMiddleware):
        assert rule.apply_to_tool_results is True


def test_email_is_redacted_because_the_gate_fails_the_run_over_it():
    stack = build_middleware(settings())

    assert "email" in {rule.pii_type for rule in only(stack, PIIMiddleware)}


def test_the_model_cannot_loop_forever():
    stack = build_middleware(settings())

    assert only(stack, ModelCallLimitMiddleware)[0].exit_behavior == "end"


def test_tool_errors_are_returned_to_the_model_rather_than_raised():
    """This is the ReAct arm's repair path. Without it a rejected query kills
    the turn instead of giving the model the violation to fix."""
    assert only(build_middleware(settings()), ToolErrorMiddleware)


# --- the repair path ---


def failure(error):
    """Call `_describe_failure` the way `ToolErrorMiddleware` does.

    Its `OnError` is `Callable[[Exception, ToolCallRequest], ...]` — two
    arguments. Calling it with one passed every unit test and then raised
    TypeError against the live provider, which scored as an agent failure
    rather than as the repair it was supposed to be.
    """
    from retail_agent.baseline.react import _describe_failure

    return _describe_failure(error, object())


def test_a_guard_rejection_is_handed_back_to_the_model_to_fix():
    from retail_agent.baseline.tools import GuardRejection

    message = failure(GuardRejection("Only SELECT is allowed."))

    assert message is not None
    assert "Only SELECT is allowed." in message


def test_a_warehouse_error_is_handed_back_too():
    from retail_agent.datasources.base import QuerySyntaxError

    message = failure(QuerySyntaxError("Unrecognized name: sale_pric"))

    assert message is not None
    assert "sale_pric" in message


def test_an_unexpected_error_propagates_rather_than_being_explained_away():
    """Returning None lets it raise. An internal bug dressed up as a repair
    prompt is how the agent silently works around its own broken code."""
    assert failure(RuntimeError("something internal")) is None
