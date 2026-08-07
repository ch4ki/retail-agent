"""Reading a preference signal out of the router's decision.

Detection moved from regex to the model because regex caught about a quarter of
realistic phrasings and fired *backwards* on negation: "don't just give me the
number, tell me why" is a request for more depth, and it recorded
`depth=summary`, then quoted the user out of context in the proposal.

The model understands the negation. What it cannot be trusted with is the quote,
so the evidence is checked against the question rather than believed. The model
can invent a preference; it cannot invent a span that is in the user's own
message.
"""

from __future__ import annotations

import pytest

from retail_agent.agent.nodes.route import RouteDecision, style_signal


def decision(**kwargs) -> RouteDecision:
    return RouteDecision(**{"intent": "analyze", **kwargs})


def test_a_signal_with_a_real_quote_is_kept():
    signal = style_signal(
        decision(style_field="depth", style_value="summary", style_evidence="cut to the chase"),
        question="cut to the chase, how many brands?",
    )

    assert signal is not None
    assert (signal.field, signal.value) == ("depth", "summary")
    assert signal.evidence == "cut to the chase"


def test_an_invented_quote_is_dropped():
    """The guarantee. A model that hallucinates a preference would otherwise
    have it quoted back at the user as something they said."""
    signal = style_signal(
        decision(style_field="depth", style_value="summary", style_evidence="keep it short"),
        question="how many brands do we carry?",
    )

    assert signal is None


def test_the_quote_check_ignores_case():
    """Models routinely normalise capitalisation when copying a span."""
    signal = style_signal(
        decision(style_field="depth", style_value="deep", style_evidence="walk me through it"),
        question="Walk me through it — why did churn spike?",
    )

    assert signal is not None


def test_no_preference_means_no_signal():
    """The common case: most questions express nothing about presentation."""
    assert style_signal(decision(), question="how many brands?") is None


def test_a_missing_quote_is_not_a_signal():
    """Without evidence the proposal has nothing to show the user, and an
    unquotable suggestion is exactly what this design refuses to make."""
    signal = style_signal(
        decision(style_field="depth", style_value="summary", style_evidence=""),
        question="cut to the chase",
    )

    assert signal is None


def test_an_unknown_value_is_dropped():
    """`style_value` is free text in the schema — it has to be, since the legal
    values differ per field — so it is validated here instead."""
    signal = style_signal(
        decision(style_field="depth", style_value="terse", style_evidence="terse please"),
        question="terse please, how many brands?",
    )

    assert signal is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("depth", "summary"),
        ("depth", "deep"),
        ("answer_format", "table"),
        ("answer_format", "bullets"),
        ("answer_format", "prose"),
    ],
)
def test_every_legal_pairing_is_accepted(field, value):
    signal = style_signal(
        decision(style_field=field, style_value=value, style_evidence="as such"),
        question="as such, show me the data",
    )

    assert signal is not None


def test_a_value_from_the_wrong_field_is_dropped():
    """`depth=table` is not a preference anyone can hold."""
    signal = style_signal(
        decision(style_field="depth", style_value="table", style_evidence="as a table"),
        question="as a table please",
    )

    assert signal is None
