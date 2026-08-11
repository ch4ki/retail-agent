"""What a turn records, and whether a checkpoint can hold it.

Everything here is a plain primitive on purpose. LangGraph round-trips custom
dataclasses today but warns that deserializing unregistered types "will be
blocked in a future version", and state is checkpointed on every super-step, so
a dataclass in state is both a deprecation and a size problem.
"""

import operator
import time
from typing import Annotated, get_args, get_origin, get_type_hints


def test_the_accumulating_fields_have_adding_reducers():
    """Without a reducer, the second tool call in a turn replaces the first
    rather than adding to it, and a turn silently reports one query."""
    from retail_agent.agent.state import TurnState

    hints = get_type_hints(TurnState, include_extras=True)

    for field in (
        "attempts",
        "events",
        "trio_ids",
        "assumed_terms",
        "preference_changes",
        "reports_written",
        "redactions",
        "calls",
    ):
        annotation = hints[field]
        assert get_origin(annotation) is Annotated, f"{field} has no reducer"
        assert get_args(annotation)[1] is operator.add, f"{field}'s reducer is not add"


def test_a_step_event_is_a_dict_of_primitives():
    """A tuple would come back from the checkpointer as a list, so consumers
    that unpack three-tuples would work in process and break after a resume."""
    from retail_agent.agent.state import step_event

    event = step_event("run_sql", time.perf_counter() - 0.01, "3 row(s)")

    assert set(event) == {"name", "ms", "detail"}
    assert event["name"] == "run_sql"
    assert event["detail"] == "3 row(s)"
    assert isinstance(event["ms"], int) and event["ms"] >= 0


def test_a_frame_becomes_plain_values():
    from retail_agent.agent.state import frame_to_state
    from retail_agent.safety.frame import MaskedFrame

    frame = MaskedFrame(
        columns=("brand", "revenue"),
        rows=(("Levi's", 12), ("Lee", 9)),
        row_count=2,
        redactions=1,
        dropped_columns=("email",),
        truncated=True,
    )

    as_state = frame_to_state(frame)

    assert as_state == {
        "columns": ["brand", "revenue"],
        "rows": [["Levi's", 12], ["Lee", 9]],
        "row_count": 2,
        "redactions": 1,
        "dropped_columns": ["email"],
        "truncated": True,
    }
    assert all(isinstance(row, list) for row in as_state["rows"]), "no tuples"


def test_every_field_survives_a_checkpoint_with_its_type():
    """The point of primitives-only. This is what would catch a MaskedFrame or
    a tuple sneaking back into the schema."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    from retail_agent.agent.state import attempt_record, frame_to_state, step_event
    from retail_agent.safety.frame import MaskedFrame

    written = {
        "attempts": [attempt_record(sql="SELECT 1", executed_sql="SELECT 1")],
        "events": [step_event("run_sql", time.perf_counter(), "ok")],
        "trio_ids": ["churn-v2"],
        "assumed_terms": ["loyal"],
        "preference_changes": [{"action": "added", "note": "be brief"}],
        "reports_written": [{"report_id": "r1", "title": "Q1"}],
        "redactions": 2,
        "calls": 3,
        "context_tokens": 1200,
        "frame": frame_to_state(
            MaskedFrame(columns=("a",), rows=((1,),), row_count=1, redactions=0)
        ),
        "executed_sql": "SELECT 1",
        "status": "ok",
    }

    serializer = JsonPlusSerializer()
    read_back = serializer.loads_typed(serializer.dumps_typed(written))

    assert read_back == written, "a field did not survive the round trip unchanged"
