from retail_agent.agent.nodes.chat import chat_node
from retail_agent.agent.nodes.plan import plan_node
from retail_agent.agent.nodes.route import route_node
from retail_agent.agent.nodes.schema_qa import schema_node
from retail_agent.agent.state import fresh_scratch, new_turn_state


def state_for(question: str):
    # The state as `start_turn` leaves it, since these call nodes directly.
    state = new_turn_state(user_id="dana", session_id="s1", question=question)
    state.update(fresh_scratch(repair_budget=2))
    return state


def test_route_reads_a_structured_decision(make_deps):
    """The label comes back as a validated field, not as text to be matched
    against a set — so an out-of-range intent cannot reach the graph."""
    deps = make_deps([{"intent": "schema"}])

    assert route_node(state_for("what data do you have?"), deps)["intent"] == "schema"


def test_plan_reads_structured_steps(make_deps):
    """The prompt and the parser used to be two artifacts that could drift; a
    planner told to emit SQL while the parser looked for `STEP:` lines silently
    collapsed every question to one step. A schema makes them one artifact."""
    deps = make_deps([{"steps": ["revenue for brand X", "revenue for brand Y"]}])

    result = plan_node(state_for("compare brand X and brand Y"), deps)

    assert [step.question for step in result["plan"]] == [
        "revenue for brand X",
        "revenue for brand Y",
    ]


def test_route_classifies_a_report_operation(make_deps):
    deps = make_deps([{"intent": "report_op"}])
    result = route_node(state_for("delete all reports mentioning Calvin Klein"), deps)
    assert result["intent"] == "report_op"


def test_route_classifies_schema_question(make_deps):
    deps = make_deps([{"intent": "schema"}])
    result = route_node(state_for("what data do you have?"), deps)
    assert result["intent"] == "schema"


def test_route_classifies_analysis_question(make_deps):
    deps = make_deps([{"intent": "analyze"}])
    result = route_node(state_for("top 10 customers by spend"), deps)
    assert result["intent"] == "analyze"


def test_route_falls_back_to_analyze_when_structured_output_fails(make_deps):
    """Constrained decoding is not universal across OpenRouter models. A reply
    the schema rejects must cost a query, not the turn."""
    deps = make_deps(["I think this is a question about data"])
    result = route_node(state_for("revenue last month"), deps)
    assert result["intent"] == "analyze"


def test_schema_node_answers_without_sql(make_deps, source):
    deps = make_deps(["We hold orders, order items, products and customers."])
    result = schema_node(state_for("what can I ask?"), deps)

    assert "orders" in result["answer"]
    assert result["status"] == "ok"
    assert source.executed == []


def test_plan_node_extracts_steps(make_deps):
    deps = make_deps(
        [{"steps": ["revenue for brand X by month", "revenue for brand Y by month"]}]
    )
    result = plan_node(state_for("compare brand X and Y"), deps)

    assert len(result["plan"]) == 2
    assert result["plan"][0].question == "revenue for brand X by month"


def test_plan_node_falls_back_to_the_raw_question(make_deps):
    """Same guard on the planner: an unusable reply becomes one step carrying
    the question verbatim."""
    deps = make_deps(["I could not decompose that"])
    result = plan_node(state_for("total revenue"), deps)

    assert len(result["plan"]) == 1
    assert result["plan"][0].question == "total revenue"


def test_plan_node_caps_step_count(make_deps):
    deps = make_deps([{"steps": [f"q{i}" for i in range(10)]}])
    result = plan_node(state_for("everything"), deps)

    assert len(result["plan"]) <= deps.settings.max_analysis_steps


def test_chat_node_answers_from_conversation(make_deps):
    deps = make_deps(["Happy to help — ask away."])
    result = chat_node(state_for("thanks!"), deps)

    assert result["answer"] == "Happy to help — ask away."
    assert result["status"] == "ok"


def test_chat_node_scans_its_output_for_pii(make_deps):
    deps = make_deps(["Sure — mail them at ada@example.com."])
    result = chat_node(state_for("how do I reach them?"), deps)

    assert "ada@example.com" not in result["answer"]


# Studio lets you submit state directly, so a turn can arrive with no user
# message. Previously that produced a plan holding one step with an empty
# question, which then generated SQL for nothing.


def test_blank_turn_is_routed_to_chat_without_calling_the_model(make_deps):
    from retail_agent.agent.state import TurnState

    deps = make_deps([])  # any LLM call would raise "ran out of replies"
    result = route_node(TurnState(messages=[]), deps)

    assert result["intent"] == "chat"
    assert deps.llm.prompts == []


def test_whitespace_only_question_is_treated_as_blank(make_deps):
    deps = make_deps([])
    assert route_node(state_for("   "), deps)["intent"] == "chat"


def test_chat_node_asks_for_input_on_a_blank_turn(make_deps):
    from retail_agent.agent.state import TurnState

    deps = make_deps([])
    result = chat_node(TurnState(messages=[]), deps)

    assert deps.llm.prompts == [], "no model call for an empty turn"
    assert result["answer"]
    assert "?" in result["answer"]


def _conversation(*pairs):
    """A state carrying prior exchanges plus a new question."""
    from langchain_core.messages import AIMessage, HumanMessage

    from retail_agent.agent.state import TurnState, fresh_scratch

    messages = []
    for role, text in pairs:
        messages.append(
            HumanMessage(content=text) if role == "user" else AIMessage(content=text)
        )
    state = TurnState(messages=messages)
    state.update(fresh_scratch(repair_budget=3))
    return state


def test_planner_sees_the_prior_exchange(make_deps):
    """"and how does that compare to April?" is unanswerable without knowing
    what "that" was. Given only the latest message the planner invents a
    subject, producing steps like "retrieve the relevant data"."""
    deps = make_deps([{"steps": ["revenue for April 2024"]}])
    state = _conversation(
        ("user", "what was total revenue in March 2024?"),
        ("assistant", "Revenue in March 2024 was $1,284,000."),
        ("user", "and how does that compare to April?"),
    )

    plan_node(state, deps)

    prompt = deps.llm.prompts[-1]
    assert "March 2024" in prompt
    assert "$1,284,000" in prompt


def test_router_sees_the_prior_exchange(make_deps):
    """The router's own definition of "chat" is "answerable from results already
    in this conversation" — a judgement it cannot make without the conversation."""
    deps = make_deps([{"intent": "analyze"}])
    state = _conversation(
        ("user", "what was total revenue in March 2024?"),
        ("assistant", "Revenue in March 2024 was $1,284,000."),
        ("user", "and how does that compare to April?"),
    )

    route_node(state, deps)

    assert "March 2024" in deps.llm.prompts[-1]


def test_history_is_bounded(make_deps):
    """History grows without limit across a session; the prompt must not."""
    deps = make_deps([{"steps": ["revenue"]}])
    pairs = []
    for i in range(20):
        pairs.append(("user", f"question number {i}"))
        pairs.append(("assistant", f"answer number {i}"))
    state = _conversation(*pairs)

    plan_node(state, deps)

    prompt = deps.llm.prompts[-1]
    assert "question number 19" in prompt, "the most recent turn must survive"
    assert "question number 0" not in prompt, "the oldest must not"


def test_planner_never_receives_a_blank_question(make_deps):
    """A blank turn writes no plan of its own; `start_turn` already left it
    empty, so the node has nothing to say and nothing to spend a call on."""
    deps = make_deps([{"steps": ["something"]}])
    result = plan_node(state_for("   "), deps)

    assert result == {}
    assert deps.llm.prompts == []
