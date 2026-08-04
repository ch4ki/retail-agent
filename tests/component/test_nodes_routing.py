from retail_agent.agent.nodes.chat import chat_node
from retail_agent.agent.nodes.plan import plan_node
from retail_agent.agent.nodes.route import route_node
from retail_agent.agent.nodes.schema_qa import schema_node
from retail_agent.agent.state import new_turn_state


def state_for(question: str):
    return new_turn_state(
        user_id="dana", session_id="s1", question=question, repair_budget=2
    )


def test_route_classifies_schema_question(make_deps):
    deps = make_deps(["schema"])
    result = route_node(state_for("what data do you have?"), deps)
    assert result["intent"] == "schema"


def test_route_classifies_analysis_question(make_deps):
    deps = make_deps(["analyze"])
    result = route_node(state_for("top 10 customers by spend"), deps)
    assert result["intent"] == "analyze"


def test_route_falls_back_to_analyze_on_unrecognised_reply(make_deps):
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
        ["STEP: revenue for brand X by month\nSTEP: revenue for brand Y by month"]
    )
    result = plan_node(state_for("compare brand X and Y"), deps)

    assert len(result["plan"]) == 2
    assert result["plan"][0].question == "revenue for brand X by month"


def test_plan_node_falls_back_to_the_raw_question(make_deps):
    deps = make_deps(["I could not decompose that"])
    result = plan_node(state_for("total revenue"), deps)

    assert len(result["plan"]) == 1
    assert result["plan"][0].question == "total revenue"


def test_plan_node_caps_step_count(make_deps):
    deps = make_deps(["\n".join(f"STEP: q{i}" for i in range(10))])
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
