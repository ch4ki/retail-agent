from langchain_core.messages import AIMessage

from retail_agent.llm.messages import message_text


def test_plain_string_content():
    assert message_text(AIMessage(content="SELECT 1")) == "SELECT 1"


def test_gemini_style_block_list():
    # What gemini-2.5/3.x actually returns: a list of blocks, each carrying a
    # thinking signature alongside the text.
    message = AIMessage(
        content=[{"type": "text", "text": "schema", "extras": {"signature": "abc"}}]
    )
    assert message_text(message) == "schema"


def test_multiple_text_blocks_are_joined():
    message = AIMessage(
        content=[
            {"type": "text", "text": "SELECT id\n"},
            {"type": "text", "text": "FROM users"},
        ]
    )
    assert message_text(message) == "SELECT id\nFROM users"


def test_non_text_blocks_are_ignored():
    message = AIMessage(
        content=[
            {"type": "thinking", "thinking": "let me consider"},
            {"type": "text", "text": "the answer"},
        ]
    )
    assert message_text(message) == "the answer"


def test_list_of_bare_strings():
    assert message_text(AIMessage(content=["a", "b"])) == "ab"


def test_empty_content_is_empty_string():
    assert message_text(AIMessage(content=[])) == ""


def test_surrounding_whitespace_is_stripped():
    message = AIMessage(content=[{"type": "text", "text": "  schema\n"}])
    assert message_text(message) == "schema"


def test_a_raw_string_is_accepted_directly():
    assert message_text("already text") == "already text"
