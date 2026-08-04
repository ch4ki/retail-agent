from retail_agent.llm.errors import describe_llm_error

QUOTA = (
    "Error calling model 'gemini-3.5-flash' (RESOURCE_EXHAUSTED): 429 "
    "RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your "
    "current quota... Quota exceeded for metric: "
    "generativelanguage.googleapis.com/generate_content_free_tier_requests, "
    "limit: 20, model: gemini-3.5-flash'}}"
)


def test_quota_error_is_one_actionable_line():
    message = describe_llm_error(Exception(QUOTA), provider="gemini")

    assert len(message.splitlines()) <= 3
    assert "quota" in message.lower()
    assert "LLM_PROVIDER" in message
    assert "RESOURCE_EXHAUSTED" not in message
    assert "generativelanguage.googleapis.com" not in message


def test_rate_limit_is_recognised():
    message = describe_llm_error(Exception("429 Too Many Requests"), provider="openai")
    assert "rate limit" in message.lower() or "quota" in message.lower()


def test_auth_error_points_at_the_key():
    message = describe_llm_error(
        Exception("401 UNAUTHENTICATED: API key not valid"), provider="gemini"
    )
    assert "GOOGLE_API_KEY" in message


def test_connection_error_mentions_reachability():
    message = describe_llm_error(
        Exception("Connection refused to localhost:11434"), provider="ollama"
    )
    assert "reach" in message.lower()


def test_unknown_error_is_truncated_not_dumped():
    message = describe_llm_error(Exception("x" * 5000), provider="gemini")
    assert len(message) < 400


def test_message_never_leaks_an_api_key():
    err = Exception("bad key: AIzaSyEXAMPLEKEYVALUE1234567890abcdefg")
    message = describe_llm_error(err, provider="gemini")
    assert "AIzaSyEXAMPLEKEYVALUE1234567890abcdefg" not in message
