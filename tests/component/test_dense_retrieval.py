"""Dense retrieval against the real embedding model.

Marked, because the model is fetched on first use:

    uv run pytest -m vector

These assert what the bundled model actually does, measured, rather than what
semantic search is supposed to do. It is a small ONNX model chosen so the
feature needs no API key and no torch, and the trade is real: on this corpus it
ranks the right trio first for roughly half of paraphrased questions. The tests
below are written around that, and the ones that would fail are recorded as
limitations rather than deleted.
"""

import pytest

from retail_agent.config import Settings
from retail_agent.knowledge.dense import build_dense_index
from retail_agent.knowledge.retrieval import lexical_rank, retrieve
from retail_agent.knowledge.seeds import SEED_TRIOS

pytestmark = pytest.mark.vector


@pytest.fixture(scope="module")
def dense(tmp_path_factory):
    settings = Settings(
        _env_file=None,
        dense_retrieval=True,
        milvus_path=str(tmp_path_factory.mktemp("milvus") / "trios.db"),
    )
    index = build_dense_index(settings)
    if index is None:
        pytest.skip("dense retrieval unavailable")
    return index


def test_a_paraphrase_with_no_shared_words_is_found(dense):
    """The reason dense retrieval exists. "repeat purchasers" shares nothing
    with "How many loyal customers do we have?" — it scores 0.517, the
    strongest match in the corpus."""
    question = "how many repeat purchasers?"

    assert lexical_rank(question, list(SEED_TRIOS)) == [], "no lexical overlap"
    found = retrieve(question, list(SEED_TRIOS), dense_rank=dense.rank)

    assert "loyal-customers" in [t.id for t in found]


def test_nonsense_is_rejected_by_the_floor(dense):
    """A vector index always returns its nearest neighbour. Without the floor
    this retrieves whichever trio is least unlike a question about France."""
    found = retrieve(
        "what is the capital of France?", list(SEED_TRIOS), dense_rank=dense.rank
    )

    assert found == []


def test_the_index_follows_the_corpus_without_a_restart(dense):
    """Promotion adds a trio mid-session; a signature check re-embeds only when
    the corpus actually changed."""
    from retail_agent.knowledge.trios import Trio

    extra = Trio(
        id="returns",
        question="What share of orders are sent back?",
        sql="SELECT 1",
        report="",
        metric_definitions={"return rate": "returned items over all items"},
        tags=("returns", "refunds"),
    )
    corpus = [*SEED_TRIOS, extra]

    ranked = dense.rank("how many items get sent back?", corpus)

    assert "returns" in [s.trio.id for s in ranked]


def test_the_ranking_is_a_similarity_not_a_distance(dense):
    """COSINE, so a score is comparable to the floor. With L2 the number is a
    distance whose scale depends on the model and the floor means nothing."""
    ranked = dense.rank("how many repeat purchasers?", list(SEED_TRIOS))

    assert ranked, "expected a hit"
    assert all(-1.0 <= s.score <= 1.0 for s in ranked)
    assert ranked == sorted(ranked, key=lambda s: -s.score), "best first"


@pytest.mark.xfail(
    reason="measured limitation of the bundled model: 'stopped buying' scores "
    "0.138 against churn-90 but 0.302 against underspending, so the wrong trio "
    "ranks first. A production embedding endpoint fixes this; a 45MB ONNX "
    "model that needs no key does not.",
    strict=False,
)
def test_stopped_buying_should_find_churn(dense):
    found = retrieve(
        "who stopped buying from us?", list(SEED_TRIOS), dense_rank=dense.rank
    )

    assert "churn-90" in [t.id for t in found]


# --- the OpenAI backend, which is the default when a key is configured ---

_KEY = Settings().openai_api_key
needs_openai = pytest.mark.skipif(not _KEY, reason="no OPENAI_API_KEY configured")


@pytest.fixture(scope="module")
def openai_dense(tmp_path_factory):
    settings = Settings(
        dense_retrieval=True,
        embedding_backend="openai",
        milvus_path=str(tmp_path_factory.mktemp("milvus-openai") / "trios.db"),
    )
    index = build_dense_index(settings)
    if index is None or index._model_name == "local":
        pytest.skip("OpenAI embedding backend unavailable")
    return index


@needs_openai
@pytest.mark.parametrize(
    "question,expected",
    [
        ("who stopped buying from us?", "churn-90"),
        ("which shoppers have gone quiet?", "churn-90"),
        ("who spends the most with us?", "top-customers"),
        ("which labels sell best?", "brand-performance"),
        ("how many repeat purchasers?", "loyal-customers"),
    ],
)
def test_paraphrases_reach_the_right_trio(openai_dense, question, expected):
    """None of these share distinctive vocabulary with the trio they should
    find — lexical search returns nothing for every one of them."""
    assert lexical_rank(question, list(SEED_TRIOS)) == []

    found = retrieve(question, list(SEED_TRIOS), dense_rank=openai_dense.rank)

    assert expected in [t.id for t in found]


@needs_openai
@pytest.mark.parametrize(
    "question",
    [
        "what is the capital of France?",
        "how do I reset my password?",
        "write me a poem about the sea",
    ],
)
def test_nonsense_retrieves_nothing(openai_dense, question):
    """What the floor buys. A wrong trio supplies a confident wrong definition
    and the agent has no way to tell that it is wrong."""
    assert retrieve(question, list(SEED_TRIOS), dense_rank=openai_dense.rank) == []


@needs_openai
def test_only_the_contenders_come_back(openai_dense):
    """Every retail trio clears the floor for an in-domain question. Without the
    dominance gate this returns five of six trios, and their definitions all go
    into the prompt."""
    found = retrieve(
        "how many repeat purchasers?", list(SEED_TRIOS), dense_rank=openai_dense.rank
    )

    assert found[0].id == "loyal-customers"
    assert len(found) < len(SEED_TRIOS) - 1, [t.id for t in found]
