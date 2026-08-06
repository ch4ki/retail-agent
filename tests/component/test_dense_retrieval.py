"""Dense retrieval against the real embedding model.

Marked, because the model is fetched on first use and a suite that needs the
network is a suite people stop running:

    uv run pytest -m vector
"""

import pytest

from retail_agent.config import Settings
from retail_agent.knowledge.retrieval import retrieve
from retail_agent.knowledge.seeds import SEED_TRIOS
from retail_agent.knowledge.dense import build_dense_index

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
    index.index(list(SEED_TRIOS))
    return index.rank


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("who stopped buying from us?", "churn-90"),
        ("which shoppers have gone quiet?", "churn-90"),
        ("who spends the most with us?", "top-customers"),
    ],
)
def test_a_question_sharing_no_words_still_finds_the_trio(dense, question, expected):
    """The reason dense retrieval exists. An analyst wrote "churned"; the
    executive asks "stopped buying". Overlap counting scores that zero."""
    found = retrieve(question, list(SEED_TRIOS), dense_rank=dense)

    assert expected in [t.id for t in found], f"{question!r} did not reach {expected}"


def test_hybrid_finds_what_lexical_alone_cannot(dense):
    question = "who stopped buying from us?"

    lexical_only = retrieve(question, list(SEED_TRIOS))
    hybrid = retrieve(question, list(SEED_TRIOS), dense_rank=dense)

    assert lexical_only == [], "no shared words, so overlap finds nothing"
    assert hybrid, "meaning finds it"


def test_an_unrelated_question_is_not_forced_to_match(dense):
    """A bad trio is worse than no trio. Dense retrieval always returns its
    nearest neighbours, so the relevance floor still has to hold."""
    found = retrieve("what is the capital of France?", list(SEED_TRIOS), dense_rank=dense)

    assert len(found) <= 2, f"expected few or none, got {[t.id for t in found]}"
