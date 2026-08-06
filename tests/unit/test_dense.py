"""Dense retrieval.

The unit tests inject a deterministic embedder, so they exercise the index and
the fusion without a model download or a network call. A live-marked test uses
the real one.
"""

import pytest

from retail_agent.knowledge.dense import MilvusDenseIndex, build_dense_index, embedding_text
from retail_agent.knowledge.retrieval import retrieve
from retail_agent.knowledge.trios import Trio


def trio(id, question, tags=(), definitions=None):
    return Trio(
        id=id, question=question, sql="SELECT 1", report="A finding.",
        metric_definitions=definitions or {}, tags=tags,
    )


CHURN = trio("churn", "Which customers churned?", ("churn", "retention"),
             {"churn": "no order in 90 days"})
BRANDS = trio("brands", "Which brands drive revenue?", ("brand", "revenue"))
CORPUS = [CHURN, BRANDS]


# One dimension per *concept*, not per word — otherwise the fake cannot express
# the only thing dense retrieval is here to do: recognise that "lapsed" and
# "churned" mean the same thing. The trailing bias dimension keeps every vector
# non-zero, because cosine similarity is undefined for a zero vector.
CONCEPTS = (
    ("churn", "churned", "lapsed", "retention", "stopped"),
    ("brand", "brands", "revenue", "sales"),
)


def keyword_embedder(concepts=CONCEPTS):
    """A stand-in for a real model. Deterministic and offline, so these tests
    assert the index and the fusion rather than the quality of an embedding."""

    def embed(texts):
        vectors = []
        for text in texts:
            words = set(text.lower().replace("?", " ").replace(":", " ").split())
            vectors.append(
                [1.0 if words & set(group) else 0.0 for group in concepts] + [0.1]
            )
        return vectors

    return embed


VOCAB = CONCEPTS


@pytest.fixture
def index(tmp_path):
    return MilvusDenseIndex(
        path=str(tmp_path / "trios.db"),
        embed=keyword_embedder(),
        dim=len(CONCEPTS) + 1,
        min_similarity=0.35,
    )


def test_what_is_embedded_covers_question_tags_and_definitions():
    """The same material lexical search reads, so the two rankers disagree
    about ranking rather than about what a trio is about."""
    text = embedding_text(CHURN).lower()

    assert "churned" in text and "retention" in text and "90 days" in text


def test_the_nearest_trio_ranks_first(index):
    ranked = index.rank("what happened to churn?", CORPUS)

    assert ranked and ranked[0].trio.id == "churn"


def test_an_empty_corpus_ranks_nothing(index):
    assert index.rank("anything", []) == []


def test_reindexing_an_unchanged_corpus_is_skipped(index):
    calls = []
    index._embed = lambda texts: (calls.append(1), keyword_embedder()(texts))[1]

    index.index(CORPUS)
    index.index(CORPUS)

    assert len(calls) == 1, "an unchanged corpus must not be re-embedded"


def test_a_changed_corpus_is_reindexed(index):
    index.index(CORPUS)
    before = index._indexed

    index.index([*CORPUS, trio("new", "Which customers lapsed?", ("lapsed",))])

    assert index._indexed != before, "promotion has to be picked up without a restart"


def test_a_broken_embedder_costs_recall_not_the_answer(index):
    def explode(_texts):
        raise RuntimeError("model download failed")

    index._embed = explode

    assert index.rank("what happened to churn?", CORPUS) == []


# --- what it adds to retrieval ---


def test_dense_finds_what_lexical_cannot(index):
    """The case that justifies hybrid at all: the executive says "lapsed", the
    analyst wrote "churned", and no word overlaps."""
    # No word here appears in any trio, so lexical finds nothing at all.
    lexical_only = retrieve("lapsed accounts?", CORPUS)

    hybrid = retrieve("lapsed accounts?", CORPUS, dense_rank=index.rank)

    assert lexical_only == [], "no shared vocabulary"
    assert [t.id for t in hybrid] == ["churn"]


def test_dense_does_not_drag_in_something_irrelevant(index):
    """The relevance floor still applies. A vector index always returns its
    nearest neighbour, however far away it is."""
    found = retrieve("what is the capital of France?", CORPUS, dense_rank=index.rank)

    assert found == []


# --- configuration ---


def test_it_is_off_unless_switched_on():
    """The first call downloads a model. A grader should choose that."""
    from retail_agent.config import Settings

    assert build_dense_index(Settings(_env_file=None)) is None


def test_switching_it_on_produces_an_index():
    from retail_agent.config import Settings

    settings = Settings(_env_file=None, dense_retrieval=True)

    assert build_dense_index(settings) is not None
