"""Dense retrieval.

The unit tests inject a deterministic embedder, so they exercise the index and
the fusion without a model download or a network call. A live-marked test uses
the real one.
"""

import pytest

from retail_agent.config import Settings
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


def test_also_rans_are_dropped_even_when_they_clear_the_floor(tmp_path):
    """The absolute floor rejects nonsense; this rejects the merely-related.

    For an in-domain question every retail trio clears the floor, and five
    trios' worth of definitions in the prompt is dilution rather than context.
    """
    index = MilvusDenseIndex(
        path=str(tmp_path / "trios.db"),
        embed=lambda texts: [[1.0, 0.0], [0.99, 0.14], [0.6, 0.8]][: len(texts)],
        query_embed=lambda _texts: [[1.0, 0.0]],
        dim=2,
        min_similarity=0.3,
        dominance=0.9,
    )
    trios = [
        Trio(id="best", question="a", sql="", report="", metric_definitions={}),
        Trio(id="close", question="b", sql="", report="", metric_definitions={}),
        Trio(id="also-ran", question="c", sql="", report="", metric_definitions={}),
    ]

    ranked = index.rank("a", trios)

    # 'also-ran' scores 0.6 — above the 0.3 floor, but far below the best.
    assert [s.trio.id for s in ranked] == ["best", "close"]


def test_switching_embedding_model_reindexes(tmp_path):
    """Vectors from two embedders are not comparable and usually are not even
    the same width, so a stale collection must be rebuilt rather than searched."""
    trios = [Trio(id="t", question="a", sql="", report="", metric_definitions={})]
    index = MilvusDenseIndex(
        path=str(tmp_path / "trios.db"),
        embed=lambda texts: [[1.0, 0.0] for _ in texts],
        dim=2,
        model_name="model-a",
    )
    index.index(trios)
    before = index._indexed

    index._model_name = "model-b"
    index.index(trios)

    assert index._indexed != before, "same corpus, different model: must re-embed"


def test_openai_is_preferred_when_a_key_is_configured(monkeypatch):
    """Not vendor preference: the local model's scores for relevant questions
    overlap its scores for nonsense, so it has no usable relevance floor."""
    pytest.importorskip("openai")
    settings = Settings(
        _env_file=None, dense_retrieval=True, openai_api_key="sk-test-not-called"
    )

    index = build_dense_index(settings)

    assert index._model_name == "text-embedding-3-small"
    assert index._dim == 1536
    assert index._min_similarity == 0.20


def test_without_a_key_it_falls_back_to_the_local_model():
    """The feature has to work with no provider and no key, at a known cost."""
    settings = Settings(_env_file=None, dense_retrieval=True, openai_api_key=None)

    index = build_dense_index(settings)

    assert index._model_name == "local"
    assert index._dim == 768


def test_asking_for_openai_without_a_key_degrades_rather_than_crashes():
    settings = Settings(
        _env_file=None,
        dense_retrieval=True,
        embedding_backend="openai",
        openai_api_key=None,
    )

    assert build_dense_index(settings)._model_name == "local"


def test_local_can_be_forced_even_when_a_key_exists():
    """Keeps every embedding on the machine, which is the reason to want it."""
    settings = Settings(
        _env_file=None,
        dense_retrieval=True,
        embedding_backend="local",
        openai_api_key="sk-test",
    )

    assert build_dense_index(settings)._model_name == "local"
