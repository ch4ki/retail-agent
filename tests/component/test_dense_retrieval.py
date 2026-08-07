"""Dense retrieval end to end, against a real Postgres with pgvector.

    docker compose up -d postgres && uv run retail-agent migrate
    uv run pytest -m "db and vector"

Marked `db` as well as `vector` because the vectors now live in the database
rather than in a file beside it.

The embedder is faked here on purpose. What is under test is the storage, the
distance query and the two floors — mechanics that must hold whichever model is
configured. The model itself is not asserted on: it bills per run and its scores
drift between versions, so the floor derived from it is recorded as a constant
in `knowledge/dense.py` rather than re-measured on every test run.
"""

from __future__ import annotations

import pytest

from retail_agent.config import Settings
from retail_agent.knowledge.dense import PgVectorIndex, embedding_text
from retail_agent.knowledge.retrieval import lexical_rank, retrieve
from retail_agent.knowledge.seeds import SEED_TRIOS
from retail_agent.knowledge.trios import PostgresTrioStore, Trio
from retail_agent.store.db import create_db_engine, run_migrations, session_factory

pytestmark = [pytest.mark.db, pytest.mark.vector]

DIM = 8

# Each concept owns one dimension, so "which trio is nearest" is decidable by
# reading the corpus rather than by trusting an opaque model.
CONCEPTS = (
    ("churn", "churned", "lapsed", "quiet", "stopped"),
    ("loyal", "repeat", "again"),
    ("top", "best", "biggest", "most"),
    ("brand", "brands", "label", "labels"),
    ("spend", "spent", "revenue", "sales"),
    ("customer", "customers", "shopper", "shoppers"),
    ("state", "region"),
)


def keyword_embedder():
    """A deterministic stand-in: one dimension per concept group."""

    def embed(texts):
        vectors = []
        for text in texts:
            words = set(text.lower().replace("?", " ").replace(":", " ").split())
            vector = [1.0 if words & set(group) else 0.0 for group in CONCEPTS]
            vector.append(0.1)  # keeps a zero vector from being undefined
            vectors.append(vector)
        return vectors

    return embed


@pytest.fixture(scope="module")
def sessions():
    settings = Settings()
    try:
        run_migrations(settings.database_url)
        engine = create_db_engine(settings.database_url)
    except Exception as err:
        pytest.skip(f"Postgres unavailable: {err}")
    yield session_factory(engine)
    engine.dispose()


@pytest.fixture
def corpus(sessions):
    """The seed trios, in the database, since the embeddings reference them."""
    from sqlalchemy import text

    with sessions.begin() as session:
        session.execute(text("TRUNCATE trios CASCADE"))
    store = PostgresTrioStore(sessions)
    store.seed(SEED_TRIOS)
    return list(SEED_TRIOS)


@pytest.fixture
def index(sessions):
    return PgVectorIndex(
        sessions,
        embed=keyword_embedder(),
        model="test-keyword",
        dim=DIM,
        min_similarity=0.35,
    )


def test_a_paraphrase_finds_the_trio_lexical_search_misses(index, corpus):
    """The reason dense retrieval exists: no distinctive word is shared with
    the trio, so lexical ranking returns nothing at all."""
    question = "which shoppers have gone quiet?"

    assert lexical_rank(question, corpus) == []

    assert "churn-90" in [hit.trio.id for hit in index.rank(question, corpus)]


def test_hybrid_retrieval_uses_the_dense_ranker(index, corpus):
    """Through `retrieve`, which is what the graph actually calls."""
    found = retrieve("which shoppers have gone quiet?", corpus, dense_rank=index.rank)

    assert "churn-90" in [trio.id for trio in found]


def test_nonsense_retrieves_nothing(index, corpus):
    """What the floor buys. A wrong trio supplies a confident wrong definition
    and the agent has no way to tell that it is wrong."""
    assert retrieve("what is the capital of France?", corpus, dense_rank=index.rank) == []


def test_an_unchanged_corpus_is_not_re_embedded(sessions, corpus):
    """Re-embedding on every turn would bill an API call per question."""
    calls = []

    def counting_embedder(texts):
        calls.append(len(texts))
        return keyword_embedder()(texts)

    index = PgVectorIndex(
        sessions, embed=counting_embedder, model="test-count", dim=DIM
    )

    index.index(corpus)
    index.index(corpus)

    assert len(calls) == 1, f"embedded {calls} times for an unchanged corpus"


def test_an_edited_trio_is_re_embedded(sessions, corpus):
    """A definition can be edited without a deploy, so the vector has to follow
    it — otherwise retrieval keeps matching against the old meaning."""
    calls = []

    def counting_embedder(texts):
        calls.append(list(texts))
        return keyword_embedder()(texts)

    index = PgVectorIndex(
        sessions, embed=counting_embedder, model="test-edit", dim=DIM
    )
    index.index(corpus)

    edited = [
        Trio(
            id=corpus[0].id,
            question="a completely different question about brands",
            sql=corpus[0].sql,
            report=corpus[0].report,
            metric_definitions=corpus[0].metric_definitions,
            tags=corpus[0].tags,
        ),
        *corpus[1:],
    ]
    index.index(edited)

    assert len(calls) == 2
    assert len(calls[1]) == 1, "only the edited trio should be re-embedded"


def test_a_different_model_does_not_read_another_models_vectors(sessions, corpus):
    """Vectors from two embedders are not comparable and are usually not even
    the same width, so the model is part of the key."""
    first = PgVectorIndex(sessions, embed=keyword_embedder(), model="model-a", dim=DIM)
    first.index(corpus)

    calls = []

    def counting_embedder(texts):
        calls.append(len(texts))
        return keyword_embedder()(texts)

    second = PgVectorIndex(sessions, embed=counting_embedder, model="model-b", dim=DIM)
    second.index(corpus)

    assert calls == [len(corpus)], "model-b must embed for itself, not reuse model-a"


def test_a_superseded_trio_is_not_returned(index, corpus):
    """Its row still exists — nothing is deleted — so it has to be excluded by
    the query rather than by having been removed."""
    live = [t for t in corpus if t.id != "churn-90"]

    found = index.rank("which shoppers have gone quiet?", live)

    assert "churn-90" not in [hit.trio.id for hit in found]


def test_an_unreachable_database_costs_recall_not_the_turn():
    """Dense retrieval is an improvement over lexical, never a dependency."""

    class Broken:
        def begin(self):
            raise RuntimeError("connection refused")

    index = PgVectorIndex(Broken(), embed=keyword_embedder(), dim=DIM)

    assert index.rank("anything", list(SEED_TRIOS)) == []


def test_what_is_embedded_matches_what_lexical_search_reads():
    text = embedding_text(SEED_TRIOS[0]).lower()

    assert SEED_TRIOS[0].question.lower()[:20] in text
