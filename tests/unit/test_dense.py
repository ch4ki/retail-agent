"""Dense retrieval over the Golden Bucket, in Postgres.

The ranking rules are pure and tested here. The storage half needs a real
database with the `vector` extension and lives in `test_pgvector_index.py`
behind `-m db`.
"""

from __future__ import annotations

import pytest

from retail_agent.config import Settings
from retail_agent.knowledge.dense import (
    DOMINANCE,
    MIN_SIMILARITY,
    build_dense_index,
    embedding_text,
    select_hits,
    similarity_from_distance,
)
from retail_agent.knowledge.retrieval import Scored
from retail_agent.knowledge.trios import Trio

CHURN = Trio(
    id="churn-90",
    question="Why did our churn rate spike last month?",
    sql="SELECT 1",
    report="",
    metric_definitions={"churn": "no completed order in 90 days"},
    tags=("retention",),
)


def trio(trio_id: str) -> Trio:
    return Trio(id=trio_id, question="q", sql="", report="", metric_definitions={})


def scored(trio_id: str, score: float) -> Scored:
    return Scored(trio=trio(trio_id), score=score)


# --- what gets embedded ---


def test_what_is_embedded_covers_question_tags_and_definitions():
    """The same material lexical search reads, so the two rankers disagree
    about ranking rather than about what a trio is about."""
    text = embedding_text(CHURN).lower()

    assert "churn" in text and "retention" in text and "90 days" in text


# --- turning a distance into a score ---


def test_cosine_distance_becomes_a_similarity():
    """pgvector's `<=>` returns a distance in [0, 2]. A floor is only meaningful
    against a similarity, and every number in the calibration is one."""
    assert similarity_from_distance(0.0) == 1.0
    assert similarity_from_distance(1.0) == 0.0
    assert similarity_from_distance(0.75) == pytest.approx(0.25)


# --- the two floors ---


def test_a_hit_below_the_absolute_floor_is_dropped():
    """A vector index always returns its nearest neighbour however far away it
    is. Without a floor "what is the capital of France?" retrieves whichever
    trio is least unlike it, and a bad trio is worse than no trio."""
    hits = select_hits(
        [scored("far", 0.10)], min_similarity=0.20, dominance=0.9
    )

    assert hits == []


def test_a_hit_above_the_floor_is_kept():
    assert [h.trio.id for h in select_hits([scored("near", 0.45)], min_similarity=0.20, dominance=0.9)] == ["near"]


def test_also_rans_are_dropped_even_when_they_clear_the_floor():
    """For an in-domain question every retail trio clears the absolute floor,
    and five trios' worth of definitions in a prompt is dilution rather than
    context."""
    hits = select_hits(
        [scored("best", 0.48), scored("close", 0.45), scored("also-ran", 0.25)],
        min_similarity=0.20,
        dominance=0.9,
    )

    assert [h.trio.id for h in hits] == ["best", "close"]


def test_the_dominance_gate_is_relative_to_the_best_hit():
    """So it tightens on a confident match and stays permissive on a weak one,
    which an absolute second threshold could not do."""
    hits = select_hits(
        [scored("best", 0.30), scored("close", 0.28)], min_similarity=0.20, dominance=0.9
    )

    assert len(hits) == 2


def test_nothing_in_means_nothing_out():
    assert select_hits([], min_similarity=0.20, dominance=0.9) == []


def test_the_floors_are_the_calibrated_ones():
    """Measured against the seed corpus with text-embedding-3-small: relevant
    questions scored 0.296 and up, unrelated ones no higher than 0.102. The
    floor sits in that gap. See `scripts/calibrate_dense.py`."""
    assert 0.102 < MIN_SIMILARITY < 0.296
    assert 0.5 < DOMINANCE < 1.0


# --- construction ---


def test_dense_retrieval_is_off_unless_asked_for():
    assert build_dense_index(Settings(_env_file=None), sessions=object()) is None


def test_it_needs_an_embedding_key():
    """Without one there is nothing to embed with, and dense retrieval degrades
    to lexical rather than failing the turn."""
    settings = Settings(_env_file=None, dense_retrieval=True, openai_api_key=None)

    assert build_dense_index(settings, sessions=object()) is None


def test_it_needs_a_database():
    """The vectors live in Postgres now. Without a session factory there is
    nowhere to put them."""
    settings = Settings(_env_file=None, dense_retrieval=True, openai_api_key="sk-test")

    assert build_dense_index(settings, sessions=None) is None


def test_it_is_built_when_both_are_present():
    settings = Settings(_env_file=None, dense_retrieval=True, openai_api_key="sk-test")

    index = build_dense_index(settings, sessions=object())

    assert index is not None
    assert index.dim == 1536
