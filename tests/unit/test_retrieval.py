"""Retrieval, and the floor that matters more than the retrieval.

A bad trio is worse than no trio: it supplies a confident wrong definition and
nothing downstream can tell that it is wrong.
"""

import pytest

from retail_agent.knowledge.retrieval import (
    Scored,
    lexical_rank,
    reciprocal_rank_fusion,
    retrieve,
    tokenize,
)
from retail_agent.knowledge.trios import Trio


def trio(id, question, tags=(), definitions=None, superseded_by=None):
    return Trio(
        id=id,
        question=question,
        sql="SELECT 1",
        report="A finding.",
        metric_definitions=definitions or {},
        tags=tags,
        superseded_by=superseded_by,
    )


CHURN = trio("churn", "Which customers churned last quarter?", ("churn", "retention"),
             {"churn": "no order in the trailing 90 days"})
BRANDS = trio("brands", "Which brands drive the most revenue?", ("brand", "revenue"))
SHIPPING = trio("shipping", "How long do orders take to ship?", ("shipping", "logistics"))
CORPUS = [CHURN, BRANDS, SHIPPING]


def test_stopwords_are_dropped():
    assert "the" not in tokenize("what is the revenue")
    assert "revenue" in tokenize("what is the revenue")


def test_the_relevant_trio_ranks_first():
    ranked = lexical_rank("why did churn spike last month?", CORPUS)

    assert ranked[0].trio.id == "churn"


def test_a_question_about_nothing_in_the_corpus_returns_nothing():
    assert lexical_rank("what is the weather today?", CORPUS) == []


def test_tags_and_definitions_are_searchable_not_just_the_question():
    ranked = lexical_rank("retention problems", CORPUS)

    assert ranked and ranked[0].trio.id == "churn", "matched on a tag"


# --- fusion ---


def test_fusion_rewards_agreement_between_rankers():
    a = [Scored(CHURN, 1.0), Scored(BRANDS, 0.5)]
    b = [Scored(CHURN, 0.9), Scored(SHIPPING, 0.4)]

    fused = reciprocal_rank_fusion([a, b])

    assert fused[0].trio.id == "churn", "ranked highly by both"


def test_fusion_uses_position_not_score():
    """Lexical overlap and cosine similarity are not comparable numbers. Fusing
    on rank is what lets the two be combined at all."""
    a = [Scored(BRANDS, 1000.0), Scored(CHURN, 999.0)]
    b = [Scored(CHURN, 0.02), Scored(BRANDS, 0.01)]

    fused = reciprocal_rank_fusion([a, b])

    assert {s.trio.id for s in fused[:2]} == {"brands", "churn"}
    assert fused[0].score == pytest.approx(fused[1].score, rel=0.2), (
        "neither wins by having bigger raw numbers"
    )


# --- the floor ---


def test_the_floor_drops_everything_but_the_strongest_match():
    """The floor is a share of the best score, so raising it to 1.0 keeps only
    what the question is actually about. This is the knob that decides whether
    a marginal trio reaches the model."""
    permissive = retrieve("why did churn spike for customers?", CORPUS, floor=0.0)
    strict = retrieve("why did churn spike for customers?", CORPUS, floor=1.0)

    assert [t.id for t in strict] == ["churn"]
    assert len(permissive) >= len(strict)


def test_nothing_relevant_returns_nothing_at_all():
    assert retrieve("what is the capital of France?", CORPUS) == []


def test_a_superseded_trio_is_never_retrieved():
    """Definitions change. A report from last quarter can still be read against
    the definition that produced it, but new questions must not be."""
    old = trio("old", "Which customers churned?", ("churn",), {"churn": "60 days"},
               superseded_by="churn")

    found = retrieve("why did churn spike?", [old])

    assert found == []


def test_the_strong_match_survives_the_floor():
    found = retrieve("why did churn spike last quarter?", CORPUS)

    assert [t.id for t in found] == ["churn"]


def test_results_are_capped():
    corpus = [
        trio(f"t{i}", f"What was revenue in Q{i}?", ("revenue",)) for i in range(10)
    ]

    assert len(retrieve("what was revenue?", corpus, top_k=3)) <= 3


# --- the optional dense half ---


def test_dense_ranking_is_optional():
    """Embeddings need a provider and a key. The agent has to work without
    one, so the feature cannot depend on it."""
    assert retrieve("why did churn spike?", CORPUS, dense_rank=None)


def test_a_dense_ranker_contributes_when_supplied():
    def dense(question, trios):
        # Pretends to understand that "lapsed" means churn, which lexical
        # matching cannot.
        return [Scored(CHURN, 0.99)]

    # No word here appears in any trio, so lexical matching finds nothing.
    lexical_only = retrieve("who stopped buying?", CORPUS)
    hybrid = retrieve("who stopped buying?", CORPUS, dense_rank=dense)

    assert lexical_only == []
    assert [t.id for t in hybrid] == ["churn"]
