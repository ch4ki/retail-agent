"""Dense retrieval over the Golden Bucket, using pgvector.

The half of hybrid search that lexical matching cannot do: finding the trio
whose analyst wrote "lapsed customers" when the executive asked about "churn".
Reciprocal Rank Fusion already existed and was only ever fed one ranking; this
supplies the other.

The vectors live in Postgres, beside the trios they describe. That is one store
to run, migrate, back up and reason about instead of two — and it means a trio
and its embedding can be written in the same transaction, so they cannot drift
apart. The alternative this replaced, an embedded Milvus Lite file, was a second
database whose only job was holding six rows.

Everything here degrades to nothing. If the extension is missing, the key is
absent or the query fails, `rank` returns no results and retrieval falls back to
lexical — which is what the whole feature did before this file existed.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Sequence

from retail_agent.knowledge.retrieval import Scored
from retail_agent.knowledge.trios import Trio

log = logging.getLogger(__name__)

# A vector index always returns its nearest neighbour, however far away it is.
# Without a floor, "what is the capital of France?" retrieves whichever trio
# happens to be least unlike it — and a bad trio is worse than no trio, because
# it supplies a confident wrong definition the agent cannot tell is wrong.
#
# Measured against the seed corpus with text-embedding-3-small, not chosen a
# priori: relevant questions scored 0.296 and up, unrelated ones reached no
# higher than 0.102, and the floor sits in that gap.
#
# It is a property of the model AND the corpus, so it goes stale when either
# moves. To re-derive: embed each seed trio, score a handful of paraphrased
# questions against the trio each should find, score some unrelated questions
# against the whole corpus, and put the floor between the two ranges. If they
# overlap, that model cannot do this job — which is why the local ONNX model
# this project first shipped with was dropped (relevant as low as 0.138,
# nonsense up to 0.222).
MIN_SIMILARITY = 0.20

# The absolute floor rejects nonsense. This rejects the also-rans: for an
# in-domain question every retail trio clears the floor, and five trios' worth
# of definitions in a prompt is dilution rather than context. Keeping only hits
# within 10% of the best cut the result set from five trios to one-to-three
# across every calibration question without dropping the right one.
DOMINANCE = 0.90

# text-embedding-3-small. Fixed, because the column is declared with it.
OPENAI_DIM = 1536
TOP_K = 5


def embedding_text(trio: Trio) -> str:
    """What gets embedded.

    The same material lexical search reads — question, tags, definitions — so
    the two rankers disagree about *ranking* rather than about what a trio is
    about.
    """
    definitions = " ".join(
        f"{term}: {meaning}" for term, meaning in trio.metric_definitions.items()
    )
    return f"{trio.question} {' '.join(trio.tags)} {definitions}".strip()


def content_hash(trio: Trio) -> str:
    """Changes when the embedded text does, so an edited definition is
    re-embedded and an untouched trio is left alone."""
    return hashlib.sha256(embedding_text(trio).encode()).hexdigest()[:32]


def similarity_from_distance(distance: float) -> float:
    """pgvector's `<=>` is cosine *distance*, in [0, 2]. Every number in the
    calibration is a similarity, and a floor only means something against one."""
    return 1.0 - float(distance)


def select_hits(
    hits: Sequence[Scored], *, min_similarity: float, dominance: float
) -> list[Scored]:
    """Apply both floors, best first."""
    above = [hit for hit in hits if hit.score >= min_similarity]
    if not above:
        return []
    ordered = sorted(above, key=lambda hit: -hit.score)
    cutoff = ordered[0].score * dominance
    return [hit for hit in ordered if hit.score >= cutoff]


class PgVectorIndex:
    """The corpus as vectors in Postgres.

    `rank` matches the `dense_rank` seam `retrieve` already accepts, so nothing
    in the retrieval code had to change to accommodate it.
    """

    def __init__(
        self,
        sessions,
        *,
        embed: Callable[[Sequence[str]], Sequence[Sequence[float]]],
        query_embed: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None = None,
        model: str = "text-embedding-3-small",
        dim: int = OPENAI_DIM,
        top_k: int = TOP_K,
        min_similarity: float = MIN_SIMILARITY,
        dominance: float = DOMINANCE,
    ) -> None:
        self._sessions = sessions
        self._embed = embed
        self._query_embed = query_embed or embed
        self.model = model
        self.dim = dim
        self._top_k = top_k
        self._min_similarity = min_similarity
        self._dominance = dominance

    # --- indexing ---

    def index(self, trios: Sequence[Trio]) -> None:
        """Embed anything whose text has changed, and nothing else.

        Keyed by (trio_id, model) and guarded by a content hash, so this is
        cheap to call on every turn: an unchanged corpus costs one SELECT and
        no embedding calls.
        """
        from sqlalchemy import select
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        from retail_agent.store.models import TrioEmbeddingRow

        wanted = {trio.id: content_hash(trio) for trio in trios}
        if not wanted:
            return

        with self._sessions.begin() as session:
            rows = session.execute(
                select(TrioEmbeddingRow.trio_id, TrioEmbeddingRow.content_hash).where(
                    TrioEmbeddingRow.model == self.model
                )
            ).all()
            current = {trio_id: digest for trio_id, digest in rows}

            stale = [t for t in trios if current.get(t.id) != wanted[t.id]]
            if not stale:
                return

            vectors = self._embed([embedding_text(t) for t in stale])
            for trio, vector in zip(stale, vectors):
                session.execute(
                    pg_insert(TrioEmbeddingRow)
                    .values(
                        trio_id=trio.id,
                        model=self.model,
                        embedding=list(vector),
                        content_hash=wanted[trio.id],
                    )
                    .on_conflict_do_update(
                        index_elements=["trio_id", "model"],
                        set_={
                            "embedding": list(vector),
                            "content_hash": wanted[trio.id],
                        },
                    )
                )

    # --- the seam `retrieve` expects ---

    def rank(self, question: str, trios: Sequence[Trio]) -> list[Scored]:
        """Trios most similar to the question, best first.

        Returns nothing on any failure. Dense retrieval is an improvement over
        lexical, never a dependency of it — a missing extension or an unreachable
        database must cost recall, not the answer.
        """
        if not trios:
            return []

        from sqlalchemy import select

        from retail_agent.store.models import TrioEmbeddingRow

        by_id = {trio.id: trio for trio in trios}

        try:
            self.index(trios)
            vector = list(self._query_embed([question])[0])

            distance = TrioEmbeddingRow.embedding.cosine_distance(vector)
            with self._sessions.begin() as session:
                rows = session.execute(
                    select(TrioEmbeddingRow.trio_id, distance.label("distance"))
                    .where(
                        TrioEmbeddingRow.model == self.model,
                        # Only score trios the caller is actually considering —
                        # a superseded one still has a row here.
                        TrioEmbeddingRow.trio_id.in_(list(by_id)),
                    )
                    .order_by(distance)
                    .limit(min(self._top_k, len(trios)))
                ).all()
        except Exception as err:
            log.warning("dense retrieval unavailable (%s); using lexical only", err)
            return []

        hits = [
            Scored(trio=by_id[trio_id], score=similarity_from_distance(distance))
            for trio_id, distance in rows
            if trio_id in by_id
        ]
        return select_hits(
            hits, min_similarity=self._min_similarity, dominance=self._dominance
        )


def build_dense_index(settings, *, sessions) -> PgVectorIndex | None:
    """The dense ranker, or nothing.

    Off unless asked for, and silently absent without either half it needs: an
    embedding key and a database. Both missing pieces degrade to lexical
    retrieval rather than failing a turn, which is the behaviour the feature had
    before it existed.
    """
    if not getattr(settings, "dense_retrieval", False):
        return None

    key = getattr(settings, "openai_api_key", None)
    if not key:
        log.warning("DENSE_RETRIEVAL is on but no OPENAI_API_KEY is set; using lexical only")
        return None

    if sessions is None:
        log.warning("DENSE_RETRIEVAL is on but Postgres is unreachable; using lexical only")
        return None

    try:
        from openai import OpenAI
    except ImportError:
        log.warning("DENSE_RETRIEVAL is on but the openai package is missing")
        return None

    model = settings.openai_embedding_model
    client = OpenAI(api_key=key)

    def embed(texts: Sequence[str]) -> list[list[float]]:
        response = client.embeddings.create(model=model, input=list(texts))
        return [item.embedding for item in response.data]

    return PgVectorIndex(sessions, embed=embed, model=model)
