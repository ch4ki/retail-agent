"""Dense retrieval over the Golden Bucket, using Milvus Lite.

The half of hybrid search that lexical matching cannot do: finding the trio
whose analyst wrote "lapsed customers" when the executive asked about "churn".
Reciprocal Rank Fusion already existed and was only ever fed one ranking; this
supplies the other.

Milvus Lite is embedded — a file, no server, no container — so the feature
survives the constraint that shaped the rest of this project: it has to run on
a grader's machine. Embeddings are local too, via the ONNX model that ships
with `pymilvus[model]`, so no provider and no API key are involved. The cost is
a one-time model download, which is why this is off unless switched on.

Everything here degrades to nothing. If the dependency is missing, the model
cannot be fetched, or the index fails to build, `rank` returns no results and
retrieval falls back to lexical — which is the behaviour the whole feature had
before this file existed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from retail_agent.knowledge.retrieval import Scored
from retail_agent.knowledge.trios import Trio

log = logging.getLogger(__name__)

COLLECTION = "trios"
# Cosine, so a hit's score is a similarity in [-1, 1] and a floor means
# something. With L2 the number is a distance whose scale depends on the model.
METRIC = "COSINE"
# A vector index always returns its nearest neighbour, however far away it is.
# Without a floor, "what is the capital of France?" retrieves whichever trio
# happens to be least unlike it — and a bad trio is worse than no trio.
MIN_SIMILARITY = 0.35
# Milvus Lite needs the dimension up front, and the bundled model is 768.
DEFAULT_DIM = 768
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


class MilvusDenseIndex:
    """A local vector index over the corpus, rebuilt when the corpus changes.

    `rank` matches the `dense_rank` seam `retrieve` already accepts, so nothing
    in the retrieval code had to change to accommodate it.
    """

    def __init__(
        self,
        *,
        path: str = "./.milvus/trios.db",
        embed: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None = None,
        dim: int = DEFAULT_DIM,
        top_k: int = TOP_K,
        min_similarity: float = MIN_SIMILARITY,
    ) -> None:
        self._path = path
        self._embed = embed
        self._dim = dim
        self._top_k = top_k
        self._min_similarity = min_similarity
        self._client = None
        self._indexed: str | None = None  # signature of the corpus in the index

    # --- lazily acquired, so importing this module costs nothing ---

    def _embedder(self):
        if self._embed is None:
            from pymilvus import model as milvus_model

            fn = milvus_model.DefaultEmbeddingFunction()
            self._embed = fn.encode_documents
            self._query_embed = fn.encode_queries
        return self._embed

    def _query_embedder(self):
        self._embedder()
        # A supplied embedder is used for both sides; the bundled model
        # distinguishes documents from queries and does better for it.
        return getattr(self, "_query_embed", self._embed)

    def _connect(self):
        if self._client is None:
            from pathlib import Path

            from pymilvus import MilvusClient

            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
            self._client = MilvusClient(self._path)
        return self._client

    # --- indexing ---

    @staticmethod
    def signature(trios: Sequence[Trio]) -> str:
        """Changes when the corpus does, so promotion is picked up without a
        restart and an unchanged corpus is never re-embedded."""
        import hashlib

        material = "|".join(sorted(f"{t.id}:{embedding_text(t)}" for t in trios))
        return hashlib.sha256(material.encode()).hexdigest()[:16]

    def index(self, trios: Sequence[Trio]) -> None:
        """(Re)build the collection. Cheap to call: it returns immediately if
        the corpus has not changed."""
        current = self.signature(trios)
        if current == self._indexed:
            return

        client = self._connect()
        if client.has_collection(COLLECTION):
            client.drop_collection(COLLECTION)
        client.create_collection(
            collection_name=COLLECTION,
            dimension=self._dim,
            auto_id=False,
            metric_type=METRIC,
        )

        if trios:
            vectors = self._embedder()([embedding_text(t) for t in trios])
            client.insert(
                collection_name=COLLECTION,
                data=[
                    {"id": position, "vector": list(vector), "trio_id": trio.id}
                    for position, (trio, vector) in enumerate(zip(trios, vectors))
                ],
            )
        self._indexed = current

    # --- the seam `retrieve` expects ---

    def rank(self, question: str, trios: Sequence[Trio]) -> list[Scored]:
        """Trios most similar to the question, best first.

        Returns nothing on any failure. Dense retrieval is an improvement over
        lexical, never a dependency of it — a missing model must cost recall,
        not the answer.
        """
        if not trios:
            return []
        try:
            self.index(trios)
            vector = self._query_embedder()([question])[0]
            hits = self._connect().search(
                collection_name=COLLECTION,
                data=[list(vector)],
                limit=min(self._top_k, len(trios)),
                output_fields=["trio_id"],
            )
        except Exception as err:
            log.warning("dense retrieval unavailable (%s); using lexical only", err)
            return []

        by_id = {trio.id: trio for trio in trios}
        ranked = []
        for hit in hits[0] if hits else []:
            trio_id = (hit.get("entity") or {}).get("trio_id")
            trio = by_id.get(trio_id)
            similarity = float(hit.get("distance", 0.0))
            if trio is not None and similarity >= self._min_similarity:
                ranked.append(Scored(trio=trio, score=similarity))
        return ranked


def build_dense_index(settings):
    """The dense ranker, or nothing.

    Off unless asked for: the first call downloads an embedding model, and a
    grader running this for the first time should not wait for that without
    having chosen to.
    """
    if not getattr(settings, "dense_retrieval", False):
        return None
    try:
        import pymilvus  # noqa: F401
    except ImportError:
        log.warning("DENSE_RETRIEVAL is on but pymilvus is not installed")
        return None
    return MilvusDenseIndex(path=settings.milvus_path)
