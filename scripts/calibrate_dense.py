"""Where the dense relevance floor comes from.

A floor is the only thing standing between the agent and a confidently wrong
definition, because a vector index always returns its nearest neighbour however
far away it is. Picking that number by intuition is guessing. This measures it.

    uv run python scripts/calibrate_dense.py

The method: score five paraphrased questions against the trio each *should*
find, score four unrelated questions against the whole corpus, and read off the
gap. A backend is usable when the weakest true match outscores the strongest
piece of nonsense. If those ranges overlap, no floor is both sensitive and
precise, and the honest move is to say so rather than to pick the number that
makes the tests pass.

That is not hypothetical — it is what happened here. The bundled ONNX model
overlaps and was demoted to the no-key fallback because of this script.
"""

from __future__ import annotations

import numpy as np

from retail_agent.config import Settings
from retail_agent.knowledge.dense import embedding_text
from retail_agent.knowledge.seeds import SEED_TRIOS

# Deliberately share no distinctive vocabulary with the trio they should find:
# any of these that lexical search could answer is not testing dense retrieval.
RELEVANT = {
    "who stopped buying from us?": "churn-90",
    "which shoppers have gone quiet?": "churn-90",
    "who spends the most with us?": "top-customers",
    "which labels sell best?": "brand-performance",
    "how many repeat purchasers?": "loyal-customers",
}

# Nothing to do with retail. Whatever the best of these scores is the highest a
# floor may sit below.
IRRELEVANT = [
    "what is the capital of France?",
    "how do I reset my password?",
    "what time is the meeting?",
    "write me a poem about the sea",
]


def cosine(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def calibrate(name: str, encode_documents, encode_queries) -> None:
    documents = encode_documents([embedding_text(t) for t in SEED_TRIOS])
    ids = [t.id for t in SEED_TRIOS]

    print(f"\n=== {name}  (dim {len(documents[0])}) ===")

    on_target, correct = [], 0
    for question, expected in RELEVANT.items():
        vector = encode_queries([question])[0]
        scores = {i: cosine(vector, d) for i, d in zip(ids, documents)}
        best = max(scores, key=scores.get)
        correct += best == expected
        on_target.append(scores[expected])
        verdict = "ok" if best == expected else f"MISRANKED to {best}"
        print(f"  {question:34} {scores[expected]:+.3f} vs {expected:18} {verdict}")

    noise = [max(cosine(encode_queries([q])[0], d) for d in documents) for q in IRRELEVANT]
    for question, score in zip(IRRELEVANT, noise):
        print(f"  {question:34} {score:+.3f}  (nearest; must be rejected)")

    weakest_true, loudest_noise = min(on_target), max(noise)
    print(f"\n  top-1 accuracy      {correct}/{len(RELEVANT)}")
    print(f"  weakest true match  {weakest_true:+.3f}")
    print(f"  loudest nonsense    {loudest_noise:+.3f}")
    if weakest_true > loudest_noise:
        print(
            f"  SEPARABLE — {weakest_true - loudest_noise:.3f} of daylight; "
            f"put the floor at {(weakest_true + loudest_noise) / 2:.2f}"
        )
    else:
        print(
            "  OVERLAPS — a question that should match scores below nonsense that "
            "should not.\n  No floor is both sensitive and precise. Prefer precision "
            "and accept the lost recall."
        )


def main() -> None:
    settings = Settings()

    from pymilvus import model as milvus_model

    local = milvus_model.DefaultEmbeddingFunction()
    calibrate("local (bundled ONNX)", local.encode_documents, local.encode_queries)

    if not settings.openai_api_key:
        print("\nOPENAI_API_KEY not set — skipping the OpenAI backends.")
        return

    from pymilvus.model.dense import OpenAIEmbeddingFunction

    for model_name in ("text-embedding-3-small", "text-embedding-3-large"):
        fn = OpenAIEmbeddingFunction(model_name=model_name, api_key=settings.openai_api_key)
        calibrate(model_name, fn.encode_documents, fn.encode_queries)


if __name__ == "__main__":
    main()
