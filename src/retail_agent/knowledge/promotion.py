"""Promoting a personal definition into the shared corpus.

§5.1's rule, and the reason this is a command rather than a background job:
nothing merges automatically. An agent that writes its own ground truth drifts,
and a poisoned corpus is expensive to recover from. Promotion is a person
deciding that what they told the agent should become what everyone gets.

Superseding rather than editing keeps history intact: a report written under
the old definition can still be explained against it.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from retail_agent.knowledge.trios import Trio


class PromotionError(ValueError):
    """Nothing to promote."""


def promote_definition(
    definitions,
    trios,
    *,
    user_id: str,
    term: str,
    promoted_by: str,
    question: str = "",
) -> Trio:
    """Turn one user's definition into a trio everyone answers from.

    The new trio carries no SQL and no report: those are the analyst-authored
    parts, and inventing them here would put unreviewed queries in front of the
    model as though a human had written them.
    """
    entry = definitions.lookup(user_id=user_id, term=term)
    if entry is None:
        raise PromotionError(
            f"You have not defined {term!r}. Ask a question that uses it and "
            f"I will ask you what it means."
        )

    key = entry.term
    trio = Trio(
        id=f"{key.replace(' ', '-')}-{uuid.uuid4().hex[:6]}",
        question=question or f"Questions about {key}",
        sql="",
        report="",
        metric_definitions={key: entry.definition},
        tags=tuple(dict.fromkeys([*key.split(), key])),
        author=promoted_by,
        approved_at=datetime.now(timezone.utc),
    )

    superseded = [t for t in trios.live() if t.defines(key)]
    trios.add(trio)
    for old in superseded:
        trios.supersede(old_id=old.id, new_id=trio.id)

    return trio
