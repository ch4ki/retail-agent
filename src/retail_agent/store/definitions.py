"""Definitions a user gave when the agent had to ask.

The Golden Bucket holds what the analytics team agreed. This holds what one
executive told the agent when the bucket had nothing — asked once, then reused,
because asking the same person what "loyal" means every week is how a safety
feature becomes an irritation.

Deliberately *below* the corpus in precedence. A trio is a reviewed, versioned
decision by the people who own the numbers; this is one person's answer typed
into a terminal. It fills gaps, it never overrides.

Per user, not global, for the same reason: one manager saying loyal means three
orders must not silently redefine the term for everyone else. Promoting a
personal definition into the shared corpus is §5.1's correction-capture path,
and it goes through human review.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

MAX_DEFINITION_CHARS = 500


@dataclass(frozen=True)
class UserDefinition:
    user_id: str
    term: str
    definition: str
    created_at: datetime | None = None


@runtime_checkable
class DefinitionStore(Protocol):
    def remember(self, *, user_id: str, term: str, definition: str) -> UserDefinition: ...

    def lookup(self, *, user_id: str, term: str) -> UserDefinition | None: ...

    def list_definitions(self, *, user_id: str) -> list[UserDefinition]: ...

    def forget(self, *, user_id: str, term: str) -> bool: ...


def _now() -> datetime:
    from datetime import timezone

    return datetime.now(timezone.utc)


class InMemoryDefinitionStore:
    def __init__(self) -> None:
        self._by_user: dict[str, dict[str, UserDefinition]] = {}

    def remember(self, *, user_id: str, term: str, definition: str) -> UserDefinition:
        entry = UserDefinition(
            user_id=user_id,
            term=term.lower().strip(),
            definition=definition.strip()[:MAX_DEFINITION_CHARS],
            created_at=_now(),
        )
        self._by_user.setdefault(user_id, {})[entry.term] = entry
        return entry

    def lookup(self, *, user_id: str, term: str) -> UserDefinition | None:
        return self._by_user.get(user_id, {}).get(term.lower().strip())

    def list_definitions(self, *, user_id: str) -> list[UserDefinition]:
        return sorted(self._by_user.get(user_id, {}).values(), key=lambda d: d.term)

    def forget(self, *, user_id: str, term: str) -> bool:
        return self._by_user.get(user_id, {}).pop(term.lower().strip(), None) is not None


def remembered(store: DefinitionStore | None, user_id: str, terms: list[str]) -> dict[str, str]:
    """The user's own definitions for these terms, if any.

    Never fails a turn: a store that is down costs a question the user has
    already answered, not the answer.
    """
    if store is None or not terms:
        return {}
    found: dict[str, str] = {}
    for term in terms:
        try:
            entry = store.lookup(user_id=user_id, term=term)
        except Exception:
            return found
        if entry is not None:
            found[term] = entry.definition
    return found


def personal_definitions_block(definitions: dict[str, str]) -> str:
    """Marked as the user's own, so the model does not present a personal
    working definition as though the analytics team had agreed it."""
    if not definitions:
        return ""
    lines = "\n".join(f"- {term}: {meaning}" for term, meaning in sorted(definitions.items()))
    return (
        "Definitions this user gave you previously. Use them, and say they are "
        f"the user's own rather than an agreed company definition:\n{lines}"
    )


def ask_for_definition(term: str, hint: str) -> str:
    """What the user is asked. Names the term, says what is undecided, and
    gives an example so the answer is usable rather than another abstraction."""
    return (
        f"No agreed definition for **{term}** — {hint}.\n"
        f"How should I define it? For example: "
        f"\"three or more completed orders\".\n"
        f"Press enter to let me choose and I will say what I assumed."
    )


class PostgresDefinitionStore:
    """One row per (user, term), upserted. Every statement carries `user_id`."""

    def __init__(self, sessions) -> None:
        self._sessions = sessions

    def remember(self, *, user_id: str, term: str, definition: str) -> UserDefinition:
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        from retail_agent.store.models import UserDefinitionRow

        key = term.lower().strip()
        body = definition.strip()[:MAX_DEFINITION_CHARS]
        with self._sessions.begin() as session:
            session.execute(
                pg_insert(UserDefinitionRow)
                .values(user_id=user_id, term=key, definition=body)
                .on_conflict_do_update(
                    index_elements=["user_id", "term"], set_={"definition": body}
                )
            )
        return UserDefinition(user_id=user_id, term=key, definition=body)

    def lookup(self, *, user_id: str, term: str) -> UserDefinition | None:
        from sqlalchemy import select

        from retail_agent.store.models import UserDefinitionRow

        with self._sessions() as session:
            row = session.scalar(
                select(UserDefinitionRow).where(
                    UserDefinitionRow.user_id == user_id,
                    UserDefinitionRow.term == term.lower().strip(),
                )
            )
        return _to_definition(row) if row else None

    def list_definitions(self, *, user_id: str) -> list[UserDefinition]:
        from sqlalchemy import select

        from retail_agent.store.models import UserDefinitionRow

        with self._sessions() as session:
            rows = session.scalars(
                select(UserDefinitionRow)
                .where(UserDefinitionRow.user_id == user_id)
                .order_by(UserDefinitionRow.term)
            ).all()
        return [_to_definition(row) for row in rows]

    def forget(self, *, user_id: str, term: str) -> bool:
        from sqlalchemy import delete

        from retail_agent.store.models import UserDefinitionRow

        with self._sessions.begin() as session:
            result = session.execute(
                delete(UserDefinitionRow).where(
                    UserDefinitionRow.user_id == user_id,
                    UserDefinitionRow.term == term.lower().strip(),
                )
            )
        return result.rowcount > 0


def _to_definition(row) -> UserDefinition:
    return UserDefinition(
        user_id=row.user_id,
        term=row.term,
        definition=row.definition,
        created_at=row.created_at,
    )


def build_definition_store(settings, on_degraded=None) -> DefinitionStore:
    """Postgres when reachable, memory when not.

    Note the consequence when it degrades: the agent still asks, but the answer
    only lasts the session. That is worse than persistence and much better than
    silently assuming.
    """
    import logging

    from retail_agent.store.db import create_db_engine, session_factory

    try:
        engine = create_db_engine(settings.database_url)
        with engine.connect():
            pass
        return PostgresDefinitionStore(session_factory(engine))
    except Exception as err:
        logging.getLogger(__name__).debug("definition store degraded: %s", err)
        if on_degraded is not None:
            on_degraded()
        return InMemoryDefinitionStore()
