"""The agent's tone, editable without a deploy.

The brief asks for a CEO to change how reports read, weekly, without a
developer. So the persona is a row, not a constant — loaded per turn behind a
short TTL cache, versioned so a bad edit can be rolled back, and attributed so a
tone change has an author.

The constraint that makes this safe is not in this file: the persona is
injected into a slot in the prompt and the safety rules are appended *after* it,
and none of the deterministic guards read it at all. Whoever edits the tone
cannot turn off PII masking or the confirmation gate, even by writing "ignore
all previous instructions" into the body.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Persona:
    name: str
    body: str
    version: int = 1
    is_active: bool = False
    updated_by: str = "system"
    updated_at: datetime | None = None


DEFAULT_PERSONA = Persona(
    name="analyst",
    body=(
        "You are a data analyst supporting retail executives. Write plainly and "
        "lead with the answer. Quantify claims. Avoid jargon."
    ),
)

# Long enough that a turn does not pay for a query, short enough that an edit
# lands while the person who made it is still watching.
CACHE_TTL_SECONDS = 60.0


@runtime_checkable
class PersonaStore(Protocol):
    def save(self, *, name: str, body: str, updated_by: str) -> Persona: ...

    def get(self, *, name: str, version: int | None = None) -> Persona | None: ...

    def list_personas(self) -> list[Persona]: ...

    def activate(self, *, name: str, version: int | None = None) -> Persona: ...

    def active(self) -> Persona | None: ...

    def seed(self, persona: Persona) -> None: ...


def _now() -> datetime:
    return datetime.now(timezone.utc)


class InMemoryPersonaStore:
    """Also the fallback when Postgres is unreachable: the agent keeps its
    voice for the session, and edits simply do not survive a restart."""

    def __init__(self) -> None:
        self._versions: dict[str, list[Persona]] = {}
        self._active: tuple[str, int] | None = None

    def save(self, *, name: str, body: str, updated_by: str) -> Persona:
        history = self._versions.setdefault(name, [])
        persona = Persona(
            name=name,
            body=body,
            version=len(history) + 1,
            updated_by=updated_by,
            updated_at=_now(),
        )
        history.append(persona)
        return persona

    def get(self, *, name: str, version: int | None = None) -> Persona | None:
        history = self._versions.get(name)
        if not history:
            return None
        if version is None:
            return history[-1]
        return next((p for p in history if p.version == version), None)

    def list_personas(self) -> list[Persona]:
        latest = []
        for name, history in sorted(self._versions.items()):
            newest = history[-1]
            latest.append(
                replace(newest, is_active=self._is_active(name, newest.version))
            )
        return latest

    def activate(self, *, name: str, version: int | None = None) -> Persona:
        persona = self.get(name=name, version=version)
        if persona is None:
            raise KeyError(f"no persona named {name!r}")
        self._active = (persona.name, persona.version)
        return replace(persona, is_active=True)

    def active(self) -> Persona | None:
        if self._active is None:
            return None
        name, version = self._active
        persona = self.get(name=name, version=version)
        return replace(persona, is_active=True) if persona else None

    def seed(self, persona: Persona) -> None:
        if persona.name not in self._versions:
            self.save(
                name=persona.name, body=persona.body, updated_by=persona.updated_by
            )
        if self._active is None:
            self.activate(name=persona.name)

    def _is_active(self, name: str, version: int) -> bool:
        return self._active == (name, version)


class CachedPersonaStore:
    """Reads `active()` at most once per TTL.

    Every turn asks for the persona at least once, and most ask three or four
    times across routing, synthesis and report bodies. Without this, changing
    the tone would mean either a restart or a query per prompt.
    """

    def __init__(
        self,
        inner: PersonaStore,
        *,
        ttl_seconds: float = CACHE_TTL_SECONDS,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._inner = inner
        self._ttl = ttl_seconds
        self._now = now
        self._cached: Persona | None = None
        self._read_at: float | None = None

    def active(self) -> Persona | None:
        if self._read_at is not None and self._now() - self._read_at < self._ttl:
            return self._cached
        self._cached = self._inner.active()
        self._read_at = self._now()
        return self._cached

    def invalidate(self) -> None:
        """After a local edit, so the person who just changed it sees it now
        rather than in a minute."""
        self._read_at = None

    def save(self, *, name: str, body: str, updated_by: str) -> Persona:
        self.invalidate()
        return self._inner.save(name=name, body=body, updated_by=updated_by)

    def activate(self, *, name: str, version: int | None = None) -> Persona:
        self.invalidate()
        return self._inner.activate(name=name, version=version)

    def seed(self, persona: Persona) -> None:
        self.invalidate()
        self._inner.seed(persona)

    def get(self, *, name: str, version: int | None = None) -> Persona | None:
        return self._inner.get(name=name, version=version)

    def list_personas(self) -> list[Persona]:
        return self._inner.list_personas()


def active_body(store: PersonaStore | None) -> str:
    """The tone text for a prompt, with the built-in default as the floor.

    A missing or empty persona must never leave the slot blank: the prompt would
    then open with the safety rules and no role, which changes how every answer
    reads for reasons nobody chose.
    """
    if store is None:
        return DEFAULT_PERSONA.body
    try:
        persona = store.active()
    except Exception:  # a persona is not worth failing a turn over
        return DEFAULT_PERSONA.body
    return (persona.body if persona and persona.body.strip() else DEFAULT_PERSONA.body)


class PostgresPersonaStore:
    """Versions are rows; activation is a column.

    A partial unique index on `is_active` means the database refuses two active
    personas, so "exactly one voice" is a schema property rather than something
    every caller has to remember to maintain.
    """

    def __init__(self, sessions) -> None:
        self._sessions = sessions

    def save(self, *, name: str, body: str, updated_by: str) -> Persona:
        from sqlalchemy import func as sa_func
        from sqlalchemy import select

        from retail_agent.store.models import PersonaRow

        with self._sessions.begin() as session:
            highest = session.scalar(
                select(sa_func.max(PersonaRow.version)).where(PersonaRow.name == name)
            )
            row = PersonaRow(
                name=name,
                body=body,
                version=(highest or 0) + 1,
                updated_by=updated_by,
            )
            session.add(row)
            session.flush()
            session.refresh(row)
        return _to_persona(row)

    def get(self, *, name: str, version: int | None = None) -> Persona | None:
        from sqlalchemy import select

        from retail_agent.store.models import PersonaRow

        query = select(PersonaRow).where(PersonaRow.name == name)
        query = (
            query.where(PersonaRow.version == version)
            if version is not None
            else query.order_by(PersonaRow.version.desc()).limit(1)
        )
        with self._sessions() as session:
            row = session.scalar(query)
        return _to_persona(row) if row else None

    def list_personas(self) -> list[Persona]:
        from sqlalchemy import func as sa_func
        from sqlalchemy import select, tuple_

        from retail_agent.store.models import PersonaRow

        latest = (
            select(PersonaRow.name, sa_func.max(PersonaRow.version))
            .group_by(PersonaRow.name)
            .subquery()
        )
        with self._sessions() as session:
            rows = session.scalars(
                select(PersonaRow)
                .where(tuple_(PersonaRow.name, PersonaRow.version).in_(select(latest)))
                .order_by(PersonaRow.name)
            ).all()
        return [_to_persona(row) for row in rows]

    def activate(self, *, name: str, version: int | None = None) -> Persona:
        from sqlalchemy import select, update

        from retail_agent.store.models import PersonaRow

        target = self.get(name=name, version=version)
        if target is None:
            raise KeyError(f"no persona named {name!r}")

        with self._sessions.begin() as session:
            # Clear first: the unique index would reject two active rows.
            session.execute(
                update(PersonaRow)
                .where(PersonaRow.is_active.is_(True))
                .values(is_active=False)
            )
            session.execute(
                update(PersonaRow)
                .where(
                    PersonaRow.name == name, PersonaRow.version == target.version
                )
                .values(is_active=True)
            )
            row = session.scalar(
                select(PersonaRow).where(
                    PersonaRow.name == name, PersonaRow.version == target.version
                )
            )
            return _to_persona(row)

    def active(self) -> Persona | None:
        from sqlalchemy import select

        from retail_agent.store.models import PersonaRow

        with self._sessions() as session:
            row = session.scalar(
                select(PersonaRow).where(PersonaRow.is_active.is_(True))
            )
        return _to_persona(row) if row else None

    def seed(self, persona: Persona) -> None:
        if self.get(name=persona.name) is None:
            self.save(
                name=persona.name, body=persona.body, updated_by=persona.updated_by
            )
        if self.active() is None:
            self.activate(name=persona.name)


def _to_persona(row) -> Persona:
    return Persona(
        name=row.name,
        body=row.body,
        version=row.version,
        is_active=row.is_active,
        updated_by=row.updated_by,
        updated_at=row.updated_at,
    )


def build_persona_store(settings, on_degraded=None) -> PersonaStore:
    """Postgres when reachable, memory when not — behind the TTL cache either
    way, so callers see one shape."""
    from retail_agent.store.db import sessions_or_none

    sessions = sessions_or_none(
        settings.database_url, name="persona store", on_degraded=on_degraded
    )
    store: PersonaStore = (
        PostgresPersonaStore(sessions) if sessions else InMemoryPersonaStore()
    )

    cached = CachedPersonaStore(store)
    try:
        cached.seed(DEFAULT_PERSONA)
    except Exception as err:  # a fresh database without the table yet
        logging.getLogger(__name__).debug("persona seed skipped: %s", err)
    return cached
