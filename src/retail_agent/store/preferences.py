"""Per-user answer preferences.

The brief's example is "Manager A prefers tables while Manager B prefers bullet
points". So this is owner-scoped in a way personas are not — the house voice is
one setting for everyone, but how an answer is laid out belongs to the person
reading it.

The settings split by where they can be enforced, and that split is real:

- `max_table_rows` and `show_attempt_footnote` are applied at render time.
  They hold whatever the model does.
- `answer_format` and `depth` become instructions in the synthesis prompt.
  Prose cannot be reformatted after it is written, so these can only be asked
  for — `style_instruction` is that request, and a model may ignore it.

Claiming the second pair is enforced would be the more comfortable story and
the false one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal, Protocol, runtime_checkable

AnswerFormat = Literal["table", "bullets", "prose"]
Depth = Literal["summary", "standard", "deep"]

FORMATS: tuple[AnswerFormat, ...] = ("table", "bullets", "prose")
DEPTHS: tuple[Depth, ...] = ("summary", "standard", "deep")

MIN_TABLE_ROWS = 1
MAX_TABLE_ROWS = 100

# The prompt block is read on every model call, so the list has to be bounded.
# Rejecting past the cap rather than dropping the oldest note keeps the user in
# charge of which preference is the one they stopped caring about.
MAX_NOTES = 20
# Long enough for a sentence, short enough that a pasted paragraph is refused
# rather than silently cut down to something they did not ask for.
MAX_NOTE_CHARS = 200


@dataclass(frozen=True)
class Preferences:
    answer_format: AnswerFormat = "table"
    depth: Depth = "standard"
    max_table_rows: int = 20
    show_attempt_footnote: bool = True


DEFAULT_PREFERENCES = Preferences()

# What each setting does, for `/prefs` — a settings screen that does not say
# what a setting does is a settings screen nobody changes.
DESCRIPTIONS = {
    "answer_format": "table | bullets | prose — how comparisons are laid out",
    "depth": "summary | standard | deep — how much explanation to write",
    "max_table_rows": f"{MIN_TABLE_ROWS}-{MAX_TABLE_ROWS} — rows before truncating",
    "show_attempt_footnote": "true | false — show masking and attempt counts",
}


class PreferenceError(ValueError):
    """A value outside what the setting accepts."""


def coerce(field: str, value: str) -> object:
    """Parse and validate one setting from CLI text.

    Rejecting here rather than at write time means a typo is a message, not a
    preference silently set to something meaningless.
    """
    if field == "answer_format":
        if value not in FORMATS:
            raise PreferenceError(f"answer_format must be one of {', '.join(FORMATS)}")
        return value
    if field == "depth":
        if value not in DEPTHS:
            raise PreferenceError(f"depth must be one of {', '.join(DEPTHS)}")
        return value
    if field == "max_table_rows":
        try:
            rows = int(value)
        except ValueError:
            raise PreferenceError("max_table_rows must be a whole number") from None
        if not MIN_TABLE_ROWS <= rows <= MAX_TABLE_ROWS:
            raise PreferenceError(
                f"max_table_rows must be between {MIN_TABLE_ROWS} and {MAX_TABLE_ROWS}"
            )
        return rows
    if field == "show_attempt_footnote":
        if value.lower() not in {"true", "false"}:
            raise PreferenceError("show_attempt_footnote must be true or false")
        return value.lower() == "true"
    raise PreferenceError(f"unknown setting {field!r}. Try /prefs.")


@runtime_checkable
class PreferenceStore(Protocol):
    def get(self, *, user_id: str) -> Preferences: ...

    def set(self, *, user_id: str, **changes) -> Preferences: ...

    def list_notes(self, *, user_id: str) -> list[str]: ...

    def replace_notes(self, *, user_id: str, notes: list[str]) -> None: ...


class InMemoryPreferenceStore:
    def __init__(self) -> None:
        self._by_user: dict[str, Preferences] = {}
        self._notes: dict[str, list[str]] = {}

    def get(self, *, user_id: str) -> Preferences:
        return self._by_user.get(user_id, DEFAULT_PREFERENCES)

    def set(self, *, user_id: str, **changes) -> Preferences:
        # Drop unset keys so `set(user_id=..., depth="deep")` cannot blank the
        # format the user chose last week.
        supplied = {k: v for k, v in changes.items() if v is not None}
        updated = replace(self.get(user_id=user_id), **supplied)
        self._by_user[user_id] = updated
        return updated

    def list_notes(self, *, user_id: str) -> list[str]:
        return list(self._notes.get(user_id, []))

    def replace_notes(self, *, user_id: str, notes: list[str]) -> None:
        self._notes[user_id] = list(notes)


def preferred(store: PreferenceStore | None, user_id: str) -> Preferences:
    """Never fail a turn over a layout setting."""
    if store is None:
        return DEFAULT_PREFERENCES
    try:
        return store.get(user_id=user_id)
    except Exception:
        return DEFAULT_PREFERENCES


def style_instruction(prefs: Preferences) -> str:
    """The prompt-side half.

    Prose cannot be reformatted after the model writes it, so layout and depth
    have to be asked for rather than enforced. The render-side half — row caps
    and footnotes — is enforced.
    """
    layout = {
        "table": "Use a markdown table whenever you compare more than two things.",
        "bullets": "Use short bullet points rather than tables or paragraphs.",
        "prose": "Write in plain paragraphs. Avoid tables and bullet lists.",
    }[prefs.answer_format]
    detail = {
        "summary": "Give the headline number and one sentence of context. Stop there.",
        "standard": "Lead with the answer, then the supporting detail.",
        "deep": (
            "Lead with the answer, then explain the drivers, the caveats, and "
            "what you would check next."
        ),
    }[prefs.depth]
    return f"{layout} {detail}"


class PostgresPreferenceStore:
    """One row per user, upserted.

    Every statement carries `user_id`; there is no cross-user read anywhere in
    this class, which is the whole isolation story.
    """

    def __init__(self, sessions) -> None:
        self._sessions = sessions

    def get(self, *, user_id: str) -> Preferences:
        from sqlalchemy import select

        from retail_agent.store.models import PreferenceRow

        with self._sessions() as session:
            row = session.scalar(
                select(PreferenceRow).where(PreferenceRow.user_id == user_id)
            )
        if row is None:
            return DEFAULT_PREFERENCES
        return Preferences(
            answer_format=row.answer_format,
            depth=row.depth,
            max_table_rows=row.max_table_rows,
            show_attempt_footnote=row.show_attempt_footnote,
        )

    def set(self, *, user_id: str, **changes) -> Preferences:
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        from retail_agent.store.models import PreferenceRow

        supplied = {k: v for k, v in changes.items() if v is not None}
        merged = replace(self.get(user_id=user_id), **supplied)
        values = {
            "user_id": user_id,
            "answer_format": merged.answer_format,
            "depth": merged.depth,
            "max_table_rows": merged.max_table_rows,
            "show_attempt_footnote": merged.show_attempt_footnote,
        }
        with self._sessions.begin() as session:
            session.execute(
                pg_insert(PreferenceRow)
                .values(**values)
                .on_conflict_do_update(
                    index_elements=["user_id"],
                    set_={k: v for k, v in values.items() if k != "user_id"},
                )
            )
        return merged

    def list_notes(self, *, user_id: str) -> list[str]:
        from sqlalchemy import select

        from retail_agent.store.models import PreferenceRow

        with self._sessions() as session:
            row = session.scalar(
                select(PreferenceRow).where(PreferenceRow.user_id == user_id)
            )
        return list(row.notes) if row is not None else []

    def replace_notes(self, *, user_id: str, notes: list[str]) -> None:
        """The whole list, every time.

        A list of twenty short strings is small enough that rewriting it costs
        nothing, and it means there is one write path rather than an insert, a
        delete and an ordering column to keep them straight.
        """
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        from retail_agent.store.models import PreferenceRow

        stored = list(notes)
        with self._sessions.begin() as session:
            session.execute(
                pg_insert(PreferenceRow)
                .values(user_id=user_id, notes=stored)
                .on_conflict_do_update(
                    index_elements=["user_id"], set_={"notes": stored}
                )
            )


def _normalised(note: str) -> str:
    """Trimmed, with runs of whitespace collapsed.

    Both a stored note and a note being matched against one go through this, so
    "keep it  short" and "keep it short" are the same preference rather than two.
    """
    return " ".join(note.split())


def notes_for(store: PreferenceStore | None, user_id: str) -> list[str]:
    """This user's notes, or none. Never raises — see the module docstring."""
    if store is None:
        return []
    try:
        return list(store.list_notes(user_id=user_id))
    except Exception:
        return []


def add_note(store: PreferenceStore, *, user_id: str, note: str) -> str:
    """Save one note, and say what happened.

    Reads through the store rather than through `notes_for`: a read that failed
    must not be mistaken for a user with no notes, because writing on top of
    that would erase the list. The caller catches.

    Dedup is checked before the cap, so someone at twenty notes repeating one
    they already have is told it is already saved rather than told they are full.
    """
    text = _normalised(note)
    if not text:
        return "empty"
    if len(text) > MAX_NOTE_CHARS:
        return "too_long"

    notes = list(store.list_notes(user_id=user_id))
    if any(_normalised(existing).lower() == text.lower() for existing in notes):
        return "duplicate"
    if len(notes) >= MAX_NOTES:
        return "full"

    store.replace_notes(user_id=user_id, notes=[*notes, text])
    return "added"


def remove_note(store: PreferenceStore, *, user_id: str, note: str) -> bool:
    """Drop the matching note. True when one went."""
    target = _normalised(note).lower()
    notes = list(store.list_notes(user_id=user_id))
    kept = [n for n in notes if _normalised(n).lower() != target]
    if len(kept) == len(notes):
        return False
    store.replace_notes(user_id=user_id, notes=kept)
    return True


def preference_block(notes: Sequence[str]) -> str:
    """The prompt-side half: a request, in the user's own words.

    Nothing here is enforced. Free text can only be asked for, and saying so
    plainly is better than a block that reads like a setting the code applies.
    """
    if not notes:
        return ""
    lines = "\n".join(f"- {note}" for note in notes)
    return f"This person has asked for:\n{lines}\nFollow these where they apply."


def build_preference_store(settings, on_degraded=None) -> PreferenceStore:
    """Postgres when reachable, memory when not."""
    from retail_agent.store.db import sessions_or_none

    sessions = sessions_or_none(
        settings.database_url, name="preference store", on_degraded=on_degraded
    )
    return (
        PostgresPreferenceStore(sessions) if sessions else InMemoryPreferenceStore()
    )
