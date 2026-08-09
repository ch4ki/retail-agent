"""SQLAlchemy models — the persistence shape of saved reports and traces.

Deliberately separate from the `Report` and `AuditEntry` dataclasses in
`reports.py`. Those are the domain types the whole application passes around,
and `InMemoryReportStore` must be able to produce them without importing
SQLAlchemy or holding a session. Rows are converted at the boundary.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ReportRow(Base):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    owner_id: Mapped[str] = mapped_column(String, nullable=False)
    session_id: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        # Every read filters on owner and excludes tombstones.
        Index(
            "reports_owner_live_idx",
            "owner_id",
            text("created_at DESC"),
            postgresql_where=text("deleted_at IS NULL"),
        ),
        # "delete all reports mentioning Client X" is resolved by this index.
        Index(
            "reports_fts_idx",
            text("to_tsvector('english', title || ' ' || body)"),
            postgresql_using="gin",
        ),
    )


class ReportAuditRow(Base):
    """One row per destructive action that actually happened.

    `action_id` is the primary key, and that is load-bearing: a replayed
    confirmation hits the conflict, inserts nothing, and so deletes nothing.
    """

    __tablename__ = "report_audit"

    action_id: Mapped[str] = mapped_column(String, primary_key=True)
    owner_id: Mapped[str] = mapped_column(String, nullable=False)
    report_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    token: Mapped[str] = mapped_column(Text, nullable=False)
    performed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    undone_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        Index("report_audit_owner_idx", "owner_id", text("performed_at DESC")),
    )


class TraceRow(Base):
    """One row per completed turn — the summary a metric is computed from."""

    __tablename__ = "traces"

    turn_id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, nullable=False)
    owner_id: Mapped[str] = mapped_column(String, nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("''"))
    redactions: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    bytes_billed: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default=text("0"))
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    # Every draft with its guard verdict, the query that ran, and its outcome.
    # JSONB rather than a table: `/trace` reads them whole, and the shape is
    # owned by `SqlAttempt` rather than by the schema.
    attempts: Mapped[list] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    # Why the answer is what it is: the agreed definitions consulted, the terms
    # the agent had to settle itself, and the preferences it wrote. Lists of
    # names, so nothing here can carry a row value.
    trios: Mapped[list] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    assumptions: Mapped[list] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    preference_changes: Mapped[list] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("traces_recent_idx", "owner_id", "created_at"),
        Index("traces_session_idx", "session_id", "created_at"),
    )


class TurnEventRow(Base):
    """One row per node execution.

    A table rather than a JSON blob because per-node latency percentiles are a
    named metric in the design, and that is a `GROUP BY node` — not something
    to reimplement in Python over parsed JSON.
    """

    __tablename__ = "turn_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    turn_id: Mapped[str] = mapped_column(String, nullable=False)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    node: Mapped[str] = mapped_column(String, nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    detail: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("''"))

    __table_args__ = (
        Index("turn_events_turn_idx", "turn_id", "seq"),
        Index("turn_events_node_idx", "node"),
    )


class PersonaRow(Base):
    """One row per version. Editing appends; nothing is overwritten, because a
    rollback is only possible if the previous body still exists."""

    __tablename__ = "personas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    updated_by: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("name", "version", name="personas_name_version_key"),
        # At most one active persona, enforced by the database rather than by
        # remembering to clear the old one first.
        Index(
            "personas_single_active_idx",
            "is_active",
            unique=True,
            postgresql_where=text("is_active"),
        ),
    )


class PreferenceRow(Base):
    """One row per user. Personas are global; these are not."""

    __tablename__ = "preferences"

    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    answer_format: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text("'table'")
    )
    depth: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text("'standard'")
    )
    max_table_rows: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("20")
    )
    show_attempt_footnote: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class UserDefinitionRow(Base):
    """What one user told the agent a term means. Personal, not the corpus."""

    __tablename__ = "user_definitions"

    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    term: Mapped[str] = mapped_column(String, primary_key=True)
    definition: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class TrioRow(Base):
    """The Golden Bucket. Rows, not a Python constant, so a definition can be
    edited and superseded without a deploy."""

    __tablename__ = "trios"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    sql: Mapped[str] = mapped_column(Text, nullable=False)
    report: Mapped[str] = mapped_column(Text, nullable=False)
    metric_definitions: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    tags: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default=text("'{}'::text[]")
    )
    author: Mapped[str] = mapped_column(String, nullable=False, server_default=text("'analyst'"))
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    # Set rather than deleting, so a report written under the old definition
    # can still be explained against it.
    superseded_by: Mapped[str | None] = mapped_column(String, nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("trios_live_idx", "superseded_by"),
    )


class TrioEmbeddingRow(Base):
    """A trio's vector, kept beside the trio it belongs to.

    Separate from `trios` rather than a column on it, because the two have
    different lifetimes: editing a definition changes the trio immediately,
    while its embedding is only valid until re-embedded. A missing row here
    means "not embedded yet", which a nullable column could not say as clearly.

    `model` is part of the key. Vectors from two embedders are not comparable
    and usually are not the same width, so switching models must not silently
    search a stale collection.
    """

    __tablename__ = "trio_embeddings"

    trio_id: Mapped[str] = mapped_column(
        String, ForeignKey("trios.id", ondelete="CASCADE"), primary_key=True
    )
    model: Mapped[str] = mapped_column(String, primary_key=True)
    # Deliberately unsized. A fixed width is only needed for an ANN index, and
    # there is none here — the corpus is a handful of trios, where an exact scan
    # is both faster and exactly right. Leaving it open means switching
    # embedding model is a config change rather than a migration, which is the
    # point of `model` being part of the key.
    embedding: Mapped[list[float]] = mapped_column(Vector(), nullable=False)
    # Hash of the text that produced the vector, so a trio whose question or
    # definitions changed is re-embedded and one that did not is left alone.
    content_hash: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

class PreferenceDeclineRow(Base):
    """How often a user has refused a suggestion.

    Separate from the counts because the two have different lifetimes: accepting
    a setting clears its evidence, and forgetting the refusals at the same time
    would let the next proposal arrive at full strength.
    """

    __tablename__ = "preference_declines"

    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    field: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[str] = mapped_column(String, primary_key=True)
    count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))

