"""trio embeddings, in pgvector

Revision ID: a1c9e73b4d20
Revises: f2a83e51c9d4
Create Date: 2026-08-07

Replaces the Milvus Lite file. The vectors now live beside the trios they
belong to, in the database that already holds them — one store to run, back up
and reason about instead of two.

Hand-written; `test_migrations_match_the_models` (db-marked) proves it matches
`models.py`.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "a1c9e73b4d20"
down_revision: Union[str, Sequence[str], None] = "f2a83e51c9d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # The compose image is pgvector/pgvector:pg16, so the extension ships with
    # the server and only needs enabling. IF NOT EXISTS keeps this idempotent
    # for a database where someone enabled it by hand.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "trio_embeddings",
        sa.Column("trio_id", sa.String(), nullable=False),
        # Part of the key: vectors from two embedders are not comparable and
        # usually are not the same width.
        sa.Column("model", sa.String(), nullable=False),
        # Unsized on purpose: a width is only required by an ANN index, and
        # there is none (see below). This lets the embedding model change
        # without a migration.
        sa.Column("embedding", Vector(), nullable=False),
        sa.Column("content_hash", sa.String(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["trio_id"], ["trios.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("trio_id", "model"),
    )

    # No ANN index. The corpus is a handful of trios, where an exact scan is
    # both faster and exactly right; ivfflat/hnsw trade recall for speed at a
    # scale this does not have. Add one when the corpus reaches thousands —
    # `CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)`.


def downgrade() -> None:
    op.drop_table("trio_embeddings")
    # The extension is left in place: another table may be using it, and
    # dropping it would cascade to their columns.
