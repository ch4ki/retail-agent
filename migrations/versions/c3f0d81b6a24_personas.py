"""personas, versioned with a single active row

Revision ID: c3f0d81b6a24
Revises: 9a1c4b2e77d1
Create Date: 2026-08-06

Hand-written; `test_migrations_match_the_models` (db-marked) is what proves it
matches `models.py`.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c3f0d81b6a24"
down_revision: Union[str, Sequence[str], None] = "9a1c4b2e77d1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "personas",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column(
            "is_active", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column("updated_by", sa.String(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", "version", name="personas_name_version_key"),
    )
    # At most one active persona, enforced by the database rather than by every
    # caller remembering to clear the previous one first.
    op.create_index(
        "personas_single_active_idx",
        "personas",
        ["is_active"],
        unique=True,
        postgresql_where=sa.text("is_active"),
    )


def downgrade() -> None:
    op.drop_index("personas_single_active_idx", table_name="personas")
    op.drop_table("personas")
