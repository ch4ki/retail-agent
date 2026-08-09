"""free-text preference notes

Revision ID: b7c3e91f5a44
Revises: a3f7c15e0b92
Create Date: 2026-08-09

Hand-written; `test_migrations_match_the_models` (db-marked) is what proves it
matches `models.py`.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "b7c3e91f5a44"
down_revision: Union[str, Sequence[str], None] = "a3f7c15e0b92"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "preferences",
        sa.Column(
            "notes",
            postgresql.JSONB(),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("preferences", "notes")
