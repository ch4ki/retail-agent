"""per-user answer preferences

Revision ID: d5e2a94c1f07
Revises: c3f0d81b6a24
Create Date: 2026-08-06

Hand-written; `test_migrations_match_the_models` (db-marked) is what proves it
matches `models.py`.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d5e2a94c1f07"
down_revision: Union[str, Sequence[str], None] = "c3f0d81b6a24"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "preferences",
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column(
            "answer_format", sa.String(), server_default=sa.text("'table'"), nullable=False
        ),
        sa.Column(
            "depth", sa.String(), server_default=sa.text("'standard'"), nullable=False
        ),
        sa.Column(
            "max_table_rows", sa.Integer(), server_default=sa.text("20"), nullable=False
        ),
        sa.Column(
            "show_attempt_footnote",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("user_id"),
    )


def downgrade() -> None:
    op.drop_table("preferences")
