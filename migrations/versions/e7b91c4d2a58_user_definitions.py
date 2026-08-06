"""definitions a user gave when asked

Revision ID: e7b91c4d2a58
Revises: d5e2a94c1f07
Create Date: 2026-08-06

Hand-written; `test_migrations_match_the_models` (db-marked) proves it matches
`models.py`.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e7b91c4d2a58"
down_revision: Union[str, Sequence[str], None] = "d5e2a94c1f07"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "user_definitions",
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("term", sa.String(), nullable=False),
        sa.Column("definition", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("user_id", "term"),
    )


def downgrade() -> None:
    op.drop_table("user_definitions")
