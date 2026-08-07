"""preference signals and declines

Revision ID: b4d17e29c803
Revises: a1c9e73b4d20
Create Date: 2026-08-07

The learning loop counted evidence in process memory, and its proposal threshold
is three — so it could only ever fire if a user expressed the same preference
three times in one sitting. These tables are what make it reachable.

Hand-written; `test_migrations_match_the_models` (db-marked) proves it matches
`models.py`.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b4d17e29c803"
down_revision: Union[str, Sequence[str], None] = "a1c9e73b4d20"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "preference_signals",
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("field", sa.String(), nullable=False),
        sa.Column("value", sa.String(), nullable=False),
        sa.Column("count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        # Overwritten per sighting: the proposal quotes "most recently ...", so
        # only the newest wording is ever read.
        sa.Column("evidence", sa.Text(), server_default=sa.text("''"), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("user_id", "field", "value"),
    )

    # Separate table, not a column: accepting a setting clears its evidence, and
    # forgetting the refusals at the same time would let the next proposal
    # arrive at full strength instead of at three times the threshold.
    op.create_table(
        "preference_declines",
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("field", sa.String(), nullable=False),
        sa.Column("value", sa.String(), nullable=False),
        sa.Column("count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.PrimaryKeyConstraint("user_id", "field", "value"),
    )


def downgrade() -> None:
    op.drop_table("preference_declines")
    op.drop_table("preference_signals")
