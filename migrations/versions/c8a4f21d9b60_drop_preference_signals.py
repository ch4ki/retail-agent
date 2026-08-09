"""Drop preference_signals.

The proposal engine it fed is gone: `note_preference` applies a preference the
user stated outright rather than accumulating evidence towards asking about one.

Written as a new revision rather than by deleting `b4d17e29c803`. That migration
ran on databases that exist, and removing it from the chain leaves them stamped
at a revision Alembic can no longer resolve — `upgrade` and `downgrade` both
fail with "Can't locate revision". Deleting applied history is not a way to undo
it. `downgrade()` recreates the table so the round-trip test still holds.

Revision ID: c8a4f21d9b60
Revises: b4d17e29c803
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c8a4f21d9b60"
down_revision: Union[str, Sequence[str], None] = "b4d17e29c803"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_table("preference_signals")


def downgrade() -> None:
    op.create_table(
        "preference_signals",
        sa.Column("id", sa.String(length=32), primary_key=True),
        sa.Column("user_id", sa.String(length=128), nullable=False, index=True),
        sa.Column("field", sa.String(length=64), nullable=False),
        sa.Column("value", sa.String(length=64), nullable=False),
        sa.Column("evidence", sa.Text(), nullable=False),
        sa.Column("seen", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("declined", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()
        ),
        sa.UniqueConstraint("user_id", "field", "value", name="uq_signal_user_field_value"),
    )
