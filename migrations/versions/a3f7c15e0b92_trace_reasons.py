"""why the answer was what it was, on the trace

Revision ID: a3f7c15e0b92
Revises: c8a4f21d9b60
Create Date: 2026-08-09

The turn already accumulated all three and threw them away before the trace
existed. Non-null with an empty default, so a row written before this reads
back as a turn that consulted nothing rather than as a null the renderer has
to guard.

Hand-written; `test_migrations_match_the_models` (db-marked) proves it matches
`models.py`.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "a3f7c15e0b92"
down_revision: Union[str, Sequence[str], None] = "c8a4f21d9b60"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_COLUMNS = ("trios", "assumptions", "preference_changes")


def upgrade() -> None:
    for name in _COLUMNS:
        op.add_column(
            "traces",
            sa.Column(
                name,
                JSONB(),
                nullable=False,
                server_default=sa.text("'[]'::jsonb"),
            ),
        )


def downgrade() -> None:
    for name in _COLUMNS:
        op.drop_column("traces", name)
