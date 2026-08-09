"""what a turn produced, and what it will cost every later turn

Revision ID: e4b8c2f10d93
Revises: b7c3e91f5a44
Create Date: 2026-08-09

Two columns for two questions the trace could not answer. `report_ids` because
the report body no longer passes through the answer, so without it a turn that
wrote a report leaves no link to it. `context_tokens` because the summarisation
threshold was being set against a number nothing measured.

Non-null with defaults, so a row written before this reads back as a turn that
produced no reports and was never measured, rather than as a null the renderer
has to guard.

Hand-written; `test_migrations_match_the_models` (db-marked) proves it matches
`models.py`.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "e4b8c2f10d93"
down_revision: Union[str, Sequence[str], None] = "b7c3e91f5a44"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "traces",
        sa.Column(
            "report_ids", JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")
        ),
    )
    op.add_column(
        "traces",
        sa.Column(
            "context_tokens", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
    )


def downgrade() -> None:
    op.drop_column("traces", "context_tokens")
    op.drop_column("traces", "report_ids")
