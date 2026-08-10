"""Drop the preference settings nothing reads.

`answer_format`, `depth` and `max_table_rows` were stored and validated, and
nothing read them: `preference_block` took their prompt slot, and the row cap
lost its reader when the CLI stopped rendering result frames. A setting that
silently does nothing is worse than one that was never offered.

What a user had set is folded into the notes list — the mechanism that
actually reaches the prompt — so the preference survives even though the
column does not. `max_table_rows` is dropped without a note: with no frame
left to truncate, a note about row counts would be noise.

`downgrade()` restores the columns at their defaults. The folded notes are
left in place: they are true statements of what the user asked for, and
removing them would drop a preference the upgrade preserved.

Revision ID: f8d2c40a1b7e
Revises: e4b8c2f10d93
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "f8d2c40a1b7e"
down_revision: Union[str, Sequence[str], None] = "e4b8c2f10d93"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE preferences SET notes = notes || jsonb_build_array(
            CASE answer_format
                WHEN 'bullets' THEN 'Use short bullet points rather than tables.'
                WHEN 'prose' THEN 'Write in plain paragraphs rather than tables.'
            END)
        WHERE answer_format IN ('bullets', 'prose')
        """
    )
    op.execute(
        """
        UPDATE preferences SET notes = notes || jsonb_build_array(
            CASE depth
                WHEN 'summary' THEN 'Give the headline number and one sentence of context.'
                WHEN 'deep' THEN 'Explain the drivers, the caveats, and what you would check next.'
            END)
        WHERE depth IN ('summary', 'deep')
        """
    )
    op.drop_column("preferences", "answer_format")
    op.drop_column("preferences", "depth")
    op.drop_column("preferences", "max_table_rows")


def downgrade() -> None:
    op.add_column(
        "preferences",
        sa.Column(
            "answer_format",
            sa.String(),
            nullable=False,
            server_default=sa.text("'table'"),
        ),
    )
    op.add_column(
        "preferences",
        sa.Column(
            "depth", sa.String(), nullable=False, server_default=sa.text("'standard'")
        ),
    )
    op.add_column(
        "preferences",
        sa.Column(
            "max_table_rows",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("20"),
        ),
    )
