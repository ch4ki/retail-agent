"""What a column's values mean to the business, as opposed to what they are.

Showing the model a column's actual values fixed one class of bug and created
another. It stopped writing `WHERE gender = 'female'` against a column holding
'F' — but it started writing `WHERE status = 'Complete'`, which is one of five
statuses, where a completed order means every status except Cancelled and
Returned. That undercounted 93,893 orders as 31,303, and it is a reasonable
reading: the word "completed" is right there in the value list.

Values alone cannot say that. The convention has to sit beside them.

These are facts about theLook, so they live with the other business knowledge
rather than in `prompts.py`. A rule in the SQL prompt applies to every query
whether or not it touches `status`, and that prompt is already a rulebook we
keep having to defend. A note here is only rendered when the column is in the
schema the model is reading.

This is the hand-written floor. The Golden Bucket carries the same conventions
in its trio metric definitions, but those only reach the prompt when a trio is
retrieved, and "how many orders were completed" may retrieve none.
"""

from __future__ import annotations

# The same convention governs both status columns: the agent joins orders to
# order_items freely, and a rule that held on one but not the other would be
# worse than no rule.
_COMPLETED = (
    "A completed order means status NOT IN ('Cancelled', 'Returned'). "
    "'Processing' and 'Shipped' count as completed — do not filter to "
    "status = 'Complete' alone, which is only one of the five statuses."
)

COLUMN_NOTES: dict[tuple[str, str], str] = {
    ("orders", "status"): _COMPLETED,
    ("order_items", "status"): _COMPLETED,
    ("order_items", "sale_price"): (
        "Revenue is SUM(sale_price) over order_items. There is no revenue "
        "column on orders."
    ),
}


def notes_for(table: str) -> dict[str, str]:
    """The conventions for one table, keyed by column. Empty when there are none."""
    return {
        column: note
        for (owner, column), note in COLUMN_NOTES.items()
        if owner == table
    }
