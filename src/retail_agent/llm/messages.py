"""Normalising model replies to text.

Providers disagree on the shape of `AIMessage.content`. OpenAI returns a plain
string; Gemini returns a list of content blocks, each carrying a thinking
signature. Calling `str()` on the latter yields a Python repr of a list, which
downstream code then tries to parse as SQL.

Every reader of model output goes through this function.
"""

from __future__ import annotations

from typing import Any


def message_text(message: Any) -> str:
    """Return the text of a model reply, whatever content shape it arrived in."""
    content = getattr(message, "content", message)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        return "".join(_block_text(block) for block in content).strip()

    return str(content).strip()


def _block_text(block: Any) -> str:
    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        # Blocks without a type are assumed to be text; anything explicitly
        # typed as something else (thinking, tool_use, image) is not output.
        if block.get("type", "text") != "text":
            return ""
        text = block.get("text")
        return text if isinstance(text, str) else ""
    return ""
