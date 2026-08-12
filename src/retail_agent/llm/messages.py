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


def chunk_text(message: Any) -> str:
    """The text of one streamed chunk, whatever content shape it arrived in.

    Same content-block handling as `message_text` — a chunk's `content` is
    string on OpenAI, a list of blocks on Gemini, exactly like a whole
    message's — but deliberately without the `.strip()`. `message_text`'s
    strip is correct for a whole reply; applied per chunk it eats the leading
    or trailing space every chunk but the first and last actually carries, so
    concatenating stripped chunks glues words together with no space between
    them. Only `_stream_turn` should call this; every other reader wants the
    normalised, stripped text `message_text` gives.
    """
    content = getattr(message, "content", message)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(_block_text(block) for block in content)

    return str(content)


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
