"""Final-output sweep for personal data.

This is the second line of defence. `mask_dataframe` is the first and the one
that matters: it stops PII entering model context at all. This catches the
residual case of a model inventing something that looks like real contact data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b")),
    (
        "phone",
        re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)"),
    ),
    (
        "coordinates",
        re.compile(r"(?<!\d)-?\d{1,3}\.\d{4,}\s*,\s*-?\d{1,3}\.\d{4,}(?!\d)"),
    ),
    (
        "street_address",
        re.compile(
            r"\b\d{1,5}\s+[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\s"
            r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b"
        ),
    ),
)


@dataclass(frozen=True)
class EgressResult:
    text: str
    findings: tuple[str, ...]


def scan_text(text: str) -> EgressResult:
    """Redact anything matching a PII pattern and report what was found."""
    cleaned = text
    findings: list[str] = []

    for label, pattern in PATTERNS:
        cleaned, count = pattern.subn(f"[redacted:{label}]", cleaned)
        if count:
            findings.append(label)

    return EgressResult(text=cleaned, findings=tuple(findings))
