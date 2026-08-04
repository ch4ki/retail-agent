"""Declarative PII masking, applied to query results before the model sees them.

Masking happens at the data boundary rather than on the final answer. The model
never receives an email address, so no prompt can make it emit one.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml

Action = Literal["allow", "hash", "initial", "truncate", "drop"]

DEFAULT_POLICY_PATH = Path(__file__).parent / "policies" / "thelook.yaml"


@dataclass(frozen=True)
class ColumnRule:
    action: Action
    keep: int = 0


@dataclass(frozen=True)
class MaskingReport:
    redactions: int
    dropped_columns: tuple[str, ...]


class PiiPolicy:
    def __init__(self, rules: Mapping[str, ColumnRule]) -> None:
        self._rules = {name.lower(): rule for name, rule in rules.items()}

    @classmethod
    def from_yaml(cls, path: Path | str = DEFAULT_POLICY_PATH) -> "PiiPolicy":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        columns = raw.get("columns", {})
        return cls(
            {
                name: ColumnRule(action=spec["action"], keep=int(spec.get("keep", 0)))
                for name, spec in columns.items()
            }
        )

    @classmethod
    def default(cls) -> "PiiPolicy":
        return cls.from_yaml(DEFAULT_POLICY_PATH)

    def rule_for(self, column: str) -> ColumnRule | None:
        return self._rules.get(column.lower())

    def restricted_columns(self) -> frozenset[str]:
        return frozenset(
            name for name, rule in self._rules.items() if rule.action != "allow"
        )


def mask_dataframe(
    df: pd.DataFrame, policy: PiiPolicy, *, salt: str
) -> tuple[pd.DataFrame, MaskingReport]:
    """Return a masked copy of `df` plus a report. Never mutates the input."""
    masked = df.copy()
    redactions = 0
    dropped: list[str] = []

    for column in list(masked.columns):
        rule = policy.rule_for(str(column))
        if rule is None or rule.action == "allow":
            continue

        non_null = int(masked[column].notna().sum())

        if rule.action == "drop":
            masked = masked.drop(columns=[column])
            dropped.append(str(column))
            redactions += non_null
            continue

        masked[column] = masked[column].map(lambda value, r=rule: _apply(value, r, salt))
        redactions += non_null

    return masked, MaskingReport(redactions=redactions, dropped_columns=tuple(dropped))


def _apply(value: object, rule: ColumnRule, salt: str) -> object:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value

    text = str(value)
    if rule.action == "hash":
        return hashlib.sha256(f"{salt}{text}".encode()).hexdigest()[:10]
    if rule.action == "initial":
        stripped = text.strip()
        return f"{stripped[0].upper()}." if stripped else text
    if rule.action == "truncate":
        return f"{text[: rule.keep]}…" if len(text) > rule.keep else text
    return text
