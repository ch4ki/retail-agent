from pathlib import Path

import pandas as pd
import pytest

from retail_agent.safety.pii import PiiPolicy, mask_dataframe

POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "src/retail_agent/safety/policies/thelook.yaml"
)


@pytest.fixture
def policy() -> PiiPolicy:
    return PiiPolicy.from_yaml(POLICY_PATH)


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2],
            "email": ["ada@example.com", "grace@example.com"],
            "first_name": ["Ada", "Grace"],
            "street_address": ["1 Main St", "2 Oak Ave"],
            "postal_code": ["94107", "10001"],
            "age": [36, 45],
        }
    )


def test_email_is_hashed_and_stable(policy, frame):
    masked, _ = mask_dataframe(frame, policy, salt="s")
    again, _ = mask_dataframe(frame, policy, salt="s")

    assert "@" not in masked["email"].iloc[0]
    assert masked["email"].iloc[0] == again["email"].iloc[0]
    assert masked["email"].iloc[0] != masked["email"].iloc[1]


def test_different_salt_gives_different_hash(policy, frame):
    a, _ = mask_dataframe(frame, policy, salt="one")
    b, _ = mask_dataframe(frame, policy, salt="two")
    assert a["email"].iloc[0] != b["email"].iloc[0]


def test_first_name_reduced_to_initial(policy, frame):
    masked, _ = mask_dataframe(frame, policy, salt="s")
    assert list(masked["first_name"]) == ["A.", "G."]


def test_street_address_column_is_dropped(policy, frame):
    masked, report = mask_dataframe(frame, policy, salt="s")
    assert "street_address" not in masked.columns
    assert "street_address" in report.dropped_columns


def test_postal_code_truncated(policy, frame):
    masked, _ = mask_dataframe(frame, policy, salt="s")
    assert list(masked["postal_code"]) == ["941…", "100…"]


def test_allowed_columns_untouched(policy, frame):
    masked, _ = mask_dataframe(frame, policy, salt="s")
    assert list(masked["id"]) == [1, 2]
    assert list(masked["age"]) == [36, 45]


def test_original_frame_is_not_mutated(policy, frame):
    mask_dataframe(frame, policy, salt="s")
    assert frame["email"].iloc[0] == "ada@example.com"
    assert "street_address" in frame.columns


def test_report_counts_masked_cells(policy, frame):
    _, report = mask_dataframe(frame, policy, salt="s")
    # 2 emails + 2 first names + 2 postal codes + 2 dropped address cells
    assert report.redactions == 8


def test_nulls_survive_masking(policy):
    frame = pd.DataFrame({"email": ["a@b.com", None]})
    masked, _ = mask_dataframe(frame, policy, salt="s")
    assert masked["email"].isna().iloc[1]


def test_column_matching_is_case_insensitive(policy):
    frame = pd.DataFrame({"EMAIL": ["a@b.com"]})
    masked, _ = mask_dataframe(frame, policy, salt="s")
    assert "@" not in masked["EMAIL"].iloc[0]


def test_unknown_columns_pass_through(policy):
    frame = pd.DataFrame({"sale_price": [12.5]})
    masked, report = mask_dataframe(frame, policy, salt="s")
    assert list(masked["sale_price"]) == [12.5]
    assert report.redactions == 0


def test_restricted_columns_exposes_non_allow_rules(policy):
    restricted = policy.restricted_columns()
    assert "email" in restricted
    assert "id" not in restricted


def test_empty_frame_is_handled(policy):
    masked, report = mask_dataframe(pd.DataFrame(), policy, salt="s")
    assert masked.empty
    assert report.redactions == 0
