from retail_agent.safety.egress import scan_text


def test_clean_text_passes_through_unchanged():
    text = "Revenue in Q1 was $1.2M across 4,300 orders."
    result = scan_text(text)
    assert result.text == text
    assert result.findings == ()


def test_email_is_redacted():
    result = scan_text("Top customer is ada@example.com with $900 spend.")
    assert "ada@example.com" not in result.text
    assert "[redacted:email]" in result.text
    assert "email" in result.findings


def test_phone_number_is_redacted():
    result = scan_text("Call them on (415) 555-0132 to follow up.")
    assert "555-0132" not in result.text
    assert "phone" in result.findings


def test_coordinates_are_redacted():
    result = scan_text("Customer located at 37.7749, -122.4194 in the Bay Area.")
    assert "37.7749" not in result.text
    assert "coordinates" in result.findings


def test_street_address_is_redacted():
    result = scan_text("Ships to 1600 Pennsylvania Avenue next week.")
    assert "Pennsylvania Avenue" not in result.text
    assert "street_address" in result.findings


def test_hashed_identifier_is_not_flagged():
    result = scan_text("Customer a1b2c3d4e5 placed 12 orders.")
    assert result.findings == ()


def test_money_and_dates_are_not_flagged():
    result = scan_text("On 2025-03-14 revenue hit 1,234.56 across 3 regions.")
    assert result.findings == ()


def test_masked_initials_are_not_flagged():
    result = scan_text("Top customers: A. D. (36, TX) and G. H. (45, CA).")
    assert result.findings == ()


def test_multiple_findings_are_all_reported():
    result = scan_text("Reach ada@example.com or (415) 555-0132.")
    assert set(result.findings) == {"email", "phone"}


def test_empty_text_is_safe():
    result = scan_text("")
    assert result.text == ""
    assert result.findings == ()
