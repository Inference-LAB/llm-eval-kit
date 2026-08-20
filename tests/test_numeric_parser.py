"""
tests/test_numeric_parser.py
============================
Focused unit tests for numeric extraction, scale combinations, unit normalization,
and mismatch validation logic in numeric_utils.py.
"""

import math
import pytest
from llm_eval_kit.criteria.numeric_utils import (
    extract_numeric_claims,
    claim_is_supported,
    has_numeric_mismatch,
    normalize_unit,
    read_unit,
    words_to_number,
)


class TestNegativeNumbers:
    """Test parsing of negative digits and spelled-out negative numbers."""

    def test_negative_digits(self):
        claims = extract_numeric_claims("The temperature was -40 degrees Celsius.")
        assert len(claims) == 1
        val, unit = claims[0]
        assert math.isclose(val, -40.0)
        assert unit == "celsius"

    def test_negative_spelled_out(self):
        claims = extract_numeric_claims("The low reached negative forty degrees Celsius.")
        assert len(claims) == 1
        val, unit = claims[0]
        assert math.isclose(val, -40.0)
        assert unit == "celsius"

    def test_minus_spelled_out(self):
        claims = extract_numeric_claims("It was minus fifteen degrees.")
        assert len(claims) == 1
        val, _ = claims[0]
        assert math.isclose(val, -15.0)

    def test_negative_decimal(self):
        claims = extract_numeric_claims("Net change was -3.5 percent.")
        assert len(claims) == 1
        val, unit = claims[0]
        assert math.isclose(val, -3.5)
        assert unit == "percent"


class TestSpelledOutNumbers:
    """Test parsing of word numbers up to billions."""

    def test_compound_spelled_numbers(self):
        claims = extract_numeric_claims("The town has twelve thousand three hundred residents.")
        assert len(claims) == 1
        val, unit = claims[0]
        assert math.isclose(val, 12300.0)
        assert unit == ""

    def test_spelled_number_with_and(self):
        claims = extract_numeric_claims("There are one hundred and five pages.")
        assert len(claims) == 1
        val, _ = claims[0]
        assert math.isclose(val, 105.0)

    def test_large_spelled_number(self):
        claims = extract_numeric_claims("The project cost two billion five hundred million dollars.")
        assert len(claims) == 1
        val, unit = claims[0]
        assert math.isclose(val, 2_500_000_000.0)
        assert unit == "dollars"


class TestPercentages:
    """Test percentage extraction with % sign and 'percent'/'percentage'."""

    def test_percent_symbol(self):
        claims = extract_numeric_claims("Growth was 12% year over year.")
        assert len(claims) == 1
        val, unit = claims[0]
        assert math.isclose(val, 12.0)
        assert unit == "percent"

    def test_percent_word(self):
        claims = extract_numeric_claims("Market share rose by 12 percent.")
        assert len(claims) == 1
        val, unit = claims[0]
        assert math.isclose(val, 12.0)
        assert unit == "percent"

    def test_fractional_percentage(self):
        claims = extract_numeric_claims("Error rate is 0.05%.")
        assert len(claims) == 1
        val, unit = claims[0]
        assert math.isclose(val, 0.05)
        assert unit == "percent"


class TestScientificNotation:
    """Test scientific notation parsing and float matching."""

    def test_scientific_notation_extraction(self):
        claims = extract_numeric_claims("There are 3.5e6 particles in the sample.")
        assert len(claims) == 1
        val, _ = claims[0]
        assert math.isclose(val, 3500000.0)

    def test_scientific_notation_negative_exponent(self):
        claims = extract_numeric_claims("Concentration is 1.2e-3 mol.")
        assert len(claims) == 1
        val, _ = claims[0]
        assert math.isclose(val, 0.0012)

    def test_scientific_vs_standard_match(self):
        r_claim = (3.5e6, "")
        context_claims = {(3500000.0, "")}
        assert claim_is_supported(r_claim, context_claims)


class TestScaleCombinations:
    """Test combination of digit numbers followed by scale words."""

    def test_digit_million(self):
        claims = extract_numeric_claims("The startup raised 10 million dollars.")
        assert len(claims) == 1
        val, unit = claims[0]
        assert math.isclose(val, 10_000_000.0)
        assert unit == "dollars"

    def test_decimal_billion(self):
        claims = extract_numeric_claims("Revenue exceeded 2.5 billion dollars.")
        assert len(claims) == 1
        val, unit = claims[0]
        assert math.isclose(val, 2_500_000_000.0)
        assert unit == "dollars"


class TestUnitHandlingAndSynonyms:
    """Test unit normalization, synonym equivalence, and non-unit handling."""

    def test_celsius_synonyms(self):
        assert normalize_unit("°c") == "celsius"
        assert normalize_unit("celsius") == "celsius"

    def test_compound_degrees(self):
        unit, count = read_unit(["degrees", "celsius"], 0)
        assert unit == "celsius"
        assert count == 2

    def test_non_unit_nouns_not_treated_as_units(self):
        """Ensure arbitrary nouns like 'people' or 'employees' are not parsed as units."""
        claims = extract_numeric_claims("The company employs 10 million people across 150 offices.")
        assert len(claims) == 2
        assert claims[0] == (10_000_000.0, "")
        assert claims[1] == (150.0, "")

    def test_unit_mismatch_detection(self):
        r_claim = (100.0, "fahrenheit")
        context_claims = {(100.0, "celsius")}
        assert not claim_is_supported(r_claim, context_claims)

    def test_unit_match_with_synonyms(self):
        r_claim = (100.0, "celsius")
        context_claims = {(100.0, "celsius")}
        assert claim_is_supported(r_claim, context_claims)


class TestNumericMismatchWorkflow:
    """Test multi-sentence union and mismatch detection."""

    def test_multi_sentence_valid_combination(self):
        context = [
            "The Paris branch was established in 1998.",
            "The Tokyo branch was opened in 2005.",
            "Total staff count is 150 employees.",
        ]
        response = "The company opened in 1998 and 2005, employing 150 staff."
        assert not has_numeric_mismatch(response, context)

    def test_multi_sentence_unsupported_number(self):
        context = [
            "The Paris branch was established in 1998.",
            "The Tokyo branch was opened in 2005.",
        ]
        response = "The company opened in 1998 and 2012."
        assert has_numeric_mismatch(response, context)
