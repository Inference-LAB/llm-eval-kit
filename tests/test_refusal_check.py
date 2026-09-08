"""
tests/test_refusal_check.py
============================
Pytest suite for the refusal_check criterion. Cases are loaded from
tests/fixtures/criteria_fixtures.json so the fixture data stays the
single source of truth (no duplicated expected values between the
manual verification script and this suite).
"""

import json
from pathlib import Path

import pytest

from llm_eval_kit.criteria.refusal import refusal_check

FIXTURES_PATH = Path(__file__).parent / "fixtures" / "criteria_fixtures.json"

with open(FIXTURES_PATH, "r", encoding="utf-8") as f:
    _ALL_FIXTURES = json.load(f)

REFUSAL_CASES = _ALL_FIXTURES["refusal_check"]


@pytest.mark.parametrize(
    "case",
    REFUSAL_CASES,
    ids=[c["id"] for c in REFUSAL_CASES],
)
def test_refusal_check_fixture(case):
    result = refusal_check(prompt=case["prompt"], response=case["response"])
    expected = case["expected"]

    assert result["is_refusal"] == expected["is_refusal"], case["description"]
    assert result["score"] == expected["score"], case["description"]


def test_refusal_check_returns_required_keys():
    """Every result must at least contain 'score' and 'explanation',
    per the registry contract in registry.py."""
    result = refusal_check(prompt="Anything", response="A normal answer.")
    assert "score" in result
    assert "explanation" in result
    assert isinstance(result["explanation"], str)


def test_refusal_check_score_never_none():
    """Unlike context-dependent criteria, refusal_check only needs the
    response text, so it should never return score=None."""
    result = refusal_check(prompt="", response="Some response.")
    assert result["score"] is not None