"""
tests/test_factual_grounding.py
================================
Pytest suite for the factual_grounding criterion. Cases are loaded from
tests/fixtures/criteria_fixtures.json.

NOTE: this criterion downloads the sentence-transformers model
(all-MiniLM-L6-v2, ~80MB) on first use. The download happens once and is
then cached locally, so subsequent runs are offline and fast. The first
run of this file may take longer than usual for that reason.
"""

import json
from pathlib import Path

import pytest

from llm_eval_kit.criteria.factual_grounding import factual_grounding

FIXTURES_PATH = Path(__file__).parent / "fixtures" / "criteria_fixtures.json"

with open(FIXTURES_PATH, "r", encoding="utf-8") as f:
    _ALL_FIXTURES = json.load(f)

GROUNDING_CASES = _ALL_FIXTURES["factual_grounding"]

# Must match MISMATCH_SCORE_CAP in factual_grounding_config.json.
# Kept here explicitly (rather than imported) so this test would fail
# loudly if someone changes the cap in config without updating the
# fixtures' expectations.
_MISMATCH_SCORE_CAP = 0.3


@pytest.mark.parametrize(
    "case",
    GROUNDING_CASES,
    ids=[c["id"] for c in GROUNDING_CASES],
)
def test_factual_grounding_fixture(case):
    result = factual_grounding(
        prompt=case["prompt"],
        response=case["response"],
        context=case["context"],
    )
    expected = case["expected"]

    if "score" in expected and expected["score"] is None:
        assert result["score"] is None, case["description"]
        return

    if "score_min" in expected:
        assert result["score"] is not None, case["description"]
        assert expected["score_min"] <= result["score"] <= expected["score_max"], (
            f"{case['description']} -- got {result['score']}"
        )

    if "score_capped" in expected:
        was_capped = result["score"] is not None and result["score"] <= _MISMATCH_SCORE_CAP
        assert was_capped == expected["score_capped"], (
            f"{case['description']} -- got score={result['score']}"
        )


def test_factual_grounding_no_context_returns_none():
    result = factual_grounding(prompt="Anything", response="Anything.", context="")
    assert result["score"] is None
    assert "explanation" in result


def test_factual_grounding_returns_required_keys():
    result = factual_grounding(
        prompt="What is the sky?",
        response="The sky is blue.",
        context="The sky appears blue due to Rayleigh scattering.",
    )
    assert "score" in result
    assert "explanation" in result
    assert isinstance(result["explanation"], str)


def test_factual_grounding_model_is_singleton():
    """The embedding model must load once per session, not once per
    evaluate() call -- see DESIGN.md and the Research Engineer's brief.
    Calling the criterion twice should reuse the same cached model
    object rather than reloading it from disk each time."""
    from llm_eval_kit.criteria import factual_grounding as fg_module

    factual_grounding(prompt="a", response="b", context="c")
    model_after_first_call = fg_module._model

    factual_grounding(prompt="d", response="e", context="f")
    model_after_second_call = fg_module._model

    assert model_after_first_call is model_after_second_call