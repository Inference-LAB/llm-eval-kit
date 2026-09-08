"""
tests/test_evaluator.py
========================
Pytest suite for the Evaluator class itself (llm_eval_kit/evaluator.py).
These tests cover the Evaluator's own logic -- looking up criteria,
computing overall_score, error handling -- independent of any single
criterion's internal behavior.

Uses refusal_check as a stand-in "real" criterion since it needs no
model download and is fast, and adds small fake criteria registered on
the fly to test edge cases (e.g. a criterion that returns score=None)
without depending on factual_grounding's context requirement.
"""

import pytest

from llm_eval_kit.evaluator import Evaluator
from llm_eval_kit.registry import CRITERIA_REGISTRY, register_criterion
import llm_eval_kit.criteria.refusal  # noqa: F401 -- ensures refusal_check is registered


def test_evaluate_runs_default_criteria_when_none_given():
    """If criteria=None, Evaluator should run every registered criterion."""
    ev = Evaluator()
    result = ev.evaluate(prompt="What is 2+2?", response="4")
    assert set(result["criteria"].keys()) == set(CRITERIA_REGISTRY.keys())


def test_evaluate_runs_only_requested_criteria():
    ev = Evaluator()
    result = ev.evaluate(
        prompt="What is 2+2?",
        response="4",
        criteria=["refusal_check"],
    )
    assert set(result["criteria"].keys()) == {"refusal_check"}


def test_evaluate_raises_on_unknown_criterion():
    ev = Evaluator()
    with pytest.raises(ValueError):
        ev.evaluate(prompt="x", response="y", criteria=["not_a_real_criterion"])


def test_evaluate_fails_fast_before_running_any_checker():
    """A typo in one criterion name should raise before any checker runs,
    even if other requested criteria are valid -- see evaluator.py's
    'fail-fast' docstring note."""
    ev = Evaluator()
    with pytest.raises(ValueError):
        ev.evaluate(
            prompt="x",
            response="y",
            criteria=["refusal_check", "totally_made_up_name"],
        )


def test_evaluate_metadata_contains_word_count_and_timing():
    ev = Evaluator()
    result = ev.evaluate(
        prompt="Explain gravity.",
        response="Gravity pulls objects toward each other.",
        criteria=["refusal_check"],
    )
    assert result["metadata"]["response_length_words"] == 6
    assert result["metadata"]["evaluation_time_ms"] >= 0


def test_overall_score_averages_available_scores():
    """A refusal (score 0.0) run alongside a criterion returning 1.0
    should average to 0.5."""

    @register_criterion("_test_always_one")
    def _always_one(prompt, response, context="", **kwargs):
        return {"score": 1.0, "explanation": "stub"}

    try:
        ev = Evaluator()
        result = ev.evaluate(
            prompt="x",
            response="I cannot help with that request.",
            criteria=["refusal_check", "_test_always_one"],
        )
        assert result["overall_score"] == pytest.approx(0.5)
    finally:
        del CRITERIA_REGISTRY["_test_always_one"]


def test_overall_score_excludes_none_scores_not_counts_as_zero():
    """A criterion returning score=None (e.g. missing context) must be
    excluded from the average entirely, not treated as a 0 -- see
    evaluator.py's comment referencing DESIGN.md Section 3.3."""

    @register_criterion("_test_always_none")
    def _always_none(prompt, response, context="", **kwargs):
        return {"score": None, "explanation": "not applicable"}

    try:
        ev = Evaluator()
        result = ev.evaluate(
            prompt="x",
            response="This is a normal, non-refusal response.",
            criteria=["refusal_check", "_test_always_none"],
        )
        # refusal_check alone scores 1.0 (no refusal detected); the None
        # criterion must not drag that average down toward 0.5.
        assert result["overall_score"] == pytest.approx(1.0)
    finally:
        del CRITERIA_REGISTRY["_test_always_none"]


def test_overall_score_is_none_when_all_criteria_return_none():
    @register_criterion("_test_always_none_2")
    def _always_none(prompt, response, context="", **kwargs):
        return {"score": None, "explanation": "not applicable"}

    try:
        ev = Evaluator()
        result = ev.evaluate(
            prompt="x",
            response="y",
            criteria=["_test_always_none_2"],
        )
        assert result["overall_score"] is None
    finally:
        del CRITERIA_REGISTRY["_test_always_none_2"]