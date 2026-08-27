"""
tests/test_criteria_all.py
==========================
Unit tests for all 4 registered criteria:
  1. refusal_check
  2. factual_grounding
  3. relevance
  4. completeness
And end-to-end evaluation using Evaluator.
"""

import pytest
import llm_eval_kit
from llm_eval_kit.registry import CRITERIA_REGISTRY
from llm_eval_kit.evaluator import Evaluator
from llm_eval_kit.criteria.relevance import relevance
from llm_eval_kit.criteria.completeness import completeness, _extract_aspects


def test_all_four_criteria_registered():
    """Verify all 4 core criteria are automatically registered on package import."""
    expected = {"refusal_check", "factual_grounding", "relevance", "completeness"}
    registered = set(CRITERIA_REGISTRY.keys())
    assert expected.issubset(registered), f"Missing criteria: {expected - registered}"


class TestRelevanceCriterion:
    """Tests for relevance criterion."""

    def test_high_relevance(self):
        prompt = "What is the capital of France?"
        response = "The capital of France is Paris."
        result = relevance(prompt=prompt, response=response)
        assert result["score"] >= 0.70
        assert "highly relevant" in result["explanation"]

    def test_low_relevance(self):
        prompt = "What is the capital of France?"
        response = "Quantum entanglement occurs when pairs of particles interact."
        result = relevance(prompt=prompt, response=response)
        assert result["score"] < 0.40
        assert "low semantic relevance" in result["explanation"]

    def test_relevance_empty_response(self):
        result = relevance(prompt="Query", response="")
        assert result["score"] == 0.0
        assert "Empty response" in result["explanation"]

    def test_relevance_empty_prompt(self):
        result = relevance(prompt="", response="Some answer")
        assert result["score"] == 0.0
        assert "Empty prompt" in result["explanation"]


class TestCompletenessCriterion:
    """Tests for completeness criterion."""

    def test_aspect_extraction(self):
        prompt = "What causes climate change, and how can we reduce global emissions?"
        aspects = _extract_aspects(prompt)
        assert len(aspects) >= 2

    def test_full_completeness(self):
        prompt = "What causes climate change and how can we reduce it?"
        response = (
            "Climate change is caused by greenhouse gas emissions and fossil fuel combustion. "
            "We can reduce it by transitioning to renewable solar energy and reforestation."
        )
        result = completeness(prompt=prompt, response=response)
        assert result["score"] == 1.0
        assert result["total_aspects"] >= 2
        assert len(result["covered_aspects"]) == result["total_aspects"]

    def test_partial_completeness(self):
        prompt = "What causes climate change and how can we reduce it?"
        response = "Climate change is caused by greenhouse gas emissions from burning fossil fuels."
        result = completeness(prompt=prompt, response=response)
        assert 0.0 < result["score"] < 1.0
        assert len(result["covered_aspects"]) < result["total_aspects"]

    def test_completeness_empty_response(self):
        result = completeness(prompt="Explain topic A and topic B", response="")
        assert result["score"] == 0.0
        assert len(result["covered_aspects"]) == 0

    def test_completeness_empty_prompt(self):
        result = completeness(prompt="", response="Answer")
        assert result["score"] == 0.0


class TestEvaluatorEndToEnd:
    """Tests Evaluator running all 4 criteria end-to-end."""

    def test_evaluator_all_four_criteria(self):
        evaluator = Evaluator()
        prompt = "What is the boiling point of water and why does it occur?"
        response = "Water boils at 100 degrees Celsius when vapor pressure equals atmospheric pressure."
        context = "Water boils at 100°C at standard atmospheric pressure."

        result = evaluator.evaluate(
            prompt=prompt,
            response=response,
            context=context,
        )

        assert "overall_score" in result
        assert result["overall_score"] is not None
        assert "refusal_check" in result["criteria"]
        assert "factual_grounding" in result["criteria"]
        assert "relevance" in result["criteria"]
        assert "completeness" in result["criteria"]
        assert result["metadata"]["response_length_words"] > 0
        assert result["metadata"]["evaluation_time_ms"] >= 0
