"""
tests/test_relevance_completeness.py
====================================
Unit tests for relevance and completeness evaluation criteria.
"""

import pytest
from llm_eval_kit.criteria.relevance import relevance
from llm_eval_kit.criteria.completeness import completeness, _extract_aspects


class TestRelevanceCriterion:
    """Unit tests for relevance criterion."""

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
    """Unit tests for completeness criterion."""

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
