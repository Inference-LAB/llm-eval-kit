"""
tests/test_edge_cases.py
========================
Edge case hardening tests:
  1. Empty / whitespace / None response strings across all criteria.
  2. 5,000-word context sentence-splitting performance and embedding verification.
  3. Urdu refusal phrase detection, Urdu sentence punctuation, and numeral handling.
"""

import time
import math
import pytest
from llm_eval_kit.criteria.refusal import refusal_check
from llm_eval_kit.criteria.factual_grounding import factual_grounding, _split_sentences
from llm_eval_kit.criteria.numeric_utils import extract_numeric_claims, has_numeric_mismatch


class TestEmptyResponses:
    """Tests empty, whitespace, and None input handling across criteria."""

    def test_refusal_empty_string(self):
        result = refusal_check(prompt="Write code", response="")
        assert result["score"] == 0.0
        assert result["is_refusal"] is True
        assert "Empty response" in result["explanation"]

    def test_refusal_whitespace_string(self):
        result = refusal_check(prompt="Write code", response="   \n\t  ")
        assert result["score"] == 0.0
        assert result["is_refusal"] is True

    def test_refusal_none_response(self):
        result = refusal_check(prompt="Write code", response=None)
        assert result["score"] == 0.0
        assert result["is_refusal"] is True

    def test_factual_grounding_empty_response_with_context(self):
        result = factual_grounding(prompt="Query", response="", context="Water boils at 100°C.")
        assert result["score"] == 0.0
        assert "Empty response" in result["explanation"]
        assert result["best_matching_sentence"] is None

    def test_factual_grounding_whitespace_response(self):
        result = factual_grounding(prompt="Query", response="   ", context="Water boils at 100°C.")
        assert result["score"] == 0.0
        assert result["best_matching_sentence"] is None

    def test_factual_grounding_none_response_with_context(self):
        result = factual_grounding(prompt="Query", response=None, context="Water boils at 100°C.")
        assert result["score"] == 0.0
        assert result["best_matching_sentence"] is None

    def test_factual_grounding_empty_context_takes_precedence(self):
        result = factual_grounding(prompt="Query", response="", context="")
        assert result["score"] is None
        assert "No context provided" in result["explanation"]


class TestLargeContextBenchmark:
    """Tests 5,000-word context scaling, regex sentence splitting latency, and factual evaluation."""

    @pytest.fixture
    def large_5000_word_context(self):
        sentences = []
        for i in range(200):
            s1 = f"Project unit number {i} operates in industrial zone {i % 12} with an annual budget of {i * 10} thousand dollars."
            s2 = f"Milestone verification was completed successfully during fiscal year {1990 + (i % 30)} under protocol {i}."
            sentences.extend([s1, s2])
        full_text = " ".join(sentences)
        assert len(full_text.split()) >= 5000
        return full_text

    def test_sentence_splitting_speed_5000_words(self, large_5000_word_context):
        t0 = time.perf_counter()
        segments = _split_sentences(large_5000_word_context)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert len(segments) >= 250
        assert elapsed_ms < 20.0

    def test_factual_grounding_on_5000_word_context(self, large_5000_word_context):
        target_response = "Project unit number 50 operates in industrial zone 2 with an annual budget of 500 thousand dollars."
        result = factual_grounding(
            prompt="Tell me about project unit 50.",
            response=target_response,
            context=large_5000_word_context,
        )

        assert result["score"] is not None
        assert result["score"] > 0.5
        assert "Project unit number 50" in result["best_matching_sentence"]


class TestUrduLanguageHandling:
    """Tests refusal detection, punctuation segmentation, and numeral parsing on Urdu text."""

    @pytest.mark.parametrize("urdu_refusal", [
        "میں اس درخواست میں مدد نہیں کر سکتا۔",
        "معذرت، میں یہ نہیں کر سکتا۔",
        "میں اس درخواست پر عمل نہیں کر سکتا۔",
        "ایک AI ماڈل کے طور پر، میں یہ کام نہیں کر سکتا۔",
        "معاف کیجیے گا، میں یہ معلومات فراہم نہیں کر سکتا۔",
    ])
    def test_urdu_refusal_detection(self, urdu_refusal):
        result = refusal_check(prompt="کچھ برا لکھیں", response=urdu_refusal)
        assert result["is_refusal"] is True
        assert result["score"] == 0.0

    def test_urdu_substantive_response(self):
        response = "پیرس فرانس کا دارالحکومت ہے۔ یہ ایک خوبصورت شہر ہے۔"
        result = refusal_check(prompt="فرانس کا دارالحکومت کیا ہے؟", response=response)
        assert result["is_refusal"] is False
        assert result["score"] == 1.0

    def test_urdu_sentence_splitting_punctuation(self):
        urdu_text = "یہ پہلا جملہ ہے۔ یہ دوسرا جملہ ہے؟ کیا یہ تیسرا جملہ ہے!"
        segments = _split_sentences(urdu_text)
        assert len(segments) == 3
        assert segments[0] == "یہ پہلا جملہ ہے۔"
        assert segments[1] == "یہ دوسرا جملہ ہے؟"
        assert segments[2] == "کیا یہ تیسرا جملہ ہے!"

    def test_urdu_eastern_arabic_numerals_extraction(self):
        claims = extract_numeric_claims("درجہ حرارت ۱۰۰ ڈگری سینٹی گریڈ ہے۔")
        assert len(claims) == 1
        val, _ = claims[0]
        assert math.isclose(val, 100.0)

    def test_urdu_eastern_arabic_numerals_grounding_match(self):
        context = ["پانی ۱۰۰ ڈگری پر ابلتا ہے۔"]
        response = "پانی 100 ڈگری پر ابلتا ہے۔"
        assert not has_numeric_mismatch(response, context)
