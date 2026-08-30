"""
tests/test_registry.py
========================
Pytest suite for llm_eval_kit/registry.py -- the criteria registry
pattern itself (registering, looking up, and guarding against
duplicate criterion names).
"""

import pytest

from llm_eval_kit.registry import (
    CRITERIA_REGISTRY,
    register_criterion,
    available_criteria,
)


def test_register_criterion_adds_to_registry():
    @register_criterion("_test_dummy_criterion")
    def _dummy(prompt, response, context="", **kwargs):
        return {"score": 1.0, "explanation": "stub"}

    try:
        assert "_test_dummy_criterion" in CRITERIA_REGISTRY
        assert CRITERIA_REGISTRY["_test_dummy_criterion"] is _dummy
    finally:
        del CRITERIA_REGISTRY["_test_dummy_criterion"]


def test_register_criterion_rejects_duplicate_name():
    @register_criterion("_test_dup_criterion")
    def _first(prompt, response, context="", **kwargs):
        return {"score": 1.0}

    try:
        with pytest.raises(ValueError):
            @register_criterion("_test_dup_criterion")
            def _second(prompt, response, context="", **kwargs):
                return {"score": 0.0}
    finally:
        del CRITERIA_REGISTRY["_test_dup_criterion"]


def test_available_criteria_returns_list_of_registered_names():
    @register_criterion("_test_listed_criterion")
    def _dummy(prompt, response, context="", **kwargs):
        return {"score": 1.0}

    try:
        names = available_criteria()
        assert isinstance(names, list)
        assert "_test_listed_criterion" in names
    finally:
        del CRITERIA_REGISTRY["_test_listed_criterion"]


def test_registered_function_is_still_directly_callable():
    """The decorator must return the original function unchanged, so it
    can still be imported and called directly (not just through the
    registry) -- e.g. by other tests that import refusal_check directly."""

    @register_criterion("_test_still_callable")
    def _dummy(prompt, response, context="", **kwargs):
        return {"score": 0.42, "explanation": "direct call"}

    try:
        result = _dummy(prompt="x", response="y")
        assert result["score"] == 0.42
    finally:
        del CRITERIA_REGISTRY["_test_still_callable"]