"""
tests/test_performance.py
===========================
Performance benchmark, per DESIGN.md Section 5.4 and the 2-second
target in Section 8.

Two things are checked, matching the design doc exactly:

1. A full Evaluator.evaluate() call across all currently registered
   criteria must complete in under 2 seconds. This measures actual
   evaluation time (Evaluator's own "evaluation_time_ms" metadata),
   not the one-time cost of downloading/loading the embedding model --
   see DESIGN.md Section 3.1 / Risk 2: model loading is a one-time
   per-session cost, not a per-evaluate() cost.

2. A regression test (DESIGN.md Section 5.4 / Risk 2) asserting the
   embedding model singleton is not reloaded across repeated
   evaluate() calls in the same session. If this ever breaks, the
   library silently gets much slower and the 2-second budget above
   would eventually be blown -- this test is what catches that early.

NOTE: this test currently benchmarks whichever criteria are registered
at run time (today: refusal_check, factual_grounding). Per the registry
pattern (registry.py / DESIGN.md Section 3.2), once relevance and
completeness are added and imported, they'll automatically be included
here too -- no change needed to this file.
"""

import time

from llm_eval_kit.evaluator import Evaluator
from llm_eval_kit.registry import CRITERIA_REGISTRY

PERFORMANCE_TARGET_MS = 2000


def test_full_evaluation_under_two_second_target():
    """A full evaluate() call across all registered criteria must finish
    in under 2 seconds, per DESIGN.md Section 8's performance target."""
    ev = Evaluator()

    # Warm-up call: triggers the one-time embedding model load, so it
    # doesn't get counted against the 2-second budget for a single
    # evaluate() call (see module docstring).
    ev.evaluate(
        prompt="Warm-up prompt.",
        response="Warm-up response.",
        context="Warm-up context sentence.",
    )

    result = ev.evaluate(
        prompt="What causes inflation?",
        response="Inflation is caused by excess money supply.",
        context=(
            "Inflation occurs when the general price level rises. "
            "Common causes include excess money supply and demand-pull factors."
        ),
    )

    elapsed_ms = result["metadata"]["evaluation_time_ms"]
    assert elapsed_ms < PERFORMANCE_TARGET_MS, (
        f"evaluate() took {elapsed_ms}ms, over the {PERFORMANCE_TARGET_MS}ms target "
        f"(criteria run: {list(result['criteria'].keys())})"
    )


def test_full_evaluation_under_two_seconds_wall_clock():
    """Same check as above, but measured with an independent wall-clock
    timer around the call itself, as a cross-check against Evaluator's
    self-reported timing."""
    ev = Evaluator()
    ev.evaluate(prompt="Warm-up.", response="Warm-up.", context="Warm-up.")

    start = time.perf_counter()
    ev.evaluate(
        prompt="What causes inflation?",
        response="Inflation is caused by excess money supply.",
        context="Inflation occurs due to excess money supply and demand-pull factors.",
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert elapsed_ms < PERFORMANCE_TARGET_MS, (
        f"evaluate() wall-clock time was {elapsed_ms:.0f}ms, over the "
        f"{PERFORMANCE_TARGET_MS}ms target"
    )


def test_repeated_evaluate_calls_do_not_reload_model():
    """Regression test per DESIGN.md Section 5.4 / Risk 2: the embedding
    model singleton must not be reloaded on each evaluate() call. If it
    were, evaluation time would grow with every call instead of staying
    roughly constant, and the 2-second budget above would eventually be
    broken as usage scales."""
    from llm_eval_kit.criteria import factual_grounding as fg_module

    ev = Evaluator()

    ev.evaluate(prompt="a", response="b", context="c")
    model_after_first = fg_module._model
    assert model_after_first is not None

    for _ in range(3):
        ev.evaluate(prompt="a", response="b", context="c")

    model_after_repeated_calls = fg_module._model
    assert model_after_first is model_after_repeated_calls


def test_benchmark_covers_all_currently_registered_criteria():
    """Sanity check that this benchmark isn't silently under-testing --
    it should exercise every criterion currently in CRITERIA_REGISTRY,
    not a hardcoded subset that falls out of date as criteria are added."""
    ev = Evaluator()
    result = ev.evaluate(
        prompt="What causes inflation?",
        response="Inflation is caused by excess money supply.",
        context="Inflation occurs due to excess money supply and demand-pull factors.",
    )
    assert set(result["criteria"].keys()) == set(CRITERIA_REGISTRY.keys())