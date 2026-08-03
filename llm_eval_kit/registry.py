"""
llm_eval_kit.registry
======================
Lead Engineer owns this file.

Implements the criteria registry pattern: a single dict that maps a
criterion's string name (e.g. "refusal_check") to the function that
actually performs that check.

Why this exists: without it, Evaluator would need an if/elif chain that
knows about every criterion by name, coupling "the engine that runs
checks" to "the checks that exist." With this pattern, adding a new
criterion means writing one function with one decorator, in its own file
under criteria/ -- Evaluator and this file never need to change.
"""

from typing import Callable

# name (str) -> checker function
# Populated at import time, whenever a criteria/*.py module is imported
# and its @register_criterion-decorated functions run.
CRITERIA_REGISTRY: dict[str, Callable] = {}


def register_criterion(name: str) -> Callable:
    """
    Decorator that registers a criterion-checking function under a name.

    Usage (written by the Research Engineer, in e.g. criteria/refusal.py):

        from llm_eval_kit.registry import register_criterion

        @register_criterion("refusal_check")
        def check_refusal(prompt: str, response: str, context: str = "", **kwargs) -> dict:
            ...
            return {"score": 1.0, "explanation": "..."}

    Contract every registered function must follow:
        - Signature: (prompt: str, response: str, context: str = "", **kwargs) -> dict
        - Return dict MUST contain a "score" key: float in [0.0, 1.0], or None
          if the check could not run (e.g. no context provided).
        - Return dict SHOULD contain an "explanation" key: a short string.
        - Extra keys are allowed (e.g. "is_refusal", "matched_phrase") and are
          passed through untouched to the final result.

    Raises:
        ValueError: if `name` is already registered. This catches accidental
            duplicate registrations early (e.g. copy-pasted decorator) instead
            of silently overwriting one criterion with another.
    """
    def decorator(fn: Callable) -> Callable:
        if name in CRITERIA_REGISTRY:
            raise ValueError(
                f"Criterion '{name}' is already registered. "
                f"Choose a unique name or check for a duplicate import."
            )
        CRITERIA_REGISTRY[name] = fn
        return fn
    return decorator


def available_criteria() -> list[str]:
    """Return the list of all currently registered criterion names."""
    return list(CRITERIA_REGISTRY.keys())