"""
llm_eval_kit.evaluator
========================
Lead Engineer owns this file.

Implements the Evaluator class -- the main entry point a user of this
library actually calls. Evaluator never knows how any individual criterion
works; it only knows how to look criteria up by name in CRITERIA_REGISTRY
and call them the same way, every time.
"""

import time
from llm_eval_kit.registry import CRITERIA_REGISTRY


class Evaluator:
    def evaluate(
        self,
        prompt: str,
        response: str,
        context: str = "",
        criteria: list[str] = None,
    ) -> dict:
        """
        Evaluate an LLM response against structured criteria.

        Args:
            prompt:   The original prompt sent to the LLM
            response: The LLM's response to evaluate
            context:  Optional reference text for grounding checks
            criteria: List of criterion names. Defaults to all registered
                      criteria if not given.

        Returns:
            dict with overall_score, per-criterion results, and metadata.

        Raises:
            ValueError: if any requested criterion name is not registered.
                Raised before any checker runs (fail-fast) -- we never want
                to burn compute on three checks only to crash on the fourth
                because of a typo.
        """
        if criteria is None:
            criteria = list(CRITERIA_REGISTRY.keys())

        # Validate all criterion names before running any of them.
        unknown = [c for c in criteria if c not in CRITERIA_REGISTRY]
        if unknown:
            raise ValueError(
                f"Unknown criteria: {unknown}. "
                f"Available: {list(CRITERIA_REGISTRY.keys())}"
            )

        start = time.time()
        results = {}
        for name in criteria:
            checker = CRITERIA_REGISTRY[name]
            results[name] = checker(prompt=prompt, response=response, context=context)

        # Criteria that return score: None (e.g. no context given) are
        # excluded from the average, not counted as 0 -- see design doc
        # Section 3.3 for why this is a deliberate choice, not an oversight.
        scores = [r["score"] for r in results.values() if r.get("score") is not None]
        overall = round(sum(scores) / len(scores), 3) if scores else None

        return {
            "overall_score": overall,
            "criteria": results,
            "metadata": {
                "response_length_words": len(response.split()),
                "evaluation_time_ms": round((time.time() - start) * 1000),
            },
        }