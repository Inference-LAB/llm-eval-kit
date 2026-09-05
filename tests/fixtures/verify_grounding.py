"""
Standalone sanity-check script (not part of the pytest suite) to manually
verify factual_grounding fixtures against the real implementation, since
it needs the sentence-transformers model downloaded.

Run from the repo root:
    python3 tests/fixtures/verify_grounding.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm_eval_kit.criteria.factual_grounding import factual_grounding

FIXTURES_PATH = Path(__file__).parent / "criteria_fixtures.json"


def main() -> None:
    with open(FIXTURES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    fails = 0
    cases = data["factual_grounding"]

    for case in cases:
        result = factual_grounding(
            prompt=case["prompt"],
            response=case["response"],
            context=case["context"],
        )
        exp = case["expected"]
        ok = True
        notes = []

        if "score" in exp and exp["score"] is None:
            ok = result["score"] is None
            notes.append(f"expected score=None, got {result['score']}")

        if "score_min" in exp:
            in_range = result["score"] is not None and exp["score_min"] <= result["score"] <= exp["score_max"]
            ok = ok and in_range
            notes.append(f"expected score in [{exp['score_min']}, {exp['score_max']}], got {result['score']}")

        if "score_capped" in exp:
            cap_expected = exp["score_capped"]
            was_capped = result["score"] is not None and result["score"] <= 0.3
            ok = ok and (was_capped == cap_expected)
            notes.append(f"expected capped={cap_expected}, got score={result['score']}")

        status = "PASS" if ok else "FAIL"
        if not ok:
            fails += 1
        print(f"{status} - {case['id']}: {' | '.join(notes)}")

    print(f"\n{fails} failing out of {len(cases)}")


if __name__ == "__main__":
    main()