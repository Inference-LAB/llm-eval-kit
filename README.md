# llm-eval-kit — Offline LLM Response Evaluation Library

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`llm-eval-kit` is a lightweight, fully offline Python library designed to evaluate the quality of LLM responses. It runs evaluation checks locally—requiring **no external APIs, no internet connection (after setup), and no paid subscriptions**.

Designed to fit seamlessly into CI/CD pipelines, this library helps automate prompt validation and model upgrades by checking response grounding, relevance, completeness, and refusal detection.

---

## Key Features

- **100% Local & Private:** Runs entirely on your own machine using a shared local embedding model (`all-MiniLM-L6-v2`) via `sentence-transformers`.
- **Extensible Registry Architecture:** Adding a new criterion requires no modification to the core engine; simply use the `@register_criterion` decorator.
- **Robust Grounding Check:** Combines semantic embedding similarity with a specialized numeric sanity check to catch mismatched values or units (e.g., distinguishing between "50°C" and "100°C").
- **Smart Refusal Detection:** Rule-based checks with context awareness to prevent false positives (e.g., distinguishing an actual refusal from conversational text like *"I cannot stress enough how important this is"*).
- **Graceful Error & Skip Handling:** Skipped checks (e.g., when context is omitted for grounding) are omitted from the overall average rather than counted as zero.
- **Fail-Fast CLI & API:** Validates all evaluation criteria names before invoking any computationally heavy embeddings.

---

## Installation

Install the package via `pip` (requires Python 3.9+):

```bash
pip install llm-eval-kit
```

*Note: On first run, the library will automatically download the local embedding model `all-MiniLM-L6-v2` (approx. 90MB) to your local cache. All subsequent executions run entirely offline.*

---

## Quick Start (Python API)

Here is a 4-line example demonstrating how to run the evaluator:

```python
from llm_eval_kit import Evaluator

# Initialize the evaluator
evaluator = Evaluator()

# Define your inputs
prompt = "What is the boiling point of water at sea level?"
response = "Water boils at 100 degrees Celsius at sea level."
context = "At sea level, the boiling point of water is 100°C (212°F)."

# Run the evaluation (specify criteria or omit to run all registered criteria)
result = evaluator.evaluate(
    prompt=prompt,
    response=response,
    context=context,
    criteria=["refusal_check", "factual_grounding", "relevance", "completeness"]
)

# Output is a structured Python dictionary
import json
print(json.dumps(result, indent=2))
```

### Example Output JSON
```json
{
  "overall_score": 0.95,
  "criteria": {
    "refusal_check": {
      "score": 1.0,
      "is_refusal": false,
      "explanation": "No refusal phrase detected near the start of the response."
    },
    "factual_grounding": {
      "score": 0.88,
      "explanation": "Score reflects semantic similarity with a numeric-consistency penalty applied, not calibrated factual accuracy. Response numeric claims match or are supported by the context.",
      "best_matching_sentence": "At sea level, the boiling point of water is 100°C (212°F)."
    },
    "relevance": {
      "score": 0.92,
      "explanation": "Response is highly relevant and directly addresses the prompt topic."
    },
    "completeness": {
      "score": 1.0,
      "explanation": "Response covers all key aspects requested in the prompt.",
      "covered_aspects": ["boiling point of water at sea level"],
      "total_aspects": 1
    }
  },
  "metadata": {
    "response_length_words": 9,
    "evaluation_time_ms": 142
  }
}
```

---

## CLI Usage

The library includes a thin, developer-friendly command-line wrapper `llm-eval` for easy integration into test runners and pipelines.

```bash
llm-eval evaluate \
  --prompt "What is the capital of France?" \
  --response "Paris is the capital of France." \
  --context "Paris is the capital and most populous city of France." \
  --criteria refusal_check \
  --criteria factual_grounding
```

### Options:
- `--prompt` (Required): The original prompt sent to the LLM.
- `--response` (Required): The LLM response to evaluate.
- `--context` (Optional): Reference text for grounding.
- `--criteria` (Optional): Repeating option to choose specific criteria (defaults to all registered criteria).

---

## Evaluation Criteria Details

### 1. Factual Grounding (`factual_grounding`)
Measures whether the response's claims are semantically aligned and supported by the provided context.
- **Numeric Validation:** Uses `numeric_utils.py` to extract spelled-out numbers (e.g., *"ten million"*), digits, decimals, negatives, and unit synonyms (e.g., *"degrees Celsius"* vs. *"°C"*). If the response asserts numbers or units not present in the context, a penalty caps the score (default cap: `0.3`).
- **Limitation:** Embedding similarity captures semantic closeness, not absolute factual truth. A subtle negation or unit-less mismatch may still yield high similarity unless checked numerically.

### 2. Refusal Detection (`refusal_check`)
Identifies if the model declined to answer the prompt.
- **Smart Tail Filtering:** If a refusal-like phrase is detected at the start of a response but is followed by a substantial tail of actual content (exceeding `120` characters), it is treated as incidental language rather than a refusal.
- **Configurable:** Refusal phrase lists, checking windows, and tail thresholds are loaded from `refusal_config.json`. Supports multilingual variants (including Urdu refusal heuristics).

### 3. Relevance (`relevance`)
Evaluates semantic relevance between the prompt and the response.
- **Dense Cosine Similarity:** Encodes the prompt and response with `all-MiniLM-L6-v2` and computes cosine similarity.
- **Defensive Pre-Encoding:** Handles empty or whitespace prompts/responses safely without unnecessary transformer calls.

### 4. Completeness (`completeness`)
Evaluates whether all sub-questions or clauses in the user's prompt were addressed by the response.
- **Aspect Extraction:** Decomposes complex prompts into semantic aspects and checks coverage against response sentences.
- **Granular Accounting:** Returns detailed breakdowns (`covered_aspects`, `total_aspects`, and explanation).

---

## Architecture & Extensibility

```
                    +------------------------+
                    |       Evaluator        |
                    +------------------------+
                                |
                                v
                    +------------------------+
                    |   CRITERIA_REGISTRY    |
                    +------------------------+
                       |         |         |
                       v         v         v
                  refusal_check  relevance  factual_grounding
```

### The Registry Pattern
`llm-eval-kit` decouples the evaluator execution from the checking logic using a lookup registry. This ensures that adding new checks requires zero modification to the `Evaluator` class.

#### How to Add a Custom Criterion:
1. Create a new file under `llm_eval_kit/criteria/my_check.py`.
2. Decorate your evaluation function with `@register_criterion("my_check")`:

```python
# llm_eval_kit/criteria/my_check.py
from llm_eval_kit.registry import register_criterion

@register_criterion("my_check")
def my_check(prompt: str, response: str, context: str = "", **kwargs) -> dict:
    # Custom checking logic
    return {
        "score": 0.95,
        "explanation": "Custom check passed successfully."
    }
```

3. Import the module inside `llm_eval_kit/__init__.py` to auto-register it at load time:
```python
from llm_eval_kit.criteria import my_check
```

---

## Known Limitations

1. **Semantic Similarity != Factual Accuracy:** Grounding checks reflect semantic alignment with the provided context passage. A response can score high simply by repeating the vocabulary and structure of the context even if a key detail is wrong (mitigated by the numeric mismatch check).
2. **Directional Refusals:** The `refusal_check` only identifies *whether* a refusal happened. It does not judge if the refusal was appropriate (e.g. refusing harmful prompts).
3. **Implicit Aspects:** Heuristic aspect-splitting for completeness checks may miss implicit sub-questions in complex prompts.
