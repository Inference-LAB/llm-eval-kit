# Examples & Demos

This directory contains standalone demonstration scripts illustrating core capabilities and edge-case handling in `llm-eval-kit`.

## Available Scripts

| Script | Scenario Demonstrated | Key Takeaway |
| :--- | :--- | :--- |
| [`demo_1.py`](demo_1.py) | **Scenario 1:** End-to-End Orchestration | Runs all 4 criteria (`refusal_check`, `factual_grounding`, `relevance`, `completeness`) in one call and returns aggregated structured JSON. |
| [`demo_2.py`](demo_2.py) | **Scenario 2:** Numeric Contradiction Catch | Verifies that numeric claims (e.g., `50°C` vs `100°C`) are checked against context, capping the grounding score at `0.30`. |
| [`demo_3.py`](demo_3.py) | **Scenario 3:** Refusal False-Positive Defense | Evaluates conversational text containing *"cannot"* (e.g., *"I cannot stress enough..."*) and correctly scores `1.0` (not a refusal). |
| [`demo_4.py`](demo_4.py) | **Scenario 4:** Genuine Refusal Detection | Detects true refusal phrases near the start of the response and scores `0.0`. |
| [`demo_5.py`](demo_5.py) | **Scenario 5:** Fail-Fast Registry Validation | Proves that unrecognized criteria names are caught and rejected at the registry boundary before compute is spent. |

## Running the Demos

Execute any script directly with Python:

```bash
python examples/demo_1.py
python examples/demo_2.py
python examples/demo_3.py
python examples/demo_4.py
python examples/demo_5.py
```
