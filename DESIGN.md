# Design Document
## llm-eval-kit — LLM Response Evaluation Library

---

## 1. Project Summary
**Section 1 · Owner: Lead Engineer — Mahrukh Baig**

llm-eval-kit is a lightweight, pip-installable Python library that checks
how good an AI's response is. You give it a prompt, the AI's response, and
optionally the source text the response should be based on. It runs four
checks factual grounding, relevance, completeness, and refusal detection 
and returns a structured score as JSON. Everything runs locally: no
internet needed after the first setup, no subscription, no external AI
needed to do the checking. It's meant to be dropped into a CI pipeline, so
a prompt change or model update can be checked automatically instead of a
person reading through answers by hand.

---

## 2. Problem Statement
**Section 2 · Owner: All three**

Anyone who builds a chatbot or an AI writing tool eventually asks the same
question: "how do I know if the model's answers are actually good?"

Right now, most people just read a few answers and guess. There's no
lightweight tool that checks an AI's response the same way every time,
without needing the internet, another AI, or a paid subscription.

llm-eval-kit fixes this. It's a small Python library you install with
`pip install llm-eval-kit`. You give it a prompt, the AI's response, and
(optionally) some source text, and it hands back a score explaining whether
the response was good and why using fast, offline, non-AI methods.

---

## 3. Architecture
**Section 3 · Owner: Lead Engineer — Mahrukh Baig**

### 3.1 Overview

Three pieces make the library work: `Evaluator`, the criteria registry, and
the four criteria functions (built by Research Engineer). `Evaluator` never
talks to a criterion function directly it only ever talks to the
registry, a lookup table connecting a criterion's name (like
`"refusal_check"`) to the function that runs it.

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

### 3.2 Criteria Registry (`registry.py`)

**What it is:** one dictionary, `CRITERIA_REGISTRY`, mapping a criterion's
name to the function that runs it. Each criterion function adds itself to
this dictionary automatically, using a decorator:
`@register_criterion("refusal_check")`.

**Why built this way:** without it, `Evaluator` would need a big
`if/elif` block checking each criterion's name directly meaning every
new criterion means editing `Evaluator`'s code. With the registry, a new
criterion is just a new file with one decorator. `Evaluator` never changes.

**Known limitation:** if a criterion file is never imported anywhere, it
silently never registers. Fix: `llm_eval_kit`'s main file will import every
criteria file automatically, so this can't happen by accident.

**How criteria are automatically discovered:** registration only happens
as a *side effect* of a criterion file being imported the decorator runs
the moment Python loads that file, not when `evaluate()` is called. To make
sure every criterion is always available without anyone manually maintaining
a list, `llm_eval_kit/__init__.py` explicitly imports every module under
`criteria/` (`refusal.py`, `factual_grounding.py`, `relevance.py`,
`completeness.py`) as soon as the package itself is imported:

```python
# llm_eval_kit/__init__.py
from llm_eval_kit.criteria import refusal, factual_grounding, relevance, completeness
```

This means a single `import llm_eval_kit` — which any user of the library
already has to do — guarantees `CRITERIA_REGISTRY` is fully populated
before `Evaluator.evaluate()` is ever called. Adding a fifth criterion later
means writing its file and adding one import line here; no other file needs
to change.

**Verified:**
```
Registered criteria: ['refusal_check', 'relevance']
refusal_check: {'score': 0.0, 'is_refusal': True, 'explanation': 'Refusal phrase detected'}
relevance: {'score': 0.75, 'explanation': 'stub for demo'}
PASS: duplicate correctly rejected -> Criterion 'relevance' is already registered.
```

### 3.3 Evaluator (`evaluator.py`)

**What it is:** the main class. `evaluate()` takes a prompt, a response, an
optional context, and a list of criteria to check. It checks all requested
criterion names are valid first (fails immediately if not), runs each one
through the registry, and averages their scores into one overall score.

**Known limitation:** a criterion that skips (returns `score: None`, e.g.
no context given) is left out of the average, not counted as 0 this needs
to be documented clearly in the README so it's not misunderstood later.

**Rules every criterion function must follow, so Evaluator can treat them
all the same way:**

| Rule | What it means |
|---|---|
| Same inputs | prompt, response, optional context |
| Must return a score | 0.0 to 1.0, or None if it couldn't run |
| Should explain itself | a short reason for the score |
| Extra info allowed | e.g. is_refusal — kept, not thrown away |
| No repeated names | duplicate registration causes an error, not a silent overwrite |

**Sample output:** calling `evaluate()` returns a structured result like this:

```json
{
  "overall_score": 0.87,
  "criteria": {
    "factual_grounding": {
      "score": 0.91,
      "explanation": "Best context match: 0.910"
    },
    "relevance": {
      "score": 0.89,
      "explanation": "Response directly addresses the prompt question"
    },
    "refusal_check": {
      "score": 1.0,
      "is_refusal": false,
      "explanation": "No refusal phrase detected"
    },
    "completeness": {
      "score": 0.68,
      "explanation": "Response covers main cause but omits cost-push and demand-pull factors"
    }
  },
  "metadata": {
    "response_length_words": 14,
    "evaluation_time_ms": 210
  }
}
```

If a criterion is skipped (e.g. `factual_grounding` with no context provided),
its entry looks like this instead, and is excluded from `overall_score`:

```json
"factual_grounding": {
  "score": null,
  "explanation": "No context provided factual grounding check skipped"
}
```

### 3.4 CLI (`cli.py`)

**What it is:** a command-line tool, `llm-eval evaluate --prompt "..."
--response "..." --criteria ...`. It does no evaluation logic itself — it
just reads the command-line input, calls `Evaluator.evaluate()`, and prints
the result as JSON. Keeping it this thin means CLI bugs and evaluation
logic bugs stay separate problems.

---

## 4. Technical Approach
**Section 4 · Owner: Research/Implementation Engineer — Muhammad Maaz**

### 4.1 Shared Embedding Model

All semantic criteria run on one shared model: sentence-transformers, using all-MiniLM-L6-v2.

- Converts text into 384-dimensional embeddings
- Lets us compare meaning, not exact wording
- Loaded once via a singleton pattern and reused across every `evaluate()` call
- Runs fully offline no API calls, no per-request cost

Singleton matters here specifically: loading the model on every call would make the library unusably slow. One load per session, shared across all four criteria.

### 4.2 Factual Grounding

**Stack:** sentence-transformers, scikit-learn (cosine_similarity)

**Why:** Measures whether the response is supported by the provided context, helping detect unsupported or hallucinated claims.

**Input:** Context, Response

**Output:** Grounding score (0–1) with an explanation

**How it works:** Context and response are embedded, then cosine similarity is computed between the two vectors. High similarity indicates the response is likely supported by the provided context.

**Important clarification (per director review):** this criterion measures
*semantic alignment* with the provided context how closely the response's
meaning matches something already stated in the context **not** whether
the response is factually correct in any absolute sense. A response can
score high here simply by sounding similar to the context, even if a key
detail (like a number) is wrong. This distinction matters and is why the
Known Limitation below exists.

**In Scope:**
- Paraphrased responses
- Context-supported factual claims
- Basic numerical inconsistencies

**Out of Scope:**
- World-knowledge fact checking
- Claims with no context to check against
- Multi-hop reasoning across multiple context passages

**Known Limitation:** Embedding similarity captures semantic closeness, not factual correctness. A response with a flipped negation or a wrong number can still score high, because the sentence structure stays semantically close.

### 4.3 Relevance

**Stack:** sentence-transformers, scikit-learn (cosine_similarity)

**Why:** Evaluates whether the response directly addresses the user's prompt.

**Input:** Prompt, Response

**Output:** Relevance score (0–1) with an explanation

**How it works:** Prompt and response are embedded using the shared model, and cosine similarity is computed between them. Higher similarity means the response stays on-topic.

**In Scope:**
- On-topic responses
- Paraphrased answers
- Semantically related responses

**Out of Scope:**
- Verifying factual accuracy
- Determining whether every required aspect of the prompt was covered

**Known Limitation:** Relevance and correctness are separate things. A response can be fully on-topic and still be wrong, or on-topic and still missing half the answer — this criterion won't catch either.

### 4.4 Completeness

**Stack:** sentence-transformers

**Why:** Measures how thoroughly the response covers the information requested in the prompt.

**Input:** Prompt, Response

**Output:** Completeness score (0–1) with an explanation

**How it works:** The prompt is decomposed into its major aspects, and the response is checked for semantic coverage of each aspect. The score reflects the proportion of requested information addressed.

**How "aspects" are identified:** the prompt is
split into candidate sub-parts using simple, rule-based heuristics 
splitting on the conjunction "and", on commas, and on question marks. For
example, "What causes climate change and how can we reduce it?" splits into
two aspects: "What causes climate change" and "how can we reduce it." Each
resulting aspect is then embedded and compared against the response using
the same similarity approach as `factual_grounding`; an aspect counts as
"covered" if its similarity to some part of the response passes a fixed
threshold (0.6). This is a heuristic, not a linguistically precise
decomposition it will not correctly split prompts with implicit,
un-conjoined sub-questions (e.g. "Explain X." with an unstated follow-up
expectation), which is why this is listed as a Known Limitation below.

**In Scope:**
- Multi-part questions
- Partial answers
- Coverage of prompt aspects

**Out of Scope:**
- Judging factual correctness
- Validating information against external knowledge

**Known Limitation:** Automatically identifying all prompt aspects is heuristic-based and may not always capture complex or implicit requirements.

### 4.5 Refusal Check

**Stack:** Python `re` (Regular Expressions)

**Why:** Detects whether the model refused to answer instead of providing a meaningful response.

**Input:** Response

**Output:** Refusal label (True/False) with an explanation

**How it works:** Common refusal phrases are detected using pattern matching, with contextual checks in place to filter out obvious false positives (e.g. "I cannot stress enough how important this is" must never be flagged as a refusal).

**In Scope:**
- Standard refusal phrases (e.g. "I can't help with that")
- Explicit refusal statements

**Out of Scope:**
- Implied refusals expressed in highly unusual wording
- Complex conversational context

**Known Limitation:** Rule-based detection can miss uncommon refusal phrasing and can incorrectly flag benign sentences that merely contain refusal-related keywords.

---

## 5. Evaluation Plan
**Section 5 · Owner: Integration/Evaluation Engineer — Warisha Arshad**

### 5.1 Overview

This section covers how llm-eval-kit itself gets tested, not what the four
criteria measure that's Section 4. The job here is proving the library
actually works: that `Evaluator` behaves correctly, that each criterion
function honors the contract from Section 3.3, and that the whole thing
runs fast enough and installs cleanly.

Three pieces: a fixture dataset with known expected results, a pytest suite
built against that dataset, and a performance benchmark checked against the
2-second target from Section 8.

### 5.2 Fixture Dataset (`fixture_dataset.json`)

**What it is:** a set of `(prompt, response, context, expected_result)`
cases where the correct output is already known before any test runs.
Without this, there's nothing to assert against a test that just checks
"did it return something" doesn't actually prove the score is right.

**Why this is the hard part:** predicting, by hand, what each criterion
should output before running it, means understanding all four criteria
well enough to know the correct answer in advance. This depends directly on
Section 4's known limitations the fixture cases are deliberately built to
sit right on those limitations, not just easy cases.

**Categories the fixture set needs to cover:**

| Category | Purpose | Example |
|---|---|---|
| Clean pass | Confirms basic correctness | Accurate, well-grounded, on-topic response |
| Known false positive | refusal_check's named trap | "I cannot stress enough how important this is..." |
| Contradiction | factual_grounding's known limitation | Response with a flipped number or negation against context |
| Off-topic | relevance's boundary case | Response that's fluent but answers a different question |
| Partial answer | completeness's boundary case | Multi-part prompt, response covers only one part |
| Missing context | Evaluator's skip-not-zero rule | factual_grounding requested with context=None |
| Duplicate/invalid criterion name | Evaluator's fail-fast rule (Section 3.3) | Unknown criterion name passed to evaluate() |

### 5.3 Test Suite (`test_evaluator.py`)

**What it is:** pytest tests run against the fixture dataset, checking
`Evaluator` and each criterion against the documented contract, not just
against "does it crash."

**Specifically testing:**
- Each criterion function returns a score between 0.0 and 1.0, or None, never anything else
- A skipped criterion (score: None) is excluded from the overall average, not treated as 0 called out explicitly in Section 3.3 as a common misunderstanding, so it gets its own test
- Evaluator rejects an unknown criterion name immediately, before running anything (fail-fast, per Section 3.3)
- Registering the same criterion name twice raises an error rather than silently overwriting already partially verified in Section 3.2's registry output, this test locks that behavior in permanently
- A criterion function that doesn't return a "score" key fails with a clear error, not an uncaught KeyError this directly covers Risk 6 in Section 7

Target: 90%+ coverage, per Section 8.3.

### 5.4 Performance Benchmark

**What it is:** a timed test running a full `evaluate()` call across all
four criteria, checked against the 2-second target from Section 8.

**Why it matters here specifically:** the embedding model singleton
(Section 3.1 / Risk 2) is the main performance risk in this library. A
regression test will assert the model is loaded exactly once across
repeated `evaluate()` calls in the same session — if that singleton ever
breaks and starts reloading per call, this test catches it immediately
instead of someone noticing the library got slow weeks later.

### 5.5 Deployment Verification

**What it is:** confirming `pip install llm-eval-kit` actually works end to
end on a clean environment, not just that the code runs inside the dev
setup.

**Checklist (maps directly to Section 8.2 / 8.3):**
- `pip install llm-eval-kit` works on a clean Python 3.9+ environment
- The four-line usage example from the Project Summary actually runs and returns a correct structured result
- CLI command (`llm-eval evaluate --prompt ... --response ... --criteria ...`) runs and prints valid JSON
- Every public function has a docstring, checked before merge, not after

### 5.6 Known Limitation

Predicting expected results by hand for the fixture dataset is inherently
subjective for borderline cases two people could reasonably disagree on
the "correct" completeness score for a partially-answered prompt. This will
be documented per-fixture with a short justification for the expected
value, so disagreements are traceable to a stated reason rather than silent
guesswork.

---

## 6. Module Ownership Table
**Section 6 · Owner: All three**

| Module / File | Owner | Depends On | Target Week |
|---|---|---|---|
| `evaluator.py` | Mahrukh Baig — Lead Engineer | `registry.py`, criteria modules | Week 2 |
| `registry.py` | Mahrukh Baig — Lead Engineer | Python decorators | Week 2 |
| `model_loader.py` | Warisha Arshad — Research/Implementation Engineer | sentence-transformers | Week 2 |
| `criteria/factual_grounding.py` | Warisha Arshad — Research/Implementation Engineer | `model_loader.py`, cosine_similarity | Week 3 |
| `criteria/relevance.py` | Warisha Arshad — Research/Implementation Engineer | `model_loader.py`, cosine_similarity | Week 3 |
| `criteria/completeness.py` | Warisha Arshad — Research/Implementation Engineer | `model_loader.py` | Week 3 |
| `criteria/refusal.py` | Warisha Arshad — Research/Implementation Engineer | Python `re` | Week 3 |
| `cli.py` | Mahrukh Baig — Lead Engineer | `evaluator.py`, Typer | Week 4 |
| `test_evaluator.py` | Muhammad Maaz — Integration/Evaluation Engineer | pytest, Evaluator | Week 4 |
| `fixture_dataset.json` | Muhammad Maaz — Integration/Evaluation Engineer | Evaluation criteria | Week 4 |
| `README.md` | Mahrukh Baig — Lead Engineer | Completed API | Week 5 |

---

## 7. Known Risks
**Section 7 · Owner: All three**

**Risk 1 (Research Engineer):** Semantic similarity may not reliably reflect
factual correctness (negation, sarcasm, and subtle contradictions can still
score high).
**Mitigation:** Document this limitation explicitly in the README and API
docs. Combine similarity scoring with lightweight rule-based checks (e.g.
simple negation detection) where feasible, rather than relying on
embeddings alone.

**Risk 2 (Research Engineer):** Repeated model loading on every
`evaluate()` call would significantly increase evaluation time and could
push performance past the 2-second benchmark.
**Mitigation:** Load the embedding model once per session using a singleton
pattern, and add a regression test that fails if model load is triggered
more than once across repeated `evaluate()` calls.

**Risk 3 (Research Engineer):** Rule-based (regex) refusal detection may
produce false positives (flagging benign text as a refusal) or false
negatives (missing an actual refusal phrased unusually).
**Mitigation:** Build the fixture dataset to explicitly include known
false-positive traps (e.g. "I cannot stress enough how important this is")
and validate `refusal_check` against them before merging. Keep the pattern
list reviewable and easy to extend.

**Risk 4 (Lead Engineer):** The criteria registry pattern, if designed
poorly, could make adding a fifth evaluation criterion in v2 require
editing multiple files instead of one.
**Mitigation:** Lead Engineer finalizes the registry/decorator design early
and reviews it with the team before any criterion is implemented against
it, specifically testing that a new criterion can be added by writing one
function only.

**Risk 5 (Lead Engineer):** A criterion file that exists in the `criteria/`
folder but is never actually imported anywhere will silently never register
in `CRITERIA_REGISTRY`. This fails quietly at first nobody notices until
someone tries to call `evaluate()` with that criterion's name and gets an
"unknown criterion" error, well after the file was written.
**Mitigation:** `llm_eval_kit/__init__.py` will explicitly import every file
under `criteria/`, so simply running `import llm_eval_kit` guarantees the
registry is fully populated. Integration Engineer should add a test that
asserts all four expected criterion names are present immediately after
import, to catch this automatically going forward.

**Risk 6 (Lead Engineer):** `Evaluator` currently assumes every registered
criterion function returns a dictionary containing a `"score"` key, per the
contract defined in Section 3.3. Nothing currently stops a criterion
function from breaking this contract — for example, forgetting to include
`"score"` at all which would crash `evaluate()` with an uncaught
`KeyError` instead of a clear, useful error message.
**Mitigation:** Flagging this for Integration Engineer to test deliberately,
using a dummy criterion function that violates the contract on purpose. The
result of that test will decide whether `Evaluator` needs an explicit
defensive check (e.g. validating the returned dict shape) before Week 2 is
considered complete.

**Risk 7 (Integration/Evaluation Engineer):** Fixture dataset expected
values are hand-predicted and inherently subjective for borderline cases
(see Section 5.6).
**Mitigation:** Mitigated by requiring a documented justification for each
fixture's expected result, so disagreements are traceable to a stated
reason rather than silent guesswork.

---

## 8. Definition of Done
**Section 8 · Owner: Lead Engineer — Mahrukh Baig**

### 8.1 What "Done" Means

llm-eval-kit v1 is done when someone can run `pip install llm-eval-kit` on
a clean computer, use it in four lines of code, and get back a correct,
structured score no internet, no account, no subscription needed.

### 8.2 Core Features Checklist
- [ ] `Evaluator` class with a working `evaluate()` method
- [ ] All four criteria work: factual_grounding, relevance, refusal_check, completeness
- [ ] Results come back as structured JSON with scores + explanations
- [ ] New criteria can be added by writing one function, without editing `evaluator.py`
- [ ] Missing context is handled gracefully, not with a crash
- [ ] CLI works: `llm-eval evaluate --prompt "..." --response "..." --criteria ...`

### 8.3 Quality Checklist
- [ ] 15+ test cases with known expected results, including tricky edge cases
- [ ] 90%+ test coverage
- [ ] Every public function has a docstring
- [ ] `pip install llm-eval-kit` works on a clean Python 3.9+ environment
- [ ] README covers installation, a worked example, all four criteria, and known limitations

### 8.4 Not Required for v1
- Training or fine-tuning any model
- Fact-checking against outside/internet knowledge
- Evaluating many responses in one call (batch evaluation)
- Async or streaming evaluation
- A web dashboard or hosted version