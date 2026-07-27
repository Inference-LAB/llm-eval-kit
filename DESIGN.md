# Design Document
## llm-eval-kit — LLM Response Evaluation Library

---

## 1. Project Summary
**Section 1 · Owner: Lead Engineer — Mahrukh Baig**

llm-eval-kit is a lightweight, pip-installable Python library that checks
how good an AI's response is. You give it a prompt, the AI's response, and
optionally the source text the response should be based on. It runs four
checks — factual grounding, relevance, completeness, and refusal detection —
and returns a structured score as JSON. Everything runs locally: no
internet needed after the first setup, no subscription, no external AI
needed to do the checking. It's meant to be dropped into a CI pipeline, so
a prompt change or model update can be checked automatically instead of a
person reading through answers by hand.

---

## 2. Problem Statement
**Section 2 · Owner: All three — write this together, not just you**

Anyone who builds a chatbot or an AI writing tool eventually asks the same question: "how do I know if the model's answers are actually good?"

Right now, most people just read a few answers and guess. There's no lightweight tool that checks an AI's response the same way every time, without needing the internet, another AI, or a paid subscription.

llm-eval-kit fixes this. It's a small Python library you install with pip install llm-eval-kit. You give it a prompt, the AI's response, and (optionally) some source text, and it hands back a score explaining whether the response was good — and why — using fast, offline, non-AI methods.

---

## 3. Architecture
**Section 3 · Owner: Lead Engineer — Mahrukh Baig**

### 3.1 Overview

Three pieces make the library work: `Evaluator`, the criteria registry, and
the four criteria functions (built by Research Engineer). `Evaluator` never
talks to a criterion function directly — it only ever talks to the
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
`if/elif` block checking each criterion's name directly — meaning every
new criterion means editing `Evaluator`'s code. With the registry, a new
criterion is just a new file with one decorator. `Evaluator` never changes.

**Known limitation:** if a criterion file is never imported anywhere, it
silently never registers. Fix: `llm_eval_kit`'s main file will import every
criteria file automatically, so this can't happen by accident.

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
no context given) is left out of the average, not counted as 0 — this needs
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

### 3.4 CLI (`cli.py`)

**What it is:** a command-line tool, `llm-eval evaluate --prompt "..."
--response "..." --criteria ...`. It does no evaluation logic itself — it
just reads the command-line input, calls `Evaluator.evaluate()`, and prints
the result as JSON. Keeping it this thin means CLI bugs and evaluation
logic bugs stay separate problems.

---

## 4. Technical Approach
**Section 4 · Owner: Research/Implementation Engineer — Warisha Arshad**

### 4.1 Shared Embedding Model

All semantic criteria run on one shared model: sentence-transformers, using all-MiniLM-L6-v2.

- Converts text into 384-dimensional embeddings
- Lets us compare meaning, not exact wording
- Loaded once via a singleton pattern and reused across every `evaluate()` call
- Runs fully offline — no API calls, no per-request cost

Singleton matters here specifically: loading the model on every call would make the library unusably slow. One load per session, shared across all four criteria.

### 4.2 Factual Grounding

**Stack:** sentence-transformers, scikit-learn (cosine_similarity)

**Why:** Measures whether the response is supported by the provided context, helping detect unsupported or hallucinated claims.

**Input:** Context, Response

**Output:** Grounding score (0–1) with an explanation

**How it works:** Context and response are embedded, then cosine similarity is computed between the two vectors. High similarity indicates the response is likely supported by the provided context.

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
**Section 5 · Owner: Integration/Evaluation Engineer — Maaz**

*(currently empty — Maaz's to fill)*

---

## 6. Module Ownership Table
**Section 6 · Owner: All three**

| Module / File | Owner | Depends On | Target Week |
|---|---|---|---|
| `evaluator.py` | Mahrukh Baig — Lead Engineer | `registry.py`, criteria modules | Week 2 |
| `registry.py` | Mahrukh Baig — Lead Engineer | Python decorators | Week 2 |
| `model_loader.py` | Warisha Arshad — Research/Implementation Engineer | sentence-transformers | Week 2 |
| `factual_grounding.py` | Warisha Arshad — Research/Implementation Engineer | `model_loader.py`, cosine_similarity | Week 3 |
| `relevance.py` | Warisha Arshad — Research/Implementation Engineer | `model_loader.py`, cosine_similarity | Week 3 |
| `completeness.py` | Warisha Arshad — Research/Implementation Engineer | `model_loader.py` | Week 3 |
| `refusal_check.py` | Warisha Arshad — Research/Implementation Engineer | Python `re` | Week 3 |
| `cli.py` | Mahrukh Baig — Lead Engineer | `evaluator.py`, Typer | Week 4 |
| `test_evaluator.py` | Maaz — Integration/Evaluation Engineer | pytest, Evaluator | Week 4 |
| `fixture_dataset.json` | Maaz — Integration/Evaluation Engineer | Evaluation criteria | Week 4 |
| `README.md` | Mahrukh Baig — Lead Engineer | Completed API | Week 5 |

---

## 7. Known Risks
**Section 7 · Owner: All three**

**Risk 1 (Research Engineer):** Semantic similarity may not reliably reflect
factual correctness (negation, sarcasm, and subtle contradictions can still
score high).

**Risk 2 (Research Engineer):** Repeated model loading on every
`evaluate()` call would significantly increase evaluation time and could
push performance past the 2-second benchmark.

**Risk 3 (Research Engineer):** Rule-based (regex) refusal detection may
produce false positives (flagging benign text as a refusal) or false
negatives (missing an actual refusal phrased unusually).

**Risk 4 (Lead Engineer):** The criteria registry pattern, if designed
poorly, could make adding a fifth evaluation criterion in v2 require
editing multiple files instead of one.

**Risk 5 (Lead Engineer):** A criterion file that exists in the `criteria/`
folder but is never actually imported anywhere will silently never register
in `CRITERIA_REGISTRY`. This fails quietly at first — nobody notices until
someone tries to call `evaluate()` with that criterion's name and gets an
"unknown criterion" error, well after the file was written.

**Risk 6 (Lead Engineer):** `Evaluator` currently assumes every registered
criterion function returns a dictionary containing a `"score"` key, per the
contract defined in Section 3.3. Nothing currently stops a criterion
function from breaking this contract — for example, forgetting to include
`"score"` at all — which would crash `evaluate()` with an uncaught
`KeyError` instead of a clear, useful error message.

---

## 8. Definition of Done
**Section 8 · Owner: Lead Engineer — Mahrukh Baig**

### 8.1 What "Done" Means

llm-eval-kit v1 is done when someone can run `pip install llm-eval-kit` on
a clean computer, use it in four lines of code, and get back a correct,
structured score — no internet, no account, no subscription needed.

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