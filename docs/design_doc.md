# llm-eval-kit — Design Document
Cohort 01 · Week 1

---

## 1. Project Summary
Owner: Lead Engineer
*[Placeholder — one paragraph overview of what llm-eval-kit is and why it exists]*

---

## 2. Problem Statement
Owner: Shared
*[Placeholder — what problem does this solve, who has it, why current approaches fall short]*

---

## 3. Technical Approach
Owner: Muhammad Maaz — Research/Implementation Engineer

### 3.1 Shared Foundation: Embedding Model

All four criteria that depend on semantic comparison (factual_grounding, relevance,
completeness) share a single embedding backbone: sentence-transformers'
all-MiniLM-L6-v2. This is loaded once per process via a singleton pattern, not
once per evaluate() call.

Manual testing (see appendix) confirms load time is trivial on CPU after the
initial ~80MB download, but repeated re-instantiation of SentenceTransformer()
inside a hot loop is the single biggest performance risk in this library, since
model loading dominates over the actual embedding computation. The singleton
must also be safe to call from a multi-threaded context (e.g. a Flask service
evaluating several responses concurrently) — first-load race conditions are a
known failure mode for naive singletons and will be covered explicitly in the
Integration Engineer's test suite.

### 3.2 factual_grounding

**Method:** Sentence-level cosine similarity between the response and each
sentence in the provided context, taking the maximum (not the average) as the
grounding score. Splitting context into sentences and taking the max avoids
diluting the signal when only one sentence in a long context actually supports
the response.

**Verified limitation (from manual testing):** Cosine similarity measures
semantic/topical closeness, not factual correctness. A paraphrased, accurate
response scored 0.982 against its context. A response that directly
contradicted its context (stating "50 degrees Celsius" instead of the correct
"100 degrees Celsius" for water's boiling point) still scored 0.822 — high
enough to risk a false "grounded" verdict under a naive fixed threshold.

**Implication for design:** factual_grounding cannot rely on similarity alone
for numerical or entity-specific claims. v1 will use similarity as the primary
signal, but will flag any response containing a number that doesn't literally
appear in the matched context sentence, as a supplementary check.

**In scope:** context-grounded factual claims, numeric mismatches against
provided context.
**Out of scope for v1:** claims not covered by the given context at all,
world-knowledge fact-checking, multi-hop reasoning across multiple context
sentences.

### 3.3 relevance

**Method:** Cosine similarity between the prompt embedding and the response
embedding directly — no context needed for this check.

**Known risk:** A response can be topically related to the prompt without
actually answering it. Pure embedding similarity will score these as falsely
relevant. v1 will document this as a known limitation — see Section 6.

### 3.4 refusal_check

**Method:** Primarily pattern-based, not embedding-based. Named failure mode
from the project brief: phrases like "I cannot stress enough how important
this is" contain refusal-adjacent language without being a refusal at all.

**Design decision:**
1. Keyword/phrase matching for common refusal openers, restricted to
   sentence-initial position.
2. Secondary check: does the response, excluding the flagged phrase, still
   contain a substantive answer to the prompt? If yes, it's not a refusal.

*[To update: real test data pending — see Section 3 tracker below]*

### 3.5 completeness

**Method:** Sentence-level coverage — decompose the prompt into implied
sub-aspects (naive approach: split on conjunctions/question marks) and check
whether each has at least one semantically similar sentence in the response.

**Open question carried into Section 6:** decomposing a prompt into "aspects"
reliably is the least well-defined part of this criterion. v1 will use a
simple heuristic rather than anything more sophisticated, flagged as a
limitation rather than solved perfectly.

### Appendix: Manual Verification Data

| Case | Context | Response | Cosine Similarity |
|------|---------|----------|-------------------|
| 1 (paraphrase, accurate) | "Paris is the capital of France." | "The capital of France is Paris." | 0.982 |
| 2 (contradiction) | "Water boils at 100 degrees Celsius at sea level." | "Water boils at 50 degrees Celsius." | 0.822 |

Test environment: Python venv, sentence-transformers 5.6.0, all-MiniLM-L6-v2, CPU-only, Windows 11.

---

## 4. Evaluation Plan
Owner: Integration/Evaluation Engineer
*[Placeholder — test fixture strategy, coverage target, performance benchmark]*

---

## 5. Module Ownership Table
Owner: Shared

| Module/File | Owner | Description |
|---|---|---|
| `evaluator.py` | Lead Engineer | Evaluator class, criteria registry |
| `cli.py` | Lead Engineer | Typer CLI |
| `criteria/factual_grounding.py` | Maaz | Grounding check |
| `criteria/relevance.py` | Maaz | Relevance check |
| `criteria/refusal_check.py` | Maaz | Refusal detection |
| `criteria/completeness.py` | Maaz | Completeness check |
| `model_loader.py` | Maaz | Singleton embedding model loader |
| `tests/` | Integration Engineer | pytest suite, fixtures |
| *[fill in remaining files as architecture solidifies]* | | |

---

## 6. Known Risks
Owner: Shared

- **factual_grounding false positives on numeric contradictions** — verified via testing, see Section 3.2. Mitigation: supplementary numeric-mismatch check.
- **relevance false positives on topically-related-but-non-answering responses** — unmitigated in v1, documented limitation.
- **completeness prompt decomposition is a naive heuristic** — may miss implicit sub-aspects not separated by conjunctions.
- **Singleton model loader thread-safety under concurrent use** — needs explicit test coverage from Integration Engineer.
- *[Add Lead/Integration Engineer risks here]*

---

## 7. Definition of Done
Owner: Lead Engineer
*[Placeholder — what counts as "Week 1 complete" / "v1 shippable"]*
