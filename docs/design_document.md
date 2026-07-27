# INFERENCE LAB
## Design Document
### llm-eval-kit — LLM Response Evaluation Library (A Draft)

---

## 1. Project Summary
*Section 1 · Owner: Lead Engineer*

llm-eval-kit is a lightweight, pip-installable Python library that scores how good an LLM's response is. Give it a prompt, a response, and (optionally) the context the response was supposed to be based on — it runs four checks (factual grounding, relevance, completeness, and refusal detection) and returns a structured JSON score. Everything runs locally: no external API calls, no subscription, and no internet needed after the first model download. It's built to be dropped into a CI pipeline so a prompt change or fine-tune can be checked automatically instead of being eyeballed by a person.

---

## 2. Problem Statement
*Section 2 · Owner: All three — collaborative*

As the use of Large Language Models (LLMs) expands across NLP research, prompt engineering, and Retrieval-Augmented Generation (RAG) systems, evaluating the quality of model responses remains a significant challenge. Current evaluation practices often rely on manual review, external LLMs as evaluators, or proprietary platforms, resulting in high costs, inconsistent results, limited reproducibility, and dependence on internet connectivity or third-party services. These limitations make it difficult for researchers to objectively compare experiments, benchmark model improvements, and produce reliable evaluation results.

Beyond research challenges, many AI projects focus primarily on model development while providing limited opportunities to develop production-oriented engineering skills. As a result, aspiring AI engineers often lack practical experience in designing extensible software architectures, implementing efficient NLP pipelines, and building systematic evaluation and testing frameworks. This gap makes it harder to demonstrate the real-world engineering competencies expected in AI research and industry.

---

## 3. Technical Approach
*Section 3 · Owner: Research/Implementation Engineer - [By Warisha Arshad]*

### 3.1 Shared Embedding Model

All semantic criteria run on one shared model: sentence-transformers, using all-MiniLM-L6-v2.

- Converts text into 384-dimensional embeddings
- Lets us compare meaning, not exact wording
- Loaded once via a singleton pattern and reused across every `evaluate()` call
- Runs fully offline — no API calls, no per-request cost

Singleton matters here specifically: loading the model on every call would make the library unusably slow. One load per session, shared across all four criteria.

### 3.2 Factual Grounding

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

### 3.3 Relevance

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

### 3.4 Completeness

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

### 3.5 Refusal Check

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

## 4. Evaluation Plan
*Section 4 · Owner: Integration/Evaluation Engineer*



---

## 5. Module Ownership Table
*Section 5 · Owner: All three*

| Module / File | Owner | Depends On | Target Week |
|---|---|---|---|
| evaluator.py | Lead Engineer | registry.py, criteria modules | Week 2 |
| registry.py | Lead Engineer | Python decorators | Week 2 |
| model_loader.py | Research/Implementation Engineer | sentence-transformers | Week 2 |
| factual_grounding.py | Research/Implementation Engineer | model_loader.py, cosine_similarity | Week 3 |
| relevance.py | Research/Implementation Engineer | model_loader.py, cosine_similarity | Week 3 |
| completeness.py | Research/Implementation Engineer | model_loader.py | Week 3 |
| refusal_check.py | Research/Implementation Engineer | Python re | Week 3 |
| cli.py | Lead Engineer | evaluator.py, Typer | Week 4 |
| test_evaluator.py | Integration/Evaluation Engineer | pytest, Evaluator | Week 4 |
| fixture_dataset.json | Integration/Evaluation Engineer | Evaluation criteria | Week 4 |
| README.md | Lead Engineer | Completed API | Week 5 |

---

## 6. Known Risks
*Section 6 · Owner: All three*

**Risk 1:** Semantic similarity may not reliably reflect factual correctness (negation, sarcasm, and subtle contradictions can still score high).
**Mitigation:** Document this limitation explicitly in the README and API docs. Combine similarity scoring with lightweight rule-based checks (e.g. simple negation detection) where feasible, rather than relying on embeddings alone.

**Risk 2:** Repeated model loading on every `evaluate()` call would significantly increase evaluation time and could push performance past the 2-second benchmark.
**Mitigation:** Load the embedding model once per session using a singleton pattern, and add a regression test that fails if model load is triggered more than once across repeated `evaluate()` calls.

**Risk 3:** Rule-based (regex) refusal detection may produce false positives (flagging benign text as a refusal) or false negatives (missing an actual refusal phrased unusually).
**Mitigation:** Build the fixture dataset to explicitly include known false-positive traps (e.g. "I cannot stress enough how important this is") and validate refusal_check against them before merging. Keep the pattern list reviewable and easy to extend.

**Risk 4:** The criteria registry pattern, if designed poorly, could make adding a fifth evaluation criterion in v2 require editing multiple files instead of one.
**Mitigation:** Lead Engineer finalizes the registry/decorator design early and reviews it with the team before any criterion is implemented against it, specifically testing that a new criterion can be added by writing one function only.

---

## 7. Definition of Done
*Section 7 · Owner: Lead Engineer*