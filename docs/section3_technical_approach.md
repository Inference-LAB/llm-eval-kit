# Section 3 — Technical Approach
Owner: Muhammad Maaz — Research/Implementation Engineer

## 3.1 Shared Foundation: Embedding Model

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

## 3.2 factual_grounding

**Method:** Sentence-level cosine similarity between the response and each
sentence in the provided context, taking the maximum (not the average) as the
grounding score. Splitting context into sentences and taking the max avoids
diluting the signal when only one sentence in a long context actually supports
the response — averaging across unrelated sentences would wash out a strong
match.

**Verified limitation (from manual testing):** Cosine similarity measures
semantic/topical closeness, not factual correctness. A paraphrased, accurate
response scored 0.982 against its context. A response that directly
contradicted its context (stating "50 degrees Celsius" instead of the correct
"100 degrees Celsius" for water's boiling point) still scored 0.822 — high
enough to risk a false "grounded" verdict under a naive fixed threshold (e.g.
"score > 0.7 = grounded").

**Implication for design:** factual_grounding cannot rely on similarity alone
for numerical or entity-specific claims. v1 scope will use similarity as the
primary signal, but will flag (not silently pass) any response containing a
number that doesn't literally appear in the matched context sentence, as a
supplementary check. This is a heuristic, not full fact-checking — genuinely
out of scope for v1 is anything requiring external knowledge verification
(e.g. checking claims against the internet).

**In scope:** context-grounded factual claims, numeric mismatches against
provided context.
**Out of scope for v1:** claims not covered by the given context at all,
world-knowledge fact-checking, multi-hop reasoning across multiple context
sentences.

## 3.3 relevance

**Method:** Cosine similarity between the prompt embedding and the response
embedding directly — no context needed for this check, since relevance is
about whether the response addresses the prompt, not whether it's grounded
in supporting material.

**Known risk:** A response can be topically related to the prompt without
actually answering it (e.g. restating the question, or answering a related-
but-different question). Pure embedding similarity will score these as
falsely relevant. v1 will document this as a known limitation rather than
attempt to solve it with a single-model approach — flagged in Section 6
(Known Risks).

## 3.4 refusal_check

**Method:** Primarily pattern-based, not embedding-based, since refusals have
recognizable lexical structure but with a specific, named failure mode from
the project brief: phrases like "I cannot stress enough how important this
is" contain refusal-adjacent language ("I cannot") without being a refusal at
all.

**Design decision:** A naive keyword match on "I cannot" / "I won't" /
"I'm unable to" is not sufficient. The check needs at minimum:
1. Keyword/phrase matching for common refusal openers, restricted to sentence-
   initial position (refusals almost always start a response, whereas
   "I cannot stress enough" typically doesn't open the response).
2. A secondary check: does the response, excluding the flagged phrase, still
   contain a substantive answer to the prompt? If yes, it's not a refusal
   regardless of the opening phrase.

This is the highest-risk criterion for false positives and will need the
largest fixture set in the Integration Engineer's test suite.

## 3.5 completeness

**Method:** Sentence-level coverage — decompose the prompt into implied
sub-aspects (naive approach: treat clauses/questions within the prompt as
separate aspects) and check whether each has at least one semantically
similar sentence in the response, using the same embedding approach as
factual_grounding.

**Open question carried into Section 6:** decomposing a prompt into
"aspects" reliably is the least well-defined part of this criterion. v1 will
likely use a simple heuristic (splitting on conjunctions and question marks)
rather than anything more sophisticated, and this will be explicitly flagged
as a limitation rather than solved perfectly in v1.

## Appendix: Manual Verification Data

| Case | Context | Response | Cosine Similarity |
|------|---------|----------|-------------------|
| 1 (paraphrase, accurate) | "Paris is the capital of France." | "The capital of France is Paris." | 0.982 |
| 2 (contradiction) | "Water boils at 100 degrees Celsius at sea level." | "Water boils at 50 degrees Celsius." | 0.822 |

Test environment: Python venv, sentence-transformers 5.6.0, all-MiniLM-L6-v2,
CPU-only, Windows 11.
