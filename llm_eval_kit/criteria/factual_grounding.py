"""
llm_eval_kit.criteria.factual_grounding
=======================================
Checks how well a response is grounded in the given context by comparing
sentence embeddings, then applies a numeric sanity check on top since embeddings
alone cannot catch swapped numbers or altered figures.

Background:
  Pure semantic similarity treats "water boils at 50C" and "water boils at 100C"
  as almost identical (~0.90 similarity) because it is the same sentence structure
  with one digit changed. To prevent false positives on hallucinations/contradictions,
  we extract numeric claims from the response and validate them against the context,
  capping the score if unsupported figures appear.

Key Architectural Decisions:
  1. Config Separation: Embedding model name and mismatch score cap are loaded
     from factual_grounding_config.json.
  2. Modular Numeric Logic: Extraction and comparison routines live in
     numeric_utils.py, keeping this file focused on the evaluation workflow.
  3. Context Multi-Sentence Union: Validates claims against all context sentences
     (subset check), allowing responses to combine multiple facts.
  4. Singleton Lazy Loading: SentenceTransformer is loaded on first invocation.
  5. Score Semantics: Measures semantic similarity with a numeric-consistency
     penalty applied, not calibrated factual accuracy.

Criterion Interface Contract:
  prompt: Accepted to satisfy the uniform (prompt, response, context, **kwargs)
          signature required by CRITERIA_REGISTRY and Evaluator, though factual
          grounding compares response against context.
"""

import json
from pathlib import Path
import re
from sentence_transformers import SentenceTransformer, util

from llm_eval_kit.criteria.numeric_utils import has_numeric_mismatch
from llm_eval_kit.registry import register_criterion

_CONFIG_PATH = Path(__file__).parent / "factual_grounding_config.json"

with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    _CONFIG = json.load(f)

MODEL_NAME = _CONFIG["model_name"]
MISMATCH_SCORE_CAP = _CONFIG["mismatch_score_cap"]

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


@register_criterion("factual_grounding")
def factual_grounding(prompt: str, response: str, context: str = "", **kwargs) -> dict:
    """
    Scores how grounded `response` is in `context` using max sentence
    similarity, then applies a numeric mismatch check on top.

    Args:
        prompt: Original prompt. Unused directly by this criterion, but
            accepted to satisfy the shared registry contract.
        response: The model's response text to evaluate.
        context: Reference context against which to evaluate grounding.
        **kwargs: Accepted for uniform registry compatibility.

    Returns:
        dict with:
            score (float | None): Cosine similarity score [0.0, 1.0], capped at
                MISMATCH_SCORE_CAP if an unsupported numeric claim is detected,
                or None if context is missing/empty.
            explanation (str): Human-readable reasoning for the score.
            best_matching_sentence (str, optional): The context sentence with
                highest semantic similarity to the response.
    """
    if context is None or not context.strip():
        return {
            "score": None,
            "explanation": "No context provided; factual_grounding is not applicable.",
        }

    sentences = _split_sentences(context)
    if not sentences:
        return {
            "score": None,
            "explanation": "Context contained no extractable sentences.",
        }

    model = _get_model()
    response_embedding = model.encode(response, convert_to_tensor=True)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

    similarities = util.cos_sim(response_embedding, sentence_embeddings)[0]
    best_idx = int(similarities.argmax())
    best_sentence = sentences[best_idx]
    score = float(similarities[best_idx])

    if has_numeric_mismatch(response, sentences):
        score = min(score, MISMATCH_SCORE_CAP)
        explanation = (
            "Score reflects semantic similarity with a numeric-consistency penalty "
            "applied, not calibrated factual accuracy. Response contains a numeric "
            "claim or unit not supported by the context, capping the score."
        )
    else:
        explanation = (
            "Score reflects semantic similarity with a numeric-consistency penalty "
            "applied, not calibrated factual accuracy. Response numeric claims match "
            "or are supported by the context."
        )

    return {
        "score": score,
        "explanation": explanation,
        "best_matching_sentence": best_sentence,
    }
