"""
llm_eval_kit.criteria.factual_grounding
=======================================
Evaluates factual grounding of an LLM response against reference context by computing
maximum sentence cosine similarity and enforcing numeric consistency penalties.
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

MODEL_NAME: str = _CONFIG["model_name"]
MISMATCH_SCORE_CAP: float = _CONFIG["mismatch_score_cap"]

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """
    Lazily initializes and returns the cached SentenceTransformer singleton model.

    Returns:
        SentenceTransformer: Cached embedding model instance loaded from MODEL_NAME.

    Known Limitations:
        - Loads the model onto CPU/GPU on first invocation, which incurs initial setup latency.
        - Memory consumption corresponds to the underlying transformer model footprint (~80-120MB RAM).
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?۔؟])\s+|\n+")


def _split_sentences(text: str) -> list[str]:
    """
    Splits input text into discrete sentence and paragraph segments.

    Args:
        text (str): Input text block to segment.

    Returns:
        list[str]: Non-empty stripped sentence and paragraph strings.

    Known Limitations:
        - Uses linear O(N) regex segmentation matching terminal punctuation (.!?۔؟) or newlines.
        - Abbreviations followed by whitespace (e.g. 'e.g. ') may cause false splits.
        - Unpunctuated text without newlines returns a single-element list.
    """
    parts = _SENTENCE_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


@register_criterion("factual_grounding")
def factual_grounding(prompt: str, response: str, context: str = "", **kwargs) -> dict:
    """
    Evaluates how well a model response is grounded in reference context using embedding
    similarity and numeric claim verification.

    Args:
        prompt (str): Original prompt. Accepted for uniform registry compatibility.
        response (str): The model's response string to evaluate. If empty or whitespace-only,
            evaluates to score 0.0.
        context (str, optional): Reference context text to evaluate grounding against. If empty
            or whitespace-only, evaluates to score None (not applicable). Defaults to "".
        **kwargs: Arbitrary keyword arguments for registry compatibility.

    Returns:
        dict: Evaluation result dictionary containing:
            - score (float | None): Cosine similarity score in [0.0, 1.0], capped at
              MISMATCH_SCORE_CAP if unsupported numeric claims are detected; 0.0 if response is empty;
              or None if context is missing/empty.
            - explanation (str): Human-readable explanation of score and penalties.
            - best_matching_sentence (str | None, optional): Context sentence yielding maximum cosine
              similarity to the response, or None if response/context is empty.

    Known Limitations:
        - Semantic similarity represents semantic alignment, not calibrated factual verification.
        - Default model ('all-MiniLM-L6-v2') is English-optimized; non-Latin scripts (e.g. Urdu)
          may exhibit lower semantic discrimination due to WordPiece tokenization.
        - Sentence splitting operates in O(N) linear time and is benchmarked for contexts up to 5,000+ words.
    """
    if context is None or not context.strip():
        return {
            "score": None,
            "explanation": "No context provided; factual_grounding is not applicable.",
        }

    if response is None or not response.strip():
        return {
            "score": 0.0,
            "explanation": "Empty response contains no factual content to evaluate.",
            "best_matching_sentence": None,
        }

    sentences = _split_sentences(context)
    if not sentences:
        return {
            "score": None,
            "explanation": "Context contained no extractable sentences.",
        }

    model = _get_model()
    response_embedding = model.encode(response, convert_to_tensor=True)
    sentence_embeddings = model.encode(sentences, batch_size=64, convert_to_tensor=True)

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
