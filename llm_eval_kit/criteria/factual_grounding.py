"""
criteria/factual_grounding.py

Checks how well a response is grounded in the given context by comparing
embeddings, then applies a numeric sanity check on top since embeddings
alone can't catch a swapped number.

Background: pure semantic similarity treats "water boils at 50C" and
"water boils at 100C" as almost identical, since it's the same sentence
with one digit changed. Tested directly - scored 0.90 similarity despite
being factually opposite. To catch that, we pull numeric claims (value, unit)
out of the response and context, capping the score to 0.3 if the response
contains a numeric claim not supported anywhere in the context.

Key Features & Scope:
  1. Numeric claims are validated against the UNION of all context sentences
     (subset check), allowing responses to combine facts across sentences.
  2. Spelled-out numbers up to billions ("twelve thousand three hundred"),
     negative numbers ("-40", "negative forty"), percentages ("12%",
     "12 percent"), and scientific notation ("3.5e6") are supported.
  3. Digit + scale combinations (e.g. "10 million dollars") correctly combine
     into single values (10000000.0, "dollars").
  4. SentenceTransformer model is lazily loaded on first call, not at module
     import time.
  5. Units are normalized through a synonym table (e.g. "degrees Celsius" and
     "°C"), and common stopwords following numbers are ignored.

Score Semantics:
  Score reflects semantic similarity with a numeric-consistency penalty applied,
  not calibrated factual accuracy.
"""

import re
from sentence_transformers import SentenceTransformer, util

from llm_eval_kit.registry import register_criterion

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_CHUNK_SCALE = {"hundred": 100}
_TOTAL_SCALE = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}

_NEGATION_WORDS = {"negative", "minus"}

_UNIT_SYNONYMS = {
    "celsius": "celsius", "°c": "celsius",
    "fahrenheit": "fahrenheit", "°f": "fahrenheit",
    "kelvin": "kelvin", "°k": "kelvin",
    "meters": "meters", "meter": "meters", "m": "meters",
    "feet": "feet", "foot": "feet", "ft": "feet",
    "kilograms": "kilograms", "kilogram": "kilograms", "kg": "kilograms",
    "pounds": "pounds", "pound": "pounds", "lbs": "pounds", "lb": "pounds",
    "dollars": "dollars", "dollar": "dollars", "usd": "dollars",
    "%": "percent", "percent": "percent", "percentage": "percent",
}

_DEGREE_QUALIFIERS = {"celsius", "fahrenheit", "kelvin"}

_UNIT_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "of",
    "is", "was", "were", "are", "that", "this", "with", "for", "from",
    "it", "its", "has", "have", "had", "now", "then", "so", "as", "by",
}

_NUMBER_TOKEN = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?%?$")


def _normalize_unit(raw_unit):
    return _UNIT_SYNONYMS.get(raw_unit, raw_unit)


def _words_to_number(tokens, start_idx):
    """
    Parses a spelled-out number starting at tokens[start_idx].
    Handles negation ("negative", "minus"), ones, tens, hundred, thousand, million, billion.
    Returns (value_float, tokens_consumed).
    """
    if start_idx >= len(tokens):
        return None, 0

    idx = start_idx
    is_negative = False

    if tokens[idx] in _NEGATION_WORDS:
        is_negative = True
        idx += 1
        if idx >= len(tokens):
            return None, 0

    total = 0
    current_chunk = 0
    matched_any = False

    while idx < len(tokens):
        w = tokens[idx]
        if w in _ONES:
            current_chunk += _ONES[w]
            matched_any = True
            idx += 1
        elif w in _TENS:
            current_chunk += _TENS[w]
            matched_any = True
            idx += 1
        elif w in _CHUNK_SCALE:
            current_chunk = (current_chunk or 1) * _CHUNK_SCALE[w]
            matched_any = True
            idx += 1
        elif w in _TOTAL_SCALE:
            scale = _TOTAL_SCALE[w]
            total += (current_chunk or 1) * scale
            current_chunk = 0
            matched_any = True
            idx += 1
        else:
            break

    total += current_chunk

    if not matched_any:
        return None, 0

    if is_negative:
        total = -total

    return float(total), idx - start_idx


def _read_unit(tokens_lower, start_idx):
    """
    Reads the unit starting at tokens_lower[start_idx], returning
    (normalized_unit, tokens_consumed). Handles "degrees celsius",
    synonym normalization, and ignores stopwords.
    """
    if start_idx >= len(tokens_lower):
        return "", 0

    first = tokens_lower[start_idx]

    if first in _UNIT_STOPWORDS:
        return "", 0

    if first == "degrees" and start_idx + 1 < len(tokens_lower):
        second = tokens_lower[start_idx + 1]
        if second in _DEGREE_QUALIFIERS:
            return _normalize_unit(second), 2

    return _normalize_unit(first), 1


def _extract_numeric_claims(text):
    """
    Extracts (value, normalized_unit) pairs from text. Supports:
      - Digit numbers with decimals, negative signs, scientific notation, trailing %
      - Spelled-out numbers (negative, ones, tens, hundred, thousand, million, billion)
      - Digit + scale combinations (e.g. "10 million")
      - Unit synonyms and stopword filtering
    """
    claims = []
    tokens = re.findall(r"[a-zA-Z°%]+|-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?%?", text)
    tokens_lower = [t.lower() for t in tokens]

    i = 0
    while i < len(tokens):
        token = tokens[i]
        token_lower = tokens_lower[i]

        if _NUMBER_TOKEN.match(token):
            unit_from_token = ""
            val_str = token
            if token.endswith("%"):
                unit_from_token = "percent"
                val_str = token[:-1]

            try:
                value = float(val_str)
            except ValueError:
                i += 1
                continue

            consumed_unit_tokens = 0
            if unit_from_token:
                unit = unit_from_token
            else:
                next_idx = i + 1
                if next_idx < len(tokens_lower) and tokens_lower[next_idx] in _TOTAL_SCALE:
                    scale = _TOTAL_SCALE[tokens_lower[next_idx]]
                    value *= scale
                    next_idx += 1
                unit, consumed_unit_tokens = _read_unit(tokens_lower, next_idx)

            claims.append((value, unit))

            advance = 1
            if not unit_from_token:
                if i + 1 < len(tokens_lower) and tokens_lower[i + 1] in _TOTAL_SCALE:
                    advance += 1
                advance += consumed_unit_tokens
            i += advance
            continue

        val, word_count = _words_to_number(tokens_lower, i)
        if val is not None and word_count > 0:
            unit, consumed_unit_tokens = _read_unit(tokens_lower, i + word_count)
            claims.append((val, unit))
            i += word_count + consumed_unit_tokens
            continue

        i += 1

    return claims


def _split_sentences(text):
    parts = _SENTENCE_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


_KNOWN_UNITS = set(_UNIT_SYNONYMS.values())


def _claim_is_supported(r_claim, context_claims):
    r_val, r_unit = r_claim
    for c_val, c_unit in context_claims:
        if r_val == c_val:
            r_is_known = r_unit in _KNOWN_UNITS
            c_is_known = c_unit in _KNOWN_UNITS

            if r_is_known or c_is_known:
                if r_unit == c_unit:
                    return True
            else:
                return True
    return False


def _has_numeric_mismatch(response, context_sentences):
    response_claims = _extract_numeric_claims(response)
    if not response_claims:
        return False

    context_claims = set()
    for sentence in context_sentences:
        context_claims.update(_extract_numeric_claims(sentence))

    for r_claim in response_claims:
        if not _claim_is_supported(r_claim, context_claims):
            return True

    return False


@register_criterion("factual_grounding")
def factual_grounding(prompt: str, response: str, context: str = "", **kwargs) -> dict:
    """
    Scores how grounded `response` is in `context` using max sentence
    similarity, then applies a numeric mismatch check on top.

    Score reflects semantic similarity with a numeric-consistency penalty
    applied, not calibrated factual accuracy.

    Returns score=None when no context is provided.
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

    if _has_numeric_mismatch(response, sentences):
        score = min(score, 0.3)
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