"""
criteria/factual_grounding.py

Checks how well a response is grounded in the given context by comparing
embeddings, then applies a numeric sanity check on top since embeddings
alone can't catch a swapped number.

Background: pure semantic similarity treats "water boils at 50C" and
"water boils at 100C" as almost identical, since it's the same sentence
with one digit changed. Tested directly - scored 0.90 similarity despite
being factually opposite. To catch that, we pull the numeric values out
of the response and the closest-matching context sentence and compare
them. Three things beyond a plain digit match:

  1. Spelled-out numbers ("fifty") are converted to digits before
     comparing, so "fifty degrees" vs "100 degrees" is still caught.
  2. Numbers are paired with whatever unit word follows them, so
     "100 Celsius" vs "100 Fahrenheit" is treated as a mismatch even
     though the digits match.
  3. Units are normalized through a small synonym table first, so
     "100 degrees Celsius" and "100°C" are recognized as the same unit
     rather than false-flagged as a mismatch just because the words
     differ. This table only covers common units (temperature, length,
     weight, currency) - an unrecognized unit is left as-is and compared
     literally, which can still under- or over-match on unusual phrasing.

Still not a real fact-checker. Wrong facts with no number in them, or a
number phrased in a way this doesn't recognize, won't be caught. Worth
stating that boundary plainly in the PR.
"""

import re

from sentence_transformers import SentenceTransformer, util

from llm_eval_kit.registry import register_criterion

_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

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

_UNIT_SYNONYMS = {
    "celsius": "celsius", "°c": "celsius",
    "fahrenheit": "fahrenheit", "°f": "fahrenheit",
    "kelvin": "kelvin", "°k": "kelvin",
    "meters": "meters", "meter": "meters", "m": "meters",
    "feet": "feet", "foot": "feet", "ft": "feet",
    "kilograms": "kilograms", "kilogram": "kilograms", "kg": "kilograms",
    "pounds": "pounds", "pound": "pounds", "lbs": "pounds", "lb": "pounds",
    "dollars": "dollars", "dollar": "dollars", "usd": "dollars",
}

_DEGREE_QUALIFIERS = {"celsius", "fahrenheit", "kelvin"}


def _normalize_unit(raw_unit):
    return _UNIT_SYNONYMS.get(raw_unit, raw_unit)


def _words_to_number(words):
    total = 0
    current = 0
    matched_anything = False

    for word in words:
        if word in _ONES:
            current += _ONES[word]
            matched_anything = True
        elif word in _TENS:
            current += _TENS[word]
            matched_anything = True
        elif word == "hundred":
            current = (current or 1) * 100
            matched_anything = True
        else:
            break

    total += current
    return total if matched_anything else None


def _read_unit(tokens_lower, start_idx):
    """
    Reads the unit starting at tokens_lower[start_idx], returning
    (normalized_unit, tokens_consumed). Handles the "degrees celsius"
    two-word case specially; otherwise reads one word and normalizes it
    through the synonym table.
    """
    if start_idx >= len(tokens_lower):
        return "", 0

    first = tokens_lower[start_idx]

    if first == "degrees" and start_idx + 1 < len(tokens_lower):
        second = tokens_lower[start_idx + 1]
        if second in _DEGREE_QUALIFIERS:
            return _normalize_unit(second), 2

    return _normalize_unit(first), 1


def _extract_numeric_claims(text):
    """
    Finds (value, normalized_unit) pairs in text. Handles digit numbers
    ("100"), short spelled-out runs ("one hundred"), and common unit
    synonyms/phrasings. Intentionally simple - meant to catch a swapped
    figure or unit, not parse arbitrary numeric language.
    """
    claims = []
    tokens = re.findall(r"[a-zA-Z°]+|\d+(?:\.\d+)?", text)
    tokens_lower = [t.lower() for t in tokens]

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if re.match(r"^\d+(?:\.\d+)?$", token):
            value = float(token)
            unit, _ = _read_unit(tokens_lower, i + 1)
            claims.append((value, unit))
            i += 1
            continue

        lookahead = tokens_lower[i:i + 3]
        value = _words_to_number(lookahead)
        if value is not None:
            consumed = 0
            for w in lookahead:
                if w in _ONES or w in _TENS or w == "hundred":
                    consumed += 1
                else:
                    break
            unit, _ = _read_unit(tokens_lower, i + consumed)
            claims.append((float(value), unit))
            i += consumed
            continue

        i += 1

    return claims


def _split_sentences(text):
    parts = _SENTENCE_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _has_numeric_mismatch(response, matched_sentence):
    response_claims = set(_extract_numeric_claims(response))
    context_claims = set(_extract_numeric_claims(matched_sentence))

    if not response_claims or not context_claims:
        return False

    return response_claims != context_claims


@register_criterion("factual_grounding")
def factual_grounding(prompt: str, response: str, context: str = "", **kwargs) -> dict:
    """
    Scores how grounded `response` is in `context` using max sentence
    similarity, then applies a numeric mismatch check on top (see module
    docstring for what that check does and doesn't catch).

    Returns score=None when there's no context to check against - that's
    a different outcome from a low score and shouldn't be conflated with
    "failed the check."
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

    response_embedding = _MODEL.encode(response, convert_to_tensor=True)
    sentence_embeddings = _MODEL.encode(sentences, convert_to_tensor=True)

    similarities = util.cos_sim(response_embedding, sentence_embeddings)[0]
    best_idx = int(similarities.argmax())
    best_sentence = sentences[best_idx]
    score = float(similarities[best_idx])

    if _has_numeric_mismatch(response, best_sentence):
        score = min(score, 0.3)
        explanation = (
            "Response contains a numeric value or unit that doesn't match the "
            "closest context sentence, so the score was capped despite high "
            "semantic similarity. Units are normalized through a small synonym "
            "table first (e.g. 'degrees Celsius' and '°C' are treated as the "
            "same unit) so this shouldn't false-flag equivalent phrasings - but "
            "it won't catch a wrong fact with no number in it, or a number/unit "
            "phrased in a way the synonym table doesn't recognize."
        )
    else:
        explanation = (
            "Semantic similarity to context, not verified factual accuracy - a "
            "response that contradicts the context in a way that doesn't involve "
            "a recognizable number/unit mismatch may still score moderately "
            "instead of 0.0."
        )

    return {
        "score": score,
        "explanation": explanation,
        "best_matching_sentence": best_sentence,
    }