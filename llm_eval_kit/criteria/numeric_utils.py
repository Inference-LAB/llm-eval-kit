"""
llm_eval_kit.criteria.numeric_utils
===================================
Utility module for extracting, normalizing, and comparing numeric claims
(value, unit) from text to perform factual consistency checks.

Key Capabilities:
  - Digits with decimals, negative signs, scientific notation, trailing %.
  - Spelled-out numbers (ones, tens, hundred, thousand, million, billion).
  - Digit + scale combinations (e.g. "10 million").
  - Unit normalization and synonym mapping (e.g. "degrees Celsius" and "°C").
  - Selective unit parsing: only recognized physical/economic units are captured,
    preventing arbitrary nouns (e.g. "people" in "10 million people") from being
    treated as units.
  - Float comparison with tolerance via math.isclose().
"""

import math
import re

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
    # Temperature
    "celsius": "celsius", "°c": "celsius",
    "fahrenheit": "fahrenheit", "°f": "fahrenheit",
    "kelvin": "kelvin", "°k": "kelvin",
    # Distance / Length
    "meters": "meters", "meter": "meters", "m": "meters",
    "kilometers": "kilometers", "kilometer": "kilometers", "km": "kilometers",
    "centimeters": "centimeters", "centimeter": "centimeters", "cm": "centimeters",
    "feet": "feet", "foot": "feet", "ft": "feet",
    "miles": "miles", "mile": "miles",
    "inches": "inches", "inch": "inches", "in": "inches",
    # Mass / Weight
    "kilograms": "kilograms", "kilogram": "kilograms", "kg": "kilograms",
    "grams": "grams", "gram": "grams", "g": "grams",
    "pounds": "pounds", "pound": "pounds", "lbs": "pounds", "lb": "pounds",
    # Currency
    "dollars": "dollars", "dollar": "dollars", "usd": "dollars", "$": "dollars",
    "euros": "euros", "euro": "euros", "eur": "euros", "€": "euros",
    # Percentages
    "%": "percent", "percent": "percent", "percentage": "percent",
    # Time
    "seconds": "seconds", "second": "seconds", "sec": "seconds", "s": "seconds",
    "minutes": "minutes", "minute": "minutes", "min": "minutes",
    "hours": "hours", "hour": "hours", "hr": "hours", "hrs": "hours",
    "days": "days", "day": "days",
    "years": "years", "year": "years",
}

_DEGREE_QUALIFIERS = {"celsius", "fahrenheit", "kelvin"}

_KNOWN_UNITS = set(_UNIT_SYNONYMS.values())

_NUMBER_TOKEN = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?%?$")


def normalize_unit(raw_unit: str) -> str:
    """Normalize a raw unit string to its canonical form via the synonym table."""
    return _UNIT_SYNONYMS.get(raw_unit.lower(), raw_unit.lower())


def words_to_number(tokens: list[str], start_idx: int) -> tuple[float | None, int]:
    """
    Parses a spelled-out number starting at tokens[start_idx].
    Handles negation ("negative", "minus"), ones, tens, hundred, thousand,
    million, billion, and optional 'and'.
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
        elif w == "and":
            # Handle optional 'and' in numbers like 'one hundred and five'
            if matched_any and idx + 1 < len(tokens) and (tokens[idx + 1] in _ONES or tokens[idx + 1] in _TENS):
                idx += 1
                continue
            else:
                break
        else:
            break

    total += current_chunk

    if not matched_any:
        return None, 0

    if is_negative:
        total = -total

    return float(total), idx - start_idx


def read_unit(tokens_lower: list[str], start_idx: int) -> tuple[str, int]:
    """
    Reads a recognized unit starting at tokens_lower[start_idx], returning
    (normalized_unit, tokens_consumed).

    Only tokens that map to known units in _UNIT_SYNONYMS or compound units
    (e.g., 'degrees celsius') are recognized. Arbitrary nouns (such as 'people'
    or 'employees') return ('', 0), preventing non-units from being consumed.
    """
    if start_idx >= len(tokens_lower):
        return "", 0

    first = tokens_lower[start_idx]

    # Handle compound "degrees [celsius/fahrenheit/kelvin]"
    if first == "degrees" and start_idx + 1 < len(tokens_lower):
        second = tokens_lower[start_idx + 1]
        if second in _DEGREE_QUALIFIERS:
            return normalize_unit(second), 2

    # Only accept known units from synonym dictionary
    if first in _UNIT_SYNONYMS:
        return normalize_unit(first), 1

    return "", 0


def extract_numeric_claims(text: str) -> list[tuple[float, str]]:
    """
    Extracts (value, normalized_unit) pairs from text. Supports:
      - Digit numbers with decimals, negative signs, scientific notation, trailing %
      - Spelled-out numbers (negative, ones, tens, hundred, thousand, million, billion)
      - Digit + scale combinations (e.g. "10 million")
      - Unit synonyms and normalization
    """
    claims = []
    tokens = re.findall(r"[a-zA-Z°%€$]+|-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?%?", text)
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
                unit, consumed_unit_tokens = read_unit(tokens_lower, next_idx)

            claims.append((value, unit))

            advance = 1
            if not unit_from_token:
                if i + 1 < len(tokens_lower) and tokens_lower[i + 1] in _TOTAL_SCALE:
                    advance += 1
                advance += consumed_unit_tokens
            i += advance
            continue

        val, word_count = words_to_number(tokens_lower, i)
        if val is not None and word_count > 0:
            unit, consumed_unit_tokens = read_unit(tokens_lower, i + word_count)
            claims.append((val, unit))
            i += word_count + consumed_unit_tokens
            continue

        i += 1

    return claims


def claim_is_supported(r_claim: tuple[float, str], context_claims: set[tuple[float, str]]) -> bool:
    """
    Checks if a response numeric claim (value, unit) is supported by any claim in the context.
    Uses math.isclose() to handle float representation differences (e.g. scientific notation, decimals).
    """
    r_val, r_unit = r_claim
    for c_val, c_unit in context_claims:
        if math.isclose(r_val, c_val, rel_tol=1e-5, abs_tol=1e-8):
            r_is_known = r_unit in _KNOWN_UNITS
            c_is_known = c_unit in _KNOWN_UNITS

            if r_is_known or c_is_known:
                if r_unit == c_unit:
                    return True
            else:
                return True
    return False


def has_numeric_mismatch(response: str, context_sentences: list[str]) -> bool:
    """
    Returns True if the response contains any numeric claim that is not supported
    by the union of claims across all context sentences (subset check).
    """
    response_claims = extract_numeric_claims(response)
    if not response_claims:
        return False

    context_claims = set()
    for sentence in context_sentences:
        context_claims.update(extract_numeric_claims(sentence))

    for r_claim in response_claims:
        if not claim_is_supported(r_claim, context_claims):
            return True

    return False
