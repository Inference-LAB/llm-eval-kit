"""
llm_eval_kit.criteria.numeric_utils
===================================
Utility module for extracting, normalizing, and validating numeric claims
(value, unit) from text to perform factual consistency checks.
"""

import math
import re

_ONES: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_TENS: dict[str, int] = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_CHUNK_SCALE: dict[str, int] = {"hundred": 100}
_TOTAL_SCALE: dict[str, int] = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}

_NEGATION_WORDS: set[str] = {"negative", "minus"}

_UNIT_SYNONYMS: dict[str, str] = {
    "celsius": "celsius", "°c": "celsius",
    "fahrenheit": "fahrenheit", "°f": "fahrenheit",
    "kelvin": "kelvin", "°k": "kelvin",
    "meters": "meters", "meter": "meters", "m": "meters",
    "kilometers": "kilometers", "kilometer": "kilometers", "km": "kilometers",
    "centimeters": "centimeters", "centimeter": "centimeters", "cm": "centimeters",
    "feet": "feet", "foot": "feet", "ft": "feet",
    "miles": "miles", "mile": "miles",
    "inches": "inches", "inch": "inches", "in": "inches",
    "kilograms": "kilograms", "kilogram": "kilograms", "kg": "kilograms",
    "grams": "grams", "gram": "grams", "g": "grams",
    "pounds": "pounds", "pound": "pounds", "lbs": "pounds", "lb": "pounds",
    "dollars": "dollars", "dollar": "dollars", "usd": "dollars", "$": "dollars",
    "euros": "euros", "euro": "euros", "eur": "euros", "€": "euros",
    "%": "percent", "percent": "percent", "percentage": "percent",
    "seconds": "seconds", "second": "seconds", "sec": "seconds", "s": "seconds",
    "minutes": "minutes", "minute": "minutes", "min": "minutes",
    "hours": "hours", "hour": "hours", "hr": "hours", "hrs": "hours",
    "days": "days", "day": "days",
    "years": "years", "year": "years",
}

_DEGREE_QUALIFIERS: set[str] = {"celsius", "fahrenheit", "kelvin"}

_KNOWN_UNITS: set[str] = set(_UNIT_SYNONYMS.values())

_URDU_TO_ASCII_DIGITS = str.maketrans({
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
})

_NUMBER_TOKEN = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?%?$")


def normalize_unit(raw_unit: str) -> str:
    """
    Normalizes a raw unit string to its canonical identifier via the synonym dictionary.

    Args:
        raw_unit (str): Raw unit token extracted from text (e.g., '°c', 'kilograms', 'USD').

    Returns:
        str: Canonical unit name (e.g., 'celsius', 'kilograms', 'dollars') or lowercased raw string if unmapped.

    Known Limitations:
        - Only covers standard units in _UNIT_SYNONYMS (temperature, length, mass, currency, percentage, time).
        - Unregistered unit strings are returned as raw lowercase strings without unit conversion.
    """
    return _UNIT_SYNONYMS.get(raw_unit.lower(), raw_unit.lower())


def words_to_number(tokens: list[str], start_idx: int) -> tuple[float | None, int]:
    """
    Parses a contiguous sequence of spelled-out English number words starting at a specified token index.

    Args:
        tokens (list[str]): List of lowercased word tokens.
        start_idx (int): Starting index in tokens to begin parsing.

    Returns:
        tuple[float | None, int]: A tuple of (parsed_float_value, tokens_consumed). If no valid number
            words match at start_idx, returns (None, 0).

    Known Limitations:
        - Supports English number words up to billions (ones, tens, hundred, thousand, million, billion).
        - Does not parse non-English spelled-out number words (e.g., Urdu or French word numbers).
        - Fractions in word form (e.g., 'three quarters', 'half') are not converted to floats.
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
    Extracts a recognized unit from tokens starting at a given index.

    Args:
        tokens_lower (list[str]): List of lowercased word tokens.
        start_idx (int): Index where candidate unit begins.

    Returns:
        tuple[str, int]: A tuple of (normalized_unit_str, tokens_consumed). If the token is not a recognized
            unit in _UNIT_SYNONYMS, returns ('', 0).

    Known Limitations:
        - Selective recognition: only tokens mapping to _UNIT_SYNONYMS or compound degree expressions
          ('degrees celsius') are recognized. Arbitrary nouns ('people', 'employees', 'cars') return ('', 0).
    """
    if start_idx >= len(tokens_lower):
        return "", 0

    first = tokens_lower[start_idx]

    if first == "degrees" and start_idx + 1 < len(tokens_lower):
        second = tokens_lower[start_idx + 1]
        if second in _DEGREE_QUALIFIERS:
            return normalize_unit(second), 2

    if first in _UNIT_SYNONYMS:
        return normalize_unit(first), 1

    return "", 0


def extract_numeric_claims(text: str) -> list[tuple[float, str]]:
    """
    Extracts all numeric claims as (value, normalized_unit) pairs from input text.

    Args:
        text (str): Input text to extract numbers from.

    Returns:
        list[tuple[float, str]]: List of (float_value, normalized_unit) tuples in order of occurrence.

    Known Limitations:
        - Parses Western ASCII digits, Eastern Arabic/Urdu digits (۰-۹), scientific notation, percentages,
          and English spelled numbers up to billions.
        - Unrecognized units following numbers are left as empty unit strings ('') rather than attached.
        - Null or empty input returns an empty list ([]).
    """
    if not text or not text.strip():
        return []

    normalized_text = text.translate(_URDU_TO_ASCII_DIGITS)
    claims = []
    tokens = re.findall(r"[a-zA-Z°%€$]+|-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?%?", normalized_text)
    tokens_lower = [t.lower() for t in tokens]

    i = 0
    while i < len(tokens):
        token = tokens[i]

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
    Verifies whether a single numeric claim from a response is supported by any claim in the context.

    Args:
        r_claim (tuple[float, str]): Response claim as (value, unit).
        context_claims (set[tuple[float, str]]): Set of (value, unit) claims extracted from the context.

    Returns:
        bool: True if an equivalent numeric value is found with compatible units; False otherwise.

    Known Limitations:
        - Compares float values with relative tolerance 1e-5 and absolute tolerance 1e-8 via math.isclose().
        - If a claim has a recognized physical unit (e.g., 'celsius'), the context claim must have the identical
          normalized unit. Unitless claims or unrecognized units match solely on numeric equality.
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
    Checks if a response asserts numeric claims not present across the full union of context sentences.

    Args:
        response (str): The model's response string.
        context_sentences (list[str]): List of context sentence strings.

    Returns:
        bool: True if the response contains any numeric claim not supported anywhere in the context (subset check);
            False if all response numeric claims are supported or if the response has no numeric claims.

    Known Limitations:
        - Validates numeric claim accuracy (precision), not response completeness (recall).
        - Multi-sentence subset check: does not require all response numbers to originate from the same sentence.
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
