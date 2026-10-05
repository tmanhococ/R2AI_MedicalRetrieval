from __future__ import annotations

import hashlib
import re
from typing import Sequence

try:
    import jieba  # type: ignore
    _JIEBA_AVAILABLE = True
except ImportError:
    _JIEBA_AVAILABLE = False


# Clinical symbols, +/- signs, units, drug/chemical patterns, negations, and words
_MEDICAL_PROTECTED_PATTERN = re.compile(
    r"""
    # 1. Standalone special clinical symbols
    (?:\+\/-|±|\+ve|-ve)
    # 2. Numbers with clinical units (e.g. 50mg/ml, 5.5mmol/l, 100mg, 10ml, 6.5%, etc.)
    |\b\d+(?:\.\d+)?(?:mg\/ml|mmol\/l|mg\/dl|mcg|μg|ug|mg|ml|iu|mmhg|kg|g|%|cm|mm)\b
    # 3. Hyphenated drug / chemical / virus codes (e.g. COVID-19, SARS-CoV-2, non-small, IL-6)
    |\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b
    # 4. Biomarkers with +/- (e.g. HER2+, Ca2+, CD4+, Rh-, CD8+, ER+, PR+)
    |[A-Za-z0-9_]+[+-]+(?![A-Za-z0-9])
    # 5. Standalone + or - symbol (e.g. from (+) or (-))
    |(?<=[(\s])[-+](?=[)\s.,;])
    # 6. CJK Ideographs (Chinese characters)
    |[\u4e00-\u9fff]
    # 7. Vietnamese syllables, English words, alphanumeric tokens
    |[A-Za-z0-9À-ỹà-ỹ_]+
    """,
    re.VERBOSE | re.UNICODE | re.IGNORECASE,
)


def tokenize_lexical(text: str, language: str = "und") -> list[str]:
    """Tokenizes text preserving medical symbols, units, negations, and language words.

    Supported languages:
    - 'vi': Vietnamese syllables + protected medical symbols + negation (không, chưa, chẳng)
    - 'en': Unicode words + protected medical symbols + negation (no, not, without, non-)
    - 'zh': jieba (if available) or CJK regex + protected medical symbols + negation (无, 不, 非)
    - 'und': Unicode words + CJK characters + protected medical symbols
    """
    if not text or not text.strip():
        return []

    # First extract all matching tokens using the medical-protected pattern
    matches = _MEDICAL_PROTECTED_PATTERN.findall(text)
    tokens: list[str] = [m.strip().lower() for m in matches if m.strip()]

    # If Chinese and jieba is available, refine multi-character CJK sequences
    lang_lower = (language or "und").strip().lower()
    if lang_lower == "zh" and _JIEBA_AVAILABLE:
        refined: list[str] = []
        for tok in tokens:
            # If tok is a CJK character or multiple CJK chars, let jieba segment if longer
            if all('\u4e00' <= ch <= '\u9fff' for ch in tok) and len(tok) > 1:
                cut = jieba.lcut(tok)
                refined.extend(c.strip().lower() for c in cut if c.strip())
            else:
                refined.append(tok)
        tokens = refined

    return tokens


def encode_token(token: str) -> str:
    """Safely encodes a single token into a deterministic sha256 hex string starting with 't'.

    Format: 't' + sha256(token.lower().utf-8).hexdigest()
    Guarantees no parser syntax errors, brackets, +/- or boolean operator corruption in Lucene/BM25.
    """
    cleaned = token.strip().lower()
    return "t" + hashlib.sha256(cleaned.encode("utf-8")).hexdigest()


def encode_tokens(tokens: Sequence[str]) -> list[str]:
    """Encodes a sequence of tokens into a list of safe sha256 hex tokens."""
    return [encode_token(t) for t in tokens if t.strip()]


def tokenize_and_encode(text: str, language: str = "und") -> str:
    """Tokenizes raw text and encodes tokens into a whitespace-delimited safe string."""
    tokens = tokenize_lexical(text, language=language)
    return " ".join(encode_tokens(tokens))
