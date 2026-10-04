"""Text normalization, tokenization, and LCS matching for ViBioMIR evaluation.

Implements exact specification from section 7.1 and 7.2 of the implementation plan:
- Normalization: NFKC -> html.unescape -> lower -> collapse whitespace -> strip.
- Tokenization: whitespace_v1 (split on whitespace).
- LCS: exact two-row dynamic programming with O(min(len(a), len(b))) memory.
- Overlap match: lcs / len(ref_tokens) >= threshold using exact Fraction arithmetic.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
import functools
import hashlib
import html
import re
import unicodedata

DEFAULT_NORMALIZATION_VERSION = "btc_order_v1"
DEFAULT_TOKENIZER_NAME = "whitespace_v1"
DEFAULT_OVERLAP_THRESHOLD = Fraction(2, 5)

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class TextMatchProfile:
    """Configuration profile for text evaluation and matching."""

    tokenizer_name: str = DEFAULT_TOKENIZER_NAME
    normalization_version: str = DEFAULT_NORMALIZATION_VERSION
    overlap_threshold: Fraction = DEFAULT_OVERLAP_THRESHOLD

    @property
    def profile_fingerprint(self) -> str:
        """Deterministic string identifier for profile settings."""
        return (
            f"tok={self.tokenizer_name}:"
            f"norm={self.normalization_version}:"
            f"thresh={self.overlap_threshold.numerator}/{self.overlap_threshold.denominator}"
        )


PROVISIONAL_SPEC_V1 = TextMatchProfile(
    tokenizer_name="whitespace_v1",
    normalization_version="btc_order_v1",
    overlap_threshold=Fraction(2, 5),
)


@functools.lru_cache(maxsize=65536)
def normalize_eval_text(
    text: str,
    normalization_version: str = DEFAULT_NORMALIZATION_VERSION,
) -> str:
    """Normalize evaluation text following exact specification.

    Order:
    1. unicodedata.normalize("NFKC", text)
    2. html.unescape(text)
    3. text.lower()
    4. re.sub(r"\\s+", " ", text).strip()

    Note: NFKC precedes html.unescape. HTML entities resolving to fullwidth
    characters (e.g. &#xFF21;) are unescaped after NFKC, resulting in fullwidth
    characters (e.g. 'ａ') rather than ASCII equivalents ('a').
    """
    if normalization_version != "btc_order_v1":
        raise ValueError(
            f"Unsupported normalization version: {normalization_version}. "
            "Supported versions: ['btc_order_v1']"
        )

    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)
    text = text.lower()
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


@functools.lru_cache(maxsize=65536)
def tokenize_eval_text(
    text: str,
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
) -> tuple[str, ...]:
    """Tokenize text into evaluation tokens.

    In provisional whitespace_v1 profile, splits strictly on whitespace.
    Languages without whitespace (e.g., Chinese) remain single tokens unless
    explicitly tokenized; no implicit Jieba or PyVi segmentation is applied.
    """
    if tokenizer_name != "whitespace_v1":
        raise ValueError(
            f"Unsupported tokenizer: {tokenizer_name}. "
            "Supported tokenizers: ['whitespace_v1']"
        )

    if not text:
        return ()

    return tuple(text.split())


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Compute the Longest Common Subsequence (LCS) length between two token sequences.

    Uses exact dynamic programming with two rows, running in O(len(a) * len(b)) time
    and O(min(len(a), len(b))) memory.
    """
    if not a or not b:
        return 0

    # Ensure b is the shorter sequence to minimize memory allocation
    if len(a) < len(b):
        a, b = b, a

    n = len(b)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for item_a in a:
        for j, item_b in enumerate(b):
            if item_a == item_b:
                curr[j + 1] = prev[j] + 1
            else:
                curr[j + 1] = curr[j] if curr[j] > prev[j + 1] else prev[j + 1]
        prev, curr = curr, prev

    return prev[n]


def overlap_matches(
    pred_tokens: Sequence[str],
    ref_tokens: Sequence[str],
    threshold: Fraction = DEFAULT_OVERLAP_THRESHOLD,
) -> bool:
    """Determine whether prediction tokens sufficiently overlap reference tokens.

    Formula:
        lcs_length(pred_tokens, ref_tokens) / len(ref_tokens) >= threshold

    Evaluated using integer cross-multiplication:
        lcs * threshold.denominator >= len(ref_tokens) * threshold.numerator

    Note:
        - len(ref_tokens) is ALWAYS the denominator. Prediction length is never used.
        - Empty ref_tokens raises ValueError to reject invalid qrels.
        - Empty pred_tokens returns False.
    """
    ref_len = len(ref_tokens)
    if ref_len == 0:
        raise ValueError("Reference tokens cannot be empty")

    if not isinstance(threshold, Fraction):
        threshold = Fraction(str(threshold))

    if threshold < 0 or threshold > 1:
        raise ValueError(f"Threshold must be in [0, 1], got {threshold}")

    if not pred_tokens:
        return False

    lcs = lcs_length(pred_tokens, ref_tokens)
    return lcs * threshold.denominator >= ref_len * threshold.numerator


class TextOverlapEvaluator:
    """Stateful evaluator with caching across an evaluation run."""

    def __init__(self, profile: TextMatchProfile | None = None) -> None:
        self.profile = profile or PROVISIONAL_SPEC_V1
        self._norm_cache: dict[tuple[str, str], str] = {}
        self._token_cache: dict[tuple[str, str], tuple[str, ...]] = {}

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def normalize(self, text: str) -> str:
        """Normalize text with profile-aware caching."""
        cache_key = (self.profile.normalization_version, self._hash_text(text))
        if cache_key not in self._norm_cache:
            self._norm_cache[cache_key] = normalize_eval_text(
                text, normalization_version=self.profile.normalization_version
            )
        return self._norm_cache[cache_key]

    def tokenize(self, text: str, is_already_normalized: bool = False) -> tuple[str, ...]:
        """Tokenize text with profile-aware caching."""
        norm_text = text if is_already_normalized else self.normalize(text)
        cache_key = (self.profile.tokenizer_name, self._hash_text(norm_text))
        if cache_key not in self._token_cache:
            self._token_cache[cache_key] = tokenize_eval_text(
                norm_text, tokenizer_name=self.profile.tokenizer_name
            )
        return self._token_cache[cache_key]

    def matches(self, pred_text: str, ref_text: str) -> bool:
        """Check if predicted chunk text matches reference chunk text."""
        pred_tokens = self.tokenize(pred_text)
        ref_tokens = self.tokenize(ref_text)
        return overlap_matches(
            pred_tokens, ref_tokens, threshold=self.profile.overlap_threshold
        )

    def clear(self) -> None:
        """Clear instance caches."""
        self._norm_cache.clear()
        self._token_cache.clear()


def clear_eval_text_caches() -> None:
    """Clear global LRU caches for evaluation normalization and tokenization."""
    normalize_eval_text.cache_clear()
    tokenize_eval_text.cache_clear()
