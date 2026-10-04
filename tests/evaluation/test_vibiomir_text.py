"""Tests for ViBioMIR text normalization, tokenization, and LCS matching.

Follows Task 6 specification and section 7.1/7.2 of the implementation plan.
"""

from fractions import Fraction
import pytest

from src.evaluation.vibiomir_text import (
    PROVISIONAL_SPEC_V1,
    TextMatchProfile,
    TextOverlapEvaluator,
    clear_eval_text_caches,
    lcs_length,
    normalize_eval_text,
    overlap_matches,
    tokenize_eval_text,
)


def test_nfkc_then_unescape_order():
    """Verify exact order: NFKC -> html.unescape -> lower -> collapse whitespace -> strip."""
    # 1. HER2＋ (fullwidth plus) -> + via NFKC, &amp; -> & via unescape, multiple spaces/newlines collapsed
    raw = "HER2＋ &amp;  A\n B"
    assert normalize_eval_text(raw) == "her2+ & a b"

    # 2. In BTC order (NFKC before unescape), HTML entity &#xFF21; is parsed to
    # fullwidth 'Ａ' after NFKC has already run.
    # Therefore, 'Ａ' is only lowercased to 'ａ' (U+FF41), NOT normalized to ASCII 'a'.
    # If unescape ran before NFKC, it would have yielded 'a'.
    assert normalize_eval_text("&#xFF21;") == "ａ"

    # 3. Empty and whitespace-only strings
    assert normalize_eval_text("") == ""
    assert normalize_eval_text("   \t\n  ") == ""

    # 4. Unknown normalization version raises ValueError
    with pytest.raises(ValueError, match="Unsupported normalization version"):
        normalize_eval_text("test", normalization_version="unknown_v2")


def test_lcs_not_substring_and_order():
    """Verify LCS is subsequence (not contiguous substring) and preserves token order."""
    # Non-contiguous subsequence: ("a", "b", "c") in ("a", "x", "b", "y", "c") has length 3
    # even though it is not a contiguous substring.
    a = ("a", "b", "c")
    b = ("a", "x", "b", "y", "c")
    assert lcs_length(a, b) == 3
    assert lcs_length(b, a) == 3

    # Reversed order does NOT match bag overlap:
    # Bag overlap of ("c", "b", "a") and ("a", "b", "c") is 3,
    # but LCS length is 1 (can only pick one token in increasing sequence order).
    assert lcs_length(("c", "b", "a"), ("a", "b", "c")) == 1

    # Empty sequences
    assert lcs_length((), ()) == 0
    assert lcs_length(("a", "b"), ()) == 0
    assert lcs_length((), ("a", "b")) == 0

    # Completely disjoint
    assert lcs_length(("a", "b"), ("c", "d")) == 0

    # Identity
    assert lcs_length(("a", "b", "c"), ("a", "b", "c")) == 3


def test_repeated_tokens():
    """Verify LCS handles duplicate tokens correctly without overcounting."""
    assert lcs_length(("a", "a", "a"), ("a", "a")) == 2
    assert lcs_length(("a", "a"), ("a", "a", "a")) == 2
    assert lcs_length(("a", "b", "a"), ("a", "a")) == 2
    assert lcs_length(("a", "b", "a"), ("b", "a")) == 2
    assert lcs_length(("x", "a", "x", "a", "x"), ("a", "a")) == 2


def test_threshold_boundary_fraction():
    """Verify exact Fraction(2, 5) boundary: 4/10 matches, 3/10 does not match."""
    # Reference of 10 tokens:
    # Threshold 2/5 means 4/10 is exactly 2/5 (match), 3/10 is < 2/5 (no match).
    ref_10 = tuple(f"t{i}" for i in range(10))

    # Prediction with 4 matching tokens
    pred_4_match = ("t0", "t1", "t2", "t3")
    assert overlap_matches(pred_4_match, ref_10, threshold=Fraction(2, 5)) is True

    # Prediction with 3 matching tokens
    pred_3_match = ("t0", "t1", "t2")
    assert overlap_matches(pred_3_match, ref_10, threshold=Fraction(2, 5)) is False

    # Reference of 5 tokens: 2/5 matches, 1/5 does not
    ref_5 = ("a", "b", "c", "d", "e")
    assert overlap_matches(("a", "b"), ref_5, threshold=Fraction(2, 5)) is True
    assert overlap_matches(("a",), ref_5, threshold=Fraction(2, 5)) is False

    # Custom threshold Fraction(1, 2)
    assert overlap_matches(("a", "b"), ("a", "b", "c", "d"), threshold=Fraction(1, 2)) is True
    assert overlap_matches(("a",), ("a", "b", "c", "d"), threshold=Fraction(1, 2)) is False


def test_wrong_denominator_prediction_tokens():
    """Verify that prediction length is NEVER used as the denominator."""
    # Case 1: Long prediction, short reference
    # ref length = 5, pred length = 50, LCS = 2.
    # 2 / len(ref) = 2/5 >= 2/5 -> True.
    # If pred length was denominator: 2/50 = 1/25 < 2/5 -> False (wrong!).
    ref_tokens = ("a", "b", "c", "d", "e")
    pred_tokens_long = ("a", "b") + tuple(f"extra_{i}" for i in range(48))
    assert len(pred_tokens_long) == 50
    assert overlap_matches(pred_tokens_long, ref_tokens, threshold=Fraction(2, 5)) is True

    # Case 2: Short prediction, long reference
    # ref length = 10, pred length = 2, LCS = 2.
    # 2 / len(ref) = 2/10 = 1/5 < 2/5 -> False.
    # If pred length was denominator: 2/2 = 1.0 >= 2/5 -> True (wrong!).
    ref_tokens_long = tuple(f"r{i}" for i in range(10))
    pred_tokens_short = ("r0", "r1")
    assert overlap_matches(pred_tokens_short, ref_tokens_long, threshold=Fraction(2, 5)) is False


def test_chinese_whitespace_profile_no_jieba():
    """Verify Chinese text without whitespace stays as a single token in whitespace_v1."""
    # Single sentence with no whitespace must remain 1 token
    tokens = tokenize_eval_text("肺癌治疗", tokenizer_name="whitespace_v1")
    assert tokens == ("肺癌治疗",)
    assert len(tokens) == 1

    # Sentence with whitespace splits on whitespace only
    tokens_spaced = tokenize_eval_text("肺癌 治疗", tokenizer_name="whitespace_v1")
    assert tokens_spaced == ("肺癌", "治疗")

    # Unsupported tokenizer raises ValueError
    with pytest.raises(ValueError, match="Unsupported tokenizer"):
        tokenize_eval_text("test", tokenizer_name="jieba_v1")


def test_empty_reference_tokens_reject():
    """Verify empty reference tokens raise ValueError to prevent division by zero."""
    with pytest.raises(ValueError, match="Reference tokens cannot be empty"):
        overlap_matches(("a", "b"), ())

    with pytest.raises(ValueError, match="Reference tokens cannot be empty"):
        overlap_matches((), ())


def test_empty_prediction_tokens():
    """Verify empty prediction tokens return False without raising."""
    ref_tokens = ("a", "b", "c")
    assert overlap_matches((), ref_tokens, threshold=Fraction(2, 5)) is False


def test_text_evaluator_cache_and_profile():
    """Verify TextOverlapEvaluator caches normalized/tokenized text across calls."""
    evaluator = TextOverlapEvaluator(profile=PROVISIONAL_SPEC_V1)
    assert evaluator.profile.tokenizer_name == "whitespace_v1"
    assert evaluator.profile.normalization_version == "btc_order_v1"
    assert evaluator.profile.overlap_threshold == Fraction(2, 5)

    raw_text = "HER2＋ &amp;  A\n B"
    norm1 = evaluator.normalize(raw_text)
    norm2 = evaluator.normalize(raw_text)
    assert norm1 == norm2 == "her2+ & a b"
    assert len(evaluator._norm_cache) == 1

    tok1 = evaluator.tokenize(raw_text)
    tok2 = evaluator.tokenize(raw_text)
    assert tok1 == tok2 == ("her2+", "&", "a", "b")
    assert len(evaluator._token_cache) == 1

    # End-to-end matching with raw strings
    assert evaluator.matches("HER2＋ &amp; A B", "her2+ & a b") is True
    assert evaluator.matches("other text", "her2+ & a b") is False

    evaluator.clear()
    assert len(evaluator._norm_cache) == 0
    assert len(evaluator._token_cache) == 0


def test_module_level_caches_clear():
    """Verify clear_eval_text_caches clears the LRU caches without errors."""
    normalize_eval_text("Sample 1")
    tokenize_eval_text("sample 1")
    clear_eval_text_caches()
