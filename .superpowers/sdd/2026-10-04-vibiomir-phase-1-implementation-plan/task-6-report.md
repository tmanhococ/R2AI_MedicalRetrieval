# Task 6 Report: Exact normalization, LCS và text match profile

## Summary
Task 6 implements the core text normalization, tokenization, dynamic programming Longest Common Subsequence (LCS), and Fraction-based overlap evaluator for ViBioMIR evaluation as defined in Sections 7.1 and 7.2 of the Phase 1 Implementation Plan.

Key deliverables:
1. **Normalization Order (`normalize_eval_text`)**: Strictly enforces `NFKC -> html.unescape -> lower -> collapse whitespace -> strip`. As proven in tests, HTML entities representing fullwidth unicode characters (such as `&#xFF21;`) unescape to fullwidth `ａ` after NFKC, matching the benchmark reference behavior without unwanted ASCII folding.
2. **Tokenization Profile (`tokenize_eval_text`)**: Implements provisional `whitespace_v1` profile that splits strictly on whitespace. Unspaced languages like Chinese (e.g. `"肺癌治疗"`) remain single tokens without unverified third-party word segmentation libraries (e.g. Jieba or PyVi).
3. **Exact Two-Row LCS (`lcs_length`)**: Implements strict Longest Common Subsequence using two rolling DP rows with $O(\min(|a|, |b|))$ memory overhead and $O(|a| \cdot |b|)$ runtime. Preserves strict token order and accurately accounts for repeated tokens without bag-of-words distortion.
4. **Fraction-based Overlap Matcher (`overlap_matches`)**: Compares prediction and reference tokens against `Fraction(2, 5)` using exact integer cross-multiplication (`lcs * threshold.denominator >= len(ref_tokens) * threshold.numerator`). Enforces reference length strictly as the denominator, rejects empty references with `ValueError`, and returns `False` for empty predictions.
5. **Stateful Evaluator & Caching (`TextOverlapEvaluator`, `TextMatchProfile`)**: Provides run-scoped evaluation profile caching keyed by `(profile_setting, text_hash)` alongside module-level LRU caches for performance during large evaluation sweeps.

---

## Commits Created
- `e3474f3`: `feat: implement ViBioMIR text overlap evaluator (Task 6)`

---

## Files Created / Modified
- `src/evaluation/vibiomir_text.py` (Created: 217 lines)
- `tests/evaluation/test_vibiomir_text.py` (Created: 183 lines)
- `src/evaluation/__init__.py` (Modified: exported new functions, profile classes, and cache helpers)
- `.superpowers/sdd/2026-10-04-vibiomir-phase-1-implementation-plan/task-6-brief.md` (Updated checkboxes)

---

## TDD Evidence

### 1. RED Phase (Initial Test Run Failure)
Command: `uv run pytest tests/evaluation/test_vibiomir_text.py -q`
Output:
```
=================================== ERRORS ====================================
___________ ERROR collecting tests/evaluation/test_vibiomir_text.py ___________
ImportError while importing test module 'D:\Research\R2AI\mvp-rag-medical\tests\evaluation\test_vibiomir_text.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\Lib\importlib\__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests\evaluation\test_vibiomir_text.py:9: in <module>
    from src.evaluation.vibiomir_text import (
E   ModuleNotFoundError: No module named 'src.evaluation.vibiomir_text'
=========================== short test summary info ===========================
ERROR tests/evaluation/test_vibiomir_text.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.17s
```

### 2. GREEN Phase (Implementation Passing)
Command: `uv run pytest tests/evaluation/test_vibiomir_text.py -v`
Output:
```
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.6.0 -- D:\Research\R2AI\mvp-rag-medical\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\Research\R2AI\mvp-rag-medical
configfile: pyproject.toml
plugins: anyio-4.15.1, asyncio-1.4.0, cov-7.1.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 10 items

tests/evaluation/test_vibiomir_text.py::test_nfkc_then_unescape_order PASSED [ 10%]
tests/evaluation/test_vibiomir_text.py::test_lcs_not_substring_and_order PASSED [ 20%]
tests/evaluation/test_vibiomir_text.py::test_repeated_tokens PASSED      [ 30%]
tests/evaluation/test_vibiomir_text.py::test_threshold_boundary_fraction PASSED [ 40%]
tests/evaluation/test_vibiomir_text.py::test_wrong_denominator_prediction_tokens PASSED [ 50%]
tests/evaluation/test_vibiomir_text.py::test_chinese_whitespace_profile_no_jieba PASSED [ 60%]
tests/evaluation/test_vibiomir_text.py::test_empty_reference_tokens_reject PASSED [ 70%]
tests/evaluation/test_vibiomir_text.py::test_empty_prediction_tokens PASSED [ 80%]
tests/evaluation/test_vibiomir_text.py::test_text_evaluator_cache_and_profile PASSED [ 90%]
tests/evaluation/test_vibiomir_text.py::test_module_level_caches_clear PASSED [100%]

============================= 10 passed in 0.04s ==============================
```

---

## Full Regression Test Suite
Command: `uv run pytest -q`
Output:
```
........................................................................ [ 26%]
........................................................................ [ 52%]
........................................................................ [ 78%]
............................................................             [100%]
276 passed in 8.78s
```
*Note: 266 baseline tests + 10 new Task 6 tests = 276 tests passed. Zero regressions.*

---

## Self-Review Checklist

- [x] **Normalization Pipeline Order**: `unicodedata.normalize("NFKC")` strictly before `html.unescape()`, followed by `.lower()` and `re.sub(r"\s+", " ", text).strip()`.
- [x] **Subsequence Semantics**: LCS preserves relative order; `("c", "b", "a")` vs `("a", "b", "c")` yields 1, not 3.
- [x] **Denominator Contract**: `len(ref_tokens)` is always the denominator; prediction length cannot inflate or deflate the threshold ratio.
- [x] **Exact Arithmetic**: Evaluated with `Fraction` cross-multiplication (`lcs * threshold.denominator >= len(ref_tokens) * threshold.numerator`), preventing float precision inaccuracies.
- [x] **Zero Divisor Guard**: Empty reference tokens raise `ValueError` as specified.
- [x] **Language Integrity**: Chinese without whitespace is not implicitly split by any external segmenter.
- [x] **Memory & Performance**: LCS algorithm uses two rolling rows bounded by $\min(|a|, |b|)$ elements. Profiles and text hashes are cached for rapid repeated evaluation.
