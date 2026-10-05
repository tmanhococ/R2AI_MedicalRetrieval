# Task 5 Report: Query Normalization & Adaptive Query Router

## 1. What was implemented
- **`src/query/query_processor.py`**:
  - `normalize_vi_query(text: str) -> str`: Normalizes Vietnamese medical queries using Unicode NFC normalization, zero-width stripping, and whitespace collapsing while strictly preserving medical abbreviations (`HA`, `ĐTĐ`, `THA`, `COPD`), units (`mg`, `ml`, `mmol/L`, `mmHg`), diagnostic markers (`(+)`, `(-)`), and medication names without destructive lowercasing or stemming.
  - `AdaptiveQueryRouter`: Routes queries based on word count threshold (`length_threshold`, default 50).
    - Short queries (`<= length_threshold`): Identical text for BM25 and Dense branches (`is_clinical_case = False`).
    - Long clinical cases (`> length_threshold`): Strips clinical boilerplate phrases (demographics, admission reasons, history headers, clinical examination boilerplate) for BM25 (`bm25_text`), while preserving full contextual information for Dense retrieval (`dense_text`).
    - Dense truncation strategies: Supports `head_tail` (head 25% up to 128 tokens + tail remainder up to `max_dense_tokens`) and `full` truncation for queries exceeding `max_dense_tokens` (default 512).
    - Accepts both `QueryRecord` instances and raw text strings.
  - `ProcessedQuery`: Frozen dataclass containing `query_id`, `original_text`, `normalized_text`, `bm25_text`, `dense_text`, `is_clinical_case`, `word_count`, and a `@property condensed_vi`.
  - `QueryVariants`: Frozen dataclass containing `vi`, `vi_condensed`, `en`, and `zh`.
  - `build_query_variants(processed: ProcessedQuery, en: str | None = None, zh: str | None = None) -> QueryVariants`: Helper to generate query variant structures for downstream multi-lingual retrieval streams.
- **`src/query/__init__.py`**: Re-exports all key interfaces.
- **`tests/test_query_processor.py`**: 12 comprehensive unit tests verifying all features and edge cases.

## 2. TDD Evidence
- **RED (failing test)**: Ran `pytest tests/test_query_processor.py -v` before implementing the module.
  - Result: Failed with `ModuleNotFoundError: No module named 'src.query'`.
- **GREEN (passing tests)**:
  - `tests/test_query_processor.py`: 12 passed in 0.13s.
  - Full suite (`pytest`): 49 passed in 0.26s across all test modules.

## 3. Files Changed
- `src/query/__init__.py` (Created)
- `src/query/query_processor.py` (Created)
- `tests/test_query_processor.py` (Created)

## 4. Self-Review Findings
- All task brief and design specifications for Task 5 have been implemented with exact signature compatibility.
- Boilerplate regex filters correctly target conversational and administrative padding without touching diagnostic markers or drug names (verified with Amlodipine, Troponin T, HA, ĐTĐ).
- NFC normalization correctly composes decomposed Vietnamese characters into canonical precomposed code points.
- Edge cases handled gracefully: empty strings, pure whitespace, missing query IDs, non-QueryRecord inputs.

## 5. Concerns
- None. Downstream Task 6 (`Translator`, `TranslationCache`, and `MultilingualQueryBuilder`) can consume `QueryVariants` and `ProcessedQuery` directly.
