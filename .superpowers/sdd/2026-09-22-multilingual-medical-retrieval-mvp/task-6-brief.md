### Task 6: Translation Protocol, Offline Cache & Fault-Tolerant Translator

**Files:**
- Create: `src/query/translator.py`
- Create: `src/query/translation_cache.py`
- Test: `tests/test_translator.py`

**Interfaces:**
- Consumes: `QueryVariants`, `ProcessedQuery` from `src.query.query_processor`
- Produces: `Translator(Protocol)`, `PrecomputedCacheTranslator`, `PassthroughTranslator`, `LocalHfTranslator`, `ApiTranslator`, `create_translator(config: dict) -> Translator`, `MultilingualQueryBuilder`

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Translator(Protocol):
    def translate(self, text: str, source_lang: str, target_lang: str) -> str | None: ...
```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_translator.py
from pathlib import Path
import json
import pytest
from src.query.translation_cache import TranslationCache
from src.query.translator import (
    PrecomputedCacheTranslator,
    PassthroughTranslator,
    MultilingualQueryBuilder,
    create_translator,
)
from src.query.query_processor import ProcessedQuery

def test_translation_cache_lookup(tmp_path: Path):
    cache_file = tmp_path / "translations.jsonl"
    cache = TranslationCache(cache_file)
    cache.set("đau ngực", "en", "chest pain")
    
    # Reload from disk
    cache2 = TranslationCache(cache_file)
    assert cache2.get("đau ngực", "en") == "chest pain"
    assert cache2.get("đau ngực", "zh") is None

def test_query_builder_fallback_when_translation_missing(tmp_path: Path):
    cache_file = tmp_path / "translations.jsonl"
    cache = TranslationCache(cache_file)
    cache.set("sốt cao", "en", "high fever")
    # ZH is missing
    translator = PrecomputedCacheTranslator(cache)
    builder = MultilingualQueryBuilder(translator=translator, target_languages=["en", "zh"])
    variants = builder.build(text_vi="sốt cao", text_for_translation="sốt cao")

    assert variants.vi == "sốt cao"
    assert variants.en == "high fever"
    assert variants.zh is None  # Graceful fallback: zh stream will simply be skipped

def test_passthrough_translator():
    translator = PassthroughTranslator()
    assert translator.translate("đau đầu", "vi", "en") is None

def test_create_translator_factory(tmp_path: Path):
    cache_file = tmp_path / "trans.jsonl"
    cfg = {
        "query_understanding": {
            "translation": {
                "provider": "precomputed_cache",
                "cache_path": str(cache_file)
            }
        }
    }
    t = create_translator(cfg)
    assert isinstance(t, PrecomputedCacheTranslator)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_translator.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Triển khai:
1. `src/query/translation_cache.py`:
   - `TranslationCache`: Key là `f"{hash_or_clean_text}_{target_lang}"`, đọc/ghi JSONL `artifacts/cache/translations.jsonl`, in-memory cache dict để lookup $O(1)$. Thread-safe và append-only.
2. `src/query/translator.py`:
   - `Translator(Protocol)`: `translate(text: str, source_lang: str, target_lang: str) -> str | None`.
   - `PrecomputedCacheTranslator`: lookup từ `TranslationCache`. Nếu không có trong cache, trả về `None` (hoặc fallback).
   - `PassthroughTranslator`: luôn trả về `None`.
   - `LocalHfTranslator` (mockable / lazy load transformer pipeline, fallback on error).
   - `create_translator(config: dict) -> Translator`: factory dựa theo `config["query_understanding"]["translation"]["provider"]`.
   - `MultilingualQueryBuilder`:
     - Nhận `Translator` và `target_languages` (mặc định `["en", "zh"]`).
     - Hàm `build(processed: ProcessedQuery | str, text_for_translation: str | None = None) -> QueryVariants`:
       - Nếu input là `ProcessedQuery`, dùng `processed.condensed_vi or processed.normalized_text` làm `text_for_translation`.
       - Dịch sang từng ngôn ngữ đích qua `translator.translate(...)`.
       - Bọc trong `try...except` để nếu dịch lỗi ở bất kỳ ngôn ngữ nào thì gán `None` thay vì crash pipeline (Fault Tolerance!).
       - Trả về `QueryVariants(vi=..., vi_condensed=..., en=..., zh=...)`.
3. Export trong `src/query/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_translator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/query/translator.py src/query/translation_cache.py src/query/__init__.py tests/test_translator.py
git commit -m "feat: implement translation caching and multilingual query builder with fault tolerance"
```
