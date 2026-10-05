### Task 5: Query Normalization & Adaptive Query Router

**Files:**
- Create: `src/query/__init__.py`
- Create: `src/query/query_processor.py`
- Test: `tests/test_query_processor.py`

**Interfaces:**
- Consumes: `QueryRecord` from `src.data.schemas`
- Produces: `normalize_vi_query(text: str) -> str`, `AdaptiveQueryRouter`, `ProcessedQuery`, `QueryVariants`

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ProcessedQuery:
    query_id: str
    original_text: str
    normalized_text: str
    bm25_text: str
    dense_text: str
    is_clinical_case: bool
    word_count: int

@dataclass(frozen=True)
class QueryVariants:
    vi: str
    vi_condensed: str | None
    en: str | None = None
    zh: str | None = None
```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_query_processor.py
import pytest
from src.data.schemas import QueryRecord
from src.query.query_processor import normalize_vi_query, AdaptiveQueryRouter, ProcessedQuery, QueryVariants

def test_normalize_vi_query_preserves_medical_terms():
    raw = "  Bệnh nhân bị   HA: 140/90 mmHg,  ĐTĐ type 2 ,  HIV(+) ?? "
    cleaned = normalize_vi_query(raw)
    assert "HA: 140/90 mmHg" in cleaned
    assert "ĐTĐ type 2" in cleaned
    assert "HIV(+)" in cleaned
    assert "  " not in cleaned

def test_adaptive_query_router_short_query():
    router = AdaptiveQueryRouter(length_threshold=50)
    query = QueryRecord(query_id="q1", text_vi="Thuốc điều trị tăng huyết áp ban đầu?")
    routed = router.route(query)
    assert routed.is_clinical_case is False
    assert routed.bm25_text == routed.dense_text
    assert routed.word_count <= 50

def test_adaptive_query_router_long_clinical_case():
    router = AdaptiveQueryRouter(length_threshold=20, strip_clinical_boilerplate=True)
    long_text = "Bệnh nhân nam 54 tuổi vào viện vì lý do đau tức ngực trái lan ra vai sau khi gắng sức, tiền sử ghi nhận tăng huyết áp 5 năm đang dùng Amlodipine 5mg, sau đó được chỉ định xét nghiệm men tim Troponin T tăng cao."
    query = QueryRecord(query_id="q2", text_vi=long_text)
    routed = router.route(query)
    assert routed.is_clinical_case is True
    # Boilerplate removed in bm25_text
    assert "bệnh nhân nam" not in routed.bm25_text.lower()
    assert "Amlodipine" in routed.bm25_text
    assert "Troponin T" in routed.bm25_text
    # Dense retains context
    assert len(routed.dense_text) >= len(routed.bm25_text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_query_processor.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Triển khai:
1. `normalize_vi_query(text: str) -> str`:
   - Unicode NFC normalization.
   - Trim spaces.
   - Medical Rule Guard: Giữ nguyên số, đơn vị, mã bệnh, tên thuốc, abbreviation chuyên ngành:
     `HA`, `ĐTĐ`, `THA`, `COPD`, `mg`, `ml`, `mmol/L`, `mmHg`, `(+)`, `(-)`.
2. `AdaptiveQueryRouter`:
   - Config: `length_threshold` (mặc định 50 từ), `strip_clinical_boilerplate` (mặc định True), `dense_strategy` ("head_tail" hoặc "full", mặc định "head_tail"), `max_dense_tokens` (mặc định 512).
   - Bộ lọc boilerplate y khoa (regex/list): loại bỏ các cụm như *"bệnh nhân nam/nữ"*, *"năm nay... tuổi"*, *"vào viện vì lý do"*, *"tiền sử ghi nhận"*, *"sau đó được chỉ định"...* khỏi `bm25_text`.
   - Đối với dense: nếu query dài > threshold và `dense_strategy == "head_tail"`, giữ nguyên phần đầu và phần kết luận cuối (hoặc toàn bộ nếu < 512 tokens).
   - Trả về `ProcessedQuery`.
3. Export qua `src/query/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_query_processor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/query/query_processor.py src/query/__init__.py tests/test_query_processor.py
git commit -m "feat: implement query normalization and adaptive query router"
```
