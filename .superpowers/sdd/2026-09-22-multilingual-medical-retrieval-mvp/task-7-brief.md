### Task 7: Multilingual Tokenizers & BM25 Indexing / Retrieval (BM25s / Rank-BM25)

**Files:**
- Create: `src/indexing/tokenizers.py`
- Create: `src/indexing/bm25_index.py`
- Create: `src/retrieval/__init__.py`
- Create: `src/retrieval/bm25_retriever.py`
- Test: `tests/test_bm25_retriever.py`

**Interfaces:**
- Consumes: `DocumentRecord`, `ChunkRecord`, `SearchHit` from `src.data.schemas`
- Produces: `get_tokenizer(lang: str, mode: str | None = None) -> Callable[[str], list[str]]`
- Produces: `BM25Indexer`
- Produces: `BM25Retriever`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bm25_retriever.py
from pathlib import Path
import pytest
from src.data.schemas import DocumentRecord, ChunkRecord
from src.indexing.bm25_index import BM25Indexer
from src.retrieval.bm25_retriever import BM25Retriever

def test_bm25_indexing_and_search(tmp_path: Path):
    docs = [
        DocumentRecord(doc_id="d1", language="vi", text="Bệnh nhân đái tháo đường cần xét nghiệm HbA1c."),
        DocumentRecord(doc_id="d2", language="vi", text="Cao huyết áp vô căn điều trị bằng Amlodipine.")
    ]
    chunks = [
        ChunkRecord(chunk_id="c1", doc_id="d1", language="vi", text="đái tháo đường xét nghiệm HbA1c"),
        ChunkRecord(chunk_id="c2", doc_id="d2", language="vi", text="huyết áp điều trị Amlodipine")
    ]
    indexer = BM25Indexer(backend="rank_bm25") # test with standard backend
    index_dir = tmp_path / "bm25_index"
    indexer.build_indexes(documents=docs, chunks=chunks, output_dir=index_dir)

    retriever = BM25Retriever(index_dir=index_dir, backend="rank_bm25")
    hits = retriever.search_chunks(query="HbA1c đái tháo đường", language="vi", top_k=5)
    
    assert len(hits) > 0
    assert hits[0].item_id == "c1"
    assert hits[0].doc_id == "d1"
    assert hits[0].raw_score > 0

def test_bm25_search_documents(tmp_path: Path):
    docs = [
        DocumentRecord(doc_id="d1", language="vi", text="Bệnh nhân đái tháo đường cần xét nghiệm HbA1c."),
        DocumentRecord(doc_id="d2", language="vi", text="Cao huyết áp vô căn điều trị bằng Amlodipine.")
    ]
    indexer = BM25Indexer(backend="rank_bm25")
    index_dir = tmp_path / "bm25_index"
    indexer.build_indexes(documents=docs, chunks=[], output_dir=index_dir)

    retriever = BM25Retriever(index_dir=index_dir, backend="rank_bm25")
    hits = retriever.search_documents(query="đái tháo đường", language="vi", top_k=2)
    assert len(hits) > 0
    assert hits[0].item_id == "d1"

def test_bm25_empty_partition_graceful(tmp_path: Path):
    indexer = BM25Indexer(backend="rank_bm25")
    index_dir = tmp_path / "bm25_index"
    indexer.build_indexes(documents=[], chunks=[], output_dir=index_dir)

    retriever = BM25Retriever(index_dir=index_dir, backend="rank_bm25")
    hits = retriever.search_chunks(query="test", language="en", top_k=5)
    assert hits == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bm25_retriever.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

1. `src/indexing/tokenizers.py`:
   - `get_tokenizer(lang: str, mode: str | None = None)`:
     - `vi`: `"syllable"` (regex `r"\w+"` lowercase) hoặc `"pyvi"` (nếu có, fallback syllable).
     - `en`: `"whitespace"` hoặc regex `r"\b\w+\b"` lowercase.
     - `zh`: `"jieba"` (nếu có, fallback char-level) hoặc `"char"` (regex `r"[\u4e00-\u9fff]|[a-zA-Z0-9]+"`).
2. `src/indexing/bm25_index.py`:
   - `BM25Indexer(backend: str = "bm25s" | "rank_bm25", tokenizers_config: dict | None = None)`:
     - Phân vùng theo `language`: `vi`, `en`, `zh`.
     - Lưu index riêng cho `documents` và `chunks` của từng ngôn ngữ (vd: `vi_documents.pkl` hoặc bm25s format + `metadata.jsonl`).
     - Hỗ trợ lưu và load deterministic.
3. `src/retrieval/bm25_retriever.py`:
   - `BM25Retriever(index_dir: Path | str, backend: str = "bm25s" | "rank_bm25", tokenizers_config: dict | None = None)`:
     - `search_documents(query: str, language: str, top_k: int) -> list[SearchHit]`
     - `search_chunks(query: str, language: str, top_k: int) -> list[SearchHit]`
     - Trả về danh sách `SearchHit` với `raw_score`, `rank` (1-indexed), `item_id`, `item_type`, `doc_id`, `language`, `source="bm25"`.
     - Nếu language partition không có dữ liệu hoặc index không tồn tại: trả về `[]` an toàn (không crash).
4. Re-export trong `src/retrieval/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_bm25_retriever.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/indexing/tokenizers.py src/indexing/bm25_index.py src/retrieval/__init__.py src/retrieval/bm25_retriever.py tests/test_bm25_retriever.py
git commit -m "feat: implement multilingual tokenizers and bm25 indexing and retrieval"
```
