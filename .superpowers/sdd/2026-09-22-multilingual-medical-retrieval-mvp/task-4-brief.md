### Task 4: Dynamic Chunker Layer (Prechunked, Auto, Hybrid)

**Files:**
- Create: `src/indexing/__init__.py`
- Create: `src/indexing/chunker.py`
- Test: `tests/test_chunker.py`

**Interfaces:**
- Consumes: `DocumentRecord`, `ChunkRecord` from `src.data.schemas`
- Produces: `Chunker(Protocol)`, `AutoChunker`, `HybridChunker`, `PrechunkedAdapter`, `create_chunker(config: dict) -> Chunker`

```python
from typing import Protocol

class Chunker(Protocol):
    def chunk_documents(
        self,
        documents: list[DocumentRecord],
        existing_chunks: list[ChunkRecord] | None = None,
    ) -> list[ChunkRecord]: ...
```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunker.py
import pytest
from src.data.schemas import DocumentRecord, ChunkRecord
from src.indexing.chunker import AutoChunker, HybridChunker, PrechunkedChunker, create_chunker

def test_auto_chunker_sliding_window():
    long_text = " ".join([f"word{i}" for i in range(100)])
    doc = DocumentRecord(doc_id="doc_long", language="vi", text=long_text)
    chunker = AutoChunker(chunk_size=40, chunk_overlap=10, min_chunk_length=5)
    chunks = chunker.chunk_documents([doc])
    
    assert len(chunks) >= 3
    assert chunks[0].chunk_id == "doc_long_c0"
    assert chunks[0].doc_id == "doc_long"
    assert chunks[0].language == "vi"
    assert chunks[1].chunk_index == 1
    assert chunks[0].metadata is not None
    assert chunks[0].metadata.get("is_auto_generated") is True

def test_hybrid_chunker_preserves_existing_chunks():
    docs = [
        DocumentRecord(doc_id="d1", language="vi", text="Doc 1 text"),
        DocumentRecord(doc_id="d2", language="en", text="Doc 2 " * 30)
    ]
    existing_chunks = [ChunkRecord(chunk_id="d1_custom", doc_id="d1", language="vi", text="Doc 1 text")]
    chunker = HybridChunker(auto_chunker=AutoChunker(chunk_size=20, chunk_overlap=5))
    result = chunker.chunk_documents(docs, existing_chunks=existing_chunks)
    
    # d1 uses existing chunk, d2 gets auto chunked
    d1_chunks = [c for c in result if c.doc_id == "d1"]
    d2_chunks = [c for c in result if c.doc_id == "d2"]
    assert len(d1_chunks) == 1 and d1_chunks[0].chunk_id == "d1_custom"
    assert len(d2_chunks) > 1

def test_create_chunker_factory():
    cfg_auto = {
        "indexing": {
            "chunking": {
                "mode": "auto",
                "auto_strategy": {"chunk_size": 100, "chunk_overlap": 20, "min_chunk_length": 10}
            }
        }
    }
    c = create_chunker(cfg_auto)
    assert isinstance(c, AutoChunker)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chunker.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Triển khai:
1. `Chunker(Protocol)` với method `chunk_documents(documents: list[DocumentRecord], existing_chunks: list[ChunkRecord] | None = None) -> list[ChunkRecord]`
2. `AutoChunker`:
   - Token/word-based sliding window: chia theo khoảng trắng hoặc tokenizer (đơn giản, nhanh, ổn định).
   - Tham số: `chunk_size` (mặc định 512), `chunk_overlap` (mặc định 64), `min_chunk_length` (mặc định 50 ký tự hoặc 10 từ).
   - Format chunk_id: `f"{doc.doc_id}_c{idx}"`.
   - Gắn `metadata={"is_auto_generated": True}`.
3. `PrechunkedChunker`:
   - Trả về `existing_chunks` nếu có. Nếu không có trả về `[]` hoặc raise nếu strict.
4. `HybridChunker`:
   - Nếu document đã có chunk trong `existing_chunks`, giữ nguyên các chunk đó.
   - Nếu document chưa có chunk nào, dùng `AutoChunker` sinh chunk cho document đó.
5. Factory function: `create_chunker(config: dict) -> Chunker`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_chunker.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/indexing/chunker.py src/indexing/__init__.py tests/test_chunker.py
git commit -m "feat: implement dynamic chunking layer with auto sliding window and hybrid mode"
```
