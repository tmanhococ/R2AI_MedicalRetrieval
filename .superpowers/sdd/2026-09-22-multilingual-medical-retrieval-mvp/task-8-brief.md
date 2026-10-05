### Task 8: Dense Indexing & Vector Search (FAISS & NumPy Fallback)

**Files:**
- Create: `src/indexing/dense_index.py`
- Create: `src/retrieval/dense_retriever.py`
- Test: `tests/test_dense_retriever.py`

**Interfaces:**
- Consumes: `ChunkRecord`, `SearchHit` from `src.data.schemas`
- Produces: `DenseIndexer`, `DenseRetriever`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dense_retriever.py
from pathlib import Path
import numpy as np
import pytest
from src.data.schemas import ChunkRecord
from src.indexing.dense_index import DenseIndexer
from src.retrieval.dense_retriever import DenseRetriever

class DummyEncoder:
    def __init__(self, dim=16):
        self.dim = dim
    def encode(self, texts, batch_size=32, normalize_embeddings=True):
        # Deterministic vector based on length and characters
        vecs = []
        for t in texts:
            val = float(len(t) % 10) / 10.0
            v = np.full(self.dim, val, dtype=np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(v)
                v = v / (norm if norm > 0 else 1.0)
            vecs.append(v)
        return np.array(vecs, dtype=np.float32)

def test_dense_indexing_and_search_numpy(tmp_path: Path):
    chunks = [
        ChunkRecord(chunk_id="c1", doc_id="d1", language="vi", text="đoạn văn bản một"),
        ChunkRecord(chunk_id="c2", doc_id="d2", language="vi", text="đoạn văn bản dài hơn nhiều hai")
    ]
    encoder = DummyEncoder()
    indexer = DenseIndexer(encoder=encoder, backend="numpy")
    index_dir = tmp_path / "dense_index"
    indexer.build_index(chunks=chunks, output_dir=index_dir)

    retriever = DenseRetriever(encoder=encoder, index_dir=index_dir, backend="numpy")
    chunk_hits = retriever.search_chunks(query="đoạn văn bản một", language="vi", top_k=2)
    assert len(chunk_hits) == 2
    assert chunk_hits[0].item_id in {"c1", "c2"}
    assert chunk_hits[0].item_type == "chunk"
    assert chunk_hits[0].source == "dense"
    assert chunk_hits[0].rank == 1

    # Document aggregation (max score)
    doc_hits = retriever.aggregate_documents(chunk_hits)
    assert len(doc_hits) == 2
    assert doc_hits[0].item_type == "document"
    assert doc_hits[0].doc_id in {"d1", "d2"}

def test_dense_instruction_prefix_support(tmp_path: Path):
    encoder = DummyEncoder()
    prefix_config = {"query": "query: ", "passage": "passage: "}
    indexer = DenseIndexer(encoder=encoder, backend="numpy", instruction_prefix=prefix_config)
    index_dir = tmp_path / "dense_prefix_index"
    chunks = [ChunkRecord(chunk_id="c1", doc_id="d1", language="vi", text="text")]
    indexer.build_index(chunks=chunks, output_dir=index_dir)

    retriever = DenseRetriever(encoder=encoder, index_dir=index_dir, backend="numpy", instruction_prefix=prefix_config)
    hits = retriever.search_chunks(query="test", language="vi", top_k=1)
    assert len(hits) == 1

def test_dense_empty_language_partition(tmp_path: Path):
    encoder = DummyEncoder()
    indexer = DenseIndexer(encoder=encoder, backend="numpy")
    index_dir = tmp_path / "dense_empty"
    indexer.build_index(chunks=[], output_dir=index_dir)

    retriever = DenseRetriever(encoder=encoder, index_dir=index_dir, backend="numpy")
    hits = retriever.search_chunks(query="test", language="en", top_k=5)
    assert hits == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dense_retriever.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

1. `src/indexing/dense_index.py`:
   - `DenseIndexer`:
     - Nhận `encoder` (hoặc model name / sentence-transformers callable), `backend: str = "faiss" | "numpy"`, `instruction_prefix: dict | None = None` (ví dụ `{"query": "query: ", "passage": "passage: "}`), `device: str = "auto"`, `normalize_embeddings: bool = True`.
     - `build_index(chunks: list[ChunkRecord], output_dir: Path | str)`:
       - Phân vùng chunk theo `language` (`vi`, `en`, `zh`).
       - Thêm `instruction_prefix["passage"]` vào text của từng chunk trước khi encode (nếu có prefix).
       - Tính embedding vector ma trận `(N, D)`.
       - Nếu `backend == "faiss"` (và faiss khả dụng): build `IndexFlatIP`.
       - Nếu `backend == "numpy"` (hoặc fallback): lưu ma trận `chunk_embeddings_{lang}.npy`.
       - Lưu `metadata_{lang}.jsonl` với `chunk_id`, `doc_id`, `language`, `chunk_index`.
       - Lưu `manifest.json`.
2. `src/retrieval/dense_retriever.py`:
   - `DenseRetriever`:
     - `encode_query(query: str) -> np.ndarray`: thêm `instruction_prefix["query"]` nếu có, chuẩn hóa vector ($L_2$ norm = 1).
     - `search_chunks(query: str, language: str, top_k: int) -> list[SearchHit]`:
       - Tính dot-product / cosine similarity (qua FAISS `index.search` hoặc NumPy `embeddings @ query_vec`).
       - Trả về `list[SearchHit]` với `source="dense"`, 1-indexed `rank`, sorted descending by score.
       - Trả về `[]` nếu language partition rỗng.
     - `aggregate_documents(chunk_hits: list[SearchHit]) -> list[SearchHit]`:
       - Nhóm chunk hits theo `doc_id`, lấy `max(raw_score)`.
       - Trả về `list[SearchHit]` với `item_type="document"`, `item_id=doc_id`, sorted descending by score, 1-indexed rank.
3. Export trong `src/indexing/__init__.py` và `src/retrieval/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dense_retriever.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/indexing/dense_index.py src/retrieval/dense_retriever.py src/indexing/__init__.py src/retrieval/__init__.py tests/test_dense_retriever.py
git commit -m "feat: implement dense vector indexing and retrieval with faiss and numpy backends"
```
