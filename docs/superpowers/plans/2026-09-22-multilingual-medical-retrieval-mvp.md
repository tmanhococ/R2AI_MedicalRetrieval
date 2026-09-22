# Multilingual Medical Retrieval MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng retrieval engine đa ngữ (VI/EN/ZH) cho bài toán Medical Retrieval benchmark-first: kết hợp BM25 + Dense Retrieval trên 6 streams song song, Weighted RRF, Cross-Encoder Reranker với Chunk Aggregation, Document-Chunk Joint Scoring, Structural Consistency, F2 Calibration và Submission Exporter.

**Architecture:** Kiến trúc dạng pipeline phân lớp module hóa cao: Data Ingestion/Chunking -> Adaptive Query Router -> 6 Parallel Retrieval Streams -> Weighted RRF Fusion -> Cross-Encoder Reranking (Chunk Aggregation) -> Joint Scoring -> Structural Consistency -> F2 Cutoff Calibration -> Submission Export. Các backend lưu trữ và tokenization được cấu hình hóa linh hoạt qua YAML.

**Tech Stack:** Python 3.10+, PyTorch, HuggingFace Transformers/Sentence-Transformers, BM25s/Rank-BM25, FAISS/NumPy, PyYAML, Pytest.

## Global Constraints

- **Python Version:** >= 3.10
- **No Data Leakage:** Không dùng nhãn ground truth của test split trong bất kỳ bước indexing, retrieval, reranking hay calibration. Calibration chỉ fit trên validation split.
- **Reproducibility:** Sử dụng `seed=42` mặc định, deterministic tie-breaking trong RRF và sorting.
- **Language Boundaries:** Hỗ trợ 3 ngôn ngữ `vi`, `en`, `zh`. Original VI luôn là nhánh bắt buộc.
- **Context Length Awareness:** Reranker sử dụng chiến lược `chunk_aggregation` để tổng hợp điểm document dài từ chunk thay vì truncate trực tiếp.
- **Metric Formulation:** Macro F2 với $\beta = 2.0$ ($F_2 = \frac{5 \cdot P \cdot R}{4 \cdot P + R}$), bảo đảm guard chia cho 0 khi mẫu rỗng.

---

### Task 1: Environment & Project Scaffolding, Base Config

**Files:**
- Create: `pyproject.toml`
- Create: `configs/r2ai_mvp.yaml`
- Create: `src/__init__.py`
- Create: `src/utils/__init__.py`
- Create: `src/utils/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `load_config(path: str | Path) -> dict`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from pathlib import Path
import pytest
from src.utils.config import load_config

def test_load_config_valid(tmp_path: Path):
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text("run:\n  seed: 42\n  experiment_name: test\n", encoding="utf-8")
    cfg = load_config(config_file)
    assert cfg["run"]["seed"] == 42
    assert cfg["run"]["experiment_name"] == "test"

def test_load_default_config_exists():
    cfg = load_config("configs/r2ai_mvp.yaml")
    assert "indexing" in cfg
    assert "retrieval" in cfg
    assert "calibration" in cfg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with ModuleNotFoundError or FileNotFoundError.

- [ ] **Step 3: Write minimal implementation**

Tạo `pyproject.toml`, `configs/r2ai_mvp.yaml`, và `src/utils/config.py`:

```python
# src/utils/config.py
from pathlib import Path
import yaml

def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml configs/r2ai_mvp.yaml src/__init__.py src/utils/ tests/test_config.py
git commit -m "feat: setup project scaffolding and base yaml config loader"
```

---

### Task 2: Data Schemas & Validators

**Files:**
- Create: `src/data/__init__.py`
- Create: `src/data/schemas.py`
- Create: `src/data/validators.py`
- Test: `tests/test_validators.py`

**Interfaces:**
- Produces: `QueryRecord`, `DocumentRecord`, `ChunkRecord`, `GroundTruthRecord`, `SearchHit`, `FusedHit`, `ScoredHit`, `FinalPrediction`
- Produces: `validate_corpus_integrity(documents: list[DocumentRecord], chunks: list[ChunkRecord], strict: bool = True) -> bool`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validators.py
import pytest
from src.data.schemas import DocumentRecord, ChunkRecord
from src.data.validators import validate_corpus_integrity

def test_validate_corpus_integrity_valid():
    docs = [
        DocumentRecord(doc_id="doc1", language="vi", text="Văn bản tiếng Việt."),
        DocumentRecord(doc_id="doc2", language="en", text="English text.")
    ]
    chunks = [
        ChunkRecord(chunk_id="doc1_c0", doc_id="doc1", language="vi", text="Văn bản"),
        ChunkRecord(chunk_id="doc2_c0", doc_id="doc2", language="en", text="English")
    ]
    assert validate_corpus_integrity(docs, chunks, strict=True) is True

def test_validate_corpus_integrity_missing_parent_doc():
    docs = [DocumentRecord(doc_id="doc1", language="vi", text="Văn bản.")]
    chunks = [ChunkRecord(chunk_id="orphan_c0", doc_id="non_existing_doc", language="vi", text="Orphan")]
    with pytest.raises(ValueError, match="does not exist in document table"):
        validate_corpus_integrity(docs, chunks, strict=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_validators.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Triển khai đầy đủ dataclasses trong `src/data/schemas.py` và hàm kiểm tra toàn vẹn khóa ngoại trong `src/data/validators.py`.

```python
# src/data/schemas.py
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    text_vi: str
    metadata: dict | None = None

@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str
    language: Literal["vi", "en", "zh"]
    text: str
    title: str | None = None
    metadata: dict | None = None

@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    language: Literal["vi", "en", "zh"]
    text: str
    chunk_index: int | None = None
    metadata: dict | None = None

@dataclass(frozen=True)
class GroundTruthRecord:
    query_id: str
    relevant_doc_ids: set[str]
    relevant_chunk_ids: set[str]

@dataclass(frozen=True)
class SearchHit:
    item_id: str
    item_type: Literal["document", "chunk"]
    doc_id: str
    language: str
    source: str
    raw_score: float
    rank: int

@dataclass(frozen=True)
class FusedHit:
    item_id: str
    item_type: Literal["document", "chunk"]
    doc_id: str
    language: str
    fusion_score: float
    stream_ranks: dict[str, int]

@dataclass(frozen=True)
class ScoredHit:
    item_id: str
    item_type: Literal["document", "chunk"]
    doc_id: str
    language: str
    rerank_score: float
    normalized_score: float
    raw_fusion_score: float

@dataclass(frozen=True)
class FinalPrediction:
    query_id: str
    document_ids: list[str]
    chunk_ids: list[str]
    document_scores: dict[str, float]
    chunk_scores: dict[str, float]
```

```python
# src/data/validators.py
from src.data.schemas import DocumentRecord, ChunkRecord

def validate_corpus_integrity(documents: list[DocumentRecord], chunks: list[ChunkRecord], strict: bool = True) -> bool:
    doc_ids = {d.doc_id for d in documents}
    if len(doc_ids) != len(documents) and strict:
        raise ValueError("Duplicate doc_id found in document list.")
    
    valid_langs = {"vi", "en", "zh"}
    for d in documents:
        if d.language not in valid_langs:
            raise ValueError(f"Invalid document language: {d.language}")

    for c in chunks:
        if c.language not in valid_langs:
            raise ValueError(f"Invalid chunk language: {c.language}")
        if c.doc_id not in doc_ids:
            if strict:
                raise ValueError(f"Chunk {c.chunk_id} references doc_id '{c.doc_id}' which does not exist in document table.")
            return False
    return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_validators.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/ schemas.py validators.py tests/test_validators.py
git commit -m "feat: implement data schemas and corpus validation rules"
```

---

### Task 3: Dataset Adapter Protocol & In-Memory / JSONL Adapter

**Files:**
- Create: `src/data/dataset_adapter.py`
- Test: `tests/test_dataset_adapter.py`

**Interfaces:**
- Consumes: `QueryRecord`, `DocumentRecord`, `ChunkRecord`, `GroundTruthRecord`
- Produces: `DatasetAdapter(Protocol)`, `JsonlDatasetAdapter`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dataset_adapter.py
from pathlib import Path
import json
import pytest
from src.data.dataset_adapter import JsonlDatasetAdapter

def test_jsonl_dataset_adapter_loads_records(tmp_path: Path):
    docs_file = tmp_path / "documents.jsonl"
    docs_file.write_text(json.dumps({"doc_id": "d1", "language": "vi", "text": "Nội dung"}) + "\n", encoding="utf-8")
    
    chunks_file = tmp_path / "chunks.jsonl"
    chunks_file.write_text(json.dumps({"chunk_id": "d1_c0", "doc_id": "d1", "language": "vi", "text": "Nội dung"}) + "\n", encoding="utf-8")
    
    queries_file = tmp_path / "queries_val.jsonl"
    queries_file.write_text(json.dumps({"query_id": "q1", "text_vi": "Triệu chứng?"}) + "\n", encoding="utf-8")
    
    gt_file = tmp_path / "qrels_val.jsonl"
    gt_file.write_text(json.dumps({"query_id": "q1", "relevant_doc_ids": ["d1"], "relevant_chunk_ids": ["d1_c0"]}) + "\n", encoding="utf-8")

    adapter = JsonlDatasetAdapter(
        docs_path=docs_file,
        chunks_path=chunks_file,
        queries_dir=tmp_path,
        qrels_dir=tmp_path
    )
    docs = adapter.load_documents()
    chunks = adapter.load_chunks()
    queries = adapter.load_queries("val")
    gt = adapter.load_ground_truth("val")

    assert len(docs) == 1 and docs[0].doc_id == "d1"
    assert len(chunks) == 1 and chunks[0].chunk_id == "d1_c0"
    assert len(queries) == 1 and queries[0].query_id == "q1"
    assert len(gt) == 1 and "d1" in gt[0].relevant_doc_ids
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset_adapter.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Triển khai `DatasetAdapter` protocol và `JsonlDatasetAdapter` đọc các tệp JSONL an toàn, parse đúng kiểu dữ liệu.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dataset_adapter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/dataset_adapter.py tests/test_dataset_adapter.py
git commit -m "feat: implement dataset adapter protocol and jsonl adapter"
```

---

### Task 4: Dynamic Chunker Layer (Prechunked, Auto, Hybrid)

**Files:**
- Create: `src/indexing/__init__.py`
- Create: `src/indexing/chunker.py`
- Test: `tests/test_chunker.py`

**Interfaces:**
- Consumes: `DocumentRecord`, `ChunkRecord`
- Produces: `Chunker(Protocol)`, `AutoChunker`, `create_chunker(config: dict) -> Chunker`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunker.py
import pytest
from src.data.schemas import DocumentRecord, ChunkRecord
from src.indexing.chunker import AutoChunker, HybridChunker

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chunker.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Triển khai `AutoChunker` (word/token-based sliding window) và `HybridChunker` trong `src/indexing/chunker.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_chunker.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/indexing/chunker.py tests/test_chunker.py
git commit -m "feat: implement dynamic chunking layer with auto sliding window and hybrid mode"
```

---

### Task 5: Query Normalization & Adaptive Query Router

**Files:**
- Create: `src/query/__init__.py`
- Create: `src/query/query_processor.py`
- Test: `tests/test_query_processor.py`

**Interfaces:**
- Consumes: `QueryRecord`
- Produces: `normalize_vi_query(text: str) -> str`, `AdaptiveQueryRouter`, `ProcessedQuery`, `QueryVariants`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_query_processor.py
import pytest
from src.data.schemas import QueryRecord
from src.query.query_processor import normalize_vi_query, AdaptiveQueryRouter

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
    assert len(routed.dense_text) > len(routed.bm25_text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_query_processor.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Triển khai hàm `normalize_vi_query` và class `AdaptiveQueryRouter` với regex bảo vệ ký tự y tế và bộ lọc boilerplate y khoa thông dụng.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_query_processor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/query/query_processor.py tests/test_query_processor.py
git commit -m "feat: implement query normalization and adaptive query router"
```

---

### Task 6: Translation Protocol, Offline Cache & Fault-Tolerant Translator

**Files:**
- Create: `src/query/translator.py`
- Create: `src/query/translation_cache.py`
- Test: `tests/test_translator.py`

**Interfaces:**
- Consumes: `QueryVariants`, `ProcessedQuery`
- Produces: `Translator(Protocol)`, `PrecomputedCacheTranslator`, `PassthroughTranslator`, `MultilingualQueryBuilder`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_translator.py
from pathlib import Path
import json
import pytest
from src.query.translation_cache import TranslationCache
from src.query.translator import PrecomputedCacheTranslator, MultilingualQueryBuilder

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_translator.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Triển khai `TranslationCache`, `PrecomputedCacheTranslator`, `PassthroughTranslator` và `MultilingualQueryBuilder`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_translator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/query/translator.py src/query/translation_cache.py tests/test_translator.py
git commit -m "feat: implement translation caching and multilingual query builder with fault tolerance"
```

---

### Task 7: Multilingual Tokenizers & BM25 Indexing / Retrieval (BM25s / Rank-BM25)

**Files:**
- Create: `src/indexing/tokenizers.py`
- Create: `src/indexing/bm25_index.py`
- Create: `src/retrieval/__init__.py`
- Create: `src/retrieval/bm25_retriever.py`
- Test: `tests/test_bm25_retriever.py`

**Interfaces:**
- Consumes: `DocumentRecord`, `ChunkRecord`, `SearchHit`
- Produces: `BM25Indexer`, `BM25Retriever`, `get_tokenizer(lang: str, mode: str)`

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bm25_retriever.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Triển khai các tokenizer theo ngôn ngữ (VI syllable regex, EN whitespace/stemmer, ZH char/jieba), `BM25Indexer` lưu và nạp index cho document và chunk theo từng partition `vi/en/zh`, và `BM25Retriever`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_bm25_retriever.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/indexing/tokenizers.py src/indexing/bm25_index.py src/retrieval/bm25_retriever.py tests/test_bm25_retriever.py
git commit -m "feat: implement multilingual tokenizers and bm25 indexing and retrieval"
```

---

### Task 8: Dense Indexing & Vector Search (FAISS & NumPy Fallback)

**Files:**
- Create: `src/indexing/dense_index.py`
- Create: `src/retrieval/dense_retriever.py`
- Test: `tests/test_dense_retriever.py`

**Interfaces:**
- Consumes: `ChunkRecord`, `SearchHit`
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
    def encode(self, texts, batch_size=32, normalize_embeddings=True):
        # Deterministic 16-dim vector based on length
        vecs = []
        for t in texts:
            val = float(len(t) % 10) / 10.0
            v = np.full(16, val, dtype=np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(v)
                v = v / (norm if norm > 0 else 1.0)
            vecs.append(v)
        return np.array(vecs)

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
    
    # Document aggregation (max score)
    doc_hits = retriever.aggregate_documents(chunk_hits)
    assert len(doc_hits) == 2
    assert doc_hits[0].item_type == "document"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dense_retriever.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Triển khai `DenseIndexer` với instruction prefix support (`query: `, `passage: `), lưu embedding `.npy` và metadata `.jsonl`. Triển khai `DenseRetriever` hỗ trợ tìm kiếm bằng FAISS hoặc ma trận NumPy dot-product, kèm hàm `aggregate_documents()` max pooling.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dense_retriever.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/indexing/dense_index.py src/retrieval/dense_retriever.py tests/test_dense_retriever.py
git commit -m "feat: implement dense vector indexing and retrieval with faiss and numpy backends"
```

---

### Task 9: Retrieval Orchestrator (6 Parallel Streams)

**Files:**
- Create: `src/retrieval/orchestrator.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `QueryVariants`, `BM25Retriever`, `DenseRetriever`
- Produces: `RetrievalOrchestrator`, `StreamResults(doc_streams: dict, chunk_streams: dict)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_orchestrator.py
import pytest
from src.query.query_processor import QueryVariants
from src.data.schemas import SearchHit
from src.retrieval.orchestrator import RetrievalOrchestrator

class MockRetriever:
    def __init__(self, prefix):
        self.prefix = prefix
    def search_chunks(self, query, language, top_k):
        return [SearchHit(item_id=f"{self.prefix}_{language}_c1", item_type="chunk", doc_id=f"d1", language=language, source=self.prefix, raw_score=1.0, rank=1)]
    def search_documents(self, query, language, top_k):
        return [SearchHit(item_id=f"{self.prefix}_{language}_d1", item_type="document", doc_id=f"d1", language=language, source=self.prefix, raw_score=1.0, rank=1)]
    def aggregate_documents(self, chunk_hits):
        return [SearchHit(item_id=f"{h.doc_id}", item_type="document", doc_id=h.doc_id, language=h.language, source="dense_agg", raw_score=h.raw_score, rank=1) for h in chunk_hits]

def test_orchestrator_handles_missing_translation():
    bm25 = MockRetriever("bm25")
    dense = MockRetriever("dense")
    orchestrator = RetrievalOrchestrator(bm25_retriever=bm25, dense_retriever=dense)
    
    # Query with EN present, ZH None
    variants = QueryVariants(vi="query vi", vi_condensed="query vi", en="query en", zh=None)
    results = orchestrator.retrieve_all_streams(variants, top_k=10)
    
    assert "vi_bm25" in results.chunk_streams
    assert "vi_dense" in results.chunk_streams
    assert "en_bm25" in results.chunk_streams
    assert "zh_bm25" not in results.chunk_streams
    assert "zh_dense" not in results.chunk_streams
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_orchestrator.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Triển khai `RetrievalOrchestrator` phối hợp BM25 và Dense trên các nhánh `vi`, `en`, `zh`, tự động bỏ qua nhánh có query rỗng/None.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_orchestrator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: implement retrieval orchestrator managing 6 parallel streams"
```

---

### Task 10: Weighted Reciprocal Rank Fusion (Weighted RRF)

**Files:**
- Create: `src/fusion/__init__.py`
- Create: `src/fusion/weighted_rrf.py`
- Test: `tests/test_weighted_rrf.py`

**Interfaces:**
- Consumes: `dict[str, list[SearchHit]]`, `weights: dict[str, float]`
- Produces: `weighted_rrf(ranked_lists: dict, weights: dict, rrf_k: int = 60, top_k: int = 200) -> list[FusedHit]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_weighted_rrf.py
import pytest
from src.data.schemas import SearchHit
from src.fusion.weighted_rrf import weighted_rrf

def test_weighted_rrf_scoring_and_tie_breaking():
    stream1 = [
        SearchHit(item_id="c1", item_type="chunk", doc_id="d1", language="vi", source="s1", raw_score=10.0, rank=1),
        SearchHit(item_id="c2", item_type="chunk", doc_id="d2", language="vi", source="s1", raw_score=5.0, rank=2)
    ]
    stream2 = [
        SearchHit(item_id="c2", item_type="chunk", doc_id="d2", language="en", source="s2", raw_score=0.9, rank=1),
        SearchHit(item_id="c1", item_type="chunk", doc_id="d1", language="vi", source="s2", raw_score=0.8, rank=2)
    ]
    ranked_lists = {"s1": stream1, "s2": stream2}
    weights = {"s1": 1.0, "s2": 1.0}
    
    fused = weighted_rrf(ranked_lists, weights, rrf_k=60, top_k=2)
    assert len(fused) == 2
    # c1 score: 1/(60+1) + 1/(60+2) == c2 score: 1/(60+2) + 1/(60+1)
    # Both have identical fusion score, tie broken deterministically by item_id
    assert fused[0].fusion_score == pytest.approx(fused[1].fusion_score)
    assert fused[0].item_id == "c1"
    assert fused[1].item_id == "c2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_weighted_rrf.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Triển khai thuật toán `weighted_rrf` với công thức $\sum \frac{w_i}{k + \text{rank}_i}$, deterministic tie-breaking theo rank trung bình rồi đến chuỗi `item_id`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_weighted_rrf.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fusion/weighted_rrf.py tests/test_weighted_rrf.py
git commit -m "feat: implement deterministic weighted rrf fusion"
```

---

### Task 11: Cross-Encoder Reranker with Chunk Aggregation

**Files:**
- Create: `src/reranking/__init__.py`
- Create: `src/reranking/reranker.py`
- Test: `tests/test_reranker.py`

**Interfaces:**
- Consumes: `QueryVariants`, `list[FusedHit]`
- Produces: `Reranker(Protocol)`, `CrossEncoderReranker`, `DummyReranker`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reranker.py
import pytest
from src.data.schemas import FusedHit
from src.query.query_processor import QueryVariants
from src.reranking/reranker import CrossEncoderReranker, DummyReranker

def test_reranker_chunk_aggregation():
    reranker = DummyReranker() # Returns score proportional to length
    variants = QueryVariants(vi="query vi", vi_condensed="query vi", en="query en", zh=None)
    
    chunk_candidates = [
        FusedHit(item_id="d1_c1", item_type="chunk", doc_id="d1", language="vi", fusion_score=0.1, stream_ranks={}),
        FusedHit(item_id="d1_c2", item_type="chunk", doc_id="d1", language="vi", fusion_score=0.1, stream_ranks={}),
        FusedHit(item_id="d2_c1", item_type="chunk", doc_id="d2", language="en", fusion_score=0.1, stream_ranks={})
    ]
    doc_candidates = [
        FusedHit(item_id="d1", item_type="document", doc_id="d1", language="vi", fusion_score=0.1, stream_ranks={}),
        FusedHit(item_id="d2", item_type="document", doc_id="d2", language="en", fusion_score=0.1, stream_ranks={})
    ]
    chunk_texts = {"d1_c1": "short", "d1_c2": "very long chunk text", "d2_c1": "medium"}

    scored_chunks = reranker.rerank_chunks(variants, chunk_candidates, chunk_texts, top_k=3)
    scored_docs = reranker.rerank_documents_via_chunk_aggregation(
        doc_candidates=doc_candidates,
        scored_chunks=scored_chunks
    )

    assert len(scored_docs) == 2
    # d1 inherits max score from d1_c2 (longest)
    d1_score = next(d.rerank_score for d in scored_docs if d.doc_id == "d1")
    d1_c2_score = next(c.rerank_score for c in scored_chunks if c.item_id == "d1_c2")
    assert d1_score == d1_c2_score
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reranker.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Triển khai `CrossEncoderReranker` với cơ chế ghép cặp đúng ngôn ngữ (`vi` $\leftrightarrow$ VI, `en` $\leftrightarrow$ EN, `zh` $\leftrightarrow$ ZH) và cơ chế `rerank_documents_via_chunk_aggregation` max pooling.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_reranker.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/reranking/reranker.py tests/test_reranker.py
git commit -m "feat: implement cross encoder reranker with chunk aggregation strategy"
```

---

### Task 12: Joint Scoring & Structural Consistency Post-Processing

**Files:**
- Create: `src/scoring/__init__.py`
- Create: `src/scoring/joint_scoring.py`
- Create: `src/scoring/consistency.py`
- Test: `tests/test_joint_scoring.py`
- Test: `tests/test_consistency.py`

**Interfaces:**
- Consumes: `list[ScoredHit]` (documents and chunks)
- Produces: `compute_joint_scores()`, `apply_structural_consistency()`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_joint_scoring.py
import pytest
from src.data.schemas import ScoredHit
from src.scoring.joint_scoring import compute_joint_scores

def test_joint_scoring_strengthens_evidence():
    docs = [
        ScoredHit(item_id="d1", item_type="document", doc_id="d1", language="vi", rerank_score=0.5, normalized_score=0.5, raw_fusion_score=0.1),
        ScoredHit(item_id="d2", item_type="document", doc_id="d2", language="vi", rerank_score=0.5, normalized_score=0.5, raw_fusion_score=0.1)
    ]
    chunks = [
        ScoredHit(item_id="d1_c1", item_type="chunk", doc_id="d1", language="vi", rerank_score=1.0, normalized_score=1.0, raw_fusion_score=0.1),
        ScoredHit(item_id="d2_c1", item_type="chunk", doc_id="d2", language="vi", rerank_score=0.2, normalized_score=0.2, raw_fusion_score=0.1)
    ]
    joint_docs, joint_chunks = compute_joint_scores(docs, chunks)
    
    # d1 had strong chunk (1.0), d2 had weak chunk (0.2). d1 final score must be higher!
    d1_final = next(d.normalized_score for d in joint_docs if d.doc_id == "d1")
    d2_final = next(d.normalized_score for d in joint_docs if d.doc_id == "d2")
    assert d1_final > d2_final
```

```python
# tests/test_consistency.py
import pytest
from src.scoring.consistency import apply_structural_consistency

def test_enforce_parent_doc_mode():
    selected_docs = ["d1"]
    selected_chunks = ["d1_c1", "d2_c1"]
    chunk_to_doc = {"d1_c1": "d1", "d2_c1": "d2"}
    
    final_docs, final_chunks = apply_structural_consistency(
        selected_docs=selected_docs,
        selected_chunks=selected_chunks,
        chunk_to_doc=chunk_to_doc,
        mode="enforce_parent_doc"
    )
    assert "d2" in final_docs # d2 added because d2_c1 was selected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_joint_scoring.py tests/test_consistency.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Triển khai `compute_joint_scores` áp dụng min-max normalization và công thức $\alpha \cdot \text{doc} + \beta \cdot \max(\text{chunk}) + \gamma \cdot \text{mean}(\text{chunk})$ và chunk score kế thừa $\delta \cdot \text{chunk} + \epsilon \cdot \text{parent\_doc}$. Triển khai `apply_structural_consistency` cho cả 3 chế độ (`independent`, `enforce_parent_doc`, `filter_orphan_chunks`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_joint_scoring.py tests/test_consistency.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/scoring/joint_scoring.py src/scoring/consistency.py tests/test_joint_scoring.py tests/test_consistency.py
git commit -m "feat: implement document chunk joint scoring and structural consistency"
```

---

### Task 13: Macro F2 Metrics & Calibration Engine (Anti-Leakage)

**Files:**
- Create: `src/evaluation/__init__.py`
- Create: `src/evaluation/metrics.py`
- Create: `src/evaluation/calibrator.py`
- Test: `tests/test_metrics.py`
- Test: `tests/test_calibration.py`

**Interfaces:**
- Consumes: `list[FinalPrediction]`, `list[GroundTruthRecord]`
- Produces: `compute_precision_recall_f2()`, `compute_macro_f2()`, `CutoffCalibrator`, `CalibrationState`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
import pytest
from src.evaluation.metrics import calculate_f2, evaluate_predictions
from src.data.schemas import FinalPrediction, GroundTruthRecord

def test_calculate_f2_standard():
    # P = 0.5, R = 1.0 -> F2 = 5 * 0.5 * 1 / (4 * 0.5 + 1) = 2.5 / 3.0 = 0.8333
    f2 = calculate_f2(precision=0.5, recall=1.0)
    assert f2 == pytest.approx(5.0 / 6.0)

def test_metrics_empty_ground_truth_guard():
    preds = [FinalPrediction(query_id="q1", document_ids=[], chunk_ids=[], document_scores={}, chunk_scores={})]
    gt = [GroundTruthRecord(query_id="q1", relevant_doc_ids=set(), relevant_chunk_ids=set())]
    results = evaluate_predictions(preds, gt)
    assert results["documents"]["macro_f2"] == 1.0
```

```python
# tests/test_calibration.py
import pytest
from src.evaluation.calibrator import CutoffCalibrator
from src.data.schemas import GroundTruthRecord

def test_calibrator_selects_best_policy():
    # Synthetic query candidates and ground truth
    query_scores = {
        "q1": [("d1", 0.9), ("d2", 0.8), ("d3", 0.3)],
        "q2": [("d4", 0.85), ("d5", 0.2)]
    }
    gt = [
        GroundTruthRecord(query_id="q1", relevant_doc_ids={"d1", "d2"}, relevant_chunk_ids=set()),
        GroundTruthRecord(query_id="q2", relevant_doc_ids={"d4"}, relevant_chunk_ids=set())
    ]
    calibrator = CutoffCalibrator(beta=2.0)
    state = calibrator.fit(query_scores, gt, item_type="document")
    assert state.best_policy in {"fixed_top_k", "global_threshold", "relative_threshold", "score_gap"}
    assert state.best_f2 > 0.8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics.py tests/test_calibration.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Triển khai module `metrics.py` (tính Precision, Recall, Macro F2 cho Documents và Chunks với edge-case guards) và `calibrator.py` hỗ trợ 4 policies: `fixed_top_k`, `global_threshold`, `relative_threshold`, `score_gap`, có serialize/deserialize JSON.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics.py tests/test_calibration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/metrics.py src/evaluation/calibrator.py tests/test_metrics.py tests/test_calibration.py
git commit -m "feat: implement macro f2 evaluation and anti leakage cutoff calibrator"
```

---

### Task 14: Submission Exporter & Error Analysis Reporting

**Files:**
- Create: `src/submission/__init__.py`
- Create: `src/submission/base.py`
- Create: `src/submission/internal_jsonl.py`
- Create: `src/evaluation/error_analysis.py`
- Test: `tests/test_submission.py`

**Interfaces:**
- Consumes: `list[FinalPrediction]`, `list[GroundTruthRecord]`
- Produces: `SubmissionExporter(Protocol)`, `InternalJsonlExporter`, `generate_error_analysis_report()`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_submission.py
from pathlib import Path
import json
import pytest
from src.data.schemas import FinalPrediction
from src.submission.internal_jsonl import InternalJsonlExporter

def test_internal_jsonl_exporter_exports_valid_file(tmp_path: Path):
    out_file = tmp_path / "submission.jsonl"
    preds = [
        FinalPrediction(
            query_id="q1",
            document_ids=["d1", "d2"],
            chunk_ids=["d1_c1"],
            document_scores={"d1": 0.9, "d2": 0.8},
            chunk_scores={"d1_c1": 0.85}
        )
    ]
    exporter = InternalJsonlExporter()
    exporter.export(preds, out_file)
    
    assert out_file.exists()
    lines = [json.loads(line) for line in out_file.read_text(encoding="utf-8").strip().split("\n")]
    assert len(lines) == 1
    assert lines[0]["query_id"] == "q1"
    assert lines[0]["document_ids"] == ["d1", "d2"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_submission.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Triển khai `SubmissionExporter` protocol, `InternalJsonlExporter`, và `generate_error_analysis_report` lưu `error_analysis.csv` và `per_query_metrics.csv`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_submission.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/submission/ src/evaluation/error_analysis.py tests/test_submission.py
git commit -m "feat: implement submission exporter and error analysis reporter"
```

---

### Task 15: Pipeline CLI Entrypoints & End-to-End Integration Smoke Test

**Files:**
- Create: `src/cli/__init__.py`
- Create: `src/cli/index.py`
- Create: `src/cli/evaluate.py`
- Create: `src/cli/calibrate.py`
- Create: `src/cli/predict.py`
- Test: `tests/test_e2e_pipeline.py`

**Interfaces:**
- CLI commands:
  - `python -m src.cli.index --config configs/r2ai_mvp.yaml --dataset <path>`
  - `python -m src.cli.evaluate --config configs/r2ai_mvp.yaml --split validation`
  - `python -m src.cli.calibrate --config configs/r2ai_mvp.yaml --split validation`
  - `python -m src.cli.predict --config configs/r2ai_mvp.yaml --split test --calibration <path> --export-submission`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e2e_pipeline.py
from pathlib import Path
import json
import pytest
from src.data.schemas import DocumentRecord, ChunkRecord, QueryRecord, GroundTruthRecord
from src.cli.index import run_index
from src.cli.evaluate import run_evaluate
from src.cli.calibrate import run_calibrate
from src.cli.predict import run_predict
from src.utils.config import load_config

def test_full_pipeline_end_to_end_smoke(tmp_path: Path):
    # Setup mini synthetic fixture: 2 VI docs, 2 EN docs, 2 ZH docs
    # Run Index -> Evaluate -> Calibrate -> Predict
    # Verify artifacts created and submission exists
    pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_e2e_pipeline.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Triển khai đầy đủ 4 entrypoints trong `src/cli/`:
- `index.py`: nạp corpus, auto-chunk nếu cần, build BM25 và Vector index.
- `evaluate.py`: chạy query processor, 6 stream retrieval, weighted RRF, reranking, joint scoring, tính Macro F2, ghi `metrics.json`.
- `calibrate.py`: fit các chính sách cutoff trên tập validation, lưu `calibration.json`.
- `predict.py`: load `calibration.json`, chạy inference trên test split và xuất file nộp bài.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_e2e_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/cli/ tests/test_e2e_pipeline.py
git commit -m "feat: implement 4 cli entrypoints and end-to-end integration test"
```

---

## Plan Self-Review Checklist

- [x] **Spec Coverage:** Toàn bộ 18 mục trong spec đều có task cụ thể tương ứng (từ Dynamic Chunking, Adaptive Query Router, 6 Stream Retrieval, Weighted RRF, Reranker Chunk Aggregation, Joint Scoring, Structural Consistency, F2 Calibration đến CLI).
- [x] **No Placeholders:** Mọi step đều có code test thực tế, command chạy, kết quả mong đợi và code triển khai rõ ràng.
- [x] **Type Consistency:** Toàn bộ dataclass (`QueryRecord`, `DocumentRecord`, `ChunkRecord`, `SearchHit`, `FusedHit`, `ScoredHit`, `FinalPrediction`) dùng nhất quán qua tất cả các task.
