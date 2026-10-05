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
git add src/data/schemas.py src/data/validators.py src/data/__init__.py tests/test_validators.py
git commit -m "feat: implement data schemas and corpus validation rules"
```
