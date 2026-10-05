### Task 3: Dataset Adapter Protocol & In-Memory / JSONL Adapter

**Files:**
- Create: `src/data/dataset_adapter.py`
- Test: `tests/test_dataset_adapter.py`

**Interfaces:**
- Consumes: `QueryRecord`, `DocumentRecord`, `ChunkRecord`, `GroundTruthRecord` from `src.data.schemas`
- Produces: `DatasetAdapter(Protocol)`, `JsonlDatasetAdapter`

```python
from typing import Protocol

class DatasetAdapter(Protocol):
    def load_documents(self) -> list[DocumentRecord]: ...
    def load_chunks(self) -> list[ChunkRecord]: ...
    def load_queries(self, split: str) -> list[QueryRecord]: ...
    def load_ground_truth(self, split: str) -> list[GroundTruthRecord]: ...
```

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

def test_jsonl_dataset_adapter_file_not_found():
    adapter = JsonlDatasetAdapter(
        docs_path="non_existent/docs.jsonl",
        chunks_path=None
    )
    with pytest.raises(FileNotFoundError):
        adapter.load_documents()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset_adapter.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Triển khai `DatasetAdapter` protocol và `JsonlDatasetAdapter` trong `src/data/dataset_adapter.py`:
- `load_documents()` -> đọc `docs_path`
- `load_chunks()` -> đọc `chunks_path` nếu có, nếu `chunks_path is None` trả về `[]`
- `load_queries(split)` -> tìm file `queries_{split}.jsonl` trong `queries_dir`
- `load_ground_truth(split)` -> tìm file `qrels_{split}.jsonl` hoặc `ground_truth_{split}.jsonl` trong `qrels_dir`, parse `relevant_doc_ids: set[str]` và `relevant_chunk_ids: set[str]`
- Re-export `DatasetAdapter` và `JsonlDatasetAdapter` trong `src/data/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dataset_adapter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/dataset_adapter.py src/data/__init__.py tests/test_dataset_adapter.py
git commit -m "feat: implement dataset adapter protocol and jsonl adapter"
```
