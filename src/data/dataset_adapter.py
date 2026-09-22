import json
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from src.data.schemas import (
    ChunkRecord,
    DocumentRecord,
    GroundTruthRecord,
    QueryRecord,
)


@runtime_checkable
class DatasetAdapter(Protocol):
    """Protocol defining the standard interface for dataset loaders."""

    def load_documents(self) -> list[DocumentRecord]: ...

    def load_chunks(self) -> list[ChunkRecord]: ...

    def load_queries(self, split: str) -> list[QueryRecord]: ...

    def load_ground_truth(self, split: str) -> list[GroundTruthRecord]: ...


def _normalize_language(lang: Any) -> Literal["vi", "en", "zh"]:
    cleaned = str(lang).strip().lower()
    mapping: dict[str, Literal["vi", "en", "zh"]] = {
        "vi": "vi",
        "vietnamese": "vi",
        "en": "en",
        "english": "en",
        "zh": "zh",
        "chinese": "zh",
    }
    if cleaned in mapping:
        return mapping[cleaned]
    raise ValueError(
        f"Invalid language '{lang}'. Must normalize to 'vi', 'en', or 'zh'."
    )


class JsonlDatasetAdapter:
    """Dataset adapter loading corpus, queries, and ground truth from JSONL files."""

    def __init__(
        self,
        docs_path: str | Path | None = None,
        chunks_path: str | Path | None = None,
        queries_dir: str | Path | None = None,
        qrels_dir: str | Path | None = None,
    ) -> None:
        self.docs_path = Path(docs_path) if docs_path is not None else None
        self.chunks_path = Path(chunks_path) if chunks_path is not None else None
        self.queries_dir = Path(queries_dir) if queries_dir is not None else None
        self.qrels_dir = Path(qrels_dir) if qrels_dir is not None else None

    def load_documents(self) -> list[DocumentRecord]:
        if self.docs_path is None or not self.docs_path.exists():
            raise FileNotFoundError(f"Documents file not found: {self.docs_path}")
        if not self.docs_path.is_file():
            raise ValueError(f"Documents path is not a file: {self.docs_path}")

        documents: list[DocumentRecord] = []
        with open(self.docs_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                metadata = data.get("metadata")
                documents.append(
                    DocumentRecord(
                        doc_id=str(data["doc_id"]),
                        language=_normalize_language(data["language"]),
                        text=str(data["text"]),
                        title=str(data["title"]) if data.get("title") is not None else None,
                        metadata=metadata if isinstance(metadata, dict) else None,
                    )
                )
        return documents

    def load_chunks(self) -> list[ChunkRecord]:
        if self.chunks_path is None:
            return []
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {self.chunks_path}")
        if not self.chunks_path.is_file():
            raise ValueError(f"Chunks path is not a file: {self.chunks_path}")

        chunks: list[ChunkRecord] = []
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                chunk_index = data.get("chunk_index")
                metadata = data.get("metadata")
                chunks.append(
                    ChunkRecord(
                        chunk_id=str(data["chunk_id"]),
                        doc_id=str(data["doc_id"]),
                        language=_normalize_language(data["language"]),
                        text=str(data["text"]),
                        chunk_index=int(chunk_index) if chunk_index is not None else None,
                        metadata=metadata if isinstance(metadata, dict) else None,
                    )
                )
        return chunks

    def load_queries(self, split: str) -> list[QueryRecord]:
        if self.queries_dir is None:
            raise FileNotFoundError("queries_dir is not specified")

        candidate_file: Path | None = None
        if self.queries_dir.is_file():
            candidate_file = self.queries_dir
        else:
            p1 = self.queries_dir / f"queries_{split}.jsonl"
            p2 = self.queries_dir / f"{split}.jsonl"
            if p1.exists() and p1.is_file():
                candidate_file = p1
            elif p2.exists() and p2.is_file():
                candidate_file = p2

        if candidate_file is None or not candidate_file.exists():
            raise FileNotFoundError(
                f"Queries file not found for split '{split}' in {self.queries_dir}"
            )

        queries: list[QueryRecord] = []
        with open(candidate_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                text_vi = str(data.get("text_vi", data.get("text", data.get("query", ""))))
                metadata = data.get("metadata")
                queries.append(
                    QueryRecord(
                        query_id=str(data["query_id"]),
                        text_vi=text_vi,
                        metadata=metadata if isinstance(metadata, dict) else None,
                    )
                )
        return queries

    def load_ground_truth(self, split: str) -> list[GroundTruthRecord]:
        if self.qrels_dir is None:
            raise FileNotFoundError("qrels_dir is not specified")

        candidate_file: Path | None = None
        if self.qrels_dir.is_file():
            candidate_file = self.qrels_dir
        else:
            p1 = self.qrels_dir / f"qrels_{split}.jsonl"
            p2 = self.qrels_dir / f"ground_truth_{split}.jsonl"
            p3 = self.qrels_dir / f"{split}.jsonl"
            if p1.exists() and p1.is_file():
                candidate_file = p1
            elif p2.exists() and p2.is_file():
                candidate_file = p2
            elif p3.exists() and p3.is_file():
                candidate_file = p3

        if candidate_file is None or not candidate_file.exists():
            raise FileNotFoundError(
                f"Ground truth file not found for split '{split}' in {self.qrels_dir}"
            )

        ground_truth: list[GroundTruthRecord] = []
        with open(candidate_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                rel_docs = data.get("relevant_doc_ids", data.get("relevant_docs", []))
                rel_chunks = data.get("relevant_chunk_ids", data.get("relevant_chunks", []))
                ground_truth.append(
                    GroundTruthRecord(
                        query_id=str(data["query_id"]),
                        relevant_doc_ids={str(x) for x in rel_docs},
                        relevant_chunk_ids={str(x) for x in rel_chunks},
                    )
                )
        return ground_truth


class InMemoryDatasetAdapter:
    """In-memory dataset adapter for testing and mocking."""

    def __init__(
        self,
        documents: list[DocumentRecord] | None = None,
        chunks: list[ChunkRecord] | None = None,
        queries: dict[str, list[QueryRecord]] | None = None,
        ground_truth: dict[str, list[GroundTruthRecord]] | None = None,
    ) -> None:
        self._documents = list(documents or [])
        self._chunks = list(chunks or [])
        self._queries = dict(queries or {})
        self._ground_truth = dict(ground_truth or {})

    def load_documents(self) -> list[DocumentRecord]:
        return list(self._documents)

    def load_chunks(self) -> list[ChunkRecord]:
        return list(self._chunks)

    def load_queries(self, split: str) -> list[QueryRecord]:
        return list(self._queries.get(split, []))

    def load_ground_truth(self, split: str) -> list[GroundTruthRecord]:
        return list(self._ground_truth.get(split, []))
