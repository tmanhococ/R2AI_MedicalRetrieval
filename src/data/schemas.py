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
