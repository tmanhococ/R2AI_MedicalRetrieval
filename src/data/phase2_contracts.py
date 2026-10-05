from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


# Supported languages in Phase 2
Phase2Language = Literal["vi", "en", "zh", "und"]


@dataclass
class ResourceReport:
    """Hardware and runtime diagnostic report for ViBioMIR Phase 2."""
    cpu_threads: int
    ram_total_bytes: int
    ram_free_bytes: int
    disk_free_bytes: int
    gpu_name: str | None
    vram_bytes: int
    runtime_versions: dict[str, str]
    measured_chunks_per_second: float | None = None
    measured_pairs_per_second: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_threads": self.cpu_threads,
            "ram_total_bytes": self.ram_total_bytes,
            "ram_total_gb": round(self.ram_total_bytes / (1024**3), 2),
            "ram_free_bytes": self.ram_free_bytes,
            "ram_free_gb": round(self.ram_free_bytes / (1024**3), 2),
            "disk_free_bytes": self.disk_free_bytes,
            "disk_free_gb": round(self.disk_free_bytes / (1024**3), 2),
            "gpu_name": self.gpu_name,
            "vram_bytes": self.vram_bytes,
            "vram_gb": round(self.vram_bytes / (1024**3), 2) if self.vram_bytes else 0.0,
            "runtime_versions": self.runtime_versions,
            "measured_chunks_per_second": self.measured_chunks_per_second,
            "measured_pairs_per_second": self.measured_pairs_per_second,
        }


@dataclass
class CoverageLedger:
    """Full-corpus coverage invariants across registry, content, and indexes."""
    official_count: int
    unique_url_count: int
    ready_doc_count: int
    terminal_gap_count: int
    unresolved_count: int
    chunk_count: int
    sparse_doc_count: int
    sparse_chunk_count: int
    dense_row_count: int
    source_scope: str
    registry_hash: str
    snapshot_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "official_count": self.official_count,
            "unique_url_count": self.unique_url_count,
            "ready_doc_count": self.ready_doc_count,
            "terminal_gap_count": self.terminal_gap_count,
            "unresolved_count": self.unresolved_count,
            "chunk_count": self.chunk_count,
            "sparse_doc_count": self.sparse_doc_count,
            "sparse_chunk_count": self.sparse_chunk_count,
            "dense_row_count": self.dense_row_count,
            "source_scope": self.source_scope,
            "registry_hash": self.registry_hash,
            "snapshot_id": self.snapshot_id,
        }


class ScopeViolationError(ValueError):
    """Raised when coverage invariants or scope requirements are violated."""


# Recognized scope labels in ViBioMIR Phase 2
SCOPE_PARTIAL_BODY_INDEX = "PARTIAL_BODY_INDEX"
SCOPE_FULL_SOURCE_PASS_WITH_GAPS = "FULL_SOURCE_PASS_WITH_GAPS"
SCOPE_FULL_BODY_INDEX = "FULL_BODY_INDEX"
SCOPE_FULL_HYBRID_WITH_GAPS = "FULL_HYBRID_WITH_GAPS"
SCOPE_FULL_HYBRID = "FULL_HYBRID"

VALID_SCOPES = (
    SCOPE_PARTIAL_BODY_INDEX,
    SCOPE_FULL_SOURCE_PASS_WITH_GAPS,
    SCOPE_FULL_BODY_INDEX,
    SCOPE_FULL_HYBRID_WITH_GAPS,
    SCOPE_FULL_HYBRID,
)


@dataclass
class IndexManifest:
    """Manifest describing an assembled sparse or dense index artifact."""
    snapshot_id: str
    item_type: str
    language: str
    member_count: int
    membership_sha256: str
    backend: str
    model_fingerprint: str | None
    tokenizer_fingerprint: str
    parameters: dict[str, Any] = field(default_factory=dict)
    artifact_paths_and_hashes: dict[str, str] = field(default_factory=dict)


@dataclass
class EmbeddingBlock:
    """Atomic block metadata for float16 embedding shards on disk."""
    block_id: str
    first_row: int
    row_count: int
    dimension: int
    vector_path: str
    mapping_path: str
    model_fingerprint: str
    snapshot_id: str
    sha256: str


@dataclass
class EvidenceSpan:
    """Source evidence span preserving document offsets without synthetic modifications."""
    span_id: str
    doc_id: int
    source_version: str
    source_sha256: str
    start_char: int
    end_char: int
    seed_chunk_ids: tuple[str, ...]
    variant_kind: Literal["base", "neighbor_block", "expanded"]
    base_span_id: str | None = None
    added_block_span_ids: tuple[str, ...] = field(default_factory=tuple)
    score: float | None = None


@dataclass
class StageManifest:
    """Manifest describing a completed pipeline stage artifact."""
    stage: str
    fingerprint: str
    snapshot_id: str
    completed_units: int
    expected_units: int
    artifact_paths_and_hashes: dict[str, str] = field(default_factory=dict)


@dataclass
class StageMeasurement:
    """Resource and timing metrics recorded during stage execution."""
    stage: str
    wall_seconds: float
    completed_units: int
    peak_core_rss_bytes: int
    peak_worker_rss_bytes: int
    peak_gpu_allocated_bytes: int
    peak_gpu_reserved_bytes: int
    input_bytes: int
    output_bytes: int


@dataclass
class FullCampaignConfig:
    scope: str = "all"
    domain_whitelist: list[str] = field(default_factory=list)
    source_limit: int | None = None
    reuse_existing_content: bool = True
    require_no_unresolved_sources: bool = True
    priority_changes_order_only: bool = True


@dataclass
class SparseConfig:
    backend: str = "lucene"
    k1: float = 1.2
    b: float = 0.75
    pretokenized: bool = True
    analyzer: str = "whitespace"
    token_encoding: str = "sha256_hex_v1"
    document_top_k_per_language: int = 500
    chunk_top_k_per_language: int = 1000


@dataclass
class DenseConfig:
    enabled: bool = True
    model_name: str = "intfloat/multilingual-e5-base"
    dimension: int = 768
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    normalize: bool = True
    vector_storage_dtype: str = "float16"
    block_rows: int = 10000
    backend: str = "faiss_cpu_ivfpq"
    metric: str = "inner_product"
    nlist: int = 4096
    pq_m: int = 64
    pq_bits: int = 8
    train_rows: int = 200000
    nprobe: int = 32
    approximate_top_k: int = 4000
    refined_top_k: int = 1000
    query_window_tokens: int = 448
    query_window_overlap: int = 64
    search_all_languages_with_original_vi: bool = True


@dataclass
class EvidenceConfig:
    document_top_k: int = 100
    base_windows_per_doc: int = 6
    base_spans_per_query_max: int = 800
    scan_all_chunks_in_candidate_doc: bool = True
    neighbor_blocks: int = 1
    expanded_soft_target_tokens: int = 600
    expansion_logit_tolerance: float = 0.5
    require_positive_neighbor: bool = True
    dedup_span_overlap: float = 0.8
    preserve_source_offsets: bool = True


@dataclass
class RerankConfig:
    enabled: bool = True
    model_name: str = "BAAI/bge-reranker-v2-m3"
    max_length: int = 512
    passage_window_tokens: int = 320
    query_window_tokens: int = 128
    query_window_overlap: int = 16
    window_aggregation: str = "max"
    use_raw_logits: bool = True
    require_real_weights: bool = True


@dataclass
class OutputPolicy:
    fit_only_allowed_dev: bool = True
    fallback_policy: str = "heuristic_ce_positive"
    fallback_raw_logit_min: float = 0.0
    min_results: int = 0
    document_cap: int = 10
    evidence_cap: int = 20
    evidence_cap_per_doc: int = 3
    fixed_count_validation: bool = False
