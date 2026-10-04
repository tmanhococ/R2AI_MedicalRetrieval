from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, NotRequired, Protocol, TypedDict, runtime_checkable


@dataclass(frozen=True)
class OfficialSource:
    """Official source document representation from links_corpus.parquet."""
    doc_id: int
    url: str

    def __post_init__(self) -> None:
        if isinstance(self.doc_id, bool) or not isinstance(self.doc_id, int):
            raise ValueError(f"doc_id must be integer, got {type(self.doc_id).__name__}: {self.doc_id}")
        if not (-9223372036854775808 <= self.doc_id <= 9223372036854775807):
            raise ValueError(f"doc_id out of signed int64 range: {self.doc_id}")
        if not isinstance(self.url, str) or not self.url.strip():
            raise ValueError(f"url must be non-empty string, got {self.url!r}")


@dataclass(frozen=True)
class OfficialQuery:
    """Official query representation from query.parquet."""
    query_id: int
    query: str

    def __post_init__(self) -> None:
        if isinstance(self.query_id, bool) or not isinstance(self.query_id, int):
            raise ValueError(f"query_id must be integer, got {type(self.query_id).__name__}: {self.query_id}")
        if not (-9223372036854775808 <= self.query_id <= 9223372036854775807):
            raise ValueError(f"query_id out of signed int64 range: {self.query_id}")
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError(f"query must be non-empty string, got {self.query!r}")


@dataclass(frozen=True)
class FetchJob:
    """Fetch job scheduled for crawler worker."""
    job_id: str
    url: str
    attempt: int


@dataclass(frozen=True)
class FetchResult:
    """Outcome of fetching an official source URL."""
    job_id: str
    state: str
    raw_path: str | None
    raw_sha256: str | None
    final_url: str | None
    mime_type: str | None
    http_status: int | None
    fetched_at: str
    error_code: str | None
    retry_at: str | None = None
    bytes_written: int = 0


@dataclass(frozen=True)
class ExtractedBlock:
    """Contiguous block of extracted text with reading order and character span."""
    block_index: int
    kind: str
    start_char: int
    end_char: int
    section: str | None


@dataclass(frozen=True)
class ExtractedDocument:
    """Structured extraction of an official source document."""
    doc_id: int
    source_version: str
    source_text: str
    source_sha256: str
    raw_sha256: str
    title: str | None
    language: str
    blocks: tuple[ExtractedBlock, ...]
    quality_flags: tuple[str, ...]


@dataclass(frozen=True)
class SpanChunk:
    """Passage chunk with strict character span provenance within source_text."""
    chunk_id: str
    doc_id: int
    source_version: str
    source_sha256: str
    start_char: int
    end_char: int
    chunk_index: int
    language: str
    section: str | None
    text: str
    index_text: str
    dense_token_count: int
    reranker_token_count: int


@dataclass(frozen=True)
class ReferenceChunk:
    """Reference chunk in development qrels."""
    doc_id: int
    chunk_text: str

    def __post_init__(self) -> None:
        if isinstance(self.doc_id, bool) or not isinstance(self.doc_id, int):
            raise ValueError(f"doc_id must be integer, got {type(self.doc_id).__name__}: {self.doc_id}")
        if not (-9223372036854775808 <= self.doc_id <= 9223372036854775807):
            raise ValueError(f"doc_id out of signed int64 range: {self.doc_id}")
        if not isinstance(self.chunk_text, str) or not self.chunk_text.strip():
            raise ValueError(f"chunk_text must be non-empty string, got {self.chunk_text!r}")


@dataclass(frozen=True)
class DevQueryMetadata:
    """Metadata describing development split query origin and coverage."""
    query_id: int
    query_text: str
    origin: str
    source_doc_ids: tuple[int, ...]
    topic: str
    question_style: str
    split: str
    pool_version: str
    judgment_coverage: float


@dataclass(frozen=True)
class EvaluationRecord:
    """Ground truth relevance judgments for an evaluation query."""
    query_id: int
    relevant_doc_ids: frozenset[int]
    relevant_chunks: tuple[ReferenceChunk, ...]
    metadata: DevQueryMetadata | None = None

    def __post_init__(self) -> None:
        if isinstance(self.query_id, bool) or not isinstance(self.query_id, int):
            raise ValueError(f"query_id must be integer, got {type(self.query_id).__name__}: {self.query_id}")
        if not (-9223372036854775808 <= self.query_id <= 9223372036854775807):
            raise ValueError(f"query_id out of signed int64 range: {self.query_id}")
        for ref in self.relevant_chunks:
            if not isinstance(ref, ReferenceChunk):
                raise TypeError(f"relevant_chunks must contain ReferenceChunk objects, got {type(ref).__name__}")
            if not ref.chunk_text or not ref.chunk_text.strip():
                raise ValueError("ReferenceChunk chunk_text cannot be empty")


@dataclass(frozen=True)
class LevelMetrics:
    """Precision, recall, and Macro F2 metrics at document or chunk level."""
    precision: float
    recall: float
    f2: float
    prediction_count: int
    reference_count: int
    correct_prediction_count: int
    found_reference_count: int


@dataclass(frozen=True)
class MacroMetrics:
    """Macro-averaged precision, recall, and F2 scores across a query dataset."""
    precision: float
    recall: float
    f2: float


@dataclass(frozen=True)
class QueryMetrics:
    """Evaluation metrics for a single query across document and chunk levels."""
    query_id: int
    documents: LevelMetrics
    chunks: LevelMetrics
    matched_reference_indices: tuple[int, ...] = ()
    matched_prediction_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class IngestReport:
    """Report produced after importing official corpus rows into registry."""
    input_rows: int
    inserted_sources: int
    unique_urls: int
    rejected_rows: int
    input_sha256: str


@dataclass(frozen=True)
class CrawlReport:
    """Summary of crawler execution."""
    leased: int = 0
    fetched: int = 0
    retry_wait: int = 0
    failed: int = 0
    blocked: int = 0
    unsupported: int = 0
    bytes_written: int = 0

    @property
    def total_jobs(self) -> int:
        return self.leased

    @property
    def succeeded(self) -> int:
        return self.fetched

    @property
    def retried(self) -> int:
        return self.retry_wait

    @property
    def skipped(self) -> int:
        return self.blocked + self.unsupported


@dataclass(frozen=True)
class ExtractionReport:
    """Summary of document extraction execution."""
    attempted: int
    ready: int
    quarantined: int
    failed: int
    flags_by_reason: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class PackedPair:
    """Token-budgeted pair inputs prepared for cross-encoder reranking."""
    query_text: str
    passage_text: str
    input_ids: tuple[int, ...] | list[int]
    attention_mask: tuple[int, ...] | list[int]
    query_window_index: int


@dataclass(frozen=True)
class MigrationResult:
    """Outcome of migrating legacy chunks without full provenance."""
    migrated_chunks: tuple[SpanChunk, ...] = ()
    quarantined_chunks: tuple[Any, ...] = ()
    regenerated_chunks: tuple[SpanChunk, ...] = ()



@dataclass(frozen=True)
class SnapshotManifest:
    """Immutable manifest pinning all artifacts and hashes of a frozen snapshot."""
    snapshot_id: str
    created_at: str
    registry_sha256: str
    queries_sha256: str
    source_membership_sha256: str
    extractor_fingerprint: str
    chunker_fingerprint: str
    source_count: int
    ready_document_count: int
    chunk_count: int
    language_counts: dict[str, int]
    quarantine_counts: dict[str, int]
    shard_paths_and_hashes: list[dict[str, str]]


@dataclass(frozen=True)
class ValidationReport:
    """Offline validation verdict for submission file."""
    valid: bool
    query_count: int
    document_count: int = 0
    chunk_count: int = 0
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    snapshot_id: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.errors, list):
            object.__setattr__(self, "errors", tuple(self.errors))
        if isinstance(self.warnings, list):
            object.__setattr__(self, "warnings", tuple(self.warnings))

    @property
    def is_valid(self) -> bool:
        return self.valid


@dataclass(frozen=True)
class ExportResult:
    """Summary of exported official submission files and checksums."""
    json_path: Path
    zip_path: Path
    validation_report_path: Path
    json_sha256: str
    zip_sha256: str
    validation_report: ValidationReport | None = None



class PerQueryMetrics(tuple):
    """Tuple of QueryMetrics supporting sequence indexing, iteration, and query_id dict lookup."""

    def __new__(cls, items: Any = ()) -> PerQueryMetrics:
        return super().__new__(cls, items)

    def __init__(self, items: Any = ()) -> None:
        self._by_id: dict[int, QueryMetrics] = {qm.query_id: qm for qm in self}

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, slice):
            return PerQueryMetrics(super().__getitem__(key))
        if isinstance(key, int):
            if key in self._by_id:
                return self._by_id[key]
            return super().__getitem__(key)
        raise KeyError(f"Query ID {key!r} not found in per_query_metrics")

    def __contains__(self, item: Any) -> bool:
        if isinstance(item, int):
            return item in self._by_id
        return super().__contains__(item)

    def get(self, query_id: int, default: Any = None) -> QueryMetrics | None:
        return self._by_id.get(query_id, default)

    def keys(self) -> Any:
        return self._by_id.keys()

    def values(self) -> Any:
        return self._by_id.values()

    def items(self) -> Any:
        return self._by_id.items()


@dataclass(frozen=True)
class EvaluationSummary:
    """Overall dataset-level macro evaluation report."""
    profile_fingerprint: str
    scorer_parity_verified: bool
    query_count: int
    macro_documents: MacroMetrics
    macro_chunks: MacroMetrics
    per_query_metrics: tuple[QueryMetrics, ...]
    inferred_final_mean: float
    warnings: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def profile_name(self) -> str:
        for part in self.profile_fingerprint.split(":"):
            if part.startswith("name="):
                return part.split("=", 1)[1]
        return self.profile_fingerprint

    @property
    def macro_f2_documents(self) -> float:
        return self.macro_documents.f2

    @property
    def macro_f2_chunks(self) -> float:
        return self.macro_chunks.f2

    @property
    def total_queries(self) -> int:
        return self.query_count


@dataclass(frozen=True)
class RetrievalTrace:
    """Complete candidate score trajectory and filtering trace for a query."""
    query_id: int
    candidate_doc_ids: frozenset[int]
    reranked_doc_ids: frozenset[int]
    final_doc_ids: frozenset[int]
    final_chunk_ids: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


class SubmissionChunk(TypedDict):
    """Chunk element in official results.json submission schema."""
    doc_id: int
    chunk_text: str
    chunk_order: NotRequired[int]


class SubmissionRow(TypedDict):
    """Query row in official results.json submission schema."""
    id: int
    relevant_docs: list[int]
    relevant_chunks: list[SubmissionChunk]


# Validated configuration dataclasses

@dataclass(frozen=True)
class CrawlConfig:
    concurrency: int = 16
    per_domain_concurrency: int = 2
    min_domain_interval_seconds: float = 1.0
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 30.0
    total_deadline_seconds: float = 60.0
    max_attempts: int = 3
    base_backoff_seconds: float = 2.0
    lease_seconds: float = 120.0
    max_redirects: int = 5
    max_html_bytes: int = 10 * 1024 * 1024
    max_pdf_bytes: int = 30 * 1024 * 1024
    respect_robots: bool = True
    raw_root: Path = Path("artifacts/vibiomir/raw")


@dataclass(frozen=True)
class ExtractionConfig:
    html_backend: str = "trafilatura"
    pdf_backend: str = "pymupdf"
    language_model_path: Path = Path("artifacts/models/lid.176.ftz")
    language_confidence_min: float = 0.70
    accepted_languages: tuple[str, ...] = ("vi", "en", "zh")
    quarantine_unknown_language: bool = True


@dataclass(frozen=True)
class ChunkConfig:
    mode: str = "auto"
    strategy_type: str = "paragraph_token_spans"
    body_max_tokens: int = 320
    long_block_overlap_tokens: int = 64
    contextual_header_max_tokens: int = 32
    drop_short_blocks: bool = False


@dataclass(frozen=True)
class PairPackingConfig:
    max_length: int = 512
    query_window_tokens: int = 128
    query_window_overlap_tokens: int = 16
    query_window_aggregation: str = "max"
    body_truncation: str = "reject"


@dataclass(frozen=True)
class SnapshotConfig:
    snapshot_root: Path
    name: str
    parquet_row_group_size: int = 5000
    target_shard_bytes: int = 67108864
    chunk_config: ChunkConfig = field(default_factory=ChunkConfig)


@dataclass(frozen=True)
class EvaluationProfile:
    name: str
    normalization_version: str
    tokenizer_name: str
    threshold: Fraction
    undefined_pr: int
    empty_both_f2: int
    scorer_parity_verified: bool

    @property
    def profile_fingerprint(self) -> str:
        """Deterministic fingerprint representing full evaluator configuration."""
        return (
            f"name={self.name}:"
            f"norm={self.normalization_version}:"
            f"tok={self.tokenizer_name}:"
            f"thresh={self.threshold.numerator}/{self.threshold.denominator}:"
            f"undef={self.undefined_pr}:"
            f"empty_both={self.empty_both_f2}:"
            f"parity={self.scorer_parity_verified}"
        )


@runtime_checkable
class SourceStore(Protocol):
    """Reader protocol providing random lookup over frozen snapshot documents and chunks."""

    def get_document(self, doc_id: int) -> ExtractedDocument: ...

    def get_chunk(self, chunk_id: str) -> SpanChunk: ...

    def contains_official_doc(self, doc_id: int) -> bool: ...
