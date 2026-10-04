from pathlib import Path
import sqlite3
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.vibiomir_contracts import (
    OfficialSource,
    OfficialQuery,
    IngestReport,
    SubmissionChunk,
    SubmissionRow,
)
from src.data.vibiomir_adapter import (
    iter_official_sources,
    load_official_queries,
)
from src.data.vibiomir_registry import (
    SourceRegistry,
    ingest_sources,
)


def _create_synthetic_corpus_parquet(
    path: Path,
    ids: list[int | float | bool | None],
    urls: list[str | None],
    id_type: pa.DataType = pa.int64(),
) -> Path:
    arrays = [
        pa.array(ids, type=id_type),
        pa.array(urls, type=pa.string()),
    ]
    schema = pa.schema([
        ("id", id_type),
        ("url", pa.string()),
    ])
    table = pa.Table.from_arrays(arrays, schema=schema)
    pq.write_table(table, path)
    return path


def _create_synthetic_query_parquet(
    path: Path,
    ids: list[int | float | bool | None],
    queries: list[str | None],
    id_type: pa.DataType = pa.int64(),
) -> Path:
    arrays = [
        pa.array(ids, type=id_type),
        pa.array(queries, type=pa.string()),
    ]
    schema = pa.schema([
        ("id", id_type),
        ("query", pa.string()),
    ])
    table = pa.Table.from_arrays(arrays, schema=schema)
    pq.write_table(table, path)
    return path


def test_sparse_int64_ids_and_alias_urls(tmp_path: Path):
    """IDs 583 and 4294967297 are preserved; alias URLs sharing same URL produce two rows in sources and one fetch job."""
    corpus_file = tmp_path / "corpus.parquet"
    db_file = tmp_path / "registry.sqlite"

    # ID 4294967297 is 2^32 + 1, exceeding signed 32-bit int
    doc_ids = [583, 4294967297]
    shared_url = "https://example.com/medical/guideline-covid"
    urls = [shared_url, shared_url]

    _create_synthetic_corpus_parquet(corpus_file, doc_ids, urls)

    # 1. Test iter_official_sources
    source_iterator = iter_official_sources(corpus_file, batch_size=10)
    sources = list(source_iterator)
    assert len(sources) == 2
    assert sources[0] == OfficialSource(doc_id=583, url=shared_url)
    assert sources[1] == OfficialSource(doc_id=4294967297, url=shared_url)

    # 2. Test ingestion into SourceRegistry
    registry = SourceRegistry(db_file)
    report = ingest_sources(iter_official_sources(corpus_file), registry)

    assert isinstance(report, IngestReport)
    assert report.input_rows == 2
    assert report.inserted_sources == 2
    assert report.unique_urls == 1
    assert report.rejected_rows == 0
    assert report.input_sha256 != ""

    # 3. Assert direct DB table state
    with sqlite3.connect(str(db_file)) as conn:
        sources_rows = conn.execute(
            "SELECT doc_id, original_url, fetch_job_id FROM sources ORDER BY doc_id ASC"
        ).fetchall()
        assert len(sources_rows) == 2
        assert sources_rows[0][0] == 583
        assert sources_rows[0][1] == shared_url
        assert sources_rows[1][0] == 4294967297
        assert sources_rows[1][1] == shared_url
        # Both must share the same fetch_job_id
        assert sources_rows[0][2] == sources_rows[1][2]

        jobs = conn.execute(
            "SELECT job_id, url, state, attempts FROM fetch_jobs"
        ).fetchall()
        assert len(jobs) == 1
        assert jobs[0][0] == sources_rows[0][2]
        assert jobs[0][1] == shared_url
        assert jobs[0][2] == "pending"
        assert jobs[0][3] == 0

    assert registry.is_complete() is True
    registry.close()


def test_reject_bool_float_null_and_duplicate_ids(tmp_path: Path):
    """Rejects bool, float, null, and duplicate IDs transactionally without masking via upsert."""
    db_file = tmp_path / "registry.sqlite"

    # A. Float ID in parquet schema should be rejected by iter_official_sources
    float_file = tmp_path / "float_corpus.parquet"
    arrays = [pa.array([1.5, 2.0], type=pa.float64()), pa.array(["http://a.com", "http://b.com"])]
    pq.write_table(pa.Table.from_arrays(arrays, names=["id", "url"]), float_file)
    with pytest.raises(ValueError, match="integer"):
        list(iter_official_sources(float_file))

    # B. Bool ID in parquet schema should be rejected
    bool_file = tmp_path / "bool_corpus.parquet"
    arrays = [pa.array([True, False], type=pa.bool_()), pa.array(["http://a.com", "http://b.com"])]
    pq.write_table(pa.Table.from_arrays(arrays, names=["id", "url"]), bool_file)
    with pytest.raises(ValueError, match="integer"):
        list(iter_official_sources(bool_file))

    # C. Null ID in parquet should be rejected
    null_id_file = tmp_path / "null_id_corpus.parquet"
    _create_synthetic_corpus_parquet(null_id_file, [1, None], ["http://a.com", "http://b.com"])
    with pytest.raises(ValueError, match="null"):
        list(iter_official_sources(null_id_file))

    # D. Null/empty URL should be rejected
    null_url_file = tmp_path / "null_url_corpus.parquet"
    arrays = [pa.array([1, 2], type=pa.int64()), pa.array(["http://a.com", "   "])]
    pq.write_table(pa.Table.from_arrays(arrays, names=["id", "url"]), null_url_file)
    with pytest.raises(ValueError, match="[Uu]rl"):
        list(iter_official_sources(null_url_file))

    # E. Duplicate IDs in input must fail transactionally and not be masked by upsert
    dup_file = tmp_path / "dup_corpus.parquet"
    _create_synthetic_corpus_parquet(
        dup_file,
        [583, 100, 583],  # Duplicate doc_id 583
        ["http://a.com", "http://b.com", "http://c.com"],
    )

    registry = SourceRegistry(db_file)
    with pytest.raises(ValueError, match="[Dd]uplicate.*doc_id"):
        ingest_sources(iter_official_sources(dup_file), registry)

    # Ingest must fail transactionally: registry is not complete
    assert registry.is_complete() is False

    # F. Cross-batch duplicate rejection: batch 1 committed, batch 2 has a duplicate doc_id
    cross_file = tmp_path / "cross_batch_dup.parquet"
    _create_synthetic_corpus_parquet(
        cross_file,
        [1, 2, 3, 4, 2, 6],  # doc_id 2 repeated across batch boundary (batch_size=3)
        [f"http://example.com/{i}" for i in [1, 2, 3, 4, 2, 6]],
    )
    cross_db = tmp_path / "cross_registry.sqlite"
    cross_reg = SourceRegistry(cross_db)
    with pytest.raises(ValueError, match="[Dd]uplicate.*doc_id"):
        ingest_sources(iter_official_sources(cross_file, batch_size=3), cross_reg, batch_size=3)

    assert cross_reg.is_complete() is False
    # Batch 1 (1, 2, 3) was committed, batch 2 was rolled back, so only 3 rows exist
    assert cross_reg.count_sources() == 3
    cross_reg.close()
    registry.close()


def test_query_exact_set_and_count(tmp_path: Path):
    """Tests exact set, signed int64, and expected_count validation for queries."""
    query_file = tmp_path / "query.parquet"
    ids = [101, 102, 103]
    queries = ["Triệu chứng sốt xuất huyết", "Điều trị cúm mùa", "Chỉ số đường huyết bình thường"]
    _create_synthetic_query_parquet(query_file, ids, queries)

    # Matching expected_count
    loaded = load_official_queries(query_file, expected_count=3)
    assert len(loaded) == 3
    assert loaded[0] == OfficialQuery(query_id=101, query=queries[0])
    assert loaded[1] == OfficialQuery(query_id=102, query=queries[1])
    assert loaded[2] == OfficialQuery(query_id=103, query=queries[2])

    # No expected_count specified
    loaded_none = load_official_queries(query_file, expected_count=None)
    assert len(loaded_none) == 3

    # Mismatched expected_count raises ValueError
    with pytest.raises(ValueError, match="Expected 1200 queries, but found 3"):
        load_official_queries(query_file, expected_count=1200)

    # Query with duplicate ID raises ValueError
    dup_query_file = tmp_path / "dup_query.parquet"
    _create_synthetic_query_parquet(dup_query_file, [101, 101], ["query 1", "query 2"])
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        load_official_queries(dup_query_file)

    # Query with empty text raises ValueError
    empty_query_file = tmp_path / "empty_query.parquet"
    _create_synthetic_query_parquet(empty_query_file, [101], ["   "])
    with pytest.raises(ValueError, match="empty"):
        load_official_queries(empty_query_file)


def test_ingest_resume_idempotent(tmp_path: Path):
    """Resume from checkpoint and idempotent re-run on completed input."""
    corpus_file = tmp_path / "corpus_10.parquet"
    db_file = tmp_path / "registry.sqlite"

    ids = list(range(1, 11))
    urls = [f"https://example.com/doc/{i}" for i in ids]
    _create_synthetic_corpus_parquet(corpus_file, ids, urls)

    registry = SourceRegistry(db_file)
    source_iter = iter_official_sources(corpus_file, batch_size=4)
    file_sha = source_iter.input_sha256

    # 1. Simulate an interrupted ingestion: commit first batch of 4 rows, leaving complete=0
    first_4 = list(iter_official_sources(corpus_file))[:4]
    first_sources = [(s.doc_id, s.url, "job_" + str(s.doc_id)) for s in first_4]
    first_jobs = [("job_" + str(s.doc_id), s.url) for s in first_4]
    registry.init_ingest_state(file_sha)
    registry.commit_ingest_batch(file_sha, first_sources, first_jobs, last_committed_row=4)

    assert registry.count_sources() == 4
    assert registry.is_complete() is False
    ingest_state = registry.get_ingest_state(file_sha)
    assert ingest_state is not None
    assert ingest_state[1] == 4  # last_committed_row == 4
    assert ingest_state[2] is False  # complete == 0

    # 2. Resume ingestion on the same file: skips first 4 rows and completes remaining 6 rows
    report1 = ingest_sources(
        iter_official_sources(corpus_file, batch_size=4),
        registry,
        batch_size=4,
    )
    assert report1.input_rows == 10
    assert report1.inserted_sources == 10
    assert registry.count_sources() == 10
    assert registry.is_complete() is True

    # 3. Re-running on completed registry is idempotent: returns IngestReport without errors
    report2 = ingest_sources(
        iter_official_sources(corpus_file, batch_size=4),
        registry,
        batch_size=4,
    )
    assert report2.input_rows == 10
    assert report2.inserted_sources == 10
    assert report2.input_sha256 == report1.input_sha256
    assert registry.count_sources() == 10

    # 4. Attempting to ingest a different file with a different checksum into same registry raises ValueError
    diff_file = tmp_path / "different_corpus.parquet"
    _create_synthetic_corpus_parquet(diff_file, [100, 101], ["http://x.com", "http://y.com"])
    with pytest.raises(ValueError, match="[Cc]hecksum"):
        ingest_sources(
            iter_official_sources(diff_file),
            registry,
        )

    registry.close()


def test_contracts_and_unit_validations(tmp_path: Path):
    """Directly tests contract invariants, bounds checking, and registry query helpers."""
    # OfficialSource contract validations
    with pytest.raises(ValueError, match="integer"):
        OfficialSource(doc_id=True, url="https://example.com")  # type: ignore

    with pytest.raises(ValueError, match="integer"):
        OfficialSource(doc_id=1.5, url="https://example.com")  # type: ignore

    with pytest.raises(ValueError, match="range"):
        OfficialSource(doc_id=2**63, url="https://example.com")

    with pytest.raises(ValueError, match="range"):
        OfficialSource(doc_id=-(2**63) - 1, url="https://example.com")

    with pytest.raises(ValueError, match="url"):
        OfficialSource(doc_id=1, url="   ")

    # OfficialQuery contract validations
    with pytest.raises(ValueError, match="integer"):
        OfficialQuery(query_id=False, query="test")  # type: ignore

    with pytest.raises(ValueError, match="range"):
        OfficialQuery(query_id=2**63, query="test")

    with pytest.raises(ValueError, match="query"):
        OfficialQuery(query_id=1, query="")

    # SourceRegistry query helpers
    db_file = tmp_path / "test_helpers.sqlite"
    with SourceRegistry(db_file) as reg:
        assert reg.count_sources() == 0
        assert reg.count_unique_urls() == 0
        assert reg.contains_doc(1) is False
        assert reg.get_source(1) is None
        assert reg.is_complete() is False

