from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sqlite3
from typing import Iterable

from src.data.vibiomir_contracts import (
    IngestReport,
    OfficialSource,
)


class SourceRegistry:
    """SQLite-backed registry managing sources, fetch_jobs, and transactional ingest state."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._init_db()

    def _init_db(self) -> None:
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA synchronous = NORMAL;")
        self.conn.execute("PRAGMA foreign_keys = ON;")

        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sources (
                    doc_id INTEGER PRIMARY KEY,
                    original_url TEXT NOT NULL,
                    fetch_job_id TEXT NOT NULL
                );
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sources_fetch_job_id
                ON sources(fetch_job_id);
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fetch_jobs (
                    job_id TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    lease_until TEXT,
                    raw_path TEXT,
                    raw_sha256 TEXT,
                    http_status INTEGER,
                    error_code TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingest_state (
                    input_sha256 TEXT PRIMARY KEY,
                    last_committed_row INTEGER NOT NULL,
                    complete INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def get_ingest_state(
        self, input_sha256: str | None = None
    ) -> tuple[str, int, bool] | None:
        """Returns (input_sha256, last_committed_row, is_complete) or None."""
        if input_sha256 is not None:
            row = self.conn.execute(
                """
                SELECT input_sha256, last_committed_row, complete
                FROM ingest_state
                WHERE input_sha256 = ?
                """,
                (input_sha256,),
            ).fetchone()
        else:
            row = self.conn.execute(
                """
                SELECT input_sha256, last_committed_row, complete
                FROM ingest_state
                ORDER BY updated_at DESC LIMIT 1
                """
            ).fetchone()

        if row:
            return (row[0], row[1], bool(row[2]))
        return None

    def init_ingest_state(self, input_sha256: str) -> None:
        now_utc = datetime.now(timezone.utc).isoformat()
        with self.conn:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO ingest_state (
                    input_sha256, last_committed_row, complete, updated_at
                ) VALUES (?, 0, 0, ?)
                """,
                (input_sha256, now_utc),
            )

    def commit_ingest_batch(
        self,
        input_sha256: str,
        sources: list[tuple[int, str, str]],
        jobs: list[tuple[str, str]],
        last_committed_row: int,
    ) -> None:
        """Atomically inserts batch of sources and fetch_jobs, updating cursor.

        Rolls back transactionally if any duplicate doc_id is encountered.
        """
        now_utc = datetime.now(timezone.utc).isoformat()

        # Check for duplicates within this batch
        seen_batch_ids: set[int] = set()
        for doc_id, _, _ in sources:
            if doc_id in seen_batch_ids:
                raise ValueError(f"Duplicate doc_id {doc_id} detected in input batch")
            seen_batch_ids.add(doc_id)

        try:
            with self.conn:
                self.conn.executemany(
                    "INSERT INTO sources (doc_id, original_url, fetch_job_id) VALUES (?, ?, ?)",
                    sources,
                )
                job_rows = [
                    (job_id, url, "pending", 0, now_utc, now_utc)
                    for job_id, url in jobs
                ]
                self.conn.executemany(
                    """
                    INSERT OR IGNORE INTO fetch_jobs (
                        job_id, url, state, attempts, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    job_rows,
                )
                self.conn.execute(
                    """
                    UPDATE ingest_state
                    SET last_committed_row = ?, updated_at = ?
                    WHERE input_sha256 = ?
                    """,
                    (last_committed_row, now_utc, input_sha256),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Duplicate doc_id or integrity error during batch ingest: {exc}") from exc

    def mark_ingest_complete(self, input_sha256: str, total_rows: int) -> None:
        now_utc = datetime.now(timezone.utc).isoformat()
        with self.conn:
            self.conn.execute(
                """
                UPDATE ingest_state
                SET complete = 1, last_committed_row = ?, updated_at = ?
                WHERE input_sha256 = ?
                """,
                (total_rows, now_utc, input_sha256),
            )

    def is_complete(self, input_sha256: str | None = None) -> bool:
        state = self.get_ingest_state(input_sha256)
        return state is not None and state[2]

    def contains_doc(self, doc_id: int) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM sources WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return row is not None

    def get_source(self, doc_id: int) -> OfficialSource | None:
        row = self.conn.execute(
            "SELECT doc_id, original_url FROM sources WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if row:
            return OfficialSource(doc_id=row[0], url=row[1])
        return None

    def count_sources(self) -> int:
        return self.conn.execute("SELECT count(*) FROM sources").fetchone()[0]

    def count_unique_urls(self) -> int:
        return self.conn.execute("SELECT count(DISTINCT original_url) FROM sources").fetchone()[0]

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "SourceRegistry":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()


def ingest_sources(
    rows: Iterable[OfficialSource],
    registry: SourceRegistry,
    input_sha256: str | None = None,
    batch_size: int = 500,
) -> IngestReport:
    """Imports official source rows into SQLite SourceRegistry with transactional batch commits and resume cursor."""
    if input_sha256 is None:
        if hasattr(rows, "input_sha256"):
            input_sha256 = getattr(rows, "input_sha256")
        elif hasattr(rows, "path"):
            path = Path(getattr(rows, "path"))
            h = hashlib.sha256()
            with path.open("rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            input_sha256 = h.hexdigest()
        elif isinstance(rows, (list, tuple)):
            h = hashlib.sha256()
            for r in rows:
                h.update(f"{r.doc_id}:{r.url}\n".encode("utf-8"))
            input_sha256 = h.hexdigest()
        else:
            raise ValueError(
                "input_sha256 must be provided when rows does not expose input_sha256 or path"
            )

    existing_state = registry.get_ingest_state()
    if existing_state is not None:
        existing_sha, last_row, is_complete = existing_state
        if existing_sha != input_sha256:
            raise ValueError(
                f"Registry already contains ingest state for different checksum {existing_sha}, "
                f"got {input_sha256}. Use a separate registry or explicit migration."
            )
        if is_complete:
            return IngestReport(
                input_rows=last_row,
                inserted_sources=registry.count_sources(),
                unique_urls=registry.count_unique_urls(),
                rejected_rows=0,
                input_sha256=input_sha256,
            )
        cursor = last_row
    else:
        cursor = 0
        registry.init_ingest_state(input_sha256)

    batch_sources: list[tuple[int, str, str]] = []
    batch_jobs: list[tuple[str, str]] = []
    current_row_idx = 0

    for row in rows:
        if not isinstance(row, OfficialSource):
            if not hasattr(row, "doc_id") or not hasattr(row, "url"):
                raise ValueError(f"Expected OfficialSource, got {type(row).__name__}: {row}")
            row = OfficialSource(doc_id=row.doc_id, url=row.url)

        if current_row_idx < cursor:
            current_row_idx += 1
            continue

        clean_url = row.url.strip()
        job_id = hashlib.sha256(clean_url.encode("utf-8")).hexdigest()

        batch_sources.append((row.doc_id, clean_url, job_id))
        batch_jobs.append((job_id, clean_url))
        current_row_idx += 1

        if len(batch_sources) >= batch_size:
            registry.commit_ingest_batch(
                input_sha256, batch_sources, batch_jobs, current_row_idx
            )
            batch_sources.clear()
            batch_jobs.clear()

    if batch_sources:
        registry.commit_ingest_batch(
            input_sha256, batch_sources, batch_jobs, current_row_idx
        )
        batch_sources.clear()
        batch_jobs.clear()

    registry.mark_ingest_complete(input_sha256, current_row_idx)

    return IngestReport(
        input_rows=current_row_idx,
        inserted_sources=registry.count_sources(),
        unique_urls=registry.count_unique_urls(),
        rejected_rows=0,
        input_sha256=input_sha256,
    )
