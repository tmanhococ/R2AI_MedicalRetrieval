import collections
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

from src.data.vibiomir_contracts import (
    FetchJob,
    FetchResult,
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
                    retry_at TEXT,
                    raw_path TEXT,
                    raw_sha256 TEXT,
                    http_status INTEGER,
                    mime_type TEXT,
                    final_url TEXT,
                    error_code TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            for col, col_type in [
                ("retry_at", "TEXT"),
                ("mime_type", "TEXT"),
                ("final_url", "TEXT"),
                ("campaign_id", "TEXT"),
                ("priority", "INTEGER DEFAULT 0"),
            ]:
                try:
                    self.conn.execute(f"ALTER TABLE fetch_jobs ADD COLUMN {col} {col_type};")
                except sqlite3.OperationalError:
                    pass
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS campaigns (
                    campaign_id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL
                );
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_fetch_jobs_state_prio_att
                ON fetch_jobs(state, priority, attempts);
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

    def sample_stratified_jobs(
        self,
        sample_size: int = 500,
        seed: int = 42,
        persist_path: Path | str | None = None,
    ) -> list[str]:
        """Draw a deterministic stratified sample of jobs by (domain, url_type).

        Guarantees proportional quota allocation across strata using the largest remainder method,
        representing every non-empty stratum when sample_size >= num_strata, and persists the sample IDs.
        """
        rows = self.conn.execute("SELECT job_id, url FROM fetch_jobs").fetchall()
        if not rows:
            return []

        strata: dict[str, list[str]] = collections.defaultdict(list)
        for jid, url in rows:
            parsed = urlparse(url)
            domain = parsed.netloc.lower().split(":")[0]
            path = parsed.path.lower()
            ext = os.path.splitext(path)[1]
            if path.endswith(".pdf") or ".pdf" in path:
                utype = "pdf"
            elif ext in (".html", ".htm"):
                utype = "html"
            elif ext in (".aspx", ".asp", ".php", ".jsp"):
                utype = "dynamic"
            elif not ext or ext == "/":
                utype = "slug"
            else:
                utype = ext.lstrip(".")[:8]
            strata[f"{domain}::{utype}"].append(jid)

        total_items = len(rows)
        effective_sample_size = min(sample_size, total_items)
        sorted_keys = sorted(strata.keys())
        num_strata = len(sorted_keys)

        quotas: dict[str, int] = {}
        if effective_sample_size >= num_strata:
            for k in sorted_keys:
                quotas[k] = 1
            remaining = effective_sample_size - num_strata
            if remaining > 0:
                props = {k: remaining * (len(strata[k]) / total_items) for k in sorted_keys}
                floors = {k: int(props[k]) for k in sorted_keys}
                rems = sorted(sorted_keys, key=lambda k: (props[k] - floors[k], k), reverse=True)
                extra_needed = remaining - sum(floors.values())
                for i in range(extra_needed):
                    floors[rems[i]] += 1
                for k in sorted_keys:
                    quotas[k] += floors[k]
        else:
            props = {k: effective_sample_size * (len(strata[k]) / total_items) for k in sorted_keys}
            floors = {k: int(props[k]) for k in sorted_keys}
            rems = sorted(sorted_keys, key=lambda k: (props[k] - floors[k], len(strata[k]), k), reverse=True)
            extra_needed = effective_sample_size - sum(floors.values())
            for i in range(extra_needed):
                floors[rems[i]] += 1
            for k in sorted_keys:
                quotas[k] = floors[k]

        rng = random.Random(seed)
        sampled_ids: list[str] = []
        strata_summary: dict[str, dict[str, Any]] = {}

        for k in sorted_keys:
            stratum_jobs = sorted(strata[k])
            q = min(quotas[k], len(stratum_jobs))
            sample = rng.sample(stratum_jobs, q) if q > 0 else []
            sampled_ids.extend(sample)
            strata_summary[k] = {
                "stratum_size": len(stratum_jobs),
                "sampled_count": len(sample),
            }

        sampled_ids.sort()

        if persist_path is not None:
            p = Path(persist_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            meta = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "seed": seed,
                "target_sample_size": sample_size,
                "actual_sample_size": len(sampled_ids),
                "strata_count": num_strata,
                "sampled_job_ids": sampled_ids,
                "strata_summary": strata_summary,
            }
            with open(p, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

        return sampled_ids

    def lease_jobs(
        self,
        limit: int,
        lease_seconds: int = 120,
        max_attempts: int = 3,
        now: datetime | None = None,
        job_ids: Sequence[str] | None = None,
        campaign_id: str | None = None,
    ) -> list[FetchJob]:
        """Leases up to limit jobs, reclaiming expired leases and elapsed retry_wait jobs."""
        if limit <= 0:
            return []

        now_dt = now or datetime.now(timezone.utc)
        now_iso = now_dt.isoformat()
        lease_until_iso = (now_dt + timedelta(seconds=lease_seconds)).isoformat()

        with self.conn:
            # 1. Reclaim expired leases that have exceeded max_attempts
            self.conn.execute(
                """
                UPDATE fetch_jobs
                SET state = 'fetch_failed',
                    lease_until = NULL,
                    error_code = 'lease_timeout_exhausted',
                    updated_at = ?
                WHERE state IN ('fetching', 'leased')
                  AND lease_until IS NOT NULL
                  AND lease_until <= ?
                  AND attempts >= ?
                """,
                (now_iso, now_iso, max_attempts),
            )
            # Reclaim remaining expired leases back to pending
            self.conn.execute(
                """
                UPDATE fetch_jobs
                SET state = 'pending',
                    lease_until = NULL,
                    updated_at = ?
                WHERE state IN ('fetching', 'leased')
                  AND lease_until IS NOT NULL
                  AND lease_until <= ?
                """,
                (now_iso, now_iso),
            )

            # 2. Transition retry_wait jobs whose backoff expired back to pending
            self.conn.execute(
                """
                UPDATE fetch_jobs
                SET state = 'pending',
                    retry_at = NULL,
                    updated_at = ?
                WHERE state = 'retry_wait'
                  AND (retry_at IS NULL OR retry_at <= ?)
                """,
                (now_iso, now_iso),
            )

            # 3. Select pending jobs (optionally restricted to job_ids and campaign_id)
            if job_ids is not None:
                self.conn.execute(
                    "CREATE TEMP TABLE IF NOT EXISTS target_lease_jobs (jid TEXT PRIMARY KEY);"
                )
                self.conn.execute("DELETE FROM target_lease_jobs;")
                self.conn.executemany(
                    "INSERT OR IGNORE INTO target_lease_jobs (jid) VALUES (?)",
                    [(j,) for j in job_ids],
                )
                if campaign_id is not None:
                    rows = self.conn.execute(
                        """
                        SELECT j.job_id, j.url, j.attempts
                        FROM fetch_jobs j
                        JOIN target_lease_jobs t ON j.job_id = t.jid
                        WHERE j.state = 'pending' AND (j.campaign_id = ? OR j.campaign_id IS NULL)
                        ORDER BY j.attempts ASC, j.priority ASC, j.rowid ASC
                        LIMIT ?
                        """,
                        (campaign_id, limit),
                    ).fetchall()
                else:
                    rows = self.conn.execute(
                        """
                        SELECT j.job_id, j.url, j.attempts
                        FROM fetch_jobs j
                        JOIN target_lease_jobs t ON j.job_id = t.jid
                        WHERE j.state = 'pending'
                        ORDER BY j.attempts ASC, j.priority ASC, j.rowid ASC
                        LIMIT ?
                        """,
                        (limit,),
                    ).fetchall()
            else:
                if campaign_id is not None:
                    rows = self.conn.execute(
                        """
                        SELECT job_id, url, attempts
                        FROM fetch_jobs
                        WHERE state = 'pending' AND (campaign_id = ? OR campaign_id IS NULL)
                        ORDER BY attempts ASC, priority ASC, rowid ASC
                        LIMIT ?
                        """,
                        (campaign_id, limit),
                    ).fetchall()
                else:
                    rows = self.conn.execute(
                        """
                        SELECT job_id, url, attempts
                        FROM fetch_jobs
                        WHERE state = 'pending'
                        ORDER BY attempts ASC, priority ASC, rowid ASC
                        LIMIT ?
                        """,
                        (limit,),
                    ).fetchall()

            if not rows:
                return []

            leased_jobs: list[FetchJob] = []
            for job_id, url, attempts in rows:
                next_attempt = attempts + 1
                self.conn.execute(
                    """
                    UPDATE fetch_jobs
                    SET state = 'fetching',
                        attempts = ?,
                        lease_until = ?,
                        updated_at = ?
                    WHERE job_id = ?
                    """,
                    (next_attempt, lease_until_iso, now_iso, job_id),
                )
                leased_jobs.append(FetchJob(job_id=job_id, url=url, attempt=next_attempt))

            return leased_jobs

    def finish_fetch(self, result: FetchResult, now: datetime | None = None) -> None:
        """Atomically updates fetch_job state, hashes, metadata, and error codes."""
        now_iso = (now or datetime.now(timezone.utc)).isoformat()
        with self.conn:
            self.conn.execute(
                """
                UPDATE fetch_jobs
                SET state = ?,
                    raw_path = ?,
                    raw_sha256 = ?,
                    final_url = ?,
                    mime_type = ?,
                    http_status = ?,
                    error_code = ?,
                    retry_at = ?,
                    lease_until = NULL,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (
                    result.state,
                    result.raw_path,
                    result.raw_sha256,
                    result.final_url,
                    result.mime_type,
                    result.http_status,
                    result.error_code,
                    result.retry_at,
                    now_iso,
                    result.job_id,
                ),
            )

    def get_fetch_job(self, job_id: str) -> dict[str, Any] | None:
        """Returns fetch_job details as dictionary or None."""
        cursor = self.conn.execute(
            """
            SELECT job_id, url, state, attempts, lease_until, retry_at,
                   raw_path, raw_sha256, http_status, mime_type, final_url,
                   error_code, campaign_id, priority, created_at, updated_at
            FROM fetch_jobs
            WHERE job_id = ?
            """,
            (job_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        cols = [col[0] for col in cursor.description]
        return dict(zip(cols, row))

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
