"""prune_to_official.py keeps only data of documents in the official corpus."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT = Path(__file__).resolve().parent.parent / "prune_to_official.py"


def _fixture(root: Path) -> Path:
    art = root / "artifacts" / "vibiomir"
    raw = art / "raw" / "ab"
    raw.mkdir(parents=True)
    for name in ("keep1", "keep2", "drop1", "shared"):
        (raw / f"{name}.gz").write_bytes(b"x")
    db = sqlite3.connect(art / "registry.sqlite")
    db.execute("create table sources (doc_id integer primary key, original_url text, fetch_job_id text)")
    db.execute("create table fetch_jobs (job_id text primary key, url text, raw_path text)")
    rows = [(1, "j1", "keep1"), (2, "j2", "drop1"), (3, "j3", "keep2"), (4, "j4", "shared"), (5, "j4", "shared")]
    for doc, job, raw_name in rows:
        db.execute("insert into sources values (?, ?, ?)", (doc, f"u{doc}", job))
        db.execute("insert or ignore into fetch_jobs values (?, ?, ?)",
                   (job, f"u{doc}", f"artifacts\\vibiomir\\raw\\ab\\{raw_name}.gz"))
    db.commit()
    db.close()
    (art / "failed_urls.jsonl").write_text(
        "".join(json.dumps({"job_id": j, "url": "x"}) + "\n" for j in ("j1", "j2", "j9")), encoding="utf-8")
    snap = art / "snapshots" / "phase2_official"
    snap.mkdir(parents=True)
    pq.write_table(pa.table({"doc_id": [1, 2, 3], "original_url": ["a", "b", "c"], "reason": ["robots_blocked"] * 3,
                             "http_status": pa.array([None, None, None], pa.int32())}), snap / "source_gaps.parquet")
    (snap / "gaps.json").write_text(json.dumps({"total_gaps": 3}), encoding="utf-8")
    pq.write_table(pa.table({"id": [1, 3, 5], "url": ["u1", "u3", "u5"]}), root / "official.parquet")
    return art


def _run(root: Path, *extra: str) -> str:
    out = subprocess.run([sys.executable, str(SCRIPT), "--root", str(root), "--official-corpus",
                          str(root / "official.parquet"), *extra], check=True, capture_output=True, text=True)
    return out.stdout


def test_dry_run_changes_nothing(tmp_path):
    art = _fixture(tmp_path)
    report = _run(tmp_path)
    assert "DRY RUN" in report
    assert (art / "raw" / "ab" / "drop1.gz").exists()
    assert sqlite3.connect(art / "registry.sqlite").execute("select count(*) from sources").fetchone()[0] == 5


def test_apply_keeps_only_official_data(tmp_path):
    art = _fixture(tmp_path)
    _run(tmp_path, "--apply")
    assert sorted(p.name for p in (art / "raw" / "ab").iterdir()) == ["keep1.gz", "keep2.gz", "shared.gz"]
    db = sqlite3.connect(art / "registry.sqlite")
    assert [r[0] for r in db.execute("select doc_id from sources order by doc_id")] == [1, 3, 5]
    assert sorted(r[0] for r in db.execute("select job_id from fetch_jobs")) == ["j1", "j3", "j4"]
    lines = (art / "failed_urls.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["job_id"] for x in lines] == ["j1"]
    gaps = pq.read_table(art / "snapshots" / "phase2_official" / "source_gaps.parquet")
    assert gaps["doc_id"].to_pylist() == [1, 3]
    assert json.loads((art / "snapshots" / "phase2_official" / "gaps.json").read_text())["total_gaps"] == 2
