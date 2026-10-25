"""slim_snapshot.py keeps only official documents and yields the same bundle as the full snapshot."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_prepare import _snapshot  # noqa: E402


def _prepare(snap: Path, root: Path, out: Path) -> None:
    subprocess.run([sys.executable, str(HERE / "prepare_upload.py"), "--snapshot", str(snap), "--queries",
                    str(root / "query.parquet"), "--out", str(out), "--official-corpus",
                    str(root / "official.parquet")], check=True, capture_output=True)


def test_slim_snapshot_keeps_official_docs_and_matches_bundle(tmp_path):
    snap = _snapshot(tmp_path)
    (snap / "manifest.json").write_text(json.dumps({"snapshot_id": "t", "chunk_count": 60}), encoding="utf-8")
    slim = tmp_path / "slim"
    subprocess.run([sys.executable, str(HERE / "slim_snapshot.py"), "--snapshot", str(snap), "--official-corpus",
                    str(tmp_path / "official.parquet"), "--out", str(slim)], check=True, capture_output=True)

    chunks = pq.read_table(slim / "chunks_00000.parquet")
    assert sorted(set(chunks["doc_id"].to_pylist())) == [1, 3]
    assert chunks.schema == pq.read_table(snap / "chunks_00000.parquet").schema
    assert sorted(pq.read_table(slim / "documents_00000.parquet")["doc_id"].to_pylist()) == [1, 3]
    m = json.loads((slim / "manifest.json").read_text(encoding="utf-8"))
    assert m["derived_from"] == "t" and m["chunk_count"] == 4 and m["ready_document_count"] == 2
    assert {s["path"] for s in m["shard_paths_and_hashes"]} == {"chunks_00000.parquet", "documents_00000.parquet"}

    _prepare(snap, tmp_path, tmp_path / "out_full")
    _prepare(slim, tmp_path, tmp_path / "out_slim")
    full = pq.read_table(tmp_path / "out_full" / "chunks" / "part_00000.parquet")
    thin = pq.read_table(tmp_path / "out_slim" / "chunks" / "part_00000.parquet")
    assert full.equals(thin)


def test_slim_snapshot_skips_shards_without_official_docs(tmp_path):
    snap = _snapshot(tmp_path)
    pq.write_table(pa.Table.from_pylist([{"doc_id": 2, "title": "x"}]), snap / "documents_00001.parquet")
    slim = tmp_path / "slim"
    subprocess.run([sys.executable, str(HERE / "slim_snapshot.py"), "--snapshot", str(snap), "--official-corpus",
                    str(tmp_path / "official.parquet"), "--out", str(slim)], check=True, capture_output=True)
    assert not (slim / "documents_00001.parquet").exists()
