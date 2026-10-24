"""prepare_upload.py on a tiny synthetic snapshot."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT = Path(__file__).resolve().parent.parent / "prepare_upload.py"


def _chunk(doc, idx, text, lang="vi"):
    return {"chunk_id": f"{doc}:{idx}", "doc_id": doc, "source_version": "v", "source_sha256": "s",
            "start_char": 0, "end_char": len(text), "chunk_index": idx, "language": lang, "section": None,
            "text": text, "index_text": text, "dense_token_count": 1, "reranker_token_count": 1}


def _snapshot(root: Path) -> Path:
    snap = root / "snap"
    snap.mkdir()
    rows = [_chunk(1, 0, "doc one a"), _chunk(1, 1, "doc one b"),
            _chunk(2, 0, "doc two off corpus"),
            _chunk(3, 0, "doc three"), _chunk(3, 1, "menu")]
    rows += [_chunk(10 + i, 0, "menu") for i in range(5)]  # boilerplate only in off-corpus docs
    # many distinct off-corpus texts: some hash beyond every in-corpus hash (regression for an IndexError)
    rows += [_chunk(100 + i, 0, f"off corpus text {i}") for i in range(50)]
    pq.write_table(pa.Table.from_pylist(rows), snap / "chunks_00000.parquet")
    pq.write_table(pa.Table.from_pylist([{"doc_id": d, "title": f"T{d}"} for d in (1, 2, 3, *range(10, 15), *range(100, 150))]),
                   snap / "documents_00000.parquet")
    pq.write_table(pa.table({"id": [1], "query": ["q"]}), root / "query.parquet")
    pq.write_table(pa.table({"id": [1, 3, 99], "url": ["u1", "u3", "u99"]}), root / "official.parquet")
    return snap


def _run(root: Path, *extra: str) -> dict:
    out = root / "out"
    subprocess.run([sys.executable, str(SCRIPT), "--snapshot", str(root / "snap"), "--queries",
                    str(root / "query.parquet"), "--out", str(out), *extra], check=True, capture_output=True)
    return json.loads((out / "upload_manifest.json").read_text())


def test_official_filter_drops_off_corpus_docs(tmp_path):
    _snapshot(tmp_path)
    m = _run(tmp_path, "--official-corpus", str(tmp_path / "official.parquet"))
    t = pq.read_table(tmp_path / "out" / "chunks" / "part_00000.parquet")
    assert sorted(set(t["doc_id"].to_pylist())) == [1, 3]
    # boilerplate is counted inside the official corpus only: "menu" occurs once there, so it is kept
    assert "menu" in t["text"].to_pylist()
    assert m["official_total"] == 3 and m["docs_out"] == 2 and m["dropped_off_corpus"] == 56
    # the bundle carries the official list so finalize can reject off-corpus docs on Modal
    assert pq.read_table(tmp_path / "out" / "official_corpus.parquet")["id"].to_pylist() == [1, 3, 99]


def test_without_official_filter_keeps_all_docs(tmp_path):
    _snapshot(tmp_path)
    m = _run(tmp_path)
    t = pq.read_table(tmp_path / "out" / "chunks" / "part_00000.parquet")
    assert 2 in set(t["doc_id"].to_pylist())
    assert "menu" not in t["text"].to_pylist()  # 6 copies corpus-wide -> boilerplate
    assert m["dropped_off_corpus"] == 0
