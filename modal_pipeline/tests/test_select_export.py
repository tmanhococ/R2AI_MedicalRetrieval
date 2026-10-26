"""select_export.py --chunk-mode doc on a tiny bundle."""
from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT = Path(__file__).resolve().parent.parent / "select_export.py"


def _setup(tmp: Path) -> tuple[Path, Path, Path]:
    chunks = tmp / "bundle" / "chunks"
    chunks.mkdir(parents=True)
    # rows 0-2 in part_00000 (doc 7 twice, doc 8), rows 3-4 in part_90000 (doc 9)
    pq.write_table(pa.table({"doc_id": [7, 7, 8], "text": ["a0", "a1", "b0"]}), chunks / "part_00000.parquet")
    pq.write_table(pa.table({"doc_id": [9, 9], "text": ["c0", "c1"]}), chunks / "part_90000.parquet")
    pool = tmp / "scored_pool.parquet"
    pq.write_table(pa.table({"qid": [1, 1, 1, 2], "row": [1, 2, 4, 0], "doc_id": [7, 8, 9, 7],
                             "ce": [5.0, 1.0, 3.0, -1.0], "doc_rrf": [0.1] * 4,
                             "text": ["a1", "b0", "c1", "a0"]}), pool)
    queries = tmp / "query.parquet"
    pq.write_table(pa.table({"id": [1, 2], "query": ["q1", "q2"]}), queries)
    return tmp / "bundle", pool, queries


def _results(out: Path) -> list[dict]:
    with zipfile.ZipFile(io.BytesIO((out / "submission.zip").read_bytes())) as zf:
        return json.loads(zf.read("results.json"))


def test_doc_mode_emits_one_whole_document_per_selected_doc(tmp_path):
    bundle, pool, queries = _setup(tmp_path)
    out = tmp_path / "out"
    subprocess.run([sys.executable, str(SCRIPT), "--pool", str(pool), "--queries", str(queries), "--out", str(out),
                    "--chunk-mode", "doc", "--bundle", str(bundle), "--doc-min", "1"], check=True, capture_output=True)
    res = _results(out)
    assert res[0]["relevant_docs"] == [7, 9, 8]
    assert [c["chunk_text"] for c in res[0]["relevant_chunks"]] == ["a0\n\na1", "c0\n\nc1", "b0"]
    assert [c["doc_id"] for c in res[0]["relevant_chunks"]] == [7, 9, 8]
    assert res[1]["relevant_chunks"] == [{"doc_id": 7, "chunk_text": "a0\n\na1", "chunk_order": 0}]


def test_doc_mode_window_is_centred_on_best_chunk(tmp_path):
    bundle, pool, queries = _setup(tmp_path)
    out = tmp_path / "out"
    subprocess.run([sys.executable, str(SCRIPT), "--pool", str(pool), "--queries", str(queries), "--out", str(out),
                    "--chunk-mode", "doc", "--bundle", str(bundle), "--doc-chars", "2", "--doc-min", "1"],
                   check=True, capture_output=True)
    res = _results(out)
    assert res[0]["relevant_chunks"][0]["chunk_text"] == "a1"  # doc 7's best row is 1 -> second chunk
    assert res[0]["relevant_chunks"][1]["chunk_text"] == "c1"
