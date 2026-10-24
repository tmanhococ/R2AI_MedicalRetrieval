"""Run every pipeline stage locally on a tiny slice of the bundle: real E5, real reranker,
real Pyserini/Lucene, optionally real NLLB. The Modal volume is replaced by a local dir.

Needs a Python 3.12 env with CUDA torch, transformers<5, pyserini, faiss-cpu, modal, and Java 21:
    uv venv --python 3.12 ../.venv-gpu
    uv pip install --python ../.venv-gpu/Scripts/python.exe --index-url https://download.pytorch.org/whl/cu124 torch
    uv pip install --python ../.venv-gpu/Scripts/python.exe "transformers>=4.44,<5" sentencepiece protobuf \
        huggingface_hub pyarrow numpy pandas modal pytest pyserini faiss-cpu

    ../.venv-gpu/Scripts/python.exe modal_pipeline/local_smoke.py --bundle artifacts/modal_upload_v2 \
        --official-corpus ../hf_check/links_corpus_vi_en_zh.parquet --work ../smoke_work
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# Same guard as the Modal BM25 image: pyserini 2.4 builds an OpenAI client at import time.
os.environ.setdefault("OPENAI_API_KEY", "unused-pyserini-import-guard")


class _LocalVolume:
    def reload(self):
        pass

    def commit(self):
        pass


def _slice_part(src: Path, dst: Path, rows: int) -> int:
    t = pq.read_table(src)
    docs = t["doc_id"].to_numpy()
    cut = min(rows, len(docs))
    while cut < len(docs) and docs[cut] == docs[cut - 1]:
        cut += 1
    pq.write_table(t.slice(0, cut), dst)
    return cut


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--official-corpus", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=3000, help="rows taken from each sampled part")
    ap.add_argument("--nq", type=int, default=10)
    ap.add_argument("--translate", action="store_true", help="run NLLB (downloads the model if not cached)")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    vol_dir = args.work / "vol"
    shutil.rmtree(vol_dir, ignore_errors=True)
    (vol_dir / "data" / "chunks").mkdir(parents=True)
    src_parts = sorted((args.bundle / "chunks").glob("part_*.parquet"))
    picks = [p for p in src_parts if p.name == "part_00000.parquet"] + [p for p in src_parts if p.name.startswith("part_9")][:1]
    for p in picks:
        n = _slice_part(p, vol_dir / "data" / "chunks" / p.name, args.rows)
        print(f"slice {p.name}: {n} rows")
    shutil.copy2(args.bundle / "query.parquet", vol_dir / "data" / "query.parquet")
    shutil.copy2(args.official_corpus, vol_dir / "data" / "official_corpus.parquet")

    spec = importlib.util.spec_from_file_location("vb_app_local", HERE / "app.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    app.VOL = vol_dir
    app.vol = app.hf_vol = _LocalVolume()
    app.E5_DIR = Path(snapshot_download(app.E5_ID, local_files_only=True))
    app.RERANK_DIR = Path(snapshot_download(app.RERANK_ID, local_files_only=True))

    nq, v = args.nq, "smoke"
    o = {**app.DEFAULT_OPTS, "nq": nq, "version": v, "dense_k": 200, "bm25_chunk_k": 200, "bm25_doc_k": 100,
         "top_docs": 20, "pool_size": 60, "rerank_batches": 2}
    timings: dict[str, float] = {}

    def stage(name, fn, *a, **kw):
        t0 = time.time()
        out = fn(*a, **kw)
        timings[name] = round(time.time() - t0, 1)
        print(f"[{name}] {timings[name]}s {json.dumps(out, default=str)[:300]}", flush=True)
        return out

    stage("meta", app.build_meta.local)
    for s in app._shards():
        stage(f"encode {s['part']}", app.encode_part.local, s["part"], 512, 0, 8192, v)
    stage("bm25 index", app.build_bm25.local, 0, 1, v)
    if args.translate:
        stage("translate", app.translate_queries.local, nq, False)
    else:
        rows = [{"id": q, "vi": t, "en": None, "zh": None} for q, t in app._load_queries(nq)]
        (vol_dir / "queries").mkdir(exist_ok=True)
        (vol_dir / "queries" / "variants.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    app._check_run_opts(app._run_dir(0, nq, v), o)
    stage("dense search", app.dense_search.local, 0, nq, o["dense_k"], o["max_len"], v)
    stage("bm25 search", app.bm25_search.local, 0, nq, o["bm25_chunk_k"], o["bm25_doc_k"], o["bm25_k1"], o["bm25_b"], v, 2)
    stage("pool", app.build_pool.local, 0, nq, json.dumps(o))
    for b in range(o["rerank_batches"]):
        stage(f"rerank {b}", app.rerank_batch.local, 0, nq, b, o["rerank_batches"], v, 512, 4096)
    final = stage("final", app.finalize.local, 0, nq, json.dumps(o))

    from vb_common import quality_report, validate_results

    run = app._run_dir(0, nq, v)
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    scored = pq.read_table(run / "scored_pool.parquet", columns=["qid", "ce"]).to_pylist()
    errors = validate_results(results, [q for q, _ in app._load_queries(nq)])
    report = {"final": final, "quality": quality_report(results, scored), "format_errors": errors, "timings": timings}
    (args.work / "smoke_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("SMOKE PASS" if not errors and final["queries"] == nq else "SMOKE FAIL")
    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()
