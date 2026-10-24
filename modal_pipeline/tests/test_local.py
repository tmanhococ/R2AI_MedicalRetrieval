"""Local tests for the Modal pipeline: shared helpers plus the CPU stages on real data.

GPU and Java stages are replaced by synthetic search/rerank outputs; everything else
(meta, BM25 tokenization, fusion pool, finalize, export) runs the real code against a
temp dir standing in for the Modal volume.

    .venv/Scripts/python.exe -m pytest modal_pipeline/tests -q
"""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

PIPELINE = Path(__file__).resolve().parent.parent
REPO = PIPELINE.parent
BUNDLE = REPO / "artifacts" / "modal_upload"
sys.path.insert(0, str(PIPELINE))

import vb_common as vc  # noqa: E402

NQ = 12
PART_ROWS = (5000, 3000)


def test_bm25_tokens_latin_and_cjk():
    assert vc.bm25_tokens("Beta-HCG dưới 0.10 U/L, HER2+") == ["beta-hcg", "dưới", "0", "10", "u", "l", "her2+"]
    assert vc.bm25_tokens("眼睛胀痛") == ["眼睛", "睛胀", "胀痛"]
    assert vc.bm25_tokens("单") == ["单"]
    nfd = "Kết quả"  # decomposed Vietnamese
    assert vc.bm25_tokens(nfd) == vc.bm25_tokens("Kết quả")


def test_query_string_caps_distinct_terms():
    q = vc.bm25_query_string(" ".join(f"w{i}" for i in range(3000)) + " w1 w1")
    assert len(set(q.split())) == vc.MAX_QUERY_TERMS
    assert q.split().count("w1") == 3


def test_selection_rules():
    rng = np.random.default_rng(0)
    rows = [{"qid": 1, "doc_id": r // 3, "ce": float(rng.uniform(-8, 4)), "doc_rrf": 0.0, "text": f"t{r}"}
            for r in range(300)]
    rows.append({"qid": 1, "doc_id": 0, "ce": 9.0, "doc_rrf": 0.0, "text": "t0"})  # duplicate text
    p = vc.SelectionParams()
    res = vc.build_results([1, 2], rows, p)
    assert [r["id"] for r in res] == [1, 2]
    assert res[1] == {"id": 2, "relevant_docs": [], "relevant_chunks": []}
    docs = set(res[0]["relevant_docs"])
    chunks = res[0]["relevant_chunks"]
    assert len(docs) <= p.doc_max and len(chunks) <= p.chunk_max
    assert all(c["doc_id"] in docs for c in chunks)
    assert len({c["chunk_text"] for c in chunks}) == len(chunks)
    per_doc = pd.Series([c["doc_id"] for c in chunks]).value_counts()
    assert per_doc.max() <= p.chunk_per_doc
    assert [c["chunk_order"] for c in chunks] == list(range(len(chunks)))
    tight = vc.build_results([1], rows, vc.SelectionParams(doc_min_logit=99, chunk_min_logit=99, doc_min=5, chunk_min=7))
    assert len(tight[0]["relevant_docs"]) == 5 and len(tight[0]["relevant_chunks"]) == 7
    assert vc.validate_results(res, [1, 2]) == []
    assert vc.validate_results(res, [1]) != []


class _FakeVolume:
    def reload(self):
        pass

    def commit(self):
        pass


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    src = BUNDLE / "chunks"
    if not (src / "part_00000.parquet").exists():
        pytest.skip("artifacts/modal_upload not prepared")
    vol = tmp_path_factory.mktemp("vol")
    (vol / "data" / "chunks").mkdir(parents=True)
    for i, n in enumerate(PART_ROWS):
        t = pq.read_table(src / f"part_{i:05d}.parquet")
        docs = t["doc_id"].to_numpy()
        cut = n
        while cut < len(docs) and docs[cut] == docs[cut - 1]:  # keep whole documents
            cut += 1
        pq.write_table(t.slice(0, cut), vol / "data" / "chunks" / f"part_{i:05d}.parquet")
    pq.write_table(pq.read_table(BUNDLE / "query.parquet"), vol / "data" / "query.parquet")

    spec = importlib.util.spec_from_file_location("vb_app_under_test", PIPELINE / "app.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    app.VOL = vol
    app.vol = _FakeVolume()
    return app, vol


def test_meta(stack):
    app, vol = stack
    meta = app.build_meta.local()
    shards = app._shards()
    rows = sum(pq.read_metadata(vol / "data" / "chunks" / s["part"]).num_rows for s in shards)
    assert meta["parts"] == 2 and meta["rows"] == rows
    assert shards[1]["offset"] == shards[0]["rows"]
    assert len(np.load(vol / "meta" / "chunk_doc.npy")) == rows


def test_tokenize_part(stack, tmp_path):
    app, vol = stack
    s = app._shards()[1]
    out_c, out_d = tmp_path / "c", tmp_path / "d"
    out_c.mkdir()
    out_d.mkdir()
    stats = app._tokenize_part((str(vol / "data" / "chunks" / s["part"]), s["offset"], str(out_c), str(out_d)))
    lines = (out_c / "part_00001.jsonl").read_text(encoding="utf-8").splitlines()
    assert stats["rows"] == s["rows"] == len(lines)
    assert json.loads(lines[0])["id"] == str(s["offset"])
    assert json.loads(lines[-1])["id"] == str(s["offset"] + s["rows"] - 1)
    doc_ids = pq.read_table(vol / "data" / "chunks" / s["part"], columns=["doc_id"])["doc_id"].to_pylist()
    doc_lines = [json.loads(x) for x in (out_d / "part_00001.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [int(d["id"]) for d in doc_lines] == list(dict.fromkeys(doc_ids))
    assert all(d["contents"] for d in doc_lines)


def test_pool_and_finalize(stack):
    app, vol = stack
    app.build_meta.local()
    shards = app._shards()
    n_rows = shards[-1]["offset"] + shards[-1]["rows"]
    chunk_doc = np.load(vol / "meta" / "chunk_doc.npy")

    queries = app._load_queries(NQ)
    (vol / "queries").mkdir(exist_ok=True)
    (vol / "queries" / "variants.jsonl").write_text(
        "".join(json.dumps({"id": q, "vi": t, "en": t, "zh": None}, ensure_ascii=False) + "\n" for q, t in queries),
        encoding="utf-8",
    )
    run = app._run_dir(2, NQ)
    run.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    docs_all = np.unique(chunk_doc)

    def fake(name, k, col):
        rows = []
        for qid, _ in queries:
            for lang in ("vi", "en"):
                items = rng.choice(docs_all if col == "doc_id" else n_rows, size=k, replace=False)
                rows += [(qid, lang, r + 1, int(it), float(k - r)) for r, it in enumerate(items)]
        pd.DataFrame(rows, columns=["qid", "lang", "rank", col, "score"]).to_parquet(run / f"{name}.parquet")

    fake("dense", 1000, "row")
    fake("bm25_chunks", 1000, "row")
    fake("bm25_docs", 300, "doc_id")

    opts = {**app.DEFAULT_OPTS, "nq": NQ, "rerank_batches": 3}
    info = app.build_pool.local(2, NQ, json.dumps(opts))
    assert info["queries"] == NQ
    pool = pd.read_parquet(run / "pool.parquet")
    assert pool.groupby("qid").size().max() <= opts["pool_size"]
    assert not pool.duplicated(["qid", "row"]).any()
    assert (chunk_doc[pool["row"].to_numpy()] == pool["doc_id"].to_numpy()).all()
    for s in shards:
        texts = pq.read_table(vol / "data" / "chunks" / s["part"], columns=["text"])["text"].to_pylist()
        part = pool[(pool["row"] >= s["offset"]) & (pool["row"] < s["offset"] + s["rows"])]
        assert len(part) > 0
        assert all(texts[r - s["offset"]] == t for r, t in zip(part["row"], part["text"]))

    (run / "rerank").mkdir(exist_ok=True)
    qids = sorted(pool["qid"].unique())
    for b in range(3):
        p = pool[pool["qid"].isin(qids[b::3])]
        pd.DataFrame({"qid": p["qid"], "row": p["row"], "ce": rng.normal(-2, 3, len(p)).astype("float32")}).to_parquet(
            run / "rerank" / f"scores_{b:03d}.parquet"
        )
    final = app.finalize.local(2, NQ, json.dumps(opts))
    assert final["queries"] == NQ and final["avg_docs"] > 0 and final["avg_chunks"] > 0
    with zipfile.ZipFile(io.BytesIO((run / "submission.zip").read_bytes())) as zf:
        assert zf.namelist() == ["results.json"]
        results = json.loads(zf.read("results.json"))
    assert vc.validate_results(results, [q for q, _ in queries]) == []
    scored = pd.read_parquet(run / "scored_pool.parquet")
    assert len(scored) == len(pool) and scored["ce"].notna().all()
