"""ViBioMIR retrieval pipeline on Modal.

Stages (each is idempotent and skips work whose output already exists):
  meta       row offsets + chunk->doc map for the uploaded parts
  bench      encode a sample on 4 parts and print the projected embedding cost
  encode     E5-base embeddings per part            -> /vol/emb/part_XXXXX.npy (float16)
  bm25       Lucene (Pyserini) chunk + doc indexes  -> /vol/bm25/<tag>/{chunks,docs}
  translate  NLLB vi->en/zh query variants          -> /vol/queries/variants.jsonl
  search     exact GPU dense search + Lucene BM25   -> /vol/runs/<run>/{dense,bm25_*}.parquet
  pool       weighted-RRF fusion + rerank pool      -> /vol/runs/<run>/pool.parquet
  rerank     bge-reranker-v2-m3 logits              -> /vol/runs/<run>/rerank/*.parquet
  final      selection + results.json + submission.zip
  all        everything above except bench

Examples (from mvp-rag-medical/):
  modal run modal_pipeline/app.py --stage bench
  modal run modal_pipeline/app.py --stage all --parts 1 --nq 30     # cheap dry run
  modal run --detach modal_pipeline/app.py --stage all                # full run
  modal run modal_pipeline/app.py --stage final --opts my_opts.json   # re-select with other settings
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import time

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so add_local_python_source finds vb_common

from vb_common import VOLUME_NAME  # noqa: E402

APP_NAME = "vibiomir"
VOL = Path("/vol")
MODELS = Path("/models")
E5_ID = "intfloat/multilingual-e5-base"
RERANK_ID = "BAAI/bge-reranker-v2-m3"
NLLB_ID = "facebook/nllb-200-distilled-1.3B"
E5_DIR = MODELS / "e5"
RERANK_DIR = MODELS / "reranker"

# GPU choices: L4 ($0.80/h) is the cheapest per token for these encoder-only models.
GPU_ENCODE = ["L4", "A10"]
GPU_RERANK = ["L4", "A10"]
GPU_SMALL = ["L4", "A10"]
MAX_GPUS = 10  # Starter plan GPU concurrency

DEFAULT_OPTS = {
    "version": "v1",            # namespace of emb/, bm25/, runs/ outputs; bump to recompute after a fix
    "retranslate": False,       # True: redo cached query translations
    "nq": 0,                    # >0: only the first nq queries (dry runs)
    "max_len": 512,             # E5 passage truncation (tokens)
    "dense_k": 1000,            # dense hits per query variant
    "bm25_chunk_k": 1000,       # BM25 chunk hits per query variant
    "bm25_doc_k": 300,          # BM25 doc hits per query variant
    "bm25_k1": 1.2,
    "bm25_b": 0.75,
    "rrf_k": 60,
    "w_dense": 1.0,
    "w_sparse_chunk": 0.75,
    "w_sparse_doc": 1.0,
    "top_docs": 150,            # fused docs guaranteed a chunk in the rerank pool
    "chunks_per_new_doc": 2,    # chunks picked lexically for top docs without a chunk hit
    "pool_size": 400,           # (query, chunk) pairs reranked per query
    "rerank_batches": 40,
    "selection": {},            # overrides for vb_common.SelectionParams
}

_HERE = Path(__file__).resolve().parent
_LEXICAL = _HERE.parent / "src" / "indexing" / "lexical_tokens.py"

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_vol = modal.Volume.from_name("vibiomir-hf-cache", create_if_missing=True)


def _download_models() -> None:
    from huggingface_hub import snapshot_download

    ignore = ["*.bin", "*.onnx", "onnx/*", "openvino/*", "*.h5", "*.msgpack", "*.ot"]
    snapshot_download(E5_ID, local_dir=str(E5_DIR), ignore_patterns=ignore)
    snapshot_download(RERANK_ID, local_dir=str(RERANK_DIR), ignore_patterns=ignore)


_base = modal.Image.debian_slim(python_version="3.12")

cpu_image = (
    _base.uv_pip_install("pyarrow>=16", "numpy>=1.26,<3", "pandas>=2.2")
    .add_local_file(str(_LEXICAL), "/root/lexical_tokens.py")
    .add_local_python_source("vb_common")
)

gpu_image = (
    _base.uv_pip_install(
        "torch>=2.4,<3",
        "transformers>=4.44,<5",
        "sentencepiece",
        "protobuf",
        "huggingface_hub",
        "pyarrow>=16",
        "numpy>=1.26,<3",
        "pandas>=2.2",
    )
    .run_function(_download_models)
    .env({"HF_HOME": "/hf", "TOKENIZERS_PARALLELISM": "true"})
    .add_local_file(str(_LEXICAL), "/root/lexical_tokens.py")
    .add_local_python_source("vb_common")
)

bm25_image = (
    _base.apt_install("curl", "ca-certificates")
    .run_commands(
        "mkdir -p /opt/jdk"
        " && curl -fsSL -o /tmp/jdk.tgz"
        " https://api.adoptium.net/v3/binary/latest/21/ga/linux/x64/jdk/hotspot/normal/eclipse"
        " && tar -xzf /tmp/jdk.tgz -C /opt/jdk --strip-components=1"
        " && rm /tmp/jdk.tgz && ln -sf /opt/jdk/bin/java /usr/local/bin/java"
    )
    .env({"JAVA_HOME": "/opt/jdk"})
    .uv_pip_install("pyserini==2.4.0", "faiss-cpu", "snowballstemmer", "pyarrow>=16", "numpy>=1.26,<3", "pandas>=2.2")
    .add_local_file(str(_LEXICAL), "/root/lexical_tokens.py")
    .add_local_python_source("vb_common")
)

app = modal.App(APP_NAME)


# ---------------------------------------------------------------- helpers

def _data_dir() -> Path:
    for cand in [VOL / "data", *sorted((VOL / "data").glob("*"))]:
        if (cand / "chunks").is_dir():
            return cand
    raise FileNotFoundError("No chunks/ under /vol/data. Upload the bundle first (modal_pipeline/upload.py).")


def _shards(parts: int = 0) -> list[dict]:
    path = VOL / "meta" / "shards.json"
    if not path.exists():
        raise FileNotFoundError("Run --stage meta first.")
    shards = json.loads(path.read_text())
    return shards[:parts] if parts else shards


def _meta_is_current() -> bool:
    """False when the uploaded parts differ from those recorded in meta/shards.json."""
    path = VOL / "meta" / "shards.json"
    if not path.exists():
        return False
    on_volume = sorted(p.name for p in (_data_dir() / "chunks").glob("part_*.parquet"))
    return on_volume == [s["part"] for s in json.loads(path.read_text())]


def _index_tag(parts: int) -> str:
    return "full" if not parts else f"p{parts}"


# Stages skip outputs that already exist, so every output path carries what produced it:
# bump the "version" opt to recompute after a code fix instead of deleting volume files.
def _emb_path(part: str, max_len: int, version: str) -> Path:
    return VOL / "emb" / f"{version}_len{max_len}" / (Path(part).stem + ".npy")


def _bm25_dir(parts: int, version: str) -> Path:
    return VOL / "bm25" / version / _index_tag(parts)


def _run_dir(parts: int, nq: int, version: str = DEFAULT_OPTS["version"]) -> Path:
    return VOL / "runs" / version / (_index_tag(parts) + (f"_q{nq}" if nq else ""))


def _index_done(target: Path, name: str, rows: int) -> bool:
    marker = target / f"{name}.done"
    return marker.exists() and json.loads(marker.read_text()).get("rows") == rows


# Opts that change search/pool/rerank outputs; a run dir must not mix two settings of them.
RUN_KEYS = ("version", "max_len", "dense_k", "bm25_chunk_k", "bm25_doc_k", "bm25_k1", "bm25_b", "rrf_k",
            "w_dense", "w_sparse_chunk", "w_sparse_doc", "top_docs", "chunks_per_new_doc", "pool_size",
            "rerank_batches")


def _check_run_opts(run: Path, o: dict) -> None:
    current = {k: o[k] for k in RUN_KEYS}
    path = run / "run_opts.json"
    if not path.exists():
        run.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(current, indent=2))
        return
    saved = json.loads(path.read_text())
    changed = sorted(k for k in RUN_KEYS if saved.get(k) != current[k])
    if changed:
        raise ValueError(
            f"{run.name} was built with different {changed}; existing outputs would be reused silently. "
            f'Pass a new "version" in --opts (e.g. "v2") to compute a fresh run.'
        )


def _load_queries(nq: int = 0) -> list[tuple[int, str]]:
    import pyarrow.parquet as pq

    t = pq.read_table(_data_dir() / "query.parquet", columns=["id", "query"])
    qs = sorted(zip(t["id"].to_pylist(), t["query"].to_pylist()))
    return qs[:nq] if nq else qs


def _load_variants(nq: int = 0) -> list[dict]:
    path = VOL / "queries" / "variants.jsonl"
    if not path.exists():
        raise FileNotFoundError("Run --stage translate first.")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda r: r["id"])
    return rows[:nq] if nq else rows


def _query_variant_list(nq: int) -> list[tuple[int, str, str]]:
    return [(v["id"], lang, v[lang]) for v in _load_variants(nq) for lang in ("vi", "en", "zh") if v.get(lang)]


def _embed(model, tok, ids: list[list[int]], token_budget: int):
    """Length-sorted, token-budgeted batches; mean pooling + L2 norm (E5 recipe)."""
    import numpy as np
    import torch

    n = len(ids)
    out = np.empty((n, model.config.hidden_size), dtype=np.float16)
    if n == 0:
        return out
    lengths = np.fromiter((len(x) for x in ids), dtype=np.int64, count=n)
    order = np.argsort(-lengths, kind="stable")
    i = 0
    while i < n:
        bs = max(1, token_budget // int(lengths[order[i]]))
        sel = order[i : i + bs]
        batch = tok.pad({"input_ids": [ids[j] for j in sel]}, return_tensors="pt")
        input_ids = batch["input_ids"].cuda(non_blocking=True)
        mask = batch["attention_mask"].cuda(non_blocking=True)
        with torch.inference_mode():
            h = model(input_ids=input_ids, attention_mask=mask).last_hidden_state.float()
            m = mask.unsqueeze(-1).float()
            e = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1).clamp(min=1.0), dim=-1)
        out[sel] = e.half().cpu().numpy()
        i += bs
    return out


def _load_e5():
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(E5_DIR))
    model = AutoModel.from_pretrained(str(E5_DIR)).half().cuda().eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    return tok, model


def _write_parquet(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)


# ---------------------------------------------------------------- stages

@app.function(image=cpu_image, volumes={VOL: vol}, cpu=2, memory=16384, timeout=3600)
def build_meta() -> dict:
    import numpy as np
    import pyarrow.parquet as pq

    vol.reload()
    parts = sorted((_data_dir() / "chunks").glob("part_*.parquet"))
    shards, docs, offset = [], [], 0
    for p in parts:
        d = pq.read_table(p, columns=["doc_id"])["doc_id"].to_numpy()
        shards.append({"part": p.name, "offset": offset, "rows": int(len(d))})
        docs.append(d)
        offset += len(d)
    chunk_doc = np.concatenate(docs).astype(np.int64)
    # Re-sharding in prepare_upload guarantees each doc is one contiguous run of rows.
    run_starts = np.flatnonzero(np.r_[True, chunk_doc[1:] != chunk_doc[:-1]])
    n_docs = len(np.unique(chunk_doc))
    if len(run_starts) != n_docs:
        raise ValueError(f"documents are not contiguous: {len(run_starts)} runs for {n_docs} docs")
    out = VOL / "meta"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "chunk_doc.npy", chunk_doc)
    (out / "shards.json").write_text(json.dumps(shards))
    vol.commit()
    return {"parts": len(parts), "rows": int(offset), "docs": int(n_docs)}


@app.function(
    image=gpu_image, gpu=GPU_ENCODE, volumes={VOL: vol}, cpu=2, memory=12288,
    timeout=2 * 3600, retries=1, max_containers=MAX_GPUS,
)
def encode_part(part: str, max_len: int = 512, sample: int = 0, token_budget: int = 32768,
                version: str = DEFAULT_OPTS["version"]) -> dict:
    import numpy as np
    import pyarrow.parquet as pq
    from vb_common import compose_passage

    out = _emb_path(part, max_len, version)
    if not sample and out.exists():
        return {"part": part, "skipped": True}
    t = pq.read_table(_data_dir() / "chunks" / part, columns=["title", "section", "text"])
    titles, sections, texts = (t[c].to_pylist() for c in ("title", "section", "text"))
    idx = range(t.num_rows)
    if sample:
        idx = np.random.default_rng(0).choice(t.num_rows, size=min(sample, t.num_rows), replace=False)
    passages = ["passage: " + compose_passage(titles[i], sections[i], texts[i]) for i in idx]

    tok, model = _load_e5()
    t0 = time.time()
    ids = tok(passages, truncation=True, max_length=max_len)["input_ids"]
    vecs = _embed(model, tok, ids, token_budget)
    secs = time.time() - t0
    if not sample:
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.stem + ".tmp.npy")
        np.save(tmp, vecs)
        os.replace(tmp, out)
        vol.commit()
    return {"part": part, "rows": len(passages), "tokens": int(sum(len(x) for x in ids)), "secs": round(secs, 1)}


@app.function(image=gpu_image, gpu=GPU_SMALL, volumes={VOL: vol, "/hf": hf_vol}, memory=16384, timeout=3600)
def translate_queries(nq: int = 0, force: bool = False) -> dict:
    """Translates the first nq queries (all when nq == 0) that are not yet in variants.jsonl."""
    import re

    vol.reload()
    out = VOL / "queries" / "variants.jsonl"
    done: dict[int, dict] = {}
    if out.exists() and not force:
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["id"]] = row
    queries = [(qid, q) for qid, q in _load_queries(nq) if qid not in done]
    if not queries:
        return {"skipped": True, "cached": len(done)}
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    # Persist the download before loading: commit() reloads the volume, which fails on open (mmapped) files.
    snapshot_download(NLLB_ID)
    hf_vol.commit()
    tok = AutoTokenizer.from_pretrained(NLLB_ID, src_lang="vie_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_ID).half().cuda().eval()

    # NLLB is sentence-level: translate sentences, then re-join per query.
    splitter = re.compile(r"(?<=[.!?…])\s+|\n+")
    q_sents = [[s.strip() for s in splitter.split(q) if s and s.strip()] or [q] for _, q in queries]
    uniq = sorted({s for ss in q_sents for s in ss}, key=len)
    variants: dict[str, dict[str, str]] = {"en": {}, "zh": {}}
    for lang, code in (("en", "eng_Latn"), ("zh", "zho_Hans")):
        bos = tok.convert_tokens_to_ids(code)
        for i in range(0, len(uniq), 48):
            batch = uniq[i : i + 48]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
            gen = model.generate(**enc, forced_bos_token_id=bos, num_beams=4, max_new_tokens=256)
            for src, dst in zip(batch, tok.batch_decode(gen, skip_special_tokens=True)):
                variants[lang][src] = dst.strip()
    for (qid, q), ss in zip(queries, q_sents):
        done[int(qid)] = {
            "id": int(qid),
            "vi": q,
            "en": " ".join(variants["en"][s] for s in ss).strip() or None,
            "zh": "".join(variants["zh"][s] for s in ss).strip() or None,
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text("".join(json.dumps(done[k], ensure_ascii=False) + "\n" for k in sorted(done)), encoding="utf-8")
    os.replace(tmp, out)
    vol.commit()
    return {"translated": len(queries), "sentences": len(uniq), "cached_total": len(done)}


def _tokenize_part(args: tuple[str, int, str, str]) -> dict:
    import pyarrow.parquet as pq
    from vb_common import bm25_tokens

    path, offset, out_chunks, out_docs = args
    t = pq.read_table(path, columns=["doc_id", "language", "title", "section", "text"])
    stem = Path(path).stem
    title_cache: dict[int, list[str]] = {}
    doc_order: list[int] = []
    doc_body: dict[int, list[str]] = {}
    n_tokens = 0
    with open(f"{out_chunks}/{stem}.jsonl", "w", encoding="utf-8") as fc:
        cols = (t[c].to_pylist() for c in ("doc_id", "language", "title", "section", "text"))
        for i, (doc, lang, title, section, text) in enumerate(zip(*cols)):
            if doc not in title_cache:
                title_cache[doc] = bm25_tokens(title, lang)
                doc_order.append(doc)
                doc_body[doc] = []
            body = " ".join(bm25_tokens(text, lang))
            head = " ".join(title_cache[doc] + bm25_tokens(section, lang))
            n_tokens += body.count(" ") + 1
            fc.write(json.dumps({"id": str(offset + i), "contents": f"{head} {body}".strip()}, ensure_ascii=False) + "\n")
            doc_body[doc].append(body)  # joined strings: per-token lists would cost GBs per worker
    with open(f"{out_docs}/{stem}.jsonl", "w", encoding="utf-8") as fd:
        for doc in doc_order:
            contents = " ".join([" ".join(title_cache[doc]), *doc_body[doc]]).strip()
            fd.write(json.dumps({"id": str(doc), "contents": contents}, ensure_ascii=False) + "\n")
    return {"part": stem, "rows": t.num_rows, "tokens": n_tokens}


@app.function(image=bm25_image, volumes={VOL: vol}, cpu=16, memory=65536, timeout=10 * 3600)
def build_bm25(parts: int = 0, threads: int = 16, version: str = DEFAULT_OPTS["version"]) -> dict:
    import multiprocessing as mp
    import shutil
    import subprocess
    import sys

    vol.reload()
    target = _bm25_dir(parts, version)
    shards = _shards(parts)
    rows = sum(s["rows"] for s in shards)
    if _index_done(target, "chunks", rows) and _index_done(target, "docs", rows):
        return {"skipped": True}
    work = Path(tempfile.gettempdir()) / "bm25"
    in_c, in_d = work / "in_chunks", work / "in_docs"
    for d in (in_c, in_d):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    t0 = time.time()
    jobs = [(str(_data_dir() / "chunks" / s["part"]), s["offset"], str(in_c), str(in_d)) for s in shards]
    if threads > 1 and "fork" in mp.get_all_start_methods():
        with mp.get_context("fork").Pool(threads) as pool:
            stats = list(pool.imap_unordered(_tokenize_part, jobs))
    else:  # local runs on Windows: no fork
        stats = [_tokenize_part(j) for j in jobs]
    report = {"tokenize_secs": round(time.time() - t0), "tokens": sum(s["tokens"] for s in stats)}
    for name, inp in (("chunks", in_c), ("docs", in_d)):
        if _index_done(target, name, rows):
            continue
        idx = work / f"idx_{name}"
        shutil.rmtree(idx, ignore_errors=True)
        t1 = time.time()
        subprocess.run(
            [sys.executable, "-m", "pyserini.index.lucene", "-collection", "JsonCollection",
             "-input", str(inp), "-index", str(idx), "-generator", "DefaultLuceneDocumentGenerator",
             "-threads", str(threads), "-pretokenized"],
            check=True,
        )
        dest = target / name
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(idx, dest)
        (target / f"{name}.done").write_text(json.dumps({"secs": round(time.time() - t1), "rows": rows, "parts": len(shards)}))
        vol.commit()
        report[f"{name}_index_secs"] = round(time.time() - t1)
    return report


@app.function(image=bm25_image, volumes={VOL: vol}, cpu=16, memory=65536, timeout=4 * 3600)
def bm25_search(parts: int, nq: int, k_chunks: int, k_docs: int, k1: float, b: float, version: str,
                threads: int = 16) -> dict:
    import shutil

    import pandas as pd
    from vb_common import bm25_query_string

    vol.reload()
    run = _run_dir(parts, nq, version)
    src = _bm25_dir(parts, version)
    qs = [(qid, lang, bm25_query_string(text, lang)) for qid, lang, text in _query_variant_list(nq)]
    qs = [q for q in qs if q[2]]
    from pyserini.analysis import JWhiteSpaceAnalyzer
    from pyserini.search.lucene import LuceneSearcher

    report = {}
    for name, k, col in (("chunks", k_chunks, "row"), ("docs", k_docs, "doc_id")):
        out = run / f"bm25_{name}.parquet"
        if out.exists():
            report[name] = "skipped"
            continue
        local = Path(tempfile.gettempdir()) / f"idx_{name}"
        if not local.exists():
            shutil.copytree(src / name, local)
        searcher = LuceneSearcher(str(local))
        searcher.set_analyzer(JWhiteSpaceAnalyzer())
        searcher.set_bm25(k1, b)
        t0 = time.time()
        res = searcher.batch_search([q for _, _, q in qs], [str(i) for i in range(len(qs))], k=k, threads=threads)
        rows = [
            (qid, lang, rank, int(h.docid), float(h.score))
            for i, (qid, lang, _) in enumerate(qs)
            for rank, h in enumerate(res.get(str(i), []), 1)
        ]
        _write_parquet(pd.DataFrame(rows, columns=["qid", "lang", "rank", col, "score"]), out)
        vol.commit()
        report[name] = {"hits": len(rows), "secs": round(time.time() - t0)}
    return report


@app.function(image=gpu_image, gpu=GPU_SMALL, volumes={VOL: vol}, cpu=4, memory=32768, timeout=2 * 3600)
def dense_search(parts: int, nq: int, k: int, max_len: int, version: str) -> dict:
    import numpy as np
    import pandas as pd
    import torch

    vol.reload()
    out = _run_dir(parts, nq, version) / "dense.parquet"
    if out.exists():
        return {"skipped": True}
    qs = _query_variant_list(nq)
    tok, model = _load_e5()
    ids = tok(["query: " + text for _, _, text in qs], truncation=True, max_length=max_len)["input_ids"]
    q = torch.from_numpy(_embed(model, tok, ids, 32768)).cuda()
    del model
    m = q.shape[0]
    best_s = torch.full((m, k), float("-inf"), device="cuda", dtype=torch.float16)
    best_i = torch.full((m, k), -1, device="cuda", dtype=torch.long)
    t0 = time.time()
    for s in _shards(parts):
        path = _emb_path(s["part"], max_len, version)
        if not path.exists():
            raise FileNotFoundError(f"missing embeddings for {s['part']}; run --stage encode")
        emb = torch.from_numpy(np.load(path)).cuda()
        if emb.shape[0] != s["rows"]:
            raise ValueError(f"{path.name}: {emb.shape[0]} vectors for {s['rows']} rows")
        scores = q @ emb.T
        top_s, top_i = scores.topk(min(k, emb.shape[0]), dim=1)
        cat_s = torch.cat([best_s, top_s], dim=1)
        cat_i = torch.cat([best_i, top_i + s["offset"]], dim=1)
        best_s, pos = cat_s.topk(k, dim=1)
        best_i = cat_i.gather(1, pos)
        del emb, scores
    bs, bi = best_s.float().cpu().numpy(), best_i.cpu().numpy()
    rows = [
        (qid, lang, r + 1, int(bi[j, r]), float(bs[j, r]))
        for j, (qid, lang, _) in enumerate(qs)
        for r in range(k)
        if bi[j, r] >= 0
    ]
    _write_parquet(pd.DataFrame(rows, columns=["qid", "lang", "rank", "row", "score"]), out)
    vol.commit()
    return {"vectors": m, "hits": len(rows), "scan_secs": round(time.time() - t0)}


@app.function(image=cpu_image, volumes={VOL: vol}, cpu=8, memory=49152, timeout=3 * 3600)
def build_pool(parts: int, nq: int, opts_json: str) -> dict:
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from vb_common import bm25_tokens

    o = {**DEFAULT_OPTS, **json.loads(opts_json)}
    vol.reload()
    run = _run_dir(parts, nq, o["version"])
    out = run / "pool.parquet"
    if out.exists():
        return {"skipped": True}
    shards = _shards(parts)
    n_rows = shards[-1]["offset"] + shards[-1]["rows"]
    chunk_doc = np.load(VOL / "meta" / "chunk_doc.npy")[:n_rows]
    first_row = np.flatnonzero(np.r_[True, chunk_doc[1:] != chunk_doc[:-1]])
    doc_first = dict(zip(chunk_doc[first_row].tolist(), first_row.tolist()))
    doc_count = dict(zip(chunk_doc[first_row].tolist(), np.diff(np.r_[first_row, n_rows]).tolist()))

    k = o["rrf_k"]
    dense = pd.read_parquet(run / "dense.parquet")
    bm25c = pd.read_parquet(run / "bm25_chunks.parquet")
    bm25d = pd.read_parquet(run / "bm25_docs.parquet")

    # Chunk-level fusion; translated variants of one query are correlated, so each
    # family contributes once per item using its best rank across variants.
    cd = dense.groupby(["qid", "row"], as_index=False)["rank"].min().rename(columns={"rank": "r_dense"})
    cs = bm25c.groupby(["qid", "row"], as_index=False)["rank"].min().rename(columns={"rank": "r_sc"})
    ch = cd.merge(cs, on=["qid", "row"], how="outer")
    ch["doc_id"] = chunk_doc[ch["row"].to_numpy()]
    ch["chunk_rrf"] = (o["w_dense"] / (k + ch["r_dense"])).fillna(0) + (o["w_sparse_chunk"] / (k + ch["r_sc"])).fillna(0)

    def doc_ranks(df: pd.DataFrame) -> pd.DataFrame:
        df = df.assign(doc_id=chunk_doc[df["row"].to_numpy()])
        first = df.groupby(["qid", "lang", "doc_id"], as_index=False)["rank"].min()
        first["drank"] = first.groupby(["qid", "lang"])["rank"].rank(method="first")
        return first.groupby(["qid", "doc_id"], as_index=False)["drank"].min()

    dd = doc_ranks(dense).rename(columns={"drank": "r_dense"})
    dsc = doc_ranks(bm25c).rename(columns={"drank": "r_sc"})
    dsd = bm25d.groupby(["qid", "doc_id"], as_index=False)["rank"].min().rename(columns={"rank": "r_sd"})
    dsd = dsd[dsd["doc_id"].isin(list(doc_first))]  # docs outside a subset run
    docs = dd.merge(dsc, on=["qid", "doc_id"], how="outer").merge(dsd, on=["qid", "doc_id"], how="outer")
    docs["doc_rrf"] = (
        (o["w_dense"] / (k + docs["r_dense"])).fillna(0)
        + (o["w_sparse_chunk"] / (k + docs["r_sc"])).fillna(0)
        + (o["w_sparse_doc"] / (k + docs["r_sd"])).fillna(0)
    )

    # Phase 1: per query, best chunk of each top doc; collect rows whose text is needed.
    ch = ch.sort_values(["qid", "chunk_rrf", "row"], ascending=[True, False, True])
    docs = docs.sort_values(["qid", "doc_rrf", "doc_id"], ascending=[True, False, True])
    ch_by_q = dict(tuple(ch.groupby("qid", sort=False)))
    docs_by_q = dict(tuple(docs.groupby("qid", sort=False)))
    plans, need = {}, set()
    for qid, qdocs in docs_by_q.items():
        qch = ch_by_q.get(qid, ch.iloc[0:0])
        best = qch.drop_duplicates("doc_id")
        best_row = dict(zip(best["doc_id"].tolist(), best["row"].tolist()))
        top = qdocs.head(o["top_docs"])["doc_id"].tolist()
        seeded = [best_row[d] for d in top if d in best_row]
        new_docs = [d for d in top if d not in best_row]
        rest = qch.head(o["pool_size"])["row"].tolist()
        plans[qid] = (seeded, new_docs, rest)
        need.update(seeded)
        need.update(rest)
        for d in new_docs:
            need.update(range(doc_first[d], doc_first[d] + min(doc_count[d], 40)))

    # Gather texts for needed rows, one pass over the parts.
    need_arr = np.array(sorted(need), dtype=np.int64)
    texts: dict[int, tuple] = {}
    for s in shards:
        lo, hi = np.searchsorted(need_arr, [s["offset"], s["offset"] + s["rows"]])
        if lo == hi:
            continue
        local = need_arr[lo:hi] - s["offset"]
        t = pq.read_table(_data_dir() / "chunks" / s["part"], columns=["title", "section", "text"]).take(local)
        for r, ti, se, tx in zip(need_arr[lo:hi].tolist(), *(t[c].to_pylist() for c in ("title", "section", "text"))):
            texts[r] = (ti, se, tx)

    # Phase 2: lexical picks for top docs that had no chunk hit, then assemble the pool.
    variants = {v["id"]: v for v in _load_variants(nq)}
    doc_rrf_map = docs.set_index(["qid", "doc_id"])["doc_rrf"]
    chunk_rrf_map = ch.set_index(["qid", "row"])["chunk_rrf"]
    out_rows = []
    tokset: dict[int, frozenset] = {}

    def row_tokens(r: int) -> frozenset:
        if r not in tokset:
            tokset[r] = frozenset(bm25_tokens(texts[r][2]))
        return tokset[r]

    for qid, (seeded, new_docs, rest) in plans.items():
        v = variants.get(qid, {})
        qtoks = set(bm25_tokens(" ".join(x for x in (v.get("vi"), v.get("en"), v.get("zh")) if x)))
        picked = []
        for d in new_docs:
            rows = range(doc_first[d], doc_first[d] + min(doc_count[d], 40))
            scored = sorted(rows, key=lambda r: (-len(qtoks & row_tokens(r)), r))
            picked.extend(scored[: o["chunks_per_new_doc"]])
        pool, seen = [], set()
        for r in seeded + picked + rest:
            if r not in seen:
                seen.add(r)
                pool.append(r)
            if len(pool) >= o["pool_size"]:
                break
        for r in pool:
            d = int(chunk_doc[r])
            ti, se, tx = texts[r]
            out_rows.append((qid, r, d, float(chunk_rrf_map.get((qid, r), 0.0)),
                             float(doc_rrf_map.get((qid, d), 0.0)), ti, se, tx))
    df = pd.DataFrame(out_rows, columns=["qid", "row", "doc_id", "chunk_rrf", "doc_rrf", "title", "section", "text"])
    _write_parquet(df, out)
    vol.commit()
    return {"queries": len(plans), "pairs": len(df), "pairs_per_query": round(len(df) / max(1, len(plans)), 1)}


@app.function(
    image=gpu_image, gpu=GPU_RERANK, volumes={VOL: vol}, cpu=2, memory=16384,
    timeout=3 * 3600, retries=1, max_containers=MAX_GPUS,
)
def rerank_batch(parts: int, nq: int, batch: int, nbatches: int, version: str = DEFAULT_OPTS["version"],
                 max_len: int = 512, token_budget: int = 16384) -> dict:
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from vb_common import compose_passage

    run = _run_dir(parts, nq, version)
    out = run / "rerank" / f"scores_{batch:03d}.parquet"
    if out.exists():
        return {"batch": batch, "skipped": True}
    pool = pd.read_parquet(run / "pool.parquet", columns=["qid", "row", "title", "section", "text"])
    qids = sorted(pool["qid"].unique())
    mine = set(qids[batch::nbatches])
    pool = pool[pool["qid"].isin(mine)].reset_index(drop=True)
    if pool.empty:  # more batches than queries in a dry run
        empty = {"qid": pd.Series([], dtype="int64"), "row": pd.Series([], dtype="int64"),
                 "ce": pd.Series([], dtype="float32")}
        _write_parquet(pd.DataFrame(empty), out)
        vol.commit()
        return {"batch": batch, "pairs": 0, "secs": 0}
    qtext = {v["id"]: v["vi"] for v in _load_variants(nq)}

    tok = AutoTokenizer.from_pretrained(str(RERANK_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(RERANK_DIR)).half().cuda().eval()
    t0 = time.time()
    passages = [compose_passage(a, b, c) for a, b, c in zip(pool["title"], pool["section"], pool["text"])]
    enc = tok([qtext[q] for q in pool["qid"]], passages, truncation="longest_first", max_length=max_len)
    ids = enc["input_ids"]
    lengths = np.fromiter((len(x) for x in ids), dtype=np.int64, count=len(ids))
    order = np.argsort(-lengths, kind="stable")
    scores = np.empty(len(ids), dtype=np.float32)
    i = 0
    while i < len(ids):
        sel = order[i : i + max(1, token_budget // int(lengths[order[i]]))]
        batch_enc = tok.pad({"input_ids": [ids[j] for j in sel]}, return_tensors="pt")
        with torch.inference_mode():
            logits = model(
                input_ids=batch_enc["input_ids"].cuda(non_blocking=True),
                attention_mask=batch_enc["attention_mask"].cuda(non_blocking=True),
            ).logits[:, 0]
        scores[sel] = logits.float().cpu().numpy()
        i += len(sel)
    _write_parquet(pd.DataFrame({"qid": pool["qid"], "row": pool["row"], "ce": scores}), out)
    vol.commit()
    return {"batch": batch, "pairs": len(ids), "secs": round(time.time() - t0)}


@app.function(image=cpu_image, volumes={VOL: vol}, cpu=2, memory=16384, timeout=3600)
def finalize(parts: int, nq: int, opts_json: str) -> dict:
    import pandas as pd
    from vb_common import SelectionParams, build_results, results_to_zip_bytes, validate_results

    o = {**DEFAULT_OPTS, **json.loads(opts_json)}
    vol.reload()
    run = _run_dir(parts, nq, o["version"])
    pool = pd.read_parquet(run / "pool.parquet", columns=["qid", "row", "doc_id", "chunk_rrf", "doc_rrf", "text"])
    files = sorted((run / "rerank").glob("scores_*.parquet"))
    if len(files) != o["rerank_batches"]:
        raise RuntimeError(f"expected {o['rerank_batches']} rerank files, found {len(files)}")
    scores = pd.concat([pd.read_parquet(f) for f in files])
    scored = pool.merge(scores, on=["qid", "row"], how="left")
    if scored["ce"].isna().any():
        raise RuntimeError(f"{int(scored['ce'].isna().sum())} pool rows have no reranker score")
    # Documents outside the organizers' corpus can never be correct; drop them before selection.
    dropped_off_corpus = 0
    official_path = _data_dir() / "official_corpus.parquet"
    if official_path.exists():
        official = pd.read_parquet(official_path, columns=["id"])["id"]
        inside = scored["doc_id"].isin(official)
        dropped_off_corpus = int((~inside).sum())
        scored = scored[inside]
    _write_parquet(scored, run / "scored_pool.parquet")

    params = SelectionParams(**o["selection"])
    query_ids = [qid for qid, _ in _load_queries(nq)]
    results = build_results(query_ids, scored.to_dict("records"), params)
    errors = validate_results(results, query_ids)
    if errors:
        raise RuntimeError(f"submission validation failed: {errors[:5]}")
    js, zbytes = results_to_zip_bytes(results)
    (run / "results.json").write_bytes(js)
    (run / "submission.zip").write_bytes(zbytes)
    (run / "selection.json").write_text(json.dumps(params.to_dict(), indent=2))
    vol.commit()
    n = len(results)
    return {
        "queries": n,
        "avg_docs": round(sum(len(r["relevant_docs"]) for r in results) / n, 1),
        "avg_chunks": round(sum(len(r["relevant_chunks"]) for r in results) / n, 1),
        "zip_mb": round(len(zbytes) / 1e6, 1),
        "dropped_off_corpus_rows": dropped_off_corpus,
    }


@app.function(image=cpu_image, volumes={VOL: vol}, timeout=24 * 3600)
def pipeline(stage: str, parts: int = 0, opts_json: str = "{}") -> list[str]:
    o = {**DEFAULT_OPTS, **json.loads(opts_json)}
    nq = o["nq"]
    log: list[str] = []

    def note(msg: str, data=None) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}" + (f": {json.dumps(data)}" if data is not None else "")
        print(line, flush=True)
        log.append(line)

    all_stages = stage == "all"
    v = o["version"]
    # Rebuild meta whenever the uploaded parts changed (e.g. full upload after the tiny run);
    # a stale shard list would make bench, encode and bm25 silently cover only the old parts.
    if all_stages or stage == "meta" or not _meta_is_current():
        note("meta", build_meta.remote())
    vol.reload()
    shards = _shards(parts)
    if all_stages or stage in ("search", "pool", "rerank", "final"):
        _check_run_opts(_run_dir(parts, nq, v), o)
        vol.commit()
    rows = sum(s["rows"] for s in shards)

    if stage == "bench":
        full = _shards()
        picks = [full[i]["part"] for i in sorted({round(j * (len(full) - 1) / 3) for j in range(4)})]
        res = list(encode_part.map(picks, kwargs={"max_len": o["max_len"], "sample": 4000, "version": v}))
        rps = sum(r["rows"] for r in res) / sum(r["secs"] for r in res)
        total = sum(s["rows"] for s in full)
        hours = total / rps / 3600
        note("bench", {"samples": res, "rows_per_sec_per_gpu": round(rps), "total_rows": total,
                       "gpu_hours": round(hours, 2), "est_usd_L4": round(hours * (0.80 + 0.19), 2),
                       "wall_minutes_on_10_gpus": round(hours * 60 / MAX_GPUS)})
        return log

    bm25_call = translate_call = None
    if all_stages or stage == "bm25":
        bm25_call = build_bm25.spawn(parts, version=v)
    if all_stages or stage == "translate":
        translate_call = translate_queries.spawn(nq, o["retranslate"])
    if all_stages or stage == "encode":
        t0 = time.time()
        res = list(encode_part.map([s["part"] for s in shards], kwargs={"max_len": o["max_len"], "version": v}))
        done = [r for r in res if not r.get("skipped")]
        note("encode", {"parts": len(res), "encoded_now": len(done), "rows": rows,
                        "gpu_secs": round(sum(r["secs"] for r in done)), "wall_secs": round(time.time() - t0)})
    if bm25_call:
        note("bm25 index", bm25_call.get())
    if translate_call:
        note("translate", translate_call.get())

    if all_stages or stage == "search":
        d = dense_search.spawn(parts, nq, o["dense_k"], o["max_len"], v)
        s = bm25_search.spawn(parts, nq, o["bm25_chunk_k"], o["bm25_doc_k"], o["bm25_k1"], o["bm25_b"], v)
        note("dense search", d.get())
        note("bm25 search", s.get())
    if all_stages or stage == "pool":
        note("pool", build_pool.remote(parts, nq, json.dumps(o)))
    if all_stages or stage == "rerank":
        nb = o["rerank_batches"]
        res = list(rerank_batch.map([parts] * nb, [nq] * nb, range(nb), [nb] * nb, [v] * nb))
        done = [r for r in res if not r.get("skipped")]
        note("rerank", {"batches": nb, "ran_now": len(done), "pairs": sum(r["pairs"] for r in done),
                        "gpu_secs": sum(r["secs"] for r in done)})
    if all_stages or stage == "final":
        note("final", finalize.remote(parts, nq, json.dumps(o)))
    return log


@app.local_entrypoint()
def main(stage: str = "all", parts: int = 0, nq: int = 0, opts: str = "{}", download: bool = True) -> None:
    # --opts takes inline JSON or a path to a .json file (easier than quoting JSON in PowerShell).
    user = json.loads(Path(opts).read_text(encoding="utf-8") if opts.endswith(".json") else opts)
    o = {**DEFAULT_OPTS, **user}
    if nq:
        o["nq"] = nq
    if o["nq"] and "rerank_batches" not in user:
        o["rerank_batches"] = max(1, min(DEFAULT_OPTS["rerank_batches"], o["nq"] // 10))
    for line in pipeline.remote(stage, parts, json.dumps(o)):
        print(line)
    if download and stage in ("all", "final"):
        remote = _run_dir(parts, o["nq"], o["version"]).relative_to(VOL).as_posix()  # runs/<version>/<tag>
        dest = Path("artifacts") / ("modal_" + remote)
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("submission.zip", "selection.json", "scored_pool.parquet"):
            with open(dest / name, "wb") as f:
                for chunk in vol.read_file(f"{remote}/{name}"):
                    f.write(chunk)
            print(f"downloaded {dest / name}")
