"""Fetch title + abstract for the official corpus's PubMed links from Europe PMC.

NCBI E-utilities currently returns "Access Denied ... temporarily blocked" for this machine's IP, so records come
from Europe PMC's public REST API (EMBL-EBI), which serves the same MEDLINE titles and abstracts.

Resumable: one JSONL file per batch of 250 PMIDs; finished batches are skipped. At most 2 requests/s.

    .venv/Scripts/python.exe modal_pipeline/fetch_pubmed.py \
        --official-corpus data/official/links_corpus_vi_en_zh.parquet --out artifacts/pubmed
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request

SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
USER_AGENT = "vibiomir_retrieval (research; batch abstract fetch)"
BATCH = 250
MIN_INTERVAL = 0.5  # seconds between requests
_PMID_URL = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)")
_H4 = re.compile(r"<h4>(.*?)</h4>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")


def pmid_from_url(url: str) -> int | None:
    m = _PMID_URL.search(url)
    return int(m.group(1)) if m else None


def _clean(text: str | None) -> str:
    return " ".join(html.unescape(_TAG.sub("", text or "")).split())


def epmc_query(pmids: list[int]) -> str:
    return "EXT_ID:(" + " OR ".join(map(str, pmids)) + ") AND SRC:MED"


def parse_epmc_result(r: dict) -> dict:
    """{pmid, title, journal, sections: [[label|None, text], ...]} from one Europe PMC core result."""
    sections: list[list] = []
    abstract = r.get("abstractText") or ""
    parts = _H4.split(abstract)  # [lead, label1, text1, label2, text2, ...]
    if _clean(parts[0]):
        sections.append([None, _clean(parts[0])])
    for label, body in zip(parts[1::2], parts[2::2]):
        if _clean(body):
            sections.append([_clean(label) or None, _clean(body)])
    journal = ((r.get("journalInfo") or {}).get("journal") or {}).get("title")
    return {"pmid": int(r["pmid"]), "title": _clean(r.get("title")), "journal": journal or None, "sections": sections}


def _search(pmids: list[int], retries: int = 6) -> list[dict]:
    params = urllib.parse.urlencode({
        "query": epmc_query(pmids), "resultType": "core", "format": "json", "pageSize": 1000,
    })
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(f"{SEARCH}?{params}", headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data.get("resultList", {}).get("result", [])
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
            wait = min(120, 2 ** attempt)
            print(f"  request failed ({exc}); retry {attempt}/{retries} in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"Europe PMC search failed for PMIDs {pmids[0]}..{pmids[-1]}")


def main() -> None:
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser()
    ap.add_argument("--official-corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("artifacts/pubmed"))
    ap.add_argument("--limit", type=int, default=0, help="only the first N PMIDs (smoke test)")
    args = ap.parse_args()

    t = pq.read_table(args.official_corpus)
    pmid_to_doc = {}
    for doc_id, url in zip(t["id"].to_pylist(), t["url"].to_pylist()):
        pmid = pmid_from_url(url)
        if pmid is not None:
            pmid_to_doc[pmid] = doc_id
    pmids = sorted(pmid_to_doc)[: args.limit or None]
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(pmids):,} PMIDs, {(len(pmids) + BATCH - 1) // BATCH} batches -> {args.out}", flush=True)

    last = 0.0
    t0 = time.time()
    got = missing = 0
    for b in range(0, len(pmids), BATCH):
        path = args.out / f"batch_{b // BATCH:05d}.jsonl"
        if path.exists():
            continue
        ids = pmids[b : b + BATCH]
        time.sleep(max(0.0, MIN_INTERVAL - (time.time() - last)))
        last = time.time()
        recs = {}
        for r in _search(ids):
            if r.get("pmid") and r.get("source") in (None, "MED"):
                rec = parse_epmc_result(r)
                recs[rec["pmid"]] = rec
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for pmid in ids:
                rec = recs.get(pmid)
                if rec is None:
                    missing += 1
                    rec = {"pmid": pmid, "title": "", "journal": None, "sections": [], "missing": True}
                else:
                    got += 1
                rec["doc_id"] = pmid_to_doc[pmid]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp.replace(path)
        n = b // BATCH + 1
        if n % 100 == 0:
            print(f"  [{time.time() - t0:5.0f}s] {n} batches, fetched {got:,}, missing {missing:,}", flush=True)
    print(f"done in {time.time() - t0:.0f}s: fetched {got:,}, missing {missing:,} (this run)", flush=True)


if __name__ == "__main__":
    main()
