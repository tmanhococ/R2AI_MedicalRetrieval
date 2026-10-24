"""Shared helpers for the Modal pipeline and the local re-selection script.

Pure Python + pyarrow; no torch / Java imports at module level.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import io
import json
import re
import unicodedata
import zipfile

_CJK_RUN = re.compile(r"[㐀-䶿一-鿿豈-﫿]+")
MAX_QUERY_TERMS = 1000  # Lucene BooleanQuery default max clause count is 1024


def compose_passage(title: str | None, section: str | None, text: str) -> str:
    """Same layout as the snapshot's index_text: '<title> - <section>\\n\\n<text>'."""
    head = " - ".join(x.strip() for x in (title, section) if x and x.strip())
    return f"{head}\n\n{text}" if head else text


_lexical = None


def _tokenize_lexical(text: str) -> list[str]:
    global _lexical
    if _lexical is None:
        try:
            import lexical_tokens as _lexical  # copied into the Modal image
        except ImportError:
            # Repo checkout: load the file directly; importing src.indexing pulls in faiss/bm25s.
            import importlib.util
            from pathlib import Path

            path = Path(__file__).resolve().parent.parent / "src" / "indexing" / "lexical_tokens.py"
            spec = importlib.util.spec_from_file_location("lexical_tokens", path)
            _lexical = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_lexical)
    return _lexical.tokenize_lexical(text, language="und")


def bm25_tokens(text: str | None) -> list[str]:
    """Repo lexical tokenizer for non-CJK text, overlapping bigrams for CJK runs.

    The repo tokenizer emits single CJK characters; bigrams are the standard
    Lucene CJK treatment and are far more selective for Chinese BM25.
    """
    if not text:
        return []
    text = unicodedata.normalize("NFC", text)
    out: list[str] = []
    pos = 0
    for m in _CJK_RUN.finditer(text):
        if m.start() > pos:
            out.extend(_tokenize_lexical(text[pos : m.start()]))
        run = m.group()
        if len(run) == 1:
            out.append(run)
        else:
            out.extend(run[i : i + 2] for i in range(len(run) - 1))
        pos = m.end()
    if pos < len(text):
        out.extend(_tokenize_lexical(text[pos:]))
    return out


def bm25_query_string(text: str | None) -> str:
    """Whitespace-joined tokens with at most MAX_QUERY_TERMS distinct terms."""
    kept: list[str] = []
    distinct: set[str] = set()
    for tok in bm25_tokens(text):
        if tok not in distinct:
            if len(distinct) >= MAX_QUERY_TERMS:
                continue
            distinct.add(tok)
        kept.append(tok)
    return " ".join(kept)


@dataclass
class SelectionParams:
    doc_max: int = 100
    doc_min: int = 20
    doc_min_logit: float = -3.0
    chunk_max: int = 100
    chunk_min: int = 20
    chunk_min_logit: float = -3.0
    chunk_per_doc: int = 2
    chunks_from_selected_docs_only: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def select_for_query(cands: list[dict], p: SelectionParams) -> tuple[list[int], list[dict]]:
    """cands: dicts with doc_id, ce (reranker logit), doc_rrf, text.

    Documents are scored by their best chunk logit. A list keeps every item at or
    above the logit threshold, is topped up to the min count by rank, and is cut
    at the max count. Macro F2 weights recall 4x, and gold sets are large
    (median ~60 documents per query on the public split), so defaults are generous.
    """
    doc_best: dict[int, float] = {}
    doc_rrf: dict[int, float] = {}
    for c in cands:
        d = int(c["doc_id"])
        if c["ce"] > doc_best.get(d, float("-inf")):
            doc_best[d] = c["ce"]
        doc_rrf[d] = max(doc_rrf.get(d, 0.0), c.get("doc_rrf") or 0.0)
    ranked_docs = sorted(doc_best, key=lambda d: (-doc_best[d], -doc_rrf[d], d))
    docs = [d for d in ranked_docs if doc_best[d] >= p.doc_min_logit][: p.doc_max]
    if len(docs) < p.doc_min:
        docs = ranked_docs[: min(p.doc_min, p.doc_max)]
    doc_set = set(docs)

    pool = sorted(cands, key=lambda c: (-c["ce"], int(c["doc_id"])))
    if p.chunks_from_selected_docs_only:
        pool = [c for c in pool if int(c["doc_id"]) in doc_set]
    per_doc: dict[int, int] = {}
    seen_text: set[str] = set()
    ranked: list[dict] = []
    for c in pool:
        d = int(c["doc_id"])
        key = " ".join(c["text"].split())
        if per_doc.get(d, 0) >= p.chunk_per_doc or key in seen_text:
            continue
        per_doc[d] = per_doc.get(d, 0) + 1
        seen_text.add(key)
        ranked.append(c)
    chunks = [c for c in ranked if c["ce"] >= p.chunk_min_logit][: p.chunk_max]
    if len(chunks) < p.chunk_min:
        chunks = ranked[: min(p.chunk_min, p.chunk_max)]
    return docs, chunks


def build_results(query_ids: list[int], scored_rows: list[dict], p: SelectionParams) -> list[dict]:
    """scored_rows: flat rows with qid, doc_id, ce, doc_rrf, text. Every query id gets a row."""
    by_q: dict[int, list[dict]] = {}
    for r in scored_rows:
        by_q.setdefault(int(r["qid"]), []).append(r)
    results = []
    for qid in sorted(query_ids):
        docs, chunks = select_for_query(by_q.get(qid, []), p)
        results.append(
            {
                "id": int(qid),
                "relevant_docs": [int(d) for d in docs],
                "relevant_chunks": [
                    {"doc_id": int(c["doc_id"]), "chunk_text": c["text"], "chunk_order": i}
                    for i, c in enumerate(chunks)
                ],
            }
        )
    return results


def validate_results(results: list[dict], query_ids: list[int]) -> list[str]:
    errors = []
    ids = [r["id"] for r in results]
    if sorted(ids) != sorted(set(int(q) for q in query_ids)) or len(ids) != len(set(ids)):
        errors.append(f"query ids mismatch: got {len(ids)} rows, expected {len(set(query_ids))} unique ids")
    for r in results:
        if not all(isinstance(d, int) and not isinstance(d, bool) for d in r["relevant_docs"]):
            errors.append(f"query {r['id']}: non-integer doc id")
        if len(set(r["relevant_docs"])) != len(r["relevant_docs"]):
            errors.append(f"query {r['id']}: duplicate doc ids")
        for c in r["relevant_chunks"]:
            if not isinstance(c["doc_id"], int) or not c["chunk_text"].strip():
                errors.append(f"query {r['id']}: bad chunk entry")
                break
    return errors


def results_to_zip_bytes(results: list[dict]) -> tuple[bytes, bytes]:
    """Returns (results.json bytes, deterministic zip bytes with results.json at the root)."""
    js = (json.dumps(results, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    buf = io.BytesIO()
    info = zipfile.ZipInfo("results.json", date_time=(1980, 1, 1, 0, 0, 0))
    info.external_attr = 0o644 << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr(info, js)
    return js, buf.getvalue()


def quality_report(results: list[dict], scored_rows: list[dict]) -> dict:
    """Sanity signals that a validated submission is not degenerate.

    Passes when queries are not all answered by the same document and the median
    query's best reranker logit is positive (bge-reranker logits > 0 mean "relevant").
    """
    best: dict[int, float] = {}
    for r in scored_rows:
        q = int(r["qid"])
        best[q] = max(best.get(q, float("-inf")), float(r["ce"]))
    tops = sorted(best.values())
    median_top = tops[len(tops) // 2] if tops else float("-inf")
    n = len(results)
    distinct = len({r["relevant_docs"][0] for r in results if r["relevant_docs"]})
    report = {
        "queries": n,
        "empty_queries": sum(1 for r in results if not r["relevant_docs"]),
        "distinct_top_docs": distinct,
        "median_top_ce": round(median_top, 3),
    }
    report["passed"] = n > 0 and distinct >= min(10, max(1, n // 3)) and median_top > 0
    return report
