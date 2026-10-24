"""PubMed records -> bundle parts."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build_extra_parts import pubmed_chunks, write_pubmed_parts  # noqa: E402


def _rec(doc_id, sections, title="A title", missing=False):
    return {"doc_id": doc_id, "pmid": doc_id + 1000, "title": title, "journal": None,
            "sections": sections, **({"missing": True} if missing else {})}


def test_short_abstract_is_one_chunk_with_first_label():
    rows = pubmed_chunks(_rec(7, [["Background", "a b c"], ["Results", "d e"]]))
    assert [r["text"] for r in rows] == ["a b c\n\nd e"]
    assert rows[0] == {"doc_id": 7, "chunk_index": 0, "language": "en", "section": "Background",
                       "text": "a b c\n\nd e", "title": "A title"}


def test_long_abstract_splits_on_section_boundaries_under_word_cap():
    secs = [[f"S{i}", " ".join(["w"] * 120)] for i in range(5)]  # 600 words
    rows = pubmed_chunks(_rec(1, secs), max_words=300)
    assert [r["chunk_index"] for r in rows] == list(range(len(rows)))
    assert all(len(r["text"].split()) <= 300 for r in rows)
    assert sum(len(r["text"].split()) for r in rows) == 600
    assert [r["section"] for r in rows] == ["S0", "S2", "S4"]


def test_single_overlong_section_is_split_by_words():
    rows = pubmed_chunks(_rec(1, [[None, " ".join(f"w{i}" for i in range(700))]]), max_words=300)
    assert [len(r["text"].split()) for r in rows] == [300, 300, 100]


def test_title_only_and_missing_records():
    assert [r["text"] for r in pubmed_chunks(_rec(3, [], title="Only title"))] == ["Only title"]
    assert pubmed_chunks(_rec(4, [], title="", missing=True)) == []


def test_write_parts_sorted_contiguous_and_named_after_main_parts(tmp_path):
    src = tmp_path / "pubmed"
    src.mkdir()
    recs = [_rec(d, [[None, "x y"]]) for d in (30, 10, 20)] + [_rec(40, [], title="", missing=True)]
    (src / "batch_00000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    stats = write_pubmed_parts(src, tmp_path / "chunks", first_index=90000, rows_per_part=2)
    names = sorted(p.name for p in (tmp_path / "chunks").glob("*.parquet"))
    assert names == ["part_90000.parquet", "part_90001.parquet"]
    docs = [d for n in names for d in pq.read_table(tmp_path / "chunks" / n)["doc_id"].to_pylist()]
    assert docs == [10, 20, 30]
    assert stats == {"records": 4, "docs": 3, "chunks": 3, "skipped_missing": 1}
