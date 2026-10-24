# Pipeline v2: align with the official corpus, add PubMed, recover lost pages

Branch `feature/modal-pipeline-v2` (from `feature/modal-pipeline` at `fb9bbae`).

## Why

The organizers' current corpus (`links_corpus_vi_en_zh.parquet` on Hugging Face, 1,220,598 links) is not the
4,394,718-link file the crawl used. Measured on 2026-10-25:

| | v1 bundle | inside the official corpus |
|---|---|---|
| documents | 2,658,669 | 676,190 (vi 530,776, zh 145,412, en 2) |
| chunks | 6,674,146 | 2,659,416 |

- 75% of indexed documents are outside the official corpus: every one predicted is a guaranteed miss, and they
  are 60% of the embedding bill.
- The 240,000 English PubMed links (20% of the corpus) were never fetched.
- 75,837 official pages were crawled but extracted as empty text (ask.39.net 32,100, laodong.vn 31,265);
  their raw HTML is on disk.

Official coverage today is 55.4%. v2 targets about 80%, with no off-corpus documents.

## Hard limits

- Edit only `modal_pipeline/**`. Never touch `src/**`, configs, the snapshot, raw payloads, or the user's two WIP files.
- Network: only NCBI E-utilities for PubMed, at most 3 requests/s, no API key, no email sent; tool name `vibiomir_retrieval`.
  No re-crawl of robots-blocked sites.
- No Modal GPU spend (the account has no payment method). New data goes to a new volume `vibiomir-v2`; nothing on
  the old volume is deleted.
- Every task ends with its gate; `pytest modal_pipeline/tests -q` stays green after every commit.

## Tasks and gates

### V1. Official-corpus filter
`prepare_upload.py --official-corpus PATH` drops chunks whose `doc_id` is not in the official id set and records
`docs_out_official`, `official_total` in the manifest.
Gate: unit test on a 3-doc fixture (one off-corpus doc dropped); real run reports 676,190 docs / 2,659,416 chunks
(± chunks added by V3).

### V2. PubMed fetch
`fetch_pubmed.py`: reads PMIDs from the official corpus, POSTs `efetch` in batches of 200, writes resumable
`batch_XXXXX.jsonl` (doc_id, pmid, title, sections[(label, text)]). Parser handles labeled sections, inline markup,
missing abstracts and `PubmedBookArticle`.
Gate: parser tests on an inline XML fixture; live smoke of 400 PMIDs returns >= 95% records with a title; full run
covers >= 95% of the 240,000 PMIDs.

### V3. Recover empty-text pages
Re-extract official `quarantined:empty_text` pages from raw HTML with a fallback extractor (trafilatura with
recall settings, then lxml main-content text), keep pages with >= 200 characters of text, chunk them like the snapshot.
Gate: on a 200-page sample per top domain, >= 50% recovered, and a manual read of 5 samples shows real article text.
(Skipped and recorded if raw HTML is not available.)

### V4. Build extra parts
`build_extra_parts.py` turns PubMed records and recovered pages into parts with the bundle schema
(doc_id, chunk_index, language, section, text, title), named `part_9xxxx.parquet` so they sort after the main parts.
Every chunk text is an exact substring of the document text; documents are contiguous.
Gate: tests for substring/contiguity/chunk size; `build_meta` harness accepts the combined bundle.

### V5. Pipeline integration
Volume name `vibiomir-v2` shared by `app.py` and `upload.py`; `finalize` rejects any doc id outside the official
corpus when `data/official_ids.npy` exists; docs updated.
Gate: pytest green incl. a test that an off-corpus doc in the pool never reaches `results.json`.

### V6. Bundle, upload, hand-off
Build the v2 bundle, upload it to `vibiomir-v2` (ingress is free), write `RUN_STATUS_V2.md` with measured coverage
before/after and the morning commands; commit, push, whole-branch review.
Gate: remote sizes equal local; sha on remote equals local.
