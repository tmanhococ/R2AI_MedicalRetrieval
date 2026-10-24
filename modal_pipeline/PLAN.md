# Modal pipeline: local hardening + tiny cloud validation (overnight 2026-10-24)

Goal: by morning the user only has to run the full pipeline on Modal. Tonight: harden `modal_pipeline/`,
prove it locally on real data, commit + push on a new branch, and prove it end to end on Modal
with 1 part and 30 queries.

## Hard limits (do not cross)

- Files the agent may create/modify: `modal_pipeline/**` (this plan is `modal_pipeline/PLAN.md`).
  Never edit `src/**`, `configs/**`, `tests/**`, the snapshot, or `artifacts/modal_upload/**`.
- Never stage or commit `src/cli/index_sparse.py` and `src/indexing/lucene.py` (user WIP), `.env`, or anything under `artifacts/`.
- Modal spend tonight: **at most 1 USD** (that is all the credit on the account). Only `--parts 1 --nq 30` runs.
  No `--stage bench`, no full run, no H100/A100. Upload only `chunks/part_00000.parquet`, `query.parquet`, `upload_manifest.json`.
- Max 2 retries per failing Modal command; after that stop and record the failure in `RUN_STATUS.md`.
- If Modal refuses GPUs (no payment method), stop the cloud part, record it, do not work around it.
- Never print the token values; load them from `.env` into the environment only.
- Never delete anything on the Modal volume.

## Tasks and gates

### T1. Code changes (local, scope = `modal_pipeline/`)
1. `translate_queries` translates only the first `nq` queries when `nq > 0` and merges into an existing
   `variants.jsonl` (translate only missing ids), so the tiny run does not pay for 1,200 translations
   and the full run reuses them.
2. `upload.py` gets `--only` (glob list) to upload a subset.
3. `modal_pipeline/tests/test_local.py` (pytest): tokenizer/selection/export unit tests plus a CPU-stage
   harness (meta -> tokenize part -> fake search -> pool -> fake rerank -> finalize) on the first
   5,000 rows of `artifacts/modal_upload/chunks/part_00000.parquet`, with the Modal volume replaced by a temp dir.

Gate T1: `pyflakes modal_pipeline` prints nothing; `python -c "import app"` with the modal SDK succeeds and lists 10 functions.

### T2. Local tiny test
Gate T2: `pytest modal_pipeline/tests -q` passes. The harness must assert: meta row count == rows read;
pool <= pool_size per query and every pool row's doc_id == chunk_doc[row]; pool text == parquet text for that row;
finalize returns `queries == nq`; results validate with `vb_common.validate_results`; zip contains only `results.json`.

### T3. Commit on `feature/modal-pipeline`
Gate T3: `git show --stat HEAD` lists only `modal_pipeline/**`; `git status --short` still shows the
two user WIP files as ` M` (unstaged, untouched).

### T4. Push
Gate T4: `git ls-remote origin refs/heads/feature/modal-pipeline` prints the same sha as `git rev-parse HEAD`.
If push needs interactive credentials, skip and record it (the commit stays local).

### T5. Tiny run on Modal
Preconditions: `.env` exists with MODAL_TOKEN_ID / MODAL_TOKEN_SECRET; `modal` installed in `.venv`;
`modal profile current` (or a `Volume.from_name(...).listdir` call) succeeds.
Steps: upload subset -> `modal run modal_pipeline/app.py --stage all --parts 1 --nq 30`.
Gate T5: exit code 0; log has lines for meta, encode, bm25 index, translate, dense search, bm25 search, pool, rerank,
final; final reports `queries: 30` and `avg_docs > 0`; `artifacts/modal_runs/p1_q30/submission.zip` exists,
contains only `results.json` with 30 ids that pass `validate_results`.

### T6. Morning hand-off
Write `modal_pipeline/RUN_STATUS.md`: what passed (gate evidence), timings and GPU seconds per stage, anything that
failed, and the exact commands for the morning (upload the remaining parts, bench, full run). Commit + push it.
Gate T6: file exists on the pushed branch.
