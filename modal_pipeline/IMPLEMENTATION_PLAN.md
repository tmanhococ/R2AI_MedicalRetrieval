# Modal Tiny Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove `modal_pipeline/` end to end on Modal with 1 part and 30 queries, then hand off one-command instructions for the full run.

**Architecture:** T1–T4 of `PLAN.md` are done (commit `16f85b0`, pushed to `feature/modal-pipeline`, 6/6 local tests pass). This plan covers T5 (tiny cloud run) and T6 (hand-off). Each task ends with a gate that is a command plus the exact output that means pass.

**Tech Stack:** Modal SDK 1.6.2 (installed in `.venv`), Python 3.12, pytest, git. Auth: `~/.modal.toml`, profile `mle29022` (no `.env`).

**Spec:** `modal_pipeline/PLAN.md`

## Global Constraints

- Work in `D:\Research\R2AI\mvp-rag-medical`, on branch `feature/modal-pipeline` only.
- Edit only `modal_pipeline/**`. Never edit `src/**`, `configs/**`, `tests/**`, `artifacts/modal_upload/**`, or the snapshot.
- Never stage `src/cli/index_sparse.py`, `src/indexing/lucene.py` (user WIP), or anything under `artifacts/`.
- Modal spend cap: **1.00 USD total** (all of the account's credit). Allowed runs: only `--parts 1 --nq 30`. No `--stage bench`, no full run, no GPU other than the `["L4", "A10"]` in `app.py`.
- Upload only `query.parquet`, `upload_manifest.json`, `chunks/part_00000.parquet`.
- Volume deletes are allowed only under `runs/p1_q30/` and `bm25/p1/`, and only to rerun a stage after a code fix. Never touch `data/`, `emb/` or `queries/`.
- Each distinct failure gets at most 2 fix-and-retry cycles; then stop and record it in `RUN_STATUS.md`.
- If Modal refuses GPUs (no payment method, or credit exhausted): stop all cloud work and record it. Do not work around it.
- Every Modal CLI call runs with `PYTHONUTF8=1` (Windows console + Vietnamese text).
- Wall-clock cap for one tiny run: 45 minutes. If it is exceeded, run `.venv/Scripts/modal.exe app stop vibiomir` and record it.

## Review Focus

1. **GPU refused / billing error** on the first GPU stage. Expected: the run stops, nothing is retried, `RUN_STATUS.md` names the error. (Task 3, Step 4)
2. **Image build failure** (Adoptium JDK URL, pyserini/torch resolution, pyjnius, model download). Expected: fix in `app.py` image definitions only, rebuild, retry ≤ 2. (Task 3, Step 4)
3. **Pyserini API mismatch** (`JWhiteSpaceAnalyzer`, `batch_search`, `-pretokenized` flag). Expected: fix in `build_bm25` / `bm25_search` only, delete `bm25/p1/` or `runs/p1_q30/bm25_*.parquet`, rerun from `--stage bm25` or `--stage search`. (Task 3, Step 4)
4. **Stale partial output after a fix.** Stages skip outputs that already exist, so a fixed stage must have its old output removed or the fix is silently not exercised. Expected: the retry log shows the fixed stage ran (no `"skipped": true`). (Task 3, Step 5 gate)
5. **Output that validates but means nothing** (e.g. every query returns the same doc, or pool text misaligned with rows). Expected: the quality smoke check in Task 3 Step 6 fails loudly. (Task 3, Step 6)

---

### Task 1: Preflight (no spend)

**Files:**
- Modify: `modal_pipeline/PLAN.md` (T5 preconditions: `.env` → `~/.modal.toml` profile `mle29022`)

**Interfaces:**
- Consumes: nothing.
- Produces: a verified Modal auth and a clean branch for Tasks 2–4.

- [ ] **Step 1: Check the branch and working tree**

Run: `git branch --show-current; git status --short`
Expected: `feature/modal-pipeline`; status shows only ` M src/cli/index_sparse.py` and ` M src/indexing/lucene.py`.

- [ ] **Step 2: Verify Modal auth**

Run: `PYTHONUTF8=1 .venv/Scripts/modal.exe profile current`
Expected: `mle29022`.
Run: `.venv/Scripts/python.exe -c "import modal; v=modal.Volume.from_name('vibiomir', create_if_missing=True); print('ok', len(v.listdir('/')))"`
Expected: a line starting with `ok`.

- [ ] **Step 3: Re-run the local gate**

Run: `.venv/Scripts/python.exe -m pytest modal_pipeline/tests -q -p no:cacheprovider`
Expected: `6 passed`.

- [ ] **Step 4: Update `PLAN.md` T5 preconditions to the profile-based auth, then commit**

```bash
git add modal_pipeline/PLAN.md modal_pipeline/IMPLEMENTATION_PLAN.md
git commit -m "docs: switch Modal auth precondition to profile and add tiny-run implementation plan"
```
Gate: `git show --stat HEAD` lists only `modal_pipeline/` files.

### Task 2: Upload the tiny subset (no credit used; ingress is free)

**Files:** none changed.

**Interfaces:**
- Consumes: `modal_pipeline/upload.py` CLI: `upload.py SRC --only PATTERN...`.
- Produces: volume `vibiomir` with `/data/query.parquet`, `/data/upload_manifest.json`, `/data/chunks/part_00000.parquet`.

- [ ] **Step 1: Upload**

Run: `PYTHONUTF8=1 .venv/Scripts/python.exe modal_pipeline/upload.py artifacts/modal_upload --only query.parquet upload_manifest.json chunks/part_00000.parquet`
Expected: ends with `Upload complete.`

- [ ] **Step 2: Gate: remote sizes equal local sizes**

Run: `.venv/Scripts/python.exe -c "import modal,os; v=modal.Volume.from_name('vibiomir'); r={e.path.lstrip('/'):e.size for e in v.listdir('/data',recursive=True) if e.type.name=='FILE'}; L={'data/'+p:os.path.getsize('artifacts/modal_upload/'+p) for p in ['query.parquet','upload_manifest.json','chunks/part_00000.parquet']}; print('PASS' if all(r.get(k)==s for k,s in L.items()) and len([k for k in r if k.startswith('data/chunks/')])==1 else ('FAIL', r))"`
Expected: `PASS`.

### Task 3: Tiny end-to-end run on Modal

**Files:**
- Modify (only if a stage fails): `modal_pipeline/app.py`, `modal_pipeline/vb_common.py`
- Test (only for CPU-logic fixes): `modal_pipeline/tests/test_local.py`

**Interfaces:**
- Consumes: `modal run modal_pipeline/app.py --stage {all|meta|encode|bm25|translate|search|pool|rerank|final} --parts 1 --nq 30`. Run tag: `p1_q30`; index tag: `p1`.
- Produces: `artifacts/modal_runs/p1_q30/{submission.zip, selection.json, scored_pool.parquet}`; per-stage log lines used by Task 4.

- [ ] **Step 1: Launch in the background with a log**

Run (background, timeout 45 min): `mkdir -p artifacts/modal_runs/p1_q30 && PYTHONUTF8=1 .venv/Scripts/modal.exe run modal_pipeline/app.py --stage all --parts 1 --nq 30 > artifacts/modal_runs/p1_q30/run.log 2>&1; echo EXIT=$?`

- [ ] **Step 2: Watch for terminal states**

Monitor `artifacts/modal_runs/p1_q30/run.log` with a filter that matches every stage line and every failure: `\] (meta|encode|bm25 index|translate|dense search|bm25 search|pool|rerank|final)|Traceback|Error|error|FAILED|Killed|OOM|payment|billing|quota|EXIT=`.

- [ ] **Step 3: Gate on success**

Expected in `run.log`: one line each for `meta`, `encode`, `bm25 index`, `translate`, `dense search`, `bm25 search`, `pool`, `rerank`, `final`; the `final` line contains `"queries": 30`; `EXIT=0`; and `downloaded artifacts\modal_runs\p1_q30\submission.zip`.

- [ ] **Step 4: On failure: diagnose, fix, retry (max 2 per distinct failure)**

Use superpowers:systematic-debugging. Classify the failure:
- GPU or billing refused → stop Task 3 and go to Task 4 (record it).
- Image build error → fix the image definition in `app.py`.
- Stage code error → fix the stage. If the bug is in CPU logic (meta, tokenize, pool, finalize, vb_common), first add a failing test to `test_local.py` reproducing it, then fix, then `pytest modal_pipeline/tests -q` must pass.

Commit each fix: `git commit -m "fix(modal): <what>"`. Then rerun only from the failed stage: delete that stage's output under `runs/p1_q30/` or `bm25/p1/` with `modal volume rm -r vibiomir <path>` if it exists, and run `--stage <stage>`. Follow with `--stage pool`, `--stage rerank`, `--stage final` as needed. Before every retry, add up the GPU/CPU seconds already reported and stop if the estimated spend is ≥ 0.80 USD.

- [ ] **Step 5: Gate: the stage that was fixed actually ran**

Expected: the retry's log line for the fixed stage has no `"skipped": true`.

- [ ] **Step 6: Gate: submission validity + quality smoke check**

Run:
```
.venv/Scripts/python.exe -c "
import sys,json,zipfile; sys.path.insert(0,'modal_pipeline'); import vb_common as vc, pyarrow.parquet as pq
q=sorted(zip(*[pq.read_table('artifacts/modal_upload/query.parquet')[c].to_pylist() for c in ('id','query')]))[:30]
z=zipfile.ZipFile('artifacts/modal_runs/p1_q30/submission.zip'); assert z.namelist()==['results.json']
res=json.loads(z.read('results.json')); err=vc.validate_results(res,[i for i,_ in q])
hit=sum(1 for r,(i,t) in zip(res,q) if r['relevant_chunks'] and len(set(vc.bm25_tokens(t))&set(vc.bm25_tokens(r['relevant_chunks'][0]['chunk_text'])))>=2)
top=len({r['relevant_docs'][0] for r in res if r['relevant_docs']})
print('errors',err,'lexical_hits',hit,'distinct_top_docs',top); print('PASS' if not err and hit>=15 and top>=10 else 'FAIL')"
```
Expected: `errors [] …` and `PASS`. Here `lexical_hits` counts queries whose top chunk shares ≥ 2 tokens with the query, and `distinct_top_docs` counts distinct top-1 docs across the 30 queries.

### Task 4: Hand-off (`RUN_STATUS.md`) and push

**Files:**
- Create: `modal_pipeline/RUN_STATUS.md`

**Interfaces:**
- Consumes: `run.log` from Task 3 and the gate outputs of Tasks 1–3.
- Produces: the morning instructions.

- [ ] **Step 1: Write `RUN_STATUS.md` (Vietnamese) with exactly these sections**

1. **Kết quả**: PASS or FAIL per task, each with its gate evidence line.
2. **Thời gian và chi phí**: per-stage wall/GPU seconds from `run.log`, and the estimated USD (L4 0.80/h plus 0.19/h overhead; CPU 0.047/core-h; RAM 0.008/GiB-h).
3. **Lỗi đã gặp và cách sửa**: commit sha for each fix.
4. **Sáng mai chạy gì**, as copy-paste commands: add credit or a card if needed; `upload.py artifacts/modal_upload` (rest of the parts); `--stage bench`; `modal run --detach modal_pipeline/app.py --stage all`; how to download; `select_export.py` variants.
5. **Lưu ý**: before the full run, `build_meta` reruns automatically, and the embedding for part 0 is reused.

- [ ] **Step 2: Commit and push**

```bash
git add modal_pipeline/RUN_STATUS.md
git commit -m "docs: record tiny Modal run results and morning runbook"
GIT_TERMINAL_PROMPT=0 git push origin feature/modal-pipeline
```

- [ ] **Step 3: Gate**

Run: `git rev-parse HEAD; git ls-remote origin refs/heads/feature/modal-pipeline; git status --short`
Expected: both shas are equal; status shows only the two user WIP files as ` M`.
