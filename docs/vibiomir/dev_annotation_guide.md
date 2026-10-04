# ViBioMIR Development Set Annotation and Error Analysis Guide

This guide establishes the official annotation protocol, data schema, quality standards, and provenance-based error analysis workflows for the ViBioMIR evaluation and development sets.

---

## 1. Objectives & Principles

In the ViBioMIR benchmark, ground truth relevance labels for the official 1,200 test queries are withheld by the organizers. To systematically develop, debug, and tune the retrieval pipeline, we curate a high-fidelity internal development set (`data/dev/`).

### Core Principles
1. **No Synthetic Shortcuts**: Queries generated solely by automated prompting from extracted articles do not reflect real clinical information-seeking behavior. Development queries must reflect authentic clinical questions, symptom inquiries, and medical guidelines.
2. **Strict Provenance**: Every query, document judgment, and reference passage chunk must trace back to its origin, annotator/reviewer, and source character span.
3. **Conservative Evaluation**: Unjudged documents in a pool are **never** assumed to be negative. Missing labels must be tracked via `judgment_coverage`.

---

## 2. Namespace Separation & Anti-Leakage

To prevent catastrophic evaluation leakage and collisions between internal development benchmarks and official competition sets:

### Namespace Allocation
- **Official Query IDs**: Reserved exclusively for competition test queries ($1 \dots 1{,}200$).
- **Development Query IDs**: Must reside in a distinct, non-overlapping integer range starting at `9001` (e.g., $9001, 9002, \dots$).
- **Enforcement**: `load_dev_records()` rejects any query ID present in the official query ID set with a `ValueError`.

### Split Protection
- Development records must declare `split: "dev"` (or `"fit"` / `"holdout"`).
- Any record labeled with forbidden splits (`"test"`, `"public"`, `"private"`, `"official_test"`, `"test_set"`) is rejected immediately by the loader to prevent accidental ingestion of submission artifacts as training data.

---

## 3. Data Schema & Metadata Contract

Development qrels are stored as JSONL lines adhering to `EvaluationRecord` and `DevQueryMetadata`:

```json
{
  "query_id": 9001,
  "relevant_doc_ids": [1042, 5819],
  "relevant_chunks": [
    {
      "doc_id": 1042,
      "chunk_text": "Triệu chứng suy tim sung huyết thường gặp bao gồm khó thở khi gắng sức...",
      "char_start": 340,
      "char_end": 425
    }
  ],
  "metadata": {
    "query_id": 9001,
    "query_text": "Dấu hiệu nhận biết và chẩn đoán suy tim sung huyết?",
    "origin": "clinical_case_hospital_k",
    "source_doc_ids": [1042, 5819],
    "topic": "cardiology",
    "question_style": "symptom_diagnosis",
    "split": "dev",
    "pool_version": "pool_v1",
    "judgment_coverage": 0.85
  }
}
```

### Field Definitions

| Field | Type | Description |
|---|---|---|
| `query_id` | `int` | Integer ID strictly $\ge 9001$. |
| `query_text` | `str` | Full natural query string in Vietnamese or English. |
| `origin` | `str` | Provenance source identifier (annotator ID, hospital clinical log, guideline QA). |
| `source_doc_ids` | `tuple[int, ...]` | List of document IDs consulted during query creation. |
| `topic` | `str` | Medical specialty (e.g., `cardiology`, `oncology`, `pharmacology`, `pediatrics`, `infectious_disease`). |
| `question_style` | `str` | Structural query type (`factoid`, `symptom_diagnosis`, `treatment_guideline`, `drug_interaction`, `complex_clinical`). |
| `split` | `str` | Sub-split name (`dev`, `fit`, or `holdout`). Never `test`. |
| `pool_version` | `str` | Identifier of retriever pool from which judgments were elicited. |
| `judgment_coverage` | `float` | Fraction of candidate pool documents that were reviewed ($0.0 \dots 1.0$). |

---

## 4. Multi-Retriever Pooling Protocol

Evaluating retrieval solely against documents retrieved by a single system produces massive systemic bias: documents found by that system are judged, while documents only found by other architectures are unjudged and penalized.

### Pooling Procedure
1. **Candidate Pool Assembly**:
   - For each query, execute first-stage retrieval with multiple heterogeneous systems:
     - Sparse: BM25S (lexical match on titles + bodies)
     - Dense: Multilingual-E5-large (bi-encoder semantic embedding)
     - Fusion: Weighted RRF candidate union ($K = 20$ from each retriever).
2. **De-biasing & Shuffling**:
   - Merge candidates into a pool (typically 20–40 unique document IDs).
   - Randomize presentation order before presenting to annotators to eliminate rank bias.
3. **Tri-State Relevance Judgments**:
   - **Judged Positive**: Document contains direct, verifiable answers or clinical guidance relevant to the query.
   - **Judged Negative**: Document was examined and determined to be non-relevant.
   - **Unjudged**: Documents in the corpus outside the pool, or pool documents not inspected due to time budget.
4. **Golden Rule for Unjudged Documents**:
   > [!IMPORTANT]
   > **Never convert unjudged into negative by default.**
   > Incomplete pools undercount relevant documents. Treating unjudged as negative distorts cutoff calibration and under-estimates system recall.

---

## 5. Exact Reference Chunk Span Extraction

ViBioMIR evaluation penalizes rewritten, hallucinated, or synthesized passages. All reference chunks must originate strictly from the verified source document.

### Guidelines for Chunk Annotation:
1. **Exact Verbatim Extraction**:
   - The annotator identifies the exact minimal span in the document's cleaned `source_text` that directly answers or substantiates the query.
   - Copy the exact characters from `ExtractedDocument.source_text`.
   - Record `char_start` and `char_end` character offsets.
2. **No Text Synthesis**:
   - Never paraphrase, summarize, fix typos, or translate the source text.
   - The string must match character-for-character with `source_text[char_start:char_end]`.
3. **Matching Verification**:
   - Ground truth chunks will be evaluated using the ViBioMIR specification:
     - Normalization order: NFKC $\rightarrow$ `html.unescape` $\rightarrow$ lowercase $\rightarrow$ collapse whitespace $\rightarrow$ strip.
     - LCS token ratio $\ge 40\%$ ($\frac{2}{5}$) between candidate and reference chunk under the same `doc_id`.
     - Chunk matching requires exact document ID identity: chunk text matching across differing document IDs is considered a false positive.

---

## 6. Progressive Annotation Rollout & Stratification

Annotation progresses in two strictly sequenced phases:

### Phase A: Pilot Annotation (30 Queries)
- **Goal**: Calibrate guideline clarity, establish Inter-Annotator Agreement (Cohen's $\kappa > 0.80$ on document relevance), and validate loader/pipeline scripts.
- Two annotators independently annotate the 30 queries.
- Discrepancies are adjudicated by a senior medical reviewer.
- Confirm that `build_error_breakdown()` runs cleanly on all 30 pilot traces.

### Phase B: Production Scale (100–200 Queries)
- **Goal**: Build a robust, statistically significant offline evaluation benchmark.
- **Stratification Requirements**:
  1. **Topic Distribution**:
     - $\ge 25\%$ Cardiology & Cardiovascular
     - $\ge 20\%$ Oncology & Hematology
     - $\ge 20\%$ Infectious Diseases & Immunology
     - $\ge 15\%$ Pharmacology & Therapeutics
     - $\ge 20\%$ General Medicine & Pediatrics
  2. **Query Language Distribution**:
     - $\ge 70\%$ Vietnamese (primary focus)
     - $\le 30\%$ English or mixed biomedical nomenclature
  3. **Query Length / Complexity**:
     - Short search keyword style (2–5 words): 30%
     - Question style (6–15 words): 40%
     - Clinical vignette / scenario (16+ words): 30%

---

## 7. Anti-Leakage Group Splitting (Fit vs. Holdout)

When splitting the development set into `fit` (used for tuning cutoff thresholds $K_{\text{docs}}$ and $K_{\text{chunks}}$) and `holdout` (used for final unbiased validation):

1. **Source Group Clustering**:
   - Documents originating from identical domains, URL subdirectories, or near-duplicate source clusters must not be split across `fit` and `holdout`.
   - All queries whose positive documents touch cluster $C_i$ are assigned entirely to either `fit` or `holdout`.
2. **Partition Proportions**:
   - 60% `fit` (parameter calibration)
   - 40% `holdout` (frozen validation)
3. **Validation Lock**:
   - The calibrator must never fit on the holdout split. `CutoffCalibrator` verifies split provenance prior to tuning.

---

## 8. Error Analysis & Failure Breakdown with Provenance

To understand retrieval degradation and prioritize engineering improvements, each query trace is analyzed via `build_error_breakdown()`.

### The Retrieval Drop Funnel

```
               [ Gold Documents in EvaluationRecord ]
                                 |
           Does doc exist in SourceStore snapshot?
                  /                              \
               [NO]                              [YES]
                 |                                 |
         missing_source_docs              Is doc in candidate_doc_ids?
     (Crawl / Ingestion failure)                 /             \
                                              [NO]             [YES]
                                                |                |
                                     missing_candidate_docs    Is doc in reranked_doc_ids?
                                    (1st-stage BM25/E5 drop)     /              \
                                                              [NO]              [YES]
                                                                |                 |
                                                      lost_after_rerank    Is doc in final_doc_ids?
                                                     (Reranker inversion)     /          \
                                                                            [NO]         [YES]
                                                                              |            |
                                                                     lost_after_cutoff   SUCCESS
                                                                     (Cutoff policy)  (Doc Retrieved)
```

### Evidence Funnel (Chunk Level)
- When a gold reference chunk is not matched by any predicted chunk:
  - If its parent `doc_id` was **not** retrieved in `final_doc_ids`: Document retrieval failure cascade.
  - If its parent `doc_id` **was** retrieved in `final_doc_ids`: True evidence extraction failure (`evidence_missing_given_doc`).

### Aggregate Diagnostics
Using `generate_error_breakdown_summary(breakdowns)`:
- High `total_missing_source_docs` $\rightarrow$ Expand crawler queue and crawl coverage.
- High `total_missing_candidate_docs` $\rightarrow$ Increase initial candidate depth $K_{\text{candidate}}$ or enrich sparse index tokens.
- High `total_lost_after_rerank_docs` $\rightarrow$ Investigate cross-encoder context truncation or score scale compression.
- High `total_lost_after_cutoff_docs` $\rightarrow$ Relax cutoff threshold $K_{\text{docs}}$.
- High `total_evidence_missing_given_doc` $\rightarrow$ Adjust chunk splitting boundaries or increase passage extraction budget.
