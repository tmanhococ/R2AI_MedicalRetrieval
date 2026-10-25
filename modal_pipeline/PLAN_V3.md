# v3: closing the gap to the top of the leaderboard

Baseline (v2 full run, public leaderboard):

| | FINAL | DOCS F2 | CHUNKS F2 | DOCS P | DOCS R | CHUNKS P | CHUNKS R |
|---|---|---|---|---|---|---|---|
| v2 | 0.1296 | 0.1965 | 0.0628 | 0.1645 | 0.2433 | 0.0538 | 0.0774 |
| top 3 | 0.4083 | 0.4082 | 0.4083 | 0.2443 | 0.5522 | 0.2454 | 0.5505 |

## Diagnosis

1. **Chunks lose two thirds of the doc hits.** For the top teams chunk P/R equals doc P/R, so every correct
   document also yields a matching chunk. Ours: doc P 0.165 but chunk P 0.054. A predicted chunk counts when
   token-LCS(pred, gold) / len(gold) >= 0.4, and long predictions are not penalised. Our chunks are ~300-word
   windows (2 per doc); a gold chunk elsewhere in the page, or longer than our window, fails the 40% bar.
   Hypothesis: one prediction per document carrying the whole document text matches whenever the document is
   right, which is exactly the top teams' signature.
2. **Doc recall 0.24 vs 0.55.** With precision 0.24 and ~60-70 gold docs per query, the top teams output roughly
   150 documents per query; we output 85 (cap 100) and only rerank ~150 candidate documents per query.
3. **Coverage ceiling.** Only 75.1% of the official corpus has text (missing: 160k robots-blocked, 76k behind
   anti-bot pages, 61k fetch failures). If gold is spread like the corpus, recall cannot exceed ~0.75.
4. Not a major cause: the boilerplate filter. Long texts (>= 600 chars) repeated 5-49 times are ~44k rows,
   under 2% of the 2.67M kept chunks.
5. Smaller issues seen in outputs: NLLB mistranslates some idioms ("nổi nóng" -> "发热"), pulling in
   off-topic Chinese pages.

## Experiments, cheapest first (each gated by one leaderboard submission)

| # | Change | Where / cost | Expected effect |
|---|---|---|---|
| E1 | One chunk per predicted doc = full document text (from the local bundle) | local, 0 USD | CHUNKS F2 -> ~DOCS F2; FINAL ~0.13 -> ~0.20 |
| E2 | Output size sweep: 100 / 150 / 200 docs per query; doc score = max reranker logit, then fusion rank | local, 0 USD (within the scored pool) | DOCS R up; pick the best size |
| E3 | Wider candidate set: top_docs 300, pool 800 pairs/query, rerank on doc-level passages (title + best chunks) | Modal, ~1.5-2 USD | more correct docs reach the reranker |
| E4 | Stronger dense retriever (BAAI/bge-m3 dense, or multilingual-e5-large), version "v2" | Modal, ~4-6 USD | better vi->zh/en and long-query recall |
| E5 | Better query translation (LLM or larger MT) + Vietnamese query condensation for long questions | Modal or API, ~1 USD | fewer cross-lingual misses like the idiom above |
| E6 | Coverage: retry the 61k fetch failures politely; robots/anti-bot pages only if you decide the rules allow it | local network | raises the recall ceiling above 0.75 |

Order: E1 -> E2 (free, today) -> E3 -> E4 -> E5; E6 in parallel if you choose.

## Data needed

The scorer's full result (the zip with `details.jsonl`, like `artifacts/experiments/vibiomir_phase1_A/scoring_result.zip`)
for the v2 submission. It gives per-query gold counts and, per predicted chunk, the best overlap with gold; that
measures diagnosis 1 directly (how often our doc is right but the chunk misses, and by how much) and lets E2 pick
the output size from data instead of guesses.
