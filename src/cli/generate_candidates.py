from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.query.query_processor import QueryVariants, ProcessedQuery, build_query_variants, AdaptiveQueryRouter
from src.query.translator import create_translator
from src.retrieval.hybrid import HybridCandidateRetriever
from src.indexing.lucene import LuceneSparseBackend
from src.indexing.faiss_index import FaissDenseBackend
from src.data.phase2_contracts import SparseConfig, DenseConfig

logger = logging.getLogger(__name__)

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multilingual hybrid candidates for queries.")
    parser.add_argument("--queries", type=str, required=True, help="Path to queries JSONL/JSON file")
    parser.add_argument("--sparse-index-dir", type=str, required=True, help="Path to Lucene sparse index directory")
    parser.add_argument("--dense-index-dir", type=str, required=True, help="Path to FAISS dense index directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to output directory for candidates")
    parser.add_argument("--config", type=str, default="", help="Path to YAML config file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config if provided
    config = {}
    if args.config and Path(args.config).exists():
        try:
            import yaml
            with open(args.config, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning("Failed to load config file %s: %s", args.config, exc)

    translator = create_translator(config)
    sparse_backend = LuceneSparseBackend(index_dir=args.sparse_index_dir)
    dense_backend = FaissDenseBackend(index_dir=args.dense_index_dir)

    retriever = HybridCandidateRetriever(
        sparse_backend=sparse_backend,
        dense_backend=dense_backend,
    )

    queries_path = Path(args.queries)
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")

    router = AdaptiveQueryRouter()
    translator_client = create_translator(config)
    from src.query.translator import MultilingualQueryBuilder
    qb = MultilingualQueryBuilder(translator=translator_client)

    output_file = output_dir / "candidates.jsonl"
    count = 0

    with open(queries_path, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            item = json.loads(line)
            qid = item.get("query_id", item.get("id", f"q_{count}"))
            text_vi = item.get("text_vi", item.get("text", ""))

            processed = router.process(qid, text_vi)
            variants = qb.build(processed)

            fused_hits, evidence_seeds = retriever.retrieve(variants, top_k=200)

            record = {
                "query_id": qid,
                "variants": {
                    "vi": variants.vi,
                    "vi_condensed": variants.vi_condensed,
                    "en": variants.en,
                    "zh": variants.zh,
                },
                "candidates": [
                    {
                        "doc_id": h.doc_id,
                        "score": h.fusion_score,
                        "rank": h.rank,
                        "stream_ranks": h.stream_ranks,
                    }
                    for h in fused_hits
                ]
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    logger.info("Successfully generated candidates for %d queries to %s", count, output_file)

if __name__ == "__main__":
    main()
