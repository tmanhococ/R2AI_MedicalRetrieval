from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.data.phase2_contracts import RerankConfig, EvidenceSpan
from src.query.query_processor import QueryVariants
from src.data.source_store import EvidenceSourceStore
from src.reranking.neural import NeuralReranker, score_evidence

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("rerank_candidates_cli")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run neural BGE reranking on candidate evidence spans.")
    parser.add_argument("--candidates", required=True, type=Path, help="Path to input candidates JSON/Parquet file.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Path to output directory.")
    parser.add_argument("--store-dir", required=False, type=Path, help="Path to source store database/directory.")
    parser.add_argument("--config", required=False, type=Path, help="Path to JSON config.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output if present.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for reranking.")
    parser.add_argument("--model-name", type=str, default="BAAI/bge-reranker-v2-m3", help="Model name.")

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting rerank candidates CLI with args: %s", args)

    config = RerankConfig(model_name=args.model_name)
    if args.config and args.config.exists():
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
                config = RerankConfig(**{k: v for k, v in cfg_data.items() if k in RerankConfig.__dataclass_fields__})
        except Exception as e:
            logger.warning("Failed to load config from %s: %s. Using default.", args.config, e)

    reranker = NeuralReranker(model_name=config.model_name, batch_size=args.batch_size, require_real_weights=config.require_real_weights)

    # Load candidates
    candidates_data = []
    if args.candidates.suffix == ".json":
        with open(args.candidates, "r", encoding="utf-8") as f:
            candidates_data = json.load(f)
    else:
        import pyarrow.parquet as pq
        table = pq.read_table(args.candidates)
        candidates_data = table.to_pylist()

    store = None
    if args.store_dir and args.store_dir.exists():
        store = EvidenceSourceStore(args.store_dir)

    # Process each query / record
    output_records = []
    for item in candidates_data:
        q_dict = item.get("query", {"vi": item.get("query_text", "")})
        query = QueryVariants(
            vi=q_dict.get("vi", ""),
            en=q_dict.get("en"),
            zh=q_dict.get("zh"),
        )
        spans_raw = item.get("spans", [])
        spans = [
            EvidenceSpan(
                span_id=s.get("span_id"),
                doc_id=s.get("doc_id"),
                source_version=s.get("source_version", "v1"),
                source_sha256=s.get("source_sha256", ""),
                start_char=s.get("start_char", 0),
                end_char=s.get("end_char", 0),
                seed_chunk_ids=tuple(s.get("seed_chunk_ids", [])),
                variant_kind=s.get("variant_kind", "base"),
                base_span_id=s.get("base_span_id"),
                added_block_span_ids=tuple(s.get("added_block_span_ids", [])),
                score=s.get("score"),
            )
            for s in spans_raw
        ]

        scored_spans = score_evidence(query, spans, reranker=reranker, store=store, config=config, cache_dir=args.output_dir)
        
        # Sort by score desc
        scored_spans.sort(key=lambda x: x.score if x.score is not None else float("-inf"), reverse=True)

        item_out = dict(item)
        item_out["spans"] = [
            {
                "span_id": s.span_id,
                "doc_id": s.doc_id,
                "source_version": s.source_version,
                "source_sha256": s.source_sha256,
                "start_char": s.start_char,
                "end_char": s.end_char,
                "seed_chunk_ids": list(s.seed_chunk_ids),
                "variant_kind": s.variant_kind,
                "base_span_id": s.base_span_id,
                "added_block_span_ids": list(s.added_block_span_ids),
                "score": s.score,
            }
            for s in scored_spans
        ]
        output_records.append(item_out)

    output_path = args.output_dir / "reranked_candidates.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_records, f, ensure_ascii=False, indent=2)

    logger.info("Successfully wrote reranked candidates to %s", output_path)


if __name__ == "__main__":
    main()
