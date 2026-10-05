from __future__ import annotations

import argparse
from pathlib import Path
import json
import logging

from src.data.phase2_contracts import DenseConfig
from src.indexing.dense_encoder import DenseEncoder
from src.indexing.embedding_store import write_embedding_blocks

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Encode corpus chunks with multilingual E5 into float16 disk blocks.")
    parser.add_argument("--chunks-path", type=str, required=True, help="Path to JSON/JSONL chunks file.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for embedding blocks.")
    parser.add_argument("--snapshot-id", type=str, default="snap_default", help="Snapshot ID.")
    parser.add_argument("--block-rows", type=int, default=10000, help="Number of rows per embedding block.")
    parser.add_argument("--model-name", type=str, default="intfloat/multilingual-e5-base", help="E5 model name.")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume, overwrite existing blocks.")
    parser.add_argument("--use-mock", action="store_true", help="Use mock encoder for testing.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    chunks_path = Path(args.chunks_path)
    output_dir = Path(args.output_dir)

    logger.info(f"Loading chunks from {chunks_path}")
    chunks = []
    if chunks_path.suffix == ".jsonl":
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
    else:
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

    config = DenseConfig(
        model_name=args.model_name,
        block_rows=args.block_rows,
    )

    if args.use_mock:
        encoder = DenseEncoder.get_mock_encoder(dimension=config.dimension)
    else:
        encoder = DenseEncoder(
            model_name=config.model_name,
            dimension=config.dimension,
            query_prefix=config.query_prefix,
            passage_prefix=config.passage_prefix,
            normalize=config.normalize,
        )

    logger.info(f"Writing embedding blocks for {len(chunks)} chunks to {output_dir}")
    blocks = write_embedding_blocks(
        chunks=chunks,
        encoder=encoder,
        output=output_dir,
        config=config,
        snapshot_id=args.snapshot_id,
        resume=not args.no_resume,
    )

    logger.info(f"Successfully wrote {len(blocks)} embedding blocks.")

if __name__ == "__main__":
    main()
