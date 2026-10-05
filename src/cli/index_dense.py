from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.data.phase2_contracts import DenseConfig
from src.indexing.embedding_store import VectorStore
from src.indexing.faiss_index import build_faiss_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dense FAISS index from embedding blocks.")
    parser.add_argument("--embedding-dir", required=True, type=Path, help="Directory containing embedding blocks and embedding_manifest.json")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write dense index artifacts and manifest")
    parser.add_argument("--nlist", type=int, default=4096)
    parser.add_argument("--pq-m", type=int, default=64)
    parser.add_argument("--nprobe", type=int, default=32)
    args = parser.parse_args()

    store = VectorStore(args.embedding_dir)
    config = DenseConfig(
        nlist=args.nlist,
        pq_m=args.pq_m,
        nprobe=args.nprobe,
    )

    logger.info("Building FAISS index from embedding dir: %s", args.embedding_dir)
    manifest = build_faiss_index(store.blocks, config, args.output_dir, vector_store=store)
    logger.info("Successfully built FAISS index backend=%s, members=%d at %s", manifest.backend, manifest.member_count, args.output_dir)


if __name__ == "__main__":
    main()
