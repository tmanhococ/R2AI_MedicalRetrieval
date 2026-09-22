from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

from src.data.dataset_adapter import JsonlDatasetAdapter
from src.data.validators import validate_corpus
from src.indexing.bm25_index import BM25Indexer
from src.indexing.chunker import create_chunker
from src.indexing.dense_index import DenseIndexer
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def run_index(
    config: dict[str, Any],
    dataset_dir: Path | str,
    output_dir: Path | str | None = None,
    encoder: Any = None,
) -> dict[str, Any]:
    """Execute corpus validation, chunking, and dual BM25 + Dense vector indexing.

    Args:
        config: Application configuration dictionary.
        dataset_dir: Path to the dataset directory containing corpus documents/chunks.
        output_dir: Directory where indexes and manifests will be written.
        encoder: Optional pre-instantiated embedding encoder for testing or custom embeddings.

    Returns:
        Dictionary containing index paths and summary statistics.
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    # Determine index destination directory
    if output_dir is not None:
        out_path = Path(output_dir)
    else:
        run_out = config.get("run", {}).get("output_dir", "artifacts")
        out_path = Path(run_out) / "indexes"
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load corpus
    adapter = JsonlDatasetAdapter(dataset_path)
    documents = adapter.load_documents()
    chunks = adapter.load_chunks()

    # 2. Initial validation
    strict_val = config.get("indexing", {}).get("validation", {}).get("strict_foreign_keys", True)
    validate_corpus(documents, chunks, strict=strict_val)

    # 3. Chunking if chunks missing or mode != "prechunked"
    chunk_cfg = config.get("indexing", {}).get("chunking", {})
    mode = chunk_cfg.get("mode", "hybrid")
    if not chunks or mode != "prechunked":
        chunker = create_chunker(chunk_cfg)
        chunks = chunker.chunk_documents(documents, existing_chunks=chunks)
        # Re-validate with new chunks
        validate_corpus(documents, chunks, strict=strict_val)

    # Persist generated chunks to output_dir for downstream stages if not already in dataset
    chunks_out = out_path / "chunks.jsonl"
    with open(chunks_out, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(
                json.dumps(
                    {
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "language": c.language,
                        "text": c.text,
                        "chunk_index": c.chunk_index,
                        "metadata": c.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    retrieval_cfg = config.get("retrieval", {})
    bm25_cfg = retrieval_cfg.get("bm25", {})
    dense_cfg = retrieval_cfg.get("dense", {})

    bm25_output_dir = out_path / "bm25"
    dense_output_dir = out_path / "dense"

    bm25_manifest: dict[str, Any] = {}
    dense_manifest: dict[str, Any] = {}

    # 4. Build BM25 Index
    if bm25_cfg.get("enabled", True):
        bm25_indexer = BM25Indexer(
            backend=bm25_cfg.get("backend", "bm25s"),
            tokenizers_config=bm25_cfg.get("tokenizers"),
        )
        bm25_manifest = bm25_indexer.build_indexes(
            documents=documents,
            chunks=chunks,
            output_dir=bm25_output_dir,
        )

    # 5. Build Dense Index
    if dense_cfg.get("enabled", True):
        dense_encoder = encoder if encoder is not None else dense_cfg.get("encoder") or dense_cfg.get("model_name", "intfloat/multilingual-e5-large")
        dense_indexer = DenseIndexer(
            encoder=dense_encoder,
            backend=dense_cfg.get("backend", "faiss"),
            instruction_prefix=dense_cfg.get("instruction_prefix"),
            device=dense_cfg.get("device", "auto"),
            normalize_embeddings=dense_cfg.get("normalize_embeddings", True),
            batch_size=dense_cfg.get("batch_size", 64),
        )
        dense_manifest = dense_indexer.build_index(
            chunks=chunks,
            output_dir=dense_output_dir,
        )

    return {
        "output_dir": str(out_path),
        "bm25_index_dir": str(bm25_output_dir),
        "dense_index_dir": str(dense_output_dir),
        "num_documents": len(documents),
        "num_chunks": len(chunks),
        "bm25_manifest": bm25_manifest,
        "dense_manifest": dense_manifest,
    }


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Multilingual Medical Retrieval Indexing CLI entrypoint."
    )
    parser.add_argument(
        "-c",
        "--config",
        default="configs/r2ai_mvp.yaml",
        help="Path to YAML configuration file (default: configs/r2ai_mvp.yaml)",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        help="Path to dataset directory containing documents/chunks JSONL files",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Optional directory to write index files",
    )

    parsed = parser.parse_args(args)
    cfg = load_config(parsed.config)
    results = run_index(
        config=cfg,
        dataset_dir=parsed.dataset,
        output_dir=parsed.output_dir,
    )
    print(f"Indexing completed successfully:")
    print(f"  Documents indexed: {results['num_documents']}")
    print(f"  Chunks indexed:    {results['num_chunks']}")
    print(f"  BM25 index dir:    {results['bm25_index_dir']}")
    print(f"  Dense index dir:   {results['dense_index_dir']}")


if __name__ == "__main__":
    main()
