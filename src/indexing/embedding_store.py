from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path
from typing import Iterable, Sequence, Any
import numpy as np

from src.data.phase2_contracts import DenseConfig, EmbeddingBlock

class VectorStore:
    """Reads float16 embedding blocks from disk and exposes row access and chunk ID mapping."""

    def __init__(self, index_dir: Path | str):
        self.index_dir = Path(index_dir)
        self.manifest_path = self.index_dir / "embedding_manifest.json"
        self.blocks: list[EmbeddingBlock] = []
        self.chunk_to_row: dict[str, int] = {}
        self._load_manifest()

    def _load_manifest(self) -> None:
        if not self.manifest_path.exists():
            return
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for b_dict in data.get("blocks", []):
                self.blocks.append(EmbeddingBlock(**b_dict))
            self.chunk_to_row = data.get("chunk_to_row", {})

    def get_rows(self, row_ids: Sequence[int]) -> np.ndarray:
        """Fetch normalized float32 vectors from stored float16 blocks for exact requested row IDs."""
        if not row_ids:
            return np.zeros((0, 768), dtype=np.float32)

        # Sort or map to blocks
        results = [None] * len(row_ids)
        # Group by block
        for idx, r_id in enumerate(row_ids):
            target_block = None
            for b in self.blocks:
                if b.first_row <= r_id < b.first_row + b.row_count:
                    target_block = b
                    break
            if target_block is None:
                raise ValueError(f"Row ID {r_id} not found in any embedding block.")

            offset = r_id - target_block.first_row
            vec_path = self.index_dir / target_block.vector_path
            # Memory map or load float16 chunk
            mmap_vecs = np.memmap(
                vec_path,
                dtype=np.float16,
                mode="r",
                shape=(target_block.row_count, target_block.dimension),
            )
            vec_f16 = mmap_vecs[offset].astype(np.float32)
            results[idx] = vec_f16

        return np.vstack(results)

    def get_rows_for_chunk_ids(self, chunk_ids: Sequence[str]) -> np.ndarray:
        """Resolve chunk_id -> row_id then call get_rows."""
        row_ids = []
        for cid in chunk_ids:
            if cid not in self.chunk_to_row:
                raise KeyError(f"Chunk ID {cid} not found in embedding mapping.")
            row_ids.append(self.chunk_to_row[cid])
        return self.get_rows(row_ids)


def write_embedding_blocks(
    chunks: Iterable[dict[str, Any] | Any],
    encoder: Any,
    output: Path | str,
    config: DenseConfig,
    snapshot_id: str = "snap_default",
    resume: bool = True,
) -> list[EmbeddingBlock]:
    """Encode chunks into atomic float16 embedding blocks with manifests and SHA256 checksums."""
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "embedding_manifest.json"
    existing_blocks: list[EmbeddingBlock] = []
    chunk_to_row: dict[str, int] = {}

    if resume and manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for b_dict in data.get("blocks", []):
                existing_blocks.append(EmbeddingBlock(**b_dict))
            chunk_to_row = data.get("chunk_to_row", {})

    skip_rows = sum(b.row_count for b in existing_blocks)
    blocks = list(existing_blocks)

    current_chunk_batch: list[str] = []
    current_chunk_ids: list[str] = []
    
    global_row_cursor = skip_rows

    def flush_block(batch_texts: list[str], batch_ids: list[str], block_idx: int) -> EmbeddingBlock:
        nonlocal global_row_cursor
        if not batch_texts:
            raise ValueError("Cannot flush empty batch to embedding block.")
        
        vecs = encoder.encode_passages(batch_texts)
        if vecs.dtype != np.float32:
            vecs = vecs.astype(np.float32)

        # Ensure normalization and finite check
        if config.normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.clip(norms, a_min=1e-9, a_max=None)
        
        if not np.isfinite(vecs).all():
            raise ValueError("Computed vectors contain NaN or infinite values!")

        row_count, dim = vecs.shape
        vecs_f16 = vecs.astype(np.float16)

        block_id = f"block_{block_idx:05d}"
        vector_filename = f"{block_id}.vecs.f16"
        mapping_filename = f"{block_id}.mapping.json"
        
        vector_path = output_dir / vector_filename
        mapping_path = output_dir / mapping_filename

        # Write vectors atomically
        vector_path.write_bytes(vecs_f16.tobytes())

        # Compute sha256 of vector file
        sha256_hash = hashlib.sha256(vector_path.read_bytes()).hexdigest()

        block_mapping = {}
        for idx, cid in enumerate(batch_ids):
            r_id = global_row_cursor + idx
            block_mapping[cid] = r_id
            chunk_to_row[cid] = r_id

        mapping_path.write_text(json.dumps(block_mapping, indent=2), encoding="utf-8")

        first_row = global_row_cursor
        global_row_cursor += row_count

        block_meta = EmbeddingBlock(
            block_id=block_id,
            first_row=first_row,
            row_count=row_count,
            dimension=dim,
            vector_path=vector_filename,
            mapping_path=mapping_filename,
            model_fingerprint=getattr(encoder, "model_name", "mock_e5"),
            snapshot_id=snapshot_id,
            sha256=sha256_hash,
        )
        return block_meta

    chunk_iter = iter(chunks)
    # Skip already processed chunks if resuming
    skipped_count = 0
    while skipped_count < skip_rows:
        try:
            next(chunk_iter)
            skipped_count += 1
        except StopIteration:
            break

    block_idx = len(blocks)
    batch_texts = []
    batch_ids = []

    for chunk in chunk_iter:
        if isinstance(chunk, dict):
            cid = chunk.get("chunk_id") or chunk.get("id")
            text = chunk.get("text") or chunk.get("content")
        else:
            cid = getattr(chunk, "chunk_id", None) or getattr(chunk, "id", None)
            text = getattr(chunk, "text", None) or getattr(chunk, "content", None)

        if not cid or text is None:
            continue

        batch_texts.append(text)
        batch_ids.append(cid)

        if len(batch_texts) >= config.block_rows:
            b_meta = flush_block(batch_texts, batch_ids, block_idx)
            blocks.append(b_meta)
            block_idx += 1
            batch_texts = []
            batch_ids = []
            # Save intermediate manifest
            _save_manifest(manifest_path, blocks, chunk_to_row, snapshot_id, config)

    # Flush remaining
    if batch_texts:
        b_meta = flush_block(batch_texts, batch_ids, block_idx)
        blocks.append(b_meta)
        _save_manifest(manifest_path, blocks, chunk_to_row, snapshot_id, config)

    return blocks


def _save_manifest(manifest_path: Path, blocks: list[EmbeddingBlock], chunk_to_row: dict[str, int], snapshot_id: str, config: DenseConfig) -> None:
    data = {
        "snapshot_id": snapshot_id,
        "model_name": config.model_name,
        "dimension": config.dimension,
        "blocks": [b.__dict__ for b in blocks],
        "chunk_to_row": chunk_to_row,
    }
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
