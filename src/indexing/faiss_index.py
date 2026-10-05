from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np

from src.data.phase2_contracts import DenseConfig, EmbeddingBlock, IndexManifest
from src.data.schemas import SearchHit
from src.indexing.embedding_store import VectorStore

logger = logging.getLogger(__name__)


def compute_sha256(file_path: Path) -> str:
    import hashlib
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def build_faiss_index(
    blocks: Sequence[EmbeddingBlock],
    config: DenseConfig,
    output: Path,
    vector_store: VectorStore | None = None,
) -> IndexManifest:
    """Builds a FAISS index (IVF-PQ or Flat fallback) over embedding blocks and writes manifest."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    if vector_store is None:
        vector_store = VectorStore(blocks)

    d = config.dimension
    total_rows = sum(b.row_count for b in blocks)
    logger.info("Building FAISS index for %d rows, dimension %d", total_rows, d)

    use_flat = total_rows < 1000 or total_rows < config.nlist * 4

    if use_flat:
        logger.info("Total rows (%d) too small for IVF-PQ; falling back to IndexFlatIP", total_rows)
        index = faiss.IndexFlatIP(d)
        backend_name = "faiss_cpu_flat"
    else:
        quantizer = faiss.IndexFlatIP(d)
        nlist = min(config.nlist, max(1, total_rows // 39))
        pq_m = config.pq_m
        pq_bits = config.pq_bits
        if d % pq_m != 0:
            raise ValueError(f"Dimension {d} must be divisible by pq_m {pq_m}")
        
        try:
            index = faiss.IndexIVFPQ(quantizer, d, nlist, pq_m, pq_bits, faiss.METRIC_INNER_PRODUCT)
            backend_name = "faiss_cpu_ivfpq"
        except Exception as e:
            logger.warning("Failed to initialize IndexIVFPQ (%s), falling back to IndexFlatIP", e)
            index = faiss.IndexFlatIP(d)
            backend_name = "faiss_cpu_flat"

    if isinstance(index, faiss.IndexIVF) and not index.is_trained:
        train_rows_target = min(config.train_rows, total_rows)
        sample_vectors = []
        collected = 0
        for block in blocks:
            if collected >= train_rows_target:
                break
            take = min(block.row_count, train_rows_target - collected)
            rows = vector_store.get_rows(range(block.first_row, block.first_row + take))
            if len(rows) > 0:
                sample_vectors.append(rows.astype(np.float32))
                collected += len(rows)
        
        if sample_vectors:
            train_data = np.vstack(sample_vectors)
            norms = np.linalg.norm(train_data, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            train_data = train_data / norms
            logger.info("Training FAISS index with %d vectors", len(train_data))
            index.train(train_data)

    added_total = 0
    for block in blocks:
        if block.row_count == 0:
            continue
        vectors = vector_store.get_rows(range(block.first_row, block.first_row + block.row_count))
        if len(vectors) == 0:
            continue
        vecs_f32 = vectors.astype(np.float32)
        norms = np.linalg.norm(vecs_f32, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        vecs_f32 = vecs_f32 / norms

        index.add(vecs_f32)
        added_total += len(vecs_f32)

    logger.info("Added %d vectors to FAISS index", added_total)

    index_path = output / "index.faiss"
    faiss.write_index(index, str(index_path))

    index_sha = compute_sha256(index_path)
    artifact_paths_and_hashes = {
        "index.faiss": index_sha,
    }

    snapshot_id = blocks[0].snapshot_id if blocks else "unknown"
    language = "und"

    manifest = IndexManifest(
        snapshot_id=snapshot_id,
        item_type="chunk",
        language=language,
        member_count=added_total,
        membership_sha256=index_sha,
        backend=backend_name,
        model_fingerprint=blocks[0].model_fingerprint if blocks else None,
        tokenizer_fingerprint="multilingual-e5-base",
        parameters={
            "dimension": d,
            "nlist": getattr(config, "nlist", 4096),
            "pq_m": getattr(config, "pq_m", 64),
            "pq_bits": getattr(config, "pq_bits", 8),
            "nprobe": getattr(config, "nprobe", 32),
        },
        artifact_paths_and_hashes=artifact_paths_and_hashes,
    )

    manifest_path = output / "index_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.__dict__, f, indent=2)

    return manifest


def exact_cosine_oracle(
    query_vector: np.ndarray,
    vector_store: VectorStore,
    top_k: int = 200,
) -> list[SearchHit]:
    """Exact blockwise scan oracle computing cosine similarity."""
    q = query_vector.astype(np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm > 0:
        q = q / q_norm

    scores = []
    for block in vector_store.blocks:
        if block.row_count == 0:
            continue
        rows = vector_store.get_rows(range(block.first_row, block.first_row + block.row_count))
        if len(rows) == 0:
            continue
        vecs = rows.astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        vecs = vecs / norms

        block_scores = np.dot(vecs, q)
        
        mapping = {}
        if Path(block.mapping_path).exists():
            with open(block.mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)

        for idx, score in enumerate(block_scores):
            global_row_id = block.first_row + idx
            chunk_id = mapping.get(str(global_row_id), mapping.get(str(idx), f"chunk_{global_row_id}"))
            scores.append((float(score), chunk_id))

    scores.sort(key=lambda x: x[0], reverse=True)
    top_scores = scores[:top_k]

    hits = []
    for rank, (score, chunk_id) in enumerate(top_scores, start=1):
        hits.append(SearchHit(
            item_id=chunk_id,
            item_type="chunk",
            doc_id="doc_unknown",
            language="und",
            source="exact_oracle",
            raw_score=score,
            rank=rank,
        ))
    return hits


class FaissDenseBackend:
    """FAISS dense search backend with exact raw vector refinement."""

    def __init__(
        self,
        index_path: Path | str,
        vector_store: VectorStore | None = None,
        config: DenseConfig | None = None,
    ):
        self.index_path = Path(index_path)
        self.vector_store = vector_store
        self.config = config or DenseConfig()
        self.index = faiss.read_index(str(self.index_path))
        
        if isinstance(self.index, faiss.IndexIVF):
            self.index.nprobe = self.config.nprobe

    def search(self, query_vectors: np.ndarray, top_k: int = 1000) -> list[SearchHit]:
        """Searches approximate candidates and refines with exact raw float16 vectors."""
        q = query_vectors.astype(np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        
        q_norm = np.linalg.norm(q, axis=1, keepdims=True)
        q_norm[q_norm == 0.0] = 1.0
        q = q / q_norm

        approx_k = max(self.config.approximate_top_k, top_k * 4)
        distances, indices = self.index.search(q, approx_k)

        row_ids = indices[0]
        valid_mask = row_ids >= 0
        valid_row_ids = row_ids[valid_mask]

        if len(valid_row_ids) == 0 or self.vector_store is None:
            hits = []
            for rank, (idx, dist) in enumerate(zip(row_ids[0][:top_k], distances[0][:top_k]), start=1):
                if idx < 0:
                    continue
                hits.append(SearchHit(
                    item_id=f"chunk_{idx}",
                    item_type="chunk",
                    doc_id="doc_unknown",
                    language="und",
                    source="faiss_approx",
                    raw_score=float(dist),
                    rank=rank,
                ))
            return hits

        raw_rows = self.vector_store.get_rows(valid_row_ids.tolist())
        if len(raw_rows) == 0:
            return []

        vecs = raw_rows.astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        vecs = vecs / norms

        q_single = q[0]
        refined_scores = np.dot(vecs, q_single)

        scored_candidates = []
        for row_id, score in zip(valid_row_ids, refined_scores):
            chunk_id = self._row_to_chunk(int(row_id))
            scored_candidates.append((float(score), chunk_id))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        final_top = scored_candidates[:top_k]

        hits = []
        for rank, (score, chunk_id) in enumerate(final_top, start=1):
            hits.append(SearchHit(
                item_id=chunk_id,
                item_type="chunk",
                doc_id="doc_unknown",
                language="und",
                source="faiss_refined",
                raw_score=score,
                rank=rank,
            ))
        return hits

    def _row_to_chunk(self, row_id: int) -> str:
        if not self.vector_store:
            return f"chunk_{row_id}"
        for block in self.vector_store.blocks:
            if block.first_row <= row_id < block.first_row + block.row_count:
                import json
                if Path(block.mapping_path).exists():
                    with open(block.mapping_path, "r", encoding="utf-8") as f:
                        mapping = json.load(f)
                    local_id = row_id - block.first_row
                    return mapping.get(str(row_id), mapping.get(str(local_id), f"chunk_{row_id}"))
        return f"chunk_{row_id}"
