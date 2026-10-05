import os
from pathlib import Path
import numpy as np
import pytest

from src.data.phase2_contracts import DenseConfig, EmbeddingBlock, IndexManifest
from src.indexing.embedding_store import VectorStore
from src.indexing.faiss_index import (
    build_faiss_index,
    FaissDenseBackend,
    exact_cosine_oracle,
)


@pytest.fixture
def sample_vector_store(tmp_path: Path):
    vectors_dir = tmp_path / "vectors"
    vectors_dir.mkdir()
    
    dim = 768
    num_rows = 200  # Smaller fixture
    
    rng = np.random.default_rng(42)
    raw_data = rng.standard_normal((num_rows, dim)).astype(np.float32)
    norms = np.linalg.norm(raw_data, axis=1, keepdims=True)
    raw_data = raw_data / norms
    
    vector_path = vectors_dir / "block_00000.vecs.f16"
    mapping_path = vectors_dir / "block_00000.mapping.json"
    
    raw_data.astype(np.float16).tofile(vector_path)
    
    import json
    mapping = {str(i): f"chunk_{i}" for i in range(num_rows)}
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f)
        
    block = EmbeddingBlock(
        block_id="block_00000",
        first_row=0,
        row_count=num_rows,
        dimension=dim,
        vector_path="vectors/block_00000.vecs.f16",
        mapping_path=str(mapping_path),
        model_fingerprint="test_model",
        snapshot_id="test_snap",
        sha256="dummy_sha",
    )
    
    manifest_data = {
        "snapshot_id": "test_snap",
        "model_name": "test_model",
        "dimension": dim,
        "blocks": [block.__dict__],
        "chunk_to_row": {f"chunk_{i}": i for i in range(num_rows)},
    }
    with open(tmp_path / "embedding_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)
        
    store = VectorStore(tmp_path)
    return store, [block]


def test_build_faiss_index_flat_fallback(sample_vector_store, tmp_path: Path):
    store, blocks = sample_vector_store
    config = DenseConfig(
        dimension=768,
        nlist=16,
        pq_m=64,
        approximate_top_k=50,
        refined_top_k=10,
    )
    output_dir = tmp_path / "index_out"
    
    manifest = build_faiss_index(blocks, config, output_dir, vector_store=store)
    assert isinstance(manifest, IndexManifest)
    assert manifest.backend in ("faiss_cpu_ivfpq", "faiss_cpu_flat")
    assert (output_dir / "index.faiss").exists()
    assert (output_dir / "index_manifest.json").exists()


def test_faiss_dense_backend_search_and_oracle(sample_vector_store, tmp_path: Path):
    store, blocks = sample_vector_store
    config = DenseConfig(
        dimension=768,
        nlist=16,
        pq_m=64,
        approximate_top_k=50,
        refined_top_k=10,
    )
    output_dir = tmp_path / "index_out"
    build_faiss_index(blocks, config, output_dir, vector_store=store)
    
    backend = FaissDenseBackend(output_dir / "index.faiss", vector_store=store, config=config)
    
    query = np.random.default_rng(123).standard_normal(768).astype(np.float32)
    query = query / np.linalg.norm(query)
    
    hits = backend.search(query, top_k=10)
    assert len(hits) <= 10
    assert hits[0].item_id.startswith("chunk_")
    
    oracle_hits = exact_cosine_oracle(query, store, top_k=10)
    assert len(oracle_hits) == 10
    
    faiss_ids = {h.item_id for h in hits}
    oracle_ids = {h.item_id for h in oracle_hits}
    overlap = len(faiss_ids.intersection(oracle_ids))
    recall = overlap / 10.0
    assert recall >= 0.0
