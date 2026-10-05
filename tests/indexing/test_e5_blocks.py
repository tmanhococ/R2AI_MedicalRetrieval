import os
import pytest
import numpy as np
from pathlib import Path
import tempfile
import json
import hashlib

from src.data.phase2_contracts import DenseConfig, EmbeddingBlock
from src.indexing.dense_encoder import DenseEncoder
from src.indexing.embedding_store import write_embedding_blocks, VectorStore

def test_dense_config_defaults():
    config = DenseConfig()
    assert config.dimension == 768
    assert config.query_prefix == "query: "
    assert config.passage_prefix == "passage: "
    assert config.normalize is True
    assert config.vector_storage_dtype == "float16"

def test_fake_encoder_for_testing():
    # Verify we can use a mock/fake encoder in tests as allowed by task spec
    encoder = DenseEncoder.get_mock_encoder(dimension=768)
    passages = ["passage 1 test", "passage 2 test"]
    vecs = encoder.encode_passages(passages)
    assert vecs.shape == (2, 768)
    assert vecs.dtype == np.float32
    # check normalization
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)

def test_dense_encoder_nan_check():
    encoder = DenseEncoder.get_mock_encoder(dimension=768)
    # inject NaN behavior or verify validation
    with pytest.raises(ValueError):
        # We can test validation function directly if available
        DenseEncoder.validate_vectors(np.array([[float('nan'), 0.0, *([0.0]*766)]], dtype=np.float32))

def test_write_embedding_blocks_and_vector_store(tmp_path):
    chunks = [
        {"chunk_id": f"chunk_{i}", "doc_id": i // 2, "chunk_index": i % 2, "text": f"Medical text {i}", "title": f"Doc {i//2}", "section": "abstract"}
        for i in range(25)
    ]
    config = DenseConfig(block_rows=10) # 3 blocks: 10, 10, 5
    encoder = DenseEncoder.get_mock_encoder(dimension=768)
    
    output_dir = tmp_path / "dense_index"
    blocks = write_embedding_blocks(chunks, encoder, output_dir, config, snapshot_id="snap_test", resume=False)
    
    assert len(blocks) == 3
    assert blocks[0].row_count == 10
    assert blocks[1].row_count == 10
    assert blocks[2].row_count == 5
    
    # Test VectorStore
    store = VectorStore(output_dir)
    rows = store.get_rows([0, 5, 15, 24])
    assert rows.shape == (4, 768)
    assert rows.dtype == np.float32
    # check normalization
    np.testing.assert_allclose(np.linalg.norm(rows, axis=1), 1.0, atol=1e-5)
    
    # Test get_rows_for_chunk_ids
    chunk_rows = store.get_rows_for_chunk_ids(["chunk_0", "chunk_12", "chunk_24"])
    assert chunk_rows.shape == (3, 768)
    np.testing.assert_allclose(chunk_rows[0], store.get_rows([0])[0])
    np.testing.assert_allclose(chunk_rows[1], store.get_rows([12])[0])
    np.testing.assert_allclose(chunk_rows[2], store.get_rows([24])[0])

def test_resume_embedding_blocks(tmp_path):
    chunks = [
        {"chunk_id": f"chunk_{i}", "doc_id": 0, "chunk_index": i, "text": f"Text {i}", "title": "Doc", "section": "main"}
        for i in range(15)
    ]
    config = DenseConfig(block_rows=5)
    encoder = DenseEncoder.get_mock_encoder(dimension=768)
    output_dir = tmp_path / "dense_resume"
    
    # Write first 2 blocks (10 rows)
    blocks_part = write_embedding_blocks(chunks[:10], encoder, output_dir, config, snapshot_id="snap_test", resume=False)
    assert len(blocks_part) == 2
    
    # Resume with full chunks
    blocks_full = write_embedding_blocks(chunks, encoder, output_dir, config, snapshot_id="snap_test", resume=True)
    assert len(blocks_full) == 3
    assert blocks_full[2].row_count == 5
