import os
import pytest
import json
import zipfile
from pathlib import Path
from src.utils.checkpoint import InferenceCheckpointManager
from src.submission.vibiomir_json_zip import VibiomirExporter
from src.data.source_store import EvidenceSourceStore
from src.data.schemas import FinalPrediction

def test_phase2_stages_integration(tmp_path):
    # Setup test fixture files & dirs
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Test InferenceCheckpointManager & Checkpointing
    manager = InferenceCheckpointManager(output_dir=checkpoint_dir, run_fingerprint="fp1")
    assert not manager.is_query_completed("Q1")
    
    # Save a checkpoint
    manager.save_checkpoint("Q1", status="completed", prediction={"query_id": "Q1", "document_ids": [1500]})
    assert manager.is_query_completed("Q1")
    
    # Test Exporter & EvidenceSourceStore
    class MockUnderlying:
        def contains_official_doc(self, doc_id): return True
        def get_document(self, doc_id):
            class Doc:
                source_text = "This is a medical document about biomarker expression in cancer."
                language = "en"
                doc_id = doc_id
                source_version = "1.0"
                source_sha256 = "abc"
            return Doc()
    
    source_store = EvidenceSourceStore(underlying=MockUnderlying())
    exporter = VibiomirExporter(json_filename="results.json", zip_filename="submission_B.zip")
    
    predictions = [
        FinalPrediction(
            query_id="1",
            document_ids=[1500],
            chunk_ids=[],
            document_scores={1500: 0.95},
            chunk_scores={}
        )
    ]
    
    res = exporter.export(predictions, queries=None, store=source_store, output_dir=output_dir)
    zip_path = output_dir / "submission_B.zip"
    assert zip_path.exists()
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        namelist = zf.namelist()
        assert "results.json" in namelist
