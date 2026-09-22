from pathlib import Path
import pytest
from src.utils.config import load_config

def test_load_config_valid(tmp_path: Path):
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text("run:\n  seed: 42\n  experiment_name: test\n", encoding="utf-8")
    cfg = load_config(config_file)
    assert cfg["run"]["seed"] == 42
    assert cfg["run"]["experiment_name"] == "test"

def test_load_default_config_exists():
    cfg = load_config("configs/r2ai_mvp.yaml")
    assert "indexing" in cfg
    assert "retrieval" in cfg
    assert "calibration" in cfg

def test_load_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("configs/non_existent_config.yaml")
