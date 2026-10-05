"""Multilingual Medical Retrieval MVP source package."""
import os
from pathlib import Path

# Redirect large cache directories to drive D: to prevent filling drive C:
_D_CACHE = Path("D:/Research/R2AI/.cache")
if _D_CACHE.parent.exists():
    _hf_home = str(_D_CACHE / "huggingface")
    _torch_home = str(_D_CACHE / "torch")
    _tmp_dir = str(_D_CACHE / "tmp")
    os.environ.setdefault("HF_HOME", _hf_home)
    os.environ.setdefault("TRANSFORMERS_CACHE", _hf_home)
    os.environ.setdefault("TORCH_HOME", _torch_home)
    os.environ.setdefault("TMPDIR", _tmp_dir)
