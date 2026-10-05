### Task 1: Environment & Project Scaffolding, Base Config

**Files:**
- Create: `pyproject.toml`
- Create: `configs/r2ai_mvp.yaml`
- Create: `src/__init__.py`
- Create: `src/utils/__init__.py`
- Create: `src/utils/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `load_config(path: str | Path) -> dict`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with ModuleNotFoundError or FileNotFoundError.

- [ ] **Step 3: Write minimal implementation**

Tạo `pyproject.toml`, `configs/r2ai_mvp.yaml` (dựa theo schema trong docs/superpowers/spec/spec.md), và `src/utils/config.py`:

```python
# src/utils/config.py
from pathlib import Path
import yaml

def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml configs/r2ai_mvp.yaml src/__init__.py src/utils/ tests/test_config.py
git commit -m "feat: setup project scaffolding and base yaml config loader"
```
