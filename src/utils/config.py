from pathlib import Path
from typing import Any, Union
import yaml


def load_config(config_path: Union[str, Path]) -> dict[str, Any]:
    """Load and parse a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the specified config file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
        return loaded if isinstance(loaded, dict) else {}
