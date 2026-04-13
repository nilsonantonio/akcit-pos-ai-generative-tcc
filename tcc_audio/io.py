"""Small IO helpers shared by CLI scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read CSV as strings while preserving empty cells for validation."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

