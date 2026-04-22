"""Runtime helpers shared by dataset, inference, and metrics scripts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from tcc_audio.io import ensure_parent_dir, read_csv
from tcc_audio.schema import EVAL_SAMPLES_REQUIRED_COLUMNS, SAMPLES_OPTIONAL_COLUMNS


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_sample_columns(samples: pd.DataFrame) -> pd.DataFrame:
    for column in EVAL_SAMPLES_REQUIRED_COLUMNS + SAMPLES_OPTIONAL_COLUMNS:
        if column not in samples.columns:
            samples[column] = ""
    return samples


def load_samples(path: str | Path) -> pd.DataFrame:
    samples = read_csv(path)
    return ensure_sample_columns(samples)


def save_samples(samples: pd.DataFrame, path: str | Path) -> None:
    ensure_parent_dir(path)
    samples.to_csv(path, index=False)


def stringify_csv_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def refresh_sample_status(samples: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "wer",
        "speaker_similarity",
        "nisqa",
        "f0_rmse",
        "rtf",
        "train_gpu_hours",
        "inference_seconds",
        "cost_usd",
    ]
    audio_exists = samples["audio_path"].astype(str).apply(lambda value: Path(value).exists() if str(value).strip() else False)
    generated_mask = audio_exists & samples["failure_reason"].astype(str).str.strip().eq("")
    samples.loc[generated_mask & samples["status"].astype(str).str.lower().eq("pending"), "status"] = "generated"
    completed_mask = generated_mask.copy()
    for column in metric_columns:
        completed_mask &= samples[column].astype(str).str.strip().ne("")
    samples.loc[completed_mask, "status"] = "ok"
    return samples


def resolve_path(path_str: str | Path, project_root: str | Path = ".") -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path(project_root) / path


@dataclass
class EmbeddingIndex:
    frame: pd.DataFrame

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingIndex":
        frame = read_csv(path)
        return cls(frame=frame)

    def by_speaker(self, speaker_id: str) -> pd.Series:
        matches = self.frame[self.frame["speaker_id"].eq(speaker_id)]
        if matches.empty:
            raise KeyError(f"speaker_id not found in embeddings index: {speaker_id}")
        return matches.iloc[0]


def load_speaker_reference_map(path: str | Path) -> dict[str, dict[str, str]]:
    frame = read_csv(path)
    required = {"speaker_id", "reference_audio"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in speaker selection file: {', '.join(sorted(missing))}")
    output: dict[str, dict[str, str]] = {}
    for _, row in frame.iterrows():
        output[row["speaker_id"]] = row.to_dict()
    return output


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def update_rows(samples: pd.DataFrame, sample_ids: Iterable[str], **updates: object) -> pd.DataFrame:
    sample_ids = list(sample_ids)
    mask = samples["sample_id"].isin(sample_ids)
    for key, value in updates.items():
        samples.loc[mask, key] = value
    return samples


def require_dependency(module_name: str, install_hint: str):
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise SystemExit(f"Missing dependency '{module_name}'. Install with: {install_hint}") from exc
