"""Shared CLI defaults and config-backed path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from tcc_audio.io import read_yaml

DEFAULT_CONFIG_PATH = Path("configs/speecht5_minimal.yaml")
DEFAULT_PROMPTS_PATH = Path("data/prompts/ptbr_test_prompts.csv")
DEFAULT_RAW_DIR = Path("data/raw/common_voice_pt")
DEFAULT_RAW_TSV = DEFAULT_RAW_DIR / "validated.tsv"
DEFAULT_RAW_CLIPS_DIR = DEFAULT_RAW_DIR / "clips"
DEFAULT_RAW_METADATA_PATH = Path("data/manifests/common_voice_metadata.csv")
DEFAULT_PROCESSED_DIR = Path("data/processed/common_voice_pt")
DEFAULT_PROCESSED_METADATA_PATH = Path("data/manifests/common_voice_processed.csv")
DEFAULT_MANIFEST_PATH = Path("data/manifests/data_manifest.csv")
DEFAULT_SPEAKER_SELECTION_PATH = Path("data/manifests/speaker_selection.csv")
DEFAULT_EMBEDDINGS_DIR = Path("artifacts/embeddings")
DEFAULT_EMBEDDINGS_INDEX_PATH = DEFAULT_EMBEDDINGS_DIR / "speaker_embeddings.csv"
DEFAULT_RUN_MATRIX_PATH = Path("artifacts/run_matrix.csv")
DEFAULT_SAMPLES_PATH = Path("artifacts/evaluation/samples.csv")
DEFAULT_AUDIO_BASE_DIR = Path("artifacts/audio")
DEFAULT_CHECKPOINT_DIR = Path("artifacts/checkpoints/lora")
DEFAULT_EVALUATION_DIR = Path("artifacts/evaluation")
DEFAULT_METRICS_PATH = DEFAULT_EVALUATION_DIR / "metrics_summary.csv"
DEFAULT_COSTS_PATH = DEFAULT_EVALUATION_DIR / "cost_summary.csv"
DEFAULT_DATASET_INVENTORY_JSON = Path("artifacts/dataset_inventory.json")
DEFAULT_REPORT_ASSETS_DIR = Path("report_assets")
DEFAULT_NISQA_PATH = Path("NISQA")
DEFAULT_TRAINING_COST_SUMMARY_PATH = DEFAULT_EVALUATION_DIR / "manual_training_costs_summary.csv"


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = config.get(name, {})
    return section if isinstance(section, Mapping) else {}


def resolve_config_path(config_path: str | Path | None = None) -> Path | None:
    if config_path not in (None, ""):
        resolved = Path(config_path)
        if not resolved.exists():
            raise SystemExit(f"Config not found: {resolved}")
        return resolved
    if DEFAULT_CONFIG_PATH.exists():
        return DEFAULT_CONFIG_PATH
    return None


def load_cli_config(config_path: str | Path | None = None) -> tuple[Path | None, dict[str, Any]]:
    resolved = resolve_config_path(config_path)
    if resolved is None:
        return None, {}
    return resolved, read_yaml(resolved)


def resolve_data_path(config: Mapping[str, Any], key: str, fallback: str | Path) -> Path:
    return Path(_section(config, "data").get(key) or fallback)


def resolve_deliverable_path(config: Mapping[str, Any], key: str, fallback: str | Path) -> Path:
    return Path(_section(config, "deliverables").get(key) or fallback)


def resolve_samples_path(config: Mapping[str, Any]) -> Path:
    return resolve_deliverable_path(config, "samples", DEFAULT_SAMPLES_PATH)


def resolve_evaluation_dir(config: Mapping[str, Any]) -> Path:
    samples_path = resolve_samples_path(config)
    return samples_path.parent if samples_path.parent != Path("") else DEFAULT_EVALUATION_DIR


def resolve_sample_rate(config: Mapping[str, Any], fallback: int = 16000) -> int:
    value = _section(config, "data").get("sample_rate", fallback)
    return int(value)
