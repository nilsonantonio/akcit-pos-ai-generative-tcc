"""Helpers for removing SpeechT5 LoRA training artifacts and conditions."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import yaml

from tcc_audio.io import read_yaml
from tcc_audio.runtime import load_samples, save_samples

DEFAULT_CONFIG_PATH = Path("configs/speecht5_minimal.yaml")
DEFAULT_SAMPLES_PATH = Path("artifacts/evaluation/samples.csv")
DEFAULT_CHECKPOINT_DIR = Path("artifacts/checkpoints/lora")
DEFAULT_AUDIO_BASE_DIR = Path("artifacts/audio")
DEFAULT_EVALUATION_DIR = Path("artifacts/evaluation")
DEFAULT_REPORT_ASSETS_DIR = Path("report_assets")
EVALUATION_SUMMARY_FILES = [
    "metrics_summary.csv",
    "cost_summary.csv",
    "metrics_by_speaker.csv",
    "cost_by_speaker.csv",
]


@dataclass
class CleanupPaths:
    config_path: Path
    samples_path: Path
    checkpoint_dir: Path
    audio_base_dir: Path
    evaluation_dir: Path
    report_assets_dir: Path
    run_matrix_path: Path | None


@dataclass
class CleanupResult:
    condition_ids: list[str]
    removed_materialized_rows: int
    removed_checkpoint_dirs: list[Path]
    removed_audio_dirs: list[Path]
    removed_evaluation_files: list[Path]
    removed_report_assets_dir: bool


@dataclass
class RemoveConditionResult:
    cleanup: CleanupResult
    removed_base_rows: int
    removed_run_matrix: bool
    updated_config: bool


def _iter_lora_conditions(config: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    output: list[Mapping[str, Any]] = []
    for condition in config.get("conditions", []):
        if not isinstance(condition, Mapping):
            continue
        train_strategy = str(condition.get("train_strategy", "")).strip().lower()
        if train_strategy == "lora" or "lora" in condition:
            output.append(condition)
    return output


def _resolve_lora_condition_ids(config: Mapping[str, Any]) -> list[str]:
    return [str(condition["id"]) for condition in _iter_lora_conditions(config)]


def _materialized_mask(samples: pd.DataFrame) -> pd.Series:
    return samples["checkpoint_step"].astype(str).str.strip().ne("")


def _load_config(config_path: str | Path) -> dict[str, Any]:
    return read_yaml(config_path)


def resolve_cleanup_paths(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    samples_path: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    audio_base_dir: str | Path | None = None,
    evaluation_dir: str | Path | None = None,
    report_assets_dir: str | Path | None = None,
) -> CleanupPaths:
    config_path = Path(config_path)
    config = _load_config(config_path)
    deliverables = config.get("deliverables", {})
    deliverables = deliverables if isinstance(deliverables, Mapping) else {}

    resolved_samples = Path(samples_path or deliverables.get("samples") or DEFAULT_SAMPLES_PATH)
    resolved_checkpoint_dir = Path(checkpoint_dir or DEFAULT_CHECKPOINT_DIR)
    resolved_audio_base_dir = Path(audio_base_dir or DEFAULT_AUDIO_BASE_DIR)
    resolved_evaluation_dir = Path(evaluation_dir or resolved_samples.parent or DEFAULT_EVALUATION_DIR)
    resolved_report_assets_dir = Path(report_assets_dir or deliverables.get("report_assets_dir") or DEFAULT_REPORT_ASSETS_DIR)
    run_matrix_value = deliverables.get("run_matrix")
    resolved_run_matrix = Path(run_matrix_value) if run_matrix_value else None

    return CleanupPaths(
        config_path=config_path,
        samples_path=resolved_samples,
        checkpoint_dir=resolved_checkpoint_dir,
        audio_base_dir=resolved_audio_base_dir,
        evaluation_dir=resolved_evaluation_dir,
        report_assets_dir=resolved_report_assets_dir,
        run_matrix_path=resolved_run_matrix,
    )


def _resolve_target_conditions(config: Mapping[str, Any], condition_ids: Iterable[str]) -> list[str]:
    requested = [str(value).strip() for value in condition_ids if str(value).strip()]
    if not requested:
        raise ValueError("At least one --condition is required.")

    has_all = any(value == "all" for value in requested)
    if has_all and len(requested) > 1:
        raise ValueError("`all` cannot be combined with specific condition ids.")

    available = _resolve_lora_condition_ids(config)
    if has_all:
        if not available:
            raise ValueError("No LoRA conditions found in the experiment config.")
        return available

    missing = sorted(set(requested).difference(available))
    if missing:
        raise ValueError(f"Unknown LoRA condition ids: {', '.join(missing)}")
    return requested


def _confirm_operation(
    *,
    action: str,
    condition_ids: Iterable[str],
    paths: CleanupPaths,
    bypass: bool,
    updates_config: bool,
    removes_base_rows: bool,
) -> None:
    if bypass:
        return

    condition_labels = ", ".join(condition_ids)
    print(f"About to {action}.")
    print(f"Conditions: {condition_labels}")
    print(f"Checkpoints root: {paths.checkpoint_dir}")
    print(f"Audio root: {paths.audio_base_dir}")
    print(f"Samples file: {paths.samples_path}")
    print(f"Evaluation dir: {paths.evaluation_dir}")
    print(f"Report assets dir: {paths.report_assets_dir}")
    print(f"Will update config: {'yes' if updates_config else 'no'}")
    print(f"Will remove base sample rows: {'yes' if removes_base_rows else 'no'}")
    response = input("Type 'yes' to continue: ").strip().lower()
    if response != "yes":
        raise SystemExit("Aborted by user.")


def _remove_tree(path: Path) -> bool:
    if not path.exists():
        return False
    shutil.rmtree(path)
    return True


def _remove_file(path: Path) -> bool:
    if not path.exists():
        return False
    path.unlink()
    return True


def cleanup_training_results(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    condition_ids: Iterable[str],
    samples_path: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    audio_base_dir: str | Path | None = None,
    evaluation_dir: str | Path | None = None,
    report_assets_dir: str | Path | None = None,
    bypass: bool = False,
) -> CleanupResult:
    config = _load_config(config_path)
    target_conditions = _resolve_target_conditions(config, condition_ids)
    paths = resolve_cleanup_paths(
        config_path=config_path,
        samples_path=samples_path,
        checkpoint_dir=checkpoint_dir,
        audio_base_dir=audio_base_dir,
        evaluation_dir=evaluation_dir,
        report_assets_dir=report_assets_dir,
    )
    _confirm_operation(
        action="clear SpeechT5 training results",
        condition_ids=target_conditions,
        paths=paths,
        bypass=bypass,
        updates_config=False,
        removes_base_rows=False,
    )

    removed_checkpoint_dirs: list[Path] = []
    removed_audio_dirs: list[Path] = []
    removed_evaluation_files: list[Path] = []

    for condition_id in target_conditions:
        checkpoint_root = paths.checkpoint_dir / condition_id
        if _remove_tree(checkpoint_root):
            removed_checkpoint_dirs.append(checkpoint_root)
        audio_root = paths.audio_base_dir / condition_id
        if _remove_tree(audio_root):
            removed_audio_dirs.append(audio_root)

    removed_materialized_rows = 0
    if paths.samples_path.exists():
        samples = load_samples(paths.samples_path)
        mask = samples["condition"].isin(target_conditions) & _materialized_mask(samples)
        removed_materialized_rows = int(mask.sum())
        if removed_materialized_rows:
            samples = samples.loc[~mask].copy()
            save_samples(samples, paths.samples_path)

    for filename in EVALUATION_SUMMARY_FILES:
        candidate = paths.evaluation_dir / filename
        if _remove_file(candidate):
            removed_evaluation_files.append(candidate)

    removed_report_assets_dir = _remove_tree(paths.report_assets_dir)

    return CleanupResult(
        condition_ids=target_conditions,
        removed_materialized_rows=removed_materialized_rows,
        removed_checkpoint_dirs=removed_checkpoint_dirs,
        removed_audio_dirs=removed_audio_dirs,
        removed_evaluation_files=removed_evaluation_files,
        removed_report_assets_dir=removed_report_assets_dir,
    )


def _save_config(config: Mapping[str, Any], path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True), encoding="utf-8")


def remove_condition(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    condition_id: str,
    samples_path: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    audio_base_dir: str | Path | None = None,
    evaluation_dir: str | Path | None = None,
    report_assets_dir: str | Path | None = None,
    bypass: bool = False,
) -> RemoveConditionResult:
    normalized_condition = str(condition_id).strip()
    if not normalized_condition or normalized_condition == "all":
        raise ValueError("`remove condition` requires one specific LoRA condition id.")

    config = _load_config(config_path)
    available = _resolve_lora_condition_ids(config)
    if normalized_condition not in available:
        raise ValueError(f"Unknown LoRA condition id: {normalized_condition}")

    paths = resolve_cleanup_paths(
        config_path=config_path,
        samples_path=samples_path,
        checkpoint_dir=checkpoint_dir,
        audio_base_dir=audio_base_dir,
        evaluation_dir=evaluation_dir,
        report_assets_dir=report_assets_dir,
    )
    _confirm_operation(
        action="remove a SpeechT5 condition",
        condition_ids=[normalized_condition],
        paths=paths,
        bypass=bypass,
        updates_config=True,
        removes_base_rows=True,
    )

    cleanup = cleanup_training_results(
        config_path=paths.config_path,
        condition_ids=[normalized_condition],
        samples_path=paths.samples_path,
        checkpoint_dir=paths.checkpoint_dir,
        audio_base_dir=paths.audio_base_dir,
        evaluation_dir=paths.evaluation_dir,
        report_assets_dir=paths.report_assets_dir,
        bypass=True,
    )

    removed_base_rows = 0
    if paths.samples_path.exists():
        samples = load_samples(paths.samples_path)
        mask = samples["condition"].eq(normalized_condition)
        removed_base_rows = int(mask.sum())
        if removed_base_rows:
            samples = samples.loc[~mask].copy()
            save_samples(samples, paths.samples_path)

    conditions = list(config.get("conditions", []))
    config["conditions"] = [
        condition
        for condition in conditions
        if not (isinstance(condition, Mapping) and str(condition.get("id", "")).strip() == normalized_condition)
    ]
    updated_config = len(config["conditions"]) != len(conditions)
    if not updated_config:
        raise ValueError(f"Condition `{normalized_condition}` was not found in config.")
    _save_config(config, paths.config_path)

    removed_run_matrix = bool(paths.run_matrix_path and _remove_file(paths.run_matrix_path))

    return RemoveConditionResult(
        cleanup=cleanup,
        removed_base_rows=removed_base_rows,
        removed_run_matrix=removed_run_matrix,
        updated_config=updated_config,
    )


def build_cleanup_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clear SpeechT5 LoRA training artifacts for selected conditions.")
    parser.add_argument("-c", "--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("-s", "--samples")
    parser.add_argument("-k", "--checkpoint-dir")
    parser.add_argument("-a", "--audio-base-dir")
    parser.add_argument("-e", "--evaluation-dir")
    parser.add_argument("-r", "--report-assets-dir")
    parser.add_argument(
        "-C",
        "--condition",
        action="append",
        required=True,
        help="LoRA condition id to clear. Repeat the flag to select multiple conditions, or pass `all`.",
    )
    parser.add_argument("-y", "--bypass", action="store_true", help="Skip confirmation prompt.")
    return parser


def build_remove_condition_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remove one SpeechT5 LoRA condition and its artifacts.")
    parser.add_argument("-c", "--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("-s", "--samples")
    parser.add_argument("-k", "--checkpoint-dir")
    parser.add_argument("-a", "--audio-base-dir")
    parser.add_argument("-e", "--evaluation-dir")
    parser.add_argument("-r", "--report-assets-dir")
    parser.add_argument(
        "-C",
        "--condition",
        action="append",
        required=True,
        help="Specific LoRA condition id to remove. This script accepts exactly one condition.",
    )
    parser.add_argument("-y", "--bypass", action="store_true", help="Skip confirmation prompt.")
    return parser
