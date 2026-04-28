"""Audio quality metrics and refresh workflow for processed WAV metadata."""

from __future__ import annotations

import argparse
import math
import shutil
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tcc_audio.cli_defaults import (
    DEFAULT_DATASET_INVENTORY_JSON,
    DEFAULT_DATASET_SLICE_ANALYSIS_JSON,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_EMBEDDINGS_INDEX_PATH,
    DEFAULT_EVALUATION_DIR,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_PROCESSED_METADATA_PATH,
    DEFAULT_REPORT_ASSETS_DIR,
    DEFAULT_RUN_MATRIX_PATH,
    DEFAULT_SAMPLES_PATH,
    DEFAULT_SPEAKER_SELECTION_PATH,
    load_cli_config,
    resolve_data_path,
    resolve_deliverable_path,
    resolve_evaluation_dir,
    resolve_samples_path,
)
from tcc_audio.io import ensure_parent_dir, read_csv
from tcc_audio.runtime import resolve_path

QUALITY_METRIC_COLUMNS = ["audio_rms", "audio_peak", "audio_silence_ratio"]
QUALITY_REFRESH_COMMAND = "python3 scripts/refresh_processed_audio_quality.py"
DEFAULT_RMS_MIN = 0.005
DEFAULT_RMS_MAX = 0.20
DEFAULT_PEAK_MIN = 0.02
DEFAULT_SILENCE_RATIO_MAX = 0.45
FRAME_DURATION_SECONDS = 0.02
FRAME_SILENCE_RMS_THRESHOLD = 0.001


@dataclass(frozen=True)
class AudioQualityThresholds:
    rms_min: float = DEFAULT_RMS_MIN
    rms_max: float = DEFAULT_RMS_MAX
    peak_min: float = DEFAULT_PEAK_MIN
    silence_ratio_max: float = DEFAULT_SILENCE_RATIO_MAX

    def validate(self) -> None:
        if self.rms_min < 0:
            raise ValueError("rms_min must be >= 0")
        if self.rms_max < 0:
            raise ValueError("rms_max must be >= 0")
        if self.rms_min > self.rms_max:
            raise ValueError("rms_min must be <= rms_max")
        if self.peak_min < 0:
            raise ValueError("peak_min must be >= 0")
        if self.silence_ratio_max < 0 or self.silence_ratio_max > 1:
            raise ValueError("silence_ratio_max must be between 0 and 1")


@dataclass(frozen=True)
class AudioQualitySummary:
    total_rows: int
    passing_rows: int
    excluded_rows: int
    excluded_by_low_rms: int
    excluded_by_high_rms: int
    excluded_by_low_peak: int
    excluded_by_high_silence_ratio: int


@dataclass(frozen=True)
class DownstreamArtifacts:
    files: list[Path]
    dirs: list[Path]


@dataclass(frozen=True)
class DownstreamInvalidationResult:
    removed_files: list[Path]
    removed_dirs: list[Path]


@dataclass(frozen=True)
class RefreshProcessedAudioQualityResult:
    metadata_path: Path
    row_count: int
    summary: AudioQualitySummary
    downstream_artifacts: DownstreamArtifacts
    invalidation: DownstreamInvalidationResult | None


def require_audio_quality_columns(frame: pd.DataFrame, *, label: str = "Processed metadata") -> None:
    missing = [column for column in QUALITY_METRIC_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{label} is missing audio quality columns: {', '.join(missing)}. "
            f"Run `{QUALITY_REFRESH_COMMAND}` on the processed metadata first."
        )


def _read_wav_mono_float32(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frame_count = handle.getnframes()
        payload = handle.readframes(frame_count)

    if sample_width != 2:
        raise ValueError(f"Processed WAV must use 16-bit PCM audio: {path}")

    audio = np.frombuffer(payload, dtype="<i2").astype(np.float32)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    audio /= 32768.0
    return audio, sample_rate


def compute_audio_quality_metrics(path: str | Path) -> dict[str, float]:
    audio, sample_rate = _read_wav_mono_float32(path)
    if audio.size == 0:
        raise ValueError(f"Processed WAV is empty: {path}")

    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float32)))
    peak = float(np.max(np.abs(audio)))
    frame_size = max(int(round(sample_rate * FRAME_DURATION_SECONDS)), 1)
    frame_count = max(int(math.ceil(audio.size / frame_size)), 1)
    padded = np.pad(audio, (0, frame_count * frame_size - audio.size))
    frames = padded.reshape(frame_count, frame_size)
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1, dtype=np.float32))
    silence_ratio = float(np.mean(frame_rms < FRAME_SILENCE_RMS_THRESHOLD))

    return {
        "audio_rms": rms,
        "audio_peak": peak,
        "audio_silence_ratio": silence_ratio,
    }


def append_audio_quality_metrics(
    metadata: pd.DataFrame,
    *,
    project_root: str | Path = ".",
) -> pd.DataFrame:
    if "audio_path" not in metadata.columns:
        raise ValueError("Processed metadata must contain audio_path.")

    enriched = metadata.copy()
    metrics_rows: list[dict[str, float]] = []
    for row_number, audio_path_value in enumerate(enriched["audio_path"].tolist(), start=2):
        audio_path = str(audio_path_value).strip()
        if not audio_path:
            raise ValueError(f"Line {row_number}: blank audio_path in processed metadata.")
        resolved = resolve_path(audio_path, project_root=project_root)
        if not resolved.exists():
            raise FileNotFoundError(f"Processed audio not found: {resolved}")
        metrics_rows.append(compute_audio_quality_metrics(resolved))

    metrics_frame = pd.DataFrame(metrics_rows, columns=QUALITY_METRIC_COLUMNS)
    for column in QUALITY_METRIC_COLUMNS:
        enriched[column] = metrics_frame[column].astype(float)
    return enriched


def _quality_failure_masks(
    metadata: pd.DataFrame,
    *,
    thresholds: AudioQualityThresholds,
) -> dict[str, pd.Series]:
    require_audio_quality_columns(metadata)
    thresholds.validate()
    return {
        "low_rms": metadata["audio_rms"].lt(thresholds.rms_min),
        "high_rms": metadata["audio_rms"].gt(thresholds.rms_max),
        "low_peak": metadata["audio_peak"].lt(thresholds.peak_min),
        "high_silence_ratio": metadata["audio_silence_ratio"].gt(thresholds.silence_ratio_max),
    }


def summarize_audio_quality(
    metadata: pd.DataFrame,
    *,
    thresholds: AudioQualityThresholds,
) -> AudioQualitySummary:
    masks = _quality_failure_masks(metadata, thresholds=thresholds)
    excluded_mask = pd.Series(False, index=metadata.index)
    for mask in masks.values():
        excluded_mask |= mask
    excluded_rows = int(excluded_mask.sum())
    total_rows = int(len(metadata))
    return AudioQualitySummary(
        total_rows=total_rows,
        passing_rows=total_rows - excluded_rows,
        excluded_rows=excluded_rows,
        excluded_by_low_rms=int(masks["low_rms"].sum()),
        excluded_by_high_rms=int(masks["high_rms"].sum()),
        excluded_by_low_peak=int(masks["low_peak"].sum()),
        excluded_by_high_silence_ratio=int(masks["high_silence_ratio"].sum()),
    )


def _artifacts_root(run_matrix_path: Path, evaluation_dir: Path) -> Path:
    run_matrix_parent = run_matrix_path.parent
    if run_matrix_parent not in {Path(""), Path(".")}:
        return run_matrix_parent
    evaluation_parent = evaluation_dir.parent
    if evaluation_parent not in {Path(""), Path(".")}:
        return evaluation_parent
    return Path("artifacts")


def resolve_downstream_artifacts(config_path: str | Path | None = None) -> DownstreamArtifacts:
    _, config = load_cli_config(config_path)
    speaker_selection_path = resolve_data_path(config, "speaker_selection_path", DEFAULT_SPEAKER_SELECTION_PATH)
    manifest_path = resolve_data_path(config, "manifest_path", DEFAULT_MANIFEST_PATH)
    embeddings_index_path = resolve_data_path(config, "speaker_embeddings_index", DEFAULT_EMBEDDINGS_INDEX_PATH)
    run_matrix_path = resolve_deliverable_path(config, "run_matrix", DEFAULT_RUN_MATRIX_PATH)
    samples_path = resolve_samples_path(config) if config else DEFAULT_SAMPLES_PATH
    evaluation_dir = resolve_evaluation_dir(config) if config else DEFAULT_EVALUATION_DIR
    report_assets_dir = resolve_deliverable_path(config, "report_assets_dir", DEFAULT_REPORT_ASSETS_DIR)
    artifacts_root = _artifacts_root(run_matrix_path, evaluation_dir)
    checkpoint_dir = artifacts_root / "checkpoints" / "lora"
    audio_base_dir = artifacts_root / "audio"

    files = [
        speaker_selection_path,
        manifest_path,
        embeddings_index_path,
        run_matrix_path,
        samples_path,
        artifacts_root / DEFAULT_DATASET_INVENTORY_JSON.name,
        artifacts_root / DEFAULT_DATASET_SLICE_ANALYSIS_JSON.name,
    ]
    dirs = [
        embeddings_index_path.parent if embeddings_index_path.parent != Path("") else DEFAULT_EMBEDDINGS_DIR,
        checkpoint_dir,
        audio_base_dir,
        evaluation_dir,
        report_assets_dir,
    ]
    return DownstreamArtifacts(files=files, dirs=dirs)


def _confirm_invalidation(paths: DownstreamArtifacts, *, bypass: bool) -> None:
    if bypass:
        return

    print("About to invalidate downstream artifacts generated after processed-audio quality refresh.")
    for path in paths.files:
        print(f"file: {path}")
    for path in paths.dirs:
        print(f"dir: {path}")
    response = input("Type 'yes' to continue: ").strip().lower()
    if response != "yes":
        raise SystemExit("Aborted by user.")


def _remove_file(path: Path) -> bool:
    if not path.exists():
        return False
    path.unlink()
    return True


def _remove_tree(path: Path) -> bool:
    if not path.exists():
        return False
    shutil.rmtree(path)
    return True


def invalidate_downstream_artifacts(
    *,
    config_path: str | Path | None = None,
    bypass: bool = False,
) -> DownstreamInvalidationResult:
    targets = resolve_downstream_artifacts(config_path=config_path)
    _confirm_invalidation(targets, bypass=bypass)

    removed_files: list[Path] = []
    for path in targets.files:
        if _remove_file(path):
            removed_files.append(path)

    removed_dirs: list[Path] = []
    for path in targets.dirs:
        if _remove_tree(path):
            removed_dirs.append(path)

    return DownstreamInvalidationResult(removed_files=removed_files, removed_dirs=removed_dirs)


def refresh_processed_audio_quality(
    metadata_path: str | Path,
    *,
    project_root: str | Path = ".",
    thresholds: AudioQualityThresholds | None = None,
    config_path: str | Path | None = None,
    invalidate_downstream: bool = False,
    bypass: bool = False,
) -> RefreshProcessedAudioQualityResult:
    resolved_thresholds = thresholds or AudioQualityThresholds()
    resolved_thresholds.validate()
    refreshed = append_audio_quality_metrics(read_csv(metadata_path), project_root=project_root)
    ensure_parent_dir(metadata_path)
    refreshed.to_csv(metadata_path, index=False)

    downstream_artifacts = resolve_downstream_artifacts(config_path=config_path)
    invalidation = None
    if invalidate_downstream:
        invalidation = invalidate_downstream_artifacts(config_path=config_path, bypass=bypass)

    return RefreshProcessedAudioQualityResult(
        metadata_path=Path(metadata_path),
        row_count=len(refreshed),
        summary=summarize_audio_quality(refreshed, thresholds=resolved_thresholds),
        downstream_artifacts=downstream_artifacts,
        invalidation=invalidation,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh audio quality metrics on processed WAV metadata without rerunning ffmpeg."
    )
    parser.add_argument("-c", "--config", help="Optional experiment config used to resolve downstream artifact paths.")
    parser.add_argument(
        "-m",
        "--metadata",
        help=f"Processed metadata CSV. Defaults to {DEFAULT_PROCESSED_METADATA_PATH}.",
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--rms-min", type=float, default=DEFAULT_RMS_MIN)
    parser.add_argument("--rms-max", type=float, default=DEFAULT_RMS_MAX)
    parser.add_argument("--peak-min", type=float, default=DEFAULT_PEAK_MIN)
    parser.add_argument("--silence-ratio-max", type=float, default=DEFAULT_SILENCE_RATIO_MAX)
    parser.add_argument("--invalidate-downstream", action="store_true")
    parser.add_argument("-y", "--bypass", action="store_true", help="Skip confirmation prompt when invalidating downstream artifacts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config_path, config = load_cli_config(args.config)
    metadata_path = args.metadata or str(
        resolve_data_path(config, "processed_metadata_path", DEFAULT_PROCESSED_METADATA_PATH)
    )
    result = refresh_processed_audio_quality(
        metadata_path=metadata_path,
        project_root=args.project_root,
        thresholds=AudioQualityThresholds(
            rms_min=args.rms_min,
            rms_max=args.rms_max,
            peak_min=args.peak_min,
            silence_ratio_max=args.silence_ratio_max,
        ),
        config_path=config_path,
        invalidate_downstream=args.invalidate_downstream,
        bypass=args.bypass,
    )
    print(f"Updated {result.row_count} processed metadata rows in {result.metadata_path}")
    print(
        "Audio quality summary: "
        f"passing={result.summary.passing_rows} excluded={result.summary.excluded_rows} "
        f"low_rms={result.summary.excluded_by_low_rms} high_rms={result.summary.excluded_by_high_rms} "
        f"low_peak={result.summary.excluded_by_low_peak} high_silence_ratio={result.summary.excluded_by_high_silence_ratio}"
    )
    if result.invalidation is None:
        print("Downstream artifacts to refresh next:")
        for path in result.downstream_artifacts.files:
            print(f"- file {path}")
        for path in result.downstream_artifacts.dirs:
            print(f"- dir {path}")
    else:
        print(
            "Invalidated downstream artifacts: "
            f"removed_files={len(result.invalidation.removed_files)} "
            f"removed_dirs={len(result.invalidation.removed_dirs)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
