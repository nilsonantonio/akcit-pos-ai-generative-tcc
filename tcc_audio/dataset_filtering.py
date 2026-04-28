"""Shared dataset filtering for training-slice decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from tcc_audio.processed_audio_quality import (
    DEFAULT_PEAK_MIN,
    DEFAULT_RMS_MAX,
    DEFAULT_RMS_MIN,
    DEFAULT_SILENCE_RATIO_MAX,
    AudioQualityThresholds,
    QUALITY_METRIC_COLUMNS,
    require_audio_quality_columns,
)

DEFAULT_MIN_CLIPS_PER_SPEAKER = 20
DEFAULT_MAX_CLIPS_PER_SPEAKER = 120
DEFAULT_MIN_DURATION_PER_SPEAKER_S = 60.0
DEFAULT_MIN_AUDIO_DURATION_S = 2.0
DEFAULT_MAX_AUDIO_DURATION_S = 8.0


@dataclass
class TrainingSliceResult:
    prepared_rows: pd.DataFrame
    quality_filtered: pd.DataFrame
    clip_filtered: pd.DataFrame
    final_rows: pd.DataFrame
    speaker_stats_before_cap: pd.DataFrame
    speaker_stats_final: pd.DataFrame
    warnings: list[str]
    invalid_duration_rows: int
    negative_duration_rows: int
    invalid_quality_rows: int
    excluded_by_low_rms: int
    excluded_by_high_rms: int
    excluded_by_low_peak: int
    excluded_by_high_silence_ratio: int
    capped_speakers: int
    candidate_speakers: int
    eligible_speakers: int
    excluded_speakers: int
    excluded_by_min_clips: int
    excluded_by_min_duration: int


def _normalize_text_column(frame: pd.DataFrame, column: str, default: str = "unknown") -> pd.Series:
    return frame[column].fillna("").astype(str).str.strip().replace("", default)


def _mode_or_unknown(values: pd.Series) -> str:
    normalized = values.fillna("").astype(str).str.strip().replace("", "unknown")
    mode = normalized.mode()
    return str(mode.iat[0]) if not mode.empty else "unknown"


def _validate_filter_bounds(
    *,
    min_audio_duration_s: float,
    max_audio_duration_s: float,
    min_clips_per_speaker: int,
    max_clips_per_speaker: int,
    min_duration_per_speaker_s: float,
    quality_thresholds: AudioQualityThresholds,
) -> None:
    if min_audio_duration_s < 0:
        raise ValueError("min_audio_duration_s must be >= 0")
    if max_audio_duration_s < 0:
        raise ValueError("max_audio_duration_s must be >= 0")
    if min_audio_duration_s > max_audio_duration_s:
        raise ValueError("min_audio_duration_s must be <= max_audio_duration_s")
    if min_clips_per_speaker <= 0:
        raise ValueError("min_clips_per_speaker must be > 0")
    if max_clips_per_speaker <= 0:
        raise ValueError("max_clips_per_speaker must be > 0")
    if min_clips_per_speaker > max_clips_per_speaker:
        raise ValueError("min_clips_per_speaker must be <= max_clips_per_speaker")
    if min_duration_per_speaker_s < 0:
        raise ValueError("min_duration_per_speaker_s must be >= 0")
    quality_thresholds.validate()


def _require_columns(metadata: pd.DataFrame, required: list[str]) -> None:
    missing = [column for column in required if column not in metadata.columns]
    if missing:
        raise ValueError(f"Missing dataset filter input columns: {', '.join(missing)}")


def _prepare_metadata(metadata: pd.DataFrame) -> tuple[pd.DataFrame, list[str], int, int, int]:
    prepared = metadata.copy()
    warnings: list[str] = []
    prepared["_row_order"] = range(len(prepared))
    prepared["duration_s"] = pd.to_numeric(prepared["duration_s"], errors="coerce")

    invalid_duration_rows = int(prepared["duration_s"].isna().sum())
    if invalid_duration_rows:
        warnings.append(f"Discarded {invalid_duration_rows} rows with invalid duration_s.")

    negative_duration_rows = int(prepared["duration_s"].lt(0).fillna(False).sum())
    if negative_duration_rows:
        warnings.append(f"Discarded {negative_duration_rows} rows with negative duration_s.")

    require_audio_quality_columns(prepared)
    for column in QUALITY_METRIC_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    invalid_quality_rows = int(prepared[QUALITY_METRIC_COLUMNS].isna().any(axis=1).sum())
    if invalid_quality_rows:
        warnings.append(f"Discarded {invalid_quality_rows} rows with invalid audio quality metrics.")

    prepared = prepared.dropna(subset=["duration_s"])
    prepared = prepared[prepared["duration_s"] >= 0].copy()
    prepared = prepared.dropna(subset=QUALITY_METRIC_COLUMNS)

    for column in ["source_speaker_id", "gender", "locale", "variant"]:
        if column in prepared.columns:
            prepared[column] = _normalize_text_column(prepared, column)

    return prepared, warnings, invalid_duration_rows, negative_duration_rows, invalid_quality_rows


def _speaker_stats(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby("source_speaker_id", dropna=False)
        .agg(
            clip_count=("duration_s", "count"),
            total_duration_s=("duration_s", "sum"),
        )
        .reset_index()
        .sort_values(["clip_count", "total_duration_s", "source_speaker_id"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    return grouped


def _clip_cap_sort_columns(frame: pd.DataFrame) -> list[str]:
    columns = ["source_speaker_id", "duration_s"]
    if "utterance_id" in frame.columns:
        columns.append("utterance_id")
    elif "audio_path" in frame.columns:
        columns.append("audio_path")
    else:
        columns.append("_row_order")
    return columns


def apply_training_slice_filters(
    metadata: pd.DataFrame,
    *,
    min_audio_duration_s: float = DEFAULT_MIN_AUDIO_DURATION_S,
    max_audio_duration_s: float = DEFAULT_MAX_AUDIO_DURATION_S,
    min_clips_per_speaker: int = DEFAULT_MIN_CLIPS_PER_SPEAKER,
    max_clips_per_speaker: int = DEFAULT_MAX_CLIPS_PER_SPEAKER,
    min_duration_per_speaker_s: float = DEFAULT_MIN_DURATION_PER_SPEAKER_S,
    rms_min: float = DEFAULT_RMS_MIN,
    rms_max: float = DEFAULT_RMS_MAX,
    peak_min: float = DEFAULT_PEAK_MIN,
    silence_ratio_max: float = DEFAULT_SILENCE_RATIO_MAX,
) -> TrainingSliceResult:
    _require_columns(metadata, ["source_speaker_id", "duration_s"])
    quality_thresholds = AudioQualityThresholds(
        rms_min=rms_min,
        rms_max=rms_max,
        peak_min=peak_min,
        silence_ratio_max=silence_ratio_max,
    )
    _validate_filter_bounds(
        min_audio_duration_s=min_audio_duration_s,
        max_audio_duration_s=max_audio_duration_s,
        min_clips_per_speaker=min_clips_per_speaker,
        max_clips_per_speaker=max_clips_per_speaker,
        min_duration_per_speaker_s=min_duration_per_speaker_s,
        quality_thresholds=quality_thresholds,
    )

    prepared, warnings, invalid_duration_rows, negative_duration_rows, invalid_quality_rows = _prepare_metadata(metadata)
    low_rms_mask = prepared["audio_rms"].lt(quality_thresholds.rms_min)
    high_rms_mask = prepared["audio_rms"].gt(quality_thresholds.rms_max)
    low_peak_mask = prepared["audio_peak"].lt(quality_thresholds.peak_min)
    high_silence_mask = prepared["audio_silence_ratio"].gt(quality_thresholds.silence_ratio_max)
    quality_mask = ~(low_rms_mask | high_rms_mask | low_peak_mask | high_silence_mask)
    quality_filtered = prepared[quality_mask].copy()

    clip_filtered = quality_filtered[
        quality_filtered["duration_s"].ge(min_audio_duration_s) & quality_filtered["duration_s"].le(max_audio_duration_s)
    ].copy()

    speaker_stats_before_cap = _speaker_stats(clip_filtered) if not clip_filtered.empty else pd.DataFrame(
        columns=["source_speaker_id", "clip_count", "total_duration_s"]
    )
    candidate_speakers = int(len(speaker_stats_before_cap))
    eligible_mask = (
        speaker_stats_before_cap["clip_count"].ge(min_clips_per_speaker)
        & speaker_stats_before_cap["total_duration_s"].ge(min_duration_per_speaker_s)
    )
    eligible_speakers_frame = speaker_stats_before_cap[eligible_mask].copy()
    eligible_speaker_ids = set(eligible_speakers_frame["source_speaker_id"].tolist())
    excluded_by_min_clips = int(speaker_stats_before_cap["clip_count"].lt(min_clips_per_speaker).sum())
    excluded_by_min_duration = int(speaker_stats_before_cap["total_duration_s"].lt(min_duration_per_speaker_s).sum())
    eligible_speakers = int(len(eligible_speakers_frame))
    excluded_speakers = int(candidate_speakers - eligible_speakers)
    capped_speakers = int(eligible_speakers_frame["clip_count"].gt(max_clips_per_speaker).sum()) if eligible_speakers else 0

    eligible_rows = clip_filtered[clip_filtered["source_speaker_id"].isin(eligible_speaker_ids)].copy()
    if eligible_rows.empty:
        final_rows = eligible_rows
    else:
        sort_columns = _clip_cap_sort_columns(eligible_rows)
        ascending = [True, False] + [True] * (len(sort_columns) - 2)
        final_rows = (
            eligible_rows.sort_values(sort_columns, ascending=ascending)
            .groupby("source_speaker_id", group_keys=False)
            .head(max_clips_per_speaker)
            .copy()
        )

    speaker_stats_final = _speaker_stats(final_rows) if not final_rows.empty else pd.DataFrame(
        columns=["source_speaker_id", "clip_count", "total_duration_s"]
    )
    if not speaker_stats_final.empty:
        extra_rows: list[dict[str, Any]] = []
        for speaker_id, speaker_rows in final_rows.groupby("source_speaker_id", dropna=False):
            row: dict[str, Any] = {"source_speaker_id": str(speaker_id)}
            if "locale" in speaker_rows.columns:
                row["locale"] = _mode_or_unknown(speaker_rows["locale"])
            if "variant" in speaker_rows.columns:
                row["variant"] = _mode_or_unknown(speaker_rows["variant"])
            if "gender" in speaker_rows.columns:
                row["gender"] = _mode_or_unknown(speaker_rows["gender"])
            extra_rows.append(row)
        extras = pd.DataFrame(extra_rows)
        speaker_stats_final = speaker_stats_final.merge(extras, on="source_speaker_id", how="left")

    return TrainingSliceResult(
        prepared_rows=prepared,
        quality_filtered=quality_filtered,
        clip_filtered=clip_filtered,
        final_rows=final_rows,
        speaker_stats_before_cap=speaker_stats_before_cap,
        speaker_stats_final=speaker_stats_final,
        warnings=warnings,
        invalid_duration_rows=invalid_duration_rows,
        negative_duration_rows=negative_duration_rows,
        invalid_quality_rows=invalid_quality_rows,
        excluded_by_low_rms=int(low_rms_mask.sum()),
        excluded_by_high_rms=int(high_rms_mask.sum()),
        excluded_by_low_peak=int(low_peak_mask.sum()),
        excluded_by_high_silence_ratio=int(high_silence_mask.sum()),
        capped_speakers=capped_speakers,
        candidate_speakers=candidate_speakers,
        eligible_speakers=eligible_speakers,
        excluded_speakers=excluded_speakers,
        excluded_by_min_clips=excluded_by_min_clips,
        excluded_by_min_duration=excluded_by_min_duration,
    )
