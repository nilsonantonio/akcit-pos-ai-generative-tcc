"""Training-slice analysis for a final dataset recorte."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from tcc_audio.cli_defaults import DEFAULT_RAW_METADATA_PATH
from tcc_audio.dataset_filtering import (
    DEFAULT_MAX_AUDIO_DURATION_S,
    DEFAULT_MAX_CLIPS_PER_SPEAKER,
    DEFAULT_MIN_AUDIO_DURATION_S,
    DEFAULT_MIN_CLIPS_PER_SPEAKER,
    DEFAULT_MIN_DURATION_PER_SPEAKER_S,
    apply_training_slice_filters,
)
from tcc_audio.io import ensure_parent_dir, read_csv


def _to_minutes(seconds: float) -> float:
    return float(seconds) / 60.0


def _frame_totals(frame: pd.DataFrame) -> dict[str, Any]:
    total_duration_s = float(frame["duration_s"].sum()) if not frame.empty else 0.0
    return {
        "total_clips": int(len(frame)),
        "total_duration_s": total_duration_s,
        "total_minutes": _to_minutes(total_duration_s),
        "unique_speakers": int(frame["source_speaker_id"].nunique()) if not frame.empty else 0,
    }


def _grouped_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []

    grouped = (
        frame.groupby(["locale", "variant", "gender"], dropna=False)
        .agg(
            clip_count=("duration_s", "count"),
            total_duration_s=("duration_s", "sum"),
            unique_speakers=("source_speaker_id", "nunique"),
        )
        .reset_index()
        .sort_values(["locale", "variant", "gender"])
    )

    rows: list[dict[str, Any]] = []
    for _, row in grouped.iterrows():
        total_duration_s = float(row["total_duration_s"])
        rows.append(
            {
                "locale": str(row["locale"]),
                "variant": str(row["variant"]),
                "gender": str(row["gender"]),
                "clip_count": int(row["clip_count"]),
                "total_duration_s": total_duration_s,
                "total_minutes": _to_minutes(total_duration_s),
                "unique_speakers": int(row["unique_speakers"]),
            }
        )
    return rows


def _distribution(values: pd.Series) -> dict[str, float | None]:
    if values.empty:
        return {"min": None, "p25": None, "median": None, "p75": None, "p90": None, "max": None}
    return {
        "min": float(values.min()),
        "p25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "p75": float(values.quantile(0.75)),
        "p90": float(values.quantile(0.90)),
        "max": float(values.max()),
    }


def analyze_dataset_slice(
    *,
    metadata_path: str | Path = DEFAULT_RAW_METADATA_PATH,
    min_audio_duration_s: float = DEFAULT_MIN_AUDIO_DURATION_S,
    max_audio_duration_s: float = DEFAULT_MAX_AUDIO_DURATION_S,
    min_clips_per_speaker: int = DEFAULT_MIN_CLIPS_PER_SPEAKER,
    max_clips_per_speaker: int = DEFAULT_MAX_CLIPS_PER_SPEAKER,
    min_duration_per_speaker_s: float = DEFAULT_MIN_DURATION_PER_SPEAKER_S,
) -> dict[str, Any]:
    metadata = read_csv(metadata_path)
    required = {"source_speaker_id", "duration_s", "locale", "variant", "gender"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"Metadata must contain columns: {', '.join(missing)}")

    filtered = apply_training_slice_filters(
        metadata,
        min_audio_duration_s=min_audio_duration_s,
        max_audio_duration_s=max_audio_duration_s,
        min_clips_per_speaker=min_clips_per_speaker,
        max_clips_per_speaker=max_clips_per_speaker,
        min_duration_per_speaker_s=min_duration_per_speaker_s,
    )

    speaker_stats = filtered.speaker_stats_final.copy()
    analysis = {
        "filters": {
            "metadata_path": str(metadata_path),
            "min_audio_duration_s": float(min_audio_duration_s),
            "max_audio_duration_s": float(max_audio_duration_s),
            "min_clips_per_speaker": int(min_clips_per_speaker),
            "max_clips_per_speaker": int(max_clips_per_speaker),
            "min_duration_per_speaker_s": float(min_duration_per_speaker_s),
        },
        "steps": {
            "clip_filter": _frame_totals(filtered.clip_filtered),
            "speaker_eligibility": {
                "candidate_speakers": int(filtered.candidate_speakers),
                "eligible_speakers": int(filtered.eligible_speakers),
                "excluded_speakers": int(filtered.excluded_speakers),
                "excluded_by_min_clips": int(filtered.excluded_by_min_clips),
                "excluded_by_min_duration": int(filtered.excluded_by_min_duration),
            },
            "clip_cap": {
                "max_clips_per_speaker": int(max_clips_per_speaker),
                "capped_speakers": int(filtered.capped_speakers),
            },
        },
        "final_dataset": _frame_totals(filtered.final_rows),
        "grouped_by_locale_variant_gender": _grouped_summary(filtered.final_rows),
        "speaker_distribution": {
            "clips_per_speaker": _distribution(speaker_stats["clip_count"]) if not speaker_stats.empty else _distribution(pd.Series(dtype=float)),
            "duration_per_speaker_s": _distribution(speaker_stats["total_duration_s"]) if not speaker_stats.empty else _distribution(pd.Series(dtype=float)),
        },
        "speaker_stats": [
            {
                "source_speaker_id": str(row["source_speaker_id"]),
                "clip_count": int(row["clip_count"]),
                "total_duration_s": float(row["total_duration_s"]),
                "total_minutes": _to_minutes(float(row["total_duration_s"])),
                "locale": str(row.get("locale", "unknown")),
                "variant": str(row.get("variant", "unknown")),
                "gender": str(row.get("gender", "unknown")),
            }
            for _, row in speaker_stats.sort_values(["clip_count", "total_duration_s", "source_speaker_id"], ascending=[False, False, True]).iterrows()
        ],
        "warnings": filtered.warnings,
    }
    return analysis


def _render_distribution_line(label: str, distribution: Mapping[str, Any]) -> str:
    return (
        f"- {label} min={distribution['min']} p25={distribution['p25']} "
        f"median={distribution['median']} p75={distribution['p75']} p90={distribution['p90']} max={distribution['max']}"
    )


def render_dataset_slice_analysis(analysis: Mapping[str, Any]) -> str:
    filters = analysis["filters"]
    clip_filter = analysis["steps"]["clip_filter"]
    speaker_eligibility = analysis["steps"]["speaker_eligibility"]
    clip_cap = analysis["steps"]["clip_cap"]
    final_dataset = analysis["final_dataset"]
    grouped = analysis["grouped_by_locale_variant_gender"]
    distribution = analysis["speaker_distribution"]

    lines = [
        "DATASET FINAL SLICE",
        "FILTERS",
        (
            f"- min_audio_duration_s={filters['min_audio_duration_s']} "
            f"max_audio_duration_s={filters['max_audio_duration_s']} "
            f"min_clips_per_speaker={filters['min_clips_per_speaker']} "
            f"max_clips_per_speaker={filters['max_clips_per_speaker']} "
            f"min_duration_per_speaker_s={filters['min_duration_per_speaker_s']}"
        ),
        "STEP 1 CLIP FILTER",
        (
            f"- total_clips={clip_filter['total_clips']} total_minutes={clip_filter['total_minutes']:.3f} "
            f"unique_speakers={clip_filter['unique_speakers']}"
        ),
        "STEP 2 SPEAKER ELIGIBILITY",
        (
            f"- candidate_speakers={speaker_eligibility['candidate_speakers']} "
            f"eligible_speakers={speaker_eligibility['eligible_speakers']} "
            f"excluded_speakers={speaker_eligibility['excluded_speakers']} "
            f"excluded_by_min_clips={speaker_eligibility['excluded_by_min_clips']} "
            f"excluded_by_min_duration={speaker_eligibility['excluded_by_min_duration']}"
        ),
        "STEP 3 CLIP CAP",
        f"- max_clips_per_speaker={clip_cap['max_clips_per_speaker']} capped_speakers={clip_cap['capped_speakers']}",
        "FINAL DATASET",
        (
            f"- total_clips={final_dataset['total_clips']} total_minutes={final_dataset['total_minutes']:.3f} "
            f"unique_speakers={final_dataset['unique_speakers']}"
        ),
        "GROUPED BY locale / variant / gender",
    ]

    for row in grouped:
        lines.append(
            f"- locale={row['locale']} variant={row['variant']} gender={row['gender']} "
            f"clip_count={row['clip_count']} total_minutes={row['total_minutes']:.3f} unique_speakers={row['unique_speakers']}"
        )

    lines.extend(
        [
            "SPEAKER DISTRIBUTION",
            _render_distribution_line("clips_per_speaker", distribution["clips_per_speaker"]),
            _render_distribution_line("duration_per_speaker_s", distribution["duration_per_speaker_s"]),
        ]
    )

    warnings = list(analysis.get("warnings", []))
    if warnings:
        lines.append("WARNINGS")
        for warning in warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate the final dataset recorte using the training-slice rules.")
    parser.add_argument("-m", "--metadata", default=str(DEFAULT_RAW_METADATA_PATH), help="Path to the raw metadata CSV.")
    parser.add_argument("--min-audio-duration-s", type=float, default=DEFAULT_MIN_AUDIO_DURATION_S)
    parser.add_argument("--max-audio-duration-s", type=float, default=DEFAULT_MAX_AUDIO_DURATION_S)
    parser.add_argument("--min-clips-per-speaker", type=int, default=DEFAULT_MIN_CLIPS_PER_SPEAKER)
    parser.add_argument("--max-clips-per-speaker", type=int, default=DEFAULT_MAX_CLIPS_PER_SPEAKER)
    parser.add_argument("--min-duration-per-speaker-s", type=float, default=DEFAULT_MIN_DURATION_PER_SPEAKER_S)
    parser.add_argument("-j", "--json-out", help="Optional path to persist the analysis JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        analysis = analyze_dataset_slice(
            metadata_path=args.metadata,
            min_audio_duration_s=args.min_audio_duration_s,
            max_audio_duration_s=args.max_audio_duration_s,
            min_clips_per_speaker=args.min_clips_per_speaker,
            max_clips_per_speaker=args.max_clips_per_speaker,
            min_duration_per_speaker_s=args.min_duration_per_speaker_s,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(render_dataset_slice_analysis(analysis), end="")
    if args.json_out:
        ensure_parent_dir(args.json_out)
        Path(args.json_out).write_text(json.dumps(analysis, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        print(f"Wrote JSON analysis to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
