"""Duration-filtered dataset analysis for raw Common Voice metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from tcc_audio.cli_defaults import DEFAULT_RAW_METADATA_PATH
from tcc_audio.io import ensure_parent_dir, read_csv

DEFAULT_DURATION_BIN_EDGES = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, float("inf")]
DEFAULT_DURATION_BIN_LABELS = ["<=2", "2-4", "4-6", "6-8", "8-10", "10-15", "15+"]
DEFAULT_SPEAKER_THRESHOLDS_MINUTES = [1, 5, 10, 20, 30]


def _normalize_text_column(frame: pd.DataFrame, column: str, default: str = "unknown") -> pd.Series:
    return frame[column].fillna("").astype(str).str.strip().replace("", default)


def _to_minutes(seconds: float) -> float:
    return float(seconds) / 60.0


def _validate_filter_bounds(min_seconds: float | None, max_seconds: float | None) -> tuple[float | None, float | None]:
    if min_seconds is not None and min_seconds < 0:
        raise ValueError("min_seconds must be >= 0")
    if max_seconds is not None and max_seconds < 0:
        raise ValueError("max_seconds must be >= 0")
    if min_seconds is not None and max_seconds is not None and min_seconds > max_seconds:
        raise ValueError("min_seconds must be <= max_seconds")
    return min_seconds, max_seconds


def _load_metadata_frame(metadata_path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    metadata = read_csv(metadata_path)
    required = {"source_speaker_id", "gender", "duration_s", "locale", "variant"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"Metadata must contain columns: {', '.join(missing)}")

    warnings: list[str] = []
    metadata = metadata.copy()
    metadata["duration_s"] = pd.to_numeric(metadata["duration_s"], errors="coerce")

    invalid_numeric = int(metadata["duration_s"].isna().sum())
    if invalid_numeric:
        warnings.append(f"Discarded {invalid_numeric} rows with invalid duration_s.")

    negative_duration = int(metadata["duration_s"].lt(0).fillna(False).sum())
    if negative_duration:
        warnings.append(f"Discarded {negative_duration} rows with negative duration_s.")

    metadata = metadata.dropna(subset=["duration_s"])
    metadata = metadata[metadata["duration_s"] >= 0].copy()

    for column in ["source_speaker_id", "gender", "locale", "variant"]:
        metadata[column] = _normalize_text_column(metadata, column)
    return metadata, warnings


def _apply_duration_filter(frame: pd.DataFrame, min_seconds: float | None, max_seconds: float | None) -> pd.DataFrame:
    filtered = frame
    if min_seconds is not None:
        filtered = filtered[filtered["duration_s"] >= min_seconds]
    if max_seconds is not None:
        filtered = filtered[filtered["duration_s"] <= max_seconds]
    return filtered.copy()


def _grouped_duration_summary(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    if frame.empty:
        return []

    grouped = (
        frame.groupby(columns, dropna=False)
        .agg(
            clip_count=("duration_s", "count"),
            total_duration_s=("duration_s", "sum"),
            unique_speakers=("source_speaker_id", "nunique"),
        )
        .reset_index()
    )

    rows: list[dict[str, Any]] = []
    for _, row in grouped.sort_values(columns).iterrows():
        payload = {column: str(row[column]) for column in columns}
        payload.update(
            {
                "clip_count": int(row["clip_count"]),
                "total_duration_s": float(row["total_duration_s"]),
                "total_minutes": _to_minutes(float(row["total_duration_s"])),
                "unique_speakers": int(row["unique_speakers"]),
            }
        )
        rows.append(payload)
    return rows


def _duration_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "clip_count": 0,
            "min_s": None,
            "p25_s": None,
            "median_s": None,
            "mean_s": None,
            "p75_s": None,
            "p90_s": None,
            "p95_s": None,
            "max_s": None,
        }

    durations = frame["duration_s"]
    return {
        "clip_count": int(len(frame)),
        "min_s": float(durations.min()),
        "p25_s": float(durations.quantile(0.25)),
        "median_s": float(durations.median()),
        "mean_s": float(durations.mean()),
        "p75_s": float(durations.quantile(0.75)),
        "p90_s": float(durations.quantile(0.90)),
        "p95_s": float(durations.quantile(0.95)),
        "max_s": float(durations.max()),
    }


def _duration_bins(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return [
            {"duration_bin": label, "clip_count": 0, "total_duration_s": 0.0, "total_minutes": 0.0, "unique_speakers": 0}
            for label in DEFAULT_DURATION_BIN_LABELS
        ]

    bucketed = frame.copy()
    bucketed["duration_bin"] = pd.cut(
        bucketed["duration_s"],
        bins=DEFAULT_DURATION_BIN_EDGES,
        labels=DEFAULT_DURATION_BIN_LABELS,
        include_lowest=True,
        right=True,
    )

    grouped = (
        bucketed.groupby("duration_bin", observed=False)
        .agg(
            clip_count=("duration_s", "count"),
            total_duration_s=("duration_s", "sum"),
            unique_speakers=("source_speaker_id", "nunique"),
        )
        .reset_index()
    )

    rows: list[dict[str, Any]] = []
    for _, row in grouped.iterrows():
        total_duration = float(row["total_duration_s"]) if pd.notna(row["total_duration_s"]) else 0.0
        rows.append(
            {
                "duration_bin": str(row["duration_bin"]),
                "clip_count": int(row["clip_count"]),
                "total_duration_s": total_duration,
                "total_minutes": _to_minutes(total_duration),
                "unique_speakers": int(row["unique_speakers"]),
            }
        )
    return rows


def _speaker_thresholds(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return [
            {"threshold_minutes": minutes, "speaker_count": 0}
            for minutes in DEFAULT_SPEAKER_THRESHOLDS_MINUTES
        ]

    speaker_duration = frame.groupby("source_speaker_id", dropna=False)["duration_s"].sum()
    rows: list[dict[str, Any]] = []
    for minutes in DEFAULT_SPEAKER_THRESHOLDS_MINUTES:
        rows.append(
            {
                "threshold_minutes": int(minutes),
                "speaker_count": int((speaker_duration >= (minutes * 60)).sum()),
            }
        )
    return rows


def _speaker_concentration(frame: pd.DataFrame, top_speakers: int) -> dict[str, Any]:
    if top_speakers <= 0:
        raise ValueError("top_speakers must be > 0")

    if frame.empty:
        return {
            "top_n": int(top_speakers),
            "top_n_clip_share_pct": 0.0,
            "top_n_duration_share_pct": 0.0,
            "speakers": [],
        }

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

    total_clips = int(len(frame))
    total_duration_s = float(frame["duration_s"].sum())
    leaders = grouped.head(top_speakers).copy()
    rows: list[dict[str, Any]] = []
    for _, row in leaders.iterrows():
        duration_s = float(row["total_duration_s"])
        rows.append(
            {
                "source_speaker_id": str(row["source_speaker_id"]),
                "clip_count": int(row["clip_count"]),
                "total_duration_s": duration_s,
                "total_minutes": _to_minutes(duration_s),
                "clip_share_pct": float((row["clip_count"] / total_clips) * 100.0) if total_clips else 0.0,
                "duration_share_pct": float((duration_s / total_duration_s) * 100.0) if total_duration_s else 0.0,
            }
        )

    top_clip_count = int(leaders["clip_count"].sum()) if not leaders.empty else 0
    top_duration = float(leaders["total_duration_s"].sum()) if not leaders.empty else 0.0
    return {
        "top_n": int(top_speakers),
        "top_n_clip_share_pct": float((top_clip_count / total_clips) * 100.0) if total_clips else 0.0,
        "top_n_duration_share_pct": float((top_duration / total_duration_s) * 100.0) if total_duration_s else 0.0,
        "speakers": rows,
    }


def analyze_dataset_slice(
    *,
    metadata_path: str | Path = DEFAULT_RAW_METADATA_PATH,
    min_seconds: float | None = None,
    max_seconds: float | None = None,
    top_speakers: int = 5,
) -> dict[str, Any]:
    min_seconds, max_seconds = _validate_filter_bounds(min_seconds, max_seconds)
    metadata, warnings = _load_metadata_frame(metadata_path)
    filtered = _apply_duration_filter(metadata, min_seconds, max_seconds)

    total_duration_s = float(filtered["duration_s"].sum()) if not filtered.empty else 0.0
    analysis = {
        "filters": {
            "metadata_path": str(metadata_path),
            "min_seconds": min_seconds,
            "max_seconds": max_seconds,
        },
        "overall_totals": {
            "total_clips": int(len(filtered)),
            "total_duration_s": total_duration_s,
            "total_minutes": _to_minutes(total_duration_s),
            "unique_speakers": int(filtered["source_speaker_id"].nunique()) if not filtered.empty else 0,
        },
        "grouped_by_locale_variant_gender": _grouped_duration_summary(filtered, ["locale", "variant", "gender"]),
        "per_variant_gender": _grouped_duration_summary(filtered, ["variant", "gender"]),
        "duration_bins": _duration_bins(filtered),
        "speaker_thresholds": _speaker_thresholds(filtered),
        "speaker_concentration": _speaker_concentration(filtered, top_speakers),
        "duration_summary": _duration_summary(filtered),
        "warnings": warnings,
    }
    return analysis


def _list_value(values: list[str]) -> str:
    return ", ".join(values) if values else "n/a"


def render_dataset_slice_analysis(analysis: Mapping[str, Any]) -> str:
    totals = analysis["overall_totals"]
    filters = analysis["filters"]
    concentration = analysis["speaker_concentration"]
    lines = [
        "DATASET SLICE ANALYSIS",
        f"metadata_path={filters['metadata_path']}",
        f"min_seconds={filters['min_seconds']} max_seconds={filters['max_seconds']}",
        f"total_clips={totals['total_clips']} total_minutes={totals['total_minutes']:.3f} unique_speakers={totals['unique_speakers']}",
        "grouped_by_locale_variant_gender:",
    ]

    for row in analysis["grouped_by_locale_variant_gender"]:
        lines.append(
            f"- locale={row['locale']} variant={row['variant']} gender={row['gender']} clip_count={row['clip_count']} total_minutes={row['total_minutes']:.3f} unique_speakers={row['unique_speakers']}"
        )

    lines.append("")
    lines.append("per_variant_gender:")
    for row in analysis["per_variant_gender"]:
        lines.append(
            f"- variant={row['variant']} gender={row['gender']} clip_count={row['clip_count']} total_minutes={row['total_minutes']:.3f} unique_speakers={row['unique_speakers']}"
        )

    summary = analysis["duration_summary"]
    lines.extend(
        [
            "",
            "duration_summary:",
            f"- clip_count={summary['clip_count']} min_s={summary['min_s']} p25_s={summary['p25_s']} median_s={summary['median_s']} mean_s={summary['mean_s']} p75_s={summary['p75_s']} p90_s={summary['p90_s']} p95_s={summary['p95_s']} max_s={summary['max_s']}",
            "",
            "duration_bins:",
        ]
    )
    for row in analysis["duration_bins"]:
        lines.append(
            f"- duration_bin={row['duration_bin']} clip_count={row['clip_count']} total_minutes={row['total_minutes']:.3f} unique_speakers={row['unique_speakers']}"
        )

    lines.append("")
    lines.append("speaker_thresholds:")
    for row in analysis["speaker_thresholds"]:
        lines.append(f"- threshold_minutes={row['threshold_minutes']} speaker_count={row['speaker_count']}")

    lines.extend(
        [
            "",
            "speaker_concentration:",
            f"- top_n={concentration['top_n']} top_n_clip_share_pct={concentration['top_n_clip_share_pct']:.3f} top_n_duration_share_pct={concentration['top_n_duration_share_pct']:.3f}",
        ]
    )
    for row in concentration["speakers"]:
        lines.append(
            f"- source_speaker_id={row['source_speaker_id']} clip_count={row['clip_count']} total_minutes={row['total_minutes']:.3f} clip_share_pct={row['clip_share_pct']:.3f} duration_share_pct={row['duration_share_pct']:.3f}"
        )

    warnings = list(analysis.get("warnings", []))
    if warnings:
        lines.append("")
        lines.append("WARNINGS")
        for warning in warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Common Voice metadata slices using duration filters.")
    parser.add_argument("-m", "--metadata", default=str(DEFAULT_RAW_METADATA_PATH), help="Path to the raw metadata CSV.")
    parser.add_argument("--min-seconds", type=float, help="Keep clips with duration_s >= this value.")
    parser.add_argument("--max-seconds", type=float, help="Keep clips with duration_s <= this value.")
    parser.add_argument("--range-seconds", nargs=2, type=float, metavar=("MIN", "MAX"), help="Inclusive duration range shortcut.")
    parser.add_argument("--top-speakers", type=int, default=5, help="Number of dominant speakers to display.")
    parser.add_argument("-j", "--json-out", help="Optional path to persist the analysis JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.range_seconds is not None and (args.min_seconds is not None or args.max_seconds is not None):
        raise SystemExit("--range-seconds cannot be combined with --min-seconds or --max-seconds")

    min_seconds = args.min_seconds
    max_seconds = args.max_seconds
    if args.range_seconds is not None:
        min_seconds, max_seconds = args.range_seconds

    try:
        analysis = analyze_dataset_slice(
            metadata_path=args.metadata,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            top_speakers=args.top_speakers,
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
