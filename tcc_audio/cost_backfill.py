"""Backfill training cost fields in samples.csv from sample timestamps."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

SUMMARY_COLUMNS = [
    "condition",
    "speaker_id",
    "matched_rows",
    "timed_rows",
    "filled_train_gpu_hours_cells",
    "filled_cost_usd_cells",
    "total_runtime_hours",
    "train_gpu_hours_per_sample",
    "cost_usd_per_sample",
]


def _parse_timestamp(raw_value: object) -> datetime | None:
    value = str(raw_value).strip()
    if not value:
        return None
    return datetime.fromisoformat(value)


def _stringify_csv_value(value: object) -> str:
    return "" if value is None else str(value)


def _load_rows(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def _save_rows(rows: list[dict[str, str]], fieldnames: list[str], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _refresh_status(rows: list[dict[str, str]]) -> None:
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
    for row in rows:
        audio_path = (row.get("audio_path") or "").strip()
        audio_exists = bool(audio_path) and Path(audio_path).exists()
        failure_reason = (row.get("failure_reason") or "").strip()
        generated = audio_exists and failure_reason == ""
        status = (row.get("status") or "").strip().lower()
        if generated and status == "pending":
            row["status"] = "generated"
        if generated and all((row.get(column) or "").strip() != "" for column in metric_columns):
            row["status"] = "ok"


def backfill_training_costs(
    samples_path: str | Path,
    gpu_hourly_rate: float = 0.0,
    out_path: str | Path | None = None,
    summary_out_path: str | Path | None = None,
    overwrite: bool = False,
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    if gpu_hourly_rate < 0:
        raise ValueError("gpu_hourly_rate must be non-negative")

    rows, fieldnames = _load_rows(samples_path)
    groups: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        key = ((row.get("condition") or "").strip(), (row.get("speaker_id") or "").strip())
        groups.setdefault(key, []).append(index)

    summary_rows: list[dict[str, object]] = []
    for (condition, speaker_id), indices in sorted(groups.items()):
        total_runtime_seconds = 0.0
        timed_rows = 0
        for index in indices:
            row = rows[index]
            started_at = _parse_timestamp(row.get("run_started_at", ""))
            finished_at = _parse_timestamp(row.get("run_finished_at", ""))
            if started_at is None or finished_at is None:
                continue
            total_runtime_seconds += max((finished_at - started_at).total_seconds(), 0.0)
            timed_rows += 1

        if timed_rows == 0:
            continue

        matched_rows = len(indices)
        total_runtime_hours = total_runtime_seconds / 3600.0
        train_gpu_hours_per_sample = total_runtime_hours / matched_rows
        cost_usd_per_sample = train_gpu_hours_per_sample * gpu_hourly_rate

        filled_train = 0
        filled_cost = 0
        for index in indices:
            row = rows[index]
            if overwrite or (row.get("train_gpu_hours") or "").strip() == "":
                row["train_gpu_hours"] = _stringify_csv_value(train_gpu_hours_per_sample)
                filled_train += 1
            if overwrite or (row.get("cost_usd") or "").strip() == "":
                row["cost_usd"] = _stringify_csv_value(cost_usd_per_sample)
                filled_cost += 1

        summary_rows.append(
            {
                "condition": condition,
                "speaker_id": speaker_id,
                "matched_rows": matched_rows,
                "timed_rows": timed_rows,
                "filled_train_gpu_hours_cells": filled_train,
                "filled_cost_usd_cells": filled_cost,
                "total_runtime_hours": total_runtime_hours,
                "train_gpu_hours_per_sample": train_gpu_hours_per_sample,
                "cost_usd_per_sample": cost_usd_per_sample,
            }
        )

    _refresh_status(rows)
    _save_rows(rows, fieldnames, out_path or samples_path)

    if summary_out_path:
        summary_path = Path(summary_out_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS)
            writer.writeheader()
            writer.writerows(summary_rows)

    return rows, summary_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill train_gpu_hours and cost_usd from run_started_at/run_finished_at in samples.csv."
    )
    parser.add_argument("--samples", required=True, help="Path to artifacts/evaluation/samples.csv.")
    parser.add_argument(
        "--gpu-hourly-rate",
        type=float,
        required=True,
        help="GPU hourly rate used to convert derived runtime hours into cost_usd.",
    )
    parser.add_argument("--out", help="Optional output samples.csv path. Defaults to overwriting --samples.")
    parser.add_argument("--summary-out", help="Optional path for a CSV summary of the backfill operation.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing train_gpu_hours and cost_usd cells.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _, summary = backfill_training_costs(
        samples_path=args.samples,
        gpu_hourly_rate=args.gpu_hourly_rate,
        out_path=args.out,
        summary_out_path=args.summary_out,
        overwrite=args.overwrite,
    )
    print(f"Updated {len(summary)} condition/speaker groups")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
