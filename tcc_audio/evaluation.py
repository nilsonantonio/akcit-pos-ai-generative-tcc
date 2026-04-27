"""Aggregate automatic and human-facing evaluation metrics."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tcc_audio.cli_defaults import DEFAULT_EVALUATION_DIR, DEFAULT_SAMPLES_PATH, load_cli_config, resolve_evaluation_dir, resolve_samples_path
from tcc_audio.io import ensure_parent_dir, read_csv
from tcc_audio.runtime import load_samples
from tcc_audio.schema import EVAL_SAMPLES_REQUIRED_COLUMNS, NUMERIC_METRIC_COLUMNS

METRIC_SUMMARY_COLUMNS = [
    "condition",
    "text_variant",
    "metric",
    "n",
    "mean",
    "median",
    "std",
    "ci95_low",
    "ci95_high",
    "test",
    "statistic",
    "p_value",
]

COST_SUMMARY_COLUMNS = [
    "condition",
    "samples",
    "total_cost_usd",
    "total_train_gpu_hours",
    "mean_rtf",
    "mean_inference_seconds",
]

SPEAKER_METRIC_SUMMARY_COLUMNS = [
    "condition",
    "speaker_id",
    "text_variant",
    "metric",
    "n",
    "mean",
    "median",
    "std",
    "ci95_low",
    "ci95_high",
]

SPEAKER_COST_SUMMARY_COLUMNS = [
    "condition",
    "speaker_id",
    "samples",
    "total_cost_usd",
    "total_train_gpu_hours",
    "mean_rtf",
    "mean_inference_seconds",
]


def _validate_samples(samples: pd.DataFrame) -> None:
    missing = [column for column in EVAL_SAMPLES_REQUIRED_COLUMNS if column not in samples.columns]
    if missing:
        raise ValueError(f"Missing samples.csv columns: {', '.join(missing)}")


def _bootstrap_ci(values: pd.Series, iterations: int = 2000, seed: int = 42) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    if len(numeric) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(numeric, size=len(numeric), replace=True).mean()) for _ in range(iterations)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _analysis_condition(frame: pd.DataFrame) -> pd.Series:
    checkpoint_label = frame["checkpoint_label"].astype(str).str.strip()
    return checkpoint_label.where(checkpoint_label.ne(""), frame["condition"].astype(str))


def aggregate_metrics(samples_path: str | Path, out_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = load_samples(samples_path)
    _validate_samples(samples)

    completed = samples[samples["status"].str.lower().eq("ok")].copy()
    completed["analysis_condition"] = _analysis_condition(completed)
    for column in NUMERIC_METRIC_COLUMNS:
        completed[column] = pd.to_numeric(completed[column], errors="coerce")

    metric_rows: list[dict[str, object]] = []
    group_columns = ["analysis_condition", "text_variant"]
    for keys, group in completed.groupby(group_columns, dropna=False):
        condition, text_variant = keys
        for metric in NUMERIC_METRIC_COLUMNS:
            values = group[metric].dropna()
            ci_low, ci_high = _bootstrap_ci(values)
            metric_rows.append(
                {
                    "condition": condition,
                    "text_variant": text_variant,
                    "metric": metric,
                    "n": int(values.count()),
                    "mean": float(values.mean()) if len(values) else np.nan,
                    "median": float(values.median()) if len(values) else np.nan,
                    "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                }
            )
    metrics_summary = pd.DataFrame(metric_rows, columns=METRIC_SUMMARY_COLUMNS)

    if completed.empty:
        cost_summary = pd.DataFrame(columns=COST_SUMMARY_COLUMNS)
    else:
        cost_summary = (
            completed.groupby("analysis_condition", dropna=False)
            .agg(
                samples=("sample_id", "count"),
                total_cost_usd=("cost_usd", "sum"),
                total_train_gpu_hours=("train_gpu_hours", "sum"),
                mean_rtf=("rtf", "mean"),
                mean_inference_seconds=("inference_seconds", "mean"),
            )
            .reset_index()
            .rename(columns={"analysis_condition": "condition"})
        )

    comparisons = _paired_condition_tests(completed)
    if not comparisons.empty:
        metrics_summary = pd.concat([metrics_summary, comparisons], ignore_index=True, sort=False)
        metrics_summary = metrics_summary.reindex(columns=METRIC_SUMMARY_COLUMNS)

    speaker_metrics_summary = _aggregate_metrics_by_speaker(completed)
    speaker_cost_summary = _aggregate_costs_by_speaker(completed)

    if out_dir:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        metrics_path = out_path / "metrics_summary.csv"
        cost_path = out_path / "cost_summary.csv"
        metrics_by_speaker_path = out_path / "metrics_by_speaker.csv"
        cost_by_speaker_path = out_path / "cost_by_speaker.csv"
        ensure_parent_dir(metrics_path)
        metrics_summary.to_csv(metrics_path, index=False)
        cost_summary.to_csv(cost_path, index=False)
        speaker_metrics_summary.to_csv(metrics_by_speaker_path, index=False)
        speaker_cost_summary.to_csv(cost_by_speaker_path, index=False)

    return metrics_summary, cost_summary


def _paired_condition_tests(samples: pd.DataFrame) -> pd.DataFrame:
    """Run paired Wilcoxon tests between every evaluated checkpoint label."""
    rows: list[dict[str, object]] = []
    index_cols = ["speaker_id", "prompt_id", "text_variant"]
    condition_labels = sorted(samples["analysis_condition"].dropna().astype(str).unique())

    for metric in ["wer", "speaker_similarity", "nisqa", "f0_rmse"]:
        for left_condition, right_condition in combinations(condition_labels, 2):
            left = samples[samples["analysis_condition"].eq(left_condition)][index_cols + [metric]].rename(
                columns={metric: "left"}
            )
            right = samples[samples["analysis_condition"].eq(right_condition)][index_cols + [metric]].rename(
                columns={metric: "right"}
            )
            paired = left.merge(right, on=index_cols, how="inner").dropna()
            if len(paired) < 2:
                continue
            try:
                statistic, p_value = stats.wilcoxon(paired["left"], paired["right"])
            except ValueError:
                statistic, p_value = (np.nan, np.nan)
            rows.append(
                {
                    "condition": f"{left_condition} vs {right_condition}",
                    "text_variant": "paired",
                    "metric": metric,
                    "n": int(len(paired)),
                    "mean": float((paired["left"] - paired["right"]).mean()),
                    "median": float((paired["left"] - paired["right"]).median()),
                    "std": float((paired["left"] - paired["right"]).std(ddof=1)),
                    "test": "wilcoxon",
                    "statistic": float(statistic) if not np.isnan(statistic) else np.nan,
                    "p_value": float(p_value) if not np.isnan(p_value) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _aggregate_metrics_by_speaker(samples: pd.DataFrame) -> pd.DataFrame:
    metric_rows: list[dict[str, object]] = []
    group_columns = ["analysis_condition", "speaker_id", "text_variant"]
    for condition, speaker_id, text_variant in samples.groupby(group_columns, dropna=False).groups:
        group = samples.loc[
            samples["analysis_condition"].eq(condition)
            & samples["speaker_id"].eq(speaker_id)
            & samples["text_variant"].eq(text_variant)
        ]
        for metric in NUMERIC_METRIC_COLUMNS:
            values = group[metric].dropna()
            ci_low, ci_high = _bootstrap_ci(values)
            metric_rows.append(
                {
                    "condition": condition,
                    "speaker_id": speaker_id,
                    "text_variant": text_variant,
                    "metric": metric,
                    "n": int(values.count()),
                    "mean": float(values.mean()) if len(values) else np.nan,
                    "median": float(values.median()) if len(values) else np.nan,
                    "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                }
            )
    return pd.DataFrame(metric_rows, columns=SPEAKER_METRIC_SUMMARY_COLUMNS)


def _aggregate_costs_by_speaker(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame(columns=SPEAKER_COST_SUMMARY_COLUMNS)
    return (
        samples.groupby(["analysis_condition", "speaker_id"], dropna=False)
        .agg(
            samples=("sample_id", "count"),
            total_cost_usd=("cost_usd", "sum"),
            total_train_gpu_hours=("train_gpu_hours", "sum"),
            mean_rtf=("rtf", "mean"),
            mean_inference_seconds=("inference_seconds", "mean"),
        )
        .reset_index()
        .rename(columns={"analysis_condition": "condition"})
        .reindex(columns=SPEAKER_COST_SUMMARY_COLUMNS)
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate experiment sample metrics.")
    parser.add_argument("-c", "--config", help="Optional experiment config used to resolve samples/evaluation paths.")
    parser.add_argument("-s", "--samples", help="Path to samples.csv.")
    parser.add_argument("-o", "--out-dir", help="Output directory for summaries.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _, config = load_cli_config(args.config)
    samples_path = args.samples or str(resolve_samples_path(config) if config else DEFAULT_SAMPLES_PATH)
    out_dir = args.out_dir or str(resolve_evaluation_dir(config) if config else DEFAULT_EVALUATION_DIR)
    metrics, costs = aggregate_metrics(samples_path, out_dir)
    print(f"Wrote {len(metrics)} metric summary rows and {len(costs)} cost summary rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
