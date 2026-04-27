"""Create report assets from evaluation outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tcc_audio.cli_defaults import (
    DEFAULT_COSTS_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_REPORT_ASSETS_DIR,
    DEFAULT_SAMPLES_PATH,
    load_cli_config,
    resolve_deliverable_path,
)
from tcc_audio.io import ensure_parent_dir, read_csv
from tcc_audio.runtime import load_samples


def _maybe_load_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except Exception:
        return frame.to_csv(index=False)


def make_report_assets(
    samples_path: str | Path,
    metrics_path: str | Path,
    costs_path: str | Path,
    out_dir: str | Path,
) -> dict[str, Path]:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    samples = load_samples(samples_path)
    metrics = read_csv(metrics_path)
    costs = read_csv(costs_path)
    metrics_by_speaker_path = Path(metrics_path).with_name("metrics_by_speaker.csv")
    costs_by_speaker_path = Path(costs_path).with_name("cost_by_speaker.csv")
    metrics_by_speaker = read_csv(metrics_by_speaker_path) if metrics_by_speaker_path.exists() else pd.DataFrame()
    costs_by_speaker = read_csv(costs_by_speaker_path) if costs_by_speaker_path.exists() else pd.DataFrame()

    outputs = {
        "metrics_markdown": out_root / "metrics_summary.md",
        "costs_markdown": out_root / "cost_summary.md",
        "overview_markdown": out_root / "overview.md",
    }

    outputs["metrics_markdown"].write_text(_frame_to_markdown(metrics), encoding="utf-8")
    outputs["costs_markdown"].write_text(_frame_to_markdown(costs), encoding="utf-8")
    if not metrics_by_speaker.empty:
        outputs["metrics_by_speaker_markdown"] = out_root / "metrics_by_speaker.md"
        outputs["metrics_by_speaker_markdown"].write_text(
            _frame_to_markdown(metrics_by_speaker), encoding="utf-8"
        )
    if not costs_by_speaker.empty:
        outputs["costs_by_speaker_markdown"] = out_root / "cost_by_speaker.md"
        outputs["costs_by_speaker_markdown"].write_text(
            _frame_to_markdown(costs_by_speaker), encoding="utf-8"
        )

    completed = samples[samples["status"].astype(str).str.lower().eq("ok")]
    overview = [
        "# Report Assets Overview",
        "",
        f"- completed_samples: {len(completed)}",
        f"- unique_conditions: {completed['checkpoint_label'].astype(str).str.strip().replace('', pd.NA).dropna().nunique() or completed['condition'].nunique()}",
        f"- unique_speakers: {completed['speaker_id'].nunique()}",
        f"- unique_prompts: {completed['prompt_id'].nunique()}",
    ]
    outputs["overview_markdown"].write_text("\n".join(overview) + "\n", encoding="utf-8")

    plt = _maybe_load_matplotlib()
    if plt is not None and not metrics.empty:
        for metric_name in ["wer", "speaker_similarity", "nisqa", "f0_rmse"]:
            subset = metrics[metrics["metric"].eq(metric_name) & metrics["text_variant"].ne("paired")]
            if subset.empty:
                continue
            figure_path = out_root / f"{metric_name}.png"
            plt.figure(figsize=(10, 4))
            plt.bar(subset["condition"], pd.to_numeric(subset["mean"], errors="coerce"))
            plt.xticks(rotation=25, ha="right")
            plt.title(metric_name)
            plt.tight_layout()
            plt.savefig(figure_path)
            plt.close()
            outputs[f"{metric_name}_chart"] = figure_path

    if plt is not None and not metrics_by_speaker.empty:
        for metric_name in ["wer", "speaker_similarity", "nisqa", "f0_rmse"]:
            subset = metrics_by_speaker[metrics_by_speaker["metric"].eq(metric_name)]
            if subset.empty:
                continue
            figure_path = out_root / f"{metric_name}_by_speaker.png"
            pivot = subset.pivot(index="condition", columns="speaker_id", values="mean").apply(
                pd.to_numeric, errors="coerce"
            )
            if pivot.dropna(how="all").empty:
                continue
            pivot.plot(kind="bar", figsize=(12, 4))
            plt.xticks(rotation=25, ha="right")
            plt.title(f"{metric_name} by speaker")
            plt.tight_layout()
            plt.savefig(figure_path)
            plt.close()
            outputs[f"{metric_name}_by_speaker_chart"] = figure_path

    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build report assets for the TCC presentation.")
    parser.add_argument("-c", "--config", help="Optional experiment config used to resolve deliverable paths.")
    parser.add_argument("-s", "--samples")
    parser.add_argument("-m", "--metrics")
    parser.add_argument("-k", "--costs")
    parser.add_argument("-o", "--out-dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _, config = load_cli_config(args.config)
    samples_path = args.samples or str(resolve_deliverable_path(config, "samples", DEFAULT_SAMPLES_PATH))
    metrics_path = args.metrics or str(resolve_deliverable_path(config, "metrics_summary", DEFAULT_METRICS_PATH))
    costs_path = args.costs or str(resolve_deliverable_path(config, "cost_summary", DEFAULT_COSTS_PATH))
    out_dir = args.out_dir or str(resolve_deliverable_path(config, "report_assets_dir", DEFAULT_REPORT_ASSETS_DIR))
    outputs = make_report_assets(samples_path, metrics_path, costs_path, out_dir)
    print(f"Wrote {len(outputs)} report asset files to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
