"""Create report assets from evaluation outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tcc_audio.io import ensure_parent_dir, read_csv


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
    human_eval_path: str | Path | None,
    out_dir: str | Path,
) -> dict[str, Path]:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    samples = read_csv(samples_path)
    metrics = read_csv(metrics_path)
    costs = read_csv(costs_path)
    human = read_csv(human_eval_path) if human_eval_path and Path(human_eval_path).exists() else pd.DataFrame()

    outputs = {
        "metrics_markdown": out_root / "metrics_summary.md",
        "costs_markdown": out_root / "cost_summary.md",
        "human_markdown": out_root / "human_eval_summary.md",
        "overview_markdown": out_root / "overview.md",
    }

    outputs["metrics_markdown"].write_text(_frame_to_markdown(metrics), encoding="utf-8")
    outputs["costs_markdown"].write_text(_frame_to_markdown(costs), encoding="utf-8")
    outputs["human_markdown"].write_text(
        _frame_to_markdown(human) if not human.empty else "No human evaluation results available.\n",
        encoding="utf-8",
    )

    completed = samples[samples["status"].astype(str).str.lower().eq("ok")]
    overview = [
        "# Report Assets Overview",
        "",
        f"- completed_samples: {len(completed)}",
        f"- unique_conditions: {completed['condition'].nunique()}",
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

    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build report assets for the TCC presentation.")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--costs", required=True)
    parser.add_argument("--human-eval")
    parser.add_argument("--out-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    outputs = make_report_assets(args.samples, args.metrics, args.costs, args.human_eval, args.out_dir)
    print(f"Wrote {len(outputs)} report asset files to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
