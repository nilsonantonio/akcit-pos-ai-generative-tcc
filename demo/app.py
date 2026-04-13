#!/usr/bin/env python3
"""Minimal Gradio demo for comparing generated audio samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load_samples(path: str | Path) -> pd.DataFrame:
    samples = pd.read_csv(path, dtype=str, keep_default_na=False)
    samples = samples[samples["status"].str.lower().eq("ok")].copy()
    return samples


def _format_metrics(row: pd.Series) -> str:
    metric_names = ["wer", "speaker_similarity", "nisqa", "f0_rmse", "rtf", "cost_usd"]
    lines = [f"**{row['condition']}**", f"Prompt: `{row['prompt_id']}`", f"Speaker: `{row['speaker_id']}`"]
    for metric in metric_names:
        if metric in row and str(row[metric]).strip():
            lines.append(f"- `{metric}`: {row[metric]}")
    return "\n".join(lines)


def build_app(samples_path: str | Path):
    try:
        import gradio as gr
    except ImportError as exc:
        raise SystemExit("gradio is not installed. Install with: pip install '.[demo]'") from exc

    samples = _load_samples(samples_path)
    if samples.empty:
        raise SystemExit(f"No completed samples found in {samples_path}")

    prompt_ids = sorted(samples["prompt_id"].unique())
    speaker_ids = sorted(samples["speaker_id"].unique())
    conditions = sorted(samples["condition"].unique())

    def select_samples(prompt_id: str, speaker_id: str, condition_a: str, condition_b: str, condition_c: str):
        selected = []
        for condition in [condition_a, condition_b, condition_c]:
            rows = samples[
                samples["prompt_id"].eq(prompt_id)
                & samples["speaker_id"].eq(speaker_id)
                & samples["condition"].eq(condition)
            ]
            if rows.empty:
                selected.append((None, f"No sample found for `{condition}`."))
                continue
            row = rows.iloc[0]
            audio_path = row["audio_path"] if Path(row["audio_path"]).exists() else None
            selected.append((audio_path, _format_metrics(row)))
        return (
            selected[0][0],
            selected[0][1],
            selected[1][0],
            selected[1][1],
            selected[2][0],
            selected[2][1],
        )

    with gr.Blocks(title="TCC Audio PT-BR Demo") as app:
        gr.Markdown("# TCC Audio PT-BR Demo\nComparacao A/B/C de amostras e metricas.")
        with gr.Row():
            prompt = gr.Dropdown(prompt_ids, value=prompt_ids[0], label="Prompt")
            speaker = gr.Dropdown(speaker_ids, value=speaker_ids[0], label="Speaker")
        with gr.Row():
            condition_a = gr.Dropdown(conditions, value=conditions[0], label="Condicao A")
            condition_b = gr.Dropdown(conditions, value=conditions[min(1, len(conditions) - 1)], label="Condicao B")
            condition_c = gr.Dropdown(conditions, value=conditions[min(2, len(conditions) - 1)], label="Condicao C")
        run = gr.Button("Carregar comparacao")
        with gr.Row():
            with gr.Column():
                audio_a = gr.Audio(label="Audio A")
                metrics_a = gr.Markdown()
            with gr.Column():
                audio_b = gr.Audio(label="Audio B")
                metrics_b = gr.Markdown()
            with gr.Column():
                audio_c = gr.Audio(label="Audio C")
                metrics_c = gr.Markdown()
        run.click(
            select_samples,
            inputs=[prompt, speaker, condition_a, condition_b, condition_c],
            outputs=[audio_a, metrics_a, audio_b, metrics_b, audio_c, metrics_c],
        )
    return app


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the minimal Gradio demo.")
    parser.add_argument("--samples", required=True, help="Path to artifacts/evaluation/samples.csv.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    app = build_app(args.samples)
    app.launch(server_name=args.host, server_port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

