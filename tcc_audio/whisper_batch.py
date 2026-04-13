"""Batch ASR transcription with Whisper via Transformers pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from tcc_audio.runtime import load_samples, now_utc_iso, refresh_sample_status, save_samples


def _load_asr_pipeline():
    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:
        raise SystemExit("Transformers and torch are required for Whisper batch transcription.") from exc
    device = 0 if torch.cuda.is_available() else -1
    return pipeline, device


def run_whisper_batch(
    samples_path: str | Path,
    out_path: str | Path,
    model_name: str = "openai/whisper-small",
    only_missing: bool = True,
) -> None:
    pipeline_factory, device = _load_asr_pipeline()
    transcriber = pipeline_factory(
        "automatic-speech-recognition",
        model=model_name,
        device=device,
    )
    samples = load_samples(samples_path)
    subset = samples[samples["audio_path"].astype(str).str.strip().ne("")]
    if only_missing:
        subset = subset[subset["asr_text"].astype(str).str.strip().eq("")]

    for _, row in subset.iterrows():
        sample_id = row["sample_id"]
        audio_path = Path(row["audio_path"])
        if not audio_path.exists():
            samples.loc[samples["sample_id"].eq(sample_id), "failure_reason"] = f"missing audio: {audio_path}"
            continue
        try:
            started = now_utc_iso()
            result = transcriber(str(audio_path))
            samples.loc[samples["sample_id"].eq(sample_id), "asr_text"] = result["text"].strip()
            samples.loc[samples["sample_id"].eq(sample_id), "run_started_at"] = started
            samples.loc[samples["sample_id"].eq(sample_id), "run_finished_at"] = now_utc_iso()
        except Exception as exc:  # pragma: no cover - runtime integration path
            samples.loc[samples["sample_id"].eq(sample_id), "failure_reason"] = str(exc)

    samples = refresh_sample_status(samples)
    save_samples(samples, out_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Whisper on all generated audio samples.")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-name", default="openai/whisper-small")
    parser.add_argument("--all", action="store_true", help="Recompute ASR even when asr_text already exists.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_whisper_batch(args.samples, args.out, args.model_name, only_missing=not args.all)
    print(f"Updated ASR transcripts in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
