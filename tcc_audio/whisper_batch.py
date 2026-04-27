"""Batch ASR transcription with Whisper via Transformers pipeline."""

from __future__ import annotations

import argparse
import platform
from contextlib import contextmanager
from pathlib import Path

from tcc_audio.cli_defaults import DEFAULT_SAMPLES_PATH, load_cli_config, resolve_samples_path
from tcc_audio.config import (
    DEFAULT_ASR_LANGUAGE,
    DEFAULT_ASR_TASK,
    resolve_asr_language,
    resolve_asr_model,
    resolve_asr_task,
)
from tcc_audio.runtime import load_samples, now_utc_iso, refresh_sample_status, save_samples


def _load_asr_pipeline():
    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:
        raise SystemExit("Transformers and torch are required for Whisper batch transcription.") from exc

    if torch.cuda.is_available():
        device = 0
    elif platform.system() == "Darwin" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = -1

    return pipeline, device


def _build_generate_kwargs(transcriber, task: str, language: str) -> dict[str, object]:
    generate_kwargs: dict[str, object] = {"task": task, "language": language}
    suppress_tokens = getattr(transcriber.generation_config, "suppress_tokens", None)
    begin_suppress_tokens = getattr(transcriber.generation_config, "begin_suppress_tokens", None)
    if suppress_tokens is not None:
        generate_kwargs["suppress_tokens"] = list(suppress_tokens)
    if begin_suppress_tokens is not None:
        generate_kwargs["begin_suppress_tokens"] = list(begin_suppress_tokens)
    return generate_kwargs


@contextmanager
def _without_default_whisper_suppression(transcriber):
    generation_configs = []
    pipeline_config = getattr(transcriber, "generation_config", None)
    model = getattr(transcriber, "model", None)
    model_config = getattr(model, "generation_config", None)
    for config in (pipeline_config, model_config):
        if config is not None and all(id(existing) != id(config) for existing in generation_configs):
            generation_configs.append(config)

    original_values = []
    for config in generation_configs:
        original_values.append(
            (
                config,
                getattr(config, "suppress_tokens", None),
                getattr(config, "begin_suppress_tokens", None),
            )
        )
        config.suppress_tokens = None
        config.begin_suppress_tokens = None

    try:
        yield
    finally:
        for config, suppress_tokens, begin_suppress_tokens in original_values:
            config.suppress_tokens = suppress_tokens
            config.begin_suppress_tokens = begin_suppress_tokens


def run_whisper_batch(
    samples_path: str | Path,
    out_path: str | Path,
    model_name: str = "openai/whisper-small",
    only_missing: bool = True,
    task: str = DEFAULT_ASR_TASK,
    language: str = DEFAULT_ASR_LANGUAGE,
) -> None:
    pipeline_factory, device = _load_asr_pipeline()
    transcriber = pipeline_factory(
        "automatic-speech-recognition",
        model=model_name,
        device=device,
    )
    generate_kwargs = _build_generate_kwargs(transcriber, task, language)
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
            with _without_default_whisper_suppression(transcriber):
                result = transcriber(
                    str(audio_path),
                    generate_kwargs=generate_kwargs,
                )
            samples.loc[samples["sample_id"].eq(sample_id), "asr_text"] = result["text"].strip()
            samples.loc[samples["sample_id"].eq(sample_id), "run_started_at"] = started
            samples.loc[samples["sample_id"].eq(sample_id), "run_finished_at"] = now_utc_iso()
        except Exception as exc:  # pragma: no cover - runtime integration path
            samples.loc[samples["sample_id"].eq(sample_id), "failure_reason"] = str(exc)

    samples = refresh_sample_status(samples)
    save_samples(samples, out_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Whisper on all generated audio samples.")
    parser.add_argument("-c", "--config", help="Optional experiment config used to resolve evaluation.asr.name.")
    parser.add_argument("-s", "--samples")
    parser.add_argument("-o", "--out")
    parser.add_argument("--model-name")
    parser.add_argument("--task", choices=("transcribe", "translate"))
    parser.add_argument("--language")
    parser.add_argument("--all", action="store_true", help="Recompute ASR even when asr_text already exists.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _, config = load_cli_config(args.config)
    samples_path = args.samples or str(resolve_samples_path(config) if config else DEFAULT_SAMPLES_PATH)
    out_path = args.out or samples_path
    model_name = args.model_name or resolve_asr_model(config)
    task = args.task or resolve_asr_task(config)
    language = args.language or resolve_asr_language(config)
    run_whisper_batch(
        samples_path,
        out_path,
        model_name,
        only_missing=not args.all,
        task=task,
        language=language,
    )
    print(f"Updated ASR transcripts in {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
