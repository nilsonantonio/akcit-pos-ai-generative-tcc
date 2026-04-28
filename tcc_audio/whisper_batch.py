"""Batch ASR transcription with Whisper via Transformers pipeline."""

from __future__ import annotations

import argparse
import platform
import tempfile
from copy import deepcopy
from pathlib import Path

from tcc_audio.cli_defaults import DEFAULT_SAMPLES_PATH, load_cli_config, resolve_samples_path
from tcc_audio.config import (
    DEFAULT_ASR_LANGUAGE,
    DEFAULT_ASR_TASK,
    resolve_asr_language,
    resolve_asr_model,
    resolve_asr_task,
)
from tcc_audio.io import ensure_parent_dir
from tcc_audio.runtime import load_samples, now_utc_iso, refresh_sample_status


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


def _describe_device(device: object) -> str:
    if isinstance(device, int):
        return "CUDA" if device >= 0 else "CPU"
    device_type = getattr(device, "type", str(device)).lower()
    if device_type.startswith("cuda"):
        return "CUDA"
    if device_type.startswith("mps"):
        return "MPS"
    if device_type.startswith("cpu"):
        return "CPU"
    return device_type.upper()


def _build_generation_config(transcriber, task: str, language: str):
    generation_config = deepcopy(transcriber.generation_config)
    generation_config.task = task
    generation_config.language = language
    generation_config.forced_decoder_ids = None

    suppress_tokens = getattr(generation_config, "suppress_tokens", None)
    begin_suppress_tokens = getattr(generation_config, "begin_suppress_tokens", None)
    if suppress_tokens is not None:
        generation_config.suppress_tokens = list(suppress_tokens)
    if begin_suppress_tokens is not None:
        generation_config.begin_suppress_tokens = list(begin_suppress_tokens)
    return generation_config


def _save_samples_atomic(samples, path: str | Path) -> None:
    destination = Path(path)
    ensure_parent_dir(destination)

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.stem}_",
            suffix=destination.suffix,
            delete=False,
        ) as handle:
            samples.to_csv(handle, index=False)
            tmp_path = Path(handle.name)
        tmp_path.replace(destination)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


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
    samples = load_samples(samples_path)
    subset = samples[samples["audio_path"].astype(str).str.strip().ne("")]
    already_transcribed = subset[subset["asr_text"].astype(str).str.strip().ne("")]
    if only_missing:
        subset = subset[subset["asr_text"].astype(str).str.strip().eq("")]

    pending_total = len(subset)
    print(f"ASR device: {_describe_device(device)}")
    print(f"ASR model: {model_name}")
    print(f"Samples input: {samples_path}")
    print(f"Samples output: {out_path}")
    print(f"Rows total: {len(samples)}")
    print(f"Rows with audio: {len(samples[samples['audio_path'].astype(str).str.strip().ne('')])}")
    print(f"Rows already transcribed: {len(already_transcribed)}")
    print(f"Rows pending this run: {pending_total}")

    if pending_total == 0:
        samples = refresh_sample_status(samples)
        _save_samples_atomic(samples, out_path)
        print("No pending ASR rows.")
        return

    completed = 0
    failed = 0

    for current, row in enumerate(subset.itertuples(index=False), start=1):
        sample_id = row.sample_id
        audio_path = Path(row.audio_path)
        sample_mask = samples["sample_id"].eq(sample_id)
        status = "ok"

        if not audio_path.exists():
            samples.loc[sample_mask, "failure_reason"] = f"missing audio: {audio_path}"
            status = "missing_audio"
            failed += 1
        else:
            try:
                result = transcriber(
                    str(audio_path),
                    generation_config=_build_generation_config(transcriber, task, language),
                )
                samples.loc[sample_mask, "asr_text"] = result["text"].strip()
                samples.loc[sample_mask, "failure_reason"] = ""
                completed += 1
            except Exception as exc:  # pragma: no cover - runtime integration path
                samples.loc[sample_mask, "failure_reason"] = str(exc)
                status = "error"
                failed += 1

        samples = refresh_sample_status(samples)
        _save_samples_atomic(samples, out_path)

        remaining = pending_total - current
        print(
            f"[{current}/{pending_total}] sample_id={sample_id} "
            f"status={status} completed={completed} failed={failed} remaining={remaining}"
        )
        if status != "ok":
            print(f"audio_path={audio_path}")


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
