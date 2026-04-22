"""Optional Parler-TTS reference runner for demo samples."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from tcc_audio.io import read_csv, read_yaml
from tcc_audio.runtime import load_samples, now_utc_iso, save_samples, stringify_csv_value


def _load_parler():
    try:
        import soundfile as sf
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Parler-TTS runtime not installed. Install parler-tts and transformers.") from exc
    return sf, torch, ParlerTTSForConditionalGeneration, AutoTokenizer


def run_parler_reference(
    config_path: str | Path,
    samples_path: str | Path,
    speaker_selection_path: str | Path,
    limit: int | None = None,
) -> pd.DataFrame:
    sf, torch, ParlerTTSForConditionalGeneration, AutoTokenizer = _load_parler()
    config = read_yaml(config_path)
    samples = load_samples(samples_path)
    speaker_selection = read_csv(speaker_selection_path)
    descriptions = {
        row["speaker_id"]: row.get(
            "parler_description",
            "A clear and natural Portuguese Brazilian voice, recorded in a clean studio.",
        )
        for _, row in speaker_selection.iterrows()
    }

    model_name = "parler-tts/parler-tts-mini-multilingual-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    sample_rate = model.audio_encoder.config.sampling_rate

    subset = samples[samples["condition"].eq("parler_reference_optional")].copy()
    if limit:
        subset = subset.head(limit)

    for _, row in subset.iterrows():
        sample_id = row["sample_id"]
        try:
            started = now_utc_iso()
            description = descriptions.get(row["speaker_id"], descriptions[next(iter(descriptions))])
            prompt_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
            input_ids = tokenizer(row["target_text"], return_tensors="pt").input_ids.to(device)
            timer = time.perf_counter()
            with torch.no_grad():
                audio = model.generate(input_ids=input_ids, prompt_input_ids=prompt_ids)
            inference_seconds = time.perf_counter() - timer
            output_path = Path(row["audio_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, audio.cpu().numpy().squeeze(), sample_rate)
            samples.loc[samples["sample_id"].eq(sample_id), "model_name"] = model_name
            samples.loc[samples["sample_id"].eq(sample_id), "run_started_at"] = started
            samples.loc[samples["sample_id"].eq(sample_id), "run_finished_at"] = now_utc_iso()
            samples.loc[samples["sample_id"].eq(sample_id), "inference_seconds"] = stringify_csv_value(inference_seconds)
            samples.loc[samples["sample_id"].eq(sample_id), "status"] = "generated"
        except Exception as exc:  # pragma: no cover - runtime integration path
            samples.loc[samples["sample_id"].eq(sample_id), "failure_reason"] = str(exc)
            samples.loc[samples["sample_id"].eq(sample_id), "status"] = "failed"

    save_samples(samples, samples_path)
    return samples


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Parler-TTS reference generation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--speaker-selection", default="data/manifests/speaker_selection.csv")
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    frame = run_parler_reference(args.config, args.samples, args.speaker_selection, args.limit)
    print(f"Processed {len(frame[frame['condition'].eq('parler_reference_optional')])} Parler rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
