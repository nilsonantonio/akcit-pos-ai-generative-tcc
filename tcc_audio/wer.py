"""WER utilities for Whisper transcript outputs."""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd

from tcc_audio.cli_defaults import DEFAULT_SAMPLES_PATH, load_cli_config, resolve_samples_path
from tcc_audio.runtime import audio_exists_mask, load_samples, refresh_sample_status, save_samples_atomic


def normalize_for_wer(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(text).lower())
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.split() if normalized else []


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_tokens = normalize_for_wer(reference)
    hyp_tokens = normalize_for_wer(hypothesis)
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0

    previous = list(range(len(hyp_tokens) + 1))
    for i, ref_token in enumerate(ref_tokens, start=1):
        current = [i]
        for j, hyp_token in enumerate(hyp_tokens, start=1):
            substitution = previous[j - 1] + (0 if ref_token == hyp_token else 1)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1] / len(ref_tokens)


def compute_wer_from_asr(
    samples_path: str | Path,
    out_path: str | Path | None = None,
    asr_column: str = "asr_text",
) -> pd.DataFrame:
    samples = load_samples(samples_path)
    if "target_text" not in samples.columns:
        raise ValueError("samples.csv must contain target_text")
    if asr_column not in samples.columns:
        raise ValueError(f"samples.csv must contain {asr_column}")

    audio_exists = audio_exists_mask(samples)

    def score(row: pd.Series) -> object:
        row_audio_exists = bool(audio_exists.loc[row.name])
        if not row_audio_exists or str(row["failure_reason"]).strip():
            return row.get("wer", "")
        if str(row[asr_column]).strip() == "":
            return row.get("wer", "")
        return word_error_rate(row["target_text"], row[asr_column])

    samples["wer"] = samples.apply(score, axis=1)
    samples = refresh_sample_status(samples)
    if out_path:
        save_samples_atomic(samples, out_path)
    return samples


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute WER from Whisper transcript text in samples.csv.")
    parser.add_argument("-c", "--config", help="Optional experiment config used to resolve samples.csv.")
    parser.add_argument("-s", "--samples", help="Path to samples.csv.")
    parser.add_argument("-o", "--out", help="Output samples.csv path.")
    parser.add_argument("--asr-column", default="asr_text")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _, config = load_cli_config(args.config)
    samples_path = args.samples or str(resolve_samples_path(config) if config else DEFAULT_SAMPLES_PATH)
    out_path = args.out or samples_path
    samples = compute_wer_from_asr(samples_path, out_path, args.asr_column)
    computed = pd.to_numeric(samples["wer"], errors="coerce").notna().sum()
    print(f"Computed WER for {computed} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
