"""Audio preprocessing for the TCC dataset."""

from __future__ import annotations

import argparse
import subprocess
import wave
from pathlib import Path

import pandas as pd

from tcc_audio.cli_defaults import (
    DEFAULT_PROCESSED_DIR,
    DEFAULT_PROCESSED_METADATA_PATH,
    DEFAULT_RAW_METADATA_PATH,
    load_cli_config,
    resolve_sample_rate,
)
from tcc_audio.io import ensure_parent_dir, read_csv


def wav_duration_seconds(path: str | Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def preprocess_audio_dataset(
    metadata_path: str | Path,
    out_dir: str | Path,
    out_metadata: str | Path,
    sample_rate: int = 16000,
    overwrite: bool = False,
    project_root: str | Path = ".",
) -> pd.DataFrame:
    metadata = read_csv(metadata_path)
    output_rows: list[dict[str, object]] = []
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for _, row in metadata.iterrows():
        source_path = Path(row["audio_path"])
        if not source_path.is_absolute():
            source_path = Path(project_root) / source_path
        speaker_id = row["source_speaker_id"]
        target_dir = out_root / speaker_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{row['utterance_id']}.wav"

        if overwrite or not target_path.exists():
            command = [
                "ffmpeg",
                "-y",
                "-i",
                str(source_path),
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-sample_fmt",
                "s16",
                str(target_path),
            ]
            try:
                subprocess.run(command, check=True, capture_output=True)
            except FileNotFoundError as exc:
                raise SystemExit("ffmpeg is required for audio preprocessing.") from exc

        processed = row.to_dict()
        processed["audio_path"] = str(target_path)
        processed["duration_s"] = wav_duration_seconds(target_path)
        output_rows.append(processed)

    processed_frame = pd.DataFrame(output_rows)
    ensure_parent_dir(out_metadata)
    processed_frame.to_csv(out_metadata, index=False)
    return processed_frame


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Common Voice clips to WAV mono and compute duration.")
    parser.add_argument("-c", "--config", help="Optional experiment config used to resolve sample_rate.")
    parser.add_argument("-m", "--metadata", default=str(DEFAULT_RAW_METADATA_PATH), help="Input metadata CSV.")
    parser.add_argument("-o", "--out-dir", default=str(DEFAULT_PROCESSED_DIR), help="Output directory for processed WAV files.")
    parser.add_argument("-u", "--out-metadata", default=str(DEFAULT_PROCESSED_METADATA_PATH), help="Output metadata CSV.")
    parser.add_argument("--sample-rate", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--project-root", default=".")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _, config = load_cli_config(args.config)
    sample_rate = args.sample_rate if args.sample_rate is not None else resolve_sample_rate(config)
    processed = preprocess_audio_dataset(
        metadata_path=args.metadata,
        out_dir=args.out_dir,
        out_metadata=args.out_metadata,
        sample_rate=sample_rate,
        overwrite=args.overwrite,
        project_root=args.project_root,
    )
    print(f"Wrote {len(processed)} processed metadata rows to {args.out_metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
