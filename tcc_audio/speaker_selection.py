"""Select a speaker subset globally and build the data manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tcc_audio.cli_defaults import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_PROCESSED_METADATA_PATH,
    DEFAULT_SPEAKER_SELECTION_PATH,
)
from tcc_audio.io import ensure_parent_dir, read_csv
from tcc_audio.speecht5_text import normalize_text_for_speecht5


INPUT_COLUMNS = [
    "source_speaker_id",
    "gender",
    "utterance_id",
    "duration_s",
    "audio_path",
    "target_text",
    "license",
    "source",
]


def _validate_input(metadata: pd.DataFrame) -> None:
    missing = [column for column in INPUT_COLUMNS if column not in metadata.columns]
    if missing:
        raise ValueError(f"Missing speaker selection input columns: {', '.join(missing)}")


def _validation_count(selected_count: int, val_ratio: float) -> int:
    if selected_count <= 1:
        return 0
    return min(max(1, int(round(selected_count * val_ratio))), selected_count - 1)


def select_speakers(
    metadata_path: str | Path,
    manifest_out: str | Path,
    speaker_selection_out: str | Path,
    speaker_target_count: int = 4,
    minutes_per_speaker: int = 20,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = read_csv(metadata_path)
    _validate_input(metadata)
    metadata["duration_s"] = pd.to_numeric(metadata["duration_s"], errors="coerce")
    metadata = metadata.dropna(subset=["duration_s"])
    metadata = metadata[metadata["duration_s"] > 0].copy()

    rng = metadata.sample(frac=1.0, random_state=seed)
    target_seconds = minutes_per_speaker * 60
    selected_manifest_rows: list[dict[str, object]] = []
    selected_speaker_rows: list[dict[str, object]] = []

    ranked = (
        rng.groupby("source_speaker_id", as_index=False)["duration_s"]
        .sum()
        .sort_values(["duration_s", "source_speaker_id"], ascending=[False, True])
    )
    chosen_speakers = ranked.head(speaker_target_count)

    for _, speaker in chosen_speakers.iterrows():
        source_speaker_id = speaker["source_speaker_id"]
        speaker_rows = rng[rng["source_speaker_id"].eq(source_speaker_id)].copy()
        speaker_rows = speaker_rows.sort_values("duration_s", ascending=False)

        cumulative = 0.0
        selected_rows = []
        for _, row in speaker_rows.iterrows():
            if cumulative >= target_seconds:
                break
            cumulative += float(row["duration_s"])
            selected_rows.append(row)

        if not selected_rows:
            continue

        selected_df = pd.DataFrame(selected_rows)
        # Preserve at least one training clip for every selected speaker.
        val_count = _validation_count(len(selected_df), val_ratio)
        speaker_id = f"speaker_{len(selected_speaker_rows) + 1:02d}"
        reference_row = selected_df.iloc[0]

        selected_speaker_rows.append(
            {
                "speaker_id": speaker_id,
                "source_speaker_id": source_speaker_id,
                "gender": str(reference_row.get("gender", "")),
                "total_duration_s": float(speaker["duration_s"]),
                "selected_train_duration_s": float(selected_df.iloc[val_count:]["duration_s"].sum()),
                "selected_val_duration_s": float(selected_df.iloc[:val_count]["duration_s"].sum()),
                "reference_audio": reference_row["audio_path"],
                "reference_duration_s": float(reference_row["duration_s"]),
                "license": reference_row["license"],
                "source": reference_row["source"],
                "notes": "Auto-selected from processed metadata; manually audit audio before training.",
            }
        )

        for index, (_, row) in enumerate(selected_df.iterrows()):
            split = "val" if index < val_count else "train"
            selected_manifest_rows.append(
                {
                    "speaker_id": speaker_id,
                    "utterance_id": row["utterance_id"],
                    "split": split,
                    "duration_s": float(row["duration_s"]),
                    "source": row["source"],
                    "license": row["license"],
                    "audio_path": row["audio_path"],
                    "reference_audio": reference_row["audio_path"],
                    "target_text": row["target_text"],
                    "target_text_speecht5": normalize_text_for_speecht5(row["target_text"]),
                    "text_variant": "normalized",
                }
            )

    manifest = pd.DataFrame(selected_manifest_rows)
    speaker_selection = pd.DataFrame(selected_speaker_rows)

    ensure_parent_dir(manifest_out)
    ensure_parent_dir(speaker_selection_out)
    manifest.to_csv(manifest_out, index=False)
    speaker_selection.to_csv(speaker_selection_out, index=False)
    return manifest, speaker_selection


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a speaker manifest from processed Common Voice metadata using global duration ranking."
    )
    parser.add_argument("-m", "--metadata", default=str(DEFAULT_PROCESSED_METADATA_PATH), help="Path to the processed Common Voice metadata CSV.")
    parser.add_argument("-o", "--manifest-out", default=str(DEFAULT_MANIFEST_PATH), help="Output data_manifest.csv path.")
    parser.add_argument("-s", "--speaker-selection-out", default=str(DEFAULT_SPEAKER_SELECTION_PATH), help="Output speaker selection CSV path.")
    parser.add_argument(
        "-n",
        "--speaker-target-count",
        type=int,
        default=4,
        help="Total number of speakers to select globally by available duration.",
    )
    parser.add_argument("--minutes-per-speaker", type=int, default=20)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest, speaker_selection = select_speakers(
        metadata_path=args.metadata,
        manifest_out=args.manifest_out,
        speaker_selection_out=args.speaker_selection_out,
        speaker_target_count=args.speaker_target_count,
        minutes_per_speaker=args.minutes_per_speaker,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"Selected {speaker_selection['speaker_id'].nunique() if not speaker_selection.empty else 0} speakers")
    print(f"Wrote {len(manifest)} manifest rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
