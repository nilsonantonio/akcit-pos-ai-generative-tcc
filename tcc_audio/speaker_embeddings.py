"""Extract speaker embeddings from reference audio files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tcc_audio.io import ensure_parent_dir, read_csv
from tcc_audio.speechbrain_compat import encode_audio_path, load_encoder_classifier


def extract_speaker_embeddings(
    speaker_selection_path: str | Path,
    out_index: str | Path,
    out_dir: str | Path,
    model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
) -> pd.DataFrame:
    EncoderClassifier = load_encoder_classifier()
    selection = read_csv(speaker_selection_path)
    required = {"speaker_id", "reference_audio"}
    missing = required.difference(selection.columns)
    if missing:
        raise ValueError(f"Missing speaker selection columns: {', '.join(sorted(missing))}")

    classifier = EncoderClassifier.from_hparams(source=model_name)
    rows: list[dict[str, object]] = []
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for _, row in selection.iterrows():
        speaker_id = row["speaker_id"]
        reference_audio = Path(row["reference_audio"])
        embedding = encode_audio_path(classifier, reference_audio)
        embedding_path = out_root / f"{speaker_id}.npy"
        np.save(embedding_path, embedding)
        rows.append(
            {
                "speaker_id": speaker_id,
                "reference_audio_path": str(reference_audio),
                "speaker_embedding_path": str(embedding_path),
                "embedding_model": model_name,
                "embedding_dim": embedding.shape[0],
            }
        )

    index = pd.DataFrame(rows)
    ensure_parent_dir(out_index)
    index.to_csv(out_index, index=False)
    return index


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract ECAPA speaker embeddings from reference audio.")
    parser.add_argument("--speaker-selection", required=True, help="Path to speaker_selection.csv.")
    parser.add_argument("--out-index", required=True, help="Output CSV with embedding metadata.")
    parser.add_argument("--out-dir", required=True, help="Directory for .npy embedding files.")
    parser.add_argument("--model-name", default="speechbrain/spkrec-ecapa-voxceleb")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    frame = extract_speaker_embeddings(
        speaker_selection_path=args.speaker_selection,
        out_index=args.out_index,
        out_dir=args.out_dir,
        model_name=args.model_name,
    )
    print(f"Wrote {len(frame)} speaker embeddings to {args.out_index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
