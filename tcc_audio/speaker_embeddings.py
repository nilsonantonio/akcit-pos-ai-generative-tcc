"""Extract speaker embeddings from reference audio files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tcc_audio.config import (
    load_experiment_config,
    resolve_tts_speaker_embedding_dim,
    resolve_tts_speaker_embedding_model,
)
from tcc_audio.cli_defaults import (
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_EMBEDDINGS_INDEX_PATH,
    DEFAULT_SPEAKER_SELECTION_PATH,
    load_cli_config,
    resolve_data_path,
)
from tcc_audio.io import ensure_parent_dir, read_csv
from tcc_audio.speechbrain_compat import encode_audio_path, load_encoder_classifier


def extract_speaker_embeddings(
    speaker_selection_path: str | Path,
    out_index: str | Path,
    out_dir: str | Path,
    config_path: str | Path | None = None,
    model_name: str | None = None,
    expected_dim: int | None = None,
) -> pd.DataFrame:
    config = load_experiment_config(config_path)
    resolved_model_name = model_name or resolve_tts_speaker_embedding_model(config)
    resolved_expected_dim = expected_dim if expected_dim is not None else resolve_tts_speaker_embedding_dim(config)
    EncoderClassifier = load_encoder_classifier()
    selection = read_csv(speaker_selection_path)
    required = {"speaker_id", "reference_audio"}
    missing = required.difference(selection.columns)
    if missing:
        raise ValueError(f"Missing speaker selection columns: {', '.join(sorted(missing))}")

    classifier = EncoderClassifier.from_hparams(source=resolved_model_name)
    rows: list[dict[str, object]] = []
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for _, row in selection.iterrows():
        speaker_id = row["speaker_id"]
        reference_audio = Path(row["reference_audio"])
        embedding = encode_audio_path(classifier, reference_audio)
        if embedding.shape[0] != resolved_expected_dim:
            raise ValueError(
                f"Embedding model '{resolved_model_name}' produced {embedding.shape[0]} dims for {speaker_id}, "
                f"but the TTS synthesis pipeline expects {resolved_expected_dim}-d speaker embeddings."
            )
        embedding_path = out_root / f"{speaker_id}.npy"
        np.save(embedding_path, embedding)
        rows.append(
            {
                "speaker_id": speaker_id,
                "reference_audio_path": str(reference_audio),
                # This path is consumed by SpeechT5 synthesis, not by speaker-similarity evaluation.
                "speaker_embedding_path": str(embedding_path),
                "embedding_model": resolved_model_name,
                "embedding_dim": embedding.shape[0],
            }
        )

    index = pd.DataFrame(rows)
    ensure_parent_dir(out_index)
    index.to_csv(out_index, index=False)
    return index


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract SpeechT5-compatible speaker embeddings from reference audio.")
    parser.add_argument("-c", "--config", help="Optional experiment config used to resolve the TTS embedding model.")
    parser.add_argument("-s", "--speaker-selection", help="Path to speaker_selection.csv.")
    parser.add_argument("-o", "--out-index", help="Output CSV with embedding metadata.")
    parser.add_argument("-d", "--out-dir", help="Directory for .npy embedding files.")
    parser.add_argument("--model-name", help="Override the TTS speaker embedding model from config.")
    parser.add_argument("--expected-dim", type=int, help="Override the expected TTS embedding dimension from config.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config_path, config = load_cli_config(args.config)
    speaker_selection_path = args.speaker_selection or str(
        resolve_data_path(config, "speaker_selection_path", DEFAULT_SPEAKER_SELECTION_PATH)
    )
    out_index = args.out_index or str(
        resolve_data_path(config, "speaker_embeddings_index", DEFAULT_EMBEDDINGS_INDEX_PATH)
    )
    out_dir = args.out_dir or str(Path(out_index).parent if out_index else DEFAULT_EMBEDDINGS_DIR)
    frame = extract_speaker_embeddings(
        config_path=config_path,
        speaker_selection_path=speaker_selection_path,
        out_index=out_index,
        out_dir=out_dir,
        model_name=args.model_name,
        expected_dim=args.expected_dim,
    )
    print(f"Wrote {len(frame)} speaker embeddings to {out_index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
