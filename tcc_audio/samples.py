"""Create the evaluation samples ledger from the run matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tcc_audio.io import ensure_parent_dir, read_csv
from tcc_audio.runtime import load_speaker_reference_map
from tcc_audio.schema import EVAL_SAMPLES_REQUIRED_COLUMNS, SAMPLES_OPTIONAL_COLUMNS


def initialize_samples(
    run_matrix_path: str | Path,
    out_path: str | Path,
    audio_base_dir: str | Path = "artifacts/audio",
    speaker_selection_path: str | Path | None = None,
    speaker_embeddings_path: str | Path | None = None,
) -> pd.DataFrame:
    run_matrix = read_csv(run_matrix_path)
    required = ["run_id", "condition", "speaker_id", "prompt_id", "text_variant", "target_text"]
    missing = [column for column in required if column not in run_matrix.columns]
    if missing:
        raise ValueError(f"Missing run matrix columns: {', '.join(missing)}")

    speaker_reference_map = {}
    if speaker_selection_path:
        speaker_reference_map = load_speaker_reference_map(speaker_selection_path)

    speaker_embeddings: dict[str, dict[str, str]] = {}
    if speaker_embeddings_path:
        embeddings_frame = read_csv(speaker_embeddings_path)
        if "speaker_id" not in embeddings_frame.columns:
            raise ValueError("speaker_embeddings CSV must contain speaker_id")
        speaker_embeddings = {row["speaker_id"]: row.to_dict() for _, row in embeddings_frame.iterrows()}

    rows: list[dict[str, object]] = []
    for _, run in run_matrix.iterrows():
        audio_path = (
            Path(audio_base_dir)
            / run["condition"]
            / run["speaker_id"]
            / f"{run['prompt_id']}__{run['text_variant']}.wav"
        )
        row = {column: "" for column in EVAL_SAMPLES_REQUIRED_COLUMNS}
        for column in SAMPLES_OPTIONAL_COLUMNS:
            row[column] = ""
        speaker_meta = speaker_reference_map.get(run["speaker_id"], {})
        embedding_meta = speaker_embeddings.get(run["speaker_id"], {})
        row.update(
            {
                "sample_id": run["run_id"],
                "run_id": run["run_id"],
                "condition": run["condition"],
                "speaker_id": run["speaker_id"],
                "prompt_id": run["prompt_id"],
                "text_variant": run["text_variant"],
                "target_text": run["target_text"],
                "audio_path": str(audio_path),
                "reference_audio_path": speaker_meta.get("reference_audio", ""),
                "speaker_embedding_path": embedding_meta.get("speaker_embedding_path", ""),
                "model_name": run.get("model_name", ""),
                "status": "pending",
                "failure_reason": "",
                "lora_gate_status": run.get("lora_gate_fallback", "") if run["condition"] == "speecht5_lora" else "",
            }
        )
        rows.append(row)

    samples = pd.DataFrame(rows)
    ensure_parent_dir(out_path)
    samples.to_csv(out_path, index=False)
    return samples


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Initialize artifacts/evaluation/samples.csv from run_matrix.csv.")
    parser.add_argument("--run-matrix", required=True, help="Path to artifacts/run_matrix.csv.")
    parser.add_argument("--out", required=True, help="Output samples.csv path.")
    parser.add_argument("--audio-base-dir", default="artifacts/audio")
    parser.add_argument("--speaker-selection", help="Optional speaker_selection.csv for reference_audio lookup.")
    parser.add_argument("--speaker-embeddings", help="Optional speaker_embeddings.csv for embedding lookup.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    samples = initialize_samples(
        args.run_matrix,
        args.out,
        args.audio_base_dir,
        speaker_selection_path=args.speaker_selection,
        speaker_embeddings_path=args.speaker_embeddings,
    )
    print(f"Wrote {len(samples)} sample rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
