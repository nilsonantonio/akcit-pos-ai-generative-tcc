"""Create the evaluation samples ledger from the run matrix."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from tcc_audio.io import ensure_parent_dir, read_csv
from tcc_audio.runtime import load_samples, load_speaker_reference_map, save_samples, stringify_csv_value
from tcc_audio.schema import EVAL_SAMPLES_REQUIRED_COLUMNS, SAMPLES_OPTIONAL_COLUMNS


def initialize_samples(
    run_matrix_path: str | Path,
    out_path: str | Path,
    audio_base_dir: str | Path = "artifacts/audio",
    speaker_selection_path: str | Path | None = None,
    speaker_embeddings_path: str | Path | None = None,
) -> pd.DataFrame:
    del audio_base_dir
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
                "audio_path": "",
                "reference_audio_path": speaker_meta.get("reference_audio", ""),
                # This field stores the embedding consumed by the TTS synthesis step.
                "speaker_embedding_path": embedding_meta.get("speaker_embedding_path", ""),
                "model_name": run.get("model_name", ""),
                "status": "pending",
                "failure_reason": "",
                "checkpoint_label": "",
                "checkpoint_step": "",
                "checkpoint_path": "",
                "checkpoint_run_ts": "",
                "training_scope": run.get("training_scope", "per_speaker"),
                "training_unit": "",
                "lora_gate_status": "",
            }
        )
        rows.append(row)

    samples = pd.DataFrame(rows)
    ensure_parent_dir(out_path)
    samples.to_csv(out_path, index=False)
    return samples


def _is_materialized_row(frame: pd.DataFrame) -> pd.Series:
    return frame["checkpoint_step"].astype(str).str.strip().ne("")


def reset_condition_checkpoint_rows(samples: pd.DataFrame, condition_id: str) -> pd.DataFrame:
    materialized = _is_materialized_row(samples)
    return samples[~(samples["condition"].eq(condition_id) & materialized)].copy()


def _build_checkpoint_audio_path(
    audio_base_dir: str | Path,
    condition_id: str,
    checkpoint_run_ts: str,
    training_scope: str,
    training_unit: str,
    checkpoint_step: int,
    speaker_id: str,
    prompt_id: str,
    text_variant: str,
) -> Path:
    base = Path(audio_base_dir) / condition_id / checkpoint_run_ts / training_unit / f"step_{checkpoint_step}"
    if training_scope == "unique":
        base = base / speaker_id
    return base / f"{prompt_id}__{text_variant}.wav"


def materialize_checkpoint_samples(
    samples_path: str | Path,
    condition_id: str,
    checkpoint_step: int,
    checkpoint_path: str | Path,
    checkpoint_run_ts: str,
    training_scope: str,
    training_unit: str,
    audio_base_dir: str | Path,
    total_train_gpu_hours: float,
    gpu_hourly_rate: float,
) -> list[str]:
    samples = load_samples(samples_path)
    base_mask = samples["condition"].eq(condition_id) & ~_is_materialized_row(samples)
    if training_scope == "per_speaker":
        base_mask &= samples["speaker_id"].eq(training_unit)
    base_rows = samples[base_mask].copy()
    if base_rows.empty:
        return []

    train_gpu_hours_per_sample = total_train_gpu_hours / max(len(base_rows), 1)
    cost_usd_per_sample = train_gpu_hours_per_sample * gpu_hourly_rate
    checkpoint_label = f"{condition_id}@step{checkpoint_step}"

    rows: list[dict[str, object]] = []
    sample_ids: list[str] = []
    for _, row in base_rows.iterrows():
        materialized = row.to_dict()
        sample_id = f"{row['run_id']}__{checkpoint_run_ts}__step{checkpoint_step}"
        materialized.update(
            {
                "sample_id": sample_id,
                "audio_path": str(
                    _build_checkpoint_audio_path(
                        audio_base_dir=audio_base_dir,
                        condition_id=condition_id,
                        checkpoint_run_ts=checkpoint_run_ts,
                        training_scope=training_scope,
                        training_unit=training_unit,
                        checkpoint_step=checkpoint_step,
                        speaker_id=row["speaker_id"],
                        prompt_id=row["prompt_id"],
                        text_variant=row["text_variant"],
                    )
                ),
                "checkpoint_label": checkpoint_label,
                "checkpoint_step": stringify_csv_value(checkpoint_step),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_run_ts": checkpoint_run_ts,
                "training_scope": training_scope,
                "training_unit": training_unit,
                "run_started_at": "",
                "run_finished_at": "",
                "failure_reason": "",
                "wer": "",
                "speaker_similarity": "",
                "nisqa": "",
                "f0_rmse": "",
                "rtf": "",
                "train_gpu_hours": stringify_csv_value(train_gpu_hours_per_sample),
                "inference_seconds": "",
                "cost_usd": stringify_csv_value(cost_usd_per_sample),
                "status": "pending",
                "lora_gate_status": "stable_lora",
                "asr_text": "",
                "total_train_gpu_hours": stringify_csv_value(total_train_gpu_hours),
            }
        )
        rows.append(materialized)
        sample_ids.append(sample_id)

    combined = pd.concat([samples, pd.DataFrame(rows)], ignore_index=True)
    save_samples(combined, samples_path)
    return sample_ids


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Initialize artifacts/evaluation/samples.csv from run_matrix.csv.")
    parser.add_argument("--run-matrix", required=True, help="Path to artifacts/run_matrix.csv.")
    parser.add_argument("--out", required=True, help="Output samples.csv path.")
    parser.add_argument("--audio-base-dir", default="artifacts/audio")
    parser.add_argument("--speaker-selection", help="Optional speaker_selection.csv for reference_audio lookup.")
    parser.add_argument("--speaker-embeddings", help="Optional TTS speaker_embeddings.csv for synthesis embedding lookup.")
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
