"""Create and materialize the evaluation samples ledger."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from tcc_audio.cli_defaults import (
    DEFAULT_AUDIO_BASE_DIR,
    DEFAULT_EMBEDDINGS_INDEX_PATH,
    DEFAULT_RUN_MATRIX_PATH,
    DEFAULT_SAMPLES_PATH,
    DEFAULT_SPEAKER_SELECTION_PATH,
    load_cli_config,
    resolve_data_path,
    resolve_deliverable_path,
)
from tcc_audio.io import ensure_parent_dir, read_csv
from tcc_audio.runtime import load_samples, load_speaker_reference_map, save_samples_atomic, stringify_csv_value
from tcc_audio.schema import EVAL_SAMPLES_REQUIRED_COLUMNS, SAMPLES_OPTIONAL_COLUMNS


def empty_samples_frame() -> pd.DataFrame:
    rows = [{column: "" for column in EVAL_SAMPLES_REQUIRED_COLUMNS + SAMPLES_OPTIONAL_COLUMNS}]
    return pd.DataFrame(rows).iloc[0:0].copy()


def initialize_samples(
    run_matrix_path: str | Path,
    out_path: str | Path,
    audio_base_dir: str | Path = "artifacts/audio",
    speaker_selection_path: str | Path | None = None,
    speaker_embeddings_path: str | Path | None = None,
) -> pd.DataFrame:
    del run_matrix_path, audio_base_dir, speaker_selection_path, speaker_embeddings_path
    samples = empty_samples_frame()
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
    run_matrix_path: str | Path,
    condition_id: str,
    checkpoint_step: int,
    checkpoint_path: str | Path,
    checkpoint_run_ts: str,
    training_scope: str,
    training_unit: str,
    audio_base_dir: str | Path,
    total_train_gpu_hours: float,
    gpu_hourly_rate: float,
    speaker_selection_path: str | Path | None = None,
    speaker_embeddings_path: str | Path | None = None,
    speaker_ids: Iterable[str] | None = None,
    prompt_ids: Iterable[str] | None = None,
) -> list[str]:
    samples = load_samples(samples_path)
    run_matrix = read_csv(run_matrix_path)
    required = ["run_id", "condition", "speaker_id", "prompt_id", "text_variant", "target_text"]
    missing = [column for column in required if column not in run_matrix.columns]
    if missing:
        raise ValueError(f"Missing run matrix columns: {', '.join(missing)}")

    selected_speakers = {str(value).strip() for value in (speaker_ids or []) if str(value).strip()}
    selected_prompts = {str(value).strip() for value in (prompt_ids or []) if str(value).strip()}

    base_rows = run_matrix[run_matrix["condition"].eq(condition_id)].copy()
    if training_scope == "per_speaker":
        base_rows = base_rows[base_rows["speaker_id"].eq(training_unit)].copy()
    if selected_speakers:
        base_rows = base_rows[base_rows["speaker_id"].isin(selected_speakers)].copy()
    if selected_prompts:
        base_rows = base_rows[base_rows["prompt_id"].isin(selected_prompts)].copy()
    if base_rows.empty:
        return []

    speaker_reference_map = load_speaker_reference_map(speaker_selection_path) if speaker_selection_path else {}
    speaker_embeddings: dict[str, dict[str, str]] = {}
    if speaker_embeddings_path:
        embeddings_frame = read_csv(speaker_embeddings_path)
        if "speaker_id" not in embeddings_frame.columns:
            raise ValueError("speaker_embeddings CSV must contain speaker_id")
        speaker_embeddings = {row["speaker_id"]: row.to_dict() for _, row in embeddings_frame.iterrows()}

    train_gpu_hours_per_sample = total_train_gpu_hours / max(len(base_rows), 1)
    cost_usd_per_sample = train_gpu_hours_per_sample * gpu_hourly_rate
    checkpoint_label = f"{condition_id}@step{checkpoint_step}"
    existing_sample_ids = set(samples["sample_id"].astype(str).tolist())

    rows: list[dict[str, object]] = []
    sample_ids: list[str] = []
    for _, row in base_rows.iterrows():
        sample_id = f"{row['run_id']}__{checkpoint_run_ts}__step{checkpoint_step}"
        if sample_id in existing_sample_ids:
            sample_ids.append(sample_id)
            continue
        speaker_id = str(row["speaker_id"])
        speaker_meta = speaker_reference_map.get(speaker_id, {})
        embedding_meta = speaker_embeddings.get(speaker_id, {})
        materialized = {column: "" for column in EVAL_SAMPLES_REQUIRED_COLUMNS + SAMPLES_OPTIONAL_COLUMNS}
        materialized.update(
            {
                "sample_id": sample_id,
                "run_id": row["run_id"],
                "condition": row["condition"],
                "speaker_id": speaker_id,
                "prompt_id": row["prompt_id"],
                "text_variant": row["text_variant"],
                "target_text": row["target_text"],
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
                "reference_audio_path": speaker_meta.get("reference_audio", ""),
                "speaker_embedding_path": embedding_meta.get("speaker_embedding_path", ""),
                "model_name": row.get("model_name", checkpoint_label),
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

    if not rows:
        return sample_ids
    combined = pd.concat([samples, pd.DataFrame(rows)], ignore_index=True)
    save_samples_atomic(combined, samples_path)
    return sample_ids


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Initialize artifacts/evaluation/samples.csv from run_matrix.csv.")
    parser.add_argument("-c", "--config", help="Optional experiment config used to resolve run_matrix/samples paths.")
    parser.add_argument("-r", "--run-matrix", help="Path to artifacts/run_matrix.csv.")
    parser.add_argument("-o", "--out", help="Output samples.csv path.")
    parser.add_argument("-a", "--audio-base-dir", default=str(DEFAULT_AUDIO_BASE_DIR))
    parser.add_argument("-s", "--speaker-selection", help="Optional speaker_selection.csv for reference_audio lookup.")
    parser.add_argument("-e", "--speaker-embeddings", help="Optional TTS speaker_embeddings.csv for synthesis embedding lookup.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _, config = load_cli_config(args.config)
    run_matrix_path = args.run_matrix or str(resolve_deliverable_path(config, "run_matrix", DEFAULT_RUN_MATRIX_PATH))
    out_path = args.out or str(resolve_deliverable_path(config, "samples", DEFAULT_SAMPLES_PATH))
    speaker_selection_path = args.speaker_selection or str(
        resolve_data_path(config, "speaker_selection_path", DEFAULT_SPEAKER_SELECTION_PATH)
    )
    speaker_embeddings_path = args.speaker_embeddings or str(
        resolve_data_path(config, "speaker_embeddings_index", DEFAULT_EMBEDDINGS_INDEX_PATH)
    )
    samples = initialize_samples(
        run_matrix_path,
        out_path,
        args.audio_base_dir,
        speaker_selection_path=speaker_selection_path,
        speaker_embeddings_path=speaker_embeddings_path,
    )
    print(f"Wrote {len(samples)} sample rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
