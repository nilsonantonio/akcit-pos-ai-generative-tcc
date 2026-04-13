"""Build and summarize the human evaluation package."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import pandas as pd

from tcc_audio.io import ensure_parent_dir, read_csv


CORE_COMPARISONS = [
    ("speecht5_zero_shot", "speecht5_few_shot_decoder_ft"),
    ("speecht5_zero_shot", "speecht5_lora"),
    ("speecht5_few_shot_decoder_ft", "speecht5_lora"),
]


def build_human_eval_pack(
    samples_path: str | Path,
    out_dir: str | Path,
    seed: int = 42,
) -> pd.DataFrame:
    rng = random.Random(seed)
    samples = read_csv(samples_path)
    samples = samples[samples["status"].astype(str).str.lower().eq("ok")].copy()
    out_root = Path(out_dir)
    audio_dir = out_root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    batch_index = 1
    for left_condition, right_condition in CORE_COMPARISONS:
        left = samples[samples["condition"].eq(left_condition)]
        right = samples[samples["condition"].eq(right_condition)]
        paired = left.merge(
            right,
            on=["speaker_id", "prompt_id", "text_variant"],
            suffixes=("_left", "_right"),
        )
        for _, row in paired.iterrows():
            if rng.random() < 0.5:
                left_sample_id, right_sample_id = row["sample_id_left"], row["sample_id_right"]
                left_condition_name, right_condition_name = left_condition, right_condition
                left_audio, right_audio = row["audio_path_left"], row["audio_path_right"]
            else:
                left_sample_id, right_sample_id = row["sample_id_right"], row["sample_id_left"]
                left_condition_name, right_condition_name = right_condition, left_condition
                left_audio, right_audio = row["audio_path_right"], row["audio_path_left"]

            left_target = audio_dir / f"{left_sample_id}.wav"
            right_target = audio_dir / f"{right_sample_id}.wav"
            if Path(left_audio).exists():
                shutil.copy2(left_audio, left_target)
            if Path(right_audio).exists():
                shutil.copy2(right_audio, right_target)

            rows.append(
                {
                    "batch_id": f"batch_{batch_index:04d}",
                    "prompt_id": row["prompt_id"],
                    "speaker_id": row["speaker_id"],
                    "left_sample_id": left_sample_id,
                    "right_sample_id": right_sample_id,
                    "left_condition": left_condition_name,
                    "right_condition": right_condition_name,
                    "mos_left": "",
                    "mos_right": "",
                    "smos_left": "",
                    "smos_right": "",
                    "preferred_condition": "",
                    "notes": "",
                }
            )
            batch_index += 1

    pack = pd.DataFrame(rows)
    ensure_parent_dir(out_root / "human_eval_pack.csv")
    pack.to_csv(out_root / "human_eval_pack.csv", index=False)
    return pack


def import_human_eval_results(results_path: str | Path, out_path: str | Path) -> pd.DataFrame:
    results = read_csv(results_path)
    summary_rows: list[dict[str, object]] = []

    for side, condition_column, mos_column, smos_column in [
        ("left", "left_condition", "mos_left", "smos_left"),
        ("right", "right_condition", "mos_right", "smos_right"),
    ]:
        grouped = (
            results.groupby(condition_column, dropna=False)
            .agg(
                mos_mean=(mos_column, lambda values: pd.to_numeric(values, errors="coerce").mean()),
                smos_mean=(smos_column, lambda values: pd.to_numeric(values, errors="coerce").mean()),
                count=("batch_id", "count"),
            )
            .reset_index()
            .rename(columns={condition_column: "condition"})
        )
        grouped["side"] = side
        summary_rows.append(grouped)

    preference = (
        results.groupby("preferred_condition", dropna=False)
        .size()
        .reset_index(name="preferred_count")
        .rename(columns={"preferred_condition": "condition"})
    )
    summary = pd.concat(summary_rows, ignore_index=True)
    summary = summary.merge(preference, on="condition", how="left")
    ensure_parent_dir(out_path)
    summary.to_csv(out_path, index=False)
    return summary


def build_pack_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the human evaluation package.")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def build_import_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import filled human evaluation results.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    return parser

