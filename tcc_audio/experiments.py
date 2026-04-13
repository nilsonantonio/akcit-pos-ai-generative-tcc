"""Generate the official run matrix from the YAML experiment config."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tcc_audio.io import ensure_parent_dir, read_csv, read_yaml


def _bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column].astype(str).str.lower().eq("true")


def generate_run_matrix(config_path: str | Path, out_path: str | Path | None = None) -> pd.DataFrame:
    config = read_yaml(config_path)
    data_config = config.get("data", {})
    conditions = config.get("conditions", [])
    prompts_path = data_config.get("prompts_path")
    if not prompts_path:
        raise ValueError("Config must define data.prompts_path")

    prompts = read_csv(prompts_path)
    speaker_count = int(data_config.get("speaker_target_count", 4))
    speakers = [f"speaker_{index:02d}" for index in range(1, speaker_count + 1)]

    rows: list[dict[str, object]] = []
    for condition in conditions:
        condition_id = condition["id"]
        normalization_modes = condition.get("normalization_modes", ["normalized"])
        speaker_subset_count = condition.get("speaker_subset_count", speaker_count)
        prompt_subset_count = condition.get("prompt_subset_count")

        selected_speakers = speakers[: int(speaker_subset_count)]
        selected_prompts = prompts.copy()
        if prompt_subset_count:
            commercial_mask = _bool_series(selected_prompts, "include_commercial_subset")
            subset = selected_prompts[commercial_mask].head(int(prompt_subset_count))
            if len(subset) < int(prompt_subset_count):
                subset = selected_prompts.head(int(prompt_subset_count))
            selected_prompts = subset

        for speaker_id in selected_speakers:
            for _, prompt in selected_prompts.iterrows():
                for text_variant in normalization_modes:
                    target_text = prompt["raw_text"] if text_variant == "raw" else prompt["normalized_text"]
                    run_id = f"{condition_id}__{speaker_id}__{prompt['prompt_id']}__{text_variant}"
                    rows.append(
                        {
                            "run_id": run_id,
                            "condition": condition_id,
                            "condition_label": condition.get("label", condition_id),
                            "regime": condition.get("regime", ""),
                            "model_name": condition.get("model_name", ""),
                            "train_strategy": condition.get("train_strategy", ""),
                            "uses_speaker_embeddings": bool(condition.get("uses_speaker_embeddings", False)),
                            "official_hypothesis_arm": bool(condition.get("official_hypothesis_arm", True)),
                            "speaker_id": speaker_id,
                            "prompt_id": prompt["prompt_id"],
                            "category": prompt["category"],
                            "text_variant": text_variant,
                            "target_text": target_text,
                            "minutes_per_speaker": condition.get(
                                "minutes_per_speaker", data_config.get("minutes_per_speaker", "")
                            ),
                            "reference_clip_min_s": data_config.get("reference_clip_len_s", {}).get("min", ""),
                            "reference_clip_max_s": data_config.get("reference_clip_len_s", {}).get("max", ""),
                            "sample_rate": data_config.get("sample_rate", ""),
                            "compute_target": config.get("project", {}).get("compute_target", ""),
                            "seed": config.get("project", {}).get("seed", ""),
                            "lora_gate_preferred": condition.get("lora_gate", {}).get("preferred", ""),
                            "lora_gate_fallback": condition.get("lora_gate", {}).get("fallback", ""),
                        }
                    )

    matrix = pd.DataFrame(rows)
    if out_path:
        ensure_parent_dir(out_path)
        matrix.to_csv(out_path, index=False)
    return matrix


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the SpeechT5 minimal run matrix.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    parser.add_argument("--out", help="Output CSV path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    matrix = generate_run_matrix(args.config, args.out)
    print(f"Generated {len(matrix)} runs")
    if args.out:
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
