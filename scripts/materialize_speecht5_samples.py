#!/usr/bin/env python3
"""Materialize SpeechT5 samples from saved checkpoints."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tcc_audio.cli_defaults import (
    DEFAULT_AUDIO_BASE_DIR,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_RUN_MATRIX_PATH,
    DEFAULT_SAMPLES_PATH,
    load_cli_config,
    resolve_deliverable_path,
)
from tcc_audio.io import read_yaml
from tcc_audio.speecht5_runner import materialize_condition_samples


def build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Materialize samples.csv from saved SpeechT5 checkpoints.")
    parser.add_argument("-c", "--config")
    parser.add_argument("-s", "--samples")
    parser.add_argument("-r", "--run-matrix")
    parser.add_argument("-k", "--checkpoint-dir")
    parser.add_argument("-a", "--audio-base-dir", default=str(DEFAULT_AUDIO_BASE_DIR))
    parser.add_argument("-C", "--condition", action="append", default=[])
    parser.add_argument("--checkpoint-run-ts")
    parser.add_argument("--all-checkpoints", action="store_true")
    parser.add_argument("--checkpoint-step", action="append", type=int, default=[])
    parser.add_argument("--speaker-id", action="append", default=[])
    parser.add_argument("--prompt-id", action="append", default=[])
    parser.add_argument("--gpu-hourly-rate", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config_path, config = load_cli_config(args.config)
    if config_path is None:
        raise SystemExit("Config not found. Pass --config or keep configs/speecht5_minimal.yaml available.")
    resolved_config = read_yaml(config_path)
    samples_path = args.samples or str(resolve_deliverable_path(config, "samples", DEFAULT_SAMPLES_PATH))
    run_matrix_path = args.run_matrix or str(resolve_deliverable_path(config, "run_matrix", DEFAULT_RUN_MATRIX_PATH))
    checkpoint_dir = args.checkpoint_dir or str(DEFAULT_CHECKPOINT_DIR)

    materialized_total = 0
    for condition in resolved_config.get("conditions", []):
        if not isinstance(condition, dict):
            continue
        if args.condition and str(condition.get("id")) not in set(args.condition):
            continue
        materialized_total += len(
            materialize_condition_samples(
                config_path=config_path,
                samples_path=samples_path,
                run_matrix_path=run_matrix_path,
                checkpoint_dir=checkpoint_dir,
                condition=condition,
                checkpoint_run_ts=args.checkpoint_run_ts,
                best_only=not args.all_checkpoints,
                checkpoint_steps=args.checkpoint_step,
                speaker_ids=args.speaker_id,
                prompt_ids=args.prompt_id,
                audio_base_dir=args.audio_base_dir,
                gpu_hourly_rate_override=args.gpu_hourly_rate,
            )
        )
    print(f"Materialized {materialized_total} sample rows into {samples_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
