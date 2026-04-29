#!/usr/bin/env python3
"""Run SpeechT5 inference over materialized samples."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tcc_audio.cli_defaults import DEFAULT_SAMPLES_PATH, load_cli_config, resolve_samples_path
from tcc_audio.speecht5_runner import run_condition_inference


def build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Generate audio for materialized SpeechT5 samples.")
    parser.add_argument("-c", "--config")
    parser.add_argument("-s", "--samples")
    parser.add_argument("-C", "--condition", action="append", default=[])
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--speaker-id")
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--prompt-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config_path, config = load_cli_config(args.config)
    if config_path is None:
        raise SystemExit("Config not found. Pass --config or keep configs/speecht5_minimal.yaml available.")
    samples_path = args.samples or str(resolve_samples_path(config) if config else DEFAULT_SAMPLES_PATH)
    if not args.condition:
        raise SystemExit("At least one --condition is required.")
    for condition_id in args.condition:
        run_condition_inference(
            config_path=config_path,
            samples_path=samples_path,
            condition_id=condition_id,
            checkpoint_dir=args.checkpoint_dir,
            speaker_id=args.speaker_id,
            checkpoint_step=args.checkpoint_step,
            prompt_ids=args.prompt_id,
            only_pending=not args.force,
            force=args.force,
            limit=args.limit,
        )
    print(f"Updated generated audio rows in {samples_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
