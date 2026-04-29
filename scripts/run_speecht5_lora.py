#!/usr/bin/env python3
"""Run SpeechT5 LoRA fine-tuning only."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tcc_audio.cli_defaults import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_MANIFEST_PATH,
    load_cli_config,
    resolve_data_path,
)
from tcc_audio.speecht5_runner import build_lora_arg_parser, run_lora_pipeline


def main(argv: list[str] | None = None) -> int:
    args = build_lora_arg_parser().parse_args(argv)
    config_path, config = load_cli_config(args.config)
    if config_path is None:
        raise SystemExit("Config not found. Pass --config or keep configs/speecht5_minimal.yaml available.")
    manifest_path = args.manifest or str(resolve_data_path(config, "manifest_path", DEFAULT_MANIFEST_PATH))
    checkpoint_dir = args.checkpoint_dir or str(DEFAULT_CHECKPOINT_DIR)
    run_lora_pipeline(
        config_path=config_path,
        manifest_path=manifest_path,
        checkpoint_dir=checkpoint_dir,
        gpu_hourly_rate=args.gpu_hourly_rate,
        condition_ids=args.condition,
        checkpoint_run_ts=args.checkpoint_run_ts,
        resume_train=args.resume_train,
        speaker_ids=args.speaker_id,
    )
    print("SpeechT5 LoRA training completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
