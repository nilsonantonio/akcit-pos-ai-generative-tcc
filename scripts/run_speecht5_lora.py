#!/usr/bin/env python3
"""Run SpeechT5 LoRA fine-tuning and inference."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tcc_audio.speecht5_runner import build_arg_parser, run_lora_pipeline


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser("SpeechT5 LoRA").parse_args(argv)
    if not args.manifest or not args.checkpoint_dir:
        raise SystemExit("--manifest and --checkpoint-dir are required for LoRA training")
    run_lora_pipeline(
        config_path=args.config,
        manifest_path=args.manifest,
        samples_path=args.samples,
        checkpoint_dir=args.checkpoint_dir,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        per_device_batch_size=args.per_device_batch_size,
        gpu_hourly_rate=args.gpu_hourly_rate,
    )
    print("SpeechT5 LoRA pipeline completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

