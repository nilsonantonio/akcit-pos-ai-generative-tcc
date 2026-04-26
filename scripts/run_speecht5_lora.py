#!/usr/bin/env python3
"""Run SpeechT5 LoRA fine-tuning and inference."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tcc_audio.speecht5_runner import build_lora_arg_parser, run_lora_pipeline


def main(argv: list[str] | None = None) -> int:
    args = build_lora_arg_parser().parse_args(argv)
    run_lora_pipeline(
        config_path=args.config,
        manifest_path=args.manifest,
        samples_path=args.samples,
        checkpoint_dir=args.checkpoint_dir,
        gpu_hourly_rate=args.gpu_hourly_rate,
        condition_ids=args.condition,
        audio_base_dir=args.audio_base_dir,
    )
    print("SpeechT5 LoRA pipeline completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
