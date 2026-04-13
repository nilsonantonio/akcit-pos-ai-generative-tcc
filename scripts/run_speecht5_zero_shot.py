#!/usr/bin/env python3
"""Run SpeechT5 zero-shot inference."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tcc_audio.speecht5_runner import build_arg_parser, run_condition_inference


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser("SpeechT5 zero-shot").parse_args(argv)
    run_condition_inference(args.config, args.samples, "speecht5_zero_shot", limit=args.limit)
    print("SpeechT5 zero-shot inference completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

