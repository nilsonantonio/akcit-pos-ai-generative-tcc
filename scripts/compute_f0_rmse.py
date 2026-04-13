#!/usr/bin/env python3
"""Compute F0 RMSE between generated audio and reference audio."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tcc_audio.audio_metrics import build_arg_parser, compute_f0_rmse


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser("F0 RMSE").parse_args(argv)
    compute_f0_rmse(args.samples, args.out)
    print(f"Updated F0 RMSE in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

