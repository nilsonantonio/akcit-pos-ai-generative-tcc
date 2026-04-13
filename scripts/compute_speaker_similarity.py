#!/usr/bin/env python3
"""Compute speaker similarity scores."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tcc_audio.audio_metrics import build_arg_parser, compute_speaker_similarity


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser("speaker similarity").parse_args(argv)
    compute_speaker_similarity(args.samples, args.out)
    print(f"Updated speaker similarity in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

