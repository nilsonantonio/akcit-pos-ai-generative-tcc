#!/usr/bin/env python3
"""Compute NISQA quality scores."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tcc_audio.audio_metrics import build_arg_parser, compute_nisqa


def main(argv: list[str] | None = None) -> int:
    args = build_nisqa_arg_parser().parse_args(argv)
    compute_nisqa(args.samples, args.out, nisqa_path=args.nisqa_path)
    print(f"Updated NISQA in {args.out}")
    return 0


def build_nisqa_arg_parser():
    parser = build_arg_parser("NISQA")
    parser.add_argument("--nisqa-path", help="Local path to a NISQA checkout or package root.")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
