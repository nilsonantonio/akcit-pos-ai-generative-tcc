#!/usr/bin/env python3
"""Compute NISQA quality scores."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tcc_audio.cli_defaults import (
    DEFAULT_NISQA_PATH,
    DEFAULT_SAMPLES_PATH,
    load_cli_config,
    resolve_samples_path,
)
from tcc_audio.audio_metrics import build_arg_parser, compute_nisqa


def main(argv: list[str] | None = None) -> int:
    args = build_nisqa_arg_parser().parse_args(argv)
    _, config = load_cli_config(args.config)
    samples_path = args.samples or str(resolve_samples_path(config) if config else DEFAULT_SAMPLES_PATH)
    out_path = args.out or samples_path
    nisqa_path = args.nisqa_path or str(DEFAULT_NISQA_PATH)
    compute_nisqa(samples_path, out_path, nisqa_path=nisqa_path)
    print(f"Updated NISQA in {out_path}")
    return 0


def build_nisqa_arg_parser():
    parser = build_arg_parser("NISQA")
    parser.add_argument("-c", "--config", help="Optional experiment config used to resolve samples.csv.")
    parser.add_argument("--nisqa-path", help="Local path to a NISQA checkout or package root.")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
