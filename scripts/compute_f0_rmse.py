#!/usr/bin/env python3
"""Compute F0 RMSE between generated audio and reference audio."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tcc_audio.cli_defaults import DEFAULT_SAMPLES_PATH, load_cli_config, resolve_samples_path
from tcc_audio.audio_metrics import build_arg_parser, compute_f0_rmse


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser("F0 RMSE")
    parser.add_argument("-c", "--config", help="Optional experiment config used to resolve samples.csv.")
    args = parser.parse_args(argv)
    _, config = load_cli_config(args.config)
    samples_path = args.samples or str(resolve_samples_path(config) if config else DEFAULT_SAMPLES_PATH)
    out_path = args.out or samples_path
    compute_f0_rmse(samples_path, out_path)
    print(f"Updated F0 RMSE in {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
