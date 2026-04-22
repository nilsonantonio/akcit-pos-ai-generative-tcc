#!/usr/bin/env python3
"""Simulate human evaluation for testing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tcc_audio.human_eval import simulate_human_eval, build_simulate_arg_parser


def main(argv: list[str] | None = None) -> int:
    args = build_simulate_arg_parser().parse_args(argv)
    simulate_human_eval(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
