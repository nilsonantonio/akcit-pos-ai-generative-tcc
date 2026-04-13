#!/usr/bin/env python3
"""Import completed human evaluation results."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tcc_audio.human_eval import build_import_arg_parser, import_human_eval_results


def main(argv: list[str] | None = None) -> int:
    args = build_import_arg_parser().parse_args(argv)
    summary = import_human_eval_results(args.results, args.out)
    print(f"Wrote {len(summary)} human evaluation summary rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

