#!/usr/bin/env python3
"""Build the blind human evaluation package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tcc_audio.human_eval import build_human_eval_pack, build_pack_arg_parser


def main(argv: list[str] | None = None) -> int:
    args = build_pack_arg_parser().parse_args(argv)
    pack = build_human_eval_pack(args.samples, args.out_dir, args.seed)
    print(f"Wrote {len(pack)} human evaluation rows to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

