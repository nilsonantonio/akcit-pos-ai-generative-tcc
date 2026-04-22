#!/usr/bin/env python3
"""Compute speaker similarity scores."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tcc_audio.audio_metrics import build_arg_parser, compute_speaker_similarity


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser("speaker similarity")
    parser.add_argument("--config", help="Optional experiment config used to resolve evaluation.speaker_similarity_model.")
    parser.add_argument("--model-name", help="Override the speaker similarity model from config.")
    args = parser.parse_args(argv)
    compute_speaker_similarity(args.samples, args.out, model_name=args.model_name, config_path=args.config)
    print(f"Updated speaker similarity in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
