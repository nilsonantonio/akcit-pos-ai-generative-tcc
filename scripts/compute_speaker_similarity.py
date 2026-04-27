#!/usr/bin/env python3
"""Compute speaker similarity scores."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tcc_audio.cli_defaults import DEFAULT_SAMPLES_PATH, load_cli_config, resolve_samples_path
from tcc_audio.audio_metrics import build_arg_parser, compute_speaker_similarity


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser("speaker similarity")
    parser.add_argument("-c", "--config", help="Optional experiment config used to resolve evaluation.speaker_similarity_model.")
    parser.add_argument("--model-name", help="Override the speaker similarity model from config.")
    args = parser.parse_args(argv)
    config_path, config = load_cli_config(args.config)
    samples_path = args.samples or str(resolve_samples_path(config) if config else DEFAULT_SAMPLES_PATH)
    out_path = args.out or samples_path
    compute_speaker_similarity(samples_path, out_path, model_name=args.model_name, config_path=config_path)
    print(f"Updated speaker similarity in {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
