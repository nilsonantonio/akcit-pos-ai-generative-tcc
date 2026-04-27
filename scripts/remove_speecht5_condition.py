#!/usr/bin/env python3
"""Remove a SpeechT5 LoRA condition and all related artifacts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tcc_audio.training_cleanup import build_remove_condition_arg_parser, remove_condition


def main(argv: list[str] | None = None) -> int:
    args = build_remove_condition_arg_parser().parse_args(argv)
    if len(args.condition) != 1:
        raise SystemExit("remove_speecht5_condition accepts exactly one --condition.")
    condition_id = args.condition[0]
    result = remove_condition(
        config_path=args.config,
        condition_id=condition_id,
        samples_path=args.samples,
        checkpoint_dir=args.checkpoint_dir,
        audio_base_dir=args.audio_base_dir,
        evaluation_dir=args.evaluation_dir,
        report_assets_dir=args.report_assets_dir,
        bypass=args.bypass,
    )
    print(
        "Removed SpeechT5 condition "
        f"{condition_id}. "
        f"Removed {result.cleanup.removed_materialized_rows} materialized rows "
        f"and {result.removed_base_rows} base rows."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
