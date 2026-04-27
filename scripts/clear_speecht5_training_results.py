#!/usr/bin/env python3
"""Clear SpeechT5 LoRA training artifacts for one or more conditions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tcc_audio.training_cleanup import build_cleanup_arg_parser, cleanup_training_results


def main(argv: list[str] | None = None) -> int:
    args = build_cleanup_arg_parser().parse_args(argv)
    result = cleanup_training_results(
        config_path=args.config,
        condition_ids=args.condition,
        samples_path=args.samples,
        checkpoint_dir=args.checkpoint_dir,
        audio_base_dir=args.audio_base_dir,
        evaluation_dir=args.evaluation_dir,
        report_assets_dir=args.report_assets_dir,
        bypass=args.bypass,
    )
    print(
        "Cleared SpeechT5 training results for "
        f"{', '.join(result.condition_ids)}. "
        f"Removed {result.removed_materialized_rows} materialized sample rows."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
