#!/usr/bin/env python3
"""Run optional Parler-TTS reference generation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tcc_audio.parler_reference import main


if __name__ == "__main__":
    raise SystemExit(main())

