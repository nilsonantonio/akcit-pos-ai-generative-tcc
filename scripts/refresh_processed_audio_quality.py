#!/usr/bin/env python3
"""Refresh audio quality metrics on processed WAV metadata."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tcc_audio.processed_audio_quality import main


if __name__ == "__main__":
    raise SystemExit(main())
