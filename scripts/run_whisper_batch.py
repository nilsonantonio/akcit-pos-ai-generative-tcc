#!/usr/bin/env python3
"""Run Whisper ASR in batch mode."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tcc_audio.whisper_batch import main


if __name__ == "__main__":
    raise SystemExit(main())

