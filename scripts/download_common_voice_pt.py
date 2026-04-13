#!/usr/bin/env python3
"""CLI wrapper for Common Voice PT download and extraction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tcc_audio.common_voice_download import main


if __name__ == "__main__":
    raise SystemExit(main())
