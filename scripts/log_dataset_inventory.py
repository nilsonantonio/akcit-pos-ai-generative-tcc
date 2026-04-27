#!/usr/bin/env python3
"""CLI wrapper for dataset inventory reporting."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tcc_audio.dataset_inventory import main


if __name__ == "__main__":
    raise SystemExit(main())
