#!/usr/bin/env python3
"""Backfill train_gpu_hours and cost_usd from saved training totals or sample timestamps."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tcc_audio.cost_backfill import main


if __name__ == "__main__":
    raise SystemExit(main())
