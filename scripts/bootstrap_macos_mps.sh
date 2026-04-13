#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-macos-mps.txt

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg nao encontrado. Instale com: brew install ffmpeg" >&2
  exit 1
fi

python - <<'PY'
import sys
import torch

if not torch.backends.mps.is_available():
    print("MPS nao esta disponivel neste Mac. O caminho oficial exige GPU via PyTorch MPS no host.", file=sys.stderr)
    raise SystemExit(1)

print("MPS disponivel:", torch.backends.mps.is_available())
PY

python -B -c "from tests.smoke_test import *; test_prompt_file_has_expected_contract(); test_run_matrix_generation(); test_speaker_selection_from_curated_metadata(); test_metric_aggregation_contract(); test_wer_computation(); test_prepare_common_voice_metadata(); test_human_eval_and_report_assets(); print('smoke tests passed')"

echo "Bootstrap macOS MPS finalizado."

