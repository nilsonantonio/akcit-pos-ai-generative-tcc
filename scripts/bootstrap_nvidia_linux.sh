#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-linux-gpu.txt

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg nao encontrado. Instale via apt/yum no host ou use o container NVIDIA." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi nao encontrado. Este bootstrap exige Linux com GPU NVIDIA e driver instalado." >&2
  exit 1
fi

python3 - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("CUDA nao esta disponivel. O caminho oficial Docker/NVIDIA exige GPU CUDA visivel.", file=sys.stderr)
    raise SystemExit(1)

print("CUDA disponivel:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
PY

python3 -B -c "from tests.smoke_test import *; test_prompt_file_has_expected_contract(); test_run_matrix_generation(); test_speaker_selection_from_curated_metadata(); test_metric_aggregation_contract(); test_wer_computation(); test_prepare_common_voice_metadata(); test_human_eval_and_report_assets(); print('smoke tests passed')"

echo "Bootstrap Linux NVIDIA finalizado."

