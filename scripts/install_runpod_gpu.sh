#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-linux-gpu.txt

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg nao encontrado. Instale no template base do RunPod ou use o Docker GPU."
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi nao encontrado. O template do RunPod precisa expor a GPU NVIDIA." >&2
  exit 1
fi

python3 - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("CUDA nao esta disponivel dentro do pod. Verifique o template/driver do RunPod.", file=sys.stderr)
    raise SystemExit(1)

print("CUDA disponivel:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
PY

echo "Ambiente RunPod preparado."
