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

python -B tests/smoke_test.py extended

echo "Bootstrap macOS MPS finalizado."
