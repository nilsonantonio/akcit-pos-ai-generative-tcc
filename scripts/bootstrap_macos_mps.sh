#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

python3 -m venv .venv
source .venv/bin/activate

python - <<'PY'
import sys

try:
    import lzma  # noqa: F401
    import _lzma  # noqa: F401
except ModuleNotFoundError:
    print(
        "Este Python foi compilado sem suporte a lzma (_lzma). "
        "No macOS com Homebrew/asdf: rode `brew install xz`, reinstale o Python e recrie a .venv.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY

python -m pip install --upgrade pip
python -m pip install -r requirements-macos-mps.txt

if ! command -v git >/dev/null 2>&1; then
  echo "git nao encontrado. Instale git para baixar o checkout local do NISQA." >&2
  exit 1
fi

if [ ! -d "${PROJECT_ROOT}/NISQA/nisqa" ]; then
  git clone https://github.com/gabrielmittag/NISQA.git "${PROJECT_ROOT}/NISQA"
else
  echo "NISQA ja presente em ${PROJECT_ROOT}/NISQA"
fi

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

echo "Bootstrap macOS MPS finalizado. NISQA disponivel em ${PROJECT_ROOT}/NISQA"
