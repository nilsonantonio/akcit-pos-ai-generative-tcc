#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

python3 - <<'PY'
import sys

try:
    import lzma  # noqa: F401
    import _lzma  # noqa: F401
except ModuleNotFoundError:
    print(
        "Este Python foi compilado sem suporte a lzma (_lzma). "
        "Instale xz/liblzma-dev no host, reinstale o Python e recrie o ambiente virtual.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-linux-gpu.txt

if ! command -v git >/dev/null 2>&1; then
  echo "git nao encontrado. Instale git para baixar o checkout local do NISQA." >&2
  exit 1
fi

ensure_nisqa_checkout() {
  local nisqa_root="${PROJECT_ROOT}/NISQA"

  if [ ! -d "${nisqa_root}/.git" ] || [ ! -f "${nisqa_root}/nisqa/NISQA_model.py" ] || [ ! -f "${nisqa_root}/weights/nisqa_tts.tar" ]; then
    echo "NISQA ausente ou incompleto em ${nisqa_root}. Recriando checkout local..."
    rm -rf "${nisqa_root}"
    git clone https://github.com/gabrielmittag/NISQA.git "${nisqa_root}"
  else
    echo "NISQA valido ja presente em ${nisqa_root}"
  fi

  if [ ! -d "${nisqa_root}/.git" ] || [ ! -f "${nisqa_root}/nisqa/NISQA_model.py" ] || [ ! -f "${nisqa_root}/weights/nisqa_tts.tar" ]; then
    echo "Checkout do NISQA invalido em ${nisqa_root}. Verifique o clone e a presenca de weights/nisqa_tts.tar." >&2
    exit 1
  fi
}

ensure_nisqa_checkout

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

python3 -B tests/smoke_test.py extended

echo "Bootstrap Linux NVIDIA finalizado. Ambiente ok com checkout valido do NISQA em ${PROJECT_ROOT}/NISQA"
