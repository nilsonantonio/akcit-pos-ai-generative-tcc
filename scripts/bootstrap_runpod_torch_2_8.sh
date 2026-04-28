#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

MIN_DRIVER_VERSION="570.133.07"
RECOMMENDED_DRIVER_VERSION="575.51.02"

if [ "$(uname -s)" != "Linux" ]; then
  echo "Este bootstrap exige Linux com GPU NVIDIA no host do pod." >&2
  exit 1
fi

for required_cmd in python3 git ffmpeg nvidia-smi; do
  if ! command -v "${required_cmd}" >/dev/null 2>&1; then
    echo "${required_cmd} nao encontrado. Ajuste o template base do RunPod." >&2
    exit 1
  fi
done

driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | tr -d '[:space:]')"
if [ -z "${driver_version}" ]; then
  echo "Nao foi possivel determinar a versao do driver NVIDIA via nvidia-smi." >&2
  exit 1
fi

python3 - "${driver_version}" "${MIN_DRIVER_VERSION}" "${RECOMMENDED_DRIVER_VERSION}" <<'PY'
import sys

current, minimum, recommended = sys.argv[1:4]

def parse(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))

current_v = parse(current)
minimum_v = parse(minimum)
recommended_v = parse(recommended)

if current_v < minimum_v:
    print(
        f"Driver NVIDIA {current} abaixo do minimo {minimum} para compatibilidade CUDA 12.x.",
        file=sys.stderr,
    )
    raise SystemExit(1)

if current_v < recommended_v:
    print(
        f"Aviso: driver NVIDIA {current} abaixo do recomendado {recommended} para alinhamento com CUDA 12.4 GA.",
        file=sys.stderr,
    )
PY

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-linux-gpu.txt
python -m pip install -r requirements-linux-gpu-cu128.txt

python - <<'PY'
import sys
import torch

expected_torch_prefix = "2.8"
if not torch.__version__.split("+", 1)[0].startswith(expected_torch_prefix):
    print(
        f"Versao inesperada do torch no host RunPod: {torch.__version__}. Esperado {expected_torch}.",
        file=sys.stderr,
    )
    raise SystemExit(1)

cuda_version = torch.version.cuda or ""
if not cuda_version.startswith("12.8"):
    print(
        f"Versao CUDA inesperada no torch do host RunPod: {cuda_version!r}. Esperado prefixo '12.4'.",
        file=sys.stderr,
    )
    raise SystemExit(1)

if not torch.cuda.is_available():
    print("CUDA nao esta disponivel no host RunPod.", file=sys.stderr)
    raise SystemExit(1)

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA disponivel:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
PY

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

python -B tests/smoke_test.py core

echo "Bootstrap RunPod host-native finalizado. Ative a .venv com: source .venv/bin/activate"
