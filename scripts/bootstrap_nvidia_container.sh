#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

for required_cmd in python3 ffmpeg git; do
  if ! command -v "${required_cmd}" >/dev/null 2>&1; then
    echo "${required_cmd} nao encontrado no container CUDA." >&2
    exit 1
  fi
done

python3 - <<'PY'
import sys
import torch

expected_torch = "2.6.0"
if torch.__version__.split("+", 1)[0] != expected_torch:
    print(
        f"Versao inesperada do torch no container: {torch.__version__}. Esperado {expected_torch}.",
        file=sys.stderr,
    )
    raise SystemExit(1)

cuda_version = torch.version.cuda or ""
if not cuda_version.startswith("12.4"):
    print(
        f"Versao CUDA inesperada no torch do container: {cuda_version!r}. Esperado prefixo '12.4'.",
        file=sys.stderr,
    )
    raise SystemExit(1)

if not torch.cuda.is_available():
    print("CUDA nao esta disponivel dentro do container.", file=sys.stderr)
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

python3 scripts/validate_manifest.py --help >/dev/null
python3 scripts/run_speecht5_lora.py --help >/dev/null
python3 scripts/run_whisper_batch.py --help >/dev/null

echo "Bootstrap Linux NVIDIA container finalizado."
