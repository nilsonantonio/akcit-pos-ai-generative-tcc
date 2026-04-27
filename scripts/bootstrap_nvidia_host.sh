#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CUDA_IMAGE="nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04"
MIN_DRIVER_VERSION="525.60.13"
RECOMMENDED_DRIVER_VERSION="550.54.14"

if [ "$(uname -s)" != "Linux" ]; then
  echo "Este bootstrap exige Linux nativo com GPU NVIDIA." >&2
  exit 1
fi

for required_cmd in docker nvidia-smi nvidia-ctk; do
  if ! command -v "${required_cmd}" >/dev/null 2>&1; then
    echo "${required_cmd} nao encontrado. Instale Docker, driver NVIDIA e NVIDIA Container Toolkit no host." >&2
    exit 1
  fi
done

if ! docker compose version >/dev/null 2>&1; then
  echo "docker compose nao encontrado. Instale o plugin oficial do Docker Compose." >&2
  exit 1
fi

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

docker run --rm --gpus all "${CUDA_IMAGE}" nvidia-smi >/dev/null

echo "Bootstrap Linux NVIDIA host finalizado. Host pronto para Docker CUDA 12.4."
