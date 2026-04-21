#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/install_runpod_gpu.sh"

python3 -B tests/smoke_test.py core

echo "Bootstrap RunPod finalizado."
