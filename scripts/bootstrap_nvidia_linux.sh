#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Aviso: scripts/bootstrap_nvidia_linux.sh e legado. Use scripts/bootstrap_nvidia_host.sh." >&2
bash "${SCRIPT_DIR}/bootstrap_nvidia_host.sh"
