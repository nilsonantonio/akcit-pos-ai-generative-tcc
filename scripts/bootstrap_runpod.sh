#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/install_runpod_gpu.sh"

python3 -B -c "from tests.smoke_test import *; test_prompt_file_has_expected_contract(); test_run_matrix_generation(); test_speaker_selection_from_curated_metadata(); test_metric_aggregation_contract(); test_wer_computation(); print('smoke tests passed')"

echo "Bootstrap RunPod finalizado."
