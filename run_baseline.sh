#!/usr/bin/env bash
set -euo pipefail

# One-command entry point for the currently supported official-rule baselines.
#
# Examples:
#   bash run_baseline.sh Random
#   bash run_baseline.sh Range --limit-per-type 1
#   bash run_baseline.sh "Kalman filter" --workers 8
#
# The default runs the complete official test-prefix split and writes outputs
# under artifacts/official_offline_test. Add --resume after an interruption.

MODEL="${1:-}"
if [[ -z "${MODEL}" ]]; then
  echo "Usage: bash run_baseline.sh {Random|Range|Kalman filter} [options]" >&2
  exit 2
fi
shift

case "${MODEL}" in
  Random|Range|"Kalman filter") ;;
  *)
    echo "Unsupported model: ${MODEL}" >&2
    echo "Supported models: Random, Range, Kalman filter" >&2
    exit 2
    ;;
esac

DATA_ROOT="${RISKBENCH_DATA_ROOT:-/data/dongzk/RiskBench/RiskBench_Dataset}"
OUTPUT_ROOT="${RISKBENCH_OUTPUT_ROOT:-artifacts/official_offline_test}"
WORKERS="${RISKBENCH_WORKERS:-8}"

exec conda run -n riskbench python scripts/run_official_offline_test_split.py \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --method "${MODEL}" \
  --workers "${WORKERS}" \
  "$@"
