#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-}"
if [[ -z "${MODEL}" ]]; then
  echo "Usage: bash evaluate_baseline.sh {Random|Range|Kalman filter}" >&2
  exit 2
fi

case "${MODEL}" in
  Random|Range|"Kalman filter") ;;
  *)
    echo "Unsupported model: ${MODEL}" >&2
    exit 2
    ;;
esac

DATA_ROOT="${RISKBENCH_DATA_ROOT:-/data/dongzk/RiskBench/RiskBench_Dataset}"
MODEL_ROOT="${RISKBENCH_OUTPUT_ROOT:-artifacts/official_offline_test}"
RESULT_ROOT="${RISKBENCH_METRICS_ROOT:-artifacts/official_offline_metrics}"

exec conda run -n riskbench python \
  codes/RiskBench/risk_identification/Risk_identification_tool/ROI_tool.py \
  --method "${MODEL}" \
  --data_type all \
  --metadata_root "${DATA_ROOT}/metadata" \
  --model_root "${MODEL_ROOT}" \
  --result_path "${RESULT_ROOT}" \
  --save_result
