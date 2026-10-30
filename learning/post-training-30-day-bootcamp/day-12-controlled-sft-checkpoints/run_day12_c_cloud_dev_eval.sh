#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-25_percent}"
RECOVERY_RUN_ID="${DAY12_RECOVERY_RUN_ID:-C}"
case "${TARGET}" in
  25_percent|60_percent|100_percent)
    ;;
  *)
    echo "usage: $0 [25_percent|60_percent|100_percent]" >&2
    exit 2
    ;;
esac
if [[ ! "${RECOVERY_RUN_ID}" =~ ^[C-L]$ ]]; then
  echo "DAY12_RECOVERY_RUN_ID must be one uppercase rollout ID from C through L" >&2
  exit 2
fi

BOOTCAMP_ROOT="/root/autodl-tmp/day11-ready/post-training-30-day-bootcamp"
RUN_ROOT="${DAY12_RUN_ROOT:-/root/autodl-tmp/runs/day12-20260806}"
EVAL_ROOT="${DAY12_EVAL_ROOT:-${RUN_ROOT}/eval}"
GENERATION_PYTHON="/root/miniconda3/bin/python"
SANDBOX_PYTHON="/root/autodl-tmp/envs/day12-e2b/bin/python"
MANIFEST="${BOOTCAMP_ROOT}/artifacts/eval/day10-frozen-eval-manifest.json"
RUNNER="${BOOTCAMP_ROOT}/day-10-frozen-eval-baseline/run_day10_baseline.py"
SCORER="${BOOTCAMP_ROOT}/day-10-frozen-eval-baseline/score_day10_code_e2b.py"
ANALYZER="${BOOTCAMP_ROOT}/day-10-frozen-eval-baseline/analyze_day10_baseline.py"
SANDBOX_CONFIG="${BOOTCAMP_ROOT}/day-10-frozen-eval-baseline/day10_e2b_sandbox_config.json"
PREFLIGHT="${BOOTCAMP_ROOT}/day-12-controlled-sft-checkpoints/preflight_day12_e2b.py"
CREDENTIAL_FILE="/root/autodl-tmp/secrets/day12-e2b.env"

MODEL_NAME="${RECOVERY_RUN_ID}-${TARGET}"
MODEL_PATH="${RUN_ROOT}/inference-exports/${MODEL_NAME}"
MODEL_ID="day12/${MODEL_NAME}"
EXPORT_MANIFEST="${MODEL_PATH}/export_manifest.json"
PREDICTIONS="${EVAL_ROOT}/${MODEL_NAME}-dev-predictions.jsonl"
CODE_RESULTS="${EVAL_ROOT}/${MODEL_NAME}-dev-code-e2b.jsonl"
SUMMARY="${EVAL_ROOT}/${MODEL_NAME}-dev-summary.json"
BASE_PREDICTIONS="${EVAL_ROOT}/cloud-base-dev-predictions.jsonl"
BASE_CODE_RESULTS="${EVAL_ROOT}/cloud-base-dev-code-e2b.jsonl"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "${EVAL_ROOT}/logs"

require_line_count() {
  local path="$1"
  local expected="$2"
  local actual
  if [[ ! -f "${path}" ]]; then
    echo "missing artifact: ${path}" >&2
    return 1
  fi
  actual="$(wc -l < "${path}" | tr -d ' ')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "unexpected row count for ${path}: expected ${expected}, got ${actual}" >&2
    return 1
  fi
}

first_json_field() {
  "${GENERATION_PYTHON}" -c \
    'import json, sys; print(json.loads(open(sys.argv[1], encoding="utf-8").readline())[sys.argv[2]])' \
    "$1" "$2"
}

if [[ ! -f "${EXPORT_MANIFEST}" ]]; then
  echo "missing trained export: ${EXPORT_MANIFEST}" >&2
  exit 1
fi

MODEL_REVISION="$("${GENERATION_PYTHON}" -c \
  'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' \
  "${EXPORT_MANIFEST}")"

(
  set -a
  source "${CREDENTIAL_FILE}"
  set +a
  "${SANDBOX_PYTHON}" "${PREFLIGHT}"
)

if [[ -f "${PREDICTIONS}" ]]; then
  require_line_count "${PREDICTIONS}" 112
  echo "generation=resume-skip model=${MODEL_NAME}"
else
  "${GENERATION_PYTHON}" "${RUNNER}" \
    --manifest "${MANIFEST}" \
    --output "${PREDICTIONS}" \
    --model-path "${MODEL_PATH}" \
    --model-id "${MODEL_ID}" \
    --model-revision "${MODEL_REVISION}" \
    --split dev \
    --device cuda 2>&1 | tee "${EVAL_ROOT}/logs/${MODEL_NAME}-generation.log"
  require_line_count "${PREDICTIONS}" 112
fi

require_line_count "${BASE_PREDICTIONS}" 112
if [[ "$(first_json_field "${PREDICTIONS}" comparison_key)" != \
      "$(first_json_field "${BASE_PREDICTIONS}" comparison_key)" ]]; then
  echo "comparison key mismatch between ${MODEL_NAME} and cloud-base" >&2
  exit 1
fi

if [[ -f "${CODE_RESULTS}" ]]; then
  require_line_count "${CODE_RESULTS}" 28
  echo "e2b=resume-skip model=${MODEL_NAME}"
else
  (
    set -a
    source "${CREDENTIAL_FILE}"
    set +a
    "${SANDBOX_PYTHON}" "${SCORER}" \
      --manifest "${MANIFEST}" \
      --predictions "${PREDICTIONS}" \
      --sandbox-config "${SANDBOX_CONFIG}" \
      --output "${CODE_RESULTS}"
  ) 2>&1 | tee "${EVAL_ROOT}/logs/${MODEL_NAME}-e2b.log"
  require_line_count "${CODE_RESULTS}" 28
fi

require_line_count "${BASE_CODE_RESULTS}" 28
if [[ "$(first_json_field "${CODE_RESULTS}" complete_comparison_key)" != \
      "$(first_json_field "${BASE_CODE_RESULTS}" complete_comparison_key)" ]]; then
  echo "complete comparison key mismatch between ${MODEL_NAME} and cloud-base" >&2
  exit 1
fi

if [[ -f "${SUMMARY}" ]]; then
  echo "summary=resume-skip model=${MODEL_NAME}"
else
  "${GENERATION_PYTHON}" "${ANALYZER}" \
    --manifest "${MANIFEST}" \
    --predictions "${PREDICTIONS}" \
    --code-results "${CODE_RESULTS}" \
    --output "${SUMMARY}"
fi

touch "${EVAL_ROOT}/${MODEL_NAME}-DEV_EVAL_COMPLETE"
echo "status=day12_recovery_cloud_dev_eval_complete"
echo "model=${MODEL_NAME}"
echo "summary=${SUMMARY}"
