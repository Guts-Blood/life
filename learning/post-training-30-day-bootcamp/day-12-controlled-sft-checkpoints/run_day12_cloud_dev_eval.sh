#!/usr/bin/env bash
set -euo pipefail

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

MODEL_NAMES=(
  "cloud-base"
  "A-25_percent"
  "A-60_percent"
  "A-100_percent"
  "B-25_percent"
  "B-60_percent"
  "B-100_percent"
)
MODEL_PATHS=(
  "/root/autodl-tmp/day11-ready/models/Qwen3-0.6B-Base"
  "${RUN_ROOT}/inference-exports/A-25_percent"
  "${RUN_ROOT}/inference-exports/A-60_percent"
  "${RUN_ROOT}/inference-exports/A-100_percent"
  "${RUN_ROOT}/inference-exports/B-25_percent"
  "${RUN_ROOT}/inference-exports/B-60_percent"
  "${RUN_ROOT}/inference-exports/B-100_percent"
)
MODEL_IDS=(
  "Qwen/Qwen3-0.6B-Base"
  "day12/A-25_percent"
  "day12/A-60_percent"
  "day12/A-100_percent"
  "day12/B-25_percent"
  "day12/B-60_percent"
  "day12/B-100_percent"
)
MODEL_REVISIONS=(
  "ddc928429ed09d9ad603fd762053d0434c15e865"
  "b03885ed204ad72bdbeedee4a8a082f65891d93d92da5ec7dfafa98f733a5170"
  "f1fd2036251e732046033780f96a7087c068fc6c4c9bdc35148b1ac1039c5a99"
  "914dea020f489b6b05ee0071b7a5aa29f2e4bbd0ea25b790d51a86a5cf06aa81"
  "5e3ce2915dbd59e4b343a47dca0f17b1000ccf40fca8c1fe844844b9cc8ed317"
  "45613b58e50a98bd7bcdfc18c9b675b9dba0f3bc0b4eded4b8f3e15bb4404dac"
  "20d474ec04a00f990f64949fb5438b98683baba8499be5e85ec960a714e3e475"
)

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

run_prediction() {
  local index="$1"
  local name="${MODEL_NAMES[${index}]}"
  local output="${EVAL_ROOT}/${name}-dev-predictions.jsonl"
  local log="${EVAL_ROOT}/logs/${name}-generation.log"
  if [[ -f "${output}" ]]; then
    require_line_count "${output}" 112
    echo "generation=resume-skip model=${name}"
    return
  fi
  "${GENERATION_PYTHON}" "${RUNNER}" \
    --manifest "${MANIFEST}" \
    --output "${output}" \
    --model-path "${MODEL_PATHS[${index}]}" \
    --model-id "${MODEL_IDS[${index}]}" \
    --model-revision "${MODEL_REVISIONS[${index}]}" \
    --split dev \
    --device cuda 2>&1 | tee "${log}"
  require_line_count "${output}" 112
}

run_code_score() {
  local name="$1"
  local predictions="${EVAL_ROOT}/${name}-dev-predictions.jsonl"
  local output="${EVAL_ROOT}/${name}-dev-code-e2b.jsonl"
  local log="${EVAL_ROOT}/logs/${name}-e2b.log"
  if [[ -f "${output}" ]]; then
    require_line_count "${output}" 28
    echo "e2b=resume-skip model=${name}"
    return
  fi
  (
    set -a
    source "${CREDENTIAL_FILE}"
    set +a
    "${SANDBOX_PYTHON}" "${SCORER}" \
      --manifest "${MANIFEST}" \
      --predictions "${predictions}" \
      --sandbox-config "${SANDBOX_CONFIG}" \
      --output "${output}"
  ) 2>&1 | tee "${log}"
  require_line_count "${output}" 28
}

run_summary() {
  local name="$1"
  local output="${EVAL_ROOT}/${name}-dev-summary.json"
  if [[ -f "${output}" ]]; then
    echo "summary=resume-skip model=${name}"
    return
  fi
  "${GENERATION_PYTHON}" "${ANALYZER}" \
    --manifest "${MANIFEST}" \
    --predictions "${EVAL_ROOT}/${name}-dev-predictions.jsonl" \
    --code-results "${EVAL_ROOT}/${name}-dev-code-e2b.jsonl" \
    --output "${output}"
}

(
  set -a
  source "${CREDENTIAL_FILE}"
  set +a
  "${SANDBOX_PYTHON}" "${PREFLIGHT}"
)

base_comparison_key=""
for index in "${!MODEL_NAMES[@]}"; do
  run_prediction "${index}"
  prediction_path="${EVAL_ROOT}/${MODEL_NAMES[${index}]}-dev-predictions.jsonl"
  comparison_key="$(first_json_field "${prediction_path}" comparison_key)"
  if [[ -z "${base_comparison_key}" ]]; then
    base_comparison_key="${comparison_key}"
  elif [[ "${comparison_key}" != "${base_comparison_key}" ]]; then
    echo "comparison key mismatch for ${MODEL_NAMES[${index}]}" >&2
    exit 1
  fi
done

base_complete_comparison_key=""
for name in "${MODEL_NAMES[@]}"; do
  run_code_score "${name}"
  code_path="${EVAL_ROOT}/${name}-dev-code-e2b.jsonl"
  complete_comparison_key="$(first_json_field "${code_path}" complete_comparison_key)"
  if [[ -z "${base_complete_comparison_key}" ]]; then
    base_complete_comparison_key="${complete_comparison_key}"
  elif [[ "${complete_comparison_key}" != "${base_complete_comparison_key}" ]]; then
    echo "complete comparison key mismatch for ${name}" >&2
    exit 1
  fi
  run_summary "${name}"
done

touch "${EVAL_ROOT}/DEV_EVAL_COMPLETE"
echo "status=day12_cloud_dev_eval_complete"
echo "comparison_key=${base_comparison_key}"
echo "complete_comparison_key=${base_complete_comparison_key}"
