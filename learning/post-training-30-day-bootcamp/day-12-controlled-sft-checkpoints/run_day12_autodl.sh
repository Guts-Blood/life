#!/usr/bin/env bash
set -euo pipefail

DAY12_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAY12_BOOTCAMP_ROOT="$(cd "${DAY12_DIR}/.." && pwd)"
DAY12_PYTHON="${DAY12_PYTHON:-/root/miniconda3/bin/python}"
DAY12_MODEL_PATH="${DAY12_MODEL_PATH:-/root/autodl-tmp/day11-ready/models/Qwen3-0.6B-Base}"
DAY12_RUN_ROOT="${DAY12_RUN_ROOT:-/root/autodl-tmp/runs/day12-20260806}"
DAY12_EXPORT_ROOT="${DAY12_EXPORT_ROOT:-${DAY12_RUN_ROOT}/inference-exports}"
DAY12_TRAINER="${DAY12_DIR}/run_day12_training.py"
DAY12_PREPARER="${DAY12_DIR}/prepare_day12_experiment.py"
DAY12_REQUIREMENTS="${DAY12_DIR}/requirements-day12.txt"
CONFIG_A="${DAY12_BOOTCAMP_ROOT}/artifacts/configs/day12-controlled-sft-A.yaml"
CONFIG_B="${DAY12_BOOTCAMP_ROOT}/artifacts/configs/day12-controlled-sft-B.yaml"
CONFIG_C="${DAY12_BOOTCAMP_ROOT}/artifacts/configs/day12-controlled-sft-C.yaml"
CONFIG_D="${DAY12_BOOTCAMP_ROOT}/artifacts/configs/day12-controlled-sft-D.yaml"
SCHEDULE_A="${DAY12_BOOTCAMP_ROOT}/artifacts/data/day12-training-schedule-A.json"
SCHEDULE_B="${DAY12_BOOTCAMP_ROOT}/artifacts/data/day12-training-schedule-B.json"
SCHEDULE_C="${DAY12_BOOTCAMP_ROOT}/artifacts/data/day12-training-schedule-C.json"
SCHEDULE_D="${DAY12_BOOTCAMP_ROOT}/artifacts/data/day12-training-schedule-D.json"
RUN_A="${DAY12_RUN_ROOT}/run-A"
RUN_B="${DAY12_RUN_ROOT}/run-B"
RUN_C="${DAY12_RUN_ROOT}/run-C"
RUN_D="${DAY12_RUN_ROOT}/run-D"

bootstrap() {
  "${DAY12_PYTHON}" -m pip install --disable-pip-version-check -r "${DAY12_REQUIREMENTS}"
  "${DAY12_PYTHON}" -m pip check
}

prepare() {
  "${DAY12_PYTHON}" "${DAY12_PREPARER}"
}

prepare_c() {
  "${DAY12_PYTHON}" "${DAY12_DIR}/prepare_day12_mix_c.py"
}

prepare_d() {
  "${DAY12_PYTHON}" "${DAY12_DIR}/prepare_day12_run_d.py"
}

preflight() {
  mkdir -p "${DAY12_RUN_ROOT}" "${DAY12_EXPORT_ROOT}"
  "${DAY12_PYTHON}" "${DAY12_TRAINER}" preflight \
    --config "${CONFIG_A}" \
    --schedule "${SCHEDULE_A}" \
    --model-path "${DAY12_MODEL_PATH}" \
    --run-root "${DAY12_RUN_ROOT}" \
    --output "${DAY12_BOOTCAMP_ROOT}/artifacts/logs/day12-preflight-A.json"
  "${DAY12_PYTHON}" "${DAY12_TRAINER}" preflight \
    --config "${CONFIG_B}" \
    --schedule "${SCHEDULE_B}" \
    --model-path "${DAY12_MODEL_PATH}" \
    --run-root "${DAY12_RUN_ROOT}" \
    --output "${DAY12_BOOTCAMP_ROOT}/artifacts/logs/day12-preflight-B.json"
}

preflight_c() {
  mkdir -p "${DAY12_RUN_ROOT}" "${DAY12_EXPORT_ROOT}"
  "${DAY12_PYTHON}" "${DAY12_TRAINER}" preflight \
    --config "${CONFIG_C}" \
    --schedule "${SCHEDULE_C}" \
    --model-path "${DAY12_MODEL_PATH}" \
    --run-root "${DAY12_RUN_ROOT}" \
    --output "${DAY12_BOOTCAMP_ROOT}/artifacts/logs/day12-preflight-C.json"
}

preflight_d() {
  mkdir -p "${DAY12_RUN_ROOT}" "${DAY12_EXPORT_ROOT}"
  "${DAY12_PYTHON}" "${DAY12_TRAINER}" preflight \
    --config "${CONFIG_D}" \
    --schedule "${SCHEDULE_D}" \
    --model-path "${DAY12_MODEL_PATH}" \
    --run-root "${DAY12_RUN_ROOT}" \
    --output "${DAY12_BOOTCAMP_ROOT}/artifacts/logs/day12-preflight-D.json"
}

smoke() {
  "${DAY12_PYTHON}" "${DAY12_TRAINER}" train \
    --config "${CONFIG_A}" \
    --schedule "${SCHEDULE_A}" \
    --model-path "${DAY12_MODEL_PATH}" \
    --run-dir "${DAY12_RUN_ROOT}/smoke-A" \
    --export-root "${DAY12_EXPORT_ROOT}" \
    --target 25_percent \
    --smoke-steps 10
  "${DAY12_PYTHON}" "${DAY12_TRAINER}" train \
    --config "${CONFIG_B}" \
    --schedule "${SCHEDULE_B}" \
    --model-path "${DAY12_MODEL_PATH}" \
    --run-dir "${DAY12_RUN_ROOT}/smoke-B" \
    --export-root "${DAY12_EXPORT_ROOT}" \
    --target 25_percent \
    --smoke-steps 10
}

smoke_c() {
  "${DAY12_PYTHON}" "${DAY12_TRAINER}" train \
    --config "${CONFIG_C}" \
    --schedule "${SCHEDULE_C}" \
    --model-path "${DAY12_MODEL_PATH}" \
    --run-dir "${DAY12_RUN_ROOT}/smoke-C" \
    --export-root "${DAY12_EXPORT_ROOT}" \
    --target 25_percent \
    --smoke-steps 10
}

smoke_d() {
  "${DAY12_PYTHON}" "${DAY12_TRAINER}" train \
    --config "${CONFIG_D}" \
    --schedule "${SCHEDULE_D}" \
    --model-path "${DAY12_MODEL_PATH}" \
    --run-dir "${DAY12_RUN_ROOT}/smoke-D" \
    --export-root "${DAY12_EXPORT_ROOT}" \
    --target 25_percent \
    --smoke-steps 10
}

run_stage() {
  local run_id="$1"
  local config="$2"
  local schedule="$3"
  local target="$4"
  local resume_from="${5:-}"
  local run_dir="${DAY12_RUN_ROOT}/run-${run_id}"
  local command=(
    "${DAY12_PYTHON}" "${DAY12_TRAINER}" train
    --config "${config}"
    --schedule "${schedule}"
    --model-path "${DAY12_MODEL_PATH}"
    --run-dir "${run_dir}"
    --export-root "${DAY12_EXPORT_ROOT}"
    --target "${target}"
  )
  if [[ -n "${resume_from}" ]]; then
    command+=(--resume-from "${resume_from}")
  fi
  "${command[@]}"
}

run_a() {
  run_stage A "${CONFIG_A}" "${SCHEDULE_A}" 25_percent
  run_stage A "${CONFIG_A}" "${SCHEDULE_A}" 60_percent "${RUN_A}/checkpoints/checkpoint-25_percent"
  run_stage A "${CONFIG_A}" "${SCHEDULE_A}" 100_percent "${RUN_A}/checkpoints/checkpoint-60_percent"
}

run_b() {
  run_stage B "${CONFIG_B}" "${SCHEDULE_B}" 25_percent
  run_stage B "${CONFIG_B}" "${SCHEDULE_B}" 60_percent "${RUN_B}/checkpoints/checkpoint-25_percent"
  run_stage B "${CONFIG_B}" "${SCHEDULE_B}" 100_percent "${RUN_B}/checkpoints/checkpoint-60_percent"
}

run_c_25() {
  run_stage C "${CONFIG_C}" "${SCHEDULE_C}" 25_percent
}

run_c_continue() {
  run_stage C "${CONFIG_C}" "${SCHEDULE_C}" 60_percent "${RUN_C}/checkpoints/checkpoint-25_percent"
  run_stage C "${CONFIG_C}" "${SCHEDULE_C}" 100_percent "${RUN_C}/checkpoints/checkpoint-60_percent"
}

run_d_25() {
  run_stage D "${CONFIG_D}" "${SCHEDULE_D}" 25_percent
}

run_d_continue() {
  run_stage D "${CONFIG_D}" "${SCHEDULE_D}" 60_percent "${RUN_D}/checkpoints/checkpoint-25_percent"
  run_stage D "${CONFIG_D}" "${SCHEDULE_D}" 100_percent "${RUN_D}/checkpoints/checkpoint-60_percent"
}

case "${1:-}" in
  bootstrap)
    bootstrap
    ;;
  prepare)
    prepare
    ;;
  prepare-c)
    prepare_c
    ;;
  prepare-d)
    prepare_d
    ;;
  preflight)
    preflight
    ;;
  preflight-c)
    preflight_c
    ;;
  preflight-d)
    preflight_d
    ;;
  smoke)
    smoke
    ;;
  smoke-c)
    smoke_c
    ;;
  smoke-d)
    smoke_d
    ;;
  run-a)
    run_a
    ;;
  run-b)
    run_b
    ;;
  run-c-25)
    run_c_25
    ;;
  run-c-continue)
    run_c_continue
    ;;
  run-d-25)
    run_d_25
    ;;
  run-d-continue)
    run_d_continue
    ;;
  *)
    echo "usage: $0 {bootstrap|prepare|prepare-c|prepare-d|preflight|preflight-c|preflight-d|smoke|smoke-c|smoke-d|run-a|run-b|run-c-25|run-c-continue|run-d-25|run-d-continue}" >&2
    exit 2
    ;;
esac
