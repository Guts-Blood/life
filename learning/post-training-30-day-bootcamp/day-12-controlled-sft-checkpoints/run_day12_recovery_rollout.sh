#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-}"
ACTION="${2:-}"
if [[ ! "${RUN_ID}" =~ ^[C-L]$ ]]; then
  echo "usage: $0 <C-L> {preflight|smoke|run-25|continue}" >&2
  exit 2
fi

DAY12_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTCAMP_ROOT="$(cd "${DAY12_DIR}/.." && pwd)"
PYTHON_BIN="${DAY12_PYTHON:-/root/miniconda3/bin/python}"
MODEL_PATH="${DAY12_MODEL_PATH:-/root/autodl-tmp/day11-ready/models/Qwen3-0.6B-Base}"
RUN_ROOT="${DAY12_RUN_ROOT:-/root/autodl-tmp/runs/day12-20260806}"
EXPORT_ROOT="${DAY12_EXPORT_ROOT:-${RUN_ROOT}/inference-exports}"
TRAINER="${DAY12_DIR}/run_day12_training.py"
CONFIG="${BOOTCAMP_ROOT}/artifacts/configs/day12-controlled-sft-${RUN_ID}.yaml"
SCHEDULE="${BOOTCAMP_ROOT}/artifacts/data/day12-training-schedule-${RUN_ID}.json"
RUN_DIR="${RUN_ROOT}/run-${RUN_ID}"

train_stage() {
  local target="$1"
  local resume_from="${2:-}"
  local command=(
    "${PYTHON_BIN}" "${TRAINER}" train
    --config "${CONFIG}"
    --schedule "${SCHEDULE}"
    --model-path "${MODEL_PATH}"
    --run-dir "${RUN_DIR}"
    --export-root "${EXPORT_ROOT}"
    --target "${target}"
  )
  if [[ -n "${resume_from}" ]]; then
    command+=(--resume-from "${resume_from}")
  fi
  "${command[@]}"
}

case "${ACTION}" in
  preflight)
    "${PYTHON_BIN}" "${TRAINER}" preflight \
      --config "${CONFIG}" \
      --schedule "${SCHEDULE}" \
      --model-path "${MODEL_PATH}" \
      --run-root "${RUN_ROOT}" \
      --output "${BOOTCAMP_ROOT}/artifacts/logs/day12-preflight-${RUN_ID}.json"
    ;;
  smoke)
    "${PYTHON_BIN}" "${TRAINER}" train \
      --config "${CONFIG}" \
      --schedule "${SCHEDULE}" \
      --model-path "${MODEL_PATH}" \
      --run-dir "${RUN_ROOT}/smoke-${RUN_ID}" \
      --export-root "${EXPORT_ROOT}" \
      --target 25_percent \
      --smoke-steps 10
    ;;
  run-25)
    train_stage 25_percent
    ;;
  continue)
    train_stage 60_percent "${RUN_DIR}/checkpoints/checkpoint-25_percent"
    train_stage 100_percent "${RUN_DIR}/checkpoints/checkpoint-60_percent"
    ;;
  *)
    echo "usage: $0 <C-L> {preflight|smoke|run-25|continue}" >&2
    exit 2
    ;;
esac
