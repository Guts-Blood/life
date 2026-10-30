#!/usr/bin/env bash
set -euo pipefail

DAY11_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAY11_BOOTCAMP_ROOT="$(cd "${DAY11_DIR}/.." && pwd)"
DAY11_CONFIG="${DAY11_BOOTCAMP_ROOT}/artifacts/configs/day11-qwen3-0.6b-tiny-overfit.yaml"
DAY11_REQUIREMENTS="${DAY11_DIR}/requirements-day11.txt"
DAY11_TRAINER="${DAY11_DIR}/run_day11_training.py"
DAY11_PYTHON="${DAY11_PYTHON:-python}"
DAY11_MODEL_PATH="${DAY11_MODEL_PATH:-/root/autodl-tmp/day11-ready/models/Qwen3-0.6B-Base}"
DAY11_RUN_ROOT="${DAY11_RUN_ROOT:-/root/autodl-tmp/runs/day11-$(date -u +%Y%m%d-%H%M%S)}"

bootstrap() {
  "${DAY11_PYTHON}" -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda)'
  "${DAY11_PYTHON}" -m pip install --disable-pip-version-check -r "${DAY11_REQUIREMENTS}"
  "${DAY11_PYTHON}" -m pip check
}

preflight() {
  mkdir -p "${DAY11_RUN_ROOT}"
  "${DAY11_PYTHON}" "${DAY11_TRAINER}" preflight \
    --config "${DAY11_CONFIG}" \
    --model-path "${DAY11_MODEL_PATH}" \
    --run-root "${DAY11_RUN_ROOT}" \
    --output "${DAY11_BOOTCAMP_ROOT}/artifacts/logs/day11-autodl-preflight.json"
  nvidia-smi
  "${DAY11_PYTHON}" -m pip freeze > "${DAY11_BOOTCAMP_ROOT}/artifacts/logs/day11-autodl-pip-freeze.txt"
}

resume_probe() {
  local reference_run="${DAY11_RUN_ROOT}/resume-reference"
  local resumed_run="${DAY11_RUN_ROOT}/resume-interrupted"
  "${DAY11_PYTHON}" "${DAY11_TRAINER}" train \
    --config "${DAY11_CONFIG}" \
    --model-path "${DAY11_MODEL_PATH}" \
    --run-dir "${reference_run}" \
    --max-steps 6 \
    --stop-after-step 6 \
    --checksum-through-step 6
  "${DAY11_PYTHON}" "${DAY11_TRAINER}" train \
    --config "${DAY11_CONFIG}" \
    --model-path "${DAY11_MODEL_PATH}" \
    --run-dir "${resumed_run}" \
    --max-steps 6 \
    --stop-after-step 3 \
    --save-steps 3 \
    --checksum-through-step 6
  "${DAY11_PYTHON}" "${DAY11_TRAINER}" train \
    --config "${DAY11_CONFIG}" \
    --model-path "${DAY11_MODEL_PATH}" \
    --run-dir "${resumed_run}" \
    --max-steps 6 \
    --stop-after-step 6 \
    --resume-from "${resumed_run}/checkpoints/checkpoint-step-000003" \
    --checksum-through-step 6
  "${DAY11_PYTHON}" "${DAY11_TRAINER}" compare-resume \
    --reference-run "${reference_run}" \
    --resumed-run "${resumed_run}" \
    --interruption-step 3 \
    --total-steps 6 \
    --output-json "${DAY11_BOOTCAMP_ROOT}/artifacts/reports/day11-resume-audit.json" \
    --output-markdown "${DAY11_BOOTCAMP_ROOT}/artifacts/reports/day11-resume-audit.md"
}

main_overfit() {
  "${DAY11_PYTHON}" "${DAY11_TRAINER}" train \
    --config "${DAY11_CONFIG}" \
    --model-path "${DAY11_MODEL_PATH}" \
    --run-dir "${DAY11_RUN_ROOT}/main" \
    --save-steps 25 \
    --checksum-through-step 3 \
    --evaluate \
    --save-final-checkpoint \
    --artifact-report-dir "${DAY11_BOOTCAMP_ROOT}/artifacts/reports"
}

verify_result() {
  "${DAY11_PYTHON}" - "${DAY11_RUN_ROOT}" "${DAY11_BOOTCAMP_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
bootcamp_root = Path(sys.argv[2])
resume = json.loads((bootcamp_root / "artifacts/reports/day11-resume-audit.json").read_text())
summary = json.loads((run_root / "main/run-summary.json").read_text())
if resume["status"] != "pass":
    raise SystemExit("resume audit did not pass")
if not summary["threshold_reached"]:
    raise SystemExit("tiny-overfit accuracy threshold was not reached")
result = {
    "status": "day11_pass",
    "run_root": str(run_root),
    "resume_status": resume["status"],
    "final_step": summary["optimizer_step"],
    "final_token_accuracy": summary["final_teacher_forced"]["token_accuracy"],
    "final_checkpoint": summary["final_checkpoint"],
    "sync_back": [
        str(bootcamp_root / "artifacts/configs/day11-qwen3-0.6b-tiny-overfit.yaml"),
        str(bootcamp_root / "artifacts/data/day11-tiny-overfit.jsonl"),
        str(bootcamp_root / "artifacts/data/day11-tiny-overfit-manifest.json"),
        str(bootcamp_root / "artifacts/logs/day11-first-three-steps.jsonl"),
        str(bootcamp_root / "artifacts/logs/day11-run-summary.json"),
        str(bootcamp_root / "artifacts/reports/day11-resume-audit.json"),
        str(bootcamp_root / "artifacts/reports/day11-resume-audit.md"),
        str(bootcamp_root / "artifacts/reports/day11-sft-step-audit.md"),
        str(run_root / "main"),
        str(run_root / "resume-interrupted/checkpoints/checkpoint-step-000003"),
    ],
}
output = run_root / "DAY11-PASS.json"
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
PY
}

case "${1:-all}" in
  bootstrap)
    bootstrap
    ;;
  preflight)
    preflight
    ;;
  resume-probe)
    resume_probe
    ;;
  train)
    main_overfit
    ;;
  all)
    bootstrap
    preflight
    resume_probe
    main_overfit
    verify_result
    ;;
  *)
    echo "usage: $0 {bootstrap|preflight|resume-probe|train|all}" >&2
    exit 2
    ;;
esac
