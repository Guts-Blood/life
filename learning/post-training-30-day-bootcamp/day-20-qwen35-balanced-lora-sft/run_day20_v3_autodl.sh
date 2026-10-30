#!/usr/bin/env bash
set -euo pipefail

BOOTCAMP_ROOT="${DAY20_V3_BOOTCAMP_ROOT:-/root/autodl-tmp/post-training-30-day-bootcamp}"
HERE="${BOOTCAMP_ROOT}/day-20-qwen35-balanced-lora-sft"
DATA_ROOT="${DAY20_V3_DATA_ROOT:-/root/autodl-tmp/qwen35-v3}"
VENV="${DAY20_V3_VENV:-/root/autodl-tmp/qwen35-v2/venv}"
PYTHON_BIN="${DAY20_V3_PYTHON:-${VENV}/bin/python}"
SANDBOX_PYTHON="${DAY20_V3_SANDBOX_PYTHON:-/root/autodl-tmp/envs/day12-e2b/bin/python}"
MS_SWIFT_ROOT="${DAY20_V3_MS_SWIFT_ROOT:-/root/autodl-tmp/ms-swift}"
MS_SWIFT_COMMIT="${DAY20_V3_MS_SWIFT_COMMIT:-565a1ad586a21d24b23931c52d2c62b49c39bee8}"
BASE_MODEL="${DAY20_V3_BASE_MODEL:-/root/autodl-tmp/Qwen--Qwen3.5-4B-Base/1001bb4d826a52d1f399e183466143f4da7b741b}"
STATE_FILE="${DAY20_V3_STATE_FILE:-${DATA_ROOT}/state/current-day20-v3-run-root}"
PARENT_RUN_ROOT="${DAY20_V3_PARENT_RUN_ROOT:-}"

EVAL_MANIFEST="${DAY20_V3_EVAL_MANIFEST:-${BOOTCAMP_ROOT}/artifacts/eval/day10-frozen-eval-manifest.json}"
SCORERS="${DAY20_V3_SCORERS:-${BOOTCAMP_ROOT}/day-10-frozen-eval-baseline/day10_scorers.py}"
FROZEN_E2B_SCORER="${DAY20_V3_FROZEN_E2B_SCORER:-${BOOTCAMP_ROOT}/day-10-frozen-eval-baseline/score_day10_code_e2b.py}"
SANDBOX_CONFIG="${DAY20_V3_SANDBOX_CONFIG:-${BOOTCAMP_ROOT}/day-10-frozen-eval-baseline/day10_e2b_sandbox_config.json}"
HUMANEVAL_SOURCE="${DAY20_V3_HUMANEVAL_SOURCE:-/root/autodl-tmp/runs/day19-qwen35-20260808T100142Z/configs/frozen-inputs/HumanEval.jsonl.gz}"
E2B_CREDENTIAL_FILE="${DAY20_V3_E2B_CREDENTIAL_FILE:-/root/autodl-tmp/secrets/day20-v2-e2b.env}"
E2B_CREDENTIAL_ATTESTATION="${DAY20_V3_E2B_CREDENTIAL_ATTESTATION:-/root/autodl-tmp/secrets/day20-v2-e2b-attestation.json}"

PREPARER="${HERE}/prepare_day20_v3.py"
TARGET_ENCODING_MODULE="${HERE}/day20_target_encoding_v3.py"
TRAIN_RUNTIME="${HERE}/day20_train_runtime_v3.py"
TRAIN_PLUGIN="${HERE}/day20_train_plugin_v3.py"
CORE_TRAIN_RUNTIME="${HERE}/day20_train_runtime_v2.py"
CORE_TRAIN_PLUGIN="${HERE}/day20_train_plugin_v2.py"
E2B_PREFLIGHT="${HERE}/day20_e2b_preflight_v2.py"
EVALUATOR="${HERE}/evaluate_day20_v2.py"
RESCORER="${HERE}/rescore_day20_qwen35_v3.py"
CODE_SCORER="${HERE}/score_day20_code_e2b_v3.py"
PROBE_SELECTOR="${HERE}/select_day20_v2.py"
MAIN_SELECTOR="${HERE}/select_day20_main_v2.py"
EXPECTED_GPU_PATTERN="${DAY20_V3_EXPECTED_GPU_PATTERN:-RTX PRO 6000 Blackwell Server Edition}"
MIN_FREE_GIB="${DAY20_V3_MIN_FREE_GIB:-80}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export USE_HF=1
export CUDA_HOME="${VENV}"
export CUDA_PATH="${CUDA_HOME}"
export NVTE_CUDA_INCLUDE_DIR="${CUDA_HOME}/targets/x86_64-linux/include"
export PATH="${VENV}/bin:${PATH}"
if [[ -d "${VENV}/lib/python3.12/site-packages/nvidia/cudnn/lib" ]]; then
  export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/cudnn/lib:${CUDA_HOME}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

require_file() {
  [[ -f "$1" ]] || { echo "required file is missing: $1" >&2; exit 1; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "required directory is missing: $1" >&2; exit 1; }
}

require_absent() {
  [[ ! -e "$1" ]] || { echo "refusing to overwrite artifact: $1" >&2; exit 1; }
}

acquire_lock() {
  command -v flock >/dev/null || { echo "required command is missing: flock" >&2; exit 1; }
  mkdir -p "$(dirname "${STATE_FILE}")"
  exec 9>"${STATE_FILE}.lock"
  flock -n 9 || { echo "another Day 20 v3 operation is active" >&2; exit 1; }
}

load_planned_run_root() {
  require_file "${STATE_FILE}"
  RUN_ROOT="$(sed -n '1p' "${STATE_FILE}")"
  [[ "${RUN_ROOT}" == /root/autodl-tmp/runs/day20-v3-qwen35-lora-* ]] || {
    echo "unsafe Day 20 v3 run root in state file" >&2
    exit 1
  }
}

load_run_root() {
  load_planned_run_root
  require_dir "${RUN_ROOT}"
  require_file "${RUN_ROOT}/.day20-v3-run-root"
  [[ "$(sed -n '1p' "${RUN_ROOT}/.day20-v3-run-root")" == "day20-qwen35-target-encoding-v3" ]] || {
    echo "Day 20 v3 run marker drifted" >&2
    exit 1
  }
}

init_run() {
  acquire_lock
  require_file "${PYTHON_BIN}"
  require_dir "${BASE_MODEL}"
  local run_id run_root temporary_state
  require_absent "${STATE_FILE}"
  run_id="$(date -u +%Y%m%dT%H%M%SZ)"
  run_root="/root/autodl-tmp/runs/day20-v3-qwen35-lora-${run_id}"
  require_absent "${run_root}"
  temporary_state="${STATE_FILE}.new.$$"
  require_absent "${temporary_state}"
  printf '%s\n' "${run_root}" > "${temporary_state}"
  mv -n -- "${temporary_state}" "${STATE_FILE}"
  printf '%s\n' "${run_root}"
}

prepare_run() {
  acquire_lock
  load_planned_run_root
  require_file "${PYTHON_BIN}"
  require_file "${PREPARER}"
  require_file "${TARGET_ENCODING_MODULE}"
  [[ -n "${PARENT_RUN_ROOT}" ]] || {
    echo "DAY20_V3_PARENT_RUN_ROOT is required for preparation" >&2
    exit 1
  }
  require_dir "${PARENT_RUN_ROOT}"
  require_file "${PARENT_RUN_ROOT}/DAY20-V2-MANIFEST.json"
  if [[ -f "${RUN_ROOT}/DAY20-V3-MANIFEST.json" ]]; then
    verify_prepared_run_unlocked
    mkdir -p "${RUN_ROOT}"/{adapters,evidence,eval,exports,logs}
    echo "verified existing Day 20 v3 preparation"
    return
  fi
  require_absent "${RUN_ROOT}"
  local prepare_log
  mkdir -p "${DATA_ROOT}/logs"
  prepare_log="${DATA_ROOT}/logs/prepare-$(basename "${RUN_ROOT}").log"
  require_absent "${prepare_log}"
  "${PYTHON_BIN}" "${PREPARER}" \
    --parent-run-root "${PARENT_RUN_ROOT}" \
    --run-root "${RUN_ROOT}" \
    --target-encoding-module "${TARGET_ENCODING_MODULE}" \
    --expected-ms-swift-commit "${MS_SWIFT_COMMIT}" \
    2>&1 | tee "${prepare_log}"
  mkdir -p "${RUN_ROOT}"/{adapters,evidence,eval,exports,logs}
  require_absent "${RUN_ROOT}/logs/prepare.log"
  mv -- "${prepare_log}" "${RUN_ROOT}/logs/prepare.log"
  verify_prepared_run_unlocked
}

run_e2b_preflight_unlocked() {
  require_file "${SANDBOX_PYTHON}"
  require_file "${E2B_PREFLIGHT}"
  require_file "${FROZEN_E2B_SCORER}"
  require_file "${SANDBOX_CONFIG}"
  require_file "${HUMANEVAL_SOURCE}"
  require_file "${E2B_CREDENTIAL_FILE}"
  require_file "${E2B_CREDENTIAL_ATTESTATION}"
  local marker="${RUN_ROOT}/evidence/E2B-PREFLIGHT.json"
  if [[ -f "${marker}" ]]; then
    "${PYTHON_BIN}" "${E2B_PREFLIGHT}" verify \
      --run-root "${RUN_ROOT}" --marker "${marker}" >/dev/null
    echo "verified existing E2B preflight"
    return
  fi
  "${SANDBOX_PYTHON}" "${E2B_PREFLIGHT}" run \
    --run-root "${RUN_ROOT}" \
    --frozen-scorer "${FROZEN_E2B_SCORER}" \
    --sandbox-config "${SANDBOX_CONFIG}" \
    --humaneval-source "${HUMANEVAL_SOURCE}" \
    --credential-file "${E2B_CREDENTIAL_FILE}" \
    --credential-attestation "${E2B_CREDENTIAL_ATTESTATION}" \
    --marker "${marker}" \
    2>&1 | tee -a "${RUN_ROOT}/logs/e2b-preflight.log"
  "${PYTHON_BIN}" "${E2B_PREFLIGHT}" verify \
    --run-root "${RUN_ROOT}" --marker "${marker}" >/dev/null
}

e2b_preflight() {
  acquire_lock
  load_run_root
  run_e2b_preflight_unlocked
}

system_preflight_unlocked() {
  require_file "${PYTHON_BIN}"
  require_dir "${BASE_MODEL}"
  require_dir "${MS_SWIFT_ROOT}"
  require_file "${RUN_ROOT}/DAY20-V3-MANIFEST.json"
  require_file "${RUN_ROOT}/DAY20-V2-MANIFEST.json"
  require_file "${RUN_ROOT}/TARGET-ENCODING-AUDIT.json"
  require_file "${TARGET_ENCODING_MODULE}"
  require_file "${TRAIN_RUNTIME}"
  require_file "${TRAIN_PLUGIN}"
  require_file "${CORE_TRAIN_RUNTIME}"
  require_file "${CORE_TRAIN_PLUGIN}"
  require_file "${SCORERS}"
  require_file "${EVALUATOR}"
  require_file "${RESCORER}"
  require_file "${CODE_SCORER}"
  require_file "${PROBE_SELECTOR}"
  require_file "${MAIN_SELECTOR}"
  command -v git >/dev/null || { echo "required command is missing: git" >&2; exit 1; }
  command -v nvidia-smi >/dev/null || { echo "required command is missing: nvidia-smi" >&2; exit 1; }
  [[ "$(id -u)" == 0 ]] || { echo "Day 20 v3 AutoDL run must execute as root" >&2; exit 1; }
  [[ "$(git -C "${MS_SWIFT_ROOT}" rev-parse HEAD)" == "${MS_SWIFT_COMMIT}" ]] || {
    echo "pinned ms-swift commit drifted" >&2
    exit 1
  }
  [[ -z "$(git -C "${MS_SWIFT_ROOT}" status --porcelain)" ]] || {
    echo "ms-swift worktree is dirty" >&2
    exit 1
  }
  local gpu_name free_kib
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | sed -n '1p')"
  [[ "${gpu_name}" == *"${EXPECTED_GPU_PATTERN}"* ]] || {
    echo "GPU 0 does not match ${EXPECTED_GPU_PATTERN}: ${gpu_name}" >&2
    exit 1
  }
  [[ "${MIN_FREE_GIB}" =~ ^[0-9]+$ ]] || { echo "MIN_FREE_GIB must be an integer" >&2; exit 1; }
  free_kib="$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $4}')"
  (( free_kib >= MIN_FREE_GIB * 1024 * 1024 )) || {
    echo "less than ${MIN_FREE_GIB} GiB is free under /root/autodl-tmp" >&2
    exit 1
  }
  PYTHONPATH="${HERE}${PYTHONPATH:+:${PYTHONPATH}}" CUDA_VISIBLE_DEVICES=0 \
    "${PYTHON_BIN}" - "${MS_SWIFT_COMMIT}" "${EXPECTED_GPU_PATTERN}" <<'PY'
import json, sys
import torch
from day20_train_plugin_v2 import runtime_identity

commit, gpu_pattern = sys.argv[1:]
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise SystemExit("CUDA/BF16 preflight failed")
name = torch.cuda.get_device_name(0)
if gpu_pattern not in name:
    raise SystemExit(f"torch GPU identity differs: {name}")
print(json.dumps({"gpu": name, "runtime": runtime_identity(commit)}, sort_keys=True))
PY
}

preflight() {
  acquire_lock
  load_run_root
  # The cheap fail-closed gates run before any GPU-bearing Swift process.
  verify_prepared_run_unlocked
  run_e2b_preflight_unlocked
  system_preflight_unlocked
  echo "Day 20 v3 core, target audit, E2B, runtime, and GPU preflight passed"
}

require_training_gate() {
  verify_prepared_run_unlocked
  require_file "${RUN_ROOT}/DAY20-V2-MANIFEST.json"
  require_file "${RUN_ROOT}/DAY20-V3-MANIFEST.json"
  require_file "${RUN_ROOT}/TARGET-ENCODING-AUDIT.json"
  require_file "${RUN_ROOT}/evidence/E2B-PREFLIGHT.json"
  "${PYTHON_BIN}" "${E2B_PREFLIGHT}" verify \
    --run-root "${RUN_ROOT}" \
    --marker "${RUN_ROOT}/evidence/E2B-PREFLIGHT.json" >/dev/null
  system_preflight_unlocked
}

json_field() {
  "${PYTHON_BIN}" -c 'import json,sys; value=json.loads(sys.stdin.read()); print(value[sys.argv[1]])' "$1"
}

manifest_config_path() {
  local kind="$1" lr="$2"
  "${PYTHON_BIN}" - "${RUN_ROOT}/DAY20-V2-MANIFEST.json" "${kind}" "${lr}" <<'PY'
import json, sys
path, kind, lr = sys.argv[1:]
value = json.load(open(path, encoding="utf-8"))
identity = value["configs"]["probes"][lr] if kind == "probe" else value["configs"]["main_template"]
print(identity["path"])
PY
}

verify_prepared_run_unlocked() {
  require_file "${RUN_ROOT}/.day20-v3-run-root"
  [[ "$(sed -n '1p' "${RUN_ROOT}/.day20-v3-run-root")" == "day20-qwen35-target-encoding-v3" ]] || {
    echo "Day 20 v3 prepared-run marker drifted" >&2
    exit 1
  }
  require_file "${RUN_ROOT}/DAY20-V3-MANIFEST.json"
  require_file "${RUN_ROOT}/DAY20-V2-MANIFEST.json"
  require_file "${RUN_ROOT}/TARGET-ENCODING-AUDIT.json"
  require_file "${TARGET_ENCODING_MODULE}"
  require_file "${RUN_ROOT}/data/probe-v2.jsonl"
  require_file "${RUN_ROOT}/data/main-v2.jsonl"
  require_file "${RUN_ROOT}/data/diagnostic-v2.jsonl"
  local lr
  for lr in 1e-4 3e-5 6e-5 8e-5; do
    "${PYTHON_BIN}" "${TRAIN_RUNTIME}" validate-prepared \
      --outer-manifest "${RUN_ROOT}/DAY20-V3-MANIFEST.json" \
      --core-manifest "${RUN_ROOT}/DAY20-V2-MANIFEST.json" \
      --target-audit "${RUN_ROOT}/TARGET-ENCODING-AUDIT.json" \
      --target-module "${TARGET_ENCODING_MODULE}" \
      --dataset "${RUN_ROOT}/data/probe-v2.jsonl" \
      --training-config "$(manifest_config_path probe "${lr}")" \
      --run-kind probe --seed 20260809 --learning-rate "${lr}" >/dev/null
  done
  "${PYTHON_BIN}" "${TRAIN_RUNTIME}" validate-prepared \
    --outer-manifest "${RUN_ROOT}/DAY20-V3-MANIFEST.json" \
    --core-manifest "${RUN_ROOT}/DAY20-V2-MANIFEST.json" \
    --target-audit "${RUN_ROOT}/TARGET-ENCODING-AUDIT.json" \
    --target-module "${TARGET_ENCODING_MODULE}" \
    --dataset "${RUN_ROOT}/data/main-v2.jsonl" \
    --training-config "$(manifest_config_path main 1e-4)" \
    --run-kind main --seed 20260809 --learning-rate 1e-4 >/dev/null
}

resolve_main_config() {
  local learning_rate="$1" seed="$2"
  local template output
  template="$(manifest_config_path main "${learning_rate}")"
  output="${RUN_ROOT}/configs/main-s${seed}-lr${learning_rate}.json"
  if [[ -f "${output}" ]]; then
    printf '%s\n' "${output}"
    return
  fi
  "${PYTHON_BIN}" "${CORE_TRAIN_PLUGIN}" resolve-main-config \
    --manifest "${RUN_ROOT}/DAY20-V2-MANIFEST.json" \
    --template "${template}" \
    --learning-rate "${learning_rate}" \
    --seed "${seed}" \
    --output "${output}" >/dev/null
  printf '%s\n' "${output}"
}

next_attempt_name() {
  local root="$1" maximum=0 path number
  if [[ -d "${root}" ]]; then
    while IFS= read -r path; do
      number="${path##*-}"
      if [[ "${number}" =~ ^[0-9]{3}$ ]] && (( 10#${number} > maximum )); then
        maximum=$((10#${number}))
      fi
    done < <(find "${root}" -mindepth 1 -maxdepth 1 -type d -name 'attempt-[0-9][0-9][0-9]' -print)
  fi
  (( maximum < 999 )) || { echo "training attempt namespace exhausted" >&2; exit 1; }
  printf 'attempt-%03d\n' $((maximum + 1))
}

verify_sealed_attempt() {
  local summary="$1" evidence_dir
  evidence_dir="$(dirname "${summary}")"
  require_file "${summary}"
  require_file "${evidence_dir}/target-binding-config.json"
  require_file "${evidence_dir}/target-binding-attestation.json"
  require_dir "${evidence_dir}/checkpoint-envelopes"
  "${PYTHON_BIN}" "${TRAIN_PLUGIN}" verify-sealed-summary \
    --summary "${summary}" --run-root "${RUN_ROOT}" \
    --binding-config "${evidence_dir}/target-binding-config.json" \
    --attestation "${evidence_dir}/target-binding-attestation.json" \
    --envelope-dir "${evidence_dir}/checkpoint-envelopes" >/dev/null
}

run_training() {
  local kind="$1" learning_rate="$2" seed="$3"
  acquire_lock
  load_run_root
  require_training_gate
  require_file "${TRAIN_RUNTIME}"
  require_file "${TRAIN_PLUGIN}"
  require_dir "${MS_SWIFT_ROOT}"
  local outer_manifest core_manifest target_audit dataset config validation runtime_sha256 path_root evidence_root canonical_summary final_label final_candidate
  outer_manifest="${RUN_ROOT}/DAY20-V3-MANIFEST.json"
  core_manifest="${RUN_ROOT}/DAY20-V2-MANIFEST.json"
  target_audit="${RUN_ROOT}/TARGET-ENCODING-AUDIT.json"
  if [[ "${kind}" == "probe" ]]; then
    dataset="${RUN_ROOT}/data/probe-v2.jsonl"
    config="$(manifest_config_path probe "${learning_rate}")"
    path_root="${RUN_ROOT}/adapters/probe/s${seed}/lr${learning_rate}"
    evidence_root="${RUN_ROOT}/evidence/probe/s${seed}/lr${learning_rate}"
    final_label="t24000"
    final_candidate="probe-s${seed}-lr${learning_rate}-${final_label}"
  else
    dataset="${RUN_ROOT}/data/main-v2.jsonl"
    config="$(resolve_main_config "${learning_rate}" "${seed}")"
    path_root="${RUN_ROOT}/adapters/main/s${seed}/lr${learning_rate}"
    evidence_root="${RUN_ROOT}/evidence/main/s${seed}/lr${learning_rate}"
    final_label="final"
    final_candidate="main-s${seed}-lr${learning_rate}-${final_label}"
  fi
  require_file "${dataset}"
  require_file "${config}"
  validation="$("${PYTHON_BIN}" "${TRAIN_RUNTIME}" validate-prepared \
    --outer-manifest "${outer_manifest}" --core-manifest "${core_manifest}" \
    --target-audit "${target_audit}" --target-module "${TARGET_ENCODING_MODULE}" \
    --dataset "${dataset}" --training-config "${config}" \
    --run-kind "${kind}" --seed "${seed}" --learning-rate "${learning_rate}")"
  runtime_sha256="$(PYTHONPATH="${HERE}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" -c \
    'import sys; from day20_train_plugin_v2 import runtime_identity; print(runtime_identity(sys.argv[1])["runtime_sha256"])' \
    "${MS_SWIFT_COMMIT}")"
  canonical_summary="${evidence_root}/training-summary.json"
  if [[ -f "${canonical_summary}" ]]; then
    "${PYTHON_BIN}" "${CORE_TRAIN_PLUGIN}" verify-summary \
      --summary "${canonical_summary}" --run-root "${RUN_ROOT}" \
      --candidate "${final_candidate}" >/dev/null
    local linked_summary="" possible_summary
    while IFS= read -r possible_summary; do
      if [[ "${possible_summary}" -ef "${canonical_summary}" ]]; then
        linked_summary="${possible_summary}"
        break
      fi
    done < <(find "${evidence_root}" -mindepth 2 -maxdepth 2 -type f \
      -name training-summary.json -print | sort)
    [[ -n "${linked_summary}" ]] || {
      echo "canonical v2 summary is not linked to a v3 attempt" >&2
      exit 1
    }
    verify_sealed_attempt "${linked_summary}"
    echo "verified completed ${kind}: ${final_candidate}"
    return
  fi
  mkdir -p "${path_root}" "${evidence_root}"

  # A process can die after the v2 callback publishes its summary but before
  # the runner seals v3 envelopes. Complete that CPU-only step before deciding
  # whether another GPU training attempt is needed.
  local completed_summary completed_evidence
  while IFS= read -r completed_summary; do
    if "${PYTHON_BIN}" "${CORE_TRAIN_PLUGIN}" verify-summary \
      --summary "${completed_summary}" --run-root "${RUN_ROOT}" \
      --candidate "${final_candidate}" >/dev/null 2>&1; then
      completed_evidence="$(dirname "${completed_summary}")"
      require_file "${completed_evidence}/target-binding-config.json"
      require_file "${completed_evidence}/target-binding-attestation.json"
      "${PYTHON_BIN}" "${TRAIN_PLUGIN}" seal-summary \
        --summary "${completed_summary}" --run-root "${RUN_ROOT}" \
        --binding-config "${completed_evidence}/target-binding-config.json" \
        --attestation "${completed_evidence}/target-binding-attestation.json" \
        --output-dir "${completed_evidence}/checkpoint-envelopes" >/dev/null
      verify_sealed_attempt "${completed_summary}"
      require_absent "${canonical_summary}"
      ln -- "${completed_summary}" "${canonical_summary}"
      echo "sealed completed ${kind} without another GPU training attempt: ${final_candidate}"
      return
    fi
  done < <(find "${evidence_root}" -mindepth 2 -maxdepth 2 -type f \
    -name training-summary.json -print | sort)

  local attempt output_dir evidence_dir callback_config binding_config resume_checkpoint="" attempt_summary attestation
  while IFS= read -r attempt; do
    output_dir="${path_root}/${attempt}"
    evidence_dir="${evidence_root}/${attempt}"
    callback_config="${evidence_dir}/callback-config.json"
    binding_config="${evidence_dir}/target-binding-config.json"
    [[ -f "${callback_config}" && -f "${binding_config}" && ! -f "${evidence_dir}/training-summary.json" ]] || continue
    "${PYTHON_BIN}" "${TRAIN_PLUGIN}" verify-binding-config \
      --binding-config "${binding_config}" >/dev/null
    if resume_checkpoint="$("${PYTHON_BIN}" "${CORE_TRAIN_RUNTIME}" latest-resumable \
      --output-dir "${output_dir}" --run-root "${RUN_ROOT}" \
      --run-kind "${kind}" --seed "${seed}" --learning-rate "${learning_rate}" \
      --dataset-sha256 "$(printf '%s' "${validation}" | json_field dataset_file_sha256)" \
      --config-sha256 "$(printf '%s' "${validation}" | json_field training_config_file_sha256)" \
      --manifest-sha256 "$(printf '%s' "${validation}" | json_field manifest_sha256)" \
      --temporal-mix-sha256 "$(printf '%s' "${validation}" | json_field temporal_mix_sha256)" \
      --runtime-sha256 "${runtime_sha256}" \
      2>/dev/null)"; then
      break
    fi
    resume_checkpoint=""
  done < <(find "${path_root}" -mindepth 1 -maxdepth 1 -type d -name 'attempt-[0-9][0-9][0-9]' -printf '%f\n' | sort -r)

  if [[ -z "${resume_checkpoint}" ]]; then
    attempt="$(next_attempt_name "${path_root}")"
    output_dir="${path_root}/${attempt}"
    evidence_dir="${evidence_root}/${attempt}"
    callback_config="${evidence_dir}/callback-config.json"
    binding_config="${evidence_dir}/target-binding-config.json"
    require_absent "${output_dir}"
    require_absent "${evidence_dir}"
    mkdir "${output_dir}" "${evidence_dir}"
    "${PYTHON_BIN}" "${CORE_TRAIN_PLUGIN}" build-callback-config \
      --run-root "${RUN_ROOT}" --manifest "${core_manifest}" \
      --dataset "${dataset}" --training-config "${config}" \
      --run-kind "${kind}" --seed "${seed}" --learning-rate "${learning_rate}" \
      --evidence-dir "${evidence_dir}" \
      --e2b-preflight "${RUN_ROOT}/evidence/E2B-PREFLIGHT.json" \
      --expected-ms-swift-commit "${MS_SWIFT_COMMIT}" \
      --output "${callback_config}" >/dev/null
    "${PYTHON_BIN}" "${TRAIN_PLUGIN}" build-binding-config \
      --outer-manifest "${outer_manifest}" --core-manifest "${core_manifest}" \
      --target-audit "${target_audit}" --target-module "${TARGET_ENCODING_MODULE}" \
      --dataset "${dataset}" --training-config "${config}" \
      --v2-callback-config "${callback_config}" --evidence-dir "${evidence_dir}" \
      --run-kind "${kind}" --seed "${seed}" --learning-rate "${learning_rate}" \
      --output "${binding_config}" >/dev/null
  fi
  "${PYTHON_BIN}" "${TRAIN_PLUGIN}" verify-binding-config \
    --binding-config "${binding_config}" >/dev/null

  local target_regex save_limit log
  target_regex="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["training"]["target_regex"])' "${config}")"
  save_limit="$(printf '%s' "${validation}" | "${PYTHON_BIN}" -c 'import json,sys; print(len(json.load(sys.stdin)["checkpoint_tokens"]))')"
  log="${RUN_ROOT}/logs/${kind}-s${seed}-lr${learning_rate}-${attempt}.log"
  local -a resume_args=()
  [[ -z "${resume_checkpoint}" ]] || resume_args=(--resume_from_checkpoint "${resume_checkpoint}")

  DAY20_V2_REGISTER_SWIFT_CALLBACK=1 \
  DAY20_V2_CALLBACK_CONFIG="${callback_config}" \
  DAY20_V3_REGISTER_SWIFT_PLUGIN=1 \
  DAY20_V3_BINDING_CONFIG="${binding_config}" \
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" -m swift.cli.sft \
    --model "${BASE_MODEL}" \
    --model_type qwen3_5 \
    --template day20_qwen3_5_target_v3 \
    --dataset "${dataset}" \
    --output_dir "${output_dir}" \
    --add_version false \
    --tuner_type lora \
    --target_regex "${target_regex}" \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_bias none \
    --torch_dtype bfloat16 \
    --bf16 true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --max_length 2304 \
    --truncation_strategy delete \
    --packing false \
    --padding_free false \
    --enable_thinking false \
    --add_non_thinking_prefix true \
    --loss_scale default+ignore_empty_think \
    --split_dataset_ratio 0 \
    --dataset_shuffle false \
    --train_dataloader_shuffle false \
    --dataset_num_proc 1 \
    --dataloader_num_workers 0 \
    --dataloader_drop_last false \
    --group_by_length false \
    --strict true \
    --num_train_epochs 1 \
    --seed "${seed}" \
    --data_seed "${seed}" \
    --learning_rate "${learning_rate}" \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --optim adamw_torch \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-8 \
    --weight_decay 0 \
    --max_grad_norm 1.0 \
    --gradient_checkpointing true \
    --logging_strategy steps \
    --logging_steps 1 \
    --logging_first_step true \
    --report_to none \
    --save_strategy no \
    --save_total_limit "${save_limit}" \
    --save_only_model false \
    --external_plugins "${TRAIN_PLUGIN}" \
    --callbacks day20_v2_evidence day20_v3_target_binding \
    "${resume_args[@]}" \
    2>&1 | tee -a "${log}"

  attempt_summary="${evidence_dir}/training-summary.json"
  attestation="${evidence_dir}/target-binding-attestation.json"
  require_file "${attempt_summary}"
  require_file "${attestation}"
  "${PYTHON_BIN}" "${CORE_TRAIN_PLUGIN}" verify-summary \
    --summary "${attempt_summary}" --run-root "${RUN_ROOT}" \
    --candidate "${final_candidate}" >/dev/null
  "${PYTHON_BIN}" "${TRAIN_PLUGIN}" verify-attestation \
    --attestation "${attestation}" --binding-config "${binding_config}" >/dev/null
  "${PYTHON_BIN}" "${TRAIN_PLUGIN}" seal-summary \
    --summary "${attempt_summary}" --run-root "${RUN_ROOT}" \
    --binding-config "${binding_config}" --attestation "${attestation}" \
    --output-dir "${evidence_dir}/checkpoint-envelopes" >/dev/null
  verify_sealed_attempt "${attempt_summary}"
  require_absent "${canonical_summary}"
  ln -- "${attempt_summary}" "${canonical_summary}"
  echo "completed ${kind}: ${final_candidate}"
}

train_probe() {
  local learning_rate="$1"
  [[ "${learning_rate}" =~ ^(1e-4|3e-5|6e-5|8e-5)$ ]] || {
    echo "probe LR must be 1e-4, 3e-5, 6e-5, or 8e-5" >&2
    exit 1
  }
  run_training probe "${learning_rate}" 20260809
}

train_main() {
  local learning_rate="$1" seed="$2"
  [[ "${learning_rate}" =~ ^(1e-4|3e-5|6e-5|8e-5)$ ]] || {
    echo "main LR is outside the frozen Day 20 bracket" >&2
    exit 1
  }
  [[ "${seed}" == 20260809 || "${seed}" == 20260810 ]] || {
    echo "main seed must be 20260809 or 20260810" >&2
    exit 1
  }
  run_training main "${learning_rate}" "${seed}"
}

candidate_field() {
  local candidate="$1" field="$2"
  PYTHONPATH="${HERE}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" - \
    "${candidate}" "${field}" <<'PY'
import sys
from day20_candidate_factory_v2 import parse_candidate_id

candidate, field = sys.argv[1:]
identity = parse_candidate_id(candidate)
value = getattr(identity, field)
if value is None:
    raise SystemExit(f"candidate field is unavailable: {field}")
print(value)
PY
}

training_summary_for_candidate() {
  local candidate="$1" kind seed learning_rate
  kind="$(candidate_field "${candidate}" run_kind)"
  seed="$(candidate_field "${candidate}" seed)"
  learning_rate="$(candidate_field "${candidate}" learning_rate)"
  printf '%s\n' \
    "${RUN_ROOT}/evidence/${kind}/s${seed}/lr${learning_rate}/training-summary.json"
}

checkpoint_for_candidate() {
  local candidate="$1" role summary attempt_summary="" possible_summary
  role="$(candidate_field "${candidate}" model_role)"
  if [[ "${role}" == base ]]; then
    printf '%s\n' "${BASE_MODEL}"
    return
  fi
  summary="$(training_summary_for_candidate "${candidate}")"
  require_file "${summary}"
  while IFS= read -r possible_summary; do
    if [[ "${possible_summary}" -ef "${summary}" ]]; then
      attempt_summary="${possible_summary}"
      break
    fi
  done < <(find "$(dirname "${summary}")" -mindepth 2 -maxdepth 2 -type f \
    -name training-summary.json -print | sort)
  [[ -n "${attempt_summary}" ]] || {
    echo "candidate summary is not linked to a sealed v3 attempt: ${candidate}" >&2
    exit 1
  }
  verify_sealed_attempt "${attempt_summary}"
  "${PYTHON_BIN}" "${CORE_TRAIN_PLUGIN}" checkpoint-path \
    --summary "${summary}" --run-root "${RUN_ROOT}" --candidate "${candidate}"
}

evaluate_candidate() {
  local candidate="$1"
  acquire_lock
  load_run_root
  require_training_gate
  require_file "${EVALUATOR}"
  require_file "${RESCORER}"
  require_file "${EVAL_MANIFEST}"
  require_file "${SCORERS}"
  local scope role checkpoint raw_predictions raw_summary log
  scope="$(candidate_field "${candidate}" scope)"
  role="$(candidate_field "${candidate}" model_role)"
  checkpoint="$(checkpoint_for_candidate "${candidate}")"
  raw_predictions="${RUN_ROOT}/eval/${candidate}-${scope}-raw-predictions-v2.jsonl"
  raw_summary="${RUN_ROOT}/eval/${candidate}-${scope}-raw-summary-v2.json"
  log="${RUN_ROOT}/logs/${candidate}-eval-v2.log"
  local -a adapter_args=()
  if [[ "${role}" == lora ]]; then
    local training_summary
    training_summary="$(training_summary_for_candidate "${candidate}")"
    adapter_args=(--adapter "${checkpoint}" --training-summary "${training_summary}")
  fi
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${EVALUATOR}" \
    --model "${BASE_MODEL}" \
    "${adapter_args[@]}" \
    --eval-manifest "${EVAL_MANIFEST}" \
    --experiment-manifest "${RUN_ROOT}/DAY20-V2-MANIFEST.json" \
    --scorers "${SCORERS}" \
    --candidate "${candidate}" \
    --eval-scope "${scope}" \
    --output-dir "${RUN_ROOT}/eval" \
    2>&1 | tee -a "${log}"
  require_file "${raw_predictions}"
  require_file "${raw_summary}"
  "${PYTHON_BIN}" "${RESCORER}" \
    --raw-predictions "${raw_predictions}" \
    --raw-summary "${raw_summary}" \
    --eval-manifest "${EVAL_MANIFEST}" \
    --experiment-manifest "${RUN_ROOT}/DAY20-V2-MANIFEST.json" \
    --scorers "${SCORERS}" \
    --output-dir "${RUN_ROOT}/eval" \
    --scope "${scope}" \
    2>&1 | tee -a "${log}"
}

sandbox_candidate() {
  local candidate="$1"
  acquire_lock
  load_run_root
  require_training_gate
  require_file "${SANDBOX_PYTHON}"
  require_file "${CODE_SCORER}"
  local scope predictions summary output summary_output log
  scope="$(candidate_field "${candidate}" scope)"
  predictions="${RUN_ROOT}/eval/${candidate}.qwen35-v3.predictions.jsonl"
  summary="${RUN_ROOT}/eval/${candidate}.qwen35-v3.json"
  output="${RUN_ROOT}/eval/${candidate}-code-e2b-qwen35-v3.jsonl"
  summary_output="${RUN_ROOT}/eval/${candidate}-code-e2b-qwen35-v3-summary.json"
  log="${RUN_ROOT}/logs/${candidate}-code-e2b-v3.log"
  require_file "${predictions}"
  require_file "${summary}"
  (
    local key_line
    key_line="$(sed '/^[[:space:]]*$/d' "${E2B_CREDENTIAL_FILE}")"
    [[ "${key_line}" == E2B_API_KEY=* && "${key_line#E2B_API_KEY=}" != "" ]] || {
      echo "Day 20 v3 E2B credential format drifted" >&2
      exit 1
    }
    export E2B_API_KEY="${key_line#E2B_API_KEY=}"
    exec "${SANDBOX_PYTHON}" "${CODE_SCORER}" \
      --predictions "${predictions}" \
      --eval-summary "${summary}" \
      --frozen-e2b-scorer "${FROZEN_E2B_SCORER}" \
      --sandbox-config "${SANDBOX_CONFIG}" \
      --humaneval-source "${HUMANEVAL_SOURCE}" \
      --e2b-preflight "${RUN_ROOT}/evidence/E2B-PREFLIGHT.json" \
      --run-root "${RUN_ROOT}" \
      --output "${output}" \
      --summary-output "${summary_output}"
  ) 2>&1 | tee -a "${log}"
}

probe_candidate_ids() {
  local bracket_completed="$1"
  PYTHONPATH="${HERE}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" - \
    "${bracket_completed}" <<'PY'
import sys
from select_day20_v2 import expected_probe_cohort

for candidate in expected_probe_cohort(bracket_completed=sys.argv[1] == "true"):
    print(candidate)
PY
}

select_probes() {
  local phase="$1"
  [[ "${phase}" == stage-a || "${phase}" == bracket ]] || {
    echo "selection phase must be stage-a or bracket" >&2
    exit 1
  }
  acquire_lock
  load_run_root
  require_file "${PROBE_SELECTOR}"
  local bracket_completed=false output candidate status canonical
  output="${RUN_ROOT}/PROBE-SELECTION-STAGE-A.json"
  if [[ "${phase}" == bracket ]]; then
    bracket_completed=true
    output="${RUN_ROOT}/PROBE-SELECTION.json"
  fi
  local -a candidate_args=()
  while IFS= read -r candidate; do
    candidate_args+=(
      --candidate-e2b-summary
      "${RUN_ROOT}/eval/${candidate}-code-e2b-qwen35-v3-summary.json"
    )
  done < <(probe_candidate_ids "${bracket_completed}")
  local -a bracket_arg=()
  [[ "${bracket_completed}" == false ]] || bracket_arg=(--bracket-completed)
  "${PYTHON_BIN}" "${PROBE_SELECTOR}" \
    --base-e2b-summary \
    "${RUN_ROOT}/eval/base-probe-code-e2b-qwen35-v3-summary.json" \
    "${candidate_args[@]}" \
    "${bracket_arg[@]}" \
    --output "${output}"
  status="$("${PYTHON_BIN}" -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' \
    "${output}")"
  if [[ "${phase}" == stage-a && "${status}" != run_bracket ]]; then
    canonical="${RUN_ROOT}/PROBE-SELECTION.json"
    if [[ -f "${canonical}" ]]; then
      cmp -s -- "${output}" "${canonical}" || {
        echo "canonical probe selection differs from Stage A" >&2
        exit 1
      }
    else
      ln -- "${output}" "${canonical}"
    fi
  fi
}

run_probe_candidate_cohort() {
  local learning_rate="$1" candidate
  bash "${HERE}/run_day20_v3_autodl.sh" train-probe "${learning_rate}"
  while IFS= read -r candidate; do
    bash "${HERE}/run_day20_v3_autodl.sh" eval "${candidate}"
    bash "${HERE}/run_day20_v3_autodl.sh" sandbox "${candidate}"
  done < <(
    PYTHONPATH="${HERE}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" - \
      "${learning_rate}" <<'PY'
import sys
from day20_candidate_factory_v2 import (
    PRIMARY_SEED, PROBE_CHECKPOINT_TOKENS, candidate_id,
)

for label in PROBE_CHECKPOINT_TOKENS:
    print(candidate_id(
        run_kind="probe", seed=PRIMARY_SEED,
        learning_rate=sys.argv[1], checkpoint=label,
    ))
PY
  )
}

probe_funnel() {
  bash "${HERE}/run_day20_v3_autodl.sh" preflight
  bash "${HERE}/run_day20_v3_autodl.sh" eval base-probe
  bash "${HERE}/run_day20_v3_autodl.sh" sandbox base-probe
  run_probe_candidate_cohort 1e-4
  bash "${HERE}/run_day20_v3_autodl.sh" select-probe stage-a
  local run_root status
  run_root="$(sed -n '1p' "${STATE_FILE}")"
  status="$("${PYTHON_BIN}" -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' \
    "${run_root}/PROBE-SELECTION-STAGE-A.json")"
  if [[ "${status}" == run_bracket ]]; then
    local learning_rate
    for learning_rate in 3e-5 6e-5 8e-5; do
      run_probe_candidate_cohort "${learning_rate}"
    done
    bash "${HERE}/run_day20_v3_autodl.sh" select-probe bracket
    status="$("${PYTHON_BIN}" -c \
      'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' \
      "${run_root}/PROBE-SELECTION.json")"
  fi
  if [[ "${status}" == run_main ]]; then
    echo "probe funnel selected a passing candidate"
  else
    echo "probe funnel completed with no passing candidate; Base remains active"
  fi
}

verified_probe_selection_field() {
  local field="$1"
  PYTHONPATH="${HERE}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" - \
    "${RUN_ROOT}/PROBE-SELECTION.json" "${field}" <<'PY'
import sys
from pathlib import Path
from select_day20_v2 import verify_selection

value = verify_selection(Path(sys.argv[1]))
for component in sys.argv[2].split("."):
    value = value[component]
if value is None:
    raise SystemExit(f"selection field is empty: {sys.argv[2]}")
print(value)
PY
}

main_candidate_ids() {
  local seed="$1" learning_rate="$2"
  PYTHONPATH="${HERE}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" - \
    "${seed}" "${learning_rate}" <<'PY'
import sys
from day20_candidate_factory_v2 import MAIN_CHECKPOINT_TOKENS, candidate_id

for label in MAIN_CHECKPOINT_TOKENS:
    print(candidate_id(
        run_kind="main", seed=int(sys.argv[1]),
        learning_rate=sys.argv[2], checkpoint=label,
    ))
PY
}

select_primary_main() {
  acquire_lock
  load_run_root
  require_file "${MAIN_SELECTOR}"
  local learning_rate candidate
  learning_rate="$(verified_probe_selection_field selected_learning_rate)"
  local -a primary_args=()
  while IFS= read -r candidate; do
    primary_args+=(
      --primary-summary
      "${RUN_ROOT}/eval/${candidate}-code-e2b-qwen35-v3-summary.json"
    )
  done < <(main_candidate_ids 20260809 "${learning_rate}")
  "${PYTHON_BIN}" "${MAIN_SELECTOR}" primary \
    --base-summary "${RUN_ROOT}/eval/base-full-code-e2b-qwen35-v3-summary.json" \
    "${primary_args[@]}" \
    --output-dir "${RUN_ROOT}"
}

main_stage() {
  load_run_root
  local selection_status learning_rate candidate
  selection_status="$(verified_probe_selection_field status)"
  [[ "${selection_status}" == run_main ]] || {
    echo "probe selection does not authorize Main: ${selection_status}" >&2
    exit 1
  }
  learning_rate="$(verified_probe_selection_field selected_learning_rate)"
  bash "${HERE}/run_day20_v3_autodl.sh" train-main "${learning_rate}" 20260809
  bash "${HERE}/run_day20_v3_autodl.sh" eval base-full
  bash "${HERE}/run_day20_v3_autodl.sh" sandbox base-full
  while IFS= read -r candidate; do
    bash "${HERE}/run_day20_v3_autodl.sh" eval "${candidate}"
    bash "${HERE}/run_day20_v3_autodl.sh" sandbox "${candidate}"
  done < <(main_candidate_ids 20260809 "${learning_rate}")
  bash "${HERE}/run_day20_v3_autodl.sh" select-main
}

primary_selection_field() {
  local field="$1"
  PYTHONPATH="${HERE}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" - \
    "${RUN_ROOT}/PRIMARY-MAIN-SELECTION.json" "${field}" <<'PY'
import sys
from pathlib import Path
from select_day20_main_v2 import verify_primary_selection

value = verify_primary_selection(Path(sys.argv[1]))
for component in sys.argv[2].split("."):
    value = value[component]
if value is None:
    print("__NONE__")
else:
    print(value)
PY
}

publish_confirmation() {
  local confirmation_summary="${1:-}"
  acquire_lock
  load_run_root
  require_file "${MAIN_SELECTOR}"
  local -a summary_arg=()
  [[ -z "${confirmation_summary}" ]] || summary_arg=(
    --confirmation-summary "${confirmation_summary}"
  )
  "${PYTHON_BIN}" "${MAIN_SELECTOR}" confirmation \
    --primary-selection "${RUN_ROOT}/PRIMARY-MAIN-SELECTION.json" \
    "${summary_arg[@]}" \
    --output-dir "${RUN_ROOT}"
}

confirmation_stage() {
  load_run_root
  require_file "${RUN_ROOT}/PRIMARY-MAIN-SELECTION.json"
  local winner learning_rate checkpoint confirmation_candidate confirmation_summary
  winner="$(primary_selection_field decision.primary_winner)"
  if [[ "${winner}" == __NONE__ ]]; then
    bash "${HERE}/run_day20_v3_autodl.sh" select-confirmation
    echo "primary Main had no eligible checkpoint; recorded Base fallback"
    return
  fi
  learning_rate="$(candidate_field "${winner}" learning_rate)"
  checkpoint="$(candidate_field "${winner}" checkpoint_label)"
  confirmation_candidate="$(PYTHONPATH="${HERE}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" - "${learning_rate}" "${checkpoint}" <<'PY'
import sys
from day20_candidate_factory_v2 import CONFIRMATION_SEED, candidate_id
print(candidate_id(
    run_kind="main", seed=CONFIRMATION_SEED,
    learning_rate=sys.argv[1], checkpoint=sys.argv[2],
))
PY
)"
  bash "${HERE}/run_day20_v3_autodl.sh" train-main "${learning_rate}" 20260810
  bash "${HERE}/run_day20_v3_autodl.sh" eval "${confirmation_candidate}"
  bash "${HERE}/run_day20_v3_autodl.sh" sandbox "${confirmation_candidate}"
  load_run_root
  confirmation_summary="${RUN_ROOT}/eval/${confirmation_candidate}-code-e2b-qwen35-v3-summary.json"
  bash "${HERE}/run_day20_v3_autodl.sh" select-confirmation "${confirmation_summary}"
}

finalize_run() {
  acquire_lock
  load_run_root
  require_file "${MAIN_SELECTOR}"
  require_file "${RUN_ROOT}/FINAL-PROMOTION.json"
  local verified action selected
  verified="$("${PYTHON_BIN}" "${MAIN_SELECTOR}" verify-final \
    --promotion "${RUN_ROOT}/FINAL-PROMOTION.json")"
  action="$(printf '%s' "${verified}" | "${PYTHON_BIN}" -c \
    'import json,sys; print(json.load(sys.stdin)["decision"]["action"])')"
  selected="$(printf '%s' "${verified}" | "${PYTHON_BIN}" -c \
    'import json,sys; print(json.load(sys.stdin)["decision"]["selected_candidate"])')"
  printf 'final_action=%s\nselected_candidate=%s\nmerge_performed=false\n' \
    "${action}" "${selected}"
}

status() {
  acquire_lock
  load_run_root
  printf 'run_root=%s\n' "${RUN_ROOT}"
  for path in \
    "${RUN_ROOT}/DAY20-V3-MANIFEST.json" \
    "${RUN_ROOT}/DAY20-V2-MANIFEST.json" \
    "${RUN_ROOT}/TARGET-ENCODING-AUDIT.json" \
    "${RUN_ROOT}/evidence/E2B-PREFLIGHT.json" \
    "${RUN_ROOT}/PROBE-SELECTION-STAGE-A.json" \
    "${RUN_ROOT}/PROBE-SELECTION.json" \
    "${RUN_ROOT}/PRIMARY-MAIN-SELECTION.json" \
    "${RUN_ROOT}/FINAL-PROMOTION.json"; do
    if [[ -f "${path}" ]]; then
      printf 'present=%s\n' "${path}"
    else
      printf 'missing=%s\n' "${path}"
    fi
  done
  find "${RUN_ROOT}/evidence" -name training-summary.json -type f -print | sort
  find "${RUN_ROOT}/eval" \( -name '*.qwen35-v3.json' -o \
    -name '*-code-e2b-qwen35-v3-summary.json' \) -type f -print | sort
}

usage() {
  echo "usage: $0 {init|prepare|preflight|e2b-preflight|train-probe LR|train-main LR SEED|eval CANDIDATE|sandbox CANDIDATE|select-probe stage-a|bracket|probe-funnel|main|select-main|confirm|select-confirmation [SUMMARY]|finalize|status}" >&2
  exit 1
}

case "${1:-}" in
  init) [[ $# == 1 ]] || usage; init_run ;;
  prepare) [[ $# == 1 ]] || usage; prepare_run ;;
  preflight) [[ $# == 1 ]] || usage; preflight ;;
  e2b-preflight) [[ $# == 1 ]] || usage; e2b_preflight ;;
  train-probe) [[ $# == 2 ]] || usage; train_probe "$2" ;;
  train-main) [[ $# == 3 ]] || usage; train_main "$2" "$3" ;;
  eval) [[ $# == 2 ]] || usage; evaluate_candidate "$2" ;;
  sandbox) [[ $# == 2 ]] || usage; sandbox_candidate "$2" ;;
  select-probe) [[ $# == 2 ]] || usage; select_probes "$2" ;;
  probe-funnel) [[ $# == 1 ]] || usage; probe_funnel ;;
  main) [[ $# == 1 ]] || usage; main_stage ;;
  select-main) [[ $# == 1 ]] || usage; select_primary_main ;;
  confirm) [[ $# == 1 ]] || usage; confirmation_stage ;;
  select-confirmation)
    [[ $# == 1 || $# == 2 ]] || usage
    publish_confirmation "${2:-}"
    ;;
  finalize) [[ $# == 1 ]] || usage; finalize_run ;;
  status) [[ $# == 1 ]] || usage; status ;;
  *) usage ;;
esac
