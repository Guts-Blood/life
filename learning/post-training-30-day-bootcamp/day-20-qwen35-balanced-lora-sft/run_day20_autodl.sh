#!/usr/bin/env bash
set -euo pipefail

BOOTCAMP_ROOT="/root/autodl-tmp/post-training-30-day-bootcamp"
HERE="${BOOTCAMP_ROOT}/day-20-qwen35-balanced-lora-sft"
DAY19_DIR="${BOOTCAMP_ROOT}/day-19-qwen35-sft-comparison"
DATA_ROOT="/root/autodl-tmp/qwen35-v2"
VENV="${DATA_ROOT}/venv"
PYTHON_BIN="${VENV}/bin/python"
CUDNN_LIBRARY_DIR="${VENV}/lib/python3.12/site-packages/nvidia/cudnn/lib"
MS_SWIFT_ROOT="/root/autodl-tmp/ms-swift"
MS_SWIFT_COMMIT="565a1ad586a21d24b23931c52d2c62b49c39bee8"
BASE_MODEL="/root/autodl-tmp/Qwen--Qwen3.5-4B-Base/1001bb4d826a52d1f399e183466143f4da7b741b"
STATE_FILE="${DATA_ROOT}/state/current-day20-run-root"
SOURCE_ROOT="${DAY20_SOURCE_ROOT:-${DATA_ROOT}/day20-sources}"
SOURCE_FETCH_CONFIG="${DAY20_SOURCE_FETCH_CONFIG:-${DATA_ROOT}/day20-source-config.json}"

PREPARER="${HERE}/prepare_day20.py"
SOURCE_ADAPTER="${HERE}/day20_source_adapter.py"
TRAIN_PLUGIN="${HERE}/day20_train_plugin.py"
EVALUATOR="${HERE}/evaluate_day20.py"
RESCORER="${HERE}/rescore_day20_qwen35_v2.py"
CODE_SCORER="${HERE}/score_day20_code_e2b_v2.py"
FINALIZER="${HERE}/finalize_day20.py"
EVAL_MANIFEST="${BOOTCAMP_ROOT}/artifacts/eval/day10-frozen-eval-manifest.json"
SCORERS="${BOOTCAMP_ROOT}/day-10-frozen-eval-baseline/day10_scorers.py"
FROZEN_E2B_SCORER="${BOOTCAMP_ROOT}/day-10-frozen-eval-baseline/score_day10_code_e2b.py"
SANDBOX_CONFIG="${BOOTCAMP_ROOT}/day-10-frozen-eval-baseline/day10_e2b_sandbox_config.json"
HUMANEVAL_SOURCE="/root/autodl-tmp/runs/day19-qwen35-20260808T100142Z/configs/frozen-inputs/HumanEval.jsonl.gz"
SANDBOX_PYTHON="/root/autodl-tmp/envs/day12-e2b/bin/python"
E2B_CREDENTIAL_FILE="/root/autodl-tmp/secrets/day20-e2b.env"
E2B_CREDENTIAL_ATTESTATION="/root/autodl-tmp/secrets/day20-e2b-attestation.json"

PROBE_LRS=(1e-5 3e-5 1e-4)

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export USE_HF=1
export CUDA_HOME="${VENV}"
export CUDA_PATH="${CUDA_HOME}"
export NVTE_CUDA_INCLUDE_DIR="${CUDA_HOME}/targets/x86_64-linux/include"
export PATH="${VENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDNN_LIBRARY_DIR}:${CUDA_HOME}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PIP_CACHE_DIR="${DATA_ROOT}/cache/pip"
export TMPDIR="${DATA_ROOT}/tmp"

require_file() {
  [[ -f "$1" ]] || { echo "required file is missing: $1" >&2; exit 1; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "required directory is missing: $1" >&2; exit 1; }
}

require_absent() {
  [[ ! -e "$1" ]] || { echo "refusing to overwrite artifact: $1" >&2; exit 1; }
}

require_mode_0600_root() {
  local path="$1"
  require_file "${path}"
  [[ "$(stat -c '%a' -- "${path}")" == "600" ]] || {
    echo "credential artifact must have mode 0600: ${path}" >&2
    exit 1
  }
  [[ "$(stat -c '%U' -- "${path}")" == "root" ]] || {
    echo "credential artifact must be owned by root: ${path}" >&2
    exit 1
  }
}

acquire_global_lock() {
  command -v flock >/dev/null || {
    echo "required command is missing: flock" >&2
    exit 1
  }
  mkdir -p "$(dirname "${STATE_FILE}")"
  exec 9>"${STATE_FILE}.lock"
  flock -n 9 || {
    echo "another Day 20 operation holds the global run-state lock" >&2
    exit 1
  }
}

acquire_init_lock() {
  acquire_global_lock
  exec 7>"${STATE_FILE}.init.lock"
  flock -n 7 || {
    echo "another Day 20 init operation is already running" >&2
    exit 1
  }
}

acquire_run_operation_locks() {
  local operation="$1"
  [[ "${operation}" =~ ^[a-z-]+$ ]] || {
    echo "unsafe Day 20 operation lock name: ${operation}" >&2
    exit 1
  }
  acquire_global_lock
  load_run_root
  require_dir "${RUN_ROOT}/.locks"
  exec 8>"${RUN_ROOT}/.locks/run.lock"
  flock -n 8 || {
    echo "another operation is active for Day 20 run ${RUN_ROOT}" >&2
    exit 1
  }
  exec 7>"${RUN_ROOT}/.locks/${operation}.lock"
  flock -n 7 || {
    echo "another Day 20 ${operation} operation is already running" >&2
    exit 1
  }
}

require_remote_contract() {
  [[ "$(id -un)" == "root" ]] || {
    echo "Day 20 requires remote user root" >&2
    exit 1
  }
  require_file "${PYTHON_BIN}"
  require_dir "${CUDNN_LIBRARY_DIR}"
  require_dir "${BASE_MODEL}"
  require_dir "${MS_SWIFT_ROOT}"
  require_dir "${BOOTCAMP_ROOT}"
  local path
  for path in "${PREPARER}" "${SOURCE_ADAPTER}" "${TRAIN_PLUGIN}" "${EVALUATOR}" "${RESCORER}" \
    "${CODE_SCORER}" "${FINALIZER}" "${EVAL_MANIFEST}" "${SCORERS}"; do
    require_file "${path}"
  done
  local gpu_name
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
  [[ "${gpu_name}" == *H800* ]] || {
    echo "Day 20 requires GPU 0 to be an H800, found: ${gpu_name}" >&2
    exit 1
  }
  [[ "$(git -C "${MS_SWIFT_ROOT}" rev-parse HEAD)" == "${MS_SWIFT_COMMIT}" ]] || {
    echo "pinned ms-swift commit drifted" >&2
    exit 1
  }
  [[ -z "$(git -C "${MS_SWIFT_ROOT}" status --porcelain)" ]] || {
    echo "ms-swift worktree must be clean" >&2
    exit 1
  }
  "${PYTHON_BIN}" - <<'PY'
import shutil
free = shutil.disk_usage('/root/autodl-tmp').free
if free < 80 * 1024**3:
    raise SystemExit(f'Day 20 requires at least 80 GiB free, found {free / 1024**3:.2f} GiB')
PY
}

load_run_root() {
  require_file "${STATE_FILE}"
  local declared
  declared="$(<"${STATE_FILE}")"
  [[ "${declared}" == /root/autodl-tmp/runs/day20-qwen35-lora-* ]] || {
    echo "unsafe Day 20 run root in state: ${declared}" >&2
    exit 1
  }
  RUN_ROOT="$(realpath -e -- "${declared}")"
  [[ "${RUN_ROOT}" == "${declared}" ]] || {
    echo "Day 20 run root must not resolve through a symlink: ${declared}" >&2
    exit 1
  }
  require_file "${RUN_ROOT}/.day20-run-root"
  [[ "$(<"${RUN_ROOT}/.day20-run-root")" == "day20-qwen35-balanced-lora-sft-v1" ]] || {
    echo "Day 20 run marker is invalid" >&2
    exit 1
  }
}

json_field() {
  local path="$1"
  local dotted="$2"
  "${PYTHON_BIN}" - "${path}" "${dotted}" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding='utf-8'))
for key in sys.argv[2].split('.'):
    value = value[key]
if isinstance(value, bool):
    print(str(value).lower())
elif isinstance(value, (dict, list)):
    print(json.dumps(value, separators=(',', ':'), sort_keys=True))
else:
    print(value)
PY
}

require_candidate() {
  case "$1" in
    base|base-probe|early|mid|final|probe-1e-5|probe-3e-5|probe-1e-4) ;;
    *) echo "unknown Day 20 candidate: $1" >&2; exit 1 ;;
  esac
}

init_run() {
  require_remote_contract
  if [[ -f "${STATE_FILE}" ]]; then
    local current
    current="$(<"${STATE_FILE}")"
    if [[ -n "${current}" && -d "${current}" && ! -f "${current}/DAY20-RESULTS.json" ]]; then
      if [[ -f "${current}/evidence/prepare-in-progress" \
        && ! -f "${current}/DAY20-MANIFEST.json" ]]; then
        echo "retaining abandoned prepare attempt: ${current}" >&2
      else
        echo "unfinished Day 20 run already exists: ${current}" >&2
        exit 1
      fi
    fi
  fi
  local timestamp state_attempt
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  RUN_ROOT="/root/autodl-tmp/runs/day20-qwen35-lora-${timestamp}"
  require_absent "${RUN_ROOT}"
  mkdir "${RUN_ROOT}"
  printf '%s\n' 'day20-qwen35-balanced-lora-sft-v1' > "${RUN_ROOT}/.day20-run-root"
  mkdir -p "${RUN_ROOT}"/{.locks,adapters,configs,data,eval,evidence,exports,logs,tmp}
  state_attempt="${STATE_FILE}.$$"
  require_absent "${state_attempt}"
  printf '%s\n' "${RUN_ROOT}" > "${state_attempt}"
  mv -T -- "${state_attempt}" "${STATE_FILE}"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader > "${RUN_ROOT}/evidence/gpu-inventory.txt"
  "${PYTHON_BIN}" - "${RUN_ROOT}" "${BASE_MODEL}" <<'PY'
import hashlib, json, platform, socket, sys
from datetime import datetime, timezone
from pathlib import Path
root, model = map(Path, sys.argv[1:])
payload = {
    'schema_version': 1,
    'domain': 'day20.host_inventory',
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'hostname': socket.gethostname(),
    'platform': platform.platform(),
    'base_model': str(model.resolve()),
}
payload['inventory_sha256'] = hashlib.sha256(json.dumps(
    payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
(root / 'evidence/host-inventory.json').write_text(
    json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY
  echo "${RUN_ROOT}"
}

prepare_data() {
  load_run_root
  require_absent "${RUN_ROOT}/DAY20-MANIFEST.json"
  local prepare_marker="${RUN_ROOT}/evidence/prepare-in-progress"
  local prepare_complete="${RUN_ROOT}/evidence/prepare-complete"
  require_absent "${prepare_marker}"
  require_absent "${prepare_complete}"
  printf '%s\n' "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${prepare_marker}"
  local skill normalized_ready=true candidate_source
  local -a source_args=()
  local -A normalized_sources=()
  if [[ ! -f "${SOURCE_ROOT}/SOURCE-ADAPTER-MANIFEST.json" ]]; then
    normalized_ready=false
  fi
  for skill in general math finance code; do
    candidate_source="${SOURCE_ROOT}/${skill}.normalized.qwen35.jsonl"
    if [[ ! -f "${candidate_source}" && -f "${SOURCE_ROOT}/${skill}.jsonl" ]]; then
      candidate_source="${SOURCE_ROOT}/${skill}.jsonl"
    fi
    if [[ ! -f "${candidate_source}" ]]; then
      normalized_ready=false
    else
      normalized_sources["${skill}"]="${candidate_source}"
    fi
  done
  if [[ "${normalized_ready}" == true ]]; then
    source_args=(
      --source "general=${normalized_sources[general]}"
      --source "math=${normalized_sources[math]}"
      --source "finance=${normalized_sources[finance]}"
      --source "code=${normalized_sources[code]}"
    )
  else
    require_file "${SOURCE_FETCH_CONFIG}"
    source_args=(
      --fetch-adapter day20_source_adapter:fetch_sources
      --fetch-config "${SOURCE_FETCH_CONFIG}"
    )
  fi
  if ! CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${PREPARER}" \
      --run-root "${RUN_ROOT}" \
      --bootcamp-root "${BOOTCAMP_ROOT}" \
      --model "${BASE_MODEL}" \
      "${source_args[@]}" \
      --eval-manifest "${EVAL_MANIFEST}" \
      --target-tokens 256000 \
      --probe-tokens 16000 \
      --max-length 2304 \
      2>&1 | tee "${RUN_ROOT}/logs/prepare.log"; then
    echo "prepare failed; retained ${RUN_ROOT}; run init to start a clean run" >&2
    return 1
  fi
  require_file "${RUN_ROOT}/DAY20-MANIFEST.json"
  mv -T -- "${prepare_marker}" "${prepare_complete}"
}

training_summary_path() {
  local kind="$1"
  local lr="${2:-}"
  if [[ "${kind}" == "probe" ]]; then
    printf '%s\n' "${RUN_ROOT}/evidence/probe-${lr}/training-summary.json"
  else
    printf '%s\n' "${RUN_ROOT}/evidence/main/training-summary.json"
  fi
}

checkpoint_name_for_kind() {
  if [[ "$1" == "probe" ]]; then
    printf '%s\n' probe
  else
    printf '%s\n' final
  fi
}

publish_completed_training_attempt() {
  local kind="$1"
  local output_root="$2"
  local evidence_root="$3"
  local canonical_summary="$4"
  local checkpoint_name attempt_evidence attempt_name attempt_output attempt_summary checkpoint
  checkpoint_name="$(checkpoint_name_for_kind "${kind}")"
  while IFS= read -r attempt_evidence; do
    [[ -n "${attempt_evidence}" ]] || continue
    attempt_name="${attempt_evidence##*/}"
    attempt_output="${output_root}/${attempt_name}"
    attempt_summary="${attempt_evidence}/training-summary.json"
    [[ -d "${attempt_output}" && -f "${attempt_summary}" ]] || continue
    if checkpoint="$("${PYTHON_BIN}" "${TRAIN_PLUGIN}" checkpoint-path \
      --summary "${attempt_summary}" --name "${checkpoint_name}" \
      --run-root "${RUN_ROOT}" 2>/dev/null)"; then
      require_absent "${canonical_summary}"
      ln -- "${attempt_summary}" "${canonical_summary}"
      "${PYTHON_BIN}" "${TRAIN_PLUGIN}" checkpoint-path \
        --summary "${canonical_summary}" --name "${checkpoint_name}" \
        --run-root "${RUN_ROOT}" >/dev/null
      echo "recovered completed ${kind} training from ${attempt_name}"
      return 0
    fi
  done < <(find "${evidence_root}" -mindepth 1 -maxdepth 1 -type d \
    -name 'attempt-[0-9][0-9][0-9]' -print | sort -r)
  return 1
}

run_swift_training() {
  local kind="$1"
  local learning_rate="$2"
  local manifest="${RUN_ROOT}/DAY20-MANIFEST.json"
  require_file "${manifest}"
  local validation dataset target_regex expected_tokens output_root evidence_root output_dir evidence_dir log targets save_limit save_only_model summary training_config checkpoint_name
  validation="$("${PYTHON_BIN}" "${TRAIN_PLUGIN}" validate-inputs \
    --manifest "${manifest}" --dataset-kind "${kind}" --learning-rate "${learning_rate}")"
  dataset="$("${PYTHON_BIN}" -c 'import json,sys; print(json.loads(sys.argv[1])["dataset"])' "${validation}")"
  target_regex="$("${PYTHON_BIN}" -c 'import json,sys; print(json.loads(sys.argv[1])["target_regex"])' "${validation}")"
  expected_tokens="$("${PYTHON_BIN}" -c 'import json,sys; print(json.loads(sys.argv[1])["expected_supervised_tokens"])' "${validation}")"
  if [[ "${kind}" == "probe" ]]; then
    output_root="${RUN_ROOT}/adapters/probe-${learning_rate}"
    evidence_root="${RUN_ROOT}/evidence/probe-${learning_rate}"
    targets='{"probe":16000}'
    save_limit=1
    save_only_model=true
    training_config="$("${PYTHON_BIN}" -c 'import json,sys; print(json.loads(sys.argv[1])["prepared_config"])' "${validation}")"
  else
    output_root="${RUN_ROOT}/adapters/main"
    evidence_root="${RUN_ROOT}/evidence/main"
    targets='{"early":64000,"mid":153600,"final":256000}'
    save_limit=3
    save_only_model=false
    training_config="${RUN_ROOT}/configs/main-selected.json"
  fi
  require_file "${training_config}"
  summary="$(training_summary_path "${kind}" "${learning_rate}")"
  checkpoint_name="$(checkpoint_name_for_kind "${kind}")"
  if [[ -f "${summary}" ]]; then
    "${PYTHON_BIN}" "${TRAIN_PLUGIN}" checkpoint-path --summary "${summary}" \
      --name "${checkpoint_name}" \
      --run-root "${RUN_ROOT}" >/dev/null
    echo "verified completed ${kind} training at lr=${learning_rate}"
    return
  fi
  mkdir -p "${output_root}" "${evidence_root}"
  if publish_completed_training_attempt \
    "${kind}" "${output_root}" "${evidence_root}" "${summary}"; then
    return
  fi
  local attempt_name="" attempt_number=0 candidate_attempt candidate_number
  local newest="" resume_source_evidence="" resume_step=0 candidate_checkpoint candidate_step
  local -a resume_args=()
  if [[ "${kind}" == "main" ]]; then
    while IFS= read -r candidate_attempt; do
      [[ -n "${candidate_attempt}" ]] || continue
      candidate_number="${candidate_attempt##*/}"
      local resume_evidence="${evidence_root}/${candidate_number}"
      if [[ ! -f "${resume_evidence}/step-metrics.jsonl" \
        || ! -f "${resume_evidence}/trainable-inventory.json" \
        || ! -f "${resume_evidence}/trainer-token-reaudit.json" \
        || ! -f "${resume_evidence}/five-step-safety.json" \
        || ! -f "${resume_evidence}/runtime-identity.json" \
        || -f "${resume_evidence}/training-summary.json" ]]; then
        continue
      fi
      if candidate_checkpoint="$("${PYTHON_BIN}" "${TRAIN_PLUGIN}" resume-checkpoint \
        --output-dir "${candidate_attempt}" --run-root "${RUN_ROOT}" 2>/dev/null)"; then
        candidate_step="${candidate_checkpoint##*-}"
        if [[ "${candidate_step}" =~ ^[0-9]+$ ]] \
          && (( 10#${candidate_step} > resume_step )); then
          resume_step=$((10#${candidate_step}))
          newest="${candidate_checkpoint}"
          resume_source_evidence="${resume_evidence}"
        fi
      fi
    done < <(find "${output_root}" -mindepth 1 -maxdepth 1 -type d \
      -name 'attempt-[0-9][0-9][0-9]' -print | sort -r)
  fi
  if [[ -n "${newest}" ]]; then
    resume_args=(--resume_from_checkpoint "${newest}")
  fi
  if [[ -z "${attempt_name}" ]]; then
    for candidate_attempt in "${output_root}" "${evidence_root}"; do
      while IFS= read -r candidate_number; do
        [[ -n "${candidate_number}" ]] || continue
        candidate_number="${candidate_number##*-}"
        if [[ "${candidate_number}" =~ ^[0-9]{3}$ ]] \
          && (( 10#${candidate_number} > attempt_number )); then
          attempt_number=$((10#${candidate_number}))
        fi
      done < <(find "${candidate_attempt}" -mindepth 1 -maxdepth 1 -type d \
        -name 'attempt-[0-9][0-9][0-9]' -print)
    done
    (( attempt_number < 999 )) || {
      echo "Day 20 training attempt namespace is exhausted" >&2
      exit 1
    }
    attempt_number=$((attempt_number + 1))
    printf -v attempt_name 'attempt-%03d' "${attempt_number}"
    require_absent "${output_root}/${attempt_name}"
    require_absent "${evidence_root}/${attempt_name}"
    mkdir "${output_root}/${attempt_name}" "${evidence_root}/${attempt_name}"
  fi
  output_dir="${output_root}/${attempt_name}"
  evidence_dir="${evidence_root}/${attempt_name}"
  require_dir "${evidence_dir}"
  if [[ -n "${resume_source_evidence}" ]]; then
    "${PYTHON_BIN}" "${TRAIN_PLUGIN}" stage-resume-evidence \
      --source-evidence "${resume_source_evidence}" \
      --destination-evidence "${evidence_dir}" \
      --checkpoint "${newest}" --run-root "${RUN_ROOT}" >/dev/null
  fi
  log="${RUN_ROOT}/logs/$(basename "${output_root}")-${attempt_name}-train.log"
  DAY20_REGISTER_SWIFT_CALLBACK=1 \
  DAY20_PLUGIN_EVIDENCE_DIR="${evidence_dir}" \
  DAY20_DATASET_PATH="${dataset}" \
  DAY20_EXPECTED_SUPERVISED_TOKENS="${expected_tokens}" \
  DAY20_CHECKPOINT_TARGETS_JSON="${targets}" \
  DAY20_TARGET_REGEX="${target_regex}" \
  DAY20_RUN_KIND="${kind}" \
  DAY20_EXPECTED_LR="${learning_rate}" \
  DAY20_TRAINING_CONFIG="${training_config}" \
  DAY20_EXPECTED_MS_SWIFT_COMMIT="${MS_SWIFT_COMMIT}" \
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" -m swift.cli.sft \
    --model "${BASE_MODEL}" \
    --model_type qwen3_5 \
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
    --seed 20260809 \
    --data_seed 20260809 \
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
    --save_only_model "${save_only_model}" \
    --external_plugins "${TRAIN_PLUGIN}" \
    --callbacks day20_evidence \
    "${resume_args[@]}" \
    2>&1 | tee -a "${log}"
  local attempt_summary="${evidence_dir}/training-summary.json"
  require_file "${attempt_summary}"
  require_absent "${summary}"
  ln -- "${attempt_summary}" "${summary}"
  "${PYTHON_BIN}" "${TRAIN_PLUGIN}" checkpoint-path --summary "${summary}" \
    --name "${checkpoint_name}" \
    --run-root "${RUN_ROOT}" >/dev/null
}

train_probe() {
  local learning_rate
  learning_rate="$("${PYTHON_BIN}" -c 'import sys; sys.path.insert(0, sys.argv[1]); from day20_train_plugin import canonical_lr; print(canonical_lr(sys.argv[2]))' "${HERE}" "$1")"
  load_run_root
  run_swift_training probe "${learning_rate}"
}

checkpoint_for_candidate() {
  local candidate="$1"
  require_candidate "${candidate}"
  case "${candidate}" in
    base|base-probe) printf '%s\n' "${BASE_MODEL}" ;;
    early|mid|final)
      "${PYTHON_BIN}" "${TRAIN_PLUGIN}" checkpoint-path \
        --summary "${RUN_ROOT}/evidence/main/training-summary.json" \
        --name "${candidate}" --run-root "${RUN_ROOT}"
      ;;
    probe-*)
      local learning_rate="${candidate#probe-}"
      "${PYTHON_BIN}" "${TRAIN_PLUGIN}" checkpoint-path \
        --summary "${RUN_ROOT}/evidence/probe-${learning_rate}/training-summary.json" \
        --name probe --run-root "${RUN_ROOT}"
      ;;
  esac
}

training_summary_for_candidate() {
  local candidate="$1"
  case "${candidate}" in
    early|mid|final)
      printf '%s\n' "${RUN_ROOT}/evidence/main/training-summary.json"
      ;;
    probe-*)
      printf '%s\n' "${RUN_ROOT}/evidence/probe-${candidate#probe-}/training-summary.json"
      ;;
    *)
      echo "candidate has no LoRA training summary: ${candidate}" >&2
      exit 1
      ;;
  esac
}

evaluate_candidate() {
  local candidate="$1"
  require_candidate "${candidate}"
  load_run_root
  local checkpoint raw_summary raw_predictions v2_summary v2_predictions log
  checkpoint="$(checkpoint_for_candidate "${candidate}")"
  raw_summary="${RUN_ROOT}/eval/${candidate}.raw.json"
  raw_predictions="${RUN_ROOT}/eval/${candidate}.raw.predictions.jsonl"
  v2_summary="${RUN_ROOT}/eval/${candidate}.qwen35-v2.json"
  v2_predictions="${RUN_ROOT}/eval/${candidate}.qwen35-v2.predictions.jsonl"
  log="${RUN_ROOT}/logs/${candidate}-eval.log"
  if [[ -f "${v2_summary}" && -f "${v2_predictions}" ]]; then
    echo "verified existing normalized evaluation: ${candidate}"
    return
  fi
  if [[ ! -f "${raw_summary}" && ! -f "${raw_predictions}" ]]; then
    local -a adapter_args=() sample_args=()
    if [[ "${candidate}" != base && "${candidate}" != base-probe ]]; then
      local training_summary
      training_summary="$(training_summary_for_candidate "${candidate}")"
      require_file "${training_summary}"
      adapter_args=(--adapter "${checkpoint}" --training-summary "${training_summary}")
    fi
    if [[ "${candidate}" == base-probe || "${candidate}" == probe-* ]]; then
      sample_args=(--sample-limit 32)
    fi
    CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${EVALUATOR}" \
      --model "${BASE_MODEL}" \
      "${adapter_args[@]}" \
      --experiment-manifest "${RUN_ROOT}/DAY20-MANIFEST.json" \
      --eval-manifest "${EVAL_MANIFEST}" \
      --scorers "${SCORERS}" \
      --candidate "${candidate}" \
      --checkpoint "${checkpoint}" \
      --output "${raw_summary}" \
      "${sample_args[@]}" \
      2>&1 | tee -a "${log}"
  else
    require_file "${raw_summary}"
    require_file "${raw_predictions}"
  fi
  require_absent "${v2_summary}"
  require_absent "${v2_predictions}"
  "${PYTHON_BIN}" "${RESCORER}" \
    --experiment-manifest "${RUN_ROOT}/DAY20-MANIFEST.json" \
    --manifest "${EVAL_MANIFEST}" \
    --scorers "${SCORERS}" \
    --predictions "${raw_predictions}" \
    --eval-summary "${raw_summary}" \
    --output "${v2_predictions}" \
    --summary-output "${v2_summary}" \
    2>&1 | tee -a "${log}"
}

load_new_e2b_credential() {
  require_mode_0600_root "${E2B_CREDENTIAL_FILE}"
  require_mode_0600_root "${E2B_CREDENTIAL_ATTESTATION}"
  local actual_sha key_line nonempty_lines
  actual_sha="$(sha256sum -- "${E2B_CREDENTIAL_FILE}" | cut -d' ' -f1)"
  "${PYTHON_BIN}" - "${E2B_CREDENTIAL_ATTESTATION}" "${actual_sha}" <<'PY'
import hashlib, json, sys
path, credential_sha = sys.argv[1:]
value = json.load(open(path, encoding='utf-8'))
expected = hashlib.sha256(json.dumps(
    {key: item for key, item in value.items() if key != 'attestation_sha256'},
    ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
if (
    value.get('schema_version') != 1
    or value.get('domain') != 'day20.e2b_credential_attestation'
    or value.get('status') != 'rotated'
    or value.get('credential_file_sha256') != credential_sha
    or not isinstance(value.get('created_at_utc'), str)
    or not value['created_at_utc']
    or value.get('attestation_sha256') != expected
):
    raise SystemExit('Day 20 E2B credential attestation is invalid')
PY
  nonempty_lines="$(sed '/^[[:space:]]*$/d' "${E2B_CREDENTIAL_FILE}" | wc -l)"
  [[ "${nonempty_lines}" == "1" ]] || {
    echo "Day 20 E2B credential file must contain exactly one assignment" >&2
    exit 1
  }
  key_line="$(sed '/^[[:space:]]*$/d' "${E2B_CREDENTIAL_FILE}")"
  [[ "${key_line}" == E2B_API_KEY=* && "${key_line#E2B_API_KEY=}" != "" ]] || {
    echo "Day 20 E2B credential file has an invalid format" >&2
    exit 1
  }
  unset E2B_API_KEY
  E2B_API_KEY="${key_line#E2B_API_KEY=}"
  export E2B_API_KEY
}

require_e2b_runtime() {
  require_file "${SANDBOX_PYTHON}"
  "${SANDBOX_PYTHON}" - <<'PY'
import importlib.metadata
version = importlib.metadata.version('e2b')
if version != '2.37.0':
    raise SystemExit(f'expected pinned e2b 2.37.0, found {version}')
PY
}

sandbox_candidate() {
  local candidate="$1"
  require_candidate "${candidate}"
  load_run_root
  local predictions="${RUN_ROOT}/eval/${candidate}.qwen35-v2.predictions.jsonl"
  local summary="${RUN_ROOT}/eval/${candidate}.qwen35-v2.json"
  local output="${RUN_ROOT}/eval/${candidate}-code-e2b-qwen35-v2.jsonl"
  local summary_output="${RUN_ROOT}/eval/${candidate}-code-e2b-qwen35-v2-summary.json"
  require_file "${predictions}"
  require_file "${summary}"
  if [[ -f "${output}" && -f "${summary_output}" ]]; then
    echo "verified existing E2B evaluation: ${candidate}"
    return
  fi
  require_absent "${output}"
  require_absent "${summary_output}"
  require_file "${FROZEN_E2B_SCORER}"
  require_file "${SANDBOX_CONFIG}"
  require_file "${HUMANEVAL_SOURCE}"
  require_e2b_runtime
  load_new_e2b_credential
  local -a smoke=()
  local smoke_marker="${RUN_ROOT}/evidence/e2b-live-smoke-attested"
  if [[ ! -f "${smoke_marker}" ]]; then
    smoke=(--live-smoke)
  fi
  "${SANDBOX_PYTHON}" "${CODE_SCORER}" \
    --experiment-manifest "${RUN_ROOT}/DAY20-MANIFEST.json" \
    --manifest "${EVAL_MANIFEST}" \
    --predictions "${predictions}" \
    --eval-summary "${summary}" \
    --frozen-e2b-scorer "${FROZEN_E2B_SCORER}" \
    --sandbox-config "${SANDBOX_CONFIG}" \
    --humaneval-source "${HUMANEVAL_SOURCE}" \
    --credential-attestation "${E2B_CREDENTIAL_ATTESTATION}" \
    --output "${output}" \
    --summary-output "${summary_output}" \
    "${smoke[@]}" \
    2>&1 | tee -a "${RUN_ROOT}/logs/${candidate}-e2b.log"
  if [[ ${#smoke[@]} -gt 0 ]]; then
    require_absent "${smoke_marker}"
    printf '%s\n' "candidate=${candidate}" > "${smoke_marker}"
  fi
  unset E2B_API_KEY
}

run_probe() {
  local learning_rate
  learning_rate="$("${PYTHON_BIN}" -c 'import sys; sys.path.insert(0, sys.argv[1]); from day20_train_plugin import canonical_lr; print(canonical_lr(sys.argv[2]))' "${HERE}" "$1")"
  train_probe "${learning_rate}"
  evaluate_candidate "probe-${learning_rate}"
  sandbox_candidate "probe-${learning_rate}"
}

select_probes() {
  load_run_root
  local output="${RUN_ROOT}/PROBE-SELECTION.json"
  if [[ -f "${output}" ]]; then
    "${PYTHON_BIN}" "${TRAIN_PLUGIN}" verify-selection \
      --manifest "${RUN_ROOT}/DAY20-MANIFEST.json" \
      --selection "${output}" >/dev/null
    echo "verified existing probe selection: $(json_field "${output}" selected_lr)"
    return
  fi
  local -a args=()
  local learning_rate candidate
  for learning_rate in "${PROBE_LRS[@]}"; do
    candidate="probe-${learning_rate}"
    args+=(--probe "${learning_rate}" \
      "${RUN_ROOT}/eval/${candidate}.qwen35-v2.json" \
      "${RUN_ROOT}/eval/${candidate}-code-e2b-qwen35-v2-summary.json")
  done
  "${PYTHON_BIN}" "${TRAIN_PLUGIN}" select-probe \
    --day20-manifest "${RUN_ROOT}/DAY20-MANIFEST.json" \
    --eval-manifest "${EVAL_MANIFEST}" \
    --base-summary "${RUN_ROOT}/eval/base-probe.qwen35-v2.json" \
    --base-e2b-summary "${RUN_ROOT}/eval/base-probe-code-e2b-qwen35-v2-summary.json" \
    "${args[@]}" \
    --output "${output}"
}

probe_all() {
  load_run_root
  evaluate_candidate base-probe
  sandbox_candidate base-probe
  local learning_rate
  for learning_rate in "${PROBE_LRS[@]}"; do
    run_probe "${learning_rate}"
  done
  select_probes
}

train_main() {
  load_run_root
  local selection="${RUN_ROOT}/PROBE-SELECTION.json"
  require_file "${selection}"
  "${PYTHON_BIN}" "${TRAIN_PLUGIN}" verify-selection \
    --manifest "${RUN_ROOT}/DAY20-MANIFEST.json" \
    --selection "${selection}" >/dev/null
  local learning_rate
  learning_rate="$(json_field "${selection}" selected_lr)"
  local resolved_config="${RUN_ROOT}/configs/main-selected.json"
  if [[ ! -f "${resolved_config}" ]]; then
    "${PYTHON_BIN}" "${TRAIN_PLUGIN}" resolve-main-config \
      --manifest "${RUN_ROOT}/DAY20-MANIFEST.json" \
      --selection "${selection}" \
      --output "${resolved_config}"
  fi
  run_swift_training main "${learning_rate}"
}

finalize_run() {
  load_run_root
  if [[ ! -f "${RUN_ROOT}/DAY20-RESULTS.json" ]]; then
    "${PYTHON_BIN}" "${FINALIZER}" --run-root "${RUN_ROOT}"
  fi
  if [[ ! -f "${RUN_ROOT}/DAY20-PASS.json" ]]; then
    if [[ "$(json_field "${RUN_ROOT}/DAY20-RESULTS.json" status)" == \
      "no_eligible_day20_lora_anchor" ]]; then
      echo "Day 20 has no eligible LoRA anchor; Base remains active"
      return
    fi
    echo "Day 20 finalization is incomplete: PASS is missing for eligible results" >&2
    exit 1
  fi
  local verification winner verified_base checkpoint export_dir staging_dir
  verification="$("${PYTHON_BIN}" "${TRAIN_PLUGIN}" verify-winner \
    --run-root "${RUN_ROOT}")"
  winner="$("${PYTHON_BIN}" -c \
    'import json,sys; print(json.loads(sys.argv[1])["candidate"])' \
    "${verification}")"
  checkpoint="$("${PYTHON_BIN}" -c \
    'import json,sys; print(json.loads(sys.argv[1])["checkpoint"])' \
    "${verification}")"
  verified_base="$("${PYTHON_BIN}" -c \
    'import json,sys; print(json.loads(sys.argv[1])["base_model"])' \
    "${verification}")"
  case "${winner}" in early|mid|final) ;; *) echo "unsafe Day 20 winner: ${winner}" >&2; exit 1 ;; esac
  [[ "${verified_base}" == "${BASE_MODEL}" ]] || {
    echo "Day 20 winner evidence points to a different Base model" >&2
    exit 1
  }
  export_dir="${RUN_ROOT}/exports/${winner}-merged"
  if [[ -d "${export_dir}" ]]; then
    "${PYTHON_BIN}" "${TRAIN_PLUGIN}" verify-export \
      --run-root "${RUN_ROOT}" --export-dir "${export_dir}" \
      --candidate "${winner}" >/dev/null
    echo "verified existing merged winner export: ${export_dir}"
    return
  fi
  require_absent "${export_dir}"
  staging_dir="${RUN_ROOT}/exports/.${winner}-merged-attempt-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  require_absent "${staging_dir}"
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" -m swift.cli.export \
    --model "${BASE_MODEL}" \
    --model_type qwen3_5 \
    --adapters "${checkpoint}" \
    --merge_lora true \
    --torch_dtype bfloat16 \
    --output_dir "${staging_dir}" \
    --exist_ok false \
    2>&1 | tee -a "${RUN_ROOT}/logs/${winner}-merge.log"
  require_dir "${staging_dir}"
  "${PYTHON_BIN}" "${TRAIN_PLUGIN}" seal-export \
    --run-root "${RUN_ROOT}" --export-dir "${staging_dir}" \
    --published-dir "${export_dir}" --candidate "${winner}" >/dev/null
  require_absent "${export_dir}"
  [[ "$(dirname -- "${staging_dir}")" == "$(dirname -- "${export_dir}")" ]] || {
    echo "merged export staging and publication directories differ" >&2
    exit 1
  }
  mv -T -- "${staging_dir}" "${export_dir}"
  "${PYTHON_BIN}" "${TRAIN_PLUGIN}" verify-export \
    --run-root "${RUN_ROOT}" --export-dir "${export_dir}" \
    --candidate "${winner}" >/dev/null
  echo "published verified merged winner export: ${export_dir}"
}

case "${1:-}" in
  init)
    [[ $# -eq 1 ]] || { echo "usage: $0 init" >&2; exit 1; }
    acquire_init_lock
    init_run
    ;;
  prepare)
    [[ $# -eq 1 ]] || { echo "usage: $0 prepare" >&2; exit 1; }
    acquire_run_operation_locks prepare
    prepare_data
    ;;
  probe)
    [[ $# -eq 2 ]] || { echo "usage: $0 probe LR" >&2; exit 1; }
    acquire_run_operation_locks probe
    run_probe "$2"
    ;;
  probe-all)
    [[ $# -eq 1 ]] || { echo "usage: $0 probe-all" >&2; exit 1; }
    acquire_run_operation_locks probe-all
    probe_all
    ;;
  train-main)
    [[ $# -eq 1 ]] || { echo "usage: $0 train-main" >&2; exit 1; }
    acquire_run_operation_locks train-main
    train_main
    ;;
  eval)
    [[ $# -eq 2 ]] || { echo "usage: $0 eval CANDIDATE" >&2; exit 1; }
    acquire_run_operation_locks eval
    evaluate_candidate "$2"
    ;;
  sandbox)
    [[ $# -eq 2 ]] || { echo "usage: $0 sandbox CANDIDATE" >&2; exit 1; }
    acquire_run_operation_locks sandbox
    sandbox_candidate "$2"
    ;;
  finalize)
    [[ $# -eq 1 ]] || { echo "usage: $0 finalize" >&2; exit 1; }
    acquire_run_operation_locks finalize
    finalize_run
    ;;
  *)
    echo "usage: $0 {init|prepare|probe LR|probe-all|train-main|eval CANDIDATE|sandbox CANDIDATE|finalize}" >&2
    exit 1
    ;;
esac
