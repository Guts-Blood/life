#!/usr/bin/env bash
set -euo pipefail

BOOTCAMP_ROOT="/root/autodl-tmp/post-training-30-day-bootcamp"
HERE="${BOOTCAMP_ROOT}/day-19-qwen35-sft-comparison"
DATA_ROOT="/root/autodl-tmp/qwen35-v2"
VENV="${DATA_ROOT}/venv"
CACHE_ROOT="${DATA_ROOT}/cache"
BUILD_TMP="${DATA_ROOT}/tmp"
PYTHON_BIN="${VENV}/bin/python"
STATE_FILE="/root/autodl-tmp/qwen35-v2/state/current-day19-run-root"
DAY19_DIAGNOSTIC_RUN="/root/autodl-tmp/runs/day19-qwen35-20260808T100142Z"
DAY18_RUN="/root/autodl-tmp/runs/day18-qwen35-20260808T073811Z"
MODEL_PATH="/root/autodl-tmp/Qwen--Qwen3.5-4B-Base/1001bb4d826a52d1f399e183466143f4da7b741b"
MCORE_START="${DAY18_RUN}/c0-mcore-tp1"
DAY18_DIR="${BOOTCAMP_ROOT}/day-18-megatron-minimum-codepath"
MEGATRON_SFT="${DAY18_DIR}/day18_megatron_sft.py"
PROBE_PLUGIN="${DAY18_DIR}/day18_probe_plugin.py"
CHECKPOINT_AUDITOR="${DAY18_DIR}/audit_day18_checkpoint.py"
TEXT_EXPORT="${DAY18_DIR}/day18_text_export.py"
EXPORT_FIXTURE="${BOOTCAMP_ROOT}/artifacts/data/day18-qwen35-golden-text-2.jsonl"
PREPARER="${HERE}/prepare_day19.py"
EVALUATOR="${HERE}/evaluate_day19.py"
FINALIZER="${HERE}/finalize_day19.py"
CODE_SCORER="${HERE}/score_day19_code_e2b.py"
V2_RESCORER="${HERE}/rescore_day19_qwen35_v2.py"
V2_CODE_SCORER="${HERE}/score_day19_code_e2b_v2.py"
EVAL_MANIFEST="${BOOTCAMP_ROOT}/artifacts/eval/day10-frozen-eval-manifest.json"
SCORERS="${BOOTCAMP_ROOT}/day-10-frozen-eval-baseline/day10_scorers.py"
FROZEN_E2B_SCORER="${BOOTCAMP_ROOT}/day-10-frozen-eval-baseline/score_day10_code_e2b.py"
SANDBOX_CONFIG="${BOOTCAMP_ROOT}/day-10-frozen-eval-baseline/day10_e2b_sandbox_config.json"
SANDBOX_PYTHON="${DAY19_SANDBOX_PYTHON:-/root/autodl-tmp/envs/day12-e2b/bin/python}"
E2B_CREDENTIAL_FILE="${DAY19_E2B_CREDENTIAL_FILE:-/root/autodl-tmp/secrets/day12-e2b.env}"
RECIPES=(baseline-a baseline-b best-e)
EVAL_RECIPES=(baseline-a baseline-b best-e untouched-c0)

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export USE_HF=1
export USE_MCORE_GDN=1
export CUDA_HOME="${DAY19_CUDA_HOME:-${VENV}}"
export CUDA_PATH="${CUDA_HOME}"
export NVTE_CUDA_INCLUDE_DIR="${DAY19_NVTE_CUDA_INCLUDE_DIR:-${CUDA_HOME}/targets/x86_64-linux/include}"
export PATH="${VENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ -x "${PYTHON_BIN}" ]]; then
  CUDNN_ROOT="$("${PYTHON_BIN}" -c 'import nvidia.cudnn; print(nvidia.cudnn.__path__[0])' 2>/dev/null || true)"
  if [[ -n "${CUDNN_ROOT}" && -d "${CUDNN_ROOT}/lib" ]]; then
    export CUDNN_PATH="${CUDNN_ROOT}"
    export CUDNN_HOME="${CUDNN_ROOT}"
    export LD_LIBRARY_PATH="${CUDNN_ROOT}/lib:${LD_LIBRARY_PATH}"
  fi
fi
if [[ -x "${VENV}/bin/x86_64-conda-linux-gnu-gcc" && -x "${VENV}/bin/x86_64-conda-linux-gnu-g++" ]]; then
  export CC="${VENV}/bin/x86_64-conda-linux-gnu-gcc"
  export CXX="${VENV}/bin/x86_64-conda-linux-gnu-g++"
fi
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export TMPDIR="${BUILD_TMP}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "required file is missing: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "required directory is missing: $1" >&2
    exit 1
  fi
}

load_run_root() {
  require_file "${STATE_FILE}"
  RUN_ROOT="$(<"${STATE_FILE}")"
  if [[ "${RUN_ROOT}" != /root/autodl-tmp/runs/day19-qwen35-* ]]; then
    echo "unsafe Day 19 run root in state: ${RUN_ROOT}" >&2
    exit 1
  fi
  require_file "${RUN_ROOT}/.day19-run-root"
}

load_diagnostic_run_root() {
  require_dir "${DAY19_DIAGNOSTIC_RUN}"
  RUN_ROOT="$(realpath -e -- "${DAY19_DIAGNOSTIC_RUN}")"
  if [[ "${RUN_ROOT}" != "/root/autodl-tmp/runs/day19-qwen35-20260808T100142Z" ]]; then
    echo "canonical Day 19 diagnostic root drifted: ${RUN_ROOT}" >&2
    exit 1
  fi
  require_file "${RUN_ROOT}/.day19-run-root"
  require_file "${RUN_ROOT}/DAY19-PASS.json"
  require_file "${RUN_ROOT}/DAY19-RESULTS.json"
}

require_disk_headroom() {
  "${PYTHON_BIN}" - <<'PY'
import shutil
free = shutil.disk_usage('/root/autodl-tmp').free
if free < 80 * 1024**3:
    raise SystemExit(f'Day 19 requires at least 80 GiB free, found {free / 1024**3:.2f} GiB')
print(f'free_disk_gib={free / 1024**3:.2f}')
PY
}

require_recipe() {
  case "$1" in
    baseline-a|baseline-b|best-e) ;;
    *) echo "unknown recipe: $1" >&2; exit 1 ;;
  esac
}

require_eval_recipe() {
  case "$1" in
    baseline-a|baseline-b|best-e|untouched-c0) ;;
    *) echo "unknown evaluation recipe: $1" >&2; exit 1 ;;
  esac
}

init_run() {
  if [[ "$(id -un)" != root ]]; then
    echo "Day 19 requires remote user root" >&2
    exit 1
  fi
  require_file "${DAY18_RUN}/DAY18-PASS.json"
  require_dir "${MCORE_START}"
  require_dir "${MODEL_PATH}"
  require_dir "${BOOTCAMP_ROOT}"
  require_file "${PYTHON_BIN}"
  require_disk_headroom
  if [[ -f "${STATE_FILE}" ]]; then
    existing="$(<"${STATE_FILE}")"
    if [[ -n "${existing}" && -d "${existing}" && ! -f "${existing}/DAY19-PASS.json" ]]; then
      echo "unfinished Day 19 run already exists: ${existing}" >&2
      exit 1
    fi
  fi
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  RUN_ROOT="/root/autodl-tmp/runs/day19-qwen35-${timestamp}"
  if [[ -e "${RUN_ROOT}" ]]; then
    echo "refusing to reuse run root: ${RUN_ROOT}" >&2
    exit 1
  fi
  mkdir "${RUN_ROOT}"
  touch "${RUN_ROOT}/.day19-run-root"
  mkdir -p "${RUN_ROOT}"/{configs,data,data-manifests,eval,evidence,checkpoints,logs,tmp}
  printf '%s\n' "${RUN_ROOT}" > "${STATE_FILE}"
  RUN_ROOT="${RUN_ROOT}" "${PYTHON_BIN}" - <<'PY'
import json, os, platform, shutil, socket, subprocess
from datetime import datetime, timezone
from pathlib import Path
root = Path(os.environ['RUN_ROOT'])
payload = {
    'schema_version': 1,
    'domain': 'day19.host_inventory',
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'hostname': socket.gethostname(),
    'user': subprocess.check_output(['id', '-un'], text=True).strip(),
    'platform': platform.platform(),
    'free_disk_bytes': shutil.disk_usage('/root/autodl-tmp').free,
    'nvidia_smi': subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu',
        '--format=csv,noheader'
    ], text=True).splitlines(),
}
(root / 'evidence/host-inventory.json').write_text(
    json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
)
PY
  echo "${RUN_ROOT}"
}

prepare_data() {
  load_run_root
  require_disk_headroom
  if [[ -f "${RUN_ROOT}/DAY19-MANIFEST.json" ]]; then
    echo "Day 19 preparation already exists: ${RUN_ROOT}/DAY19-MANIFEST.json" >&2
    exit 1
  fi
  "${PYTHON_BIN}" "${PREPARER}" \
    --run-root "${RUN_ROOT}" \
    --bootcamp-root "${BOOTCAMP_ROOT}" \
    --model "${MODEL_PATH}" \
    --day18-run "${DAY18_RUN}" \
    --target-tokens 64608 \
    --max-length 2304 \
    > "${RUN_ROOT}/logs/prepare.log" 2>&1
  tail -n 1 "${RUN_ROOT}/logs/prepare.log"
}

config_value() {
  local config="$1"
  local expression="$2"
  "${PYTHON_BIN}" -c "import json; d=json.load(open('${config}')); print(${expression})"
}

train_recipe() {
  local recipe="$1"
  require_recipe "${recipe}"
  load_run_root
  require_file "${RUN_ROOT}/DAY19-MANIFEST.json"
  require_disk_headroom
  local config="${RUN_ROOT}/configs/${recipe}.json"
  local dataset="${RUN_ROOT}/data/${recipe}.jsonl"
  local output_dir="${RUN_ROOT}/checkpoints/${recipe}"
  local log="${RUN_ROOT}/logs/${recipe}-train.log"
  local ranks="${RUN_ROOT}/evidence/${recipe}-ranks"
  local audit="${RUN_ROOT}/evidence/${recipe}-checkpoint-audit.json"
  require_file "${config}"
  require_file "${dataset}"
  if [[ -e "${output_dir}" || -e "${ranks}" || -e "${audit}" ]]; then
    echo "refusing to overwrite training artifacts for ${recipe}" >&2
    exit 1
  fi
  local train_iters
  train_iters="$(config_value "${config}" "d['training']['train_iters']")"
  DAY18_PLUGIN_EVIDENCE_DIR="${ranks}" \
  DAY18_EXPECT_TP=2 DAY18_EXPECT_DP=1 \
  DAY18_EXPECT_START_ITERATION=0 DAY18_EXPECT_FINAL_ITERATION="${train_iters}" \
  CUDA_VISIBLE_DEVICES=0,1 "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone --nproc_per_node=2 "${MEGATRON_SFT}" \
    --mcore_model "${MCORE_START}" \
    --output_dir "${output_dir}" \
    --bridge_backend mcore-bridge \
    --model_type qwen3_5 \
    --dataset "${dataset}" \
    --add_version false \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --language_model_only false \
    --mtp_num_layers 1 \
    --mtp_loss_scaling_factor 0.1 \
    --tensor_model_parallel_size 2 \
    --pipeline_model_parallel_size 1 \
    --context_parallel_size 1 \
    --sequence_parallel false \
    --micro_batch_size 1 \
    --global_batch_size 1 \
    --train_iters "${train_iters}" \
    --max_length 2304 \
    --packing false \
    --padding_free false \
    --enable_thinking false \
    --add_non_thinking_prefix true \
    --loss_scale default+ignore_empty_think \
    --split_dataset_ratio 0 \
    --dataset_shuffle false \
    --train_dataloader_shuffle false \
    --strict true \
    --dataset_num_proc 1 \
    --dataloader_num_workers 0 \
    --seed 20260807 \
    --data_seed 20260807 \
    --recompute_granularity none \
    --attention_backend unfused \
    --gradient_accumulation_fusion false \
    --cross_entropy_loss_fusion false \
    --masked_softmax_fusion false \
    --bias_dropout_fusion false \
    --bias_activation_fusion false \
    --use_distributed_optimizer true \
    --lr 1e-5 \
    --lr_decay_style constant \
    --lr_warmup_fraction 0.05 \
    --weight_decay 0.1 \
    --logging_steps 1 \
    --save_steps "${train_iters}" \
    --save_safetensors false \
    --async_save false \
    --no_save_optim true \
    --no_save_rng true \
    --finetune true \
    --no_load_optim true \
    --no_load_rng true \
    --external_plugins "${PROBE_PLUGIN}" \
    --callbacks day18_evidence \
    2>&1 | tee "${log}"
  "${PYTHON_BIN}" "${CHECKPOINT_AUDITOR}" \
    --checkpoint "${output_dir}/checkpoint-${train_iters}" \
    --expected-iteration "${train_iters}" \
    --minimum-bytes 7000000000 \
    --checkpoint-kind model-only \
    --output "${audit}"
  require_disk_headroom
}

train_all() {
  local recipe
  for recipe in "${RECIPES[@]}"; do
    train_recipe "${recipe}"
  done
}

record_export_cleanup() {
  local recipe="$1"
  local temp_export="$2"
  local output="${RUN_ROOT}/eval/${recipe}-export-validation.json"
  RECIPE="${recipe}" TEMP_EXPORT="${temp_export}" RUN_ROOT="${RUN_ROOT}" \
    "${PYTHON_BIN}" - <<'PY'
import hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

root = Path(os.environ['RUN_ROOT'])
recipe = os.environ['RECIPE']
temp = Path(os.environ['TEMP_EXPORT'])
if temp.exists():
    raise SystemExit(f'temporary export still exists: {temp}')
summary = root / 'eval' / f'{recipe}.json'
audit = root / 'evidence' / f'{recipe}-checkpoint-audit.json'
payload = {
    'schema_version': 1,
    'domain': 'day19.temporary_export_validation',
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'recipe': recipe,
    'eval_summary': str(summary),
    'eval_summary_file_sha256': digest(summary),
    'checkpoint_audit': str(audit),
    'checkpoint_audit_file_sha256': digest(audit),
    'temporary_export_path': str(temp),
    'temporary_export_deleted': True,
    'recoverability': 'deterministic export from the retained model-only checkpoint',
}
(root / 'eval' / f'{recipe}-export-validation.json').write_text(
    json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
)
PY
}

evaluate_recipe() {
  local recipe="$1"
  require_recipe "${recipe}"
  load_run_root
  require_disk_headroom
  local config="${RUN_ROOT}/configs/${recipe}.json"
  local train_iters
  train_iters="$(config_value "${config}" "d['training']['train_iters']")"
  local checkpoint="${RUN_ROOT}/checkpoints/${recipe}/checkpoint-${train_iters}"
  local temp_export="${RUN_ROOT}/tmp/hf-${recipe}"
  local eval_output="${RUN_ROOT}/eval/${recipe}.json"
  local export_log="${RUN_ROOT}/logs/${recipe}-export.log"
  local eval_log="${RUN_ROOT}/logs/${recipe}-eval.log"
  require_dir "${checkpoint}"
  require_file "${RUN_ROOT}/evidence/${recipe}-checkpoint-audit.json"
  if [[ -e "${temp_export}" || -e "${eval_output}" ]]; then
    echo "refusing to overwrite evaluation artifacts for ${recipe}" >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone --nproc_per_node=1 "${TEXT_EXPORT}" \
    --fixture "${EXPORT_FIXTURE}" \
    --mcore_model "${checkpoint}" \
    --model_type qwen3_5 \
    --bridge_backend mcore-bridge \
    --to_hf true \
    --output_dir "${temp_export}" \
    --exist_ok false \
    --torch_dtype bfloat16 \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --context_parallel_size 1 \
    --language_model_only false \
    --mtp_num_layers 1 \
    --mtp_loss_scaling_factor 0.1 \
    --padding_free false \
    --attention_backend unfused \
    --test_convert_precision true \
    --test_convert_dtype float32 \
    2>&1 | tee "${export_log}"
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${EVALUATOR}" \
    --model "${temp_export}" \
    --eval-manifest "${EVAL_MANIFEST}" \
    --scorers "${SCORERS}" \
    --recipe "${recipe}" \
    --checkpoint "${checkpoint}" \
    --output "${eval_output}" \
    2>&1 | tee -a "${eval_log}"
  if [[ "${temp_export}" != "${RUN_ROOT}/tmp/hf-${recipe}" ]]; then
    echo "unsafe temporary export path: ${temp_export}" >&2
    exit 1
  fi
  require_file "${temp_export}/config.json"
  find "${temp_export}" -maxdepth 1 -type f -name '*.safetensors' -print -quit | grep -q .
  rm -rf -- "${temp_export}"
  record_export_cleanup "${recipe}" "${temp_export}"
  require_disk_headroom
}

evaluate_all() {
  local recipe
  for recipe in "${RECIPES[@]}"; do
    evaluate_recipe "${recipe}"
  done
}

evaluate_c0() {
  load_diagnostic_run_root
  require_dir "${MODEL_PATH}"
  local eval_output="${RUN_ROOT}/eval/untouched-c0.json"
  local eval_log="${RUN_ROOT}/logs/untouched-c0-eval.log"
  if [[ -e "${eval_output}" || -e "${eval_output%.json}.predictions.jsonl" ]]; then
    echo "refusing to overwrite untouched C0 evaluation artifacts" >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${EVALUATOR}" \
    --model "${MODEL_PATH}" \
    --eval-manifest "${EVAL_MANIFEST}" \
    --scorers "${SCORERS}" \
    --recipe untouched-c0 \
    --checkpoint "${MODEL_PATH}" \
    --output "${eval_output}" \
    2>&1 | tee -a "${eval_log}"
}

rescore_v2_recipe() {
  local recipe="$1"
  require_eval_recipe "${recipe}"
  load_diagnostic_run_root
  require_file "${V2_RESCORER}"
  require_file "${EVAL_MANIFEST}"
  require_file "${SCORERS}"
  local predictions="${RUN_ROOT}/eval/${recipe}.predictions.jsonl"
  local eval_summary="${RUN_ROOT}/eval/${recipe}.json"
  local output="${RUN_ROOT}/eval/${recipe}.qwen35-v2.predictions.jsonl"
  local summary_output="${RUN_ROOT}/eval/${recipe}.qwen35-v2.json"
  require_file "${predictions}"
  require_file "${eval_summary}"
  "${PYTHON_BIN}" "${V2_RESCORER}" \
    --manifest "${EVAL_MANIFEST}" \
    --scorers "${SCORERS}" \
    --predictions "${predictions}" \
    --eval-summary "${eval_summary}" \
    --output "${output}" \
    --summary-output "${summary_output}" \
    2>&1 | tee -a "${RUN_ROOT}/logs/${recipe}-qwen35-v2-rescore.log"
}

rescore_v2_all() {
  local recipe
  for recipe in "${EVAL_RECIPES[@]}"; do
    rescore_v2_recipe "${recipe}"
  done
}

score_code_all() {
  load_run_root
  require_file "${CODE_SCORER}"
  require_file "${FROZEN_E2B_SCORER}"
  require_file "${SANDBOX_CONFIG}"
  require_file "${EVAL_MANIFEST}"
  require_file "${SANDBOX_PYTHON}"
  local source="${RUN_ROOT}/configs/frozen-inputs/HumanEval.jsonl.gz"
  require_file "${source}"
  if [[ -z "${E2B_API_KEY:-}" ]]; then
    require_file "${E2B_CREDENTIAL_FILE}"
    local credential_mode
    credential_mode="$(stat -c '%a' "${E2B_CREDENTIAL_FILE}")"
    if [[ "${credential_mode}" != "600" ]]; then
      echo "E2B credential file must have mode 0600" >&2
      exit 1
    fi
    set -a
    source "${E2B_CREDENTIAL_FILE}"
    set +a
  fi
  if [[ -z "${E2B_API_KEY:-}" ]]; then
    echo "E2B_API_KEY is not available" >&2
    exit 1
  fi
  local index recipe
  local -a smoke
  for index in "${!RECIPES[@]}"; do
    recipe="${RECIPES[${index}]}"
    smoke=()
    if [[ "${index}" == 0 ]]; then
      smoke=(--live-smoke)
    fi
    "${SANDBOX_PYTHON}" "${CODE_SCORER}" \
      --recipe "${recipe}" \
      --manifest "${EVAL_MANIFEST}" \
      --predictions "${RUN_ROOT}/eval/${recipe}.predictions.jsonl" \
      --eval-summary "${RUN_ROOT}/eval/${recipe}.json" \
      --frozen-e2b-scorer "${FROZEN_E2B_SCORER}" \
      --sandbox-config "${SANDBOX_CONFIG}" \
      --humaneval-source "${source}" \
      --output "${RUN_ROOT}/eval/${recipe}-code-e2b.jsonl" \
      --summary-output "${RUN_ROOT}/eval/${recipe}-code-e2b-summary.json" \
      "${smoke[@]}" \
      2>&1 | tee "${RUN_ROOT}/logs/${recipe}-code-e2b.log"
  done
}

load_v2_e2b_credential() {
  if [[ -z "${E2B_API_KEY:-}" ]]; then
    require_file "${E2B_CREDENTIAL_FILE}"
    local credential_mode
    credential_mode="$(stat -c '%a' "${E2B_CREDENTIAL_FILE}")"
    if [[ "${credential_mode}" != "600" ]]; then
      echo "E2B credential file must have mode 0600" >&2
      exit 1
    fi
    set -a
    source "${E2B_CREDENTIAL_FILE}"
    set +a
  fi
  if [[ -z "${E2B_API_KEY:-}" ]]; then
    echo "E2B_API_KEY is not available" >&2
    exit 1
  fi
}

score_code_v2_recipe() {
  local recipe="$1"
  require_eval_recipe "${recipe}"
  load_diagnostic_run_root
  require_file "${V2_CODE_SCORER}"
  require_file "${FROZEN_E2B_SCORER}"
  require_file "${SANDBOX_CONFIG}"
  require_file "${EVAL_MANIFEST}"
  require_file "${SANDBOX_PYTHON}"
  local source="${RUN_ROOT}/configs/frozen-inputs/HumanEval.jsonl.gz"
  require_file "${source}"
  load_v2_e2b_credential
  require_file "${RUN_ROOT}/eval/${recipe}.qwen35-v2.predictions.jsonl"
  require_file "${RUN_ROOT}/eval/${recipe}.qwen35-v2.json"
  local -a smoke
  smoke=()
  if [[ "${recipe}" == baseline-a ]]; then
    smoke=(--live-smoke)
  fi
  "${SANDBOX_PYTHON}" "${V2_CODE_SCORER}" \
    --manifest "${EVAL_MANIFEST}" \
    --predictions "${RUN_ROOT}/eval/${recipe}.qwen35-v2.predictions.jsonl" \
    --eval-summary "${RUN_ROOT}/eval/${recipe}.qwen35-v2.json" \
    --frozen-e2b-scorer "${FROZEN_E2B_SCORER}" \
    --sandbox-config "${SANDBOX_CONFIG}" \
    --humaneval-source "${source}" \
    --output "${RUN_ROOT}/eval/${recipe}-code-e2b-qwen35-v2.jsonl" \
    --summary-output "${RUN_ROOT}/eval/${recipe}-code-e2b-qwen35-v2-summary.json" \
    "${smoke[@]}" \
    2>&1 | tee -a "${RUN_ROOT}/logs/${recipe}-code-e2b-qwen35-v2.log"
}

score_code_v2_all() {
  local recipe
  for recipe in "${EVAL_RECIPES[@]}"; do
    score_code_v2_recipe "${recipe}"
  done
}

finalize_run() {
  load_run_root
  "${PYTHON_BIN}" "${FINALIZER}" --run-root "${RUN_ROOT}"
}

case "${1:-}" in
  init) init_run ;;
  prepare) prepare_data ;;
  train)
    [[ $# -eq 2 ]] || { echo "usage: $0 train RECIPE" >&2; exit 1; }
    train_recipe "$2"
    ;;
  train-all) train_all ;;
  eval)
    [[ $# -eq 2 ]] || { echo "usage: $0 eval RECIPE" >&2; exit 1; }
    evaluate_recipe "$2"
    ;;
  eval-all) evaluate_all ;;
  eval-c0) evaluate_c0 ;;
  rescore-v2)
    [[ $# -eq 2 ]] || { echo "usage: $0 rescore-v2 RECIPE" >&2; exit 1; }
    rescore_v2_recipe "$2"
    ;;
  rescore-v2-all) rescore_v2_all ;;
  sandbox-all) score_code_all ;;
  sandbox-v2)
    [[ $# -eq 2 ]] || { echo "usage: $0 sandbox-v2 RECIPE" >&2; exit 1; }
    score_code_v2_recipe "$2"
    ;;
  sandbox-v2-all) score_code_v2_all ;;
  finalize) finalize_run ;;
  *)
    echo "usage: $0 {init|prepare|train RECIPE|train-all|eval RECIPE|eval-all|eval-c0|rescore-v2 RECIPE|rescore-v2-all|sandbox-all|sandbox-v2 RECIPE|sandbox-v2-all|finalize}" >&2
    exit 1
    ;;
esac
