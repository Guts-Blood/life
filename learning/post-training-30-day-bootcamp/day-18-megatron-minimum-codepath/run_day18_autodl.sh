#!/usr/bin/env bash
set -euo pipefail

DAY18_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTCAMP_ROOT="$(cd "${DAY18_DIR}/.." && pwd)"
SOURCE_ROOT="${DAY18_SOURCE_ROOT:-$(cd "${BOOTCAMP_ROOT}/.." && pwd)}"
SWIFT_SOURCE="${DAY18_SWIFT_SOURCE:-${SOURCE_ROOT}/ms-swift}"
DATA_ROOT="${DAY18_DATA_ROOT:-/root/autodl-tmp/qwen35-v2}"
STATE_ROOT="${DAY18_STATE_ROOT:-${DATA_ROOT}/state}"
CURRENT_RUN_FILE="${STATE_ROOT}/current-run-root"
VENV="${DAY18_VENV:-${DATA_ROOT}/venv}"
CACHE_ROOT="${DAY18_CACHE_ROOT:-${DATA_ROOT}/cache}"
BUILD_TMP="${DAY18_BUILD_TMP:-${DATA_ROOT}/tmp}"
PYTHON_BIN="${VENV}/bin/python"
MEGATRON_BIN="${VENV}/bin/megatron"
MEGATRON_SFT="${DAY18_DIR}/day18_megatron_sft.py"
MODEL_REVISION="1001bb4d826a52d1f399e183466143f4da7b741b"
MODEL_PATH="${DAY18_MODEL_PATH:-/root/autodl-tmp/Qwen--Qwen3.5-4B-Base/${MODEL_REVISION}}"
FIXTURE="${BOOTCAMP_ROOT}/artifacts/data/day18-qwen35-golden-text-2.jsonl"
CONTRACT="${BOOTCAMP_ROOT}/artifacts/configs/day18-qwen35-megatron/run-contract.json"
THRESHOLDS="${BOOTCAMP_ROOT}/artifacts/configs/day18-qwen35-megatron/parity-thresholds.json"
SNAPSHOT_VERIFIER="${DAY18_DIR}/verify_qwen35_snapshot.py"
PREFLIGHT="${DAY18_DIR}/preflight_day18.py"
TEXT_EXPORT="${DAY18_DIR}/day18_text_export.py"
HF_REFERENCE="${DAY18_DIR}/day18_hf_reference.py"
LOG_VERIFIER="${DAY18_DIR}/verify_day18_logs.py"
CHECKPOINT_AUDITOR="${DAY18_DIR}/audit_day18_checkpoint.py"
MTP_COMPARATOR="${DAY18_DIR}/compare_day18_mtp_tensors.py"
SCOPE_COMPARATOR="${DAY18_DIR}/compare_day18_parameter_scopes.py"
FINALIZER="${DAY18_DIR}/finalize_day18_run.py"
CODEPATH_MANIFEST="${DAY18_DIR}/build_day18_codepath_manifest.py"
PROBE_PLUGIN="${DAY18_DIR}/day18_probe_plugin.py"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export USE_HF=1
export USE_MCORE_GDN=1
export DAY18_BOOT_IMAGE_REFERENCE="${DAY18_BOOT_IMAGE_REFERENCE:-autodl-observed-runtime:ubuntu22.04.4-py3.12.3-torch2.5.1+cu124-cuda12.4}"
export CUDA_HOME="${DAY18_CUDA_HOME:-${VENV}}"
export CUDA_PATH="${CUDA_HOME}"
export NVTE_CUDA_INCLUDE_DIR="${DAY18_NVTE_CUDA_INCLUDE_DIR:-${CUDA_HOME}/targets/x86_64-linux/include}"
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

run_root() {
  if [[ -n "${DAY18_RUN_ROOT:-}" ]]; then
    printf '%s\n' "${DAY18_RUN_ROOT}"
  elif [[ -f "${CURRENT_RUN_FILE}" ]]; then
    tr -d '\n' < "${CURRENT_RUN_FILE}"
  else
    echo "no run root; execute '$0 init' first" >&2
    return 1
  fi
}

require_run() {
  RUN_ROOT="$(run_root)"
  if [[ ! -f "${RUN_ROOT}/.day18-run-root" ]]; then
    echo "invalid or uninitialized run root: ${RUN_ROOT}" >&2
    exit 1
  fi
  C0_DIR="${RUN_ROOT}/c0-mcore-tp1"
  C1_HF_ROUNDTRIP_DIR="${RUN_ROOT}/c1-hf-roundtrip"
  C2_DIR="${RUN_ROOT}/c2-dp2"
  C2_RELOAD_DIR="${RUN_ROOT}/c2-dp2-loadcheck"
  C3_DIR="${RUN_ROOT}/c3-tp2"
  C4_DIR="${RUN_ROOT}/c4-resume-tp2"
  C4_HF_DIR="${RUN_ROOT}/c4-hf"
  C5_DIR="${RUN_ROOT}/c5-tiny-overfit-tp2"
  C5_HF_DIR="${RUN_ROOT}/c5-hf"
  LOG_DIR="${RUN_ROOT}/logs"
  EVIDENCE_DIR="${RUN_ROOT}/evidence"
  GATE_DIR="${EVIDENCE_DIR}/gates"
  mkdir -p "${LOG_DIR}" "${EVIDENCE_DIR}" "${GATE_DIR}"
}

require_gate() {
  local gate="$1"
  if [[ ! -f "${GATE_DIR}/${gate}.pass.json" ]]; then
    echo "prerequisite gate has not passed: ${gate}" >&2
    exit 1
  fi
}

mark_gate() {
  local gate="$1"
  printf '{"schema_version":1,"gate":"%s","status":"pass","completed_at_utc":"%s"}\n' \
    "${gate}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${GATE_DIR}/${gate}.pass.json"
}

require_venv() {
  if [[ ! -x "${PYTHON_BIN}" || ! -x "${MEGATRON_BIN}" || ! -f "${MEGATRON_SFT}" ]]; then
    echo "missing Day18 environment; execute '$0 bootstrap' first" >&2
    exit 1
  fi
}

require_disk_headroom() {
  local path="${1:-/root/autodl-tmp}"
  local used_percent
  used_percent="$(df -Pk "${path}" | awk 'NR==2 {gsub(/%/, "", $5); print $5}')"
  if (( used_percent > 85 )); then
    echo "disk hard stop: ${path} is ${used_percent}% used; at least 15% must remain free" >&2
    exit 1
  fi
}

require_initial_free_space() {
  local path="${1:-/root/autodl-tmp}"
  local available_kib
  available_kib="$(df -Pk "${path}" | awk 'NR==2 {print $4}')"
  if (( available_kib < 280 * 1024 * 1024 )); then
    echo "disk hard stop: Day18 requires at least 280 GiB free after upload/extract; got $((available_kib / 1024 / 1024)) GiB" >&2
    exit 1
  fi
}

require_host_envelope() {
  local disk_total_kib memory_total_kib gpu_count gpu_index gpu_name memory_mib
  local -a gpu_memory_mib gpu_names
  disk_total_kib="$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $2}')"
  if (( disk_total_kib < 341796875 )); then
    echo "host hard stop: data filesystem is below 350 GB total" >&2
    exit 1
  fi
  memory_total_kib="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
  if (( memory_total_kib < 62500000 )); then
    echo "host hard stop: system RAM is below 64 GB" >&2
    exit 1
  fi
  mapfile -t gpu_memory_mib < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
  gpu_count="${#gpu_memory_mib[@]}"
  if (( gpu_count != 2 || ${#gpu_names[@]} != 2 )); then
    echo "host hard stop: expected exactly 2 physical GPUs, got ${gpu_count}" >&2
    exit 1
  fi
  for gpu_index in 0 1; do
    gpu_name="${gpu_names[${gpu_index}]}"
    if [[ "${gpu_name}" != *H800* && "${gpu_name}" != *H100* ]]; then
      echo "host hard stop: GPU ${gpu_index} must be H800 or H100, got ${gpu_name}" >&2
      exit 1
    fi
    memory_mib="${gpu_memory_mib[${gpu_index}]}"
    memory_mib="${memory_mib//[[:space:]]/}"
    if (( memory_mib < 80000 )); then
      echo "host hard stop: GPU ${gpu_index} must expose at least 80000 MiB, got ${memory_mib}" >&2
      exit 1
    fi
  done
}

require_absent() {
  local path="$1"
  if [[ -e "${path}" ]]; then
    echo "refusing to overwrite existing gate output: ${path}" >&2
    exit 1
  fi
}

init_run() {
  mkdir -p "${STATE_ROOT}" "${PIP_CACHE_DIR}" "${TMPDIR}"
  if [[ -f "${CURRENT_RUN_FILE}" ]]; then
    echo "current run already exists: $(tr -d '\n' < "${CURRENT_RUN_FILE}")" >&2
    echo "archive it and remove only ${CURRENT_RUN_FILE} before initializing another run" >&2
    exit 1
  fi
  local root="${DAY18_RUN_ROOT:-/root/autodl-tmp/runs/day18-qwen35-$(date -u +%Y%m%dT%H%M%SZ)}"
  if [[ -e "${root}" ]]; then
    echo "refusing to reuse existing run root: ${root}" >&2
    exit 1
  fi
  mkdir -p "${root}"
  printf 'model_revision=%s\ncreated_at_utc=%s\n' "${MODEL_REVISION}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${root}/.day18-run-root"
  printf '%s\n' "${root}" > "${CURRENT_RUN_FILE}"
  echo "DAY18_RUN_ROOT=${root}"
}

host_inventory() {
  require_run
  require_disk_headroom
  require_initial_free_space
  require_host_envelope
  {
    date -u +%Y-%m-%dT%H:%M:%SZ
    uname -a
    cat /etc/os-release 2>/dev/null || true
    /root/miniconda3/bin/python --version
    nvidia-smi
    nvidia-smi topo -m
    nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,memory.free --format=csv,noheader
    df -h / /tmp /root/autodl-tmp
    free -h
    printf 'DAY18_BOOT_IMAGE_REFERENCE=%s\n' "${DAY18_BOOT_IMAGE_REFERENCE}"
    readlink -f /usr/local/cuda 2>/dev/null || true
    if [[ -x /root/miniconda3/bin/python ]]; then
      /root/miniconda3/bin/python -c 'import platform, torch; print(f"boot_python={platform.python_version()} boot_torch={torch.__version__} boot_torch_cuda={torch.version.cuda}")'
    fi
    env | sort | grep -E '^(CUDA|NCCL|NVIDIA|PYTORCH|USE_MCORE_GDN|DAY18_BOOT_IMAGE_REFERENCE)' || true
  } 2>&1 | tee "${LOG_DIR}/host-inventory.log"
  cp "${CONTRACT}" "${EVIDENCE_DIR}/run-contract.preregistered.json"
  cp "${THRESHOLDS}" "${EVIDENCE_DIR}/parity-thresholds.preregistered.json"
  mark_gate host-inventory
}

bootstrap() {
  require_run
  require_gate host-inventory
  require_disk_headroom
  mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}"
  if [[ ! -d "${SWIFT_SOURCE}" ]]; then
    echo "missing bundled ms-swift source: ${SWIFT_SOURCE}" >&2
    exit 1
  fi
  local swift_commit
  if [[ -d "${SWIFT_SOURCE}/.git" ]]; then
    swift_commit="$(git -C "${SWIFT_SOURCE}" rev-parse HEAD)"
  elif [[ -f "${SWIFT_SOURCE}/SOURCE-COMMIT" ]]; then
    swift_commit="$(tr -d '\n' < "${SWIFT_SOURCE}/SOURCE-COMMIT")"
  else
    echo "ms-swift source has neither .git nor SOURCE-COMMIT provenance" >&2
    exit 1
  fi
  if [[ "${swift_commit}" != "565a1ad586a21d24b23931c52d2c62b49c39bee8" ]]; then
    echo "ms-swift commit mismatch: ${swift_commit}" >&2
    exit 1
  fi
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "missing isolated Day18 conda prefix: ${VENV}" >&2
    echo "prepare the pinned CUDA 12.6 / torch 2.10 environment before bootstrap; system-site-packages is forbidden" >&2
    exit 1
  fi
  local runtime_prefix
  runtime_prefix="$("${PYTHON_BIN}" -c 'import sys; print(sys.prefix)')"
  if [[ "${runtime_prefix}" != "${VENV}" ]]; then
    echo "Day18 Python prefix mismatch: expected ${VENV}, got ${runtime_prefix}" >&2
    exit 1
  fi
  "${PYTHON_BIN}" -m pip install --disable-pip-version-check --no-deps -e "${SWIFT_SOURCE}"
  "${PYTHON_BIN}" -m pip check
  "${PYTHON_BIN}" -m pip freeze --all | sort > "${EVIDENCE_DIR}/pip-freeze.txt"
  "${PYTHON_BIN}" -m pip inspect > "${EVIDENCE_DIR}/pip-inspect.json"
  printf '%s\n' "${swift_commit}" > "${EVIDENCE_DIR}/ms-swift-commit.txt"
  "${PYTHON_BIN}" --version 2>&1 | tee "${LOG_DIR}/python-version.log"
  "${PYTHON_BIN}" -c 'import swift; print(swift.__version__)' | tee "${LOG_DIR}/ms-swift-version.log"
  mark_gate bootstrap
}

model_preflight() {
  require_run
  require_gate bootstrap
  require_venv
  require_disk_headroom
  "${PYTHON_BIN}" "${SNAPSHOT_VERIFIER}" \
    --model-dir "${MODEL_PATH}" \
    --hash-all \
    --output "${EVIDENCE_DIR}/model-snapshot-verified.json" \
    2>&1 | tee "${LOG_DIR}/model-snapshot-preflight.log"
  mark_gate model-preflight
}

runtime_preflight() {
  require_run
  require_gate model-preflight
  require_venv
  "${PYTHON_BIN}" "${PREFLIGHT}" runtime \
    --model-dir "${MODEL_PATH}" \
    --output "${EVIDENCE_DIR}/runtime-preflight.json" \
    2>&1 | tee "${LOG_DIR}/runtime-preflight.log"
  mark_gate runtime-preflight
}

topology_preflight() {
  require_run
  require_gate runtime-preflight
  require_venv
  mkdir -p "${EVIDENCE_DIR}/topology"
  CUDA_VISIBLE_DEVICES=0,1 "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone --nproc_per_node=2 \
    "${PREFLIGHT}" topology \
    --output-dir "${EVIDENCE_DIR}/topology" \
    --expected-world-size 2 \
    2>&1 | tee "${LOG_DIR}/topology-preflight.log"
  "${PYTHON_BIN}" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"] == "pass", p' \
    "${EVIDENCE_DIR}/topology/summary.json"
  mark_gate topology-preflight
}

c0_c1() {
  require_run
  require_gate topology-preflight
  require_venv
  require_disk_headroom
  require_absent "${C0_DIR}"
  require_absent "${C1_HF_ROUNDTRIP_DIR}"
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${HF_REFERENCE}" \
    --model "${MODEL_PATH}" \
    --fixture "${FIXTURE}" \
    --max-length 512 \
    --output "${EVIDENCE_DIR}/c1-hf-reference.json" \
    2>&1 | tee "${LOG_DIR}/c1-hf-reference.log"
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone --nproc_per_node=1 "${TEXT_EXPORT}" \
    --fixture "${FIXTURE}" \
    --model "${MODEL_PATH}" \
    --model_type qwen3_5 \
    --bridge_backend mcore-bridge \
    --to_mcore true \
    --output_dir "${C0_DIR}" \
    --exist_ok false \
    --torch_dtype bfloat16 \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --context_parallel_size 1 \
    --language_model_only false \
    --linear_decoupled_in_proj false \
    --mtp_num_layers 1 \
    --mtp_loss_scaling_factor 0.1 \
    --padding_free false \
    --attention_backend unfused \
    --test_convert_precision true \
    --test_convert_dtype float32 \
    2>&1 | tee "${LOG_DIR}/c0-c1-conversion.log"
  "${PYTHON_BIN}" "${LOG_VERIFIER}" conversion \
    --log "${LOG_DIR}/c0-c1-conversion.log" \
    --thresholds "${THRESHOLDS}" \
    --threshold-key conversion_text_logits \
    --expected-records 2 \
    --output "${EVIDENCE_DIR}/c1-conversion-parity.json"
  "${PYTHON_BIN}" "${LOG_VERIFIER}" mcore-loss \
    --log "${LOG_DIR}/c0-c1-conversion.log" \
    --hf-reference "${EVIDENCE_DIR}/c1-hf-reference.json" \
    --thresholds "${THRESHOLDS}" \
    --threshold-key hf_reference_vs_c1_mcore_loss \
    --expected-records 2 \
    --output "${EVIDENCE_DIR}/c1-single-rank-loss-parity.json"
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone --nproc_per_node=1 "${TEXT_EXPORT}" \
    --fixture "${FIXTURE}" \
    --mcore_model "${C0_DIR}" \
    --model_type qwen3_5 \
    --bridge_backend mcore-bridge \
    --to_hf true \
    --output_dir "${C1_HF_ROUNDTRIP_DIR}" \
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
    --test_convert_precision false \
    2>&1 | tee "${LOG_DIR}/c1-mtp-roundtrip-export.log"
  "${PYTHON_BIN}" "${MTP_COMPARATOR}" \
    --left "${MODEL_PATH}" --right "${C1_HF_ROUNDTRIP_DIR}" --mode exact \
    --output "${EVIDENCE_DIR}/c1-mtp-roundtrip.json"
  mark_gate c0-c1
}

training_common=()
build_training_common() {
  local no_save_optim="${1:-false}"
  local no_save_rng="${2:-false}"
  training_common=(
    --bridge_backend mcore-bridge
    --model_type qwen3_5
    --dataset "${FIXTURE}"
    --add_version false
    --tuner_type full
    --torch_dtype bfloat16
    --freeze_llm false
    --freeze_vit true
    --freeze_aligner true
    --language_model_only false
    --mtp_num_layers 1
    --mtp_loss_scaling_factor 0.1
    --pipeline_model_parallel_size 1
    --context_parallel_size 1
    --sequence_parallel false
    --global_batch_size 2
    --max_length 512
    --packing false
    --padding_free false
    --enable_thinking false
    --add_non_thinking_prefix true
    --loss_scale default+ignore_empty_think
    --split_dataset_ratio 0
    --dataset_shuffle false
    --train_dataloader_shuffle false
    --strict true
    --dataset_num_proc 1
    --dataloader_num_workers 0
    --seed 42
    --data_seed 42
    --recompute_granularity none
    --attention_backend unfused
    --gradient_accumulation_fusion false
    --cross_entropy_loss_fusion false
    --masked_softmax_fusion false
    --bias_dropout_fusion false
    --bias_activation_fusion false
    --use_distributed_optimizer true
    --lr 1e-5
    --lr_decay_style constant
    --lr_warmup_iters 0
    --weight_decay 0
    --logging_steps 1
    --save_safetensors false
    --async_save false
    --no_save_optim "${no_save_optim}"
    --no_save_rng "${no_save_rng}"
    --external_plugins "${PROBE_PLUGIN}"
    --callbacks day18_evidence
  )
}

c2_dp2() {
  require_run
  require_gate c0-c1
  require_venv
  require_disk_headroom
  require_absent "${C2_DIR}"
  require_absent "${C2_RELOAD_DIR}"
  if [[ ! -d "${C0_DIR}" ]]; then
    echo "C0 is missing: ${C0_DIR}" >&2
    exit 1
  fi
  build_training_common
  DAY18_PLUGIN_EVIDENCE_DIR="${EVIDENCE_DIR}/c2-ranks" \
  DAY18_EXPECT_TP=1 DAY18_EXPECT_DP=2 \
  DAY18_EXPECT_START_ITERATION=0 DAY18_EXPECT_FINAL_ITERATION=3 \
  CUDA_VISIBLE_DEVICES=0,1 "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone --nproc_per_node=2 "${MEGATRON_SFT}" \
    --mcore_model "${C0_DIR}" \
    --output_dir "${C2_DIR}" \
    --tensor_model_parallel_size 1 \
    --micro_batch_size 1 \
    --train_iters 3 \
    --finetune true \
    --no_load_optim true \
    --no_load_rng true \
    --save_steps 3 \
    "${training_common[@]}" \
    2>&1 | tee "${LOG_DIR}/c2-dp2.log"
  nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader > "${EVIDENCE_DIR}/c2-post-step-vram.csv"
  "${PYTHON_BIN}" "${LOG_VERIFIER}" train \
    --log "${LOG_DIR}/c2-dp2.log" \
    --checkpoint "${C2_DIR}/checkpoint-3" \
    --expected-steps 3 \
    --output "${EVIDENCE_DIR}/c2-dp2.json"
  "${PYTHON_BIN}" "${CHECKPOINT_AUDITOR}" \
    --checkpoint "${C2_DIR}/checkpoint-3" --expected-iteration 3 \
    --minimum-bytes 30000000000 --output "${EVIDENCE_DIR}/c2-checkpoint-audit.json"
  complete_c2_verification
}

complete_c2_verification() {
  "${PYTHON_BIN}" "${LOG_VERIFIER}" ranks \
    --evidence-dir "${EVIDENCE_DIR}/c2-ranks" \
    --expected-ranks 2 --expected-tp 1 --expected-dp 2 \
    --expected-start 0 --expected-final 3 --expected-steps 3 \
    --output "${EVIDENCE_DIR}/c2-ranks-summary.json"
  DAY18_PLUGIN_EVIDENCE_DIR="${EVIDENCE_DIR}/c2-loadcheck-ranks" \
  DAY18_EXPECT_TP=1 DAY18_EXPECT_DP=2 \
  DAY18_EXPECT_START_ITERATION=3 DAY18_EXPECT_FINAL_ITERATION=3 \
  CUDA_VISIBLE_DEVICES=0,1 "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone --nproc_per_node=2 "${MEGATRON_SFT}" \
    --mcore_model "${C2_DIR}/checkpoint-3" \
    --output_dir "${C2_RELOAD_DIR}" \
    --tensor_model_parallel_size 1 \
    --micro_batch_size 1 \
    --train_iters 3 \
    --finetune false \
    --no_load_optim false \
    --no_load_rng false \
    --save_steps 999999 \
    "${training_common[@]}" \
    2>&1 | tee "${LOG_DIR}/c2-dp2-loadcheck.log"
  "${PYTHON_BIN}" "${LOG_VERIFIER}" ranks \
    --evidence-dir "${EVIDENCE_DIR}/c2-loadcheck-ranks" \
    --expected-ranks 2 --expected-tp 1 --expected-dp 2 \
    --expected-start 3 --expected-final 3 --expected-steps 0 \
    --output "${EVIDENCE_DIR}/c2-loadcheck-ranks-summary.json"
  "${PYTHON_BIN}" "${LOG_VERIFIER}" compare \
    --left "${EVIDENCE_DIR}/c1-hf-reference.json" --left-field reference_loss \
    --right "${EVIDENCE_DIR}/c2-dp2.json" --right-field first_loss \
    --max-abs 0.1 \
    --output "${EVIDENCE_DIR}/c1-hf-vs-c2-loss.json"
  require_disk_headroom
  mark_gate c2-dp2
}

c2_verify_existing() {
  require_run
  require_gate c0-c1
  require_venv
  require_disk_headroom
  require_absent "${C2_RELOAD_DIR}"
  if [[ ! -d "${C2_DIR}/checkpoint-3" || ! -f "${EVIDENCE_DIR}/c2-dp2.json" ]]; then
    echo "C2 checkpoint or training evidence is missing; run c2-dp2 instead" >&2
    exit 1
  fi
  "${PYTHON_BIN}" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"] == "pass", p' \
    "${EVIDENCE_DIR}/c2-dp2.json"
  "${PYTHON_BIN}" "${CHECKPOINT_AUDITOR}" \
    --checkpoint "${C2_DIR}/checkpoint-3" --expected-iteration 3 \
    --minimum-bytes 30000000000 --output "${EVIDENCE_DIR}/c2-checkpoint-audit.json"
  build_training_common
  complete_c2_verification
}

c3_tp2() {
  require_run
  require_gate c2-dp2
  require_venv
  require_disk_headroom
  require_absent "${C3_DIR}"
  build_training_common
  DAY18_PLUGIN_EVIDENCE_DIR="${EVIDENCE_DIR}/c3-ranks" \
  DAY18_EXPECT_TP=2 DAY18_EXPECT_DP=1 \
  DAY18_EXPECT_START_ITERATION=0 DAY18_EXPECT_FINAL_ITERATION=3 \
  CUDA_VISIBLE_DEVICES=0,1 "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone --nproc_per_node=2 "${MEGATRON_SFT}" \
    --mcore_model "${C0_DIR}" \
    --output_dir "${C3_DIR}" \
    --tensor_model_parallel_size 2 \
    --micro_batch_size 2 \
    --train_iters 3 \
    --finetune true \
    --no_load_optim true \
    --no_load_rng true \
    --save_steps 3 \
    "${training_common[@]}" \
    2>&1 | tee "${LOG_DIR}/c3-tp2.log"
  nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader > "${EVIDENCE_DIR}/c3-post-step-vram.csv"
  "${PYTHON_BIN}" "${LOG_VERIFIER}" train \
    --log "${LOG_DIR}/c3-tp2.log" \
    --checkpoint "${C3_DIR}/checkpoint-3" \
    --expected-steps 3 \
    --output "${EVIDENCE_DIR}/c3-tp2.json"
  "${PYTHON_BIN}" "${CHECKPOINT_AUDITOR}" \
    --checkpoint "${C3_DIR}/checkpoint-3" --expected-iteration 3 \
    --minimum-bytes 30000000000 --output "${EVIDENCE_DIR}/c3-checkpoint-audit.json"
  "${PYTHON_BIN}" "${LOG_VERIFIER}" ranks \
    --evidence-dir "${EVIDENCE_DIR}/c3-ranks" \
    --expected-ranks 2 --expected-tp 2 --expected-dp 1 \
    --expected-start 0 --expected-final 3 --expected-steps 3 \
    --output "${EVIDENCE_DIR}/c3-ranks-summary.json"
  "${PYTHON_BIN}" "${LOG_VERIFIER}" compare \
    --left "${EVIDENCE_DIR}/c2-dp2.json" --left-field first_loss \
    --right "${EVIDENCE_DIR}/c3-tp2.json" --right-field first_loss \
    --max-abs 0.1 \
    --output "${EVIDENCE_DIR}/c2-vs-c3-loss.json"
  require_disk_headroom
  mark_gate c3-tp2
}

c4_resume_export() {
  require_run
  require_gate c3-tp2
  require_venv
  require_disk_headroom
  require_absent "${C4_DIR}"
  require_absent "${C4_HF_DIR}"
  if [[ ! -d "${C3_DIR}/checkpoint-3" ]]; then
    echo "C3 checkpoint is missing" >&2
    exit 1
  fi
  build_training_common
  DAY18_PLUGIN_EVIDENCE_DIR="${EVIDENCE_DIR}/c4-ranks" \
  DAY18_EXPECT_TP=2 DAY18_EXPECT_DP=1 \
  DAY18_EXPECT_START_ITERATION=3 DAY18_EXPECT_FINAL_ITERATION=5 \
  CUDA_VISIBLE_DEVICES=0,1 "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone --nproc_per_node=2 "${MEGATRON_SFT}" \
    --mcore_model "${C3_DIR}/checkpoint-3" \
    --output_dir "${C4_DIR}" \
    --tensor_model_parallel_size 2 \
    --micro_batch_size 2 \
    --train_iters 5 \
    --finetune false \
    --no_load_optim false \
    --no_load_rng false \
    --save_steps 5 \
    "${training_common[@]}" \
    2>&1 | tee "${LOG_DIR}/c4-resume-tp2.log"
  "${PYTHON_BIN}" "${LOG_VERIFIER}" train \
    --log "${LOG_DIR}/c4-resume-tp2.log" \
    --checkpoint "${C4_DIR}/checkpoint-5" \
    --expected-steps 2 \
    --output "${EVIDENCE_DIR}/c4-resume.json"
  "${PYTHON_BIN}" "${CHECKPOINT_AUDITOR}" \
    --checkpoint "${C4_DIR}/checkpoint-5" --expected-iteration 5 \
    --minimum-bytes 30000000000 --output "${EVIDENCE_DIR}/c4-checkpoint-audit.json"
  "${PYTHON_BIN}" "${LOG_VERIFIER}" ranks \
    --evidence-dir "${EVIDENCE_DIR}/c4-ranks" \
    --expected-ranks 2 --expected-tp 2 --expected-dp 1 \
    --expected-start 3 --expected-final 5 --expected-steps 2 \
    --output "${EVIDENCE_DIR}/c4-ranks-summary.json"
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone --nproc_per_node=1 "${TEXT_EXPORT}" \
    --fixture "${FIXTURE}" \
    --mcore_model "${C4_DIR}/checkpoint-5" \
    --model_type qwen3_5 \
    --bridge_backend mcore-bridge \
    --to_hf true \
    --output_dir "${C4_HF_DIR}" \
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
    2>&1 | tee "${LOG_DIR}/c4-export-parity.log"
  "${PYTHON_BIN}" "${LOG_VERIFIER}" conversion \
    --log "${LOG_DIR}/c4-export-parity.log" \
    --thresholds "${THRESHOLDS}" \
    --threshold-key c4_export_reload \
    --expected-records 2 \
    --output "${EVIDENCE_DIR}/c4-export-parity.json"
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${HF_REFERENCE}" \
    --model "${C4_HF_DIR}" \
    --fixture "${FIXTURE}" \
    --max-length 512 \
    --output "${EVIDENCE_DIR}/c4-hf-reference.json" \
    2>&1 | tee "${LOG_DIR}/c4-hf-reload.log"
  "${PYTHON_BIN}" "${MTP_COMPARATOR}" \
    --left "${MODEL_PATH}" --right "${C4_HF_DIR}" --mode changed \
    --output "${EVIDENCE_DIR}/c4-mtp-trained-export.json"
  require_disk_headroom
  mark_gate c4-resume-export
}

c5_tiny_overfit() {
  require_run
  require_gate c4-resume-export
  require_venv
  require_disk_headroom
  require_absent "${C5_DIR}"
  require_absent "${C5_HF_DIR}"
  if [[ ! -d "${C0_DIR}" || ! -d "${C1_HF_ROUNDTRIP_DIR}" || ! -f "${EVIDENCE_DIR}/c1-hf-reference.json" ]]; then
    echo "C0 model, C1 HF round-trip, or Base teacher-forced evidence is missing" >&2
    exit 1
  fi
  build_training_common true true
  DAY18_PLUGIN_EVIDENCE_DIR="${EVIDENCE_DIR}/c5-ranks" \
  DAY18_EXPECT_TP=2 DAY18_EXPECT_DP=1 \
  DAY18_EXPECT_START_ITERATION=0 DAY18_EXPECT_FINAL_ITERATION=150 \
  CUDA_VISIBLE_DEVICES=0,1 "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone --nproc_per_node=2 "${MEGATRON_SFT}" \
    --mcore_model "${C0_DIR}" \
    --output_dir "${C5_DIR}" \
    --tensor_model_parallel_size 2 \
    --micro_batch_size 2 \
    --train_iters 150 \
    --finetune true \
    --no_load_optim true \
    --no_load_rng true \
    --save_steps 150 \
    "${training_common[@]}" \
    2>&1 | tee "${LOG_DIR}/c5-tiny-overfit-tp2.log"
  nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader \
    > "${EVIDENCE_DIR}/c5-post-step-vram.csv"
  "${PYTHON_BIN}" "${CHECKPOINT_AUDITOR}" \
    --checkpoint "${C5_DIR}/checkpoint-150" --expected-iteration 150 \
    --minimum-bytes 7000000000 --checkpoint-kind model-only \
    --output "${EVIDENCE_DIR}/c5-checkpoint-audit.json"
  "${PYTHON_BIN}" "${LOG_VERIFIER}" ranks \
    --evidence-dir "${EVIDENCE_DIR}/c5-ranks" \
    --expected-ranks 2 --expected-tp 2 --expected-dp 1 \
    --expected-start 0 --expected-final 150 --expected-steps 150 \
    --output "${EVIDENCE_DIR}/c5-ranks-summary.json"
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone --nproc_per_node=1 "${TEXT_EXPORT}" \
    --fixture "${FIXTURE}" \
    --mcore_model "${C5_DIR}/checkpoint-150" \
    --model_type qwen3_5 \
    --bridge_backend mcore-bridge \
    --to_hf true \
    --output_dir "${C5_HF_DIR}" \
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
    2>&1 | tee "${LOG_DIR}/c5-export-parity.log"
  "${PYTHON_BIN}" "${LOG_VERIFIER}" conversion \
    --log "${LOG_DIR}/c5-export-parity.log" \
    --thresholds "${THRESHOLDS}" \
    --threshold-key c5_export_reload \
    --expected-records 2 \
    --output "${EVIDENCE_DIR}/c5-export-parity.json"
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${HF_REFERENCE}" \
    --model "${C5_HF_DIR}" \
    --fixture "${FIXTURE}" \
    --max-length 512 \
    --output "${EVIDENCE_DIR}/c5-hf-reference.json" \
    2>&1 | tee "${LOG_DIR}/c5-hf-reload.log"
  "${PYTHON_BIN}" "${SCOPE_COMPARATOR}" \
    --base "${C1_HF_ROUNDTRIP_DIR}" --trained "${C5_HF_DIR}" \
    --output "${EVIDENCE_DIR}/c5-parameter-scopes.json"
  "${PYTHON_BIN}" "${LOG_VERIFIER}" tiny-overfit \
    --base "${EVIDENCE_DIR}/c1-hf-reference.json" \
    --final "${EVIDENCE_DIR}/c5-hf-reference.json" \
    --log "${LOG_DIR}/c5-tiny-overfit-tp2.log" \
    --thresholds "${THRESHOLDS}" \
    --threshold-key c5_tiny_overfit \
    --expected-steps 150 \
    --output "${EVIDENCE_DIR}/c5-tiny-overfit.json"
  require_disk_headroom
  mark_gate c5-tiny-overfit
}

record_problem() {
  require_run
  require_venv
  "${PYTHON_BIN}" "${FINALIZER}" problem --run-root "${RUN_ROOT}" "$@"
}

finalize_run() {
  require_run
  require_gate c5-tiny-overfit
  require_venv
  "${PYTHON_BIN}" "${CODEPATH_MANIFEST}" \
    --run-root "${RUN_ROOT}" \
    --output "${EVIDENCE_DIR}/codepath-runtime-evidence.json"
  "${PYTHON_BIN}" "${FINALIZER}" finalize --run-root "${RUN_ROOT}"
  mark_gate day18-final
}

sync_evidence() {
  require_run
  require_disk_headroom
  local sync_root="/root/autodl-tmp/qwen35-v2/sync"
  local name
  local -a optional_members=()
  name="$(basename "${RUN_ROOT}")-evidence.tar"
  mkdir -p "${sync_root}"
  if [[ -f "${RUN_ROOT}/DAY18-PASS.json" ]]; then
    optional_members+=(DAY18-PASS.json)
  fi
  tar -C "${RUN_ROOT}" -cf "${sync_root}/${name}" .day18-run-root logs evidence "${optional_members[@]}"
  sha256sum "${sync_root}/${name}" > "${sync_root}/${name}.sha256"
  echo "evidence_bundle=${sync_root}/${name}"
  echo "evidence_sha256=${sync_root}/${name}.sha256"
}

case "${1:-}" in
  init) init_run ;;
  host-inventory) host_inventory ;;
  bootstrap) bootstrap ;;
  model-preflight) model_preflight ;;
  runtime-preflight) runtime_preflight ;;
  topology-preflight) topology_preflight ;;
  c0-c1) c0_c1 ;;
  c2-dp2) c2_dp2 ;;
  c2-verify-existing) c2_verify_existing ;;
  c3-tp2) c3_tp2 ;;
  c4-resume-export) c4_resume_export ;;
  c5-tiny-overfit) c5_tiny_overfit ;;
  record-problem) shift; record_problem "$@" ;;
  finalize) finalize_run ;;
  sync-evidence) sync_evidence ;;
  *)
    echo "usage: $0 {init|host-inventory|bootstrap|model-preflight|runtime-preflight|topology-preflight|c0-c1|c2-dp2|c2-verify-existing|c3-tp2|c4-resume-export|c5-tiny-overfit|record-problem|finalize|sync-evidence}" >&2
    exit 2
    ;;
esac
