#!/usr/bin/env bash
set -euo pipefail
export COPYFILE_DISABLE=1

DAY18_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${DAY18_DIR}/../../.." && pwd)"
REVISION="1001bb4d826a52d1f399e183466143f4da7b741b"
MODEL_DIR="${DAY18_LOCAL_MODEL_PATH:-${WORKSPACE_ROOT}-checkpoints/Qwen--Qwen3.5-4B-Base/${REVISION}}"
OUTPUT_ROOT="${DAY18_MODEL_PACKAGE_OUTPUT:-${WORKSPACE_ROOT}-checkpoints/packages}"
PACKAGE="${OUTPUT_ROOT}/Qwen--Qwen3.5-4B-Base@${REVISION}.tar"
PYTHON_BIN="${DAY18_LOCAL_PYTHON:-python3}"

if [[ -e "${PACKAGE}" || -e "${PACKAGE}.sha256" ]]; then
  echo "refusing to overwrite existing model package: ${PACKAGE}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}"
TEMP_DIR="$(mktemp -d "${OUTPUT_ROOT}/manifest.XXXXXX")"
cleanup() {
  rm -rf "${TEMP_DIR}"
}
trap cleanup EXIT

"${PYTHON_BIN}" "${DAY18_DIR}/verify_qwen35_snapshot.py" \
  --model-dir "${MODEL_DIR}" \
  --hash-all \
  --output "${TEMP_DIR}/snapshot-manifest.json"
"${PYTHON_BIN}" -c \
  'import json,sys; d=json.load(open(sys.argv[1], encoding="utf-8")); [print(v["sha256"], "  ", k, sep="") for k,v in sorted(d["files"].items())]' \
  "${TEMP_DIR}/snapshot-manifest.json" > "${TEMP_DIR}/MANIFEST.sha256"

find "${MODEL_DIR}" -maxdepth 1 -type f -exec basename {} \; | LC_ALL=C sort > "${TEMP_DIR}/file-list.txt"
tar -C "${MODEL_DIR}" -cf "${PACKAGE}" -T "${TEMP_DIR}/file-list.txt"
tar -C "${TEMP_DIR}" -rf "${PACKAGE}" snapshot-manifest.json MANIFEST.sha256
(
  cd "${OUTPUT_ROOT}"
  shasum -a 256 "$(basename "${PACKAGE}")" > "$(basename "${PACKAGE}").sha256"
)
echo "model_package=${PACKAGE}"
echo "model_sha256=${PACKAGE}.sha256"
