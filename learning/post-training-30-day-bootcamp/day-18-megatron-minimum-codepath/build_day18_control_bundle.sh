#!/usr/bin/env bash
set -euo pipefail
export COPYFILE_DISABLE=1

DAY18_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTCAMP_ROOT="$(cd "${DAY18_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${BOOTCAMP_ROOT}/../.." && pwd)"
SWIFT_SOURCE="${WORKSPACE_ROOT}/vendor/ms-swift"
SWIFT_COMMIT="565a1ad586a21d24b23931c52d2c62b49c39bee8"
MODEL_REVISION="1001bb4d826a52d1f399e183466143f4da7b741b"
OUTPUT_ROOT="${DAY18_BUNDLE_OUTPUT:-${WORKSPACE_ROOT}/tmp/day18-qwen35-ready}"
BUNDLE="${OUTPUT_ROOT}/day18-control-${MODEL_REVISION}.tar"

if [[ "$(git -C "${SWIFT_SOURCE}" rev-parse HEAD)" != "${SWIFT_COMMIT}" ]]; then
  echo "vendored ms-swift commit drift" >&2
  exit 1
fi
if [[ -e "${BUNDLE}" || -e "${BUNDLE}.sha256" ]]; then
  echo "refusing to overwrite existing bundle: ${BUNDLE}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
STAGE="$(mktemp -d "${OUTPUT_ROOT}/stage.XXXXXX")"
cleanup() {
  rm -rf "${STAGE}"
}
trap cleanup EXIT

mkdir -p \
  "${STAGE}/post-training-30-day-bootcamp/artifacts/configs" \
  "${STAGE}/post-training-30-day-bootcamp/artifacts/data" \
  "${STAGE}/post-training-30-day-bootcamp/artifacts/checkpoints" \
  "${STAGE}/post-training-30-day-bootcamp/artifacts/reports" \
  "${STAGE}/post-training-30-day-bootcamp/environments" \
  "${STAGE}/ms-swift"

rsync -a --exclude '__pycache__' --exclude '*.pyc' \
  "${DAY18_DIR}/" \
  "${STAGE}/post-training-30-day-bootcamp/day-18-megatron-minimum-codepath/"
rsync -a \
  "${BOOTCAMP_ROOT}/artifacts/configs/day18-qwen35-megatron/" \
  "${STAGE}/post-training-30-day-bootcamp/artifacts/configs/day18-qwen35-megatron/"
cp "${BOOTCAMP_ROOT}/artifacts/data/day18-qwen35-golden-text-2.jsonl" \
  "${STAGE}/post-training-30-day-bootcamp/artifacts/data/"
cp "${BOOTCAMP_ROOT}/artifacts/reports/day18-qwen35-megatron-compatibility.md" \
  "${STAGE}/post-training-30-day-bootcamp/artifacts/reports/"
cp "${BOOTCAMP_ROOT}/artifacts/configs/qwen35-4b-migration-contract.json" \
  "${STAGE}/post-training-30-day-bootcamp/artifacts/configs/"
if [[ -f "${BOOTCAMP_ROOT}/artifacts/checkpoints/qwen35-4b-base-s0.json" ]]; then
  cp "${BOOTCAMP_ROOT}/artifacts/checkpoints/qwen35-4b-base-s0.json" \
    "${STAGE}/post-training-30-day-bootcamp/artifacts/checkpoints/"
fi
cp "${BOOTCAMP_ROOT}/QWEN35-4B-MIGRATION-PLAN.md" "${STAGE}/post-training-30-day-bootcamp/"
cp "${BOOTCAMP_ROOT}/AUTODL.md" "${STAGE}/post-training-30-day-bootcamp/"
cp "${BOOTCAMP_ROOT}/environments/README.md" "${STAGE}/post-training-30-day-bootcamp/environments/"

git -C "${SWIFT_SOURCE}" archive "${SWIFT_COMMIT}" | tar -x -C "${STAGE}/ms-swift"
printf '%s\n' "${SWIFT_COMMIT}" > "${STAGE}/ms-swift/SOURCE-COMMIT"
printf '{"schema_version":1,"model_revision":"%s","ms_swift_commit":"%s"}\n' \
  "${MODEL_REVISION}" "${SWIFT_COMMIT}" > "${STAGE}/BUNDLE-METADATA.json"

(
  cd "${STAGE}"
  find . -type f ! -name MANIFEST.sha256 -print | LC_ALL=C sort | while IFS= read -r path; do
    shasum -a 256 "${path}"
  done > MANIFEST.sha256
)
tar -C "${STAGE}" -cf "${BUNDLE}" .
(
  cd "${OUTPUT_ROOT}"
  shasum -a 256 "$(basename "${BUNDLE}")" > "$(basename "${BUNDLE}").sha256"
)
echo "control_bundle=${BUNDLE}"
echo "control_sha256=${BUNDLE}.sha256"
