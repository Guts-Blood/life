#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${DAY18_REMOTE_HOST:?set DAY18_REMOTE_HOST, for example root@connect.example}"
SSH_PORT="${DAY18_SSH_PORT:?set DAY18_SSH_PORT}"
CONTROL_BUNDLE="${DAY18_CONTROL_BUNDLE:?set DAY18_CONTROL_BUNDLE to the verified control tar}"
MODEL_BUNDLE="${DAY18_MODEL_BUNDLE:?set DAY18_MODEL_BUNDLE to the verified model tar}"
REVISION="1001bb4d826a52d1f399e183466143f4da7b741b"
REMOTE_INCOMING="/root/autodl-tmp/qwen35-v2/incoming"
REMOTE_SOURCE="/root/autodl-tmp/qwen35-v2/source"
REMOTE_MODEL="/root/autodl-tmp/models/Qwen--Qwen3.5-4B-Base/${REVISION}"
SSH=(ssh -p "${SSH_PORT}" -o ServerAliveInterval=30 -o ServerAliveCountMax=10)
SCP=(scp -P "${SSH_PORT}" -o ServerAliveInterval=30 -o ServerAliveCountMax=10)

for path in "${CONTROL_BUNDLE}" "${CONTROL_BUNDLE}.sha256" "${MODEL_BUNDLE}" "${MODEL_BUNDLE}.sha256"; do
  if [[ ! -f "${path}" ]]; then
    echo "missing upload input: ${path}" >&2
    exit 1
  fi
done

"${SSH[@]}" "${REMOTE_HOST}" "mkdir -p '${REMOTE_INCOMING}'"

upload_one() {
  local path="$1"
  if command -v rsync >/dev/null 2>&1 && "${SSH[@]}" "${REMOTE_HOST}" "command -v rsync >/dev/null"; then
    rsync --partial --progress -e "ssh -p ${SSH_PORT} -o ServerAliveInterval=30 -o ServerAliveCountMax=10" \
      "${path}" "${REMOTE_HOST}:${REMOTE_INCOMING}/"
  else
    "${SCP[@]}" "${path}" "${REMOTE_HOST}:${REMOTE_INCOMING}/"
  fi
}

upload_one "${CONTROL_BUNDLE}"
upload_one "${CONTROL_BUNDLE}.sha256"
upload_one "${MODEL_BUNDLE}"
upload_one "${MODEL_BUNDLE}.sha256"

CONTROL_NAME="$(basename "${CONTROL_BUNDLE}")"
MODEL_NAME="$(basename "${MODEL_BUNDLE}")"
"${SSH[@]}" "${REMOTE_HOST}" "cd '${REMOTE_INCOMING}' && sha256sum -c '${CONTROL_NAME}.sha256' && sha256sum -c '${MODEL_NAME}.sha256'"
"${SSH[@]}" "${REMOTE_HOST}" \
  "for target in '${REMOTE_SOURCE}' '${REMOTE_MODEL}'; do if [ -d \"\${target}\" ] && find \"\${target}\" -mindepth 1 -print -quit | grep -q .; then echo \"refusing non-empty extraction target: \${target}\" >&2; exit 1; fi; done; mkdir -p '${REMOTE_SOURCE}' '${REMOTE_MODEL}'"
"${SSH[@]}" "${REMOTE_HOST}" "tar -C '${REMOTE_SOURCE}' -xf '${REMOTE_INCOMING}/${CONTROL_NAME}'"
"${SSH[@]}" "${REMOTE_HOST}" "cd '${REMOTE_SOURCE}' && sha256sum -c MANIFEST.sha256"
"${SSH[@]}" "${REMOTE_HOST}" "tar -C '${REMOTE_MODEL}' -xf '${REMOTE_INCOMING}/${MODEL_NAME}'"
"${SSH[@]}" "${REMOTE_HOST}" "cd '${REMOTE_MODEL}' && sha256sum -c MANIFEST.sha256"
"${SSH[@]}" "${REMOTE_HOST}" \
  "python3 '${REMOTE_SOURCE}/post-training-30-day-bootcamp/day-18-megatron-minimum-codepath/verify_qwen35_snapshot.py' --model-dir '${REMOTE_MODEL}'"
echo "remote_source=${REMOTE_SOURCE}"
echo "remote_model=${REMOTE_MODEL}"
