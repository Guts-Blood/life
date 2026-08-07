#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-}"
TARGET="${2:-25_percent}"
if [[ ! "${RUN_ID}" =~ ^[C-L]$ ]]; then
  echo "usage: $0 <C-L> [25_percent|60_percent|100_percent]" >&2
  exit 2
fi
DAY12_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DAY12_RECOVERY_RUN_ID="${RUN_ID}"
exec "${DAY12_DIR}/run_day12_c_cloud_dev_eval.sh" "${TARGET}"
