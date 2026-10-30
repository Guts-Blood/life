#!/usr/bin/env bash
set -euo pipefail

DAY12_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DAY12_RECOVERY_RUN_ID=D
exec "${DAY12_DIR}/run_day12_c_cloud_dev_eval.sh" "${1:-25_percent}"
