#!/usr/bin/env bash
set -euo pipefail

DAY11_BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DAY11_MODEL_PATH="${DAY11_BUNDLE_ROOT}/models/Qwen3-0.6B-Base"
exec bash \
  "${DAY11_BUNDLE_ROOT}/post-training-30-day-bootcamp/day-11-sft-step-tiny-overfit/run_day11_autodl.sh" \
  all
