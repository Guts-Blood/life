#!/usr/bin/env python3
"""Build the two-prompt, G=4 Day 26 slime compatibility input."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DAY26_DIR = Path(__file__).resolve().parent
BOOTCAMP_ROOT = DAY26_DIR.parent
DAY25_DIR = BOOTCAMP_ROOT / "day-25-grpo-small-model-lab"
if str(DAY25_DIR) not in sys.path:
    sys.path.insert(0, str(DAY25_DIR))

import day25_contract as contract  # noqa: E402


SOURCE = BOOTCAMP_ROOT / "artifacts/data/day25-qwen35-coding-grpo-train.jsonl"
DEFAULT_OUTPUT = BOOTCAMP_ROOT / "artifacts/data/day26-slime-qwen35-runtime-prompts.jsonl"

REWARD_FIELDS = (
    "task_id",
    "task_family_id",
    "prompt",
    "problem",
    "code_prefix",
    "entry_point",
    "test_setup_code",
    "reward_tests",
    "tests_manifest_sha256",
    "task_manifest_sha256",
    "reward_payload_sha256",
    "source_record_sha256",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = contract.load_jsonl(SOURCE)[:2]
    contract.require(len(rows) == 2, "Day 26 requires exactly two source prompts")
    output = []
    for row in rows:
        family = str(row["task_family_id"])
        metadata = {name: row[name] for name in REWARD_FIELDS}
        metadata.update(
            {
                "prompt_id": f"day26:prompt:{family}",
                "source_name": "day25.frozen_train32",
                "source_row_sha256": row["row_sha256"],
                "modality": "text",
            }
        )
        output.append({"input": row["messages"], "metadata": metadata})

    contract.write_atomic(
        args.output,
        contract.jsonl_bytes(output),
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "output": contract.relative_to_bootcamp(args.output),
                "rows": len(output),
                "group_size": contract.GROUP_SIZE,
                "effective_trajectories": len(output) * contract.GROUP_SIZE,
                "file_sha256": contract.file_sha256(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
