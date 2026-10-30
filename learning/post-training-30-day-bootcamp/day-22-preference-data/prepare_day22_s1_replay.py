#!/usr/bin/env python3
"""Validate S1 rollout rows and emit the exact retained E2B request cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import formal_s1_pair_labeler as labeler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--rollout-contract", type=Path, required=True)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--requests-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        requests_output, summary_output = labeler._preflight_output_paths(
            [args.requests_output, args.summary_output],
            overwrite=args.overwrite,
        )
        result = labeler.prepare_candidate_replay_requests(
            labeler.load_jsonl(args.rollouts.resolve()),
            rollout_contract=labeler.load_json(args.rollout_contract.resolve()),
            seed_manifest=labeler.load_json(args.seed_manifest.resolve()),
        )
        labeler._write_atomic(
            requests_output,
            labeler._jsonl_bytes(result["requests"]),
            overwrite=args.overwrite,
        )
        labeler._write_atomic(
            summary_output,
            labeler._json_bytes(result["summary"]),
            overwrite=args.overwrite,
        )
    except labeler.FormalS1LabelerError as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "status": "ready_for_e2b",
                "requests": len(result["requests"]),
                "requests_output": str(requests_output),
                "requests_file_sha256": labeler.file_sha256(requests_output),
                "summary_output": str(summary_output),
                "summary_file_sha256": labeler.file_sha256(summary_output),
                "summary_sha256": result["summary"]["summary_sha256"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
