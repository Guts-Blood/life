#!/usr/bin/env python3
"""Audit all Day 25 prompts and the real Qwen3.5 GRPO response mask without weights."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import day25_contract as contract


DEFAULT_MODEL_PATH = (
    contract.REPO_ROOT
    / "tmp/qwen35-v2/models/Qwen--Qwen3.5-4B-Base"
    / contract.MODEL_REVISION
)
DEFAULT_TRAIN = contract.ARTIFACTS / "data/day25-qwen35-coding-grpo-train.jsonl"
DEFAULT_EVAL = contract.ARTIFACTS / "data/day25-qwen35-coding-grpo-eval.jsonl"
DEFAULT_ROWS = contract.ARTIFACTS / "eval/day25-qwen35-coding-grpo-processor-audit.jsonl"
DEFAULT_SUMMARY = contract.ARTIFACTS / "eval/day25-qwen35-coding-grpo-processor-audit-summary.json"
PROMPT_TOKEN_CAP = 512
TOTAL_TOKEN_CAP = 1024
SYNTHETIC_RESPONSE = "    return 1"


class Day25ProcessorAuditError(ValueError):
    """The pinned processor/template response boundary drifted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Day25ProcessorAuditError(message)


def _subsequence_positions(values: Sequence[int], needle: Sequence[int]) -> list[int]:
    width = len(needle)
    return [
        index
        for index in range(len(values) - width + 1)
        if list(values[index : index + width]) == list(needle)
    ]


def _encoded_prompt_length(template: Any, row: Mapping[str, Any]) -> int:
    value = template.encode(
        {"messages": list(row["messages"])}, return_length=True
    )
    _require(isinstance(value, Mapping), "template returned a non-object")
    ids = value.get("input_ids")
    _require(isinstance(ids, list) and ids, "prompt input_ids are missing")
    return len(ids)


def build_audit(
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    train_path: Path = DEFAULT_TRAIN,
    eval_path: Path = DEFAULT_EVAL,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import torch
        import swift
        from swift import get_model_processor, get_template
        from swift.rl_core.data import GRPOSample
        from swift.rlhf_trainers.utils import encode_sample, get_response_prefix_ids
    except ImportError as error:
        raise Day25ProcessorAuditError("pinned ms-swift processor runtime is unavailable") from error

    _require(
        Path(swift.__file__).resolve().parent
        == (contract.REPO_ROOT / "vendor/ms-swift/swift").resolve(),
        "processor audit imported ms-swift outside the pinned checkout",
    )
    resolved_model = model_path.resolve()
    _require(resolved_model.is_dir(), f"local Qwen3.5 snapshot is missing: {resolved_model}")
    model, processor = get_model_processor(
        str(resolved_model),
        model_type="qwen3_5",
        load_model=False,
        use_hf=True,
        download_model=False,
    )
    _require(model is None, "load_model=False unexpectedly loaded weights")
    template = get_template(
        processor,
        template_type="qwen3_5",
        max_length=TOTAL_TOKEN_CAP,
        truncation_strategy="raise",
        padding_free=False,
        loss_scale="last_round",
        enable_thinking=False,
        add_non_thinking_prefix=True,
    )
    # SwiftRLHF maps GRPO to template mode "train" (not pairwise "rlhf").
    template.set_mode("train")
    meta = getattr(template, "template_meta", None)
    _require(getattr(meta, "template_type", None) == "qwen3_5", "template type drifted")
    _require(template.enable_thinking is False, "thinking must be disabled")
    _require(template.add_non_thinking_prefix is True, "non-thinking prefix must be enabled")
    _require(template.padding_free is False, "padding_free must remain disabled")
    _require(template.max_length == TOTAL_TOKEN_CAP, "template max_length drifted")

    # Rollout tokenization happens after a newline-bearing assistant prefix.  Bind the
    # first generated token to the standalone four-space token used by the frozen
    # Qwen3.5 code contract instead of re-tokenizing the whole string out of context.
    indent_ids = list(template._tokenize("    "))
    response_ids = indent_ids + list(
        template.tokenizer.encode("return 1", add_special_tokens=False)
    )
    _require(bool(response_ids), "synthetic response did not tokenize")
    prefix_ids = get_response_prefix_ids(template, sample_enable_thinking=False)
    _require(isinstance(prefix_ids, list) and prefix_ids, "non-thinking prefix IDs are missing")
    _require(indent_ids == [257] and response_ids[0] == 257, "four-space response boundary token drifted")

    datasets = {
        "train": contract.load_jsonl(train_path.resolve()),
        "eval": contract.load_jsonl(eval_path.resolve()),
    }
    rows: list[dict[str, Any]] = []
    for split, source_rows in datasets.items():
        for source in source_rows:
            contract.verify_seal(source, "row_sha256", "Day 25 data row")
            prompt_length = _encoded_prompt_length(template, source)
            _require(
                prompt_length <= PROMPT_TOKEN_CAP,
                f"prompt exceeds {PROMPT_TOKEN_CAP} tokens: {source['task_family_id']}",
            )
            sample = GRPOSample.from_row(
                {
                    **source,
                    "messages": list(source["messages"])
                    + [{"role": "assistant", "content": SYNTHETIC_RESPONSE}],
                    "response_token_ids": response_ids,
                    "finish_reason": "stop",
                }
            )
            encoded = encode_sample(sample, template)
            input_ids = list(encoded["input_ids"])
            labels = list(encoded["labels"])
            _require(len(input_ids) == len(labels), "input/label length mismatch")
            positions = _subsequence_positions(input_ids, response_ids)
            supervised_positions = [
                position
                for position in positions
                if labels[position : position + len(response_ids)] == response_ids
            ]
            _require(
                len(supervised_positions) == 1,
                f"response token boundary is ambiguous: {source['task_family_id']}",
            )
            response_start = supervised_positions[0]
            prefix_start = response_start - len(prefix_ids)
            _require(prefix_start >= 0, "response prefix start is negative")
            _require(
                input_ids[prefix_start:response_start] == prefix_ids,
                "non-thinking prefix token IDs drifted",
            )
            _require(
                labels[prefix_start:response_start] == [-100] * len(prefix_ids),
                "non-thinking prefix leaked into the GRPO loss mask",
            )
            _require(labels[response_start] == 257, "first four-space response token is masked")
            _require(len(input_ids) <= TOTAL_TOKEN_CAP, "synthetic total sequence exceeds cap")
            audit_row: dict[str, Any] = {
                "schema_name": "day25.qwen35_grpo_processor_audit_row",
                "schema_version": 1,
                "split": split,
                "task_family_id": source["task_family_id"],
                "source_row_sha256": source["row_sha256"],
                "prompt_token_count": prompt_length,
                "synthetic_total_token_count": len(input_ids),
                "response_token_count": len(response_ids),
                "response_start": response_start,
                "four_space_token_supervised": True,
                "non_thinking_prefix_masked": True,
                "zero_truncation": True,
            }
            audit_row["audit_row_sha256"] = contract.object_sha256(audit_row)
            rows.append(audit_row)

    prompt_lengths = [row["prompt_token_count"] for row in rows]
    total_lengths = [row["synthetic_total_token_count"] for row in rows]
    summary: dict[str, Any] = {
        "schema_name": "day25.qwen35_grpo_processor_audit_summary",
        "schema_version": 1,
        "status": "pass",
        "scope": "real_pinned_qwen35_processor_no_model_weights",
        "records": len(rows),
        "split_counts": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "eval")
        },
        "prompt_token_cap": PROMPT_TOKEN_CAP,
        "training_total_token_cap": TOTAL_TOKEN_CAP,
        "max_prompt_token_count": max(prompt_lengths),
        "min_prompt_token_count": min(prompt_lengths),
        "max_synthetic_total_token_count": max(total_lengths),
        "zero_prompt_truncation": True,
        "all_four_space_boundary_tokens_supervised": True,
        "all_non_thinking_prefix_tokens_masked": True,
        "template": {
            "type": "qwen3_5",
            "mode": "train",
            "enable_thinking": False,
            "add_non_thinking_prefix": True,
            "padding_free": False,
            "loss_scale": "last_round",
        },
        "processor": {
            "model_key": contract.MODEL_KEY,
            "model_revision": contract.MODEL_REVISION,
            "local_snapshot": str(resolved_model),
            "weights_loaded": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "swift_path": str(Path(swift.__file__).resolve()),
            "ms_swift_commit": contract.MS_SWIFT_COMMIT,
        },
        "inputs": {
            "train": {
                "path": contract.relative_to_bootcamp(train_path),
                "file_sha256": contract.file_sha256(train_path),
            },
            "eval": {
                "path": contract.relative_to_bootcamp(eval_path),
                "file_sha256": contract.file_sha256(eval_path),
            },
        },
        "ordered_audit_row_hashes_sha256": contract.object_sha256(
            [row["audit_row_sha256"] for row in rows]
        ),
    }
    summary["summary_sha256"] = contract.object_sha256(summary)
    return rows, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--mode", choices=("build", "check"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows, summary = build_audit(
            model_path=args.model, train_path=args.train, eval_path=args.eval
        )
        payloads = {
            args.rows.resolve(): contract.jsonl_bytes(rows),
            args.summary.resolve(): contract.json_bytes(summary),
        }
        for path, payload in payloads.items():
            if args.mode == "build":
                contract.write_atomic(path, payload, overwrite=False)
            else:
                _require(path.is_file() and path.read_bytes() == payload, f"audit output drifted: {path}")
    except (OSError, contract.Day25ContractError, Day25ProcessorAuditError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "pass",
                "mode": args.mode,
                "records": len(rows),
                "max_prompt_tokens": summary["max_prompt_token_count"],
                "summary_sha256": summary["summary_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
