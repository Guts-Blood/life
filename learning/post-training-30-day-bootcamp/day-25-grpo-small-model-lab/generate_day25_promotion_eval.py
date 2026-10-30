#!/usr/bin/env python3
"""Generate one matched greedy Day 25 promotion-eval result package."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import day25_contract as contract


DEFAULT_CONTRACT = (
    contract.ARTIFACTS / "configs/day25-qwen35-coding-grpo/promotion-eval-contract.json"
)


class Day25PromotionGenerationError(ValueError):
    """The requested model/suite is outside the frozen matched generation contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Day25PromotionGenerationError(message)


def _directory_manifest(
    path: Path, *, exclude_names: Sequence[str] = ()
) -> dict[str, dict[str, Any]]:
    root = path.expanduser().resolve()
    _require(root.is_dir() and not root.is_symlink(), f"model directory is missing: {root}")
    excluded = set(exclude_names)
    files: dict[str, dict[str, Any]] = {}
    for item in sorted(root.rglob("*")):
        _require(not item.is_symlink(), f"model directory contains a symlink: {item}")
        if item.is_file() and item.name not in excluded:
            files[item.relative_to(root).as_posix()] = {
                "bytes": item.stat().st_size,
                "sha256": contract.file_sha256(item),
            }
    _require(bool(files), f"model directory is empty: {root}")
    return files


def _load_json_sealed(path: Path, field: str, label: str) -> tuple[dict[str, Any], str]:
    value = contract.load_json(path.resolve())
    return value, contract.verify_seal(value, field, label)


def _authorize_confirmation(path: Path | None, eval_hash: str) -> None:
    _require(path is not None, "confirmation generation requires the sealed search decision")
    value, _ = _load_json_sealed(path, "decision_sha256", "search decision")
    _require(
        value.get("eval_contract_sha256") == eval_hash
        and value.get("suite_id") == "search40"
        and value.get("gate_pass") is True
        and value.get("confirmation_authorized") is True,
        "search decision does not authorize confirmation generation",
    )


def generate(
    *,
    eval_contract_path: Path,
    binding_path: Path,
    suite_id: str,
    model_role: str,
    model_path: Path,
    adapter_path: Path | None,
    search_decision_path: Path | None,
) -> dict[str, Any]:
    value, eval_hash = _load_json_sealed(
        eval_contract_path, "eval_contract_sha256", "promotion eval contract"
    )
    binding, binding_hash = _load_json_sealed(
        binding_path, "binding_sha256", "GPU binding"
    )
    _require(binding.get("cpu_contract", {}).get("content_sha256") == value["cpu_contract_sha256"], "GPU binding uses another CPU trust root")
    _require(suite_id in {"search40", "confirmation24"}, "unknown eval suite")
    _require(model_role in {"s1_parent", "grpo_g4_final"}, "unknown model role")
    if suite_id == "confirmation24":
        _authorize_confirmation(search_decision_path, eval_hash)
    else:
        _require(search_decision_path is None, "search40 must not consume a search decision")
    suite = value["suites"][suite_id]
    dataset_path = (contract.BOOTCAMP_ROOT / suite["path"]).resolve()
    rows = contract.load_jsonl(dataset_path)
    _require(contract.file_sha256(dataset_path) == suite["file_sha256"], "eval suite drifted")
    _require([row["task_family_id"] for row in rows] == suite["ordered_family_ids"], "eval order drifted")

    model = model_path.expanduser().resolve()
    expected_model = Path(binding["parent"]["model_path"]).resolve()
    _require(model == expected_model, "base model is not the bound promoted S1 export")
    model_files = _directory_manifest(
        model, exclude_names=("S1-EXPORT-MANIFEST.json",)
    )
    model_artifact_sha = contract.object_sha256(model_files)
    _require(
        model_artifact_sha == binding["parent"]["merged_export_files_sha256"],
        "promoted S1 files drifted after GPU binding",
    )
    baseline = value["checkpoint_policy"]["baseline"]
    if model_role == "s1_parent":
        _require(adapter_path is None, "S1 baseline must not load an adapter")
        identity = {
            "checkpoint_id": baseline["checkpoint_id"],
            "downstream_key": baseline["downstream_key"],
            "artifact_sha256": model_artifact_sha,
        }
        adapter = None
    else:
        _require(adapter_path is not None, "GRPO candidate requires checkpoint-10 adapter")
        adapter = adapter_path.expanduser().resolve()
        expected_adapter = (
            Path(binding["run_root"])
            / "outputs/g4_bounded_short_run/checkpoint-10"
        ).resolve()
        _require(adapter == expected_adapter, "candidate is not the bound G4 checkpoint-10")
        adapter_files = _directory_manifest(adapter)
        identity = {
            "parent_downstream_key": baseline["downstream_key"],
            "stage": "g4_bounded_short_run",
            "checkpoint_step": 10,
            "cpu_contract_sha256": value["cpu_contract_sha256"],
            "binding_sha256": binding_hash,
            "artifact_sha256": contract.object_sha256(adapter_files),
        }

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ["USE_MCORE_GDN"] = "0"
    try:
        import torch
        from swift import get_model_processor, get_template
        from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine
    except ImportError as error:
        raise Day25PromotionGenerationError("pinned GPU inference runtime is unavailable") from error
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "matched eval requires exactly one visible GPU")
    loaded_model, processor = get_model_processor(
        str(model),
        model_type="qwen3_5",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        use_hf=True,
        download_model=False,
    )
    _require(loaded_model.__class__.__name__ == "Qwen3_5ForConditionalGeneration", "wrong Qwen3.5 model class")
    template = get_template(
        processor,
        max_length=1024,
        template_type="qwen3_5",
        padding_free=False,
        enable_thinking=False,
        add_non_thinking_prefix=True,
    )
    meta = getattr(template, "template_meta", None)
    _require(
        getattr(meta, "template_type", None) == "qwen3_5"
        and template.enable_thinking is False
        and template.add_non_thinking_prefix is True,
        "matched eval template drifted",
    )
    engine_kwargs: dict[str, Any] = {"template": template, "max_batch_size": 1}
    if adapter is not None:
        engine_kwargs["adapters"] = [str(adapter)]
    engine = TransformersEngine(loaded_model, **engine_kwargs)
    generation = value["generation_contract"]
    records: list[dict[str, Any]] = []
    for ordinal, task in enumerate(rows):
        response = engine.infer(
            [InferRequest(messages=task["messages"])],
            RequestConfig(
                max_tokens=generation["max_new_tokens"],
                temperature=0,
                num_beams=1,
                repetition_penalty=1.0,
                seed=generation["seed"],
                return_details=True,
            ),
            use_tqdm=False,
        )[0]
        _require(bool(getattr(response, "choices", None)), f"no generation choice: {task['task_family_id']}")
        choice = response.choices[0]
        prompt_ids = getattr(response, "prompt_token_ids", None)
        generated_ids = getattr(choice, "token_ids", None)
        _require(isinstance(prompt_ids, list) and isinstance(generated_ids, list), "token evidence is missing")
        generated_text = template.decode_generate_ids(generated_ids, first_token=False)
        message_content = getattr(getattr(choice, "message", None), "content", "") or ""
        _require(isinstance(generated_text, str) and isinstance(message_content, str), "completion is not text")
        row: dict[str, Any] = {
            "schema_name": "day25.coding_eval_completion",
            "schema_version": 1,
            "ordinal": ordinal,
            "task_family_id": task["task_family_id"],
            "task_row_sha256": task["row_sha256"],
            "message_content": message_content,
            "message_content_sha256": contract.text_sha256(message_content),
            "generated_text": generated_text,
            "generated_text_sha256": contract.text_sha256(generated_text),
            "prompt_token_ids": prompt_ids,
            "prompt_token_ids_sha256": contract.object_sha256(prompt_ids),
            "response_token_ids": generated_ids,
            "response_token_ids_sha256": contract.object_sha256(generated_ids),
            "finish_reason": str(getattr(choice, "finish_reason", None) or "unknown"),
        }
        row["completion_sha256"] = contract.object_sha256(row)
        records.append(row)
        print(json.dumps({"ordinal": ordinal + 1, "records": len(rows), "task_family_id": task["task_family_id"]}, sort_keys=True), flush=True)
    package: dict[str, Any] = {
        "schema_name": "day25.coding_eval_completions",
        "schema_version": 1,
        "eval_contract_sha256": eval_hash,
        "suite_id": suite_id,
        "model_role": model_role,
        "model_identity": identity,
        "generation_contract_sha256": generation["generation_contract_sha256"],
        "gpu_binding_sha256": binding_hash,
        "records": records,
    }
    package["completions_sha256"] = contract.object_sha256(package)
    return package


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--suite", choices=("search40", "confirmation24"), required=True)
    parser.add_argument("--model-role", choices=("s1_parent", "grpo_g4_final"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--search-decision", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = generate(
            eval_contract_path=args.contract,
            binding_path=args.binding,
            suite_id=args.suite,
            model_role=args.model_role,
            model_path=args.model,
            adapter_path=args.adapter,
            search_decision_path=args.search_decision,
        )
        contract.write_atomic(args.output, contract.json_bytes(result), overwrite=False)
    except (OSError, contract.Day25ContractError, Day25PromotionGenerationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "complete", "records": len(result["records"]), "completions_sha256": result["completions_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
