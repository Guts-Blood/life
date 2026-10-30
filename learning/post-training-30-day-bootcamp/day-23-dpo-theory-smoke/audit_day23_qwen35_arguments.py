#!/usr/bin/env python3
"""Parse both Day 23 stages with pinned ms-swift without loading model weights."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import day23_contract as contract


DAY23_DIR = Path(__file__).resolve().parent
BOOTCAMP_ROOT = DAY23_DIR.parent
REPO_ROOT = BOOTCAMP_ROOT.parents[1]
DEFAULT_RUN_CONTRACT = (
    BOOTCAMP_ROOT
    / "artifacts/configs/day23-qwen35-coding-dpo/cpu-run-contract.json"
)
DEFAULT_MODEL_PATH = (
    REPO_ROOT
    / "tmp/qwen35-v2/models/Qwen--Qwen3.5-4B-Base"
    / contract.MODEL_REVISION
)
DEFAULT_OUTPUT = (
    BOOTCAMP_ROOT
    / "artifacts/eval/day23-qwen35-coding-dpo-argument-audit.json"
)
STAGE_METADATA = {
    "checkpoint_candidates",
    "fresh_start_from_parent",
    "mechanism_pair_ids",
    "purpose",
    "remove_common_args",
}
CLI_DEFAULT_KEYS = {
    "adapters": [],
    "ref_adapters": [],
    "ref_model": None,
    "resume_from_checkpoint": None,
}


class Day23ArgumentAuditError(ValueError):
    """The intended config does not parse under pinned ms-swift semantics."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Day23ArgumentAuditError(message)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _resolve_dataset(value: str) -> str:
    path_value, marker, sample = value.rpartition("#")
    if not marker or not sample.isdigit():
        path_value, suffix = value, ""
    else:
        suffix = f"#{sample}"
    path = (BOOTCAMP_ROOT / path_value).resolve()
    _require(path.is_file(), f"dataset is missing: {path}")
    return str(path) + suffix


def _bound_stage_config(
    value: Mapping[str, Any], stage_name: str, model_path: Path
) -> dict[str, Any]:
    common = dict(value["intended_ms_swift_args"])
    stage = dict(value["stages"][stage_name])
    for key in stage.get("remove_common_args", []):
        common.pop(str(key), None)
    runtime_overrides = {
        key: item for key, item in stage.items() if key not in STAGE_METADATA
    }
    result = common | runtime_overrides
    result["model"] = str(model_path)
    result["dataset"] = [_resolve_dataset(item) for item in result["dataset"]]
    if "val_dataset" in result:
        result["val_dataset"] = [
            _resolve_dataset(item) for item in result["val_dataset"]
        ]
    result["external_plugins"] = [
        str((BOOTCAMP_ROOT / item).resolve()) for item in result["external_plugins"]
    ]
    result["output_dir"] = f"/tmp/day23-{stage_name}-parse-only"
    _require(
        not any(item is None or item == [] for item in result.values()),
        f"{stage_name}: executable JSON contains null or empty-list values",
    )
    _require(
        not any("__BIND_" in str(item) for item in result.values()),
        f"{stage_name}: executable JSON contains unresolved placeholders",
    )
    return result


def _parse_real_json_cli(
    config: Mapping[str, Any], *, stage_name: str, directory: Path
) -> tuple[Any, list[str]]:
    try:
        from swift.arguments import RLHFArguments
        from swift.cli.main import parse_yaml_args
        from swift.utils import parse_args
    except ImportError as error:
        raise Day23ArgumentAuditError("pinned ms-swift JSON CLI is unavailable") from error
    path = directory / f"{stage_name}.json"
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    argv = [str(path)]
    previous_config = os.environ.get("SWIFT_CONFIG_FILE")
    try:
        parse_yaml_args(argv)
        parsed, remaining = parse_args(RLHFArguments, argv)
    except (OSError, SystemExit, ValueError) as error:
        raise Day23ArgumentAuditError(
            f"{stage_name} real JSON CLI parse failed: {type(error).__name__}: {error}"
        ) from error
    finally:
        if previous_config is None:
            os.environ.pop("SWIFT_CONFIG_FILE", None)
        else:
            os.environ["SWIFT_CONFIG_FILE"] = previous_config
    _require(not remaining, f"{stage_name}: unparsed JSON CLI arguments: {remaining}")
    return parsed, argv


def build_audit(
    *,
    run_contract_path: Path = DEFAULT_RUN_CONTRACT,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    run_contract_path = run_contract_path.resolve()
    model_path = model_path.resolve()
    _require(model_path.is_dir(), f"parse-only processor snapshot is missing: {model_path}")
    value = contract.load_json(run_contract_path)
    contract_sha = contract.verify_self_hash(
        value, "contract_sha256", "Day 23 CPU run contract"
    )
    _require(value.get("status") == "cpu_ready_gpu_pending", "run contract status drifted")
    try:
        import swift
        from swift.arguments import RLHFArguments
        from swift.template import TEMPLATE_MAPPING
    except ImportError as error:
        raise Day23ArgumentAuditError("pinned ms-swift argument classes are unavailable") from error
    _require(
        Path(swift.__file__).resolve().parent
        == (REPO_ROOT / "vendor/ms-swift/swift").resolve(),
        "argument audit imported ms-swift outside the pinned vendor checkout",
    )

    argument_fields = {field.name for field in dataclasses.fields(RLHFArguments)}
    common = dict(value["intended_ms_swift_args"])
    unknown_common = sorted(set(common) - argument_fields)
    _require(not unknown_common, f"unknown common ms-swift args: {unknown_common}")
    stages: dict[str, Any] = {}
    _require(
        value.get("cli_default_semantics", {}).get("omitted_keys")
        == CLI_DEFAULT_KEYS,
        "run contract CLI default omission semantics drifted",
    )
    with tempfile.TemporaryDirectory(prefix="day23-json-cli-") as temporary:
        temporary_path = Path(temporary)
        for stage_name in ("mechanism_5step", "bounded_smoke_30step"):
            stage = dict(value["stages"][stage_name])
            runtime_overrides = {
                key: item for key, item in stage.items() if key not in STAGE_METADATA
            }
            unknown_stage = sorted(set(runtime_overrides) - argument_fields)
            _require(not unknown_stage, f"unknown {stage_name} args: {unknown_stage}")
            args_dict = _bound_stage_config(value, stage_name, model_path)
            parsed, cli_argv = _parse_real_json_cli(
                args_dict, stage_name=stage_name, directory=temporary_path
            )
            _require(parsed.rlhf_type == "dpo", f"{stage_name}: rlhf_type drifted")
            _require(parsed.adapters == [], f"{stage_name}: adapters default drifted")
            _require(parsed.ref_model is None, f"{stage_name}: ref_model must remain absent")
            _require(parsed.ref_adapters == [], f"{stage_name}: ref_adapters must remain empty")
            _require(
                parsed.resume_from_checkpoint is None,
                f"{stage_name}: resume_from_checkpoint must remain absent",
            )
            _require(
                parsed.template == "day23_qwen3_5_dpo_target_v1",
                f"{stage_name}: template alias drifted",
            )
            _require(
                parsed.loss_scale == "default+ignore_empty_think",
                f"{stage_name}: loss scale drifted",
            )
            _require(parsed.max_length == 512, f"{stage_name}: max_length drifted")
            _require(parsed.beta == 0.1 and parsed.loss_type == "sigmoid", f"{stage_name}: DPO objective drifted")
            _require(parsed.tuner_type == "lora", f"{stage_name}: tuner type drifted")
            _require(parsed.freeze_vit is True and parsed.freeze_aligner is True, f"{stage_name}: multimodal freeze drifted")
            _require(parsed.max_steps == stage["max_steps"], f"{stage_name}: max steps drifted")
            _require(parsed.add_version is False, f"{stage_name}: output versioning drifted")
            _require(parsed.save_strategy == "steps", f"{stage_name}: save strategy drifted")
            _require(parsed.save_only_model is False, f"{stage_name}: save-only-model drifted")
            _require(
                parsed.load_best_model_at_end is False,
                f"{stage_name}: automatic best-model loading drifted",
            )
            _require(
                parsed.sync_ref_model is False,
                f"{stage_name}: reference synchronization must be disabled",
            )
            training_args = parsed.training_args
            _require(
                getattr(training_args, "reference_free", None) in (None, False),
                f"{stage_name}: reference-free DPO is forbidden",
            )
            _require(
                getattr(training_args, "precompute_ref_log_probs", False) is False,
                f"{stage_name}: reference precompute drifted",
            )
            _require(
                getattr(training_args, "disable_dropout", False) is True,
                f"{stage_name}: policy/reference dropout must be disabled",
            )
            stages[stage_name] = {
                "status": "pass",
                "entrypoint": "swift.cli.main.parse_yaml_args_then_swift.utils.parse_args",
                "json_cli_argument_count": len(cli_argv),
                "executable_config_keys": sorted(args_dict),
                "omitted_cli_default_keys": sorted(CLI_DEFAULT_KEYS),
                "rlhf_type": parsed.rlhf_type,
                "template": parsed.template,
                "tuner_type": parsed.tuner_type,
                "adapters": parsed.adapters,
                "ref_model": parsed.ref_model,
                "ref_adapters": parsed.ref_adapters,
                "resume_from_checkpoint": parsed.resume_from_checkpoint,
                "beta": parsed.beta,
                "loss_type": parsed.loss_type,
                "loss_scale": parsed.loss_scale,
                "max_length": parsed.max_length,
                "max_steps": parsed.max_steps,
                "add_version": parsed.add_version,
                "save_strategy": str(parsed.save_strategy),
                "save_only_model": parsed.save_only_model,
                "load_best_model_at_end": parsed.load_best_model_at_end,
                "sync_ref_model": parsed.sync_ref_model,
                "reference_free": getattr(training_args, "reference_free", None),
                "precompute_ref_log_probs": getattr(
                    training_args, "precompute_ref_log_probs", False
                ),
                "disable_dropout": getattr(training_args, "disable_dropout", None),
                "dataset_shuffle": parsed.dataset_shuffle,
                "train_dataloader_shuffle": parsed.train_dataloader_shuffle,
                "dataset_syntax": value["stages"][stage_name].get(
                    "dataset", common["dataset"]
                ),
                "val_dataset_syntax": (
                    common["val_dataset"]
                    if "val_dataset" in args_dict
                    else []
                ),
                "external_plugin_registered": (
                    "day23_qwen3_5_dpo_target_v1" in TEMPLATE_MAPPING
                ),
                "model_role": "local_base_processor_metadata_parse_only_not_policy_parent",
                "weights_loaded": False,
            }
            if stage_name == "mechanism_5step":
                train_dataset, val_dataset = parsed.load_dataset()
                _require(len(train_dataset) == 4, "mechanism dataset#4 did not load four rows")
                _require(val_dataset is None, "mechanism stage unexpectedly loaded dev data")
                expected_rows = contract.load_jsonl(
                    BOOTCAMP_ROOT / common["dataset"][0]
                )[:4]
                actual_text = [
                    {
                        "messages": row["messages"],
                        "rejected_response": row["rejected_response"],
                    }
                    for row in train_dataset
                ]
                expected_text = [
                    {
                        "messages": row["messages"],
                        "rejected_response": row["rejected_response"],
                    }
                    for row in expected_rows
                ]
                _require(
                    actual_text == expected_text,
                    "mechanism dataset#4 order/text differs from frozen first four rows",
                )
                _require(
                    stage.get("mechanism_pair_ids")
                    == [str(row["pair_id"]) for row in expected_rows],
                    "mechanism pair ID inventory drifted",
                )
                stages[stage_name]["mechanism_loader"] = {
                    "status": "pass",
                    "records": 4,
                    "validation_records": 0,
                    "pair_ids": stage["mechanism_pair_ids"],
                    "ordered_row_hashes_sha256": contract.object_sha256(
                        [row["row_sha256"] for row in expected_rows]
                    ),
                    "standard_text_fields_exact": True,
                    "dataset_shuffle": False,
                    "train_dataloader_shuffle": False,
                }
            _require(
                stages[stage_name]["external_plugin_registered"] is True,
                f"{stage_name}: external template plugin was not registered",
            )

    result: dict[str, Any] = {
        "schema_name": "day23.qwen35_dpo_argument_parse_audit",
        "schema_version": 1,
        "status": "pass",
        "scope": "pinned_real_JSON_CLI_and_external_plugin_parse_no_model_weights",
        "run_contract": {
            "path": "artifacts/configs/day23-qwen35-coding-dpo/cpu-run-contract.json",
            "file_sha256": contract.file_sha256(run_contract_path),
            "contract_sha256": contract_sha,
        },
        "pinned_ms_swift_commit": contract.MS_SWIFT_COMMIT,
        "imported_ms_swift_from_pinned_vendor": True,
        "rlhf_argument_field_count": len(argument_fields),
        "unknown_common_argument_keys": [],
        "unknown_stage_argument_keys": {
            "mechanism_5step": [],
            "bounded_smoke_30step": [],
        },
        "stages": stages,
        "runtime": {
            "python": platform.python_version(),
            "ms_swift": _package_version("ms-swift"),
            "transformers": _package_version("transformers"),
            "torch": _package_version("torch"),
            "peft": _package_version("peft"),
            "trl": _package_version("trl"),
        },
        "claim_boundary": {
            "argument_schema_and_plugin_parse": "pass",
            "model_weights_loaded": False,
            "local_base_used_as_policy_parent": False,
            "gpu_optimizer_ready": False,
            "remaining_gates": [contract.REMOTE_PAYLOAD_GATE, contract.GPU_RUNTIME_GATE],
        },
    }
    result["audit_sha256"] = contract.object_sha256(result)
    return result


def output_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode(
        "utf-8"
    ) + b"\n"


def apply_output(path: Path, payload: bytes, *, mode: str) -> None:
    path = path.resolve()
    _require(path.parent.is_dir(), f"output parent is missing: {path.parent}")
    if mode == "build":
        try:
            with path.open("xb") as handle:
                handle.write(payload)
        except FileExistsError as error:
            raise Day23ArgumentAuditError(f"refusing to overwrite: {path}") from error
    elif mode == "check":
        try:
            actual = path.read_bytes()
        except OSError as error:
            raise Day23ArgumentAuditError(f"cannot read frozen argument audit: {path}") from error
        _require(actual == payload, f"frozen argument audit drifted: {path}")
    else:
        raise Day23ArgumentAuditError(f"unsupported mode: {mode}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("build", "check"), required=True)
    parser.add_argument("--run-contract", type=Path, default=DEFAULT_RUN_CONTRACT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_audit(
            run_contract_path=args.run_contract,
            model_path=args.model,
        )
        apply_output(args.output, output_bytes(result), mode=args.mode)
    except (OSError, contract.Day23ContractError, Day23ArgumentAuditError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "valid_day23_argument_cpu_gate",
                "mode": args.mode,
                "audit_sha256": result["audit_sha256"],
                "stages": list(result["stages"]),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
