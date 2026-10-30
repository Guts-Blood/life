#!/usr/bin/env python3
"""Fail-closed Day 20 v3 target-binding and checkpoint-envelope runtime.

V3 intentionally does not redefine the frozen v2 training/checkpoint identity.
It verifies that an outer v3 manifest binds an unchanged v2-compatible core,
then adds a separately hashed target-encoding attestation and checkpoint
envelope.  Evaluation remains a consumer of the v2 core identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

from day20_train_runtime_v2 import (
    Day20TrainingRuntimeV2Error,
    file_sha256,
    load_json,
    object_sha256,
    validate_inputs as validate_v2_inputs,
    verify_checkpoint_integrity,
)


OUTER_MANIFEST_DOMAIN = "day20.qwen35_balanced_lora.experiment_manifest.v3"
TARGET_AUDIT_DOMAIN = "day20.target_encoding_audit.v3"
BINDING_CONFIG_DOMAIN = "day20.qwen35_target_binding_config.v3"
ATTESTATION_DOMAIN = "day20.qwen35_target_binding_attestation.v3"
CHECKPOINT_ENVELOPE_DOMAIN = "day20.qwen35_target_checkpoint_envelope.v3"
TARGET_RUNTIME_DOMAIN = "day20.qwen35_target_runtime_identity.v3"
EXPECTED_TARGET_TEMPLATE_ALIAS = "day20_qwen3_5_target_v3"
EXPECTED_TARGET_CONTRACT_VERSION = "day20.qwen35_code_boundary_target_v3"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class Day20TrainingRuntimeV3Error(ValueError):
    """A v3 outer/core binding, attestation, or envelope invariant failed."""


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Day20TrainingRuntimeV3Error(f"{label} must be a lowercase SHA-256")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Day20TrainingRuntimeV3Error(f"{label} must be an object")
    return value


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    expected = _hash(value.get(field), field)
    actual = object_sha256({key: item for key, item in value.items() if key != field})
    if actual != expected:
        raise Day20TrainingRuntimeV3Error(f"{field} mismatch")
    return expected


def _absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise Day20TrainingRuntimeV3Error(f"{label} path is missing")
    path = Path(value)
    if not path.is_absolute():
        raise Day20TrainingRuntimeV3Error(f"{label} path must be absolute")
    return path.resolve()


def _within(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    parent = root.resolve()
    if resolved == parent or parent not in resolved.parents:
        raise Day20TrainingRuntimeV3Error(f"{label} is outside {parent}")
    return resolved


def _verify_file_identity(
    identity: Mapping[str, Any],
    *,
    label: str,
    root: Path | None = None,
    content_field: str | None = None,
    content_value: str | None = None,
) -> Path:
    path = _absolute_path(identity.get("path"), label)
    if root is not None:
        _within(path, root, label)
    actual_file_hash = file_sha256(path)
    if _hash(identity.get("file_sha256"), f"{label} file_sha256") != actual_file_hash:
        raise Day20TrainingRuntimeV3Error(f"{label} file identity drifted")
    if content_field is not None:
        expected = _hash(identity.get("content_sha256"), f"{label} content_sha256")
        actual = content_value
        if actual is None:
            actual = _self_hash(load_json(path), content_field)
        if expected != actual:
            raise Day20TrainingRuntimeV3Error(f"{label} content identity drifted")
    return path


def _target_module_contract(module_path: Path) -> dict[str, str]:
    """Load the target module under a private name without importing ms-swift."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_day20_target_encoding_v3_identity", module_path)
    if spec is None or spec.loader is None:
        raise Day20TrainingRuntimeV3Error("cannot load target-encoding module")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:  # the identity module must also be offline import-safe
        raise Day20TrainingRuntimeV3Error(
            f"cannot import target-encoding module: {error}"
        ) from error
    alias = getattr(module, "TARGET_TEMPLATE_ALIAS", None)
    version = getattr(module, "TARGET_ENCODING_CONTRACT_VERSION", None)
    contract = getattr(module, "TARGET_ENCODING_CONTRACT", None)
    contract_hash = getattr(module, "TARGET_ENCODING_CONTRACT_SHA256", None)
    if (
        alias != EXPECTED_TARGET_TEMPLATE_ALIAS
        or version != EXPECTED_TARGET_CONTRACT_VERSION
        or not isinstance(contract, Mapping)
        or contract_hash != object_sha256(contract)
    ):
        raise Day20TrainingRuntimeV3Error("target-encoding module contract drifted")
    return {
        "template_alias": alias,
        "contract_version": version,
        "contract_sha256": _hash(contract_hash, "target contract SHA-256"),
    }


def target_runtime_identity(target_module_path: Path) -> dict[str, Any]:
    """Hash the independent v3 target/run-contract implementation layer."""
    here = Path(__file__).resolve().parent
    paths = {
        "day20_train_runtime_v3.py": here / "day20_train_runtime_v3.py",
        "day20_train_plugin_v3.py": here / "day20_train_plugin_v3.py",
        "run_day20_v3_autodl.sh": here / "run_day20_v3_autodl.sh",
        "day20_target_encoding_v3.py": target_module_path.resolve(),
    }
    result: dict[str, Any] = {
        "schema_version": 3,
        "domain": TARGET_RUNTIME_DOMAIN,
        "identity_layer": "target_encoding_envelope_and_run_contract_v3",
        "covered_by_core_v2_runtime_sha256": False,
        "implementation_files": {
            name: {"path": str(path), "file_sha256": file_sha256(path)}
            for name, path in paths.items()
        },
    }
    result["run_contract_sha256"] = object_sha256(result)
    return result


def verify_target_runtime_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    identity = dict(value)
    _self_hash(identity, "run_contract_sha256")
    if (
        identity.get("schema_version") != 3
        or identity.get("domain") != TARGET_RUNTIME_DOMAIN
        or identity.get("identity_layer")
        != "target_encoding_envelope_and_run_contract_v3"
        or identity.get("covered_by_core_v2_runtime_sha256") is not False
    ):
        raise Day20TrainingRuntimeV3Error("v3 target runtime identity drifted")
    files = _mapping(identity.get("implementation_files"), "v3 implementation files")
    if set(files) != {
        "day20_train_runtime_v3.py",
        "day20_train_plugin_v3.py",
        "run_day20_v3_autodl.sh",
        "day20_target_encoding_v3.py",
    }:
        raise Day20TrainingRuntimeV3Error("v3 implementation file cohort drifted")
    for name, raw in files.items():
        _verify_file_identity(_mapping(raw, f"v3 implementation {name}"), label=name)
    return identity


def verify_target_audit(
    audit_path: Path,
    *,
    expected_alias: str = EXPECTED_TARGET_TEMPLATE_ALIAS,
    expected_contract_version: str = EXPECTED_TARGET_CONTRACT_VERSION,
    expected_contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify the complete, self-hashed target-encoding audit."""
    path = audit_path.resolve()
    audit = load_json(path)
    audit_hash = _self_hash(audit, "audit_sha256")
    if (
        audit.get("schema_version") != 3
        or audit.get("domain") != TARGET_AUDIT_DOMAIN
        or audit.get("status") != "pass"
    ):
        raise Day20TrainingRuntimeV3Error("target-encoding audit identity drifted")
    if audit.get("template_alias") != expected_alias:
        raise Day20TrainingRuntimeV3Error("target-encoding audit template alias drifted")
    if audit.get("contract_version") != expected_contract_version:
        raise Day20TrainingRuntimeV3Error("target-encoding audit contract version drifted")
    contract_hash = _hash(audit.get("contract_sha256"), "audit contract_sha256")
    if expected_contract_sha256 is not None and contract_hash != _hash(
        expected_contract_sha256, "expected contract_sha256"
    ):
        raise Day20TrainingRuntimeV3Error("target-encoding audit contract hash drifted")

    datasets = _mapping(audit.get("datasets"), "target-encoding audit datasets")
    if set(datasets) != {"probe", "main", "diagnostic"}:
        raise Day20TrainingRuntimeV3Error(
            "target-encoding audit must cover probe, main, and diagnostic datasets"
        )
    for name in ("probe", "main"):
        evidence = _mapping(datasets[name], f"target-encoding audit {name}")
        local = _mapping(evidence.get("local"), f"target-encoding audit {name} local")
        module = _mapping(evidence.get("module"), f"target-encoding audit {name} module")
        _self_hash(module, "audit_sha256")
        if (
            local.get("status") != "pass"
            or module.get("status") != "pass"
            or module.get("schema_name") != TARGET_AUDIT_DOMAIN
            or module.get("schema_version") != 3
            or module.get("template_alias") != expected_alias
            or module.get("contract_version") != expected_contract_version
            or module.get("contract_sha256") != contract_hash
        ):
            raise Day20TrainingRuntimeV3Error(
                f"target-encoding audit {name} identity or status drifted"
            )
        records = module.get("records")
        code_records = module.get("code_records")
        non_code_records = module.get("non_code_records")
        if (
            isinstance(records, bool)
            or not isinstance(records, int)
            or records <= 0
            or isinstance(code_records, bool)
            or not isinstance(code_records, int)
            or code_records <= 0
            or isinstance(non_code_records, bool)
            or not isinstance(non_code_records, int)
            or non_code_records <= 0
            or code_records + non_code_records != records
            or local.get("records") != records
            or local.get("code_records") != code_records
            or local.get("non_code_records") != non_code_records
            or module.get("code_exact_four_space_labels") != code_records
            or module.get("code_per_row_supervised_count_preserved") != code_records
            or module.get("non_code_input_ids_labels_identical") != non_code_records
            or module.get("native_supervised_tokens")
            != module.get("target_supervised_tokens")
            or local.get("supervised_tokens") != module.get("target_supervised_tokens")
        ):
            raise Day20TrainingRuntimeV3Error(
                f"target-encoding audit {name} coverage or preservation drifted"
            )
        for label in ("ordered_sample_ids_sha256", "ordered_target_evidence_sha256"):
            _hash(local.get(label), f"target-encoding audit {name} local {label}")
        _hash(
            module.get("ordered_target_evidence_sha256"),
            f"target-encoding audit {name} module ordered evidence",
        )

    diagnostic = _mapping(datasets["diagnostic"], "target-encoding audit diagnostic")
    if (
        diagnostic.get("status") != "pass"
        or diagnostic.get("template_alias") != expected_alias
        or diagnostic.get("contract_sha256") != contract_hash
        or diagnostic.get("copy_identical") is not True
        or isinstance(diagnostic.get("records"), bool)
        or not isinstance(diagnostic.get("records"), int)
        or diagnostic.get("records") <= 0
    ):
        raise Day20TrainingRuntimeV3Error(
            "target-encoding diagnostic copy identity drifted"
        )
    _hash(diagnostic.get("file_sha256"), "diagnostic audit file_sha256")
    return {
        "path": str(path),
        "file_sha256": file_sha256(path),
        "content_sha256": audit_hash,
        "template_alias": expected_alias,
        "contract_version": expected_contract_version,
        "contract_sha256": contract_hash,
        "audit": audit,
    }


def validate_prepared_inputs(
    outer_manifest_path: Path,
    *,
    core_manifest_path: Path,
    audit_path: Path,
    target_module_path: Path,
    dataset_path: Path,
    config_path: Path,
    run_kind: str,
    seed: int,
    learning_rate: str,
) -> dict[str, Any]:
    """Validate outer v3 -> frozen v2 core plus target contract bindings."""
    outer_path = outer_manifest_path.resolve()
    outer = load_json(outer_path)
    outer_hash = _self_hash(outer, "manifest_sha256")
    if (
        outer.get("schema_version") != 3
        or outer.get("domain") != OUTER_MANIFEST_DOMAIN
        or outer.get("status") != "prepared"
    ):
        raise Day20TrainingRuntimeV3Error("v3 outer manifest identity drifted")
    run_root = _absolute_path(outer.get("run_root"), "outer manifest run_root")
    if outer_path.parent != run_root:
        raise Day20TrainingRuntimeV3Error("outer manifest path differs from run_root")

    core = _mapping(outer.get("core_v2"), "outer core_v2")
    core_identity = _mapping(core.get("manifest"), "outer core_v2 manifest")
    core_path = _verify_file_identity(
        core_identity,
        label="core v2 manifest",
        root=run_root,
        content_field="manifest_sha256",
    )
    if core_path != core_manifest_path.resolve():
        raise Day20TrainingRuntimeV3Error("requested core v2 manifest differs from outer binding")
    core_manifest = load_json(core_path)
    for field in ("datasets", "configs"):
        if outer.get(field) != core_manifest.get(field):
            raise Day20TrainingRuntimeV3Error(
                f"outer {field} do not exactly bind the v2 core identities"
            )

    target = _mapping(outer.get("target_encoding"), "outer target_encoding")
    module_identity = _mapping(target.get("module"), "outer target module")
    module_path = _verify_file_identity(module_identity, label="target module")
    if module_path != target_module_path.resolve():
        raise Day20TrainingRuntimeV3Error("requested target module differs from outer binding")
    module_contract = _target_module_contract(module_path)
    for field in ("template_alias", "contract_version", "contract_sha256"):
        if target.get(field) != module_contract[field]:
            raise Day20TrainingRuntimeV3Error(f"outer target {field} drifted")

    audit_identity = _mapping(target.get("audit"), "outer target audit")
    bound_audit_path = _verify_file_identity(
        audit_identity,
        label="target audit",
        root=run_root,
        content_field="audit_sha256",
    )
    if bound_audit_path != audit_path.resolve():
        raise Day20TrainingRuntimeV3Error("requested target audit differs from outer binding")
    audit = verify_target_audit(
        bound_audit_path,
        expected_alias=module_contract["template_alias"],
        expected_contract_version=module_contract["contract_version"],
        expected_contract_sha256=module_contract["contract_sha256"],
    )
    diagnostic_audit = _mapping(
        audit["audit"]["datasets"]["diagnostic"], "diagnostic target audit"
    )
    diagnostic_core = _mapping(
        core_manifest.get("datasets", {}).get("diagnostic"),
        "core diagnostic dataset",
    )
    if (
        _absolute_path(diagnostic_audit.get("path"), "diagnostic audit dataset")
        != _absolute_path(diagnostic_core.get("path"), "core diagnostic dataset")
        or diagnostic_audit.get("file_sha256") != diagnostic_core.get("file_sha256")
        or diagnostic_audit.get("records") != diagnostic_core.get("records")
    ):
        raise Day20TrainingRuntimeV3Error(
            "diagnostic audit differs from the core diagnostic dataset"
        )

    try:
        validated = validate_v2_inputs(
            core_path,
            dataset_path=dataset_path,
            config_path=config_path,
            run_kind=run_kind,
            seed=seed,
            learning_rate=learning_rate,
        )
    except (Day20TrainingRuntimeV2Error, OSError, TypeError, ValueError) as error:
        raise Day20TrainingRuntimeV3Error(f"v2 core validation failed: {error}") from error

    config = load_json(config_path.resolve())
    template = _mapping(config.get("template"), "training config template")
    if (
        template.get("template") != module_contract["template_alias"]
        or template.get("model_type") != "qwen3_5"
        or template.get("enable_thinking") is not False
        or template.get("add_non_thinking_prefix") is not True
        or template.get("loss_scale") != "default+ignore_empty_think"
    ):
        raise Day20TrainingRuntimeV3Error("training config target-template contract drifted")

    audit_dataset = _mapping(
        audit["audit"]["datasets"].get(run_kind), f"audit {run_kind}"
    )
    audit_local = _mapping(audit_dataset.get("local"), f"audit {run_kind} local")
    core_dataset = _mapping(
        core_manifest["datasets"].get(run_kind), f"core {run_kind} dataset"
    )
    if (
        audit_local.get("records") != core_dataset.get("records")
        or audit_local.get("supervised_tokens") != validated["supervised_tokens"]
    ):
        raise Day20TrainingRuntimeV3Error(
            f"target audit {run_kind} coverage differs from the core dataset"
        )

    return {
        "status": "pass",
        "identity_layers": {
            "core_v2": "frozen_training_and_checkpoint_runtime_v2",
            "target_v3": "target_encoding_envelope_and_run_contract_v3",
            "v3_covered_by_v2_runtime_hash": False,
        },
        "run_root": str(run_root),
        "outer_manifest": {
            "path": str(outer_path),
            "file_sha256": file_sha256(outer_path),
            "content_sha256": outer_hash,
        },
        "core_manifest": {
            "path": str(core_path),
            "file_sha256": file_sha256(core_path),
            "content_sha256": validated["manifest_sha256"],
        },
        "dataset": {
            "path": validated["dataset"],
            "file_sha256": validated["dataset_file_sha256"],
            "supervised_tokens": validated["supervised_tokens"],
            "temporal_mix_sha256": validated["temporal_mix_sha256"],
        },
        "training_config": {
            "path": validated["training_config"],
            "file_sha256": validated["training_config_file_sha256"],
            "content_sha256": _self_hash(config, "immutable_sha256"),
        },
        "target_module": {
            "path": str(module_path),
            "file_sha256": file_sha256(module_path),
            **module_contract,
        },
        "target_audit": {
            key: audit[key]
            for key in (
                "path",
                "file_sha256",
                "content_sha256",
                "template_alias",
                "contract_version",
                "contract_sha256",
            )
        },
        "run_kind": validated["run_kind"],
        "seed": validated["seed"],
        "learning_rate": validated["learning_rate"],
        # Flat v2-core compatibility fields are intentionally named as such;
        # they do not imply that the target-v3 layer is inside runtime_sha256.
        "manifest_sha256": validated["manifest_sha256"],
        "manifest_file_sha256": validated["manifest_file_sha256"],
        "dataset_file_sha256": validated["dataset_file_sha256"],
        "training_config_file_sha256": validated[
            "training_config_file_sha256"
        ],
        "temporal_mix_sha256": validated["temporal_mix_sha256"],
        "supervised_tokens": validated["supervised_tokens"],
        "config_requires_resolution": validated["config_requires_resolution"],
        "optimizer_steps": validated["optimizer_steps"],
        "checkpoint_tokens": validated["checkpoint_tokens"],
        "checkpoint_steps": validated["checkpoint_steps"],
        "checkpoint_actual_tokens": validated["checkpoint_actual_tokens"],
    }


def build_binding_config(
    *,
    outer_manifest_path: Path,
    core_manifest_path: Path,
    audit_path: Path,
    target_module_path: Path,
    dataset_path: Path,
    training_config_path: Path,
    v2_callback_config_path: Path,
    evidence_dir: Path,
    run_kind: str,
    seed: int,
    learning_rate: str,
) -> dict[str, Any]:
    validated = validate_prepared_inputs(
        outer_manifest_path,
        core_manifest_path=core_manifest_path,
        audit_path=audit_path,
        target_module_path=target_module_path,
        dataset_path=dataset_path,
        config_path=training_config_path,
        run_kind=run_kind,
        seed=seed,
        learning_rate=learning_rate,
    )
    root = Path(validated["run_root"])
    evidence = _within(evidence_dir, root / "evidence", "v3 evidence directory")
    try:
        from day20_train_plugin_v2 import verify_callback_config

        v2_callback = verify_callback_config(v2_callback_config_path.resolve())
    except Exception as error:
        raise Day20TrainingRuntimeV3Error(
            f"v2 callback config verification failed: {error}"
        ) from error
    if (
        v2_callback.get("manifest", {}).get("content_sha256")
        != validated["core_manifest"]["content_sha256"]
        or v2_callback.get("training_config", {}).get("file_sha256")
        != validated["training_config"]["file_sha256"]
        or v2_callback.get("dataset", {}).get("file_sha256")
        != validated["dataset"]["file_sha256"]
    ):
        raise Day20TrainingRuntimeV3Error("v2 callback config differs from v3 core binding")
    result: dict[str, Any] = {
        "schema_version": 3,
        "domain": BINDING_CONFIG_DOMAIN,
        "status": "prepared",
        "run_root": validated["run_root"],
        "evidence_dir": str(evidence),
        "identity_layers": validated["identity_layers"],
        "outer_manifest": validated["outer_manifest"],
        "core_manifest": validated["core_manifest"],
        "dataset": validated["dataset"],
        "training_config": validated["training_config"],
        "target_module": validated["target_module"],
        "target_audit": validated["target_audit"],
        "target_runtime": target_runtime_identity(
            Path(validated["target_module"]["path"])
        ),
        "v2_callback_config": {
            "path": str(v2_callback_config_path.resolve()),
            "file_sha256": file_sha256(v2_callback_config_path.resolve()),
            "content_sha256": v2_callback["callback_config_sha256"],
        },
        "run_kind": validated["run_kind"],
        "seed": validated["seed"],
        "learning_rate": validated["learning_rate"],
    }
    result["binding_config_sha256"] = object_sha256(result)
    return result


def verify_binding_config(path: Path) -> dict[str, Any]:
    config = load_json(path.resolve())
    _self_hash(config, "binding_config_sha256")
    if (
        config.get("schema_version") != 3
        or config.get("domain") != BINDING_CONFIG_DOMAIN
        or config.get("status") != "prepared"
    ):
        raise Day20TrainingRuntimeV3Error("v3 binding config identity drifted")
    if config.get("run_root") != str(Path(config["run_root"]).resolve()):
        raise Day20TrainingRuntimeV3Error("binding config run_root is not canonical")
    _within(
        Path(str(config.get("evidence_dir", ""))),
        Path(config["run_root"]) / "evidence",
        "v3 evidence directory",
    )
    target_module = _mapping(config.get("target_module"), "binding target module")
    target_runtime = verify_target_runtime_identity(
        _mapping(config.get("target_runtime"), "binding target runtime")
    )
    validated = validate_prepared_inputs(
        Path(config["outer_manifest"]["path"]),
        core_manifest_path=Path(config["core_manifest"]["path"]),
        audit_path=Path(config["target_audit"]["path"]),
        target_module_path=Path(target_module["path"]),
        dataset_path=Path(config["dataset"]["path"]),
        config_path=Path(config["training_config"]["path"]),
        run_kind=str(config.get("run_kind")),
        seed=config.get("seed"),
        learning_rate=str(config.get("learning_rate")),
    )
    for field in (
        "identity_layers",
        "outer_manifest",
        "core_manifest",
        "dataset",
        "training_config",
        "target_module",
        "target_audit",
        "run_kind",
        "seed",
        "learning_rate",
    ):
        if config.get(field) != validated.get(field):
            raise Day20TrainingRuntimeV3Error(f"binding config {field} drifted")
    expected_runtime = target_runtime_identity(Path(target_module["path"]))
    if target_runtime != expected_runtime:
        raise Day20TrainingRuntimeV3Error("binding target runtime implementation drifted")
    callback_identity = _mapping(config.get("v2_callback_config"), "v2 callback config")
    callback_path = _verify_file_identity(
        callback_identity,
        label="v2 callback config",
        root=Path(config["run_root"]),
        content_field="callback_config_sha256",
    )
    try:
        from day20_train_plugin_v2 import verify_callback_config

        callback = verify_callback_config(callback_path)
    except Exception as error:
        raise Day20TrainingRuntimeV3Error(
            f"v2 callback config verification failed: {error}"
        ) from error
    if callback["callback_config_sha256"] != callback_identity["content_sha256"]:
        raise Day20TrainingRuntimeV3Error("v2 callback config content drifted")
    return config


def build_attestation(
    binding_config_path: Path,
    *,
    actual_template_alias: str,
    template_class: str,
    trainer_arguments: Mapping[str, Any],
) -> dict[str, Any]:
    binding = verify_binding_config(binding_config_path)
    expected_alias = binding["target_module"]["template_alias"]
    if actual_template_alias != expected_alias:
        raise Day20TrainingRuntimeV3Error(
            f"live template alias drifted: {actual_template_alias!r} != {expected_alias!r}"
        )
    result: dict[str, Any] = {
        "schema_version": 3,
        "domain": ATTESTATION_DOMAIN,
        "status": "pass",
        "identity_layers": binding["identity_layers"],
        "run_root": binding["run_root"],
        "run_kind": binding["run_kind"],
        "seed": binding["seed"],
        "learning_rate": binding["learning_rate"],
        "binding_config": {
            "path": str(binding_config_path.resolve()),
            "file_sha256": file_sha256(binding_config_path.resolve()),
            "content_sha256": binding["binding_config_sha256"],
        },
        "outer_manifest": binding["outer_manifest"],
        "core_manifest": binding["core_manifest"],
        "training_config": binding["training_config"],
        "dataset": binding["dataset"],
        "target_module": binding["target_module"],
        "target_audit": binding["target_audit"],
        "target_runtime": binding["target_runtime"],
        "actual_template": {
            "alias": actual_template_alias,
            "class": template_class,
        },
        "trainer_arguments": dict(trainer_arguments),
    }
    result["attestation_sha256"] = object_sha256(result)
    return result


def write_json_new(path: Path, value: Mapping[str, Any]) -> Path:
    destination = path.resolve()
    if destination.exists():
        raise Day20TrainingRuntimeV3Error(f"refusing to overwrite artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def verify_attestation(path: Path, *, binding_config_path: Path | None = None) -> dict[str, Any]:
    attestation = load_json(path.resolve())
    _self_hash(attestation, "attestation_sha256")
    if (
        attestation.get("schema_version") != 3
        or attestation.get("domain") != ATTESTATION_DOMAIN
        or attestation.get("status") != "pass"
        or attestation.get("identity_layers", {}).get("v3_covered_by_v2_runtime_hash") is not False
    ):
        raise Day20TrainingRuntimeV3Error("v3 target attestation identity drifted")
    binding_identity = _mapping(attestation.get("binding_config"), "attestation binding config")
    bound_path = _verify_file_identity(
        binding_identity,
        label="attestation binding config",
        root=Path(attestation["run_root"]),
        content_field="binding_config_sha256",
    )
    if binding_config_path is not None and bound_path != binding_config_path.resolve():
        raise Day20TrainingRuntimeV3Error("attestation uses an unexpected binding config")
    binding = verify_binding_config(bound_path)
    if binding["binding_config_sha256"] != binding_identity["content_sha256"]:
        raise Day20TrainingRuntimeV3Error("attestation binding config content drifted")
    expected = {
        "identity_layers": binding["identity_layers"],
        "run_root": binding["run_root"],
        "run_kind": binding["run_kind"],
        "seed": binding["seed"],
        "learning_rate": binding["learning_rate"],
        "outer_manifest": binding["outer_manifest"],
        "core_manifest": binding["core_manifest"],
        "training_config": binding["training_config"],
        "dataset": binding["dataset"],
        "target_module": binding["target_module"],
        "target_audit": binding["target_audit"],
        "target_runtime": binding["target_runtime"],
    }
    for field, value in expected.items():
        if attestation.get(field) != value:
            raise Day20TrainingRuntimeV3Error(f"attestation {field} drifted")
    if attestation.get("actual_template", {}).get("alias") != binding["target_module"][
        "template_alias"
    ]:
        raise Day20TrainingRuntimeV3Error("attested live template alias drifted")
    return attestation


def build_checkpoint_envelope(
    *,
    checkpoint: Path,
    run_root: Path,
    training_summary_path: Path,
    binding_config_path: Path,
    attestation_path: Path,
    candidate: str,
) -> dict[str, Any]:
    binding = verify_binding_config(binding_config_path)
    attestation = verify_attestation(
        attestation_path, binding_config_path=binding_config_path
    )
    try:
        from day20_train_plugin_v2 import verify_training_summary

        summary = verify_training_summary(
            training_summary_path.resolve(), run_root=run_root.resolve(), candidate=candidate
        )
    except Exception as error:
        raise Day20TrainingRuntimeV3Error(
            f"v2 training summary verification failed: {error}"
        ) from error
    label = next(
        (name for name, value in summary["candidate_ids"].items() if value == candidate),
        None,
    )
    if label is None:
        raise Day20TrainingRuntimeV3Error("candidate is absent from v2 training summary")
    checkpoint_path = checkpoint.resolve()
    if checkpoint_path != Path(summary["checkpoints"][label]).resolve():
        raise Day20TrainingRuntimeV3Error("checkpoint differs from v2 training summary")
    try:
        integrity = verify_checkpoint_integrity(
            checkpoint_path,
            run_root=run_root.resolve(),
            run_kind=binding["run_kind"],
            seed=binding["seed"],
            learning_rate=binding["learning_rate"],
            dataset_file_sha256=binding["dataset"]["file_sha256"],
            training_config_file_sha256=binding["training_config"]["file_sha256"],
            experiment_manifest_sha256=binding["core_manifest"]["content_sha256"],
            temporal_mix_sha256=binding["dataset"]["temporal_mix_sha256"],
        )
    except (Day20TrainingRuntimeV2Error, OSError, TypeError, ValueError) as error:
        raise Day20TrainingRuntimeV3Error(f"v2 checkpoint verification failed: {error}") from error
    args_identity = integrity["files"].get("training_args.bin")
    if not isinstance(args_identity, Mapping):
        raise Day20TrainingRuntimeV3Error(
            "v2 checkpoint does not contain bound training_args.bin"
        )
    summary_hash = _self_hash(summary, "summary_sha256")
    result: dict[str, Any] = {
        "schema_version": 3,
        "domain": CHECKPOINT_ENVELOPE_DOMAIN,
        "status": "complete",
        "candidate": candidate,
        "checkpoint_label": label,
        "identity_layers": binding["identity_layers"],
        "core_v2": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_integrity_sha256": integrity["integrity_sha256"],
            "checkpoint_snapshot_sha256": integrity["snapshot_sha256"],
            "checkpoint_integrity_file_sha256": file_sha256(
                checkpoint_path / "day20-v2-checkpoint-integrity.json"
            ),
            "training_args": {
                "path": "training_args.bin",
                "bytes": args_identity.get("bytes"),
                "file_sha256": _hash(args_identity.get("sha256"), "training args hash"),
            },
            "training_config": binding["training_config"],
            "experiment_manifest": binding["core_manifest"],
            "training_summary": {
                "path": str(training_summary_path.resolve()),
                "file_sha256": file_sha256(training_summary_path.resolve()),
                "content_sha256": summary_hash,
            },
            "v2_runtime_sha256": integrity["runtime_sha256"],
        },
        "target_v3": {
            "outer_manifest": binding["outer_manifest"],
            "template": binding["target_module"],
            "target_audit": binding["target_audit"],
            "target_runtime": binding["target_runtime"],
            "binding_config": {
                "path": str(binding_config_path.resolve()),
                "file_sha256": file_sha256(binding_config_path.resolve()),
                "content_sha256": binding["binding_config_sha256"],
            },
            "attestation": {
                "path": str(attestation_path.resolve()),
                "file_sha256": file_sha256(attestation_path.resolve()),
                "content_sha256": attestation["attestation_sha256"],
            },
        },
    }
    result["envelope_sha256"] = object_sha256(result)
    return result


def verify_checkpoint_envelope(path: Path, *, run_root: Path) -> dict[str, Any]:
    envelope = load_json(path.resolve())
    _self_hash(envelope, "envelope_sha256")
    if (
        envelope.get("schema_version") != 3
        or envelope.get("domain") != CHECKPOINT_ENVELOPE_DOMAIN
        or envelope.get("status") != "complete"
        or envelope.get("identity_layers", {}).get("v3_covered_by_v2_runtime_hash") is not False
    ):
        raise Day20TrainingRuntimeV3Error("v3 checkpoint envelope identity drifted")
    core = _mapping(envelope.get("core_v2"), "envelope core_v2")
    target = _mapping(envelope.get("target_v3"), "envelope target_v3")
    rebuilt = build_checkpoint_envelope(
        checkpoint=Path(core["checkpoint"]),
        run_root=run_root,
        training_summary_path=Path(core["training_summary"]["path"]),
        binding_config_path=Path(target["binding_config"]["path"]),
        attestation_path=Path(target["attestation"]["path"]),
        candidate=str(envelope.get("candidate")),
    )
    if rebuilt != envelope:
        raise Day20TrainingRuntimeV3Error("v3 checkpoint envelope content drifted")
    return envelope


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-prepared")
    for target in (validate,):
        target.add_argument("--outer-manifest", required=True, type=Path)
        target.add_argument("--core-manifest", required=True, type=Path)
        target.add_argument("--target-audit", required=True, type=Path)
        target.add_argument("--target-module", required=True, type=Path)
        target.add_argument("--dataset", required=True, type=Path)
        target.add_argument("--training-config", required=True, type=Path)
        target.add_argument("--run-kind", required=True, choices=("probe", "main"))
        target.add_argument("--seed", required=True, type=int)
        target.add_argument("--learning-rate", required=True)
    verify_binding = commands.add_parser("verify-binding-config")
    verify_binding.add_argument("--binding-config", required=True, type=Path)
    verify_attest = commands.add_parser("verify-attestation")
    verify_attest.add_argument("--attestation", required=True, type=Path)
    verify_attest.add_argument("--binding-config", type=Path)
    verify_envelope = commands.add_parser("verify-envelope")
    verify_envelope.add_argument("--envelope", required=True, type=Path)
    verify_envelope.add_argument("--run-root", required=True, type=Path)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        if args.command == "validate-prepared":
            result: Any = validate_prepared_inputs(
                args.outer_manifest,
                core_manifest_path=args.core_manifest,
                audit_path=args.target_audit,
                target_module_path=args.target_module,
                dataset_path=args.dataset,
                config_path=args.training_config,
                run_kind=args.run_kind,
                seed=args.seed,
                learning_rate=args.learning_rate,
            )
        elif args.command == "verify-binding-config":
            result = verify_binding_config(args.binding_config)
        elif args.command == "verify-attestation":
            result = verify_attestation(
                args.attestation, binding_config_path=args.binding_config
            )
        else:
            result = verify_checkpoint_envelope(args.envelope, run_root=args.run_root)
    except (Day20TrainingRuntimeV3Error, Day20TrainingRuntimeV2Error) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
