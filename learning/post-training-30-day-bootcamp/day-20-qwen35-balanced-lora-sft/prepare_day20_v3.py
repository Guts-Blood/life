#!/usr/bin/env python3
"""Materialize a Day 20 v3 target-encoding run from an immutable v2 core run."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

from day20_contract_v2 import validate_raw_code_continuation
from day20_ordering_v2 import audit_temporal_mix
from day20_train_runtime_v2 import checkpoint_steps, supervised_token_schedule


CORE_MANIFEST_NAME = "DAY20-V2-MANIFEST.json"
V3_MANIFEST_NAME = "DAY20-V3-MANIFEST.json"
TARGET_AUDIT_NAME = "TARGET-ENCODING-AUDIT.json"
SELECTION_EQUIVALENCE_NAME = "SELECTION-EQUIVALENCE.json"
PARENT_RUN_ROOT_MARKER = ".day20-v2-run-root"
PARENT_RUN_ROOT_MARKER_CONTENT = "day20-qwen35-candidate-factory-v2"
RUN_ROOT_MARKER = ".day20-v3-run-root"
RUN_ROOT_MARKER_CONTENT = "day20-qwen35-target-encoding-v3"
CORE_MANIFEST_DOMAIN = "day20.qwen35_balanced_lora.experiment_manifest.v2"
CORE_CONFIG_DOMAIN = "day20.qwen35_balanced_lora.training_config.v2"
V3_MANIFEST_DOMAIN = "day20.qwen35_balanced_lora.experiment_manifest.v3"
TARGET_AUDIT_DOMAIN = "day20.target_encoding_audit.v3"
SELECTION_EQUIVALENCE_DOMAIN = "day20.selection_equivalence.v3"
NATIVE_TEMPLATE_TYPE = "qwen3_5"
LOSS_SCALE = "default+ignore_empty_think"
_SHA256 = set("0123456789abcdef")


class Day20PreparationV3Error(ValueError):
    """A parent, target-encoding, equivalence, or output invariant failed."""


def object_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise Day20PreparationV3Error(f"required regular file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256 for character in value)
    )


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise Day20PreparationV3Error(f"symbolic JSON input is forbidden: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Day20PreparationV3Error(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise Day20PreparationV3Error(f"JSON root must be an object: {path}")
    return value


def _require_self_hash(
    value: Mapping[str, Any], field: str, label: str
) -> str:
    expected = value.get(field)
    actual = object_sha256({key: item for key, item in value.items() if key != field})
    if not _is_sha256(expected) or expected != actual:
        raise Day20PreparationV3Error(f"{label} {field} mismatch")
    return str(expected)


def _require_regular_directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise Day20PreparationV3Error(f"{label} must be a real directory: {path}")
    return path.resolve()


def _require_within(path: Path, parent: Path, label: str) -> Path:
    resolved = path.resolve()
    root = parent.resolve()
    if resolved == root or root not in resolved.parents:
        raise Day20PreparationV3Error(f"{label} is outside {root}")
    return resolved


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise Day20PreparationV3Error(f"JSONL input must be a regular file: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise Day20PreparationV3Error(
                        f"blank JSONL row: {path}:{line_number}"
                    )
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise Day20PreparationV3Error(
                        f"JSONL row is not an object: {path}:{line_number}"
                    )
                rows.append(row)
    except (OSError, json.JSONDecodeError) as error:
        raise Day20PreparationV3Error(f"cannot load JSONL {path}: {error}") from error
    if not rows:
        raise Day20PreparationV3Error(f"JSONL input is empty: {path}")
    return rows


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Day20PreparationV3Error(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(
                json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
    except FileExistsError as error:
        raise Day20PreparationV3Error(
            f"refusing to overwrite artifact: {path}"
        ) from error


def _write_text_new(path: Path, value: str) -> None:
    if path.exists() or path.is_symlink():
        raise Day20PreparationV3Error(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(value)
    except FileExistsError as error:
        raise Day20PreparationV3Error(
            f"refusing to overwrite artifact: {path}"
        ) from error


def _write_jsonl_new(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists() or path.is_symlink():
        raise Day20PreparationV3Error(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    except FileExistsError as error:
        raise Day20PreparationV3Error(
            f"refusing to overwrite artifact: {path}"
        ) from error


def _copy_file_new(source: Path, target: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise Day20PreparationV3Error(f"copy source is not a regular file: {source}")
    if target.exists() or target.is_symlink():
        raise Day20PreparationV3Error(f"refusing to overwrite artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise Day20PreparationV3Error(
            f"refusing to overwrite artifact: {target}"
        ) from error
    try:
        with source.open("rb") as read_handle, os.fdopen(descriptor, "wb") as write_handle:
            shutil.copyfileobj(read_handle, write_handle, length=8 * 1024 * 1024)
    except Exception:
        target.unlink(missing_ok=True)
        raise


def _import_target_module(path: Path) -> ModuleType:
    module_path = path.expanduser()
    if not module_path.is_absolute():
        module_path = module_path.resolve()
    if module_path.is_symlink() or not module_path.is_file():
        raise Day20PreparationV3Error(
            f"target-encoding module must be a regular file: {module_path}"
        )
    module_name = f"day20_target_encoding_v3_{file_sha256(module_path)[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise Day20PreparationV3Error("cannot import target-encoding module")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise Day20PreparationV3Error(
            f"cannot import target-encoding module: {type(error).__name__}: {error}"
        ) from error
    return module


def _target_module_contract(module: ModuleType) -> tuple[str, str, Mapping[str, Any]]:
    template_type = getattr(
        module, "TARGET_TEMPLATE_TYPE", getattr(module, "TARGET_TEMPLATE_ALIAS", None)
    )
    contract_value = getattr(module, "target_encoding_contract", None)
    if callable(contract_value):
        contract = contract_value()
    else:
        contract = getattr(module, "TARGET_ENCODING_CONTRACT", contract_value)
    contract_version = getattr(module, "TARGET_ENCODING_CONTRACT_VERSION", None)
    declared_sha = getattr(module, "TARGET_ENCODING_CONTRACT_SHA256", None)
    if not isinstance(template_type, str) or not template_type:
        raise Day20PreparationV3Error("target module omitted TARGET_TEMPLATE_TYPE")
    if not isinstance(contract, Mapping):
        raise Day20PreparationV3Error("target module omitted target_encoding_contract")
    contract = dict(contract)
    computed_sha = object_sha256(contract)
    if declared_sha is not None and declared_sha != computed_sha:
        raise Day20PreparationV3Error("target-encoding contract SHA drifted")
    if contract_version is None:
        contract_version = contract.get("contract_version") or contract.get("version")
    if not isinstance(contract_version, str) or not contract_version:
        raise Day20PreparationV3Error("target module omitted contract version")
    return template_type, contract_version, contract


def _validate_parent(parent_root: Path) -> tuple[dict[str, Any], Path, str]:
    root = _require_regular_directory(parent_root, "parent run root")
    marker = root / PARENT_RUN_ROOT_MARKER
    if (
        marker.is_symlink()
        or not marker.is_file()
        or marker.read_text(encoding="utf-8").strip()
        != PARENT_RUN_ROOT_MARKER_CONTENT
    ):
        raise Day20PreparationV3Error("parent run-root marker drifted")
    manifest_path = root / CORE_MANIFEST_NAME
    manifest = _load_json(manifest_path)
    content_sha = _require_self_hash(manifest, "manifest_sha256", "parent manifest")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("domain") != CORE_MANIFEST_DOMAIN
        or Path(str(manifest.get("run_root", ""))).resolve() != root
    ):
        raise Day20PreparationV3Error("parent core manifest identity drifted")
    return manifest, manifest_path, content_sha


def _validate_output_root(run_root: Path, planned: Sequence[Path]) -> Path:
    if run_root.is_symlink():
        raise Day20PreparationV3Error("output run root must not be symbolic")
    if run_root.exists() and not run_root.is_dir():
        raise Day20PreparationV3Error("output run root must be a directory")
    root = run_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise Day20PreparationV3Error("output run root must be empty")
    if any(path.exists() or path.is_symlink() for path in planned):
        collisions = [str(path) for path in planned if path.exists() or path.is_symlink()]
        raise Day20PreparationV3Error(f"refusing to overwrite artifacts: {collisions}")
    for ancestor in [root, *root.parents]:
        if ancestor.exists():
            if ancestor.is_symlink() or not ancestor.is_dir():
                raise Day20PreparationV3Error(
                    f"output path has a symbolic/non-directory ancestor: {ancestor}"
                )
            break
    return root


def _verify_ms_swift(expected_commit: str) -> dict[str, Any]:
    try:
        import swift
    except ImportError as error:
        raise Day20PreparationV3Error("ms-swift runtime is unavailable") from error
    swift_root = Path(swift.__file__).resolve().parent.parent
    try:
        commit = subprocess.run(
            ["git", "-C", str(swift_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(swift_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise Day20PreparationV3Error("cannot verify ms-swift checkout") from error
    if commit != expected_commit or dirty:
        raise Day20PreparationV3Error("ms-swift commit/worktree identity drifted")
    qwen_template = swift_root / "swift" / "template" / "templates" / "qwen.py"
    return {
        "root": str(swift_root),
        "commit": commit,
        "worktree_clean": True,
        "qwen_template": {
            "path": str(qwen_template),
            "file_sha256": file_sha256(qwen_template),
        },
    }


def _identity_path(
    identity: Mapping[str, Any], parent_root: Path, label: str
) -> Path:
    path = _require_within(Path(str(identity.get("path", ""))), parent_root, label)
    if path.is_symlink() or not path.is_file():
        raise Day20PreparationV3Error(f"{label} is not a regular file")
    if identity.get("file_sha256") != file_sha256(path):
        raise Day20PreparationV3Error(f"{label} file identity drifted")
    return path


def _encode_arrays(template: Any, messages: Sequence[Mapping[str, Any]]) -> tuple[list[int], list[int]]:
    try:
        encoded = template.encode({"messages": list(messages)}, return_length=True)
    except Exception as error:
        raise Day20PreparationV3Error(
            f"target template encoding failed: {type(error).__name__}: {error}"
        ) from error
    if not isinstance(encoded, Mapping):
        raise Day20PreparationV3Error("target template returned a non-mapping")
    input_ids = encoded.get("input_ids")
    labels = encoded.get("labels")
    if not isinstance(input_ids, (list, tuple)) or not isinstance(labels, (list, tuple)):
        raise Day20PreparationV3Error("target template omitted input_ids or labels")
    try:
        tokens = [int(value) for value in input_ids]
        label_ids = [int(value) for value in labels]
    except (TypeError, ValueError) as error:
        raise Day20PreparationV3Error("target token arrays are non-integral") from error
    if not tokens or len(tokens) != len(label_ids):
        raise Day20PreparationV3Error("target token arrays have invalid lengths")
    return tokens, label_ids


def _decode_supervised(template: Any, labels: Sequence[int]) -> str:
    supervised = [value for value in labels if value != -100]
    try:
        return template.tokenizer.decode(
            supervised,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return template.tokenizer.decode(supervised, skip_special_tokens=True)


def _reencode_dataset(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_template: Any,
    label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rewritten: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    code_records = 0
    non_code_records = 0
    for index, source in enumerate(rows, 1):
        row = copy.deepcopy(dict(source))
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            raise Day20PreparationV3Error(
                f"{label} row {index} must contain one user/assistant pair"
            )
        if [message.get("role") for message in messages] != ["user", "assistant"]:
            raise Day20PreparationV3Error(f"{label} row {index} role order drifted")
        before_core = {
            key: copy.deepcopy(value)
            for key, value in row.items()
            if key
            not in {
                "qwen35_input_tokens",
                "qwen35_supervised_tokens",
                "qwen35_render_sha256",
                "qwen35_labels_sha256",
            }
        }
        old_input_tokens = row.get("qwen35_input_tokens")
        old_supervised = row.get("qwen35_supervised_tokens")
        old_render_sha = row.get("qwen35_render_sha256")
        old_labels_sha = row.get("qwen35_labels_sha256")
        if (
            isinstance(old_input_tokens, bool)
            or not isinstance(old_input_tokens, int)
            or isinstance(old_supervised, bool)
            or not isinstance(old_supervised, int)
            or not _is_sha256(old_render_sha)
            or not _is_sha256(old_labels_sha)
        ):
            raise Day20PreparationV3Error(f"{label} row {index} token evidence is invalid")
        input_ids, labels = _encode_arrays(target_template, messages)
        supervised = sum(value != -100 for value in labels[1:])
        if supervised != old_supervised:
            raise Day20PreparationV3Error(
                f"{label} row {index} supervised-token count changed"
            )
        render_sha = object_sha256(input_ids)
        labels_sha = object_sha256(labels)
        is_code = row.get("skill") == "code" and row.get("target_format") == "code_continuation"
        if is_code:
            code_records += 1
            target = messages[-1].get("content")
            code_prefix = row.get("code_prefix")
            validated = validate_raw_code_continuation(target, code_prefix)
            stored = row.get("code_continuation_v2")
            expected_code = validated.as_evidence()
            if not isinstance(stored, Mapping) or any(
                stored.get(field) != expected_code[field]
                for field in ("canonical", "canonical_sha256", "ast_sha256")
            ):
                raise Day20PreparationV3Error(
                    f"{label} row {index} Code target evidence drifted"
                )
            decoded = _decode_supervised(target_template, labels)
            if decoded != target:
                raise Day20PreparationV3Error(
                    f"{label} row {index} supervised Code target is not byte-exact"
                )
            if len(input_ids) != old_input_tokens + 1:
                raise Day20PreparationV3Error(
                    f"{label} row {index} Code input-token delta is not +1"
                )
        else:
            non_code_records += 1
            if (
                len(input_ids) != old_input_tokens
                or render_sha != old_render_sha
                or labels_sha != old_labels_sha
            ):
                raise Day20PreparationV3Error(
                    f"{label} row {index} non-Code token arrays changed"
                )
        row.update(
            {
                "qwen35_input_tokens": len(input_ids),
                "qwen35_supervised_tokens": supervised,
                "qwen35_render_sha256": render_sha,
                "qwen35_labels_sha256": labels_sha,
            }
        )
        after_core = {
            key: copy.deepcopy(value)
            for key, value in row.items()
            if key
            not in {
                "qwen35_input_tokens",
                "qwen35_supervised_tokens",
                "qwen35_render_sha256",
                "qwen35_labels_sha256",
            }
        }
        if before_core != after_core:
            raise Day20PreparationV3Error(f"{label} row {index} non-token content changed")
        rewritten.append(row)
        evidence.append(
            {
                "sample_id": row.get("sample_id"),
                "skill": row.get("skill"),
                "old_input_tokens": old_input_tokens,
                "new_input_tokens": len(input_ids),
                "supervised_tokens": supervised,
                "old_render_sha256": old_render_sha,
                "new_render_sha256": render_sha,
                "old_labels_sha256": old_labels_sha,
                "new_labels_sha256": labels_sha,
            }
        )
    return rewritten, {
        "status": "pass",
        "records": len(rewritten),
        "code_records": code_records,
        "non_code_records": non_code_records,
        "supervised_tokens": sum(
            int(row["qwen35_supervised_tokens"]) for row in rewritten
        ),
        "ordered_sample_ids_sha256": object_sha256(
            [row.get("sample_id") for row in rewritten]
        ),
        "ordered_target_evidence_sha256": object_sha256(evidence),
    }


def _call_module_audit(
    module: ModuleType,
    rows: Sequence[Mapping[str, Any]],
    *,
    native_template: Any,
    target_template: Any,
    expected_total: int,
) -> dict[str, Any]:
    audit = getattr(module, "audit_dataset_pair", None)
    if audit is None:
        audit = getattr(module, "audit_target_encoding_v3", None)
    if not callable(audit):
        raise Day20PreparationV3Error("target module omitted audit_dataset_pair")
    try:
        result = audit(
            rows,
            native_template=native_template,
            target_template=target_template,
            expected_total_supervised_tokens=expected_total,
        )
    except TypeError:
        result = audit(
            rows,
            native_template=native_template,
            target_template=target_template,
            expected_supervised_tokens=expected_total,
        )
    if not isinstance(result, Mapping) or result.get("status") != "pass":
        raise Day20PreparationV3Error("target module dataset audit failed")
    if (
        result.get("records") != len(rows)
        or result.get("target_supervised_tokens", result.get("supervised_tokens"))
        != expected_total
    ):
        raise Day20PreparationV3Error("target module dataset audit totals drifted")
    return dict(result)


def _dataset_summary(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    parent: Mapping[str, Any],
) -> dict[str, Any]:
    summary = copy.deepcopy(dict(parent))
    summary["path"] = str(path.resolve())
    summary["file_sha256"] = file_sha256(path)
    summary["records"] = len(rows)
    summary["ordered_sample_ids"] = [row["sample_id"] for row in rows]
    summary["ordered_supervised_tokens"] = [
        row["qwen35_supervised_tokens"] for row in rows
    ]
    summary["selection_sha256"] = object_sha256(summary["ordered_sample_ids"])
    summary["supervised_tokens"] = sum(summary["ordered_supervised_tokens"])
    temporal = audit_temporal_mix(rows)
    if temporal.get("status") != "pass":
        raise Day20PreparationV3Error("rewritten dataset failed temporal-mix audit")
    summary["temporal_mix_audit"] = temporal
    summary["temporal_mix_sha256"] = object_sha256(temporal)
    invariant_fields = (
        "records",
        "supervised_tokens",
        "ordered_sample_ids",
        "ordered_supervised_tokens",
        "selection_sha256",
        "ordering_version",
        "supervised_tokens_by_skill",
        "supervised_tokens_by_format",
        "records_by_format",
        "cohort_supervised_tokens",
        "temporal_mix_audit",
        "temporal_mix_sha256",
    )
    for field in invariant_fields:
        if summary.get(field) != parent.get(field):
            raise Day20PreparationV3Error(f"dataset summary changed invariant: {field}")
    return summary


def _diagnostic_summary(
    target: Path, parent_path: Path, parent: Mapping[str, Any]
) -> dict[str, Any]:
    summary = copy.deepcopy(dict(parent))
    summary["path"] = str(target.resolve())
    summary["file_sha256"] = file_sha256(target)
    for field in (
        "file_sha256",
        "records",
        "records_by_skill",
        "ordered_sample_ids",
        "selection_sha256",
        "eval_manifest_path",
        "eval_manifest_file_sha256",
    ):
        if summary.get(field) != parent.get(field):
            raise Day20PreparationV3Error(
                f"diagnostic copy changed parent invariant: {field}"
            )
    if file_sha256(parent_path) != file_sha256(target):
        raise Day20PreparationV3Error("diagnostic copy is not byte-identical")
    return summary


def _rewrite_config(
    source: Mapping[str, Any],
    *,
    path: Path,
    dataset: Mapping[str, Any],
    template_type: str,
) -> dict[str, Any]:
    config = copy.deepcopy(dict(source))
    _require_self_hash(config, "immutable_sha256", "parent training config")
    if config.get("schema_version") != 2 or config.get("domain") != CORE_CONFIG_DOMAIN:
        raise Day20PreparationV3Error("parent training config identity drifted")
    config["data"].update(
        {
            "path": dataset["path"],
            "file_sha256": dataset["file_sha256"],
            "supervised_tokens": dataset["supervised_tokens"],
            "temporal_mix_sha256": dataset["temporal_mix_sha256"],
        }
    )
    if config.get("template", {}).get("template") != NATIVE_TEMPLATE_TYPE:
        raise Day20PreparationV3Error("parent config native template identity drifted")
    if config["template"].get("loss_scale") != LOSS_SCALE:
        raise Day20PreparationV3Error("parent config loss-scale identity drifted")
    config["template"]["template"] = template_type
    config.pop("immutable_sha256", None)
    config["immutable_sha256"] = object_sha256(config)
    _write_json_new(path, config)
    return config


def _build_core_manifest(
    *,
    parent: Mapping[str, Any],
    run_root: Path,
    datasets: Mapping[str, Any],
    configs: Mapping[str, Any],
) -> dict[str, Any]:
    core = copy.deepcopy(dict(parent))
    core["created_at_utc"] = _utc_now()
    core["run_root"] = str(run_root)
    core["datasets"] = copy.deepcopy(dict(datasets))
    core["configs"] = copy.deepcopy(dict(configs))
    core.pop("manifest_sha256", None)
    core["manifest_sha256"] = object_sha256(core)
    return core


def _selection_equivalence(
    *,
    parent_manifest_path: Path,
    parent_manifest: Mapping[str, Any],
    parent_content_sha: str,
    rewritten: Mapping[str, Sequence[Mapping[str, Any]]],
    dataset_summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for label in ("probe", "main"):
        parent_identity = parent_manifest["datasets"][label]
        new_identity = dataset_summaries[label]
        rows = rewritten[label]
        schedule = checkpoint_steps(
            supervised_token_schedule(Path(str(new_identity["path"]))),
            run_kind=label,
        )
        parent_schedule = checkpoint_steps(
            supervised_token_schedule(Path(str(parent_identity["path"]))),
            run_kind=label,
        )
        fields = {
            "ordered_sample_ids": [row["sample_id"] for row in rows],
            "ordered_supervised_tokens": [
                row["qwen35_supervised_tokens"] for row in rows
            ],
            "temporal_mix_sha256": new_identity["temporal_mix_sha256"],
            "selection_sha256": new_identity["selection_sha256"],
        }
        expected = {key: parent_identity[key] for key in fields}
        if fields != expected or schedule != parent_schedule:
            raise Day20PreparationV3Error(f"{label} selection/checkpoint equivalence failed")
        datasets[label] = {
            "status": "equivalent",
            "records": len(rows),
            "supervised_tokens": sum(fields["ordered_supervised_tokens"]),
            "ordered_sample_ids_sha256": object_sha256(fields["ordered_sample_ids"]),
            "ordered_supervised_tokens_sha256": object_sha256(
                fields["ordered_supervised_tokens"]
            ),
            "temporal_mix_sha256": fields["temporal_mix_sha256"],
            "checkpoint_schedule": schedule,
        }
    result: dict[str, Any] = {
        "schema_version": 3,
        "domain": SELECTION_EQUIVALENCE_DOMAIN,
        "status": "pass",
        "parent_manifest": {
            "path": str(parent_manifest_path),
            "file_sha256": file_sha256(parent_manifest_path),
            "content_sha256": parent_content_sha,
        },
        "probe_is_subset_of_main": set(
            row["sample_id"] for row in rewritten["probe"]
        ).issubset(row["sample_id"] for row in rewritten["main"]),
        "datasets": datasets,
    }
    if not result["probe_is_subset_of_main"]:
        raise Day20PreparationV3Error("probe is no longer nested in main")
    result["equivalence_sha256"] = object_sha256(result)
    return result


def prepare_run_v3(
    *,
    parent_run_root: Path,
    run_root: Path,
    target_encoding_module: Path,
    expected_ms_swift_commit: str,
) -> dict[str, Any]:
    if (
        not isinstance(expected_ms_swift_commit, str)
        or len(expected_ms_swift_commit) != 40
        or any(character not in _SHA256 for character in expected_ms_swift_commit)
    ):
        raise Day20PreparationV3Error("expected ms-swift commit must be a bare SHA-1")
    parent, parent_manifest_path, parent_content_sha = _validate_parent(
        parent_run_root
    )
    parent_root = parent_manifest_path.parent
    module = _import_target_module(target_encoding_module)
    template_type, contract_version, contract = _target_module_contract(module)
    contract_sha = object_sha256(contract)
    module_path = Path(str(module.__file__)).resolve()
    model_identity = parent.get("base_model_identity")
    if not isinstance(model_identity, Mapping):
        raise Day20PreparationV3Error("parent base-model identity is missing")
    model_path = Path(str(model_identity.get("path", "")))
    if model_path.is_symlink() or not model_path.is_dir():
        raise Day20PreparationV3Error("parent base-model path is unavailable")

    paths = {
        "marker": run_root.resolve() / RUN_ROOT_MARKER,
        "probe": run_root.resolve() / "data" / "probe-v2.jsonl",
        "main": run_root.resolve() / "data" / "main-v2.jsonl",
        "diagnostic": run_root.resolve() / "data" / "diagnostic-v2.jsonl",
        "core_manifest": run_root.resolve() / CORE_MANIFEST_NAME,
        "v3_manifest": run_root.resolve() / V3_MANIFEST_NAME,
        "audit": run_root.resolve() / TARGET_AUDIT_NAME,
        "equivalence": run_root.resolve() / SELECTION_EQUIVALENCE_NAME,
    }
    parent_configs = parent.get("configs")
    if not isinstance(parent_configs, Mapping):
        raise Day20PreparationV3Error("parent configs are missing")
    probe_identities = parent_configs.get("probes")
    if not isinstance(probe_identities, Mapping):
        raise Day20PreparationV3Error("parent probe configs are missing")
    config_paths = {
        str(lr): run_root.resolve() / "configs" / Path(str(identity["path"])).name
        for lr, identity in probe_identities.items()
    }
    main_config_path = (
        run_root.resolve()
        / "configs"
        / Path(str(parent_configs["main_template"]["path"])).name
    )
    root = _validate_output_root(
        run_root,
        [*paths.values(), *config_paths.values(), main_config_path],
    )
    if root == parent_root or root in parent_root.parents or parent_root in root.parents:
        raise Day20PreparationV3Error("parent and output run roots must be disjoint")

    register = getattr(module, "register_target_template_v3", None)
    build = getattr(module, "build_target_template_v3", None)
    if not callable(register) or not callable(build):
        raise Day20PreparationV3Error("target module omitted register/build API")
    register()
    try:
        target_template = build(model_path, max_length=2304)
    except Exception as error:
        raise Day20PreparationV3Error(
            f"cannot build target template: {type(error).__name__}: {error}"
        ) from error
    meta = getattr(target_template, "template_meta", None)
    if (
        getattr(meta, "template_type", None) != template_type
        or getattr(target_template, "mode", None) != "train"
        or getattr(target_template, "max_length", None) != 2304
        or getattr(target_template, "enable_thinking", None) is not False
        or getattr(target_template, "add_non_thinking_prefix", None) is not True
    ):
        raise Day20PreparationV3Error("target template runtime contract drifted")
    ms_swift_identity = _verify_ms_swift(expected_ms_swift_commit)
    native_builder = getattr(module, "build_native_template_v3", None)
    if callable(native_builder):
        native_template = native_builder(model_path, max_length=2304)
    else:
        try:
            from swift import get_model_processor, get_template

            _, processor = get_model_processor(
                str(model_path),
                model_type="qwen3_5",
                load_model=False,
                use_hf=True,
                download_model=False,
            )
            native_template = get_template(
                processor,
                template_type=NATIVE_TEMPLATE_TYPE,
                max_length=2304,
                truncation_strategy="raise",
                padding_free=False,
                loss_scale=LOSS_SCALE,
                enable_thinking=False,
                add_non_thinking_prefix=True,
            )
            native_template.set_mode("train")
        except Exception as error:
            raise Day20PreparationV3Error(
                f"cannot build native audit template: {type(error).__name__}: {error}"
            ) from error

    parent_dataset_identities = parent.get("datasets")
    if not isinstance(parent_dataset_identities, Mapping):
        raise Day20PreparationV3Error("parent datasets are missing")
    parent_dataset_paths = {
        label: _identity_path(parent_dataset_identities[label], parent_root, label)
        for label in ("probe", "main", "diagnostic")
    }
    parent_rows = {
        label: _load_jsonl(parent_dataset_paths[label]) for label in ("probe", "main")
    }
    rewritten: dict[str, list[dict[str, Any]]] = {}
    local_audits: dict[str, dict[str, Any]] = {}
    module_audits: dict[str, dict[str, Any]] = {}
    for label in ("probe", "main"):
        expected = int(parent_dataset_identities[label]["supervised_tokens"])
        rewritten[label], local_audits[label] = _reencode_dataset(
            parent_rows[label], target_template=target_template, label=label
        )
        if local_audits[label]["supervised_tokens"] != expected:
            raise Day20PreparationV3Error(f"{label} supervised-token total drifted")
        module_audits[label] = _call_module_audit(
            module,
            rewritten[label],
            native_template=native_template,
            target_template=target_template,
            expected_total=expected,
        )

    root.mkdir(parents=True, exist_ok=True)
    _write_text_new(paths["marker"], RUN_ROOT_MARKER_CONTENT + "\n")
    _write_jsonl_new(paths["probe"], rewritten["probe"])
    _write_jsonl_new(paths["main"], rewritten["main"])
    _copy_file_new(parent_dataset_paths["diagnostic"], paths["diagnostic"])
    summaries = {
        "probe": _dataset_summary(
            paths["probe"], rewritten["probe"], parent_dataset_identities["probe"]
        ),
        "main": _dataset_summary(
            paths["main"], rewritten["main"], parent_dataset_identities["main"]
        ),
        "diagnostic": _diagnostic_summary(
            paths["diagnostic"],
            parent_dataset_paths["diagnostic"],
            parent_dataset_identities["diagnostic"],
        ),
    }

    probe_config_identities: dict[str, Any] = {}
    for lr, identity in probe_identities.items():
        source_path = _identity_path(identity, parent_root, f"probe config {lr}")
        config = _rewrite_config(
            _load_json(source_path),
            path=config_paths[str(lr)],
            dataset=summaries["probe"],
            template_type=template_type,
        )
        probe_config_identities[str(lr)] = {
            **{
                key: value
                for key, value in identity.items()
                if key not in {"path", "file_sha256", "immutable_sha256"}
            },
            "path": str(config_paths[str(lr)]),
            "file_sha256": file_sha256(config_paths[str(lr)]),
            "immutable_sha256": config["immutable_sha256"],
        }
    main_identity = parent_configs["main_template"]
    main_source_path = _identity_path(main_identity, parent_root, "main config")
    main_config = _rewrite_config(
        _load_json(main_source_path),
        path=main_config_path,
        dataset=summaries["main"],
        template_type=template_type,
    )
    config_identities = {
        "probes": probe_config_identities,
        "main_template": {
            **{
                key: value
                for key, value in main_identity.items()
                if key not in {"path", "file_sha256", "immutable_sha256"}
            },
            "path": str(main_config_path),
            "file_sha256": file_sha256(main_config_path),
            "immutable_sha256": main_config["immutable_sha256"],
        },
    }
    core = _build_core_manifest(
        parent=parent,
        run_root=root,
        datasets=summaries,
        configs=config_identities,
    )
    _write_json_new(paths["core_manifest"], core)

    equivalence = _selection_equivalence(
        parent_manifest_path=parent_manifest_path,
        parent_manifest=parent,
        parent_content_sha=parent_content_sha,
        rewritten=rewritten,
        dataset_summaries=summaries,
    )
    _write_json_new(paths["equivalence"], equivalence)
    target_audit: dict[str, Any] = {
        "schema_version": 3,
        "domain": TARGET_AUDIT_DOMAIN,
        "status": "pass",
        "template_alias": template_type,
        "contract_version": contract_version,
        "contract_sha256": contract_sha,
        "native_template": NATIVE_TEMPLATE_TYPE,
        "loss_scale": LOSS_SCALE,
        "provenance": {
            "module": {
                "path": str(module_path),
                "file_sha256": file_sha256(module_path),
            },
            "model_snapshot_sha256": model_identity.get("snapshot_sha256"),
            "ms_swift": ms_swift_identity,
        },
        "datasets": {
            "probe": {
                "local": local_audits["probe"],
                "module": module_audits["probe"],
            },
            "main": {
                "local": local_audits["main"],
                "module": module_audits["main"],
            },
            "diagnostic": {
                "status": "pass",
                "path": str(paths["diagnostic"]),
                "records": summaries["diagnostic"]["records"],
                "file_sha256": summaries["diagnostic"]["file_sha256"],
                "template_alias": template_type,
                "contract_sha256": contract_sha,
                "copy_identical": True,
            },
        },
    }
    target_audit["audit_sha256"] = object_sha256(target_audit)
    _write_json_new(paths["audit"], target_audit)

    v3: dict[str, Any] = {
        "schema_version": 3,
        "domain": V3_MANIFEST_DOMAIN,
        "status": "prepared",
        "created_at_utc": _utc_now(),
        "run_root": str(root),
        "parent_run": {
            "run_root": str(parent_root),
            "manifest": {
                "path": str(parent_manifest_path),
                "file_sha256": file_sha256(parent_manifest_path),
                "content_sha256": parent_content_sha,
            },
        },
        "core_v2": {
            "manifest": {
                "path": str(paths["core_manifest"]),
                "file_sha256": file_sha256(paths["core_manifest"]),
                "content_sha256": core["manifest_sha256"],
            }
        },
        "target_encoding": {
            "template_alias": template_type,
            "contract_version": contract_version,
            "contract_sha256": contract_sha,
            "module": {
                "path": str(module_path),
                "file_sha256": file_sha256(module_path),
            },
            "audit": {
                "path": str(paths["audit"]),
                "file_sha256": file_sha256(paths["audit"]),
                "content_sha256": target_audit["audit_sha256"],
            },
        },
        "selection_equivalence": {
            "path": str(paths["equivalence"]),
            "file_sha256": file_sha256(paths["equivalence"]),
            "content_sha256": equivalence["equivalence_sha256"],
        },
        "native_qwen": {
            "model_identity": copy.deepcopy(dict(model_identity)),
            "template": NATIVE_TEMPLATE_TYPE,
            "loss_scale": LOSS_SCALE,
            "ms_swift_commit": expected_ms_swift_commit,
            "ms_swift": ms_swift_identity,
        },
        "datasets": copy.deepcopy(summaries),
        "configs": copy.deepcopy(config_identities),
    }
    v3["manifest_sha256"] = object_sha256(v3)
    _write_json_new(paths["v3_manifest"], v3)
    return v3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--target-encoding-module", required=True, type=Path)
    parser.add_argument("--expected-ms-swift-commit", required=True)
    args = parser.parse_args()
    result = prepare_run_v3(
        parent_run_root=args.parent_run_root,
        run_root=args.run_root,
        target_encoding_module=args.target_encoding_module,
        expected_ms_swift_commit=args.expected_ms_swift_commit,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
