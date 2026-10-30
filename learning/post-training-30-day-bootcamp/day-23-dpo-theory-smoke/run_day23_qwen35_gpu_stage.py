#!/usr/bin/env python3
"""Run one bound Day 23 GPU stage with fail-closed runtime evidence.

This is deliberately a thin wrapper around the pinned ms-swift JSON CLI path:
``parse_yaml_args`` -> ``try_use_single_device_mode`` -> ``rlhf_main``.  The
only monkeypatches are observers.  They do not replace a forward, loss,
optimizer step, checkpoint save, sampler, or reference implementation.

The wrapper is intentionally not part of the frozen CPU contract.  Its own
file identity must be pinned by the GPU execution binding before it can run.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


BINDING_SCHEMA = "day23.qwen35_dpo_gpu_execution_binding"
BINDING_STATUS = "gpu_execution_bound"
RECEIPT_SCHEMA = "day23.qwen35_dpo_gpu_stage_receipt"
RECEIPT_VERSION = 1
CORRECTION_CLAIM_SCHEMA = "day23.qwen35_dpo_mechanism_correction_claim"
EXPECTED_PRIOR_MECHANISM_FAILURE_PATH = Path(
    "/root/autodl-tmp/runs/day23-qwen35-dpo-20260814T091033Z/"
    "evidence/mechanism-5step/failure-receipt.json"
)
STAGES = ("g2_one_step", "mechanism_5step", "bounded_smoke_30step")
EXPECTED_MAX_STEPS = {
    "g2_one_step": 1,
    "mechanism_5step": 5,
    "bounded_smoke_30step": 30,
}
EXPECTED_CHECKPOINT_STEPS = {
    "g2_one_step": (1,),
    "mechanism_5step": (5,),
    "bounded_smoke_30step": (15, 30),
}
FORBIDDEN_CONFIG_KEYS = {
    "adapters",
    "ref_adapters",
    "ref_model",
    "resume_from_checkpoint",
}
FORBIDDEN_TOPOLOGY_KEYS = {
    "deepspeed",
    "deepspeed_autotp_size",
    "device_map",
    "fsdp",
    "fsdp_config",
    "load_in_4bit",
    "load_in_8bit",
    "quant_bits",
    "quant_method",
}
REQUIRED_CHECKPOINT_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "args.json",
    "optimizer.pt",
    "rng_state_0.pth",
    "rng_state_1.pth",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
}
EXPECTED_TARGET_MODULES = 248
EXPECTED_LORA_TENSORS = 496
EXPECTED_LORA_PARAMS = 16_232_448
MIN_FREE_MEMORY_FRACTION = 0.15
EXPECTED_WORLD_SIZE = 2
EXPECTED_PER_DEVICE_TRAIN_BATCH = 4
EXPECTED_MECHANISM_PER_DEVICE_TRAIN_BATCH = 1
EXPECTED_PER_DEVICE_EVAL_BATCH = 4
EXPECTED_GRADIENT_ACCUMULATION = 1
EXPECTED_MECHANISM_GRADIENT_ACCUMULATION = 4
MECHANISM_PAIR_IDS = (
    "mbpp:task:602:s1pair:37673b32b9385aea",
    "mbpp:task:604:s1pair:388a8daec7e97c5a",
    "mbpp:task:605:s1pair:9ae278fe596ed4d4",
    "mbpp:task:610:s1pair:9e57fb63f7544777",
)
MAX_RECORDED_FORWARD_EVENTS = 16
MAX_RECORDED_REFERENCE_EVENTS = 16
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class Day23GPUStageError(RuntimeError):
    """A binding, runtime, observation, or sealed-output invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Day23GPUStageError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def object_sha256(value: Mapping[str, Any], self_field: str | None = None) -> str:
    payload = dict(value)
    if self_field is not None:
        payload.pop(self_field, None)
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256",
    )
    return str(value)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _text(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} must be non-empty text")
    return str(value)


def _absolute_path(value: Any, label: str, *, must_exist: bool) -> Path:
    raw = Path(_text(value, label)).expanduser()
    _require(raw.is_absolute(), f"{label} must be absolute: {raw}")
    resolved = raw.resolve(strict=must_exist)
    if must_exist:
        _require(resolved.exists(), f"{label} is missing: {resolved}")
    return resolved


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Day23GPUStageError(f"cannot read {label}: {path}") from error
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def verify_self_hash(
    value: Mapping[str, Any], field: str, label: str
) -> str:
    expected = _sha256(value.get(field), f"{label}.{field}")
    actual = object_sha256(value, field)
    _require(expected == actual, f"{label}.{field} does not bind its contents")
    return expected


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return repr(value)


def write_sealed_json(path: Path, value: Mapping[str, Any], self_field: str) -> str:
    """Create one self-hashed receipt without overwriting an existing file."""
    payload = dict(value)
    payload[self_field] = object_sha256(payload)
    data = json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    _require(path.is_absolute(), f"sealed receipt path must be absolute: {path}")
    _require(path.parent.is_dir(), f"receipt parent is missing: {path.parent}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            path.unlink()
        raise
    return str(payload[self_field])


def _validate_mechanism_correction_claim(
    path: Path,
    *,
    binding_sha256: str,
    config_file_sha256: str,
    prior_receipt_sha256: str,
) -> dict[str, Any]:
    value = load_json(path, "mechanism correction claim")
    claim_sha = verify_self_hash(
        value, "claim_sha256", "mechanism correction claim"
    )
    _require(
        value.get("schema_name") == CORRECTION_CLAIM_SCHEMA
        and value.get("schema_version") == 1
        and value.get("status") == "claimed_for_execution"
        and value.get("attempt_name")
        == "mechanism_5step_topology_correction_attempt2"
        and value.get("attempt_number") == 2
        and value.get("binding_sha256") == binding_sha256
        and value.get("executable_config_file_sha256")
        == config_file_sha256
        and value.get("prior_failure_receipt_sha256")
        == prior_receipt_sha256
        and value.get("one_shot") is True
        and value.get("third_attempt_forbidden") is True,
        "mechanism correction claim identity drifted",
    )
    return {
        "path": str(path.resolve()),
        "file_sha256": file_sha256(path),
        "claim_sha256": claim_sha,
        "binding_sha256": binding_sha256,
        "status": "claimed_for_execution",
    }


def _git(checkout: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise Day23GPUStageError(f"required package is absent: {name}") from error


def _process_identity() -> dict[str, Any]:
    stat = Path("/proc/self/stat")
    boot = Path("/proc/sys/kernel/random/boot_id")
    _require(stat.is_file() and boot.is_file(), "Linux process identity files are missing")
    raw = stat.read_text(encoding="utf-8").strip()
    _require(") " in raw, "cannot parse /proc/self/stat")
    fields = raw.split(") ", 1)[1].split()
    _require(len(fields) > 19 and fields[19].isdigit(), "process start ticks are invalid")
    python = Path(sys.executable).resolve(strict=True)
    return {
        "pid": os.getpid(),
        "proc_start_ticks": int(fields[19]),
        "boot_id": boot.read_text(encoding="utf-8").strip(),
        "python_executable_file_sha256": file_sha256(python),
    }


def _find_path(value: Any, names: Iterable[str]) -> str | None:
    """Find a uniquely named path leaf in binding metadata.

    Binding paths are generated, not user-authored.  This helper tolerates a
    harmless nesting change while rejecting ambiguity rather than guessing.
    """
    wanted = set(names)
    matches: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if key in wanted and isinstance(child, str):
                    matches.append(child)
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    unique = sorted(set(matches))
    _require(len(unique) <= 1, f"ambiguous bound receipt path: {unique}")
    return unique[0] if unique else None


def _receipt_paths(binding: Mapping[str, Any], stage: str) -> tuple[Path, Path]:
    evidence = _mapping(binding.get("evidence_contract"), "binding.evidence_contract")
    stage_evidence: Any = evidence.get(stage)
    if stage_evidence is None:
        stages = evidence.get("stages")
        if isinstance(stages, Mapping):
            stage_evidence = stages.get(stage)
    if stage_evidence is None:
        receipts = evidence.get("receipts")
        if isinstance(receipts, Mapping):
            stage_evidence = receipts.get(stage)
    _require(stage_evidence is not None, f"binding has no evidence contract for {stage}")
    success = _find_path(
        stage_evidence,
        {"success", "success_path", "success_receipt", "success_receipt_path"},
    )
    failure = _find_path(
        stage_evidence,
        {"failure", "failure_path", "failure_receipt", "failure_receipt_path"},
    )
    if success is None or failure is None:
        evidence_dir_raw = _find_path(stage_evidence, {"evidence_dir", "directory", "path"})
        _require(evidence_dir_raw is not None, f"{stage}: receipt paths are not bound")
        evidence_dir = _absolute_path(
            evidence_dir_raw, f"binding.evidence_contract.{stage}.evidence_dir", must_exist=True
        )
        success_path = evidence_dir / "success-receipt.json"
        failure_path = evidence_dir / "failure-receipt.json"
    else:
        success_path = _absolute_path(success, f"{stage}.success_receipt", must_exist=False)
        failure_path = _absolute_path(failure, f"{stage}.failure_receipt", must_exist=False)
    _require(success_path != failure_path, f"{stage}: success/failure receipts collide")
    _require(success_path.parent.is_dir(), f"success receipt parent missing: {success_path.parent}")
    _require(failure_path.parent.is_dir(), f"failure receipt parent missing: {failure_path.parent}")
    return success_path, failure_path


def _producer_identity(binding: Mapping[str, Any]) -> Mapping[str, Any]:
    evidence = _mapping(binding.get("evidence_contract"), "binding.evidence_contract")
    producers = _mapping(evidence.get("producers"), "binding.evidence_contract.producers")
    return _mapping(
        producers.get("stage_runner_and_live_runtime_hook"),
        "evidence_contract.producers.stage_runner_and_live_runtime_hook",
    )


def _validated_process_identity(
    value: Any, label: str, *, python_file_sha256: str
) -> dict[str, Any]:
    identity = _mapping(value, label)
    required = {
        "pid",
        "proc_start_ticks",
        "boot_id",
        "python_executable_file_sha256",
    }
    _require(set(identity) == required, f"{label} field inventory drifted")
    _require(
        isinstance(identity.get("pid"), int)
        and not isinstance(identity.get("pid"), bool)
        and identity["pid"] > 0,
        f"{label} pid is invalid",
    )
    _require(
        isinstance(identity.get("proc_start_ticks"), int)
        and not isinstance(identity.get("proc_start_ticks"), bool)
        and identity["proc_start_ticks"] >= 0,
        f"{label} process start ticks are invalid",
    )
    _require(
        isinstance(identity.get("boot_id"), str)
        and bool(identity["boot_id"].strip()),
        f"{label} boot ID is invalid",
    )
    _require(
        identity.get("python_executable_file_sha256") == python_file_sha256,
        f"{label} Python executable hash drifted",
    )
    return dict(identity)


def validate_stage_success_receipt_value(
    binding: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    stage: str,
    binding_sha256: str,
) -> dict[str, Any]:
    """Validate every security-relevant field of a runner success receipt.

    This is intentionally public so the independent evaluator and successor
    stages apply one fail-closed receipt contract instead of trusting a plain
    self-hash plus a few booleans.
    """
    _require(stage in STAGES, f"unsupported receipt stage: {stage}")
    common = _mapping(
        _mapping(binding.get("evidence_contract"), "binding.evidence_contract").get(
            "common_receipt"
        ),
        "binding.evidence_contract.common_receipt",
    )
    required_fields = common.get("required_fields")
    _require(
        isinstance(required_fields, list)
        and set(required_fields) <= set(receipt),
        "stage receipt required field inventory is incomplete",
    )
    _require(
        receipt.get("schema_name") == RECEIPT_SCHEMA
        and receipt.get("schema_version") == RECEIPT_VERSION
        and receipt.get("status") == "pass"
        and receipt.get("stage") == stage,
        "stage success receipt schema/status drifted",
    )

    paths = _mapping(binding.get("paths"), "binding.paths")
    binding_path = _absolute_path(
        paths.get("binding_manifest"), "binding.paths.binding_manifest", must_exist=True
    )
    binding_file_sha = file_sha256(binding_path)
    binding_entry = _mapping(receipt.get("binding"), "receipt.binding")
    _require(
        Path(_text(receipt.get("binding_path"), "receipt.binding_path")).resolve()
        == binding_path
        and receipt.get("binding_file_sha256") == binding_file_sha
        and receipt.get("binding_sha256") == binding_sha256
        and Path(_text(binding_entry.get("path"), "receipt.binding.path")).resolve()
        == binding_path
        and binding_entry.get("file_sha256") == binding_file_sha
        and binding_entry.get("binding_sha256") == binding_sha256,
        "stage receipt binding identity drifted",
    )

    producer_bound = _producer_identity(binding)
    producer = _mapping(receipt.get("producer"), "receipt.producer")
    _require(
        Path(_text(producer.get("path"), "receipt.producer.path")).resolve()
        == _absolute_path(
            producer_bound.get("path"), "bound stage runner", must_exist=True
        )
        and producer.get("file_sha256") == producer_bound.get("file_sha256"),
        "stage receipt producer drifted",
    )

    cpu_bound = _mapping(binding.get("cpu_contract"), "binding.cpu_contract")
    cpu = _mapping(receipt.get("cpu_contract"), "receipt.cpu_contract")
    _require(
        Path(_text(cpu.get("path"), "receipt.cpu_contract.path")).resolve()
        == _absolute_path(cpu_bound.get("path"), "bound CPU contract", must_exist=True)
        and cpu.get("file_sha256") == cpu_bound.get("file_sha256")
        and cpu.get("contract_sha256") == cpu_bound.get("contract_sha256")
        and receipt.get("cpu_contract_sha256") == cpu_bound.get("contract_sha256"),
        "stage receipt CPU contract identity drifted",
    )

    stage_bound = _mapping(
        _mapping(binding.get("stages"), "binding.stages").get(stage),
        f"binding.stages.{stage}",
    )
    config_bound = _mapping(
        stage_bound.get("executable_config"), f"binding.stages.{stage}.executable_config"
    )
    bound_config_path = _absolute_path(
        config_bound.get("path"), "bound executable config", must_exist=True
    )
    bound_config_value = load_json(bound_config_path, "bound executable config")
    config = _mapping(receipt.get("executable_config"), "receipt.executable_config")
    _require(
        Path(_text(config.get("path"), "receipt.executable_config.path")).resolve()
        == bound_config_path
        and config.get("file_sha256") == config_bound.get("file_sha256")
        and Path(
            _text(
                receipt.get("executable_config_path"),
                "receipt.executable_config_path",
            )
        ).resolve()
        == bound_config_path
        and receipt.get("executable_config_file_sha256")
        == config_bound.get("file_sha256"),
        "stage receipt executable config identity drifted",
    )
    remote = _mapping(binding.get("remote_parent"), "binding.remote_parent")
    merged = _mapping(remote.get("merged_export"), "binding.remote_parent.merged_export")
    _require(
        receipt.get("remote_parent_files_sha256") == merged.get("files_sha256"),
        "stage receipt remote parent aggregate drifted",
    )

    runtime_bound = _mapping(binding.get("runtime_parse"), "binding.runtime_parse")
    runtime = _mapping(receipt.get("runtime_identity"), "receipt.runtime_identity")
    _require(receipt.get("runtime") == receipt.get("runtime_identity"), "receipt runtime projections differ")
    python_bound = _mapping(runtime_bound.get("python_executable"), "bound Python")
    python_runtime = _mapping(runtime.get("python"), "receipt.runtime_identity.python")
    _require(
        python_runtime
        == {
            "path": python_bound.get("path"),
            "file_sha256": python_bound.get("file_sha256"),
            "version": python_bound.get("version"),
        },
        "stage receipt Python runtime drifted",
    )
    swift_runtime = _mapping(
        runtime.get("ms_swift_checkout"),
        "receipt.runtime_identity.ms_swift_checkout",
    )
    swift_file = Path(_text(swift_runtime.get("import_file"), "receipt swift import file")).resolve()
    swift_import_root = _absolute_path(
        runtime_bound.get("ms_swift_import_root"), "bound swift import root", must_exist=True
    )
    _require(
        swift_runtime.get("path") == runtime_bound.get("ms_swift_checkout")
        and swift_runtime.get("commit") == runtime_bound.get("ms_swift_commit")
        and swift_runtime.get("clean") is True
        and swift_file.is_file()
        and swift_file.is_relative_to(swift_import_root),
        "stage receipt ms-swift runtime drifted",
    )
    _require(
        runtime.get("package_versions") == runtime_bound.get("package_versions"),
        "stage receipt package runtime drifted",
    )
    process = _validated_process_identity(
        receipt.get("process_identity"),
        "stage receipt process identity",
        python_file_sha256=_sha256(
            python_bound.get("file_sha256"), "bound Python file hash"
        ),
    )

    claim = _mapping(receipt.get("claim_boundary"), "receipt.claim_boundary")
    for key in (
        "runtime_stage_passed",
        "fresh_lora_only",
        "reference_disable_adapter_observed",
        "frozen_tensor_versions_unchanged",
        "two_gpu_ddp_runtime_proven",
    ):
        _require(claim.get(key) is True, f"stage receipt claim did not prove {key}")
    _require(claim.get("heldout_consumed") is False, "stage receipt consumed heldout")

    amendment = _mapping(
        binding.get("user_authorized_gpu_amendment"),
        "binding.user_authorized_gpu_amendment",
    )
    correction = _mapping(
        amendment.get("mechanism_protocol_correction"),
        "binding mechanism protocol correction",
    )
    _require(
        receipt.get("mechanism_protocol_correction") == dict(correction),
        "stage receipt mechanism correction provenance drifted",
    )
    receipt_correction_claim = receipt.get("mechanism_correction_claim")
    if stage == "g2_one_step":
        _require(
            receipt_correction_claim is None,
            "G2 receipt unexpectedly claims the mechanism retry",
        )
    else:
        claim_spec = _mapping(
            correction.get("execution_claim"),
            "binding mechanism correction execution claim",
        )
        claim_path = _absolute_path(
            claim_spec.get("path"),
            "binding mechanism correction execution claim path",
            must_exist=True,
        )
        mechanism_config = _mapping(
            _mapping(binding.get("stages"), "binding stages").get(
                "mechanism_5step"
            ),
            "binding mechanism stage",
        )
        mechanism_config_sha = _mapping(
            mechanism_config.get("executable_config"),
            "binding mechanism executable config",
        ).get("file_sha256")
        prior = _mapping(correction.get("prior_attempt"), "prior mechanism attempt")
        expected_claim = _validate_mechanism_correction_claim(
            claim_path,
            binding_sha256=binding_sha256,
            config_file_sha256=mechanism_config_sha,
            prior_receipt_sha256=prior.get("receipt_sha256"),
        )
        _require(
            receipt_correction_claim == expected_claim,
            "stage receipt mechanism correction claim drifted",
        )

    distributed = _mapping(receipt.get("distributed_evidence"), "receipt.distributed_evidence")
    ranks = distributed.get("ranks")
    expected_train_batch = (
        EXPECTED_MECHANISM_PER_DEVICE_TRAIN_BATCH
        if stage == "mechanism_5step"
        else EXPECTED_PER_DEVICE_TRAIN_BATCH
    )
    expected_gradient_accumulation = (
        EXPECTED_MECHANISM_GRADIENT_ACCUMULATION
        if stage == "mechanism_5step"
        else EXPECTED_GRADIENT_ACCUMULATION
    )
    expected_batches_per_rank = 2 if stage == "mechanism_5step" else None
    nominal_global = (
        EXPECTED_WORLD_SIZE
        * expected_train_batch
        * expected_gradient_accumulation
    )
    _require(
        distributed.get("world_size") == EXPECTED_WORLD_SIZE
        and bound_config_value.get("per_device_train_batch_size")
        == expected_train_batch
        and bound_config_value.get("gradient_accumulation_steps")
        == expected_gradient_accumulation
        and distributed.get("configured_nominal_global_train_batch_size")
        == nominal_global
        and isinstance(distributed.get("realized_global_batch_profile"), Mapping)
        and isinstance(ranks, list)
        and len(ranks) == EXPECTED_WORLD_SIZE,
        "stage receipt distributed rank inventory drifted",
    )
    validated_ranks: list[dict[str, Any]] = []
    process_keys: set[tuple[Any, Any, Any]] = set()
    final_digests: set[str] = set()
    for expected_rank, raw_rank in enumerate(ranks):
        rank = _mapping(raw_rank, f"receipt distributed rank {expected_rank}")
        _require(
            rank.get("rank") == expected_rank
            and rank.get("local_rank") == expected_rank,
            f"receipt distributed rank {expected_rank} identity drifted",
        )
        rank_process = _validated_process_identity(
            rank.get("process_identity"),
            f"receipt distributed rank {expected_rank} process identity",
            python_file_sha256=python_bound["file_sha256"],
        )
        topology = _mapping(rank.get("topology"), f"receipt rank {expected_rank} topology")
        _require(
            topology.get("world_size") == EXPECTED_WORLD_SIZE
            and topology.get("rank") == expected_rank
            and topology.get("local_rank") == expected_rank
            and topology.get("visible_device_count") == EXPECTED_WORLD_SIZE
            and topology.get("CUDA_VISIBLE_DEVICES") == "0,1"
            and topology.get("torch_distributed_initialized") is True,
            f"receipt distributed rank {expected_rank} topology drifted",
        )
        observations = _mapping(
            rank.get("observations"), f"receipt rank {expected_rank} observations"
        )
        lora = _mapping(observations.get("lora"), f"receipt rank {expected_rank} LoRA")
        freeze = _mapping(observations.get("freeze"), f"receipt rank {expected_rank} freeze")
        reference = _mapping(
            observations.get("reference"), f"receipt rank {expected_rank} reference"
        )
        logs = _mapping(observations.get("logs"), f"receipt rank {expected_rank} logs")
        memory = _mapping(observations.get("memory"), f"receipt rank {expected_rank} memory")
        dataloader = _mapping(
            observations.get("dataloader"),
            f"receipt rank {expected_rank} dataloader",
        )
        accumulation = _mapping(
            observations.get("gradient_accumulation"),
            f"receipt rank {expected_rank} gradient accumulation",
        )
        final_digest = _sha256(lora.get("final_digest"), f"rank {expected_rank} final LoRA digest")
        _require(
            observations.get("global_step") == EXPECTED_MAX_STEPS[stage]
            and lora.get("changed") is True
            and freeze.get("all_versions_unchanged") is True
            and freeze.get("version_changes") == []
            and int(reference.get("context_count", 0)) > 0
            and logs.get("nonfinite") == []
            and observations.get("violations") == []
            and float(memory.get("minimum_observed_device_free_fraction", -1.0))
            >= MIN_FREE_MEMORY_FRACTION,
            f"receipt distributed rank {expected_rank} success invariants drifted",
        )
        accumulation_records = accumulation.get("observations")
        _require(
            dataloader.get("configured_per_device_batch") == expected_train_batch
            and isinstance(accumulation_records, list)
            and accumulation.get("training_step_calls")
            == len(accumulation_records)
            and all(
                isinstance(item, Mapping)
                and item.get("configured_gradient_accumulation_steps")
                == expected_gradient_accumulation
                and isinstance(item.get("local_pair_batch_size"), int)
                and not isinstance(item.get("local_pair_batch_size"), bool)
                and 0 < item.get("local_pair_batch_size") <= expected_train_batch
                for item in accumulation_records
            ),
            f"receipt distributed rank {expected_rank} batch/GA evidence drifted",
        )
        if stage == "mechanism_5step":
            probes = _mapping(
                observations.get("probes"),
                f"receipt rank {expected_rank} mechanism probes",
            )
            pre_probes = probes.get("pre_train")
            post_probes = probes.get("post_train")
            _require(
                isinstance(pre_probes, list)
                and len(pre_probes) == 4
                and all(isinstance(item, Mapping) for item in pre_probes),
                f"receipt rank {expected_rank} mechanism pre-probe inventory drifted",
            )
            _require(
                isinstance(post_probes, list)
                and len(post_probes) == 4
                and all(isinstance(item, Mapping) for item in post_probes),
                f"receipt rank {expected_rank} mechanism post-probe inventory drifted",
            )
            _require(
                [item.get("pair_id") for item in pre_probes]
                == list(MECHANISM_PAIR_IDS)
                and [item.get("pair_id") for item in post_probes]
                == list(MECHANISM_PAIR_IDS)
                and len(set(MECHANISM_PAIR_IDS)) == 4,
                f"receipt rank {expected_rank} canonical mechanism pair IDs drifted",
            )
            for dataset_index, (before, after) in enumerate(
                zip(pre_probes, post_probes)
            ):
                input_sha = _sha256(
                    before.get("input_ids_sha256"),
                    f"receipt rank {expected_rank} mechanism input hash",
                )
                labels_sha = _sha256(
                    before.get("labels_sha256"),
                    f"receipt rank {expected_rank} mechanism labels hash",
                )
                input_shape = before.get("input_shape")
                labels_shape = before.get("labels_shape")
                _require(
                    before.get("dataset_index") == dataset_index
                    and after.get("dataset_index") == dataset_index
                    and input_sha == after.get("input_ids_sha256")
                    and labels_sha == after.get("labels_sha256")
                    and isinstance(input_shape, list)
                    and len(input_shape) == 2
                    and input_shape[0] == 2
                    and isinstance(input_shape[1], int)
                    and not isinstance(input_shape[1], bool)
                    and 0 < input_shape[1] <= 512
                    and labels_shape == input_shape
                    and after.get("input_shape") == input_shape
                    and after.get("labels_shape") == labels_shape,
                    f"receipt rank {expected_rank} mechanism probe tensors drifted",
                )
                for probe, phase in ((before, "pre_train"), (after, "post_train")):
                    _require(
                        probe.get("phase") == phase,
                        f"receipt rank {expected_rank} mechanism probe phase drifted",
                    )
                    for key in (
                        "policy_chosen_logp",
                        "policy_rejected_logp",
                        "reference_chosen_logp",
                        "reference_rejected_logp",
                        "chosen_reward_margin",
                    ):
                        number = probe.get(key)
                        _require(
                            isinstance(number, (int, float))
                            and not isinstance(number, bool)
                            and math.isfinite(float(number)),
                            f"receipt rank {expected_rank} mechanism {key} is non-finite",
                        )
                    expected_margin = 0.1 * (
                        (
                            float(probe["policy_chosen_logp"])
                            - float(probe["reference_chosen_logp"])
                        )
                        - (
                            float(probe["policy_rejected_logp"])
                            - float(probe["reference_rejected_logp"])
                        )
                    )
                    margin = float(probe["chosen_reward_margin"])
                    _require(
                        abs(margin - expected_margin)
                        <= 1e-12 * max(1.0, abs(margin), abs(expected_margin)),
                        f"receipt rank {expected_rank} mechanism margin arithmetic drifted",
                    )
            improvements = [
                float(after["chosen_reward_margin"])
                - float(before["chosen_reward_margin"])
                for before, after in zip(pre_probes, post_probes)
            ]
            _require(
                math.fsum(improvements) / len(improvements) > 0.0
                and sum(value > 0.0 for value in improvements) >= 3,
                f"receipt rank {expected_rank} mechanism direction gate did not pass",
            )
            expected_local_pair_ids = [
                pre_probes[index].get("pair_id")
                for index in ((0, 2) if expected_rank == 0 else (1, 3))
            ] * 5
            probe_by_pair = {
                item.get("pair_id"): item
                for item in pre_probes
                if isinstance(item, Mapping)
            }
            _require(
                dataloader.get("batches_per_rank") == expected_batches_per_rank
                and accumulation.get("model_accepts_loss_kwargs") is False
                and accumulation.get("compute_loss_func_is_none") is True
                and accumulation.get("training_step_calls") == 10
                and all(
                    item.get("current_gradient_accumulation_steps") == 2
                    for item in accumulation_records
                )
                and [item.get("global_step_before") for item in accumulation_records]
                == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
                f"receipt distributed rank {expected_rank} mechanism remainder semantics drifted",
            )
            _require(
                [item.get("pair_id") for item in accumulation_records]
                == expected_local_pair_ids
                and all(
                    item.get("pair_id") in probe_by_pair
                    and item.get("input_ids_sha256")
                    == probe_by_pair[item.get("pair_id")].get(
                        "input_ids_sha256"
                    )
                    and item.get("labels_sha256")
                    == probe_by_pair[item.get("pair_id")].get("labels_sha256")
                    for item in accumulation_records
                ),
                f"receipt distributed rank {expected_rank} mechanism pair exposure drifted",
            )
        process_keys.add(
            (
                rank_process["boot_id"],
                rank_process["pid"],
                rank_process["proc_start_ticks"],
            )
        )
        final_digests.add(final_digest)
        validated_ranks.append(dict(rank))
    _require(len(process_keys) == EXPECTED_WORLD_SIZE, "stage receipt rank processes are not distinct")
    _require(len(final_digests) == 1, "stage receipt rank LoRA digests differ")
    _require(
        distributed.get("realized_global_batch_profile")
        == realized_global_batch_profile(validated_ranks, stage),
        "stage receipt realized global batch profile drifted",
    )
    _require(
        process == validated_ranks[0]["process_identity"],
        "stage receipt common process is not rank zero",
    )
    _require(
        receipt.get("observations") == validated_ranks[0]["observations"]
        and receipt.get("evidence") == receipt.get("observations"),
        "stage receipt common evidence is not rank-zero evidence",
    )
    return dict(receipt)


def _verify_cpu_validator(cpu: Mapping[str, Any]) -> None:
    validator = _mapping(
        cpu.get("independent_validator"), "binding.cpu_contract.independent_validator"
    )
    path = _absolute_path(
        validator.get("path"), "cpu_contract.independent_validator.path", must_exist=True
    )
    _require(
        file_sha256(path)
        == _sha256(validator.get("file_sha256"), "independent_validator.file_sha256"),
        "independent CPU validator source drifted",
    )
    completed = subprocess.run(
        [sys.executable, str(path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _require(
        completed.returncode == 0,
        "independent CPU validator failed before GPU execution: "
        + completed.stderr.strip(),
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    _require(lines, "independent CPU validator emitted no result")
    try:
        result = json.loads(lines[-1])
    except json.JSONDecodeError as error:
        raise Day23GPUStageError("independent CPU validator result is not JSON") from error
    _require(
        isinstance(result, Mapping)
        and result.get("status") == "valid_cpu_ready_gpu_pending"
        and result.get("cpu_ready") is True
        and result.get("gpu_optimizer_ready") is False,
        "independent CPU validator boundary drifted",
    )
    _require(
        object_sha256(result)
        == _sha256(validator.get("result_sha256"), "independent_validator.result_sha256"),
        "independent CPU validator result drifted from the binding",
    )


def _verify_remote_parent_payload(remote_parent: Mapping[str, Any]) -> Path:
    merged = _mapping(remote_parent.get("merged_export"), "binding.remote_parent.merged_export")
    root = _absolute_path(
        merged.get("path"), "binding.remote_parent.merged_export.path", must_exist=True
    )
    _require(root.is_dir() and not root.is_symlink(), "merged S1 export is not a regular directory")
    expected = _mapping(merged.get("files"), "binding.remote_parent.merged_export.files")
    actual_names: list[str] = []
    for child in sorted(root.rglob("*")):
        _require(not child.is_symlink(), f"merged S1 export contains a symlink: {child}")
        if child.is_dir():
            continue
        _require(child.is_file(), f"merged S1 export contains a non-file: {child}")
        relative = child.relative_to(root).as_posix()
        if relative == "S1-EXPORT-MANIFEST.json":
            continue
        actual_names.append(relative)
    _require(actual_names == sorted(expected), "merged S1 export file inventory drifted")
    actual: dict[str, dict[str, Any]] = {}
    for relative in actual_names:
        child = root / relative
        identity = _mapping(expected[relative], f"remote_parent.merged_export.files.{relative}")
        size = child.stat().st_size
        digest = file_sha256(child)
        _require(size == identity.get("bytes"), f"merged S1 byte count drifted: {relative}")
        _require(
            digest == _sha256(identity.get("sha256"), f"merged S1 {relative}.sha256"),
            f"merged S1 file hash drifted: {relative}",
        )
        actual[relative] = {"bytes": size, "sha256": digest}
    _require(
        object_sha256(actual)
        == _sha256(merged.get("files_sha256"), "remote_parent.merged_export.files_sha256"),
        "merged S1 aggregate hash drifted",
    )
    _require(
        Path(_text(remote_parent.get("model_argument"), "remote_parent.model_argument")).resolve()
        == root,
        "remote parent model argument differs from the verified merged export",
    )
    export_manifest = _mapping(
        remote_parent.get("merged_export_manifest"),
        "binding.remote_parent.merged_export_manifest",
    )
    manifest_path = _absolute_path(
        export_manifest.get("path"),
        "binding.remote_parent.merged_export_manifest.path",
        must_exist=True,
    )
    _require(
        file_sha256(manifest_path)
        == _sha256(export_manifest.get("file_sha256"), "merged_export_manifest.file_sha256"),
        "merged S1 export manifest file hash drifted",
    )
    return root


def _resolve_previous_receipt(
    binding: Mapping[str, Any], stage: str, binding_sha256: str
) -> dict[str, Any]:
    if stage == "g2_one_step":
        return {
            "kind": "binding",
            "binding_sha256": binding_sha256,
            "previous_gate_receipt_sha256": binding_sha256,
        }
    previous_stage = (
        "g2_one_step" if stage == "mechanism_5step" else "mechanism_5step"
    )
    previous_success, _ = _receipt_paths(binding, previous_stage)
    _require(previous_success.is_file(), f"previous success receipt is missing: {previous_success}")
    previous = load_json(previous_success, f"{previous_stage} success receipt")
    previous_sha = verify_self_hash(
        previous, "receipt_sha256", f"{previous_stage} success receipt"
    )
    previous = validate_stage_success_receipt_value(
        binding,
        previous,
        stage=previous_stage,
        binding_sha256=binding_sha256,
    )
    evidence = _mapping(binding.get("evidence_contract"), "binding.evidence_contract")
    reload_map = _mapping(
        evidence.get("checkpoint_receipts"),
        "binding.evidence_contract.checkpoint_receipts",
    )
    previous_reload_map = _mapping(
        reload_map.get(previous_stage), f"reload receipts for {previous_stage}"
    )
    previous_step = 1 if previous_stage == "g2_one_step" else 5
    reload_path = _absolute_path(
        previous_reload_map.get(str(previous_step)),
        f"{previous_stage} fresh-process reload receipt",
        must_exist=True,
    )
    import eval_day23_qwen35_preferences as preference_evaluator

    binding_manifest = _absolute_path(
        _mapping(binding.get("paths"), "binding.paths").get("binding_manifest"),
        "binding manifest",
        must_exist=True,
    )
    previous_split = "g2" if previous_stage == "g2_one_step" else "mechanism"
    try:
        evaluator_context = preference_evaluator.validate_binding(
            binding_manifest, split=previous_split
        )
        validated_reload = preference_evaluator.validate_bound_evaluation_receipt(
            evaluator_context,
            reload_path,
            split=previous_split,
            checkpoint_step=previous_step,
        )
    except preference_evaluator.Day23PreferenceEvaluationError as error:
        raise Day23GPUStageError(
            f"{previous_stage} strict fresh-process reload validation failed: {error}"
        ) from error
    reload_value = validated_reload["value"]
    reload_sha = validated_reload["evaluation_sha256"]
    checkpoint = _mapping(reload_value.get("checkpoint"), "reload receipt checkpoint")
    _require(
        Path(_text(checkpoint.get("receipt_path"), "reload source receipt path")).resolve()
        == previous_success
        and checkpoint.get("receipt_sha256") == previous_sha,
        f"{previous_stage} reload source receipt drifted",
    )
    return {
        "kind": "stage_success_receipt",
        "stage": previous_stage,
        "path": str(previous_success),
        "file_sha256": file_sha256(previous_success),
        "previous_gate_receipt_sha256": previous_sha,
        "fresh_process_reload_receipt": {
            "path": str(reload_path),
            "file_sha256": file_sha256(reload_path),
            "evaluation_sha256": reload_sha,
            "checkpoint_step": previous_step,
        },
    }


def validate_binding(
    binding_path: Path, stage: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding_path = binding_path.expanduser().resolve(strict=True)
    binding = load_json(binding_path, "GPU binding")
    binding_sha = verify_self_hash(binding, "binding_sha256", "GPU binding")
    _require(binding.get("schema_name") == BINDING_SCHEMA, "GPU binding schema drifted")
    _require(binding.get("schema_version") == 1, "GPU binding version drifted")
    _require(binding.get("status") == BINDING_STATUS, "GPU binding is not runtime-pending")
    _require(stage in STAGES, f"unsupported stage: {stage}")

    producer = _producer_identity(binding)
    runner_path = _absolute_path(
        producer.get("path"), "evidence_contract.producers.stage_runner.path", must_exist=True
    )
    this_path = Path(__file__).resolve(strict=True)
    _require(runner_path == this_path, "binding points to a different stage runner")
    runner_sha = file_sha256(this_path)
    _require(
        runner_sha == _sha256(producer.get("file_sha256"), "stage_runner.file_sha256"),
        "stage runner file hash drifted",
    )

    cpu = _mapping(binding.get("cpu_contract"), "binding.cpu_contract")
    cpu_path = _absolute_path(cpu.get("path"), "binding.cpu_contract.path", must_exist=True)
    cpu_file_sha = file_sha256(cpu_path)
    _require(
        cpu_file_sha == _sha256(cpu.get("file_sha256"), "cpu_contract.file_sha256"),
        "CPU contract file hash drifted",
    )
    cpu_value = load_json(cpu_path, "CPU contract")
    cpu_content_sha = verify_self_hash(cpu_value, "contract_sha256", "CPU contract")
    _require(
        cpu_content_sha == _sha256(cpu.get("contract_sha256"), "cpu_contract.contract_sha256"),
        "CPU contract content hash drifted",
    )
    _require(cpu_value.get("status") == "cpu_ready_gpu_pending", "CPU contract is not ready")
    _verify_cpu_validator(cpu)

    stages = _mapping(binding.get("stages"), "binding.stages")
    stage_binding = _mapping(stages.get(stage), f"binding.stages.{stage}")
    _require(
        stage_binding.get("fresh_start_from_remote_parent") is True,
        f"{stage} is not a fresh start from the remote parent",
    )
    forbidden = set(stage_binding.get("forbidden_config_keys", []))
    _require(FORBIDDEN_CONFIG_KEYS <= forbidden, f"{stage} forbidden inputs are incomplete")
    executable = _mapping(
        stage_binding.get("executable_config"), f"binding.stages.{stage}.executable_config"
    )
    config_path = _absolute_path(
        executable.get("path"), f"binding.stages.{stage}.executable_config.path", must_exist=True
    )
    config_file_sha = file_sha256(config_path)
    _require(
        config_file_sha
        == _sha256(executable.get("file_sha256"), f"{stage}.executable_config.file_sha256"),
        f"{stage} executable config hash drifted",
    )
    config = load_json(config_path, f"{stage} executable config")
    g2_binding = _mapping(stages.get("g2_one_step"), "binding.stages.g2_one_step")
    mechanism_binding = _mapping(
        stages.get("mechanism_5step"), "binding.stages.mechanism_5step"
    )
    g2_config_entry = _mapping(
        g2_binding.get("executable_config"), "bound G2 executable config"
    )
    mechanism_config_entry = _mapping(
        mechanism_binding.get("executable_config"),
        "bound mechanism executable config",
    )
    g2_config_path = _absolute_path(
        g2_config_entry.get("path"), "bound G2 config", must_exist=True
    )
    mechanism_config_path = _absolute_path(
        mechanism_config_entry.get("path"),
        "bound mechanism config",
        must_exist=True,
    )
    _require(
        file_sha256(g2_config_path) == g2_config_entry.get("file_sha256")
        and file_sha256(mechanism_config_path)
        == mechanism_config_entry.get("file_sha256"),
        "G2/mechanism config identity drifted",
    )
    g2_config = load_json(g2_config_path, "bound G2 config")
    mechanism_config = load_json(
        mechanism_config_path, "bound mechanism config"
    )
    actual_derivation_keys = sorted(
        key
        for key in set(g2_config) | set(mechanism_config)
        if g2_config.get(key) != mechanism_config.get(key)
    )
    derivation = _mapping(
        g2_binding.get("authorized_derivation"), "G2 authorized derivation"
    )
    _require(
        actual_derivation_keys
        == [
            "dataset",
            "gradient_accumulation_steps",
            "max_steps",
            "output_dir",
            "per_device_train_batch_size",
            "save_steps",
            "warmup_ratio",
        ]
        and derivation.get("only_changed_config_keys")
        == actual_derivation_keys
        and derivation.get("dataset_sample_count") == {"from": 4, "to": 8}
        and derivation.get("per_device_train_batch_size")
        == {"from": 1, "to": 4}
        and derivation.get("gradient_accumulation_steps")
        == {"from": 4, "to": 1}
        and derivation.get("max_steps") == {"from": 5, "to": 1}
        and derivation.get("save_steps") == {"from": 5, "to": 1}
        and _mapping(
            derivation.get("warmup_ratio"), "G2 warmup derivation"
        ).get("from")
        == 0.1
        and derivation["warmup_ratio"].get("to") == 0.0
        and derivation.get(
            "all_other_keys_and_values_byte_equivalent_after_output_normalization"
        )
        is True,
        "G2 authorized seven-field derivation drifted",
    )
    expected_keys = executable.get("keys")
    _require(isinstance(expected_keys, list), f"{stage} executable key inventory is absent")
    _require(sorted(config) == sorted(expected_keys), f"{stage} executable key inventory drifted")
    _require(not (set(config) & FORBIDDEN_CONFIG_KEYS), f"{stage} supplies forbidden parent/ref inputs")
    _require(not (set(config) & FORBIDDEN_TOPOLOGY_KEYS), f"{stage} supplies forbidden topology shortcuts")
    _require(all(value is not None for value in config.values()), f"{stage} config contains null")
    _require(
        all(not isinstance(value, list) or bool(value) for value in config.values()),
        f"{stage} config contains an empty list",
    )
    _require("__BIND_" not in json.dumps(config, sort_keys=True), f"{stage} has unresolved placeholders")
    _require(config.get("max_steps") == EXPECTED_MAX_STEPS[stage], f"{stage} max_steps drifted")
    _require(config.get("add_version") is False, f"{stage} output versioning must be disabled")
    _require(config.get("tuner_type") == "lora", f"{stage} must use LoRA")
    _require(config.get("rlhf_type") == "dpo", f"{stage} must use DPO")
    _require(config.get("model_type") == "qwen3_5", f"{stage} model type drifted")
    _require(config.get("save_only_model") is False, f"{stage} must save optimizer state")
    _require(config.get("load_best_model_at_end") is False, f"{stage} cannot auto-load best")
    _require(config.get("bf16") is True, f"{stage} must remain bf16")
    _require(config.get("packing") is False and config.get("padding_free") is False, f"{stage} packing drifted")
    _require(config.get("freeze_vit") is True, f"{stage} vision must be frozen")
    _require(config.get("freeze_aligner") is True, f"{stage} aligner must be frozen")
    expected_train_batch = (
        EXPECTED_MECHANISM_PER_DEVICE_TRAIN_BATCH
        if stage == "mechanism_5step"
        else EXPECTED_PER_DEVICE_TRAIN_BATCH
    )
    expected_gradient_accumulation = (
        EXPECTED_MECHANISM_GRADIENT_ACCUMULATION
        if stage == "mechanism_5step"
        else EXPECTED_GRADIENT_ACCUMULATION
    )
    _require(
        config.get("per_device_train_batch_size") == expected_train_batch
        and config.get("per_device_eval_batch_size") == EXPECTED_PER_DEVICE_EVAL_BATCH
        and config.get("gradient_accumulation_steps")
        == expected_gradient_accumulation,
        f"{stage} dual-GPU batch amendment drifted",
    )
    amendment = _mapping(
        binding.get("user_authorized_gpu_amendment"),
        "binding.user_authorized_gpu_amendment",
    )
    stage_batches = _mapping(amendment.get("stage_train_batches"), "GPU amendment stage batches")
    stage_batch = _mapping(stage_batches.get(stage), f"GPU amendment batch for {stage}")
    expected_realized_global_range = (
        [4, 4]
        if stage == "mechanism_5step"
        else ([2, 8] if stage == "bounded_smoke_30step" else [8, 8])
    )
    _require(
        amendment.get("world_size") == EXPECTED_WORLD_SIZE
        and stage_batch.get("per_device") == expected_train_batch
        and stage_batch.get("gradient_accumulation_steps")
        == expected_gradient_accumulation
        and stage_batch.get("nominal_global")
        == EXPECTED_WORLD_SIZE
        * expected_train_batch
        * expected_gradient_accumulation
        and stage_batch.get("realized_global_range")
        == expected_realized_global_range
        and amendment.get("per_device_eval_batch_size")
        == EXPECTED_PER_DEVICE_EVAL_BATCH
        and amendment.get("bounded_smoke_nominal_global_train_batch_size") == 8
        and amendment.get("distributed_sampler_padding")
        == "must_be_attested_in_runtime_receipt",
        "bound user-authorized GPU amendment drifted",
    )
    parsed_stage = _mapping(
        _mapping(
            _mapping(binding.get("runtime_parse"), "binding.runtime_parse").get(
                "stages"
            ),
            "binding.runtime_parse.stages",
        ).get(stage),
        f"binding.runtime_parse.stages.{stage}",
    )
    _require(
        parsed_stage.get("status") == "pass"
        and parsed_stage.get("per_device_train_batch_size")
        == expected_train_batch
        and parsed_stage.get("gradient_accumulation_steps")
        == expected_gradient_accumulation
        and parsed_stage.get("configured_nominal_global_train_batch_size")
        == EXPECTED_WORLD_SIZE
        * expected_train_batch
        * expected_gradient_accumulation
        and parsed_stage.get("expected_realized_global_train_batch_size_range")
        == expected_realized_global_range,
        f"{stage} bound runtime parse batch semantics drifted",
    )
    correction = _mapping(
        amendment.get("mechanism_protocol_correction"),
        "mechanism protocol correction",
    )
    prior = _mapping(correction.get("prior_attempt"), "prior mechanism attempt")
    prior_path = _absolute_path(
        prior.get("path"), "prior mechanism failure receipt", must_exist=True
    )
    _require(
        prior_path == EXPECTED_PRIOR_MECHANISM_FAILURE_PATH,
        "prior mechanism failure receipt is not at its canonical sealed path",
    )
    prior_value = load_json(prior_path, "prior mechanism failure receipt")
    prior_receipt_sha = verify_self_hash(
        prior_value,
        "receipt_sha256",
        "prior mechanism failure receipt",
    )
    _require(
        correction.get("attempt_number") == 2
        and correction.get("append_only") is True
        and correction.get("one_shot") is True
        and correction.get("third_attempt_forbidden") is True
        and prior.get("receipt_sha256")
        == "600341cb77ee2bd97b2e4a00ae26a89720a9204b8266b86b6571c6c3181adc20"
        and prior_receipt_sha == prior.get("receipt_sha256")
        and file_sha256(prior_path) == prior.get("file_sha256")
        and prior_value.get("status") == "fail"
        and prior_value.get("stage") == "mechanism_5step"
        and prior.get("status") == "fail"
        and prior.get("candidate_or_resume_source") is False,
        "mechanism retry provenance drifted",
    )
    claim_spec = _mapping(
        correction.get("execution_claim"), "mechanism correction execution claim"
    )
    claim_path = _absolute_path(
        claim_spec.get("path"),
        "mechanism correction execution claim path",
        must_exist=stage == "bounded_smoke_30step",
    )
    _require(
        claim_path
        == (prior_path.parent / "topology-correction-attempt2-claim.json").resolve()
        and claim_spec.get("schema_name") == CORRECTION_CLAIM_SCHEMA
        and claim_spec.get("schema_version") == 1
        and claim_spec.get("self_hash_field") == "claim_sha256"
        and claim_spec.get("exclusive_create_no_overwrite") is True,
        "mechanism correction execution claim contract drifted",
    )
    correction_claim: dict[str, Any] | None = None
    if stage in {"g2_one_step", "mechanism_5step"}:
        _require(
            not claim_path.exists() and not claim_path.is_symlink(),
            "the one-shot mechanism correction has already been claimed",
        )
    else:
        correction_claim = _validate_mechanism_correction_claim(
            claim_path,
            binding_sha256=binding_sha,
            config_file_sha256=_mapping(
                mechanism_binding.get("executable_config"),
                "bound mechanism executable config",
            ).get("file_sha256"),
            prior_receipt_sha256=prior_receipt_sha,
        )
    if stage == "mechanism_5step":
        _require(
            stage_batch.get("realized_global_range") == [4, 4]
            and stage_batch.get("batches_per_rank") == 2
            and stage_batch.get("microbatches_per_optimizer_step") == 2
            and stage_batch.get("partial_epoch_flush") is True,
            "mechanism realized batch contract drifted",
        )

    for key in ("model", "output_dir"):
        _absolute_path(config.get(key), f"config.{key}", must_exist=(key == "model"))
    for key in ("dataset", "val_dataset", "external_plugins"):
        if key not in config:
            continue
        _require(isinstance(config[key], list), f"config.{key} must be a list")
        for index, raw in enumerate(config[key]):
            path_text = str(raw)
            if key == "dataset":
                path_text = re.sub(r"#\d+$", "", path_text)
            _absolute_path(path_text, f"config.{key}[{index}]", must_exist=True)

    output_dir = _absolute_path(config["output_dir"], "config.output_dir", must_exist=False)
    _require(not output_dir.exists(), f"fresh output_dir already exists: {output_dir}")
    bound_output = stage_binding.get("output_dir")
    if bound_output is not None:
        _require(
            output_dir == _absolute_path(bound_output, f"binding.stages.{stage}.output_dir", must_exist=False),
            f"{stage} output_dir differs from binding",
        )
    if stage in {"g2_one_step", "mechanism_5step"}:
        _require("val_dataset" not in config, f"{stage} must not load dev")
        _require(config.get("dataset_shuffle") is False, f"{stage} dataset shuffle must be off")
        _require(
            config.get("train_dataloader_shuffle") is False,
            f"{stage} dataloader shuffle must be off",
        )
    heldout = cpu_value.get("dataset_usage", {}).get("heldout", {}).get("path")
    if isinstance(heldout, str):
        _require(heldout not in json.dumps(config, sort_keys=True), "heldout leaked into GPU config")

    remote_parent = _mapping(binding.get("remote_parent"), "binding.remote_parent")
    expected_model = _verify_remote_parent_payload(remote_parent)
    _require(
        Path(config["model"]).resolve() == expected_model,
        "config.model is not the bound promoted merged S1",
    )

    previous = _resolve_previous_receipt(binding, stage, binding_sha)
    success_receipt, failure_receipt = _receipt_paths(binding, stage)
    _require(not success_receipt.exists(), f"success receipt already exists: {success_receipt}")
    _require(not failure_receipt.exists(), f"failure receipt already exists: {failure_receipt}")
    context = {
        "binding_path": binding_path,
        "binding_file_sha256": file_sha256(binding_path),
        "binding_sha256": binding_sha,
        "runner_path": this_path,
        "runner_file_sha256": runner_sha,
        "cpu_contract_path": cpu_path,
        "cpu_contract_file_sha256": cpu_file_sha,
        "cpu_contract_sha256": cpu_content_sha,
        "stage_binding": stage_binding,
        "config_path": config_path,
        "config_file_sha256": config_file_sha,
        "config": config,
        "output_dir": output_dir,
        "success_receipt": success_receipt,
        "failure_receipt": failure_receipt,
        "previous": previous,
        "remote_parent_files_sha256": remote_parent["merged_export"]["files_sha256"],
        "mechanism_protocol_correction": dict(correction),
        "mechanism_correction_claim_path": claim_path,
        "mechanism_correction_claim": correction_claim,
    }
    return binding, context


def validate_runtime(binding: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _mapping(binding.get("runtime_parse"), "binding.runtime_parse")
    python_binding = _mapping(runtime.get("python_executable"), "runtime_parse.python_executable")
    expected_python = _absolute_path(
        python_binding.get("path"), "runtime_parse.python_executable.path", must_exist=True
    )
    actual_python = Path(sys.executable).resolve(strict=True)
    _require(actual_python == expected_python, "runner is using the wrong Python executable")
    python_sha = file_sha256(actual_python)
    _require(
        python_sha == _sha256(python_binding.get("file_sha256"), "python_executable.file_sha256"),
        "Python executable hash drifted",
    )
    _require(
        str(python_binding.get("version")) == platform.python_version(),
        "Python runtime version drifted",
    )

    checkout = _absolute_path(
        runtime.get("ms_swift_checkout"), "runtime_parse.ms_swift_checkout", must_exist=True
    )
    _require(checkout.is_dir(), "ms-swift checkout is not a directory")
    commit = _git(checkout, "rev-parse", "HEAD")
    expected_commit = _text(runtime.get("ms_swift_commit"), "runtime_parse.ms_swift_commit")
    _require(commit == expected_commit, "ms-swift checkout commit drifted")
    _require(_git(checkout, "status", "--porcelain") == "", "ms-swift checkout is dirty")
    _require(runtime.get("ms_swift_clean") is True, "binding did not require a clean checkout")

    versions_binding = _mapping(runtime.get("package_versions"), "runtime_parse.package_versions")
    package_names = {
        "ms-swift": _package_version("ms-swift"),
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
        "trl": _package_version("trl"),
        "peft": _package_version("peft"),
        "accelerate": _package_version("accelerate"),
        "datasets": _package_version("datasets"),
    }
    for name, actual in package_names.items():
        expected = versions_binding.get(name)
        if expected is None and name == "ms-swift":
            expected = versions_binding.get("ms_swift")
        _require(str(expected) == actual, f"runtime package version drifted: {name}")

    import swift

    swift_file = Path(swift.__file__).resolve(strict=True)
    import_root = _absolute_path(
        runtime.get("ms_swift_import_root"), "runtime_parse.ms_swift_import_root", must_exist=True
    )
    _require(swift_file.is_relative_to(import_root), "swift import escaped the pinned checkout")
    _require(swift_file.is_relative_to(checkout), "swift import is outside ms-swift checkout")
    return {
        "python": {
            "path": str(actual_python),
            "file_sha256": python_sha,
            "version": platform.python_version(),
        },
        "ms_swift_checkout": {
            "path": str(checkout),
            "commit": commit,
            "clean": True,
            "import_file": str(swift_file),
        },
        "package_versions": package_names,
    }


def _tensor_sha256(tensor: Any) -> str:
    import torch

    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(canonical_json(list(value.shape)))
    if value.numel():
        raw = value.view(torch.uint8).numpy()
        digest.update(memoryview(raw))
    return digest.hexdigest()


def _named_tensor_digest(named_tensors: Iterable[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(_tensor_sha256(tensor).encode("ascii") + b"\n")
    return digest.hexdigest()


def _finite_tensor(value: Any) -> bool:
    import torch

    return not isinstance(value, torch.Tensor) or bool(torch.isfinite(value).all().item())


def _parameter_entry(name: str, parameter: Any) -> dict[str, Any]:
    return {
        "name": name,
        "numel": int(parameter.numel()),
        "shape": list(parameter.shape),
        "dtype": str(parameter.dtype),
        "device": str(parameter.device),
        "requires_grad": bool(parameter.requires_grad),
    }


def _cuda_snapshot(label: str) -> dict[str, Any]:
    import torch

    devices: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        local_rank = os.environ.get("LOCAL_RANK")
        indices = (
            [int(local_rank)]
            if local_rank is not None and local_rank.isdigit()
            else list(range(torch.cuda.device_count()))
        )
        for index in indices:
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "allocated_bytes": int(torch.cuda.memory_allocated(index)),
                    "reserved_bytes": int(torch.cuda.memory_reserved(index)),
                    "max_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
                    "max_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
                    "free_bytes": int(free_bytes),
                    "total_bytes": int(total_bytes),
                    "free_fraction": float(free_bytes / total_bytes),
                }
            )
    return {"label": label, "at_utc": utc_now(), "devices": devices}


class RuntimeObserver:
    """State owned by observational wrappers around the pinned trainer."""

    def __init__(self, stage: str, config: Mapping[str, Any]) -> None:
        self.stage = stage
        self.config = dict(config)
        self.trainer: Any = None
        self.prepared_model: Any = None
        self.trainable_parameters: list[dict[str, Any]] = []
        self.frozen_summary: dict[str, Any] = {}
        self.frozen_versions: dict[int, tuple[str, int, Any]] = {}
        self.lora_initial_digest: str | None = None
        self.lora_pre_step_digest: str | None = None
        self.lora_final_digest: str | None = None
        self.optimizer: dict[str, Any] | None = None
        self.reference_depth = 0
        self.reference_context_count = 0
        self.reference_events: list[dict[str, Any]] = []
        self.forward_counts = {"policy": 0, "reference": 0}
        self.forward_events: list[dict[str, Any]] = []
        self.forward_memory_seen: set[str] = set()
        self.logs: list[dict[str, Any]] = []
        self.nonfinite: list[str] = []
        self.memory: list[dict[str, Any]] = []
        self.pre_train_probes: list[dict[str, Any]] = []
        self.post_train_probes: list[dict[str, Any]] = []
        self.result_summary: Any = None
        self.violations: list[str] = []
        self.dataloader: dict[str, Any] | None = None
        self.model_accepts_loss_kwargs: bool | None = None
        self.compute_loss_func_is_none: bool | None = None
        self.gradient_accumulation_observations: list[dict[str, Any]] = []

    def snapshot_memory(self, label: str) -> None:
        self.memory.append(_cuda_snapshot(label))

    @staticmethod
    def _lora_named_parameters(model: Any) -> list[tuple[str, Any]]:
        return [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if ".lora_A." in name or ".lora_B." in name
        ]

    def inspect_prepared_model(self, model: Any, args: Any) -> None:
        from peft import PeftModel

        _require(isinstance(model, PeftModel), "prepared DPO policy is not a PEFT model")
        _require(getattr(args, "ref_model", None) is None, "ref_model must remain None")
        _require(not getattr(args, "ref_adapters", []), "ref_adapters must remain empty")
        _require(not getattr(args, "adapters", []), "policy adapters must start empty")
        _require(
            getattr(args, "resume_from_checkpoint", None) is None,
            "DPO policy unexpectedly resumes a checkpoint",
        )
        target_regex = _text(self.config.get("target_regex"), "config.target_regex")
        pattern = re.compile(target_regex)
        named_parameters = list(model.named_parameters())
        trainable = [(name, parameter) for name, parameter in named_parameters if parameter.requires_grad]
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        _require(local_rank in (0, 1), "DDP LOCAL_RANK is missing during model preparation")
        expected_device = f"cuda:{local_rank}"
        _require(
            {str(parameter.device) for _, parameter in named_parameters}
            == {expected_device},
            "prepared policy parameters are not confined to the local DDP GPU",
        )
        lora = self._lora_named_parameters(model)
        _require(len(lora) == EXPECTED_LORA_TENSORS, "fresh LoRA tensor count drifted")
        _require(len(trainable) == len(lora), "a non-LoRA parameter is trainable")
        _require(
            sum(parameter.numel() for _, parameter in trainable) == EXPECTED_LORA_PARAMS,
            "fresh LoRA trainable parameter count drifted",
        )
        module_counts: dict[str, set[str]] = {}
        for name, parameter in trainable:
            match = re.match(r"^(.*)\.lora_([AB])\.[^.]+\.weight$", name)
            _require(match is not None, f"unexpected trainable parameter: {name}")
            module_name, side = match.groups()
            _require(pattern.fullmatch(module_name) is not None, f"trainable target escaped regex: {name}")
            module_counts.setdefault(module_name, set()).add(side)
            lowered = name.lower()
            _require(
                not any(token in lowered for token in (".visual.", ".vision.", ".aligner.", ".merger.", "embed_tokens", "lm_head")),
                f"forbidden multimodal/embed/head trainable: {name}",
            )
            _require(parameter.is_floating_point(), f"LoRA tensor is not floating point: {name}")
        _require(len(module_counts) == EXPECTED_TARGET_MODULES, "LoRA target module count drifted")
        _require(all(sides == {"A", "B"} for sides in module_counts.values()), "LoRA A/B inventory is incomplete")

        frozen_categories: dict[str, dict[str, int]] = {
            "vision": {"tensors": 0, "params": 0},
            "aligner": {"tensors": 0, "params": 0},
            "embedding": {"tensors": 0, "params": 0},
            "lm_head": {"tensors": 0, "params": 0},
        }
        frozen_tensors = 0
        frozen_params = 0
        for name, parameter in named_parameters:
            if parameter.requires_grad:
                continue
            frozen_tensors += 1
            frozen_params += int(parameter.numel())
            lowered = name.lower()
            categories: list[str] = []
            if ".visual." in lowered or ".vision." in lowered:
                categories.append("vision")
            if ".aligner." in lowered or ".merger." in lowered:
                categories.append("aligner")
            if "embed_tokens" in lowered or ".embedding" in lowered:
                categories.append("embedding")
            if "lm_head" in lowered:
                categories.append("lm_head")
            for category in categories:
                frozen_categories[category]["tensors"] += 1
                frozen_categories[category]["params"] += int(parameter.numel())
        _require(frozen_categories["vision"]["tensors"] > 0, "vision freeze inventory is empty")
        _require(frozen_categories["aligner"]["tensors"] > 0, "aligner freeze inventory is empty")
        _require(frozen_categories["embedding"]["tensors"] > 0, "embedding freeze inventory is empty")

        self.prepared_model = model
        self.trainable_parameters = [_parameter_entry(name, parameter) for name, parameter in trainable]
        self.frozen_summary = {
            "tensors": frozen_tensors,
            "params": frozen_params,
            "categories": frozen_categories,
            "lm_head_note": (
                "separate_frozen_parameter"
                if frozen_categories["lm_head"]["tensors"]
                else "no_separate_parameter_tied_or_absent"
            ),
        }
        self.lora_initial_digest = _named_tensor_digest(lora)
        self.snapshot_memory("after_fresh_lora_prepare")

    def capture_optimizer(self, trainer: Any) -> None:
        optimizer = trainer.optimizer
        _require(optimizer is not None, "trainer did not create an optimizer")
        named = list(trainer.model.named_parameters())
        by_id = {id(parameter): (name, parameter) for name, parameter in named}
        expected_ids = {id(parameter) for _, parameter in named if parameter.requires_grad}
        actual_ids: list[int] = []
        groups: list[dict[str, Any]] = []
        for index, group in enumerate(optimizer.param_groups):
            ids = [id(parameter) for parameter in group["params"]]
            actual_ids.extend(ids)
            names: list[str] = []
            for parameter_id in ids:
                _require(parameter_id in by_id, "optimizer contains an unknown parameter")
                names.append(by_id[parameter_id][0])
            hyperparameters = {
                key: json_safe(value)
                for key, value in group.items()
                if key != "params"
            }
            groups.append(
                {
                    "index": index,
                    "parameter_names": sorted(names),
                    "parameter_count": len(names),
                    "numel": sum(by_id[item][1].numel() for item in ids),
                    "hyperparameters": hyperparameters,
                }
            )
        _require(len(actual_ids) == len(set(actual_ids)), "optimizer repeats a parameter")
        _require(set(actual_ids) == expected_ids, "optimizer inventory differs from trainable inventory")
        _require(len(actual_ids) == EXPECTED_LORA_TENSORS, "optimizer LoRA tensor count drifted")
        _require(
            sum(by_id[item][1].numel() for item in actual_ids) == EXPECTED_LORA_PARAMS,
            "optimizer LoRA parameter count drifted",
        )
        self.optimizer = {
            "class": f"{optimizer.__class__.__module__}.{optimizer.__class__.__qualname__}",
            "parameter_tensors": len(actual_ids),
            "parameter_numel": sum(by_id[item][1].numel() for item in actual_ids),
            "groups": groups,
            "exactly_trainable_inventory": True,
        }
        self.snapshot_memory("after_optimizer_create")

    def capture_post_ddp_baseline(self, trainer: Any) -> None:
        if self.frozen_versions:
            return
        model = trainer.accelerator.unwrap_model(trainer.model)
        named = list(model.named_parameters())
        self.frozen_versions = {
            id(parameter): (name, int(parameter._version), parameter)
            for name, parameter in named
            if not parameter.requires_grad
        }
        self.lora_pre_step_digest = _named_tensor_digest(
            self._lora_named_parameters(model)
        )
        _require(self.frozen_versions, "post-DDP frozen baseline is empty")
        self.snapshot_memory("after_ddp_sync_before_first_training_step")

    def observe_training_step(
        self, trainer: Any, inputs: Mapping[str, Any]
    ) -> None:
        input_ids = inputs.get("input_ids")
        _require(
            hasattr(input_ids, "shape")
            and len(input_ids.shape) == 2
            and int(input_ids.shape[0]) > 0
            and int(input_ids.shape[0]) % 2 == 0,
            "training_step concatenated batch shape drifted",
        )
        local_pair_batch_size = int(input_ids.shape[0]) // 2
        input_sha = (
            _tensor_sha256(input_ids) if "input_ids" in inputs else None
        )
        labels_sha = (
            _tensor_sha256(inputs["labels"]) if "labels" in inputs else None
        )
        pair_id = None
        if self.stage == "mechanism_5step":
            matches = [
                probe["pair_id"]
                for probe in self.pre_train_probes
                if probe["input_ids_sha256"] == input_sha
                and probe["labels_sha256"] == labels_sha
            ]
            _require(
                len(matches) == 1,
                "mechanism training microbatch does not match exactly one frozen pair",
            )
            pair_id = matches[0]
        self.gradient_accumulation_observations.append(
            {
                "global_step_before": int(trainer.state.global_step),
                "configured_gradient_accumulation_steps": int(
                    trainer.args.gradient_accumulation_steps
                ),
                "current_gradient_accumulation_steps": int(
                    trainer.current_gradient_accumulation_steps
                ),
                "local_pair_batch_size": local_pair_batch_size,
                "input_ids_sha256": input_sha,
                "labels_sha256": labels_sha,
                "pair_id": pair_id,
            }
        )

    def capture_dataloader(self, trainer: Any, dataloader: Any) -> None:
        if self.dataloader is not None:
            return
        batch_sampler = getattr(dataloader, "batch_sampler", None)
        sampler = getattr(dataloader, "sampler", None)
        dataset_records = len(trainer.train_dataset)
        self.dataloader = {
            "class": f"{type(dataloader).__module__}.{type(dataloader).__qualname__}",
            "records": dataset_records,
            "batches_per_rank": len(dataloader),
            "configured_per_device_batch": int(trainer.args.per_device_train_batch_size),
            "drop_last": bool(getattr(batch_sampler, "drop_last", trainer.args.dataloader_drop_last)),
            "batch_sampler": {
                "class": (
                    f"{type(batch_sampler).__module__}.{type(batch_sampler).__qualname__}"
                    if batch_sampler is not None
                    else None
                ),
                "batch_size": json_safe(getattr(batch_sampler, "batch_size", None)),
                "num_processes": json_safe(getattr(batch_sampler, "num_processes", None)),
                "process_index": json_safe(getattr(batch_sampler, "process_index", None)),
                "even_batches": json_safe(getattr(batch_sampler, "even_batches", None)),
                "total_batch_size": json_safe(getattr(batch_sampler, "total_batch_size", None)),
            },
            "sampler_class": (
                f"{type(sampler).__module__}.{type(sampler).__qualname__}"
                if sampler is not None
                else None
            ),
        }

    def observe_reference(self, trainer: Any, phase: str) -> dict[str, Any]:
        from accelerate.utils import is_peft_model

        model = trainer.accelerator.unwrap_model(trainer.model)
        event: dict[str, Any] = {
            "phase": phase,
            "ref_model_is_none": trainer.ref_model is None,
            "ref_adapter_name": getattr(trainer, "ref_adapter_name", None),
            "is_peft_model": bool(is_peft_model(model)),
            "adapters_disabled": bool(getattr(model, "_adapters_disabled", False)),
            "active_adapters": json_safe(getattr(model, "active_adapters", None)),
        }
        if is_peft_model(model):
            status = model.get_model_status()
            event["peft_enabled"] = json_safe(status.enabled)
        return event

    def observe_forward(self, kind: str, output: Mapping[str, Any]) -> None:
        self.forward_counts[kind] += 1
        finite = all(_finite_tensor(value) for value in output.values())
        if not finite:
            self.nonfinite.append(f"{kind}_forward")
            raise Day23GPUStageError(f"non-finite {kind} forward output")
        if len(self.forward_events) < MAX_RECORDED_FORWARD_EVENTS:
            self.forward_events.append(
                {
                    "kind": kind,
                    "sequence": self.forward_counts[kind],
                    "reference_context_active": self.reference_depth > 0,
                    "output_keys": sorted(output),
                    "finite": True,
                }
            )

    def observe_log(self, trainer: Any, logs: Mapping[str, Any]) -> None:
        numeric: dict[str, float] = {}
        for key, value in logs.items():
            if hasattr(value, "item"):
                with contextlib.suppress(Exception):
                    value = value.item()
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                number = float(value)
                if not math.isfinite(number):
                    self.nonfinite.append(f"log:{key}:step:{trainer.state.global_step}")
                    raise Day23GPUStageError(f"non-finite runtime log: {key}={value}")
                numeric[str(key)] = number
        self.logs.append(
            {
                "global_step": int(trainer.state.global_step),
                "numeric": numeric,
            }
        )
        self.snapshot_memory(f"log_step_{trainer.state.global_step}")

    @contextlib.contextmanager
    def _preserve_probe_state(self, model: Any):
        import torch

        modules = list(model.modules())
        training = [bool(module.training) for module in modules]
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        try:
            model.eval()
            yield
        finally:
            for module, state in zip(modules, training):
                module.training = state
            torch.random.set_rng_state(cpu_rng)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_rng)

    def _raw_pair_ids(self, count: int) -> list[str | None]:
        dataset = self.config.get("dataset", [])
        if not isinstance(dataset, list) or not dataset:
            return [None] * count
        raw_path = re.sub(r"#\d+$", "", str(dataset[0]))
        result: list[str | None] = []
        try:
            with Path(raw_path).open("r", encoding="utf-8") as handle:
                for line in handle:
                    if len(result) >= count:
                        break
                    value = json.loads(line)
                    result.append(str(value.get("pair_id")) if value.get("pair_id") is not None else None)
        except (OSError, json.JSONDecodeError):
            return [None] * count
        return result + [None] * (count - len(result))

    def run_probes(self, trainer: Any, phase: str) -> list[dict[str, Any]]:
        import torch

        count = 4 if self.stage == "mechanism_5step" else 1
        count = min(count, len(trainer.train_dataset))
        pair_ids = self._raw_pair_ids(count)
        probes: list[dict[str, Any]] = []
        model = trainer.model
        with self._preserve_probe_state(model), torch.no_grad():
            for index in range(count):
                example = trainer.train_dataset[index]
                batch = trainer.data_collator([example])
                batch = trainer._prepare_inputs(batch)
                with trainer.template.forward_context(trainer.model, batch):
                    policy = trainer.concatenated_forward(
                        trainer.model, batch, is_ref_model=False
                    )
                    with trainer.null_ref_context():
                        reference = trainer.concatenated_forward(
                            trainer.model, batch, is_ref_model=True
                        )
                policy_chosen = float(policy["chosen_logps"].detach().cpu().item())
                policy_rejected = float(policy["rejected_logps"].detach().cpu().item())
                ref_chosen = float(reference["chosen_logps"].detach().cpu().item())
                ref_rejected = float(reference["rejected_logps"].detach().cpu().item())
                values = [policy_chosen, policy_rejected, ref_chosen, ref_rejected]
                _require(all(math.isfinite(value) for value in values), "probe log-prob is non-finite")
                beta = float(getattr(trainer, "beta", self.config.get("beta", 0.1)))
                margin = beta * (
                    (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
                )
                probes.append(
                    {
                        "phase": phase,
                        "dataset_index": index,
                        "pair_id": pair_ids[index],
                        "input_ids_sha256": _tensor_sha256(batch["input_ids"]),
                        "labels_sha256": _tensor_sha256(batch["labels"]),
                        "input_shape": list(batch["input_ids"].shape),
                        "labels_shape": list(batch["labels"].shape),
                        "policy_chosen_logp": policy_chosen,
                        "policy_rejected_logp": policy_rejected,
                        "reference_chosen_logp": ref_chosen,
                        "reference_rejected_logp": ref_rejected,
                        "chosen_reward_margin": margin,
                    }
                )
        self.snapshot_memory(f"{phase}_probe_complete")
        return probes

    def before_train(self, trainer: Any) -> None:
        self.trainer = trainer
        self.pre_train_probes = self.run_probes(trainer, "pre_train")
        tolerance = 1e-4
        for probe in self.pre_train_probes:
            _require(
                abs(probe["policy_chosen_logp"] - probe["reference_chosen_logp"]) <= tolerance
                and abs(probe["policy_rejected_logp"] - probe["reference_rejected_logp"]) <= tolerance,
                "fresh LoRA policy is not a no-op over merged S1",
            )

    def after_train(self, trainer: Any, result: Any) -> None:
        self.result_summary = json_safe(result)
        self.post_train_probes = self.run_probes(trainer, "post_train")
        self.snapshot_memory("after_trainer_train")

    def finalize(self, *, require_success: bool) -> dict[str, Any]:
        trainer = self.trainer
        if self.prepared_model is not None:
            self.lora_final_digest = _named_tensor_digest(
                self._lora_named_parameters(self.prepared_model)
            )
        frozen_changes: list[dict[str, Any]] = []
        for name, before, parameter in self.frozen_versions.values():
            after = int(parameter._version)
            if after != before:
                frozen_changes.append({"name": name, "before": before, "after": after})
        global_step = int(trainer.state.global_step) if trainer is not None else None

        required_log_keys = {"loss", "grad_norm", "rewards/chosen", "rewards/rejected", "rewards/margins"}
        seen_log_keys = {
            key for record in self.logs for key in record.get("numeric", {})
        }
        memory_final = _cuda_snapshot("finalize")
        self.memory.append(memory_final)
        device_peaks: list[dict[str, Any]] = []
        for device in memory_final["devices"]:
            total = int(device["total_bytes"])
            max_reserved = int(device["max_reserved_bytes"])
            conservative_free_fraction = max(0.0, (total - max_reserved) / total)
            device_peaks.append(
                {
                    "index": device["index"],
                    "total_bytes": total,
                    "max_allocated_bytes": device["max_allocated_bytes"],
                    "max_reserved_bytes": max_reserved,
                    "residual_free_bytes": device["free_bytes"],
                    "residual_free_fraction": device["free_fraction"],
                    "conservative_peak_free_fraction": conservative_free_fraction,
                }
            )
        observed_free_fractions = [
            float(device["free_fraction"])
            for snapshot in self.memory
            for device in snapshot.get("devices", [])
        ]
        minimum_observed_free_fraction = (
            min(observed_free_fractions) if observed_free_fractions else None
        )

        if require_success:
            _require(trainer is not None, "trainer was never constructed")
            _require(global_step == EXPECTED_MAX_STEPS[self.stage], "trainer global_step drifted")
            _require(self.optimizer is not None, "optimizer inventory was not observed")
            _require(not frozen_changes, "a frozen tensor version changed")
            _require(self.lora_initial_digest is not None, "initial LoRA digest is absent")
            _require(self.lora_final_digest is not None, "final LoRA digest is absent")
            _require(self.lora_initial_digest != self.lora_final_digest, "LoRA digest did not change")
            _require(self.reference_context_count > 0, "reference disable context was never entered")
            _require(self.forward_counts["policy"] > 0, "policy forward was never observed")
            _require(self.forward_counts["reference"] > 0, "reference forward was never observed")
            _require(not self.nonfinite, "non-finite runtime evidence was observed")
            _require(required_log_keys <= seen_log_keys, "required finite train metrics were not logged")
            _require(device_peaks, "CUDA memory evidence is absent")
            _require(
                minimum_observed_free_fraction is not None
                and minimum_observed_free_fraction >= MIN_FREE_MEMORY_FRACTION,
                "observed device free memory fell below the required 15% headroom",
            )
            _require(
                all(
                    item["conservative_peak_free_fraction"] >= MIN_FREE_MEMORY_FRACTION
                    for item in device_peaks
                ),
                "peak memory left less than the required 15% headroom",
            )
            _require(self.post_train_probes, "post-train reload probe is absent")
            _require(self.dataloader is not None, "distributed dataloader evidence is absent")
            expected_records = 154 if self.stage == "bounded_smoke_30step" else (8 if self.stage == "g2_one_step" else 4)
            _require(
                self.dataloader["records"] == expected_records
                and self.dataloader["configured_per_device_batch"]
                == int(self.config["per_device_train_batch_size"]),
                "distributed dataloader batch/record semantics drifted",
            )
            _require(
                self.gradient_accumulation_observations
                and all(
                    0 < item["local_pair_batch_size"]
                    <= int(self.config["per_device_train_batch_size"])
                    for item in self.gradient_accumulation_observations
                ),
                "runtime local pair batch-size evidence drifted",
            )
            if self.stage == "mechanism_5step":
                _require(
                    self.dataloader["batches_per_rank"] == 2
                    and self.model_accepts_loss_kwargs is False
                    and self.compute_loss_func_is_none is True
                    and len(self.gradient_accumulation_observations) == 10
                    and all(
                        item["configured_gradient_accumulation_steps"] == 4
                        and item["current_gradient_accumulation_steps"] == 2
                        for item in self.gradient_accumulation_observations
                    )
                    and [
                        item["global_step_before"]
                        for item in self.gradient_accumulation_observations
                    ]
                    == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
                    "mechanism microbatch/GA remainder semantics drifted",
                )
                local_rank = int(os.environ["LOCAL_RANK"])
                expected_local_pair_ids = [
                    self.pre_train_probes[index]["pair_id"]
                    for index in ((0, 2) if local_rank == 0 else (1, 3))
                ] * 5
                _require(
                    [
                        item["pair_id"]
                        for item in self.gradient_accumulation_observations
                    ]
                    == expected_local_pair_ids,
                    "mechanism DDP pair shard/order/exposure drifted",
                )
                _require(len(self.pre_train_probes) == len(self.post_train_probes) == 4, "mechanism probes are incomplete")
                improvements = [
                    after["chosen_reward_margin"] - before["chosen_reward_margin"]
                    for before, after in zip(self.pre_train_probes, self.post_train_probes)
                ]
                _require(sum(improvements) / len(improvements) > 0, "mean mechanism margin did not improve")
                _require(sum(value > 0 for value in improvements) >= 3, "fewer than 3/4 mechanism pairs improved")

        return {
            "global_step": global_step,
            "trainable_inventory": {
                "tensors": len(self.trainable_parameters),
                "params": sum(item["numel"] for item in self.trainable_parameters),
                "parameters": self.trainable_parameters,
            },
            "optimizer_inventory": self.optimizer,
            "freeze": {
                **self.frozen_summary,
                "version_baseline_tensors": len(self.frozen_versions),
                "version_changes": frozen_changes,
                "all_versions_unchanged": not frozen_changes,
            },
            "lora": {
                "initial_digest": self.lora_initial_digest,
                "pre_step_digest": self.lora_pre_step_digest,
                "final_digest": self.lora_final_digest,
                "changed": (
                    self.lora_initial_digest is not None
                    and self.lora_final_digest is not None
                    and self.lora_initial_digest != self.lora_final_digest
                ),
            },
            "reference": {
                "context_count": self.reference_context_count,
                "events": self.reference_events,
                "implementation": "same_peft_model_disable_adapter",
            },
            "forwards": {
                "counts": self.forward_counts,
                "events": self.forward_events,
            },
            "probes": {
                "pre_train": self.pre_train_probes,
                "post_train": self.post_train_probes,
            },
            "logs": {
                "records": self.logs,
                "seen_numeric_keys": sorted(seen_log_keys),
                "nonfinite": self.nonfinite,
            },
            "memory": {
                "minimum_free_memory_fraction": MIN_FREE_MEMORY_FRACTION,
                "snapshots": self.memory,
                "device_peaks": device_peaks,
                "minimum_observed_device_free_fraction": minimum_observed_free_fraction,
            },
            "result_summary": self.result_summary,
            "dataloader": self.dataloader,
            "gradient_accumulation": {
                "model_accepts_loss_kwargs": self.model_accepts_loss_kwargs,
                "compute_loss_func_is_none": self.compute_loss_func_is_none,
                "training_step_calls": len(
                    self.gradient_accumulation_observations
                ),
                "observations": self.gradient_accumulation_observations,
            },
            "violations": self.violations,
        }


@contextlib.contextmanager
def install_observers(observer: RuntimeObserver):
    """Install wrappers that call the pinned implementation unchanged."""
    from swift.rlhf_trainers.dpo_trainer import DPOTrainer
    from swift.rlhf_trainers.rlhf_mixin import RLHFTrainerMixin

    original_init = DPOTrainer.__init__
    original_create_optimizer = DPOTrainer.create_optimizer
    original_train = DPOTrainer.train
    original_training_step = DPOTrainer.training_step
    original_get_train_dataloader = DPOTrainer.get_train_dataloader
    original_log = DPOTrainer.log
    original_concat = DPOTrainer.concatenated_forward
    original_null_ref = RLHFTrainerMixin.null_ref_context

    def observed_init(self: Any, *args: Any, **kwargs: Any) -> None:
        model = kwargs.get("model", args[0] if args else None)
        training_args = kwargs.get("args")
        _require(model is not None and training_args is not None, "DPOTrainer init signature drifted")
        observer.inspect_prepared_model(model, training_args)
        original_init(self, *args, **kwargs)
        observer.trainer = self
        observer.model_accepts_loss_kwargs = bool(self.model_accepts_loss_kwargs)
        observer.compute_loss_func_is_none = self.compute_loss_func is None
        _require(self.ref_model is None, "runtime constructed an explicit reference model")
        _require(getattr(self, "ref_adapter_name", None) is None, "runtime selected a ref adapter")

    def observed_create_optimizer(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_create_optimizer(self, *args, **kwargs)
        observer.capture_optimizer(self)
        return result

    def observed_train(self: Any, *args: Any, **kwargs: Any) -> Any:
        observer.before_train(self)
        result = original_train(self, *args, **kwargs)
        observer.after_train(self, result)
        return result

    def observed_training_step(self: Any, *args: Any, **kwargs: Any) -> Any:
        observer.capture_post_ddp_baseline(self)
        inputs = kwargs.get("inputs", args[1] if len(args) > 1 else None)
        _require(isinstance(inputs, Mapping), "training_step inputs signature drifted")
        observer.observe_training_step(self, inputs)
        return original_training_step(self, *args, **kwargs)

    def observed_get_train_dataloader(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_get_train_dataloader(self, *args, **kwargs)
        observer.capture_dataloader(self, result)
        return result

    def observed_log(self: Any, logs: Mapping[str, Any], *args: Any, **kwargs: Any) -> Any:
        result = original_log(self, logs, *args, **kwargs)
        observer.observe_log(self, logs)
        return result

    def observed_concat(self: Any, model: Any, batch: Mapping[str, Any], *args: Any, **kwargs: Any) -> Any:
        is_ref_model = bool(kwargs.get("is_ref_model", args[0] if args else False))
        kind = "reference" if is_ref_model else "policy"
        if is_ref_model and self.ref_model is None:
            _require(observer.reference_depth > 0, "reference forward escaped disable-adapter context")
        capture_memory = kind not in observer.forward_memory_seen
        if capture_memory:
            observer.snapshot_memory(f"before_first_{kind}_forward")
        result = original_concat(self, model, batch, *args, **kwargs)
        if capture_memory:
            observer.snapshot_memory(f"after_first_{kind}_forward")
            observer.forward_memory_seen.add(kind)
        observer.observe_forward(kind, result)
        return result

    @contextlib.contextmanager
    def observed_null_ref(self: Any):
        before = observer.observe_reference(self, "before")
        _require(before["ref_model_is_none"] is True, "reference model unexpectedly exists")
        _require(before["ref_adapter_name"] is None, "reference adapter unexpectedly exists")
        with original_null_ref(self):
            observer.reference_depth += 1
            observer.reference_context_count += 1
            try:
                inside = observer.observe_reference(self, "inside")
                _require(inside["is_peft_model"] is True, "reference policy is not PEFT")
                _require(inside["adapters_disabled"] is True, "PEFT adapter was not disabled")
                _require(inside.get("peft_enabled") is False, "PEFT layers remain enabled in reference")
                if len(observer.reference_events) < MAX_RECORDED_REFERENCE_EVENTS:
                    observer.reference_events.append({"before": before, "inside": inside})
                yield
            finally:
                observer.reference_depth -= 1
        after = observer.observe_reference(self, "after")
        _require(after["adapters_disabled"] is False, "PEFT adapter stayed disabled")
        _require(after.get("peft_enabled") is True, "policy adapter was not restored")
        if observer.reference_events and "after" not in observer.reference_events[-1]:
            observer.reference_events[-1]["after"] = after

    DPOTrainer.__init__ = observed_init
    DPOTrainer.create_optimizer = observed_create_optimizer
    DPOTrainer.train = observed_train
    DPOTrainer.training_step = observed_training_step
    DPOTrainer.get_train_dataloader = observed_get_train_dataloader
    DPOTrainer.log = observed_log
    DPOTrainer.concatenated_forward = observed_concat
    RLHFTrainerMixin.null_ref_context = observed_null_ref
    try:
        yield
    finally:
        DPOTrainer.__init__ = original_init
        DPOTrainer.create_optimizer = original_create_optimizer
        DPOTrainer.train = original_train
        DPOTrainer.training_step = original_training_step
        DPOTrainer.get_train_dataloader = original_get_train_dataloader
        DPOTrainer.log = original_log
        DPOTrainer.concatenated_forward = original_concat
        RLHFTrainerMixin.null_ref_context = original_null_ref


def validate_topology(config: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    _require(visible == "0,1", "CUDA_VISIBLE_DEVICES must be fixed to exactly `0,1`")
    _require(os.environ.get("WORLD_SIZE") == "2", "WORLD_SIZE must be exactly two")
    _require(os.environ.get("LOCAL_WORLD_SIZE") == "2", "LOCAL_WORLD_SIZE must be exactly two")
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    rank = int(os.environ.get("RANK", "-1"))
    _require(local_rank in (0, 1) and rank in (0, 1), "DDP rank identity drifted")
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    _require(torch.cuda.device_count() == 2, "runtime must expose exactly two GPUs")
    _require(torch.cuda.current_device() == local_rank, "DDP process is on the wrong local GPU")
    _require(not (set(config) & FORBIDDEN_TOPOLOGY_KEYS), "config contains a topology shortcut")
    device = torch.cuda.get_device_properties(local_rank)
    topology = {
        "CUDA_VISIBLE_DEVICES": visible,
        "NPROC_PER_NODE": os.environ.get("NPROC_PER_NODE"),
        "NNODES": os.environ.get("NNODES"),
        "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
        "LOCAL_RANK": os.environ.get("LOCAL_RANK"),
        "LOCAL_WORLD_SIZE": os.environ.get("LOCAL_WORLD_SIZE"),
        "RANK": os.environ.get("RANK"),
        "torch_distributed_initialized": torch.distributed.is_initialized(),
        "visible_device_count": torch.cuda.device_count(),
        "world_size": torch.distributed.get_world_size(),
        "rank": torch.distributed.get_rank(),
        "local_rank": local_rank,
        "device": {
            "index": local_rank,
            "name": device.name,
            "total_memory_bytes": int(device.total_memory),
            "capability": list(torch.cuda.get_device_capability(local_rank)),
        },
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    _require(topology["torch_distributed_initialized"], "DDP process group is not initialized")
    _require(
        topology["world_size"] == EXPECTED_WORLD_SIZE
        and topology["rank"] == rank,
        "DDP process group topology drifted",
    )
    return topology


def initialize_ddp() -> dict[str, int]:
    import torch

    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1", "DDP requires CUDA_VISIBLE_DEVICES=0,1")
    _require(os.environ.get("WORLD_SIZE") == "2", "runner must be launched by torchrun with world size two")
    _require(os.environ.get("LOCAL_WORLD_SIZE") == "2", "runner local world size must be two")
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    _require(rank in (0, 1) and local_rank in (0, 1), "runner DDP ranks are invalid")
    _require(torch.cuda.device_count() == 2, "runner must see both GPUs")
    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl",
            timeout=dt.timedelta(minutes=10),
        )
    _require(
        torch.distributed.get_world_size() == EXPECTED_WORLD_SIZE
        and torch.distributed.get_rank() == rank,
        "initialized DDP process group drifted",
    )
    return {"rank": rank, "local_rank": local_rank, "world_size": EXPECTED_WORLD_SIZE}


def claim_mechanism_correction_execution(
    context: dict[str, Any], ddp: Mapping[str, int]
) -> dict[str, Any]:
    import torch

    path = context["mechanism_correction_claim_path"]
    _require(isinstance(path, Path), "mechanism correction claim path is absent")
    result: list[Any] = [None]
    if ddp["rank"] == 0:
        try:
            previous = _mapping(
                context.get("previous"), "mechanism previous gate"
            )
            correction = _mapping(
                context.get("mechanism_protocol_correction"),
                "mechanism protocol correction",
            )
            prior = _mapping(correction.get("prior_attempt"), "prior mechanism attempt")
            claim = {
                "schema_name": CORRECTION_CLAIM_SCHEMA,
                "schema_version": 1,
                "status": "claimed_for_execution",
                "created_at_utc": utc_now(),
                "attempt_name": "mechanism_5step_topology_correction_attempt2",
                "attempt_number": 2,
                "one_shot": True,
                "third_attempt_forbidden": True,
                "binding_path": str(context["binding_path"]),
                "binding_file_sha256": context["binding_file_sha256"],
                "binding_sha256": context["binding_sha256"],
                "executable_config_path": str(context["config_path"]),
                "executable_config_file_sha256": context["config_file_sha256"],
                "prior_failure_receipt_sha256": prior["receipt_sha256"],
                "g2_success_receipt_sha256": previous[
                    "previous_gate_receipt_sha256"
                ],
                "process_identity": _process_identity(),
                "optimizer_started_at_claim_time": False,
            }
            claim_sha = write_sealed_json(path, claim, "claim_sha256")
            result[0] = {"status": "ok", "claim_sha256": claim_sha}
        except BaseException as error:
            result[0] = {
                "status": "error",
                "type": f"{error.__class__.__module__}.{error.__class__.__qualname__}",
                "message": str(error),
            }
    torch.distributed.broadcast_object_list(result, src=0)
    outcome = _mapping(result[0], "mechanism correction claim broadcast")
    _require(
        outcome.get("status") == "ok",
        "cannot claim the one-shot mechanism correction: "
        + str(outcome.get("message")),
    )
    correction = _mapping(
        context.get("mechanism_protocol_correction"),
        "mechanism protocol correction",
    )
    prior = _mapping(correction.get("prior_attempt"), "prior mechanism attempt")
    identity = _validate_mechanism_correction_claim(
        path,
        binding_sha256=context["binding_sha256"],
        config_file_sha256=context["config_file_sha256"],
        prior_receipt_sha256=prior["receipt_sha256"],
    )
    _require(
        identity["claim_sha256"] == outcome.get("claim_sha256"),
        "mechanism correction claim broadcast hash drifted",
    )
    context["mechanism_correction_claim"] = identity
    return identity


def checkpoint_manifest(path: Path, global_step: int) -> dict[str, Any]:
    _require(path.is_dir(), f"checkpoint is missing: {path}")
    files: dict[str, dict[str, Any]] = {}
    for child in sorted(path.rglob("*")):
        _require(not child.is_symlink(), f"checkpoint contains a symlink: {child}")
        if child.is_dir():
            continue
        _require(child.is_file(), f"checkpoint contains a non-file: {child}")
        relative = child.relative_to(path).as_posix()
        files[relative] = {
            "bytes": child.stat().st_size,
            "sha256": file_sha256(child),
        }
    _require(REQUIRED_CHECKPOINT_FILES <= set(files), f"checkpoint state is incomplete: {path}")
    trainer_state = load_json(path / "trainer_state.json", "trainer state")
    _require(int(trainer_state.get("global_step", -1)) == global_step, "checkpoint trainer step drifted")
    return {
        "path": str(path),
        "global_step": global_step,
        "files": files,
        "files_sha256": hashlib.sha256(canonical_json(files)).hexdigest(),
    }


def collect_checkpoints(output_dir: Path, stage: str) -> list[dict[str, Any]]:
    expected_steps = EXPECTED_CHECKPOINT_STEPS[stage]
    actual: dict[int, Path] = {}
    if output_dir.is_dir():
        for path in output_dir.glob("checkpoint-*"):
            if not path.is_dir():
                continue
            match = re.fullmatch(r"checkpoint-(\d+)", path.name)
            if match:
                actual[int(match.group(1))] = path.resolve()
    _require(set(actual) == set(expected_steps), f"checkpoint inventory drifted: {sorted(actual)}")
    return [checkpoint_manifest(actual[step], step) for step in expected_steps]


def realized_global_batch_profile(
    gathered: Sequence[Mapping[str, Any]], stage: str
) -> dict[str, Any]:
    by_step: dict[int, dict[int, int]] = {
        step: {} for step in range(EXPECTED_MAX_STEPS[stage])
    }
    for rank_entry in gathered:
        rank = int(rank_entry["rank"])
        observations = _mapping(
            rank_entry.get("observations"), f"rank {rank} observations"
        )
        accumulation = _mapping(
            observations.get("gradient_accumulation"),
            f"rank {rank} gradient accumulation",
        )
        records = accumulation.get("observations")
        _require(isinstance(records, list) and records, f"rank {rank} has no batch observations")
        local_by_step: dict[int, int] = {}
        for raw in records:
            item = _mapping(raw, f"rank {rank} batch observation")
            step = item.get("global_step_before")
            batch_size = item.get("local_pair_batch_size")
            _require(
                isinstance(step, int)
                and not isinstance(step, bool)
                and step in by_step
                and isinstance(batch_size, int)
                and not isinstance(batch_size, bool)
                and batch_size > 0,
                f"rank {rank} batch observation values drifted",
            )
            local_by_step[step] = local_by_step.get(step, 0) + batch_size
        _require(
            set(local_by_step) == set(by_step),
            f"rank {rank} optimizer-step batch inventory drifted",
        )
        for step, count in local_by_step.items():
            by_step[step][rank] = count
    _require(
        all(set(rank_counts) == {0, 1} for rank_counts in by_step.values()),
        "distributed optimizer-step rank batch inventory drifted",
    )
    per_step = [sum(by_step[step].values()) for step in sorted(by_step)]
    expected = (
        [4] * 5
        if stage == "mechanism_5step"
        else ([8] * 19 + [2] + [8] * 10 if stage == "bounded_smoke_30step" else [8])
    )
    _require(per_step == expected, f"{stage} realized global batch sequence drifted")
    return {
        "optimizer_steps": len(per_step),
        "per_step_global_pair_counts": per_step,
        "per_step_global_pair_counts_sha256": hashlib.sha256(
            canonical_json(per_step)
        ).hexdigest(),
        "minimum": min(per_step),
        "maximum": max(per_step),
        "unique": sorted(set(per_step)),
        "partial_batch_present": len(set(per_step)) > 1,
    }


def base_receipt(
    context: Mapping[str, Any], stage: str, started_at: str, runtime: Mapping[str, Any] | None
) -> dict[str, Any]:
    previous = json_safe(context.get("previous"))
    return {
        "schema_name": RECEIPT_SCHEMA,
        "schema_version": RECEIPT_VERSION,
        "stage": stage,
        "started_at_utc": started_at,
        "binding": {
            "path": str(context.get("binding_path")),
            "file_sha256": context.get("binding_file_sha256"),
            "binding_sha256": context.get("binding_sha256"),
        },
        "binding_path": str(context.get("binding_path")),
        "binding_file_sha256": context.get("binding_file_sha256"),
        "binding_sha256": context.get("binding_sha256"),
        "producer": {
            "path": str(context.get("runner_path", Path(__file__).resolve())),
            "file_sha256": context.get("runner_file_sha256"),
        },
        "cpu_contract": {
            "path": str(context.get("cpu_contract_path")),
            "file_sha256": context.get("cpu_contract_file_sha256"),
            "contract_sha256": context.get("cpu_contract_sha256"),
        },
        "cpu_contract_sha256": context.get("cpu_contract_sha256"),
        "executable_config": {
            "path": str(context.get("config_path")),
            "file_sha256": context.get("config_file_sha256"),
        },
        "executable_config_path": str(context.get("config_path")),
        "executable_config_file_sha256": context.get("config_file_sha256"),
        "remote_parent_files_sha256": context.get("remote_parent_files_sha256"),
        "process_identity": _process_identity(),
        "previous_gate": previous,
        "previous_gate_receipt_sha256": (
            previous.get("previous_gate_receipt_sha256")
            if isinstance(previous, Mapping)
            else None
        ),
        "runtime": json_safe(runtime),
        "runtime_identity": json_safe(runtime),
        "mechanism_protocol_correction": json_safe(
            context.get("mechanism_protocol_correction")
        ),
        "mechanism_correction_claim": json_safe(
            context.get("mechanism_correction_claim")
        ),
    }


def run_bound_stage(binding_path: Path, stage: str) -> tuple[Path | None, str | None]:
    started_at = utc_now()
    started_monotonic = time.monotonic()
    binding: dict[str, Any] | None = None
    context: dict[str, Any] = {"binding_path": binding_path.expanduser().absolute()}
    runtime: dict[str, Any] | None = None
    observer: RuntimeObserver | None = None
    ddp: dict[str, int] | None = None
    try:
        ddp = initialize_ddp()
        binding, context = validate_binding(binding_path, stage)
        runtime = validate_runtime(binding)
        import torch

        if stage == "mechanism_5step":
            claim_mechanism_correction_execution(context, ddp)

        # Every rank has now independently verified that the output path is
        # absent.  No rank may enter ms-swift (which creates output_dir) until
        # all peers pass that same fresh-start gate.
        torch.distributed.barrier()

        # Match swift.cli.rlhf's JSON expansion; torchrun already owns process
        # topology, so the single-device helper must not be called here.
        from swift.cli.main import parse_yaml_args

        cli_argv = [str(context["config_path"])]
        parse_yaml_args(cli_argv)
        topology = validate_topology(context["config"])
        torch.cuda.reset_peak_memory_stats(ddp["local_rank"])
        observer = RuntimeObserver(stage, context["config"])
        observer.snapshot_memory("before_rlhf_main")

        from swift.pipelines import rlhf_main

        with install_observers(observer):
            result = rlhf_main(cli_argv)
        observer.result_summary = json_safe(result)
        observations = observer.finalize(require_success=True)
        local_rank_evidence = {
            "rank": ddp["rank"],
            "local_rank": ddp["local_rank"],
            "process_identity": _process_identity(),
            "topology": topology,
            "observations": observations,
        }
        gathered: list[Any] = [None] * ddp["world_size"]
        torch.distributed.all_gather_object(gathered, local_rank_evidence)
        _require(
            [item.get("rank") for item in gathered if isinstance(item, Mapping)] == [0, 1],
            "DDP rank evidence inventory drifted",
        )
        _require(
            len(
                {
                    item["observations"]["lora"]["final_digest"]
                    for item in gathered
                }
            )
            == 1,
            "DDP ranks ended with different LoRA tensors",
        )
        _require(
            {item["observations"]["global_step"] for item in gathered}
            == {EXPECTED_MAX_STEPS[stage]},
            "DDP ranks ended at different global steps",
        )
        _require(
            len(
                {
                    (
                        item["process_identity"]["boot_id"],
                        item["process_identity"]["pid"],
                        item["process_identity"]["proc_start_ticks"],
                    )
                    for item in gathered
                }
            )
            == EXPECTED_WORLD_SIZE,
            "DDP rank process identities are not distinct",
        )
        batch_profile = realized_global_batch_profile(gathered, stage)
        torch.distributed.barrier()
        if ddp["rank"] != 0:
            return None, None

        checkpoints = collect_checkpoints(context["output_dir"], stage)
        final_checkpoint = checkpoints[-1]

        receipt = base_receipt(context, stage, started_at, runtime)
        receipt.update(
            {
                "status": "pass",
                "completed_at_utc": utc_now(),
                "duration_seconds": time.monotonic() - started_monotonic,
                "topology": topology,
                "observations": observations,
                "evidence": observations,
                "distributed_evidence": {
                    "world_size": ddp["world_size"],
                    "configured_nominal_global_train_batch_size": (
                        ddp["world_size"]
                        * int(context["config"]["per_device_train_batch_size"])
                        * int(context["config"]["gradient_accumulation_steps"])
                    ),
                    "realized_global_batch_profile": batch_profile,
                    "ranks": gathered,
                },
                "checkpoint": final_checkpoint,
                "checkpoints": checkpoints,
                "claim_boundary": {
                    "runtime_stage_passed": True,
                    "fresh_lora_only": True,
                    "reference_disable_adapter_observed": True,
                    "frozen_tensor_versions_unchanged": True,
                    "checkpoint_reload_parity_proven": False,
                    "checkpoint_reload_parity_pending_external_fresh_process": True,
                    "heldout_consumed": False,
                    "two_gpu_ddp_runtime_proven": True,
                },
            }
        )
        receipt_path = context["success_receipt"]
        receipt_sha = write_sealed_json(receipt_path, receipt, "receipt_sha256")
        return receipt_path, receipt_sha
    except BaseException as error:
        observations: Any = None
        if observer is not None:
            with contextlib.suppress(BaseException):
                observations = observer.finalize(require_success=False)
        partial_checkpoints: list[dict[str, Any]] = []
        output_dir = context.get("output_dir")
        if isinstance(output_dir, Path) and output_dir.is_dir():
            for path in sorted(output_dir.glob("checkpoint-*")):
                match = re.fullmatch(r"checkpoint-(\d+)", path.name)
                if not match:
                    continue
                with contextlib.suppress(BaseException):
                    partial_checkpoints.append(checkpoint_manifest(path, int(match.group(1))))

        failure_receipt = context.get("failure_receipt")
        if not isinstance(failure_receipt, Path) and binding is not None:
            with contextlib.suppress(BaseException):
                failure_receipt = _receipt_paths(binding, stage)[1]
        failure = base_receipt(context, stage, started_at, runtime)
        failure.update(
            {
                "status": "fail",
                "completed_at_utc": utc_now(),
                "duration_seconds": time.monotonic() - started_monotonic,
                "error": {
                    "type": f"{error.__class__.__module__}.{error.__class__.__qualname__}",
                    "message": str(error),
                    "traceback": traceback.format_exception(error)[-12:],
                },
                "observations": observations,
                "evidence": observations,
                "partial_checkpoints": partial_checkpoints,
                "claim_boundary": {
                    "runtime_stage_passed": False,
                    "gpu_optimizer_ready": False,
                    "heldout_consumed": False,
                    "ddp_rank": ddp.get("rank") if isinstance(ddp, Mapping) else None,
                },
            }
        )
        if isinstance(failure_receipt, Path):
            try:
                receipt_sha = write_sealed_json(
                    failure_receipt, failure, "receipt_sha256"
                )
            except BaseException as seal_error:
                raise Day23GPUStageError(
                    f"stage failed and failure receipt could not be sealed: {seal_error}"
                ) from error
            raise Day23GPUStageError(
                f"stage failed; sealed failure receipt {failure_receipt} ({receipt_sha})"
            ) from error
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument(
        "--execute",
        choices=("RUN_GPU_OPTIMIZER",),
        required=True,
        help="Explicit acknowledgement that this command may run a bound optimizer stage.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt_path, receipt_sha = run_bound_stage(args.binding, args.stage)
    except BaseException as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if receipt_path is None:
        print(
            json.dumps(
                {
                    "status": "day23_gpu_stage_worker_complete",
                    "stage": args.stage,
                    "rank": int(os.environ.get("RANK", "-1")),
                },
                sort_keys=True,
            )
        )
        return 0
    print(
        json.dumps(
            {
                "status": "valid_day23_gpu_stage_receipt",
                "stage": args.stage,
                "receipt": str(receipt_path),
                "receipt_sha256": receipt_sha,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
