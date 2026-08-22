#!/usr/bin/env python3
"""Generate, resume, and merge the two-GPU Day 22 S1 MBPP rollout.

The two workers are ordinary single-GPU processes.  ``CUDA_VISIBLE_DEVICES``
selects the physical GPU; ``--shard-id`` selects a disjoint deterministic set
of task families.  Every candidate is published independently with no
overwrite, so rerunning the same worker resumes after its last durable row.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
BOOTCAMP_ROOT = Path(os.environ.get("DAY22_BOOTCAMP_ROOT", str(HERE.parent))).resolve()
DAY20_DIR = BOOTCAMP_ROOT / "day-20-qwen35-balanced-lora-sft"
if str(DAY20_DIR) not in sys.path:
    sys.path.insert(0, str(DAY20_DIR))


SCHEMA_VERSION = 1
CONTRACT_SCHEMA = "day22.s1_rollout_contract"
CANDIDATE_SCHEMA = "day22.s1_rollout_candidate"
SHARD_SCHEMA = "day22.s1_rollout_shard_summary"
MERGE_SCHEMA = "day22.s1_rollout_merge_manifest"
WINNER = "main-s20260809-lr1e-4-final"
WINNER_CHECKPOINT = (
    "/root/autodl-tmp/runs/day20-v3-qwen35-lora-20260812T105141Z/"
    "adapters/main/s20260809/lr1e-4/attempt-001/checkpoint-1904"
)
WINNER_INTEGRITY_SHA256 = (
    "d0f72be9751628c9073cc8e4104f16d8620bd598dbb8a1e97f1df9bae51170a3"
)
WINNER_CHECKPOINT_SNAPSHOT_SHA256 = (
    "c169e0bb55b20951d27a889e4ef0deae3a01fe10acabee804b212d3d8170db0a"
)
WINNER_ADAPTER_SHA256 = (
    "29dc1a7d676b7f2f88675521bc44cf81dd43b700423df3cf4adf8d6a1c4487c3"
)
MODEL_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
MS_SWIFT_COMMIT = "565a1ad586a21d24b23931c52d2c62b49c39bee8"
EXPECTED_RUNTIME = {
    "ms-swift": "4.5.0.dev0",
    "peft": "0.19.1",
    "torch": "2.10.0+cu128",
    "transformers": "5.12.1",
}
EXPECTED_FAMILIES = 330
EXPECTED_SHARDS = 2
EXPECTED_K = 6
GENERATION = {
    "max_new_tokens": 512,
    "num_beams": 1,
    "repetition_penalty": 1.0,
    "temperature": 0.8,
    "top_p": 0.95,
}
PARTITION_DOMAIN = "day22.s1_rollout.family_partition.v1"
SEED_DOMAIN = "day22.s1_rollout.candidate_seed.v1"


class Day22RolloutError(ValueError):
    """A rollout input, provenance binding, or durable output drifted."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Day22RolloutError(message)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Day22RolloutError(f"cannot read JSON: {path}") from error
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def verify_self_hash(value: Mapping[str, Any], field: str) -> str:
    expected = value.get(field)
    actual = object_sha256({key: item for key, item in value.items() if key != field})
    require(isinstance(expected, str) and expected == actual, f"{field} mismatch")
    return expected


def write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode(
        "utf-8"
    ) + b"\n"
    write_bytes_new(path, payload)


def write_bytes_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".attempt", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise Day22RolloutError(f"refusing to overwrite: {path}") from error
        temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def publish_or_verify_json(path: Path, value: Mapping[str, Any], hash_field: str) -> str:
    if path.exists():
        existing = load_json(path)
        verify_self_hash(existing, hash_field)
        require(existing == value, f"existing output differs: {path}")
        return "verified"
    write_json_new(path, value)
    return "published"


def validate_seed_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json(path.resolve())
    verify_self_hash(manifest, "manifest_sha256")
    header = manifest.get("header")
    records = manifest.get("records")
    require(isinstance(header, dict), "seed manifest header is missing")
    require(isinstance(records, list), "seed manifest records are missing")
    require(
        header.get("schema_name") == "day22.mbpp_seed_manifest"
        and header.get("schema_version") == 1,
        "wrong Day 22 seed manifest schema",
    )
    require(len(records) == EXPECTED_FAMILIES, "seed manifest must contain 330 families")
    family_ids: list[str] = []
    for row in records:
        require(isinstance(row, dict), "seed record is not an object")
        verify_self_hash(row, "record_sha256")
        family = row.get("task_family_id")
        prompt = row.get("prompt")
        require(isinstance(family, str) and family, "seed family ID is missing")
        require(
            isinstance(prompt, dict)
            and isinstance(prompt.get("text"), str)
            and prompt.get("sha256") == text_sha256(prompt["text"]),
            f"seed prompt hash drifted: {family}",
        )
        family_ids.append(family)
    require(len(set(family_ids)) == EXPECTED_FAMILIES, "duplicate seed family ID")
    selection = header.get("selection")
    require(
        isinstance(selection, dict)
        and selection.get("records") == EXPECTED_FAMILIES,
        "seed selection count drifted",
    )
    return manifest


def family_order(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        records,
        key=lambda row: (
            text_sha256(f"{PARTITION_DOMAIN}\0{row['task_family_id']}"),
            str(row["task_family_id"]),
        ),
    )


def candidate_seed(family_id: str, sample_index: int) -> int:
    payload = f"{SEED_DOMAIN}\0{family_id}\0{sample_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def candidate_id(family_id: str, sample_index: int) -> str:
    return f"{family_id}:s1:{WINNER}:sample:{sample_index:02d}"


def expected_specs(
    records: Sequence[Mapping[str, Any]], shard_count: int = EXPECTED_SHARDS
) -> list[dict[str, Any]]:
    require(shard_count == EXPECTED_SHARDS, "Day 22 rollout requires exactly two shards")
    result: list[dict[str, Any]] = []
    for family_ordinal, record in enumerate(family_order(records)):
        shard_id = family_ordinal % shard_count
        for sample_index in range(EXPECTED_K):
            family_id = str(record["task_family_id"])
            result.append(
                {
                    "candidate_id": candidate_id(family_id, sample_index),
                    "family_ordinal": family_ordinal,
                    "record": record,
                    "sample_index": sample_index,
                    "seed": candidate_seed(family_id, sample_index),
                    "shard_id": shard_id,
                }
            )
    return result


def model_file_manifest(model_path: Path) -> dict[str, dict[str, Any]]:
    model = model_path.resolve()
    require(model.is_dir() and not model.is_symlink(), "merged model directory is missing")
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(model.rglob("*")):
        require(not path.is_symlink(), f"merged model contains a symlink: {path}")
        if path.is_file():
            relative = path.relative_to(model).as_posix()
            files[relative] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    require("config.json" in files, "merged model config.json is missing")
    require(
        any(name.endswith(".safetensors") for name in files),
        "merged model weights are missing",
    )
    return files


def runtime_identity(ms_swift_root: Path) -> dict[str, Any]:
    versions: dict[str, str] = {}
    for package in EXPECTED_RUNTIME:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise Day22RolloutError(f"required runtime package is missing: {package}") from error
    require(versions == EXPECTED_RUNTIME, f"rollout runtime drifted: {versions}")
    try:
        commit = subprocess.run(
            ["git", "-C", str(ms_swift_root.resolve()), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(ms_swift_root.resolve()), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise Day22RolloutError("cannot inspect the ms-swift checkout") from error
    require(commit == MS_SWIFT_COMMIT and not dirty, "ms-swift checkout drifted or is dirty")
    identity = {
        "ms_swift_commit": commit,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": dict(sorted(versions.items())),
    }
    identity["runtime_sha256"] = object_sha256(identity)
    return identity


def lineage_manifest_identity(path: Path) -> dict[str, Any]:
    value = load_json(path.resolve())
    verify_self_hash(value, "promotion_manifest_sha256")
    downstream_key = value.get("downstream_key")
    export = value.get("inference_export")
    require(
        isinstance(downstream_key, str) and downstream_key.strip(),
        "promotion manifest downstream_key is missing",
    )
    require(isinstance(export, dict), "promotion manifest inference_export is missing")
    require(
        value.get("schema_name") == "day21.qwen35_downstream_ready_s1_promotion"
        and value.get("schema_version") == 1
        and value.get("status") == "promoted"
        and value.get("downstream_ready") is True
        and value.get("role") == "S1"
        and value.get("state", {}).get("candidate") == WINNER
        and value.get("training", {}).get("inference_template") == "qwen3_5"
        and export.get("fresh_process_exact_token_id_parity_passed") is True,
        "S1 promotion manifest is not downstream-ready for Qwen3.5 inference",
    )
    export_path = Path(str(export.get("path", ""))).expanduser()
    require(export_path.is_absolute(), "promotion manifest inference export path is invalid")
    export_manifest_sha256 = export.get("manifest_sha256")
    require(
        isinstance(export_manifest_sha256, str)
        and len(export_manifest_sha256) == 64
        and all(character in "0123456789abcdef" for character in export_manifest_sha256),
        "promotion manifest inference_export.manifest_sha256 is invalid",
    )
    require(
        WINNER.encode("utf-8") in canonical_json(value),
        "promotion manifest does not name winner",
    )
    return {
        "path": str(path.resolve()),
        "file_sha256": file_sha256(path.resolve()),
        "promotion_manifest_sha256": value["promotion_manifest_sha256"],
        "downstream_key": downstream_key,
        "merged_export_manifest_sha256": export_manifest_sha256,
        "merged_export_path": str(export_path.resolve()),
    }


def build_contract(
    *,
    seed_path: Path,
    seed_manifest: Mapping[str, Any],
    model_path: Path,
    model_files: Mapping[str, Any],
    lineage_identity: Mapping[str, Any],
    output_root: Path,
    runtime: Mapping[str, Any],
    implementation_sha256: str,
) -> dict[str, Any]:
    specs = expected_specs(seed_manifest["records"])
    per_shard = {
        str(shard): len([item for item in specs if item["shard_id"] == shard])
        for shard in range(EXPECTED_SHARDS)
    }
    contract: dict[str, Any] = {
        "schema_name": CONTRACT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "status": "frozen",
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": implementation_sha256,
        },
        "seed_pool": {
            "path": str(seed_path.resolve()),
            "file_sha256": file_sha256(seed_path.resolve()),
            "manifest_sha256": seed_manifest["manifest_sha256"],
            "families": EXPECTED_FAMILIES,
        },
        "s1": {
            "candidate": WINNER,
            "model_role": "merged_s1",
            "model_revision": MODEL_REVISION,
            "source_checkpoint": WINNER_CHECKPOINT,
            "source_checkpoint_integrity_sha256": WINNER_INTEGRITY_SHA256,
            "source_checkpoint_snapshot_sha256": WINNER_CHECKPOINT_SNAPSHOT_SHA256,
            "source_adapter_sha256": WINNER_ADAPTER_SHA256,
            "lineage_manifest": dict(lineage_identity),
            "merged_export": {
                "path": str(model_path.resolve()),
                "files": dict(sorted(model_files.items())),
                "snapshot_sha256": object_sha256(model_files),
            },
        },
        "generation": {
            **GENERATION,
            "do_sample": True,
            "enable_thinking": False,
            "add_non_thinking_prefix": True,
            "response_boundary": "exact_generated_token_ids_decode",
            "response_repair": False,
            "samples_per_family": EXPECTED_K,
            "seed_domain": SEED_DOMAIN,
        },
        "partition": {
            "domain": PARTITION_DOMAIN,
            "family_unit": "native_mbpp_task_id",
            "shards": EXPECTED_SHARDS,
            "families_per_shard": {"0": 165, "1": 165},
            "candidates_per_shard": per_shard,
            "total_candidates": EXPECTED_FAMILIES * EXPECTED_K,
        },
        "output_root": str(output_root.resolve()),
        "runtime": dict(runtime),
    }
    require(per_shard == {"0": 990, "1": 990}, "two-GPU partition is imbalanced")
    contract["contract_sha256"] = object_sha256(contract)
    return contract


def verify_contract(path: Path, *, live: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = load_json(path.resolve())
    verify_self_hash(contract, "contract_sha256")
    require(
        contract.get("schema_name") == CONTRACT_SCHEMA
        and contract.get("schema_version") == SCHEMA_VERSION
        and contract.get("status") == "frozen",
        "wrong rollout contract schema/status",
    )
    require(contract.get("generation", {}).get("samples_per_family") == EXPECTED_K, "K drifted")
    require(contract.get("partition", {}).get("shards") == EXPECTED_SHARDS, "shard count drifted")
    seed_path = Path(str(contract["seed_pool"]["path"]))
    seed_manifest = validate_seed_manifest(seed_path)
    require(
        file_sha256(seed_path) == contract["seed_pool"]["file_sha256"]
        and seed_manifest["manifest_sha256"] == contract["seed_pool"]["manifest_sha256"],
        "frozen seed manifest drifted",
    )
    require(
        str(Path(str(contract["output_root"])).resolve()) == contract["output_root"],
        "contract output root is not absolute/canonical",
    )
    if live:
        require(
            file_sha256(Path(__file__).resolve()) == contract["implementation"]["sha256"],
            "rollout implementation drifted",
        )
        lineage = contract["s1"]["lineage_manifest"]
        require(
            file_sha256(Path(lineage["path"])) == lineage["file_sha256"],
            "S1 lineage manifest drifted",
        )
        require(
            lineage_manifest_identity(Path(lineage["path"])) == lineage,
            "S1 promotion/downstream/export identity drifted",
        )
        model = Path(contract["s1"]["merged_export"]["path"])
        files = model_file_manifest(model)
        require(
            files == contract["s1"]["merged_export"]["files"]
            and object_sha256(files) == contract["s1"]["merged_export"]["snapshot_sha256"],
            "merged S1 export drifted",
        )
    return contract, seed_manifest


def row_path(output_root: Path, shard_id: int, candidate: str) -> Path:
    name = f"{text_sha256(candidate)}.json"
    return output_root / "shards" / f"shard-{shard_id}" / "rows" / name


def validate_candidate_row(
    row: Mapping[str, Any], *, contract: Mapping[str, Any], spec: Mapping[str, Any]
) -> None:
    verify_self_hash(row, "candidate_sha256")
    record = spec["record"]
    require(
        row.get("schema_name") == CANDIDATE_SCHEMA
        and row.get("schema_version") == SCHEMA_VERSION,
        "wrong candidate row schema",
    )
    require(row.get("candidate_id") == spec["candidate_id"], "candidate ID drifted")
    require(row.get("task_family_id") == record["task_family_id"], "family ID drifted")
    require(row.get("task_id") == record["task_id"], "task ID drifted")
    require(row.get("sample_index") == spec["sample_index"], "sample index drifted")
    require(row.get("generation_seed") == spec["seed"], "generation seed drifted")
    require(row.get("split") == record["split"], "split drifted")
    require(row.get("prompt") == record["prompt"], "prompt drifted")
    generator = row.get("generator")
    require(isinstance(generator, dict), "generator provenance is missing")
    require(
        generator.get("rollout_contract_sha256") == contract["contract_sha256"]
        and generator.get("shard_id") == spec["shard_id"],
        "generator contract/shard drifted",
    )
    response = row.get("response")
    require(isinstance(response, dict), "response evidence is missing")
    require(
        isinstance(response.get("text"), str)
        and response.get("sha256") == text_sha256(response["text"]),
        "response text hash drifted",
    )
    prompt_ids = response.get("prompt_token_ids")
    generated_ids = response.get("generated_token_ids")
    require(
        isinstance(prompt_ids, list)
        and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in prompt_ids)
        and response.get("prompt_token_count") == len(prompt_ids)
        and response.get("prompt_token_ids_sha256") == object_sha256(prompt_ids),
        "prompt token evidence drifted",
    )
    require(
        isinstance(generated_ids, list)
        and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in generated_ids)
        and response.get("generated_token_count") == len(generated_ids)
        and response.get("generated_token_ids_sha256") == object_sha256(generated_ids),
        "generated token evidence drifted",
    )


def build_candidate_row(
    *,
    contract: Mapping[str, Any],
    spec: Mapping[str, Any],
    message_content: str,
    prompt_token_ids: Sequence[int],
    generated_token_ids: Sequence[int],
    generated_text: str,
    finish_reason: str,
    elapsed_seconds: float,
    response_adapter: Mapping[str, Any],
    format_contract: Mapping[str, Any],
) -> dict[str, Any]:
    record = spec["record"]
    prompt_ids = list(prompt_token_ids)
    generated_ids = list(generated_token_ids)
    row: dict[str, Any] = {
        "schema_name": CANDIDATE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "candidate_id": spec["candidate_id"],
        "created_at_utc": utc_now(),
        "task_id": record["task_id"],
        "task_family_id": record["task_family_id"],
        "split": record["split"],
        "sample_index": spec["sample_index"],
        "generation_seed": spec["seed"],
        "problem": record["problem"],
        "prompt": record["prompt"],
        "code_prefix": record["code_prefix"],
        "entry_point": record["entry_point"],
        "tests": record["tests"],
        "family_keys": record["family_keys"],
        "source": record["source"],
        "response": {
            "text": generated_text,
            "sha256": text_sha256(generated_text),
            "message_content": message_content,
            "message_content_sha256": text_sha256(message_content),
            "prompt_token_ids": prompt_ids,
            "prompt_token_count": len(prompt_ids),
            "prompt_token_ids_sha256": object_sha256(prompt_ids),
            "generated_token_ids": generated_ids,
            "generated_token_count": len(generated_ids),
            "generated_token_ids_sha256": object_sha256(generated_ids),
            "finish_reason": finish_reason,
            "response_adapter": dict(response_adapter),
        },
        "format_contract": dict(format_contract),
        "generator": {
            "rollout_contract_sha256": contract["contract_sha256"],
            "candidate": contract["s1"]["candidate"],
            "model_role": contract["s1"]["model_role"],
            "downstream_key": contract["s1"]["lineage_manifest"]["downstream_key"],
            "promotion_manifest_sha256": contract["s1"]["lineage_manifest"]["promotion_manifest_sha256"],
            "merged_export_manifest_sha256": contract["s1"]["lineage_manifest"]["merged_export_manifest_sha256"],
            "merged_snapshot_sha256": contract["s1"]["merged_export"]["snapshot_sha256"],
            "source_checkpoint_integrity_sha256": contract["s1"]["source_checkpoint_integrity_sha256"],
            "source_adapter_sha256": contract["s1"]["source_adapter_sha256"],
            "lineage_manifest_file_sha256": contract["s1"]["lineage_manifest"]["file_sha256"],
            "runtime_sha256": contract["runtime"]["runtime_sha256"],
            "ms_swift_commit": contract["runtime"]["ms_swift_commit"],
            "generation": contract["generation"],
            "generation_config_sha256": object_sha256(contract["generation"]),
            "shard_id": spec["shard_id"],
            "shard_count": EXPECTED_SHARDS,
            "elapsed_seconds": round(elapsed_seconds, 6),
        },
    }
    row["candidate_sha256"] = object_sha256(row)
    validate_candidate_row(row, contract=contract, spec=spec)
    return row


def load_or_none(path: Path, *, contract: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    row = load_json(path)
    validate_candidate_row(row, contract=contract, spec=spec)
    return row


def run_shard(contract_path: Path, shard_id: int) -> None:
    require(shard_id in range(EXPECTED_SHARDS), "shard ID must be 0 or 1")
    contract, seed_manifest = verify_contract(contract_path, live=True)
    ms_swift_root = Path(os.environ.get("DAY22_MS_SWIFT_ROOT", "/root/autodl-tmp/ms-swift"))
    require(runtime_identity(ms_swift_root) == contract["runtime"], "live runtime drifted")
    specs = [item for item in expected_specs(seed_manifest["records"]) if item["shard_id"] == shard_id]
    output_root = Path(contract["output_root"])
    pending = [
        spec
        for spec in specs
        if load_or_none(row_path(output_root, shard_id, spec["candidate_id"]), contract=contract, spec=spec)
        is None
    ]
    if not pending:
        publish_shard_summary(contract, specs, shard_id)
        print(json.dumps({"shard_id": shard_id, "status": "already_complete", "rows": len(specs)}))
        return

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("USE_HF", "1")
    os.environ["USE_MCORE_GDN"] = "0"
    try:
        import torch
        from swift import get_model_processor, get_template
        from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine
        from day20_contract_v2 import Day20V2ContractError, validate_raw_code_continuation
        from qwen35_response_adapter_v3 import adapt_generated_token_decode
    except ImportError as error:
        raise Day22RolloutError("Qwen3.5 rollout runtime is unavailable") from error
    require(torch.cuda.is_available(), "CUDA is required for S1 rollout")
    require(torch.cuda.device_count() == 1, "worker must see exactly one GPU")
    model_path = Path(contract["s1"]["merged_export"]["path"])
    model, processor = get_model_processor(
        str(model_path),
        model_type="qwen3_5",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        use_hf=True,
        download_model=False,
    )
    require(
        model.__class__.__name__ == "Qwen3_5ForConditionalGeneration",
        f"wrong Qwen3.5 model class: {model.__class__.__name__}",
    )
    template = get_template(
        processor,
        max_length=4096,
        template_type="qwen3_5",
        padding_free=False,
        enable_thinking=False,
        add_non_thinking_prefix=True,
    )
    meta = getattr(template, "template_meta", None)
    require(
        getattr(meta, "template_type", None) == "qwen3_5"
        and getattr(template, "enable_thinking", None) is False
        and getattr(template, "add_non_thinking_prefix", None) is True,
        "Qwen3.5 inference template drifted",
    )
    engine = TransformersEngine(model, template=template, max_batch_size=1)
    for ordinal, spec in enumerate(pending, 1):
        seed = spec["seed"]
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        record = spec["record"]
        started = time.monotonic()
        response = engine.infer(
            [InferRequest(messages=[{"role": "user", "content": record["prompt"]["text"]}])],
            RequestConfig(
                max_tokens=GENERATION["max_new_tokens"],
                temperature=GENERATION["temperature"],
                top_p=GENERATION["top_p"],
                num_beams=GENERATION["num_beams"],
                repetition_penalty=GENERATION["repetition_penalty"],
                seed=seed,
                return_details=True,
            ),
            use_tqdm=False,
        )[0]
        require(bool(getattr(response, "choices", None)), "ms-swift returned no choice")
        choice = response.choices[0]
        prompt_ids = getattr(response, "prompt_token_ids", None)
        generated_ids = getattr(choice, "token_ids", None)
        require(isinstance(prompt_ids, list) and isinstance(generated_ids, list), "token evidence missing")
        generated_text = template.decode_generate_ids(generated_ids, first_token=False)
        require(isinstance(generated_text, str), "generated-token decode is not text")
        message_content = getattr(getattr(choice, "message", None), "content", "") or ""
        require(isinstance(message_content, str), "message content is not text")
        adapter = adapt_generated_token_decode(
            message_content,
            generated_token_ids=generated_ids,
            generated_only_text=generated_text,
        )
        try:
            validated = validate_raw_code_continuation(generated_text, record["code_prefix"])
            format_contract: dict[str, Any] = {
                "valid": True,
                "execution_eligible": True,
                "validator": "validate_raw_code_continuation",
                "evidence": validated.as_evidence(),
                "error": None,
            }
        except Day20V2ContractError as error:
            format_contract = {
                "valid": False,
                "execution_eligible": False,
                "validator": "validate_raw_code_continuation",
                "evidence": None,
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        row = build_candidate_row(
            contract=contract,
            spec=spec,
            message_content=message_content,
            prompt_token_ids=prompt_ids,
            generated_token_ids=generated_ids,
            generated_text=generated_text,
            finish_reason=str(getattr(choice, "finish_reason", None) or "unknown"),
            elapsed_seconds=time.monotonic() - started,
            response_adapter=adapter,
            format_contract=format_contract,
        )
        destination = row_path(output_root, shard_id, spec["candidate_id"])
        write_json_new(destination, row)
        print(
            json.dumps(
                {
                    "shard_id": shard_id,
                    "completed_this_run": ordinal,
                    "pending_at_start": len(pending),
                    "candidate_id": spec["candidate_id"],
                    "tokens": len(generated_ids),
                    "format_valid": format_contract["valid"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    publish_shard_summary(contract, specs, shard_id)


def publish_shard_summary(
    contract: Mapping[str, Any], specs: Sequence[Mapping[str, Any]], shard_id: int
) -> None:
    output_root = Path(contract["output_root"])
    rows: list[dict[str, Any]] = []
    for spec in specs:
        row = load_or_none(
            row_path(output_root, shard_id, spec["candidate_id"]),
            contract=contract,
            spec=spec,
        )
        require(row is not None, f"shard {shard_id} is incomplete")
        rows.append(row)
    summary: dict[str, Any] = {
        "schema_name": SHARD_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "shard_id": shard_id,
        "shard_count": EXPECTED_SHARDS,
        "rollout_contract_sha256": contract["contract_sha256"],
        "rows": len(rows),
        "candidate_ids_sha256": object_sha256([row["candidate_id"] for row in rows]),
        "candidate_rows_sha256": object_sha256([row["candidate_sha256"] for row in rows]),
    }
    summary["summary_sha256"] = object_sha256(summary)
    path = output_root / "shards" / f"shard-{shard_id}" / "SHARD-COMPLETE.json"
    publish_or_verify_json(path, summary, "summary_sha256")


def rollout_status(contract_path: Path) -> dict[str, Any]:
    contract, seed_manifest = verify_contract(contract_path, live=False)
    output_root = Path(contract["output_root"])
    result: dict[str, Any] = {
        "rollout_contract_sha256": contract["contract_sha256"],
        "expected": EXPECTED_FAMILIES * EXPECTED_K,
        "shards": {},
    }
    total = 0
    for shard_id in range(EXPECTED_SHARDS):
        specs = [item for item in expected_specs(seed_manifest["records"]) if item["shard_id"] == shard_id]
        present = 0
        for spec in specs:
            if load_or_none(
                row_path(output_root, shard_id, spec["candidate_id"]),
                contract=contract,
                spec=spec,
            ) is not None:
                present += 1
        result["shards"][str(shard_id)] = {"expected": len(specs), "present": present}
        total += present
    result["present"] = total
    result["complete"] = total == result["expected"]
    return result


def merge_shards(contract_path: Path, output_jsonl: Path, output_manifest: Path) -> None:
    contract, seed_manifest = verify_contract(contract_path, live=False)
    output_root = Path(contract["output_root"])
    by_id: dict[str, dict[str, Any]] = {}
    ordered_specs = expected_specs(seed_manifest["records"])
    for spec in ordered_specs:
        row = load_or_none(
            row_path(output_root, spec["shard_id"], spec["candidate_id"]),
            contract=contract,
            spec=spec,
        )
        require(row is not None, f"missing rollout candidate: {spec['candidate_id']}")
        require(spec["candidate_id"] not in by_id, "duplicate rollout candidate")
        by_id[spec["candidate_id"]] = row
    rows = [by_id[spec["candidate_id"]] for spec in ordered_specs]
    require(len(rows) == EXPECTED_FAMILIES * EXPECTED_K, "merged rollout count drifted")
    jsonl_payload = b"".join(canonical_json(row) + b"\n" for row in rows)
    jsonl_sha256 = hashlib.sha256(jsonl_payload).hexdigest()
    manifest: dict[str, Any] = {
        "schema_name": MERGE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "rollout_contract_sha256": contract["contract_sha256"],
        "source_seed_manifest_sha256": seed_manifest["manifest_sha256"],
        "s1_candidate": WINNER,
        "merged_s1_snapshot_sha256": contract["s1"]["merged_export"]["snapshot_sha256"],
        "families": EXPECTED_FAMILIES,
        "samples_per_family": EXPECTED_K,
        "rows": len(rows),
        "candidate_ids_sha256": object_sha256([row["candidate_id"] for row in rows]),
        "candidate_rows_sha256": object_sha256([row["candidate_sha256"] for row in rows]),
        "generation_seeds_sha256": object_sha256([row["generation_seed"] for row in rows]),
        "output": {
            "path": str(output_jsonl.resolve()),
            "bytes": len(jsonl_payload),
            "file_sha256": jsonl_sha256,
        },
    }
    manifest["manifest_sha256"] = object_sha256(manifest)
    if output_jsonl.exists() or output_manifest.exists():
        require(output_jsonl.is_file() and output_manifest.is_file(), "existing merge pair is incomplete")
        require(file_sha256(output_jsonl) == jsonl_sha256, "existing merged JSONL drifted")
        existing = load_json(output_manifest)
        verify_self_hash(existing, "manifest_sha256")
        require(existing == manifest, "existing merge manifest drifted")
        print(json.dumps(manifest, sort_keys=True))
        return
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_jsonl.name}.", suffix=".attempt", dir=output_jsonl.parent
    )
    temporary_jsonl = Path(temporary_name)
    temporary_manifest = output_manifest.with_name(f".{output_manifest.name}.{os.getpid()}.attempt")
    linked_jsonl = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(jsonl_payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with temporary_manifest.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary_jsonl, output_jsonl)
        linked_jsonl = True
        try:
            os.link(temporary_manifest, output_manifest)
        except BaseException:
            output_jsonl.unlink(missing_ok=False)
            linked_jsonl = False
            raise
    except FileExistsError as error:
        raise Day22RolloutError("merge output appeared during publication") from error
    finally:
        temporary_jsonl.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
        if linked_jsonl and not output_manifest.exists():
            output_jsonl.unlink(missing_ok=True)
    print(json.dumps(manifest, sort_keys=True))


def prepare_contract(args: argparse.Namespace) -> None:
    path = args.output_root.resolve() / "rollout-contract.json"
    if path.exists():
        contract, _ = verify_contract(path, live=True)
        require(
            Path(contract["seed_pool"]["path"]) == args.seed_manifest.resolve(),
            "existing contract uses a different seed manifest",
        )
        require(
            Path(contract["s1"]["merged_export"]["path"]) == args.model.resolve(),
            "existing contract uses a different merged S1 export",
        )
        require(
            Path(contract["s1"]["lineage_manifest"]["path"])
            == args.lineage_manifest.resolve(),
            "existing contract uses a different promotion manifest",
        )
        print(
            json.dumps(
                {"contract": str(path), "sha256": contract["contract_sha256"], "status": "verified"},
                sort_keys=True,
            )
        )
        return
    seed_path = args.seed_manifest.resolve()
    seed = validate_seed_manifest(seed_path)
    model_path = args.model.resolve()
    model_files = model_file_manifest(model_path)
    runtime = runtime_identity(args.ms_swift_root)
    lineage = lineage_manifest_identity(args.lineage_manifest)
    require(
        Path(lineage["merged_export_path"]) == model_path,
        "--model differs from the downstream-ready S1 inference export",
    )
    contract = build_contract(
        seed_path=seed_path,
        seed_manifest=seed,
        model_path=model_path,
        model_files=model_files,
        lineage_identity=lineage,
        output_root=args.output_root,
        runtime=runtime,
        implementation_sha256=file_sha256(Path(__file__).resolve()),
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root.resolve() / "rollout-contract.json"
    publish_or_verify_json(path, contract, "contract_sha256")
    print(json.dumps({"contract": str(path.resolve()), "sha256": contract["contract_sha256"]}, sort_keys=True))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="freeze S1/model/data/generation provenance")
    prepare.add_argument("--seed-manifest", required=True, type=Path)
    prepare.add_argument("--model", required=True, type=Path, help="merged S1 inference export")
    prepare.add_argument("--lineage-manifest", required=True, type=Path)
    prepare.add_argument("--output-root", required=True, type=Path)
    prepare.add_argument(
        "--ms-swift-root", type=Path, default=Path("/root/autodl-tmp/ms-swift")
    )
    run = commands.add_parser("run", help="run or resume one deterministic GPU shard")
    run.add_argument("--contract", required=True, type=Path)
    run.add_argument("--shard-id", required=True, type=int, choices=(0, 1))
    status = commands.add_parser("status", help="validate and count durable candidate rows")
    status.add_argument("--contract", required=True, type=Path)
    merge = commands.add_parser("merge", help="require both shards and publish ordered JSONL")
    merge.add_argument("--contract", required=True, type=Path)
    merge.add_argument("--output-jsonl", required=True, type=Path)
    merge.add_argument("--output-manifest", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        prepare_contract(args)
    elif args.command == "run":
        run_shard(args.contract, args.shard_id)
    elif args.command == "status":
        print(json.dumps(rollout_status(args.contract), sort_keys=True))
    elif args.command == "merge":
        merge_shards(args.contract, args.output_jsonl, args.output_manifest)
    else:  # pragma: no cover
        raise Day22RolloutError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
