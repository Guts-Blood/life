#!/usr/bin/env python3
"""Build and verify the fail-closed RSI v0002 downstream-ready S1 handoff.

The four stages are intentionally separate: recover, export-manifest, capture/
parity, and promote.  Every published JSON is no-overwrite and self-hashed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
WINNER = "main-s20260809-lr1e-4-final"
CONFIRMATION = "main-s20260810-lr1e-4-final"
BASE_MODEL_ID = "Qwen/Qwen3.5-4B-Base"
BASE_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
BASE_SNAPSHOT_SHA256 = "a176a3c982da5480a0ca98280848474133d105d2e4ab45dc37ce2c68a7c2195b"
CHECKPOINT_INTEGRITY_SHA256 = "d0f72be9751628c9073cc8e4104f16d8620bd598dbb8a1e97f1df9bae51170a3"
CHECKPOINT_SNAPSHOT_SHA256 = "c169e0bb55b20951d27a889e4ef0deae3a01fe10acabee804b212d3d8170db0a"
ADAPTER_SHA256 = "29dc1a7d676b7f2f88675521bc44cf81dd43b700423df3cf4adf8d6a1c4487c3"
INTEGRITY_FILE_SHA256 = "f7880e6f65664b96a1a579fc3c8599e44eff385683c1af08fe7c965f060d3e58"
ENVELOPE_SHA256 = "4497af38b3daed474b2d9884e7651205985fa3cd31d4d2409d629bbafdecaf7c"
ENVELOPE_FILE_SHA256 = "481017dc8e40016522be3bb98803401111f7094d63ccf0b340c80b8cfb69c4ed"
TRAINING_SUMMARY_SHA256 = "cd56a594ea04f3c5dd43644a66fe875e2dc4f86b12b13eec321d293aa8fb9cfb"
TRAINING_SUMMARY_FILE_SHA256 = "fc5dc58a4df06ffa3b46a773f1f69e4b8a149ce2d77f7ab0096e42a44b3ae4cf"
FINAL_PROMOTION_SHA256 = "de72a8f43da23e955e9525bccb365b49edcba7976018d6b58a9b34f903c2caa7"
FINAL_PROMOTION_FILE_SHA256 = "c5ce7f522706040ac1628659ccc1ee53c2ac61eb820861a092a5dcd7bf8049e4"
COLLECTOR_INVENTORY_SHA256 = "9d6a78e9954e37e74fae9698ad6729dc253cb3c8156f3008d86ad24ed7ccab5b"
CORE_RUNTIME_SHA256 = "6e1018b2528288e459a0e5707468ac8ede284ca497e03e30f91139ff2b3cc2d8"
TARGET_RUNTIME_SHA256 = "f5117f90f30e8209ab2cc3b9c2d9761c81e718cdbe29cfb1ad3e638ae3c8edf5"
MS_SWIFT_COMMIT = "565a1ad586a21d24b23931c52d2c62b49c39bee8"
INFERENCE_TEMPLATE = "qwen3_5"
TRAINING_TEMPLATE = "day20_qwen3_5_target_v3"
DOWNSTREAM_KEY = f"s1:qwen35-4b:{CHECKPOINT_SNAPSHOT_SHA256}"
EXPECTED_RUNTIME_VERSIONS = {
    "ms-swift": "4.5.0.dev0",
    "peft": "0.19.1",
    "torch": "2.10.0+cu128",
    "transformers": "5.12.1",
}
FIXED_PROMPTS = (
    "Write a Python function add(a, b) that returns a + b. Return only Python code.",
    "Write a Python function is_even(n) that returns whether n is even. Return only Python code.",
    "Write a Python function reverse_string(text) that returns text reversed. Return only Python code.",
    "Write a Python function factorial(n) for non-negative integers. Return only Python code.",
)
GENERATION = {
    "max_tokens": 64,
    "temperature": 0,
    "num_beams": 1,
    "repetition_penalty": 1.0,
    "seed": 20260813,
}


class HandoffError(ValueError):
    """A handoff input or evidence artifact violates the frozen contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def object_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise HandoffError(f"required regular file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise HandoffError(f"required JSON is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HandoffError(f"cannot load JSON: {path}") from error
    if not isinstance(value, dict):
        raise HandoffError(f"JSON root must be an object: {path}")
    return value


def verify_self_hash(value: Mapping[str, Any], field: str) -> str:
    expected = value.get(field)
    actual = object_sha256({key: item for key, item in value.items() if key != field})
    if not isinstance(expected, str) or expected != actual:
        raise HandoffError(f"{field} mismatch")
    return expected


def write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".attempt", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise HandoffError(f"refusing to overwrite artifact: {destination}") from error
    finally:
        temporary.unlink(missing_ok=True)


def file_manifest(directory: Path, *, exclude: Sequence[str] = ()) -> dict[str, dict[str, Any]]:
    root = directory.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise HandoffError(f"required regular directory is missing: {root}")
    excluded = set(exclude)
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise HandoffError(f"symbolic links are forbidden in packages: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise HandoffError(f"non-regular package entry: {path}")
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        size = path.stat().st_size
        if size <= 0:
            raise HandoffError(f"empty package file: {path}")
        files[relative] = {"bytes": size, "sha256": file_sha256(path)}
    if not files:
        raise HandoffError(f"package is empty: {root}")
    return files


def seal(value: dict[str, Any], field: str) -> dict[str, Any]:
    if field in value:
        raise HandoffError(f"refusing to reseal {field}")
    value[field] = object_sha256(value)
    return value


def add_day20_imports(bootcamp_root: Path) -> None:
    for path in (
        bootcamp_root / "day-20-qwen35-balanced-lora-sft",
        bootcamp_root / "rsi-control",
    ):
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def collect_rsi_inventory_isolated(
    *, run_root: Path, bootcamp_root: Path
) -> dict[str, Any]:
    """Run the frozen collector with a clean interpreter and parse its JSON."""

    collector = (bootcamp_root / "rsi-control/rsi_day20_collect.py").resolve()
    if not collector.is_file() or collector.is_symlink():
        raise HandoffError(f"frozen RSI collector is missing: {collector}")
    try:
        completed = subprocess.run(
            [sys.executable, str(collector), "--run-root", str(run_root.resolve())],
            cwd=str(bootcamp_root.resolve()),
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise HandoffError("isolated RSI collector process failed to execute") from error
    stdout_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not stdout_lines:
        stderr_tail = completed.stderr[-1000:].strip()
        raise HandoffError(
            "isolated RSI collector emitted no JSON"
            + (f": {stderr_tail}" if stderr_tail else "")
        )
    try:
        value = json.loads(stdout_lines[-1])
    except json.JSONDecodeError as error:
        raise HandoffError("isolated RSI collector final stdout line is not JSON") from error
    if not isinstance(value, dict):
        raise HandoffError("isolated RSI collector JSON root is not an object")
    if completed.returncode != 0:
        raise HandoffError(
            "isolated RSI collector rejected the run: "
            f"returncode={completed.returncode}, status={value.get('status')!r}, "
            f"message={value.get('message')!r}"
        )
    return value


def copy_archive_new(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination = destination.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise HandoffError(f"refusing to overwrite checkpoint archive: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.attempt-{os.getpid()}"
    if staging.exists() or staging.is_symlink():
        raise HandoffError(f"archive staging path is occupied: {staging}")
    shutil.copytree(source, staging, copy_function=shutil.copy2)
    source_files = file_manifest(source)
    archive_files = file_manifest(staging)
    if archive_files != source_files:
        raise HandoffError(f"checkpoint archive copy differs from source: {staging}")
    os.rename(staging, destination)


def build_recovery(
    *, run_root: Path, bootcamp_root: Path, archive_dir: Path, copy_archive: bool
) -> dict[str, Any]:
    root = run_root.resolve()

    if (root / ".day20-v3-run-root").read_text(encoding="utf-8").strip() != "day20-qwen35-target-encoding-v3":
        raise HandoffError("Day20 v3 run marker drifted")
    # Recreate the archived inventory in a truly clean interpreter. Importing
    # handoff/Day20 modules in the same process changes process-global template
    # state and can make a valid 44-item inventory appear to have only 34 items.
    inventory = collect_rsi_inventory_isolated(
        run_root=root, bootcamp_root=bootcamp_root
    )
    if (
        inventory.get("status") != "pass"
        or inventory.get("inventory_sha256") != COLLECTOR_INVENTORY_SHA256
    ):
        raise HandoffError(
            "recovered 44-item RSI inventory differs from archived identity: "
            f"status={inventory.get('status')!r}, "
            f"items={len(inventory.get('complete_verified_items', []))}, "
            f"sha256={inventory.get('inventory_sha256')!r}"
        )

    add_day20_imports(bootcamp_root)
    from day20_train_runtime_v2 import verify_checkpoint_integrity
    from select_day20_main_v2 import verify_final_promotion

    checkpoint = root / "adapters/main/s20260809/lr1e-4/attempt-001/checkpoint-1904"
    summary = root / "evidence/main/s20260809/lr1e-4/attempt-001/training-summary.json"
    envelope_path = (
        root
        / "evidence/main/s20260809/lr1e-4/attempt-001/checkpoint-envelopes"
        / f"{WINNER}.target-envelope-v3.json"
    )
    promotion_path = root / "FINAL-PROMOTION.json"
    integrity = verify_checkpoint_integrity(
        checkpoint,
        run_root=root,
        run_kind="main",
        seed=20260809,
        learning_rate="1e-4",
    )
    # The immutable envelope is independently bound by its archived file hash,
    # content self-hash, and the live v2 checkpoint verification below.  Do not
    # rebuild it with a later process after recovery; the v3 verifier's rebuild
    # also binds path-resolution details that are not part of the recovered
    # checkpoint bytes.
    envelope = load_json(envelope_path)
    verify_self_hash(envelope, "envelope_sha256")
    if (
        envelope.get("domain") != "day20.qwen35_target_checkpoint_envelope.v3"
        or envelope.get("status") != "complete"
        or envelope.get("candidate") != WINNER
        or envelope.get("core_v2", {}).get("checkpoint_integrity_sha256")
        != CHECKPOINT_INTEGRITY_SHA256
        or envelope.get("core_v2", {}).get("checkpoint_snapshot_sha256")
        != CHECKPOINT_SNAPSHOT_SHA256
    ):
        raise HandoffError("winner target envelope identity drifted")
    promotion = verify_final_promotion(promotion_path)
    expected = {
        "integrity_sha256": CHECKPOINT_INTEGRITY_SHA256,
        "snapshot_sha256": CHECKPOINT_SNAPSHOT_SHA256,
    }
    if any(integrity.get(key) != value for key, value in expected.items()):
        raise HandoffError("winner checkpoint identity differs from RSI v0002 ledger")
    if integrity.get("resumable") is not True or integrity.get("global_step") != 1904:
        raise HandoffError("winner checkpoint is not the resumable final checkpoint")
    if integrity.get("files", {}).get("adapter_model.safetensors", {}).get("sha256") != ADAPTER_SHA256:
        raise HandoffError("winner adapter hash differs from RSI v0002 ledger")
    if file_sha256(checkpoint / "day20-v2-checkpoint-integrity.json") != INTEGRITY_FILE_SHA256:
        raise HandoffError("winner checkpoint integrity marker file hash drifted")
    if envelope.get("envelope_sha256") != ENVELOPE_SHA256 or file_sha256(envelope_path) != ENVELOPE_FILE_SHA256:
        raise HandoffError("winner target envelope drifted")
    if file_sha256(summary) != TRAINING_SUMMARY_FILE_SHA256 or load_json(summary).get("summary_sha256") != TRAINING_SUMMARY_SHA256:
        raise HandoffError("winner training summary drifted")
    if file_sha256(promotion_path) != FINAL_PROMOTION_FILE_SHA256 or promotion.get("promotion_sha256") != FINAL_PROMOTION_SHA256:
        raise HandoffError("RSI final promotion drifted")
    decision = promotion.get("decision", {})
    if decision.get("selected_candidate") != WINNER or decision.get("confirmation_candidate") != CONFIRMATION:
        raise HandoffError("RSI final promotion selected a different candidate")
    if copy_archive:
        copy_archive_new(checkpoint, archive_dir)
    archive_files = file_manifest(archive_dir)
    if archive_files != file_manifest(checkpoint):
        raise HandoffError("checkpoint archive package differs from source")
    archive_payload_files = dict(archive_files)
    archive_payload_files.pop("day20-v2-checkpoint-integrity.json", None)
    if object_sha256(archive_payload_files) != CHECKPOINT_SNAPSHOT_SHA256:
        raise HandoffError("checkpoint archive snapshot hash drifted")
    value: dict[str, Any] = {
        "schema_name": "day21.qwen35_s1_checkpoint_archive_evidence",
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "created_at_utc": utc_now(),
        "host": socket.gethostname(),
        "candidate": WINNER,
        "source_run_root": str(root),
        "source_checkpoint": str(checkpoint),
        "archive_checkpoint": str(archive_dir.resolve()),
        "checkpoint": {
            "global_step": 1904,
            "cumulative_supervised_tokens": 320000,
            "resumable": True,
            "integrity_sha256": CHECKPOINT_INTEGRITY_SHA256,
            "snapshot_sha256": CHECKPOINT_SNAPSHOT_SHA256,
            "adapter_sha256": ADAPTER_SHA256,
            "integrity_file_sha256": INTEGRITY_FILE_SHA256,
            "files": archive_files,
        },
        "target_envelope": {
            "path": str(envelope_path),
            "file_sha256": ENVELOPE_FILE_SHA256,
            "content_sha256": ENVELOPE_SHA256,
        },
        "training_summary": {
            "path": str(summary),
            "file_sha256": TRAINING_SUMMARY_FILE_SHA256,
            "content_sha256": TRAINING_SUMMARY_SHA256,
        },
        "final_promotion": {
            "path": str(promotion_path),
            "file_sha256": FINAL_PROMOTION_FILE_SHA256,
            "content_sha256": FINAL_PROMOTION_SHA256,
        },
        "collector": {
            "status": "pass",
            "items": sum(inventory.get("counts", {}).values()),
            "inventory_sha256": COLLECTOR_INVENTORY_SHA256,
            "counts": inventory.get("counts"),
        },
        "copy_verification": "byte_manifest_exact",
    }
    return seal(value, "evidence_sha256")


def verify_recovery(path: Path) -> dict[str, Any]:
    value = load_json(path)
    verify_self_hash(value, "evidence_sha256")
    if (
        value.get("schema_name") != "day21.qwen35_s1_checkpoint_archive_evidence"
        or value.get("status") != "pass"
        or value.get("candidate") != WINNER
        or value.get("checkpoint", {}).get("integrity_sha256") != CHECKPOINT_INTEGRITY_SHA256
        or value.get("checkpoint", {}).get("snapshot_sha256") != CHECKPOINT_SNAPSHOT_SHA256
        or value.get("checkpoint", {}).get("adapter_sha256") != ADAPTER_SHA256
    ):
        raise HandoffError("checkpoint archive evidence identity drifted")
    archive = Path(str(value.get("archive_checkpoint", "")))
    if file_manifest(archive) != value.get("checkpoint", {}).get("files"):
        raise HandoffError("checkpoint archive bytes changed after publication")
    return value


def build_export_manifest(
    *, run_root: Path, export_dir: Path, recovery_path: Path, bootcamp_root: Path
) -> dict[str, Any]:
    recovery = verify_recovery(recovery_path)
    add_day20_imports(bootcamp_root)
    from day20_train_plugin import validate_hf_export

    root = run_root.resolve()
    export = export_dir.resolve()
    if export.parent != (root / "exports").resolve() or export.name != f"{WINNER}-merged":
        raise HandoffError("merged export is outside the frozen winner namespace")
    layout = validate_hf_export(export)
    files = file_manifest(export, exclude=("S1-EXPORT-MANIFEST.json",))
    if layout.get("files") != files:
        raise HandoffError("merged HF validator and handoff file manifests differ")
    for required in ("config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"):
        if required not in files:
            raise HandoffError(f"merged export is missing required asset: {required}")
    config = load_json(export / "config.json")
    if config.get("model_type") != "qwen3_5" or "Qwen3_5ForConditionalGeneration" not in config.get("architectures", []):
        raise HandoffError("merged export has the wrong Qwen3.5 architecture")
    value: dict[str, Any] = {
        "schema_name": "day21.qwen35_s1_merged_export_manifest",
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "created_at_utc": utc_now(),
        "candidate": WINNER,
        "source": {
            "recovery_evidence_path": str(recovery_path.resolve()),
            "recovery_evidence_sha256": recovery["evidence_sha256"],
            "base_model_id": BASE_MODEL_ID,
            "base_revision": BASE_REVISION,
            "base_snapshot_sha256": BASE_SNAPSHOT_SHA256,
            "checkpoint_integrity_sha256": CHECKPOINT_INTEGRITY_SHA256,
            "checkpoint_snapshot_sha256": CHECKPOINT_SNAPSHOT_SHA256,
            "adapter_sha256": ADAPTER_SHA256,
        },
        "conversion": {
            "implementation": "python -m swift.cli.export",
            "model_type": "qwen3_5",
            "merge_lora": True,
            "torch_dtype": "bfloat16",
            "source_template": TRAINING_TEMPLATE,
            "downstream_inference_template": INFERENCE_TEMPLATE,
            "ms_swift_commit": MS_SWIFT_COMMIT,
            "lineage_verified_by": "strict_fresh_process_token_id_parity",
        },
        "export_dir": str(export),
        "files": files,
        "files_sha256": object_sha256(files),
        "hf_layout": {key: item for key, item in layout.items() if key not in {"files", "files_sha256"}},
        "processor_assets_present": True,
    }
    return seal(value, "manifest_sha256")


def verify_export_manifest(path: Path) -> dict[str, Any]:
    value = load_json(path)
    verify_self_hash(value, "manifest_sha256")
    if (
        value.get("schema_name") != "day21.qwen35_s1_merged_export_manifest"
        or value.get("status") != "pass"
        or value.get("candidate") != WINNER
        or value.get("source", {}).get("adapter_sha256") != ADAPTER_SHA256
        or value.get("conversion", {}).get("downstream_inference_template") != INFERENCE_TEMPLATE
    ):
        raise HandoffError("merged export manifest identity drifted")
    export = Path(str(value.get("export_dir", "")))
    if file_manifest(export, exclude=("S1-EXPORT-MANIFEST.json",)) != value.get("files"):
        raise HandoffError("merged export bytes changed after publication")
    return value


def runtime_versions() -> dict[str, str]:
    import importlib.metadata
    import torch

    return {
        "ms-swift": importlib.metadata.version("ms-swift"),
        "peft": importlib.metadata.version("peft"),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
    }


def capture_model(
    *, mode: str, base_model: Path, checkpoint: Path, export_dir: Path
) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["USE_HF"] = "1"
    os.environ["USE_MCORE_GDN"] = "0"
    import torch
    from swift import get_model_processor, get_template
    from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine

    versions = runtime_versions()
    if versions != EXPECTED_RUNTIME_VERSIONS:
        raise HandoffError(f"parity runtime version drift: {versions}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise HandoffError("CUDA GPU0 is required for S1 parity capture")
    model_path = base_model.resolve() if mode == "adapter" else export_dir.resolve()
    model, processor = get_model_processor(
        str(model_path),
        model_type="qwen3_5",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        use_hf=True,
        download_model=False,
    )
    if model.__class__.__name__ != "Qwen3_5ForConditionalGeneration":
        raise HandoffError(f"wrong Qwen3.5 parity loader: {model.__class__.__name__}")
    template = get_template(
        processor,
        template_type=INFERENCE_TEMPLATE,
        max_length=4096,
        padding_free=False,
        enable_thinking=False,
        add_non_thinking_prefix=True,
    )
    kwargs: dict[str, Any] = {"template": template, "max_batch_size": 1}
    if mode == "adapter":
        kwargs["adapters"] = [str(checkpoint.resolve())]
    engine = TransformersEngine(model, **kwargs)
    rows: list[dict[str, Any]] = []
    for ordinal, prompt in enumerate(FIXED_PROMPTS, 1):
        response = engine.infer(
            [InferRequest(messages=[{"role": "user", "content": prompt}])],
            RequestConfig(
                max_tokens=GENERATION["max_tokens"],
                temperature=GENERATION["temperature"],
                num_beams=GENERATION["num_beams"],
                repetition_penalty=GENERATION["repetition_penalty"],
                seed=GENERATION["seed"],
                return_details=True,
            ),
            use_tqdm=False,
        )[0]
        choice = response.choices[0]
        prompt_ids = response.prompt_token_ids
        output_ids = choice.token_ids
        if not isinstance(prompt_ids, list) or not isinstance(output_ids, list):
            raise HandoffError("ms-swift omitted parity token evidence")
        rows.append(
            {
                "ordinal": ordinal,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt_token_ids": prompt_ids,
                "prompt_token_ids_sha256": object_sha256(prompt_ids),
                "output_token_ids": output_ids,
                "output_token_ids_sha256": object_sha256(output_ids),
                "finish_reason": choice.finish_reason,
            }
        )
    value: dict[str, Any] = {
        "schema_name": "day21.qwen35_s1_fresh_process_capture",
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "created_at_utc": utc_now(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "python": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "mode": mode,
        "candidate": WINNER,
        "model_path": str(model_path),
        "adapter_path": str(checkpoint.resolve()) if mode == "adapter" else None,
        "template": INFERENCE_TEMPLATE,
        "enable_thinking": False,
        "generation": GENERATION,
        "runtime_versions": versions,
        "rows": rows,
    }
    return seal(value, "capture_sha256")


def verify_capture(path: Path, expected_mode: str) -> dict[str, Any]:
    value = load_json(path)
    verify_self_hash(value, "capture_sha256")
    if (
        value.get("schema_name") != "day21.qwen35_s1_fresh_process_capture"
        or value.get("status") != "pass"
        or value.get("mode") != expected_mode
        or value.get("candidate") != WINNER
        or value.get("template") != INFERENCE_TEMPLATE
        or value.get("generation") != GENERATION
        or value.get("runtime_versions") != EXPECTED_RUNTIME_VERSIONS
        or len(value.get("rows", [])) != len(FIXED_PROMPTS)
    ):
        raise HandoffError(f"{expected_mode} fresh-process capture identity drifted")
    return value


def build_parity(
    *, adapter_capture_path: Path, merged_capture_path: Path, export_manifest_path: Path
) -> dict[str, Any]:
    adapter = verify_capture(adapter_capture_path, "adapter")
    merged = verify_capture(merged_capture_path, "merged")
    export = verify_export_manifest(export_manifest_path)
    comparisons: list[dict[str, Any]] = []
    for left, right in zip(adapter["rows"], merged["rows"], strict=True):
        prompt_exact = left["prompt_token_ids"] == right["prompt_token_ids"]
        output_exact = left["output_token_ids"] == right["output_token_ids"]
        comparison = {
            "ordinal": left["ordinal"],
            "prompt_sha256": left["prompt_sha256"],
            "prompt_token_ids_exact": prompt_exact,
            "output_token_ids_exact": output_exact,
            "adapter_prompt_token_ids_sha256": left["prompt_token_ids_sha256"],
            "merged_prompt_token_ids_sha256": right["prompt_token_ids_sha256"],
            "adapter_output_token_ids_sha256": left["output_token_ids_sha256"],
            "merged_output_token_ids_sha256": right["output_token_ids_sha256"],
            "adapter_output_tokens": len(left["output_token_ids"]),
            "merged_output_tokens": len(right["output_token_ids"]),
        }
        comparisons.append(comparison)
    if not all(row["prompt_token_ids_exact"] and row["output_token_ids_exact"] for row in comparisons):
        raise HandoffError(f"strict adapter/merged token parity failed: {comparisons}")
    if adapter.get("pid") == merged.get("pid"):
        raise HandoffError("adapter and merged captures were not fresh processes")
    value: dict[str, Any] = {
        "schema_name": "day21.qwen35_s1_adapter_merged_parity",
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "created_at_utc": utc_now(),
        "candidate": WINNER,
        "contract": {
            "fresh_process_per_mode": True,
            "same_prompt_token_ids_required": True,
            "output_token_ids_exact_required": True,
            "threshold_relaxation_allowed": False,
            "template": INFERENCE_TEMPLATE,
            "generation": GENERATION,
        },
        "adapter_capture": {
            "path": str(adapter_capture_path.resolve()),
            "file_sha256": file_sha256(adapter_capture_path),
            "content_sha256": adapter["capture_sha256"],
            "pid": adapter["pid"],
        },
        "merged_capture": {
            "path": str(merged_capture_path.resolve()),
            "file_sha256": file_sha256(merged_capture_path),
            "content_sha256": merged["capture_sha256"],
            "pid": merged["pid"],
        },
        "export_manifest_sha256": export["manifest_sha256"],
        "comparisons": comparisons,
        "passed": len(comparisons),
        "failed": 0,
    }
    return seal(value, "parity_sha256")


def verify_parity(path: Path) -> dict[str, Any]:
    value = load_json(path)
    verify_self_hash(value, "parity_sha256")
    rows = value.get("comparisons")
    if (
        value.get("schema_name") != "day21.qwen35_s1_adapter_merged_parity"
        or value.get("status") != "pass"
        or value.get("candidate") != WINNER
        or value.get("failed") != 0
        or not isinstance(rows, list)
        or len(rows) != len(FIXED_PROMPTS)
        or not all(row.get("prompt_token_ids_exact") is True and row.get("output_token_ids_exact") is True for row in rows)
    ):
        raise HandoffError("strict adapter/merged parity evidence drifted")
    return value


def build_promotion(
    *, recovery_path: Path, export_manifest_path: Path, parity_path: Path
) -> dict[str, Any]:
    recovery = verify_recovery(recovery_path)
    export = verify_export_manifest(export_manifest_path)
    parity = verify_parity(parity_path)
    value: dict[str, Any] = {
        "schema_name": "day21.qwen35_downstream_ready_s1_promotion",
        "schema_version": SCHEMA_VERSION,
        "checkpoint_id": "qwen35-4b-s1-main-s20260809-lr1e-4-final",
        "role": "S1",
        "status": "promoted",
        "downstream_ready": True,
        "downstream_key": DOWNSTREAM_KEY,
        "created_at_utc": utc_now(),
        "lineage": {
            "lineage_id": "qwen35-4b-day13-plus-v2",
            "parent_checkpoint_id": "qwen35-4b-base-s0",
            "parent_checkpoint_hash": BASE_SNAPSHOT_SHA256,
            "objective": "coding_sft",
            "model_id": BASE_MODEL_ID,
            "model_revision": BASE_REVISION,
            "architecture": "Qwen3_5ForConditionalGeneration",
            "processor_class": "Qwen3VLProcessor",
            "modality": "text_only",
            "training_scope": "text_only_coding_lora_vision_and_aligner_frozen",
        },
        "training": {
            "framework_tag": "ms-swift-4.5.0.dev0",
            "framework_sha": MS_SWIFT_COMMIT,
            "core_runtime_identity_sha256": CORE_RUNTIME_SHA256,
            "target_runtime_identity_sha256": TARGET_RUNTIME_SHA256,
            "training_template": TRAINING_TEMPLATE,
            "inference_template": INFERENCE_TEMPLATE,
            "seed": 20260809,
            "learning_rate": "1e-4",
            "trained_label_tokens": 320000,
            "optimizer_updates": 1904,
            "packing": False,
            "parameter_dtype": "bfloat16",
        },
        "state": {
            "candidate": WINNER,
            "resumable_checkpoint_path": recovery["archive_checkpoint"],
            "resumable_checkpoint_integrity_sha256": CHECKPOINT_INTEGRITY_SHA256,
            "resumable_checkpoint_snapshot_sha256": CHECKPOINT_SNAPSHOT_SHA256,
            "adapter_sha256": ADAPTER_SHA256,
            "optimizer_state_present": True,
            "scheduler_state_present": True,
            "rng_state_present": True,
            "trainer_state_present": True,
            "recovery_evidence": {
                "path": str(recovery_path.resolve()),
                "file_sha256": file_sha256(recovery_path),
                "content_sha256": recovery["evidence_sha256"],
            },
        },
        "inference_export": {
            "path": export["export_dir"],
            "artifact_hash": export["files_sha256"],
            "manifest_sha256": export["manifest_sha256"],
            "manifest": {
                "path": str(export_manifest_path.resolve()),
                "file_sha256": file_sha256(export_manifest_path),
                "content_sha256": export["manifest_sha256"],
            },
            "parity_report": {
                "path": str(parity_path.resolve()),
                "file_sha256": file_sha256(parity_path),
                "content_sha256": parity["parity_sha256"],
            },
            "resolved_loader_class": "Qwen3_5ForConditionalGeneration",
            "processor_assets_present": True,
            "text_only_modality_parity_passed": True,
            "fresh_process_exact_token_id_parity_passed": True,
        },
        "selection": {
            "decision": "promote_primary_checkpoint",
            "primary_candidate": WINNER,
            "confirmation_candidate": CONFIRMATION,
            "guardrails_passed": True,
            "frozen_suite": "fixed-full112",
            "primary_metrics": {"general": 21, "math": 21, "finance": 19, "code": 20, "total": 81},
            "confirmation_metrics": {"general": 21, "math": 21, "finance": 14, "code": 17, "total": 73},
            "final_promotion_file_sha256": FINAL_PROMOTION_FILE_SHA256,
            "final_promotion_content_sha256": FINAL_PROMOTION_SHA256,
            "collector_inventory_sha256": COLLECTOR_INVENTORY_SHA256,
        },
        "claim_boundary": {
            "fixed_full112_qualified": True,
            "independent_training_seed_same_suite_confirmation": True,
            "independent_heldout_generalization_proven": False,
            "statistically_unique_best_proven": False,
            "exact_resume_equivalence_required": False,
        },
    }
    return seal(value, "promotion_manifest_sha256")


def verify_promotion(path: Path) -> dict[str, Any]:
    value = load_json(path)
    manifest_sha = verify_self_hash(value, "promotion_manifest_sha256")
    if (
        value.get("schema_name") != "day21.qwen35_downstream_ready_s1_promotion"
        or value.get("status") != "promoted"
        or value.get("downstream_ready") is not True
        or value.get("downstream_key") != DOWNSTREAM_KEY
        or value.get("role") != "S1"
        or value.get("state", {}).get("candidate") != WINNER
        or value.get("training", {}).get("inference_template") != INFERENCE_TEMPLATE
        or value.get("inference_export", {}).get("fresh_process_exact_token_id_parity_passed") is not True
    ):
        raise HandoffError("downstream-ready S1 promotion manifest drifted")
    return value | {"_verified_promotion_manifest_sha256": manifest_sha}


def build_downstream_key(manifest_path: Path) -> dict[str, Any]:
    manifest = verify_promotion(manifest_path)
    manifest_sha = manifest.pop("_verified_promotion_manifest_sha256")
    value: dict[str, Any] = {
        "schema_name": "day21.qwen35_s1_downstream_key",
        "schema_version": SCHEMA_VERSION,
        "status": "active",
        "role": "S1",
        "checkpoint_id": manifest["checkpoint_id"],
        "downstream_key": DOWNSTREAM_KEY,
        "promotion_manifest": {
            "path": str(manifest_path.resolve()),
            "file_sha256": file_sha256(manifest_path),
            "content_sha256": manifest_sha,
        },
    }
    return seal(value, "key_sha256")


def verify_downstream_key(path: Path) -> dict[str, Any]:
    value = load_json(path)
    verify_self_hash(value, "key_sha256")
    manifest = verify_promotion(Path(value["promotion_manifest"]["path"]))
    manifest_sha = manifest["_verified_promotion_manifest_sha256"]
    if value.get("downstream_key") != DOWNSTREAM_KEY:
        raise HandoffError("S1 downstream key is not bound to the promotion manifest")
    if file_sha256(Path(value["promotion_manifest"]["path"])) != value["promotion_manifest"]["file_sha256"]:
        raise HandoffError("S1 promotion manifest file hash changed")
    return value


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    recovery = commands.add_parser("recover")
    recovery.add_argument("--run-root", required=True, type=Path)
    recovery.add_argument("--bootcamp-root", required=True, type=Path)
    recovery.add_argument("--archive-dir", required=True, type=Path)
    recovery.add_argument("--copy-archive", action="store_true")
    recovery.add_argument("--output", required=True, type=Path)
    export = commands.add_parser("export-manifest")
    export.add_argument("--run-root", required=True, type=Path)
    export.add_argument("--bootcamp-root", required=True, type=Path)
    export.add_argument("--export-dir", required=True, type=Path)
    export.add_argument("--recovery", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    capture = commands.add_parser("capture")
    capture.add_argument("--mode", required=True, choices=("adapter", "merged"))
    capture.add_argument("--base-model", required=True, type=Path)
    capture.add_argument("--checkpoint", required=True, type=Path)
    capture.add_argument("--export-dir", required=True, type=Path)
    capture.add_argument("--output", required=True, type=Path)
    parity = commands.add_parser("parity")
    parity.add_argument("--adapter-capture", required=True, type=Path)
    parity.add_argument("--merged-capture", required=True, type=Path)
    parity.add_argument("--export-manifest", required=True, type=Path)
    parity.add_argument("--output", required=True, type=Path)
    promote = commands.add_parser("promote")
    promote.add_argument("--recovery", required=True, type=Path)
    promote.add_argument("--export-manifest", required=True, type=Path)
    promote.add_argument("--parity", required=True, type=Path)
    promote.add_argument("--manifest-output", required=True, type=Path)
    promote.add_argument("--key-output", required=True, type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--manifest", required=True, type=Path)
    verify.add_argument("--key", required=True, type=Path)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "recover":
            value = build_recovery(
                run_root=args.run_root,
                bootcamp_root=args.bootcamp_root,
                archive_dir=args.archive_dir,
                copy_archive=args.copy_archive,
            )
            write_json_new(args.output, value)
        elif args.command == "export-manifest":
            value = build_export_manifest(
                run_root=args.run_root,
                export_dir=args.export_dir,
                recovery_path=args.recovery,
                bootcamp_root=args.bootcamp_root,
            )
            write_json_new(args.output, value)
        elif args.command == "capture":
            value = capture_model(
                mode=args.mode,
                base_model=args.base_model,
                checkpoint=args.checkpoint,
                export_dir=args.export_dir,
            )
            write_json_new(args.output, value)
        elif args.command == "parity":
            value = build_parity(
                adapter_capture_path=args.adapter_capture,
                merged_capture_path=args.merged_capture,
                export_manifest_path=args.export_manifest,
            )
            write_json_new(args.output, value)
        elif args.command == "promote":
            value = build_promotion(
                recovery_path=args.recovery,
                export_manifest_path=args.export_manifest,
                parity_path=args.parity,
            )
            write_json_new(args.manifest_output, value)
            key = build_downstream_key(args.manifest_output)
            write_json_new(args.key_output, key)
            value = {"manifest": value, "key": key}
        else:
            value = {
                "manifest": verify_promotion(args.manifest),
                "key": verify_downstream_key(args.key),
            }
    except (HandoffError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
