#!/usr/bin/env python3
"""Linux/Hopper runtime and two-rank topology preflight for Day18."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


EXPECTED_VERSIONS = {
    "ms-swift": "4.5.0.dev0",
    "transformers": "5.12.1",
    "megatron-core": "0.18.0",
    "mcore-bridge": "1.6.0",
    "transformer-engine": "2.16.0",
    "transformer-engine-torch": "2.16.0",
    "transformer-engine-cu12": "2.16.0",
    "flash-linear-attention": "0.4.2",
    "flash-attn": "2.8.3",
    "qwen-vl-utils": "0.0.14",
    "datasets": "4.8.4",
    "peft": "0.19.1",
    "trl": "0.29.1",
    "accelerate": "1.14.0",
    "causal-conv1d": "1.6.2.post1",
}
EXPECTED_BOOT_IMAGE_REFERENCE = (
    "autodl-observed-runtime:ubuntu22.04.4-py3.12.3-torch2.5.1+cu124-cuda12.4"
)
EXPECTED_BOOT_RUNTIME = {
    "os": "Ubuntu 22.04.4 LTS",
    "python": "3.12.3",
    "torch": "2.5.1+cu124",
    "torch_cuda": "12.4",
    "system_cuda_toolkit": "/usr/local/cuda-12.4",
}
SUPPORTED_GPU_MODELS = ("H800", "H100")
EXPECTED_COMPUTE_CAPABILITY = (9, 0)
MINIMUM_GPU_MEMORY_MIB = 80000


def gpu_model_supported(name: str) -> bool:
    return any(model in name for model in SUPPORTED_GPU_MODELS)


def inspect_boot_runtime() -> Dict[str, str]:
    os_release = {}
    for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", maxsplit=1)
            os_release[key] = value.strip().strip('"')
    base_python = subprocess.check_output(
        [
            "/root/miniconda3/bin/python",
            "-c",
            (
                "import json,platform,torch; "
                "print(json.dumps({'python':platform.python_version(),"
                "'torch':torch.__version__,'torch_cuda':torch.version.cuda}))"
            ),
        ],
        text=True,
    )
    payload = json.loads(base_python)
    payload["os"] = os_release.get("PRETTY_NAME", "")
    payload["system_cuda_toolkit"] = str(Path("/usr/local/cuda").resolve())
    return payload


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def package_version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_preflight(model_dir: Path, output: Path) -> None:
    failures: List[str] = []
    versions = {name: package_version(name) for name in EXPECTED_VERSIONS}
    for name, expected in EXPECTED_VERSIONS.items():
        if versions[name] != expected:
            failures.append(f"{name}: expected {expected}, got {versions[name]}")
    if sys.version_info[:2] != (3, 12):
        failures.append(f"Python must be 3.12.x, got {platform.python_version()}")
    if os.environ.get("USE_MCORE_GDN") != "1":
        failures.append("USE_MCORE_GDN must be exactly 1")
    boot_image_reference = os.environ.get("DAY18_BOOT_IMAGE_REFERENCE")
    if boot_image_reference != EXPECTED_BOOT_IMAGE_REFERENCE:
        failures.append(
            "AutoDL boot image reference drift: "
            f"expected {EXPECTED_BOOT_IMAGE_REFERENCE}, got {boot_image_reference!r}"
        )
    boot_runtime: Dict[str, str] = {}
    try:
        boot_runtime = inspect_boot_runtime()
        for key, expected in EXPECTED_BOOT_RUNTIME.items():
            if boot_runtime.get(key) != expected:
                failures.append(
                    f"AutoDL boot runtime drift for {key}: expected {expected}, got {boot_runtime.get(key)!r}"
                )
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        failures.append(f"AutoDL boot runtime inspection failed: {type(exc).__name__}: {exc}")

    import torch
    from transformers import AutoConfig, AutoProcessor

    if not torch.__version__.startswith("2.10.0"):
        failures.append(f"isolated runtime requires torch 2.10.0 build, got {torch.__version__}")
    if torch.version.cuda != "12.6":
        failures.append(f"isolated runtime requires torch CUDA 12.6, got {torch.version.cuda}")
    if not torch.cuda.is_available():
        failures.append("CUDA is unavailable")
    if torch.cuda.device_count() != 2:
        failures.append(f"expected exactly two visible GPUs, got {torch.cuda.device_count()}")
    gpu_inventory = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        memory_mib = properties.total_memory // (1024 * 1024)
        capability = (properties.major, properties.minor)
        gpu_inventory.append(
            {
                "index": index,
                "name": properties.name,
                "compute_capability": list(capability),
                "memory_mib": memory_mib,
            }
        )
        if not gpu_model_supported(properties.name):
            failures.append(f"GPU {index}: expected H800 or H100, got {properties.name}")
        if capability != EXPECTED_COMPUTE_CAPABILITY:
            failures.append(
                f"GPU {index}: expected compute capability {EXPECTED_COMPUTE_CAPABILITY}, got {capability}"
            )
        if memory_mib < MINIMUM_GPU_MEMORY_MIB:
            failures.append(
                f"GPU {index}: expected at least {MINIMUM_GPU_MEMORY_MIB} MiB, got {memory_mib} MiB"
            )
    gpu_names = [row["name"] for row in gpu_inventory]
    physical_gpu_inventory: List[Dict[str, Any]] = []
    try:
        nvidia_smi_output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,mig.mode.current",
                "--format=csv,noheader",
            ],
            text=True,
        )
        for line in nvidia_smi_output.splitlines():
            index, uuid, mig_mode = (value.strip() for value in line.split(",", maxsplit=2))
            physical_gpu_inventory.append(
                {"index": int(index), "uuid": uuid, "mig_mode": mig_mode}
            )
        if len(physical_gpu_inventory) != 2:
            failures.append(
                f"expected exactly two physical GPUs from nvidia-smi, got {len(physical_gpu_inventory)}"
            )
        if len({row["uuid"] for row in physical_gpu_inventory}) != len(physical_gpu_inventory):
            failures.append("nvidia-smi returned duplicate physical GPU UUIDs")
        for row in physical_gpu_inventory:
            if not row["uuid"].startswith("GPU-"):
                failures.append(f"GPU {row['index']}: expected a physical GPU UUID, got {row['uuid']}")
            if row["mig_mode"].lower() != "disabled":
                failures.append(
                    f"GPU {row['index']}: MIG must be disabled, got {row['mig_mode']}"
                )
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        failures.append(f"physical GPU inventory failed: {type(exc).__name__}: {exc}")
    if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        failures.append("CUDA BF16 is unsupported")

    config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    architectures = list(getattr(config, "architectures", []) or [])
    if "Qwen3_5ForConditionalGeneration" not in architectures:
        failures.append(f"wrong architecture: {architectures}")
    if processor.__class__.__name__ != "Qwen3VLProcessor":
        failures.append(f"wrong processor: {processor.__class__.__name__}")
    resolved_mtp_layers = getattr(getattr(config, "text_config", None), "mtp_num_hidden_layers", None)
    if resolved_mtp_layers != 1:
        failures.append(f"text_config.mtp_num_hidden_layers: expected 1, got {resolved_mtp_layers}")
    special_token_ids = {
        "vision_start_token_id": 248053,
        "vision_end_token_id": 248054,
        "image_token_id": 248056,
        "video_token_id": 248057,
    }
    for key, expected in special_token_ids.items():
        if getattr(config, key, None) != expected:
            failures.append(f"{key}: expected {expected}, got {getattr(config, key, None)}")

    imported_modules: Dict[str, str] = {}
    for module_name in (
        "swift",
        "megatron.core",
        "mcore_bridge",
        "transformer_engine",
        "transformer_engine.pytorch",
        "fla",
        "flash_attn",
        "causal_conv1d",
        "qwen_vl_utils",
        "peft",
        "trl",
        "accelerate",
    ):
        try:
            module = importlib.import_module(module_name)
            imported_modules[module_name] = str(
                getattr(module, "__file__", None)
                or getattr(getattr(module, "__spec__", None), "origin", None)
                or "namespace-package"
            )
        except Exception as exc:  # noqa: BLE001 - every failed CUDA extension import is evidence.
            failures.append(f"import {module_name} failed: {type(exc).__name__}: {exc}")

    gdn_sources: List[str] = []
    bridge_path = imported_modules.get("mcore_bridge")
    if bridge_path:
        bridge_root = Path(bridge_path).resolve().parent
        for path in bridge_root.rglob("*.py"):
            lowered = path.name.lower()
            if "qwen" not in lowered and "gdn" not in lowered:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if "USE_MCORE_GDN" in text or "GatedDeltaNet" in text:
                gdn_sources.append(str(path))
    if not gdn_sources:
        failures.append("could not evidence the mcore-bridge Qwen3.5/GDN source path")

    nccl_version: Any = None
    if torch.cuda.is_available():
        try:
            nccl_version = torch.cuda.nccl.version()
        except Exception as exc:  # noqa: BLE001
            failures.append(f"NCCL version query failed: {exc}")

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "boot_image_reference": boot_image_reference,
        "boot_runtime": boot_runtime,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "nccl": nccl_version,
        "gpu_names": gpu_names,
        "gpu_inventory": gpu_inventory,
        "physical_gpu_inventory": physical_gpu_inventory,
        "bf16_supported": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "packages": versions,
        "imports": imported_modules,
        "gdn_sources": sorted(gdn_sources),
        "use_mcore_gdn": os.environ.get("USE_MCORE_GDN"),
        "model_path": str(model_dir.resolve()),
        "architecture": architectures,
        "processor_class": processor.__class__.__name__,
        "special_token_ids": {key: getattr(config, key, None) for key in special_token_ids},
        "mtp_num_hidden_layers": resolved_mtp_layers,
    }
    write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    if failures:
        raise SystemExit(1)


def topology_preflight(output_dir: Path, expected_world_size: int) -> None:
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    failures: List[str] = []
    if world_size != expected_world_size:
        failures.append(f"expected world_size={expected_world_size}, got {world_size}")
    name = torch.cuda.get_device_name(local_rank)
    properties = torch.cuda.get_device_properties(local_rank)
    capability = (properties.major, properties.minor)
    memory_mib = properties.total_memory // (1024 * 1024)
    if not gpu_model_supported(name):
        failures.append(f"rank {rank} is not on H800/H100: {name}")
    if capability != EXPECTED_COMPUTE_CAPABILITY:
        failures.append(
            f"rank {rank} compute capability expected {EXPECTED_COMPUTE_CAPABILITY}, got {capability}"
        )
    if memory_mib < MINIMUM_GPU_MEMORY_MIB:
        failures.append(
            f"rank {rank} GPU memory expected at least {MINIMUM_GPU_MEMORY_MIB} MiB, got {memory_mib} MiB"
        )
    probe = torch.tensor([rank + 1.0], device="cuda")
    dist.all_reduce(probe)
    expected_sum = world_size * (world_size + 1) / 2
    if probe.item() != expected_sum:
        failures.append(f"NCCL all_reduce expected {expected_sum}, got {probe.item()}")
    cuda_uuid = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", str(local_rank)],
        text=True,
    ).strip()
    if not cuda_uuid:
        failures.append(f"rank {rank} could not resolve its CUDA UUID")
    payload = {
        "schema_version": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "hostname": socket.gethostname(),
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "cuda_device": name,
        "cuda_uuid": cuda_uuid,
        "compute_capability": list(capability),
        "memory_mib": memory_mib,
        "all_reduce_sum": probe.item(),
    }
    write_json(output_dir / f"rank-{rank}.json", payload)
    dist.barrier()
    if rank == 0:
        rows = [json.loads((output_dir / f"rank-{idx}.json").read_text(encoding="utf-8")) for idx in range(world_size)]
        summary_failures = [failure for row in rows for failure in row["failures"]]
        if len({row["hostname"] for row in rows}) != 1:
            summary_failures.append("ranks are not on the same host")
        if len({row["cuda_uuid"] for row in rows}) != world_size:
            summary_failures.append("ranks did not bind to unique GPUs")
        write_json(
            output_dir / "summary.json",
            {
                "schema_version": 1,
                "status": "pass" if not summary_failures else "fail",
                "failures": summary_failures,
                "ranks": rows,
            },
        )
    dist.barrier()
    if failures:
        raise SystemExit(1)
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    runtime = subparsers.add_parser("runtime")
    runtime.add_argument("--model-dir", required=True, type=Path)
    runtime.add_argument("--output", required=True, type=Path)
    topology = subparsers.add_parser("topology")
    topology.add_argument("--output-dir", required=True, type=Path)
    topology.add_argument("--expected-world-size", type=int, default=2)
    args = parser.parse_args()
    if args.command == "runtime":
        runtime_preflight(args.model_dir, args.output)
    else:
        topology_preflight(args.output_dir, args.expected_world_size)


if __name__ == "__main__":
    main()
