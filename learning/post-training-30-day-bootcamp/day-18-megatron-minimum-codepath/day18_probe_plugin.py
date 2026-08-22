"""External ms-swift callback that fails closed on Day18 rank/state ownership."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.distributed as dist
from megatron.core import mpu
from swift.megatron.callbacks import MegatronCallback, megatron_callbacks_map


def is_visual_name(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in ("visual", "vision", "aligner", "merger"))


def is_mtp_name(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in ("mtp", "multi_token", "multi-token"))


def parameter_rows(models: Iterable[torch.nn.Module]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for chunk, model in enumerate(models):
        for name, parameter in model.named_parameters():
            rows.append(
                {
                    "chunk": chunk,
                    "name": name,
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                    "requires_grad": parameter.requires_grad,
                    "is_visual_or_aligner": is_visual_name(name),
                    "is_mtp": is_mtp_name(name),
                }
            )
    return rows


def layout_sha256(rows: List[Dict[str, Any]]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def gdn_modules(models: Iterable[torch.nn.Module]) -> List[Dict[str, str]]:
    result: List[Dict[str, str]] = []
    markers = ("gateddeltanet", "gated_delta", "deltanet", "delta_net", "gdn")
    for chunk, model in enumerate(models):
        for name, module in model.named_modules():
            identity = f"{name} {module.__class__.__module__} {module.__class__.__name__}".lower()
            if not any(marker in identity for marker in markers):
                continue
            try:
                source = inspect.getfile(module.__class__)
            except (OSError, TypeError):
                source = "unknown"
            result.append(
                {
                    "chunk": str(chunk),
                    "name": name,
                    "class": f"{module.__class__.__module__}.{module.__class__.__name__}",
                    "source": source,
                }
            )
    return result


def gradient_counts(models: Iterable[torch.nn.Module]) -> Tuple[int, int]:
    trainable_with_gradient = 0
    visual_with_gradient = 0
    for model in models:
        for name, parameter in model.named_parameters():
            gradient = parameter.grad
            if gradient is None:
                gradient = getattr(parameter, "main_grad", None)
            if gradient is None:
                continue
            if parameter.requires_grad:
                trainable_with_gradient += 1
            if is_visual_name(name):
                visual_with_gradient += 1
    return trainable_with_gradient, visual_with_gradient


def mtp_gradient_stats(models: Iterable[torch.nn.Module]) -> Dict[str, int]:
    result = {
        "parameter_tensors": 0,
        "missing_gradient_tensors": 0,
        "nonfinite_gradient_tensors": 0,
        "nonzero_gradient_tensors": 0,
    }
    for model in models:
        for name, parameter in model.named_parameters():
            if not is_mtp_name(name) or not parameter.requires_grad:
                continue
            result["parameter_tensors"] += 1
            gradient = parameter.grad
            if gradient is None:
                gradient = getattr(parameter, "main_grad", None)
            if gradient is None:
                result["missing_gradient_tensors"] += 1
                continue
            if not torch.isfinite(gradient).all().item():
                result["nonfinite_gradient_tensors"] += 1
            if torch.count_nonzero(gradient).item() > 0:
                result["nonzero_gradient_tensors"] += 1
    return result


class Day18EvidenceCallback(MegatronCallback):
    def __init__(self, trainer):
        super().__init__(trainer)
        root = os.environ.get("DAY18_PLUGIN_EVIDENCE_DIR")
        if not root:
            raise RuntimeError("DAY18_PLUGIN_EVIDENCE_DIR is required")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.rank = dist.get_rank()
        self.output = self.root / f"rank-{self.rank}.json"
        self.payload: Dict[str, Any] = {}
        self._last_update_successful = None
        self._train_step_calls = 0

    def write(self) -> None:
        self.output.write_text(
            json.dumps(self.payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def install_train_step_probe(self) -> None:
        original_train_step = self.trainer.train_step

        def train_step_with_evidence(*args, **kwargs):
            result = original_train_step(*args, **kwargs)
            self._train_step_calls += 1
            self._last_update_successful = (
                result[2] if isinstance(result, tuple) and len(result) == 3 and isinstance(result[2], bool) else None
            )
            return result

        self.trainer.train_step = train_step_with_evidence

    def on_train_begin(self):
        rows = parameter_rows(self.trainer.unwrapped_models)
        visual = [row for row in rows if row["is_visual_or_aligner"]]
        trainable_visual = [row for row in visual if row["requires_grad"]]
        trainable = [row for row in rows if row["requires_grad"]]
        mtp = [row for row in rows if row["is_mtp"]]
        trainable_mtp = [row for row in mtp if row["requires_grad"]]
        gdn = gdn_modules(self.trainer.unwrapped_models)
        expected_gdn = [
            row for row in gdn
            if row["class"] == "mcore_bridge.model.modules.gated_delta_net.GatedDeltaNet"
            and "mcore_bridge" in row["source"]
        ]
        expected_tp = int(os.environ["DAY18_EXPECT_TP"])
        expected_dp = int(os.environ["DAY18_EXPECT_DP"])
        expected_start = int(os.environ.get("DAY18_EXPECT_START_ITERATION", "0"))
        failures: List[str] = []
        if os.environ.get("USE_MCORE_GDN") != "1":
            failures.append("USE_MCORE_GDN is not 1")
        if mpu.get_tensor_model_parallel_world_size() != expected_tp:
            failures.append("tensor-parallel world size mismatch")
        if mpu.get_data_parallel_world_size() != expected_dp:
            failures.append("data-parallel world size mismatch")
        if self.args.mtp_num_layers != 1:
            failures.append(f"expected mtp_num_layers=1, got {self.args.mtp_num_layers}")
        if self.state.iteration != expected_start:
            failures.append(f"expected start iteration {expected_start}, got {self.state.iteration}")
        if not trainable:
            failures.append("no trainable parameters")
        if trainable_visual:
            failures.append(f"{len(trainable_visual)} visual/aligner parameters are trainable")
        if not visual:
            failures.append("no visual/aligner parameters were found in the full conditional model")
        if not gdn:
            failures.append("no GDN module/class was found")
        elif not expected_gdn:
            failures.append("active GDN is not the pinned mcore-bridge GatedDeltaNet implementation")
        if not mtp:
            failures.append("no MCore MTP parameters were found")
        elif len(trainable_mtp) != len(mtp):
            failures.append(f"only {len(trainable_mtp)}/{len(mtp)} MTP parameter tensors are trainable")
        self.payload = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "pass" if not failures else "fail",
            "failures": failures,
            "rank": self.rank,
            "local_rank": int(os.environ["LOCAL_RANK"]),
            "world_size": dist.get_world_size(),
            "tp_rank": mpu.get_tensor_model_parallel_rank(),
            "tp_world_size": mpu.get_tensor_model_parallel_world_size(),
            "dp_rank": mpu.get_data_parallel_rank(),
            "dp_world_size": mpu.get_data_parallel_world_size(),
            "pp_rank": mpu.get_pipeline_model_parallel_rank(),
            "pp_world_size": mpu.get_pipeline_model_parallel_world_size(),
            "start_iteration": self.state.iteration,
            "freeze_vit_arg": self.args.freeze_vit,
            "freeze_aligner_arg": self.args.freeze_aligner,
            "mtp_num_layers": self.args.mtp_num_layers,
            "mtp_loss_scaling_factor": self.args.mtp_loss_scaling_factor,
            "optimizer_class": f"{self.trainer.optimizer.__class__.__module__}.{self.trainer.optimizer.__class__.__name__}",
            "parameter_count_local": len(rows),
            "trainable_parameter_tensors_local": len(trainable),
            "visual_parameter_tensors_local": len(visual),
            "trainable_visual_parameter_tensors_local": len(trainable_visual),
            "mtp_parameter_tensors_local": len(mtp),
            "trainable_mtp_parameter_tensors_local": len(trainable_mtp),
            "parameter_layout_sha256": layout_sha256(rows),
            "trainable_parameter_sample": [row["name"] for row in trainable[:50]],
            "visual_parameter_sample": [row["name"] for row in visual[:50]],
            "mtp_parameter_sample": [row["name"] for row in mtp[:50]],
            "gdn_modules": gdn,
            "expected_mcore_bridge_gdn_modules": expected_gdn,
            "steps": [],
        }
        self.install_train_step_probe()
        self.write()
        if failures:
            raise RuntimeError(f"Day18 ownership preflight failed: {failures}")

    def on_step_end(self):
        trainable_with_gradient, visual_with_gradient = gradient_counts(self.trainer.unwrapped_models)
        mtp_stats = mtp_gradient_stats(self.trainer.unwrapped_models)
        failures = self.payload["failures"]
        free_bytes, total_bytes = torch.cuda.mem_get_info(torch.cuda.current_device())
        free_fraction = free_bytes / total_bytes
        expected_train_step_calls = len(self.payload["steps"]) + 1
        if self._train_step_calls != expected_train_step_calls:
            failures.append(
                f"iteration {self.state.iteration}: expected {expected_train_step_calls} train_step calls, "
                f"got {self._train_step_calls}"
            )
        if self._last_update_successful is not True:
            failures.append(
                f"iteration {self.state.iteration}: update_successful is {self._last_update_successful!r}"
            )
        if trainable_with_gradient == 0:
            failures.append(f"iteration {self.state.iteration}: no trainable gradient/main_grad found")
        if visual_with_gradient:
            failures.append(f"iteration {self.state.iteration}: visual/aligner gradient count={visual_with_gradient}")
        if mtp_stats["missing_gradient_tensors"]:
            failures.append(
                f"iteration {self.state.iteration}: MTP tensors missing gradients={mtp_stats['missing_gradient_tensors']}"
            )
        if mtp_stats["nonfinite_gradient_tensors"]:
            failures.append(
                f"iteration {self.state.iteration}: non-finite MTP gradients={mtp_stats['nonfinite_gradient_tensors']}"
            )
        if mtp_stats["nonzero_gradient_tensors"] == 0:
            failures.append(f"iteration {self.state.iteration}: all MTP gradients are zero")
        if free_fraction < 0.10:
            failures.append(
                f"iteration {self.state.iteration}: CUDA free-memory fraction {free_fraction:.4f} is below 0.10"
            )
        self.payload["steps"].append(
            {
                "iteration": self.state.iteration,
                "update_successful": self._last_update_successful,
                "trainable_tensors_with_gradient": trainable_with_gradient,
                "visual_tensors_with_gradient": visual_with_gradient,
                "mtp_gradient_stats": mtp_stats,
                "cuda_free_bytes": free_bytes,
                "cuda_total_bytes": total_bytes,
                "cuda_free_fraction": free_fraction,
                "cuda_memory_allocated_bytes": torch.cuda.memory_allocated(),
                "cuda_memory_reserved_bytes": torch.cuda.memory_reserved(),
            }
        )
        self._last_update_successful = None
        self.payload["status"] = "pass" if not failures else "fail"
        self.write()
        if failures:
            raise RuntimeError(f"Day18 gradient ownership failed: {failures}")

    def on_train_end(self):
        expected_final = int(os.environ["DAY18_EXPECT_FINAL_ITERATION"])
        expected_iterations = list(range(self.payload["start_iteration"] + 1, expected_final + 1))
        observed_iterations = [step.get("iteration") for step in self.payload["steps"]]
        if self.state.iteration != expected_final:
            self.payload["failures"].append(
                f"expected final iteration {expected_final}, got {self.state.iteration}"
            )
        if observed_iterations != expected_iterations:
            self.payload["failures"].append(
                f"expected successful update iterations {expected_iterations}, got {observed_iterations}"
            )
        if any(step.get("update_successful") is not True for step in self.payload["steps"]):
            self.payload["failures"].append("not every recorded iteration completed a successful optimizer update")
        self.payload["final_iteration"] = self.state.iteration
        self.payload["status"] = "pass" if not self.payload["failures"] else "fail"
        self.write()
        if self.payload["failures"]:
            raise RuntimeError(f"Day18 final state failed: {self.payload['failures']}")


megatron_callbacks_map["day18_evidence"] = Day18EvidenceCallback
