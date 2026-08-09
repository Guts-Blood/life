#!/usr/bin/env python3
"""Build fail-closed runtime/source evidence for the Day 18 training codepath."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


EXPECTED_SWIFT_COMMIT = "565a1ad586a21d24b23931c52d2c62b49c39bee8"
REQUIRED_GATES = (
    "host-inventory",
    "bootstrap",
    "model-preflight",
    "runtime-preflight",
    "topology-preflight",
    "c0-c1",
    "c2-dp2",
    "c3-tp2",
    "c4-resume-export",
    "c5-tiny-overfit",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_run_file(run_root: Path, relative: str) -> Path:
    path = (run_root / relative).resolve()
    if not path.is_relative_to(run_root):
        raise ValueError(f"evidence escapes run root: {relative}")
    if not path.is_file():
        raise FileNotFoundError(f"missing runtime evidence: {relative}")
    return path


def load_json(run_root: Path, relative: str, *, require_pass: bool = False) -> Dict[str, Any]:
    path = require_run_file(run_root, relative)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{relative} is not a JSON object")
    if require_pass and payload.get("status") != "pass":
        raise ValueError(f"{relative} status is {payload.get('status')!r}, expected 'pass'")
    return payload


def require_keys(payload: Dict[str, Any], relative: str, keys: Iterable[str]) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(f"{relative} is missing fields: {missing}")


def resolve_symbol(reference: str) -> Any:
    module_name, separator, qualified_name = reference.partition(":")
    if not separator or not module_name or not qualified_name:
        raise ValueError(f"invalid symbol reference: {reference}")
    obj: Any = importlib.import_module(module_name)
    for part in qualified_name.split("."):
        obj = getattr(obj, part)
    return obj


def source_record(reference: str) -> Dict[str, Any]:
    obj = resolve_symbol(reference)
    source_name = inspect.getsourcefile(obj) or inspect.getfile(obj)
    if not source_name:
        raise ValueError(f"symbol has no inspectable source: {reference}")
    source = Path(source_name).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source for {reference} does not exist: {source}")
    line = inspect.getsourcelines(obj)[1]
    resolved_module = getattr(obj, "__module__", None)
    resolved_name = getattr(obj, "__qualname__", getattr(obj, "__name__", None))
    if not resolved_module or not resolved_name:
        raise ValueError(f"symbol identity is incomplete: {reference}")
    return {
        "requested": reference,
        "file_function": f"{source}:{resolved_name}",
        "module": resolved_module,
        "qualname": resolved_name,
        "source_file": str(source),
        "source_line": line,
        "source_sha256": sha256(source),
    }


def runtime_class_record(reference: str) -> Dict[str, Any]:
    record = source_record(reference)
    obj = resolve_symbol(reference)
    if not inspect.isclass(obj):
        raise TypeError(f"runtime class reference is not a class: {reference}")
    record["class"] = f"{obj.__module__}.{obj.__qualname__}"
    return record


def node(
    node_id: str,
    function: str,
    runtime_class: str,
    shape_rank_state: Dict[str, Any],
    evidence_paths: List[str],
    *,
    supporting_functions: Iterable[str] = (),
    supporting_runtime_classes: Iterable[str] = (),
) -> Dict[str, Any]:
    primary = source_record(function)
    runtime = runtime_class_record(runtime_class)
    return {
        "id": node_id,
        "status": "pass",
        "file_function": primary["file_function"],
        "runtime_class": runtime["class"],
        "source": primary,
        "runtime_class_source": runtime,
        "supporting_file_functions": [source_record(value) for value in supporting_functions],
        "supporting_runtime_classes": [runtime_class_record(value) for value in supporting_runtime_classes],
        "shape_rank_state": shape_rank_state,
        "evidence_paths": evidence_paths,
    }


def hf_state(payload: Dict[str, Any], relative: str) -> Dict[str, Any]:
    require_keys(payload, relative, ("batch_shape", "model_class", "model_path", "teacher_forced"))
    teacher_forced = payload["teacher_forced"]
    if not isinstance(teacher_forced, dict):
        raise ValueError(f"{relative} teacher_forced is not an object")
    require_keys(teacher_forced, f"{relative}.teacher_forced", ("supervised_tokens", "loss_weight_sum", "mean_loss"))
    batch_shape = payload["batch_shape"]
    if not isinstance(batch_shape, list) or len(batch_shape) != 2 or any(int(value) <= 0 for value in batch_shape):
        raise ValueError(f"{relative} has invalid batch_shape: {batch_shape!r}")
    return {
        "batch_shape": batch_shape,
        "supervised_tokens": teacher_forced["supervised_tokens"],
        "loss_weight_sum": teacher_forced["loss_weight_sum"],
        "mean_loss": teacher_forced["mean_loss"],
        "cuda_visible_devices": payload.get("cuda_visible_devices"),
        "model_path": payload["model_path"],
        "model_class": payload["model_class"],
    }


def distributed_state(payload: Dict[str, Any], relative: str) -> Dict[str, Any]:
    require_keys(payload, relative, ("expected", "ranks"))
    expected = payload["expected"]
    rows = payload["ranks"]
    if not isinstance(expected, dict) or not isinstance(rows, list):
        raise ValueError(f"{relative} rank summary is malformed")
    require_keys(expected, f"{relative}.expected", ("ranks", "tp", "dp", "start_iteration", "final_iteration", "steps"))
    if len(rows) != int(expected["ranks"]):
        raise ValueError(f"{relative} rank count does not match its expected envelope")
    for row in rows:
        require_keys(
            row,
            relative,
            ("rank", "parameter_layout_sha256", "optimizer_class", "expected_mcore_bridge_gdn_modules"),
        )
    return {
        "world_size": expected["ranks"],
        "tp": expected["tp"],
        "dp": expected["dp"],
        "rank_ids": sorted(row.get("rank") for row in rows),
        "start_iteration": expected["start_iteration"],
        "final_iteration": expected["final_iteration"],
        "optimizer_steps": expected["steps"],
        "parameter_layout_sha256_by_rank": {
            str(row.get("rank")): row.get("parameter_layout_sha256") for row in rows
        },
    }


def active_gdn_class(c5_ranks: Dict[str, Any]) -> str:
    modules = [
        item
        for rank in c5_ranks["ranks"]
        for item in rank.get("expected_mcore_bridge_gdn_modules", [])
    ]
    classes = {item.get("class") for item in modules}
    classes.discard(None)
    expected = {"mcore_bridge.model.modules.gated_delta_net.GatedDeltaNet"}
    if classes != expected:
        raise ValueError(f"unexpected active GDN classes: {sorted(classes)}")
    reference = "mcore_bridge.model.modules.gated_delta_net:GatedDeltaNet"
    imported_source = Path(runtime_class_record(reference)["source_file"]).resolve()
    observed_sources = {Path(item.get("source", "")).resolve() for item in modules}
    if observed_sources != {imported_source} or not imported_source.is_file():
        raise ValueError(
            f"runtime GDN source mismatch: observed={sorted(map(str, observed_sources))}, "
            f"imported={imported_source}"
        )
    return reference


def active_optimizer_class(c5_ranks: Dict[str, Any]) -> str:
    classes = {rank.get("optimizer_class") for rank in c5_ranks["ranks"]}
    classes.discard(None)
    if len(classes) != 1:
        raise ValueError(f"expected one runtime optimizer class, got {sorted(classes)}")
    dotted = classes.pop()
    module_name, separator, class_name = dotted.rpartition(".")
    if not separator:
        raise ValueError(f"invalid optimizer class: {dotted}")
    return f"{module_name}:{class_name}"


def build_manifest(run_root: Path) -> Dict[str, Any]:
    run_root = run_root.resolve()
    if not (run_root / ".day18-run-root").is_file():
        raise ValueError(f"invalid Day18 run root: {run_root}")
    if os.environ.get("USE_MCORE_GDN") != "1":
        raise ValueError("USE_MCORE_GDN must be exactly 1 while resolving the runtime codepath")

    for gate in REQUIRED_GATES:
        gate_payload = load_json(run_root, f"evidence/gates/{gate}.pass.json")
        if gate_payload.get("gate") != gate or gate_payload.get("status") != "pass":
            raise ValueError(f"invalid gate marker: {gate}")

    swift_commit_path = require_run_file(run_root, "evidence/ms-swift-commit.txt")
    swift_commit = swift_commit_path.read_text(encoding="utf-8").strip()
    if swift_commit != EXPECTED_SWIFT_COMMIT:
        raise ValueError(f"ms-swift commit drift: {swift_commit}")

    runtime = load_json(run_root, "evidence/runtime-preflight.json", require_pass=True)
    c1_hf = load_json(run_root, "evidence/c1-hf-reference.json")
    c1_conversion = load_json(run_root, "evidence/c1-conversion-parity.json", require_pass=True)
    c1_loss = load_json(run_root, "evidence/c1-single-rank-loss-parity.json", require_pass=True)
    c1_mtp = load_json(run_root, "evidence/c1-mtp-roundtrip.json", require_pass=True)
    c2_ranks = load_json(run_root, "evidence/c2-ranks-summary.json", require_pass=True)
    c2_load_ranks = load_json(run_root, "evidence/c2-loadcheck-ranks-summary.json", require_pass=True)
    c2_audit = load_json(run_root, "evidence/c2-checkpoint-audit.json", require_pass=True)
    c3_ranks = load_json(run_root, "evidence/c3-ranks-summary.json", require_pass=True)
    c4_ranks = load_json(run_root, "evidence/c4-ranks-summary.json", require_pass=True)
    c4_audit = load_json(run_root, "evidence/c4-checkpoint-audit.json", require_pass=True)
    c4_resume = load_json(run_root, "evidence/c4-resume.json", require_pass=True)
    c4_export = load_json(run_root, "evidence/c4-export-parity.json", require_pass=True)
    c4_hf = load_json(run_root, "evidence/c4-hf-reference.json")
    c5_ranks = load_json(run_root, "evidence/c5-ranks-summary.json", require_pass=True)
    c5_audit = load_json(run_root, "evidence/c5-checkpoint-audit.json", require_pass=True)
    c5_export = load_json(run_root, "evidence/c5-export-parity.json", require_pass=True)
    c5_hf = load_json(run_root, "evidence/c5-hf-reference.json")
    c5_scopes = load_json(run_root, "evidence/c5-parameter-scopes.json", require_pass=True)
    c5_overfit = load_json(run_root, "evidence/c5-tiny-overfit.json", require_pass=True)

    base_hf_state = hf_state(c1_hf, "evidence/c1-hf-reference.json")
    c4_hf_state = hf_state(c4_hf, "evidence/c4-hf-reference.json")
    final_hf_state = hf_state(c5_hf, "evidence/c5-hf-reference.json")
    c2_state = distributed_state(c2_ranks, "evidence/c2-ranks-summary.json")
    c2_load_state = distributed_state(c2_load_ranks, "evidence/c2-loadcheck-ranks-summary.json")
    c3_state = distributed_state(c3_ranks, "evidence/c3-ranks-summary.json")
    c4_state = distributed_state(c4_ranks, "evidence/c4-ranks-summary.json")
    c5_state = distributed_state(c5_ranks, "evidence/c5-ranks-summary.json")

    if runtime.get("architecture") != ["Qwen3_5ForConditionalGeneration"]:
        raise ValueError(f"unexpected conditional architecture: {runtime.get('architecture')}")
    if runtime.get("processor_class") != "Qwen3VLProcessor":
        raise ValueError(f"unexpected processor class: {runtime.get('processor_class')}")
    if base_hf_state["model_class"] != "Qwen3_5ForConditionalGeneration" or final_hf_state[
        "model_class"
    ] != "Qwen3_5ForConditionalGeneration":
        raise ValueError("HF runtime evidence did not use Qwen3_5ForConditionalGeneration")

    gdn_class = active_gdn_class(c5_ranks)
    optimizer_class = active_optimizer_class(c5_ranks)
    nodes = [
        node(
            "conditional_loader",
            "swift.model.models.qwen:Qwen3_5Loader.get_model",
            "transformers.models.qwen3_5.modeling_qwen3_5:Qwen3_5ForConditionalGeneration",
            {"shape": {"base": base_hf_state["batch_shape"], "final": final_hf_state["batch_shape"]},
             "rank": {"base_cuda_visible_devices": base_hf_state["cuda_visible_devices"],
                      "final_cuda_visible_devices": final_hf_state["cuda_visible_devices"]},
             "state": {"architecture": runtime["architecture"], "base_model": base_hf_state["model_path"],
                       "final_model": final_hf_state["model_path"]}},
            ["evidence/runtime-preflight.json", "evidence/c1-hf-reference.json", "evidence/c5-hf-reference.json"],
            supporting_runtime_classes=("swift.model.models.qwen:Qwen3_5Loader",),
        ),
        node(
            "processor_template",
            "swift.template.templates.qwen:Qwen3_5Template._swift_prepare_inputs",
            "swift.template.templates.qwen:Qwen3_5Template",
            {"shape": {"base_batch": base_hf_state["batch_shape"],
                       "supervised_tokens": base_hf_state["supervised_tokens"]},
             "rank": {"cuda_visible_devices": base_hf_state["cuda_visible_devices"]},
             "state": {"processor_class": runtime["processor_class"], "text_only": True,
                       "loss_weight_sum": base_hf_state["loss_weight_sum"]}},
            ["evidence/runtime-preflight.json", "evidence/c1-hf-reference.json"],
            supporting_functions=("swift.template.templates.qwen:Qwen3VLTemplate._encode",),
            supporting_runtime_classes=(
                "transformers.models.qwen3_vl.processing_qwen3_vl:Qwen3VLProcessor",
            ),
        ),
        node(
            "conversion_provider",
            "mcore_bridge.model.gpts.qwen3_next_gdn:Qwen3NextLoader.build_model",
            "mcore_bridge.model.mm_gpts.qwen3_5_gdn:Qwen3_5Bridge",
            {"shape": {"converted_rows": len(c1_conversion.get("records", [])),
                       "mtp_tensor_count": c1_mtp.get("expected_tensor_count")},
             "rank": {"world_size": 1, "tp": 1, "dp": 1},
             "state": {"logit_parity_status": c1_conversion["status"],
                       "mtp_roundtrip_status": c1_mtp["status"]}},
            ["evidence/c1-conversion-parity.json", "evidence/c1-mtp-roundtrip.json"],
            supporting_functions=(
                "mcore_bridge.model.gpts.qwen3_next_gdn:Qwen3NextGDNBridgeMixin._set_layer_attn",
                "mcore_bridge.bridge.gpt_bridge:GPTBridge.load_weights",
                "mcore_bridge.bridge.gpt_bridge:GPTBridge.save_weights",
            ),
            supporting_runtime_classes=(
                "mcore_bridge.model.gpts.qwen3_next_gdn:Qwen3NextLoader",
            ),
        ),
        node(
            "gdn",
            "mcore_bridge.model.modules.gated_delta_net:GatedDeltaNet.forward",
            gdn_class,
            {"shape": {"base_batch": base_hf_state["batch_shape"],
                       "parameter_layout_sha256_by_rank": c5_state["parameter_layout_sha256_by_rank"]},
             "rank": c5_state,
             "state": {"use_mcore_gdn": runtime.get("use_mcore_gdn"),
                       "gdn_sources": runtime.get("gdn_sources")}},
            ["evidence/runtime-preflight.json", "evidence/c3-ranks-summary.json", "evidence/c5-ranks-summary.json"],
        ),
        node(
            "batch_labels_loss",
            "swift.megatron.trainers.trainer:MegatronTrainer.forward_step",
            "swift.megatron.trainers.trainer:MegatronTrainer",
            {"shape": {"labels": base_hf_state["batch_shape"],
                       "supervised_tokens": base_hf_state["supervised_tokens"]},
             "rank": {"c2": c2_state, "c3": c3_state, "c5": c5_state},
             "state": {"hf_reference_loss": base_hf_state["mean_loss"],
                       "mcore_single_rank_loss": c1_loss.get("mcore_loss"),
                       "final_hf_loss": final_hf_state["mean_loss"]}},
            ["evidence/c1-hf-reference.json", "evidence/c1-single-rank-loss-parity.json",
             "evidence/c2-ranks-summary.json", "evidence/c3-ranks-summary.json",
             "evidence/c5-tiny-overfit.json"],
            supporting_functions=(
                "swift.megatron.trainers.utils:prepare_batch",
                "swift.megatron.trainers.trainer:MegatronTrainer.loss_func",
            ),
        ),
        node(
            "backward_optimizer",
            "swift.megatron.trainers.base:BaseMegatronTrainer.train_step",
            optimizer_class,
            {"shape": {"c5_parameter_layout_sha256_by_rank": c5_state["parameter_layout_sha256_by_rank"]},
             "rank": {"c2": c2_state, "c3": c3_state, "c5": c5_state},
             "state": {"c5_optimizer_steps": c5_overfit.get("optimizer_steps"),
                       "logged_gradient_norm_count": c5_overfit.get("logged_gradient_norm_count")}},
            ["evidence/c2-ranks-summary.json", "evidence/c3-ranks-summary.json",
             "evidence/c5-ranks-summary.json", "evidence/c5-tiny-overfit.json"],
            supporting_runtime_classes=("swift.megatron.trainers.base:BaseMegatronTrainer",),
        ),
        node(
            "distributed_save_load",
            "swift.megatron.utils.megatron_lm_utils:save_mcore_checkpoint",
            "swift.megatron.trainers.base:BaseMegatronTrainer",
            {"shape": {"c2_state_key_count": c2_audit.get("state_key_count"),
                       "c4_state_key_count": c4_audit.get("state_key_count")},
             "rank": {"c2_save": c2_state, "c2_fresh_load": c2_load_state, "c4_resume": c4_state},
             "state": {"c2_checkpoint_kind": c2_audit.get("checkpoint_kind"),
                       "c4_checkpoint_kind": c4_audit.get("checkpoint_kind"),
                       "c4_expected_iteration": c4_audit.get("expected_iteration"),
                       "c4_logged_steps": c4_resume.get("expected_steps")}},
            ["evidence/c2-checkpoint-audit.json", "evidence/c2-loadcheck-ranks-summary.json",
             "evidence/c4-checkpoint-audit.json", "evidence/c4-ranks-summary.json", "evidence/c4-resume.json"],
            supporting_functions=(
                "swift.megatron.utils.megatron_lm_utils:load_mcore_checkpoint",
                "swift.megatron.trainers.base:BaseMegatronTrainer._load_checkpoint",
                "swift.megatron.trainers.base:BaseMegatronTrainer.save_checkpoint",
            ),
        ),
        node(
            "hf_export",
            "swift.megatron.pipelines.export.export:MegatronExport.convert_mcore2hf",
            "swift.megatron.pipelines.export.export:MegatronExport",
            {"shape": {"c4_reload_batch": c4_hf_state["batch_shape"],
                       "c5_reload_batch": final_hf_state["batch_shape"]},
             "rank": {"export_world_size": 1, "reload_cuda_visible_devices": final_hf_state["cuda_visible_devices"]},
             "state": {"c4_parity": c4_export["status"], "c5_parity": c5_export["status"],
                       "c5_checkpoint_kind": c5_audit.get("checkpoint_kind"),
                       "parameter_scopes": c5_scopes.get("scopes")}},
            ["evidence/c4-export-parity.json", "evidence/c4-hf-reference.json",
             "evidence/c5-checkpoint-audit.json", "evidence/c5-export-parity.json",
             "evidence/c5-hf-reference.json", "evidence/c5-parameter-scopes.json"],
        ),
    ]

    for item in nodes:
        for relative in item["evidence_paths"]:
            require_run_file(run_root, relative)
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "failures": [],
        "ms_swift_commit": swift_commit,
        "packages": runtime.get("packages"),
        "runtime_imports": runtime.get("imports"),
        "node_count": len(nodes),
        "nodes": nodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    output = args.output or run_root / "evidence" / "codepath-runtime-evidence.json"
    try:
        payload = build_manifest(run_root)
    except Exception as exc:  # noqa: BLE001 - the precise resolution failure is evidence.
        payload = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "fail",
            "failures": [f"{type(exc).__name__}: {exc}"],
            "nodes": [],
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
