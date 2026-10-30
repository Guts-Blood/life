#!/usr/bin/env python3
"""Run one controlled Day 12 SFT stage on the frozen exact-token schedule."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import prepare_day12_experiment as preparation

HERE = Path(__file__).resolve().parent
BOOTCAMP_ROOT = HERE.parent
DAY08_CONTRACT = BOOTCAMP_ROOT / "day-08-sft-data-contract" / "inspect_sft_sample.py"
IGNORE_INDEX = -100


class Day12TrainingError(ValueError):
    """A training, checkpoint, or runtime invariant failed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    if not path.is_file():
        raise Day12TrainingError(f"required file is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise Day12TrainingError(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise Day12TrainingError(f"JSON root must be an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise Day12TrainingError(f"JSONL row must be an object: {path}")
            result.append(value)
    return result


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_day08_contract() -> Any:
    spec = importlib.util.spec_from_file_location("day08_sft_contract", DAY08_CONTRACT)
    if spec is None or spec.loader is None:
        raise Day12TrainingError("cannot import the Day 08 SFT data contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_base_model(model_path: Path, config: dict[str, Any]) -> None:
    if not model_path.is_dir():
        raise Day12TrainingError(f"base model directory is missing: {model_path}")
    for relative_path, expected_hash in config["model"]["files"].items():
        actual_hash = file_sha256(model_path / relative_path)
        if actual_hash != expected_hash:
            raise Day12TrainingError(
                f"base model hash mismatch for {relative_path}: {actual_hash}"
            )


def ordered_schedule_ids(
    config: dict[str, Any], schedule: dict[str, Any]
) -> tuple[list[str], dict[str, int]]:
    preparation.verify_schedule(schedule)
    header = schedule["header"]
    if header.get("run_id") != config["run"]["id"]:
        raise Day12TrainingError("schedule run ID differs from config")
    if header.get("mixture_manifest_hash") != config["data"]["manifest_hash"]:
        raise Day12TrainingError("schedule mixture hash differs from config")
    if header.get("config_hash") != preparation.object_sha256(config):
        raise Day12TrainingError("schedule resolved-config hash differs from current config")
    ordered: list[str] = []
    cursors: dict[str, int] = {}
    for checkpoint_name in config["training"]["checkpoint_order"]:
        segment = schedule["segments"][checkpoint_name]
        ordered.extend(segment["occurrence_ids"])
        cursors[checkpoint_name] = len(ordered)
    if len(ordered) != len(set(ordered)):
        raise Day12TrainingError("schedule occurrence IDs are duplicated")
    return ordered, cursors


def encode_training_records(
    config: dict[str, Any], schedule: dict[str, Any], tokenizer: Any
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    try:
        manifest = preparation.verify_mixture(config)
    except preparation.Day12PreparationError as error:
        raise Day12TrainingError(str(error)) from error
    ordered_ids, cursors = ordered_schedule_ids(config, schedule)
    records = manifest["records"]
    by_id = {record["occurrence_id"]: record for record in records}
    if set(ordered_ids) != set(by_id):
        raise Day12TrainingError("schedule IDs do not match the mixture manifest")

    template_hash = hashlib.sha256(
        (tokenizer.chat_template or "").encode("utf-8")
    ).hexdigest()
    if template_hash != config["model"]["chat_template_sha256"]:
        raise Day12TrainingError("tokenizer chat template hash differs from config")
    if type(tokenizer).__name__ != config["model"]["tokenizer_class"]:
        raise Day12TrainingError("tokenizer class differs from config")

    contract = load_day08_contract()
    encoded: list[dict[str, Any]] = []
    for occurrence_id in ordered_ids:
        raw = by_id[occurrence_id]
        sample = {
            "id": occurrence_id,
            "messages": raw["messages"],
            "metadata": {"skill": raw["skill"], "canonical_sample_id": raw["sample_id"]},
        }
        audit = contract.encode_sample(tokenizer, sample, int(config["data"]["max_length"]))
        input_ids = [row["token_id"] for row in audit["rows"]]
        labels = [row["label"] for row in audit["rows"]]
        supervised_tokens = sum(label != IGNORE_INDEX for label in labels[1:])
        if len(input_ids) != int(raw["input_token_count"]):
            raise Day12TrainingError(f"encoded length drift: {occurrence_id}")
        if supervised_tokens != int(raw["supervised_tokens_per_occurrence"]):
            raise Day12TrainingError(f"supervised-token drift: {occurrence_id}")
        encoded.append(
            {
                "id": occurrence_id,
                "skill": raw["skill"],
                "input_ids": input_ids,
                "labels": labels,
                "supervised_tokens": supervised_tokens,
            }
        )
    if sum(record["supervised_tokens"] for record in encoded) != int(
        config["training"]["checkpoint_budgets"]["100_percent"]
    ):
        raise Day12TrainingError("encoded data does not preserve the full token budget")
    return encoded, manifest, cursors


def collate(records: list[dict[str, Any]], tokenizer: Any, device: Any) -> dict[str, Any]:
    import torch

    max_length = max(len(record["input_ids"]) for record in records)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise Day12TrainingError("tokenizer has no padding or EOS token")
    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    attention_rows: list[list[int]] = []
    for record in records:
        padding = max_length - len(record["input_ids"])
        input_rows.append(record["input_ids"] + [pad_token_id] * padding)
        label_rows.append(record["labels"] + [IGNORE_INDEX] * padding)
        attention_rows.append([1] * len(record["input_ids"]) + [0] * padding)
    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long, device=device),
        "labels": torch.tensor(label_rows, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_rows, dtype=torch.long, device=device),
    }


def masked_causal_loss_sum(logits: Any, labels: Any) -> tuple[Any, int]:
    from torch.nn import functional

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels.ne(IGNORE_INDEX)
    token_count = int(valid.sum().item())
    if token_count <= 0:
        raise Day12TrainingError("microbatch has zero shifted supervised tokens")
    loss_sum = functional.cross_entropy(
        shift_logits.float().view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    return loss_sum, token_count


def set_reproducibility(seed: int, deterministic: bool) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if deterministic:
        torch.use_deterministic_algorithms(True)


class TokenWarmupScheduler:
    def __init__(self, optimizer: Any, base_lr: float, warmup_tokens: int) -> None:
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_tokens = warmup_tokens
        self.completed_tokens = 0
        self.last_lr = 0.0

    def prepare_window(self, completed_tokens: int, window_tokens: int) -> float:
        if completed_tokens != self.completed_tokens or window_tokens <= 0:
            raise Day12TrainingError("token scheduler state is inconsistent")
        target_tokens = completed_tokens + window_tokens
        multiplier = 1.0
        if self.warmup_tokens > 0:
            multiplier = min(target_tokens / self.warmup_tokens, 1.0)
        self.last_lr = self.base_lr * multiplier
        for group in self.optimizer.param_groups:
            group["lr"] = self.last_lr
        return self.last_lr

    def complete_window(self, completed_tokens: int) -> None:
        if completed_tokens < self.completed_tokens:
            raise Day12TrainingError("token scheduler cannot move backwards")
        self.completed_tokens = completed_tokens

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": "linear_warmup_by_supervised_tokens_then_constant",
            "base_lr": self.base_lr,
            "warmup_tokens": self.warmup_tokens,
            "completed_tokens": self.completed_tokens,
            "last_lr": self.last_lr,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "name": "linear_warmup_by_supervised_tokens_then_constant",
            "base_lr": self.base_lr,
            "warmup_tokens": self.warmup_tokens,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise Day12TrainingError(f"scheduler restore mismatch: {key}")
        self.completed_tokens = int(state["completed_tokens"])
        self.last_lr = float(state["last_lr"])
        for group in self.optimizer.param_groups:
            group["lr"] = self.last_lr


def trainable_parameters(model: Any) -> list[Any]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise Day12TrainingError("model has no trainable parameters")
    return parameters


def parameter_counts(model: Any) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def make_optimizer(model: Any, config: dict[str, Any]) -> Any:
    import torch

    optimizer_config = config["optimizer"]
    return torch.optim.AdamW(
        trainable_parameters(model),
        lr=float(optimizer_config["learning_rate"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        eps=float(optimizer_config["epsilon"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        fused=False,
        foreach=False,
    )


def checkpoint_file_hashes(path: Path) -> dict[str, str]:
    return {
        file.relative_to(path).as_posix(): file_sha256(file)
        for file in sorted(path.rglob("*"))
        if file.is_file() and file.name not in {"checkpoint_manifest.json", "COMPLETE"}
    }


def save_checkpoint(
    path: Path,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: TokenWarmupScheduler,
    state: dict[str, Any],
) -> None:
    import torch

    if path.exists():
        raise Day12TrainingError(f"refusing to overwrite checkpoint: {path}")
    path.mkdir(parents=True)
    model_path = path / "model"
    model.save_pretrained(model_path, safe_serialization=True)
    tokenizer.save_pretrained(model_path)
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    write_json(path / "scheduler_state.json", scheduler.state_dict())
    rng_state = {
        "python_random_state": random.getstate(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all(),
    }
    torch.save(rng_state, path / "rng_state.pt")
    write_json(path / "training_state.json", state)
    manifest = {
        "domain": "day12.full_training_checkpoint",
        "schema_version": 1,
        "created_at": utc_now(),
        "training_state_hash": preparation.object_sha256(state),
        "files": checkpoint_file_hashes(path),
    }
    write_json(path / "checkpoint_manifest.json", manifest)
    (path / "COMPLETE").write_text("complete\n", encoding="utf-8")


def verify_checkpoint(
    path: Path, config_hash: str, schedule_hash: str
) -> dict[str, Any]:
    if not (path / "COMPLETE").is_file():
        raise Day12TrainingError(f"checkpoint is incomplete: {path}")
    manifest = load_json(path / "checkpoint_manifest.json")
    if manifest.get("domain") != "day12.full_training_checkpoint":
        raise Day12TrainingError("unexpected checkpoint domain")
    for relative_path, expected_hash in manifest.get("files", {}).items():
        if file_sha256(path / relative_path) != expected_hash:
            raise Day12TrainingError(f"checkpoint file hash mismatch: {relative_path}")
    state = load_json(path / "training_state.json")
    if preparation.object_sha256(state) != manifest.get("training_state_hash"):
        raise Day12TrainingError("checkpoint training-state hash mismatch")
    if state.get("config_hash") != config_hash or state.get("schedule_hash") != schedule_hash:
        raise Day12TrainingError("checkpoint config/schedule identity mismatch")
    return state


def restore_runtime_state(
    checkpoint_path: Path,
    optimizer: Any,
    scheduler: TokenWarmupScheduler,
    device: Any,
) -> None:
    import torch

    optimizer.load_state_dict(
        torch.load(checkpoint_path / "optimizer.pt", map_location="cpu", weights_only=False)
    )
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(device)
    scheduler.load_state_dict(load_json(checkpoint_path / "scheduler_state.json"))
    rng = torch.load(
        checkpoint_path / "rng_state.pt", map_location="cpu", weights_only=False
    )
    random.setstate(rng["python_random_state"])
    torch.set_rng_state(rng["torch_cpu_rng_state"])
    torch.cuda.set_rng_state_all(rng["torch_cuda_rng_state_all"])


def export_bf16_model(
    path: Path,
    model: Any,
    tokenizer: Any,
    parent_checkpoint: Path,
    training_state: dict[str, Any],
) -> None:
    import torch
    from safetensors.torch import save_file

    if path.exists():
        raise Day12TrainingError(f"refusing to overwrite inference export: {path}")
    path.mkdir(parents=True)
    export_model = model
    if training_state["parameterization"]["type"] == "lora":
        if not callable(getattr(model, "merge_and_unload", None)):
            raise Day12TrainingError("LoRA model cannot be merged for inference export")
        export_model = model.merge_and_unload(safe_merge=True)
    state_dict: dict[str, Any] = {}
    for name, tensor in export_model.state_dict().items():
        value = tensor.detach().cpu()
        if value.is_floating_point():
            value = value.to(torch.bfloat16)
        state_dict[name] = value.contiguous()
    save_file(state_dict, path / "model.safetensors", metadata={"format": "pt"})
    del state_dict

    export_config = copy.deepcopy(export_model.config)
    export_config.torch_dtype = torch.bfloat16
    export_config.save_pretrained(path)
    if getattr(export_model, "generation_config", None) is not None:
        export_model.generation_config.save_pretrained(path)
    tokenizer.save_pretrained(path)
    files = {
        file.relative_to(path).as_posix(): file_sha256(file)
        for file in sorted(path.rglob("*"))
        if file.is_file() and file.name not in {"export_manifest.json", "COMPLETE"}
    }
    write_json(
        path / "export_manifest.json",
        {
            "domain": "day12.bf16_inference_export",
            "schema_version": 1,
            "created_at": utc_now(),
            "parent_checkpoint": str(parent_checkpoint.resolve()),
            "parent_checkpoint_manifest_sha256": file_sha256(
                parent_checkpoint / "checkpoint_manifest.json"
            ),
            "checkpoint_name": training_state["checkpoint_name"],
            "config_hash": training_state["config_hash"],
            "schedule_hash": training_state["schedule_hash"],
            "cumulative_supervised_tokens": training_state[
                "cumulative_supervised_tokens"
            ],
            "dtype": "bfloat16",
            "parameterization": training_state["parameterization"],
            "files": files,
        },
    )
    (path / "COMPLETE").write_text("complete\n", encoding="utf-8")


def configure_model(model: Any, config: dict[str, Any]) -> None:
    model.config.use_cache = False
    if config["model"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )


def adapter_settings(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("adapter", {"type": "none"})
    if not isinstance(value, dict) or value.get("type") not in {"none", "lora"}:
        raise Day12TrainingError("adapter.type must be none or lora")
    return value


def configure_parameterization(
    model: Any,
    config: dict[str, Any],
    *,
    resume_adapter_path: Path | None = None,
) -> Any:
    adapter = adapter_settings(config)
    if adapter["type"] == "none":
        if resume_adapter_path is not None:
            raise Day12TrainingError("full-parameter run cannot resume an adapter")
        return model

    from peft import LoraConfig, PeftModel, get_peft_model

    expected_version = str(adapter["peft_version"])
    actual_version = importlib.metadata.version("peft")
    if actual_version != expected_version:
        raise Day12TrainingError(
            f"PEFT version mismatch: expected {expected_version}, got {actual_version}"
        )
    if resume_adapter_path is not None:
        result = PeftModel.from_pretrained(
            model, resume_adapter_path, is_trainable=True, local_files_only=True
        )
    else:
        lora_config = LoraConfig(
            task_type=str(adapter["task_type"]),
            r=int(adapter["rank"]),
            lora_alpha=int(adapter["alpha"]),
            lora_dropout=float(adapter["dropout"]),
            target_modules=adapter["target_modules"],
            bias=str(adapter["bias"]),
            inference_mode=False,
        )
        result = get_peft_model(model, lora_config)
    if config["model"]["gradient_checkpointing"]:
        result.enable_input_require_grads()
    return result


def verify_runtime(config: dict[str, Any]) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise Day12TrainingError("Day 12 requires a CUDA GPU with BF16 support")
    properties = torch.cuda.get_device_properties(0)
    memory_gib = properties.total_memory / 1024**3
    if memory_gib < float(config["hardware"]["minimum_gpu_memory_gib"]):
        raise Day12TrainingError(f"GPU memory is too small: {memory_gib:.2f} GiB")
    adapter = adapter_settings(config)
    if adapter["type"] == "lora":
        try:
            actual_version = importlib.metadata.version("peft")
        except importlib.metadata.PackageNotFoundError as error:
            raise Day12TrainingError("LoRA run requires the peft package") from error
        if actual_version != str(adapter["peft_version"]):
            raise Day12TrainingError(
                f"PEFT version mismatch: expected {adapter['peft_version']}, got {actual_version}"
            )
    return torch, AutoModelForCausalLM, AutoTokenizer


def run_preflight(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoTokenizer

    config = preparation.load_resolved_config(args.config)
    schedule = load_json(args.schedule)
    model_path = args.model_path.resolve()
    verify_base_model(model_path, config)
    verify_runtime(config)
    disk = shutil.disk_usage(args.run_root.resolve())
    free_gib = disk.free / 1024**3
    minimum = float(config["checkpoint"]["minimum_free_disk_gib"])
    if free_gib < minimum:
        raise Day12TrainingError(
            f"free disk {free_gib:.2f} GiB is below required {minimum:.2f} GiB"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, use_fast=True
    )
    encoded, manifest, cursors = encode_training_records(config, schedule, tokenizer)
    result = {
        "status": "day12_preflight_pass",
        "created_at": utc_now(),
        "run_id": config["run"]["id"],
        "config_hash": preparation.object_sha256(config),
        "schedule_hash": schedule["header"]["schedule_hash"],
        "mixture_manifest_hash": manifest["header"]["manifest_hash"],
        "record_count": len(encoded),
        "supervised_tokens": sum(record["supervised_tokens"] for record in encoded),
        "maximum_input_tokens": max(len(record["input_ids"]) for record in encoded),
        "checkpoint_cursors": cursors,
        "free_disk_gib": free_gib,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "parameterization": adapter_settings(config),
        "peft_version": (
            importlib.metadata.version("peft")
            if adapter_settings(config)["type"] == "lora"
            else None
        ),
    }
    if args.output is not None:
        write_json(args.output.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_train(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = preparation.load_resolved_config(args.config)
    schedule = load_json(args.schedule)
    config_hash = preparation.object_sha256(config)
    schedule_hash = schedule["header"]["schedule_hash"]
    target_name = args.target
    if target_name not in config["training"]["checkpoint_order"]:
        raise Day12TrainingError(f"unknown checkpoint target: {target_name}")
    run_dir = args.run_dir.resolve()
    trace_path = run_dir / "step-trace.jsonl"
    if args.resume_from is None and run_dir.exists() and any(run_dir.iterdir()):
        raise Day12TrainingError(f"refusing to reuse non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    verify_runtime(config)
    set_reproducibility(
        int(config["seed"]), bool(config["training"]["deterministic_algorithms"])
    )
    device = torch.device("cuda:0")
    base_model_path = args.model_path.resolve()
    verify_base_model(base_model_path, config)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, local_files_only=True, use_fast=True
    )
    records, manifest, checkpoint_cursors = encode_training_records(
        config, schedule, tokenizer
    )

    state: dict[str, Any] | None = None
    model_load_path = base_model_path
    if args.resume_from is not None:
        state = verify_checkpoint(args.resume_from.resolve(), config_hash, schedule_hash)
        if adapter_settings(config)["type"] == "none":
            model_load_path = args.resume_from.resolve() / "model"
    model = AutoModelForCausalLM.from_pretrained(
        model_load_path,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation=config["model"]["attention_implementation"],
    ).to(device)
    configure_model(model, config)
    model = configure_parameterization(
        model,
        config,
        resume_adapter_path=(
            args.resume_from.resolve() / "model"
            if args.resume_from is not None
            and adapter_settings(config)["type"] == "lora"
            else None
        ),
    )
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    if parameter_dtypes != ["torch.float32"]:
        raise Day12TrainingError(f"unexpected parameter dtypes: {parameter_dtypes}")
    optimizer = make_optimizer(model, config)
    scheduler = TokenWarmupScheduler(
        optimizer,
        float(config["optimizer"]["learning_rate"]),
        int(config["scheduler"]["warmup_supervised_tokens"]),
    )

    optimizer_step = 0
    schedule_cursor = 0
    cumulative_tokens = 0
    if state is not None:
        optimizer_step = int(state["optimizer_step"])
        schedule_cursor = int(state["schedule_cursor"])
        cumulative_tokens = int(state["cumulative_supervised_tokens"])
        restore_runtime_state(args.resume_from.resolve(), optimizer, scheduler, device)
        if scheduler.completed_tokens != cumulative_tokens:
            raise Day12TrainingError("restored scheduler token position differs from checkpoint")
        trace = load_jsonl(trace_path)
        if not trace or trace[-1].get("schedule_cursor") != schedule_cursor:
            raise Day12TrainingError("trace/checkpoint schedule cursor continuity failed")
    else:
        first_name = config["training"]["checkpoint_order"][0]
        if target_name != first_name:
            raise Day12TrainingError("a fresh run must target the first checkpoint")

    target_cursor = int(checkpoint_cursors[target_name])
    target_tokens = int(config["training"]["checkpoint_budgets"][target_name])
    if schedule_cursor >= target_cursor:
        raise Day12TrainingError("target checkpoint is not ahead of current state")
    micro_batch_size = int(config["training"]["micro_batch_size"])
    accumulation_steps = int(config["training"]["gradient_accumulation_steps"])
    window_size = micro_batch_size * accumulation_steps
    max_grad_norm = float(config["optimizer"]["max_grad_norm"])
    optimizer.zero_grad(set_to_none=True)
    completed_this_process = 0

    while schedule_cursor < target_cursor:
        window_end = min(schedule_cursor + window_size, target_cursor)
        window_records = records[schedule_cursor:window_end]
        microbatches = [
            window_records[index : index + micro_batch_size]
            for index in range(0, len(window_records), micro_batch_size)
        ]
        window_tokens = sum(record["supervised_tokens"] for record in window_records)
        learning_rate = scheduler.prepare_window(cumulative_tokens, window_tokens)
        total_loss_sum = 0.0
        microbatch_trace: list[dict[str, Any]] = []
        model.train()
        for microbatch in microbatches:
            batch = collate(microbatch, tokenizer, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                )
            loss_sum, token_count = masked_causal_loss_sum(outputs.logits, batch["labels"])
            (loss_sum / window_tokens).backward()
            numeric_loss_sum = float(loss_sum.detach().float().cpu().item())
            total_loss_sum += numeric_loss_sum
            microbatch_trace.append(
                {
                    "occurrence_ids": [record["id"] for record in microbatch],
                    "supervised_tokens": token_count,
                    "mean_loss": numeric_loss_sum / token_count,
                }
            )
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            trainable_parameters(model), max_grad_norm, foreach=False
        )
        grad_norm = float(grad_norm_tensor.detach().float().cpu().item())
        if not math.isfinite(grad_norm) or grad_norm == 0.0:
            raise Day12TrainingError(
                f"invalid gradient norm at optimizer step {optimizer_step + 1}: {grad_norm}"
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if not all(parameter.grad is None for parameter in model.parameters()):
            raise Day12TrainingError("zero_grad(set_to_none=True) did not clear gradients")
        optimizer_step += 1
        schedule_cursor = window_end
        cumulative_tokens += window_tokens
        scheduler.complete_window(cumulative_tokens)
        completed_this_process += 1
        append_jsonl(
            trace_path,
            {
                "event": "optimizer_step",
                "run_id": config["run"]["id"],
                "target_checkpoint": target_name,
                "optimizer_step": optimizer_step,
                "schedule_cursor": schedule_cursor,
                "window_record_count": len(window_records),
                "window_supervised_tokens": window_tokens,
                "cumulative_supervised_tokens": cumulative_tokens,
                "loss_sum": total_loss_sum,
                "mean_loss": total_loss_sum / window_tokens,
                "grad_norm_before_clip": grad_norm,
                "max_grad_norm": max_grad_norm,
                "learning_rate": learning_rate,
                "microbatches": microbatch_trace,
                "operation_order": [
                    "token_lr",
                    "forward",
                    "fp32_loss_sum/window_tokens",
                    "backward",
                    "clip",
                    "optimizer",
                    "zero_grad",
                ],
            },
        )
        if args.smoke_steps is not None and completed_this_process >= args.smoke_steps:
            break

    if args.smoke_steps is not None:
        summary = {
            "status": "day12_smoke_pass",
            "run_id": config["run"]["id"],
            "optimizer_steps": completed_this_process,
            "schedule_cursor": schedule_cursor,
            "cumulative_supervised_tokens": cumulative_tokens,
            "maximum_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "parameterization": adapter_settings(config),
            "parameter_counts": parameter_counts(model),
        }
        write_json(run_dir / f"smoke-{config['run']['id']}-summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if schedule_cursor != target_cursor or cumulative_tokens != target_tokens:
        raise Day12TrainingError(
            f"target boundary mismatch: cursor={schedule_cursor}/{target_cursor}, "
            f"tokens={cumulative_tokens}/{target_tokens}"
        )
    training_state = {
        "domain": "day12.training_state",
        "schema_version": 1,
        "created_at": utc_now(),
        "run_id": config["run"]["id"],
        "checkpoint_name": target_name,
        "config_hash": config_hash,
        "schedule_hash": schedule_hash,
        "mixture_manifest_hash": manifest["header"]["manifest_hash"],
        "base_model_id": config["model"]["id"],
        "base_model_revision": config["model"]["revision"],
        "optimizer_step": optimizer_step,
        "schedule_cursor": schedule_cursor,
        "cumulative_supervised_tokens": cumulative_tokens,
        "scheduler_state": scheduler.state_dict(),
        "parameterization": adapter_settings(config),
        "parameter_counts": parameter_counts(model),
    }
    checkpoint_path = run_dir / "checkpoints" / f"checkpoint-{target_name}"
    save_checkpoint(
        checkpoint_path, model, tokenizer, optimizer, scheduler, training_state
    )
    verify_checkpoint(checkpoint_path, config_hash, schedule_hash)
    export_path = args.export_root.resolve() / f"{config['run']['id']}-{target_name}"
    export_bf16_model(
        export_path, model, tokenizer, checkpoint_path, training_state
    )
    summary = {
        "status": "day12_stage_complete",
        "run_id": config["run"]["id"],
        "checkpoint_name": target_name,
        "optimizer_step": optimizer_step,
        "schedule_cursor": schedule_cursor,
        "cumulative_supervised_tokens": cumulative_tokens,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_manifest_sha256": file_sha256(
            checkpoint_path / "checkpoint_manifest.json"
        ),
        "inference_export_path": str(export_path),
        "inference_export_manifest_sha256": file_sha256(
            export_path / "export_manifest.json"
        ),
        "maximum_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "parameterization": training_state["parameterization"],
        "parameter_counts": training_state["parameter_counts"],
    }
    write_json(run_dir / f"stage-{target_name}-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--config", type=Path, required=True)
    preflight_parser.add_argument("--schedule", type=Path, required=True)
    preflight_parser.add_argument("--model-path", type=Path, required=True)
    preflight_parser.add_argument("--run-root", type=Path, required=True)
    preflight_parser.add_argument("--output", type=Path)
    preflight_parser.set_defaults(function=run_preflight)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--schedule", type=Path, required=True)
    train_parser.add_argument("--model-path", type=Path, required=True)
    train_parser.add_argument("--run-dir", type=Path, required=True)
    train_parser.add_argument("--export-root", type=Path, required=True)
    train_parser.add_argument(
        "--target", choices=("25_percent", "60_percent", "100_percent"), required=True
    )
    train_parser.add_argument("--resume-from", type=Path)
    train_parser.add_argument("--smoke-steps", type=int)
    train_parser.set_defaults(function=run_train)
    return parser


def main() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "smoke_steps", None) is not None and args.smoke_steps <= 0:
        parser.error("--smoke-steps must be positive")
    try:
        args.function(args)
    except (Day12TrainingError, preparation.Day12PreparationError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
