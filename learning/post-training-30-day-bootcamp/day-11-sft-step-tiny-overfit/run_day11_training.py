#!/usr/bin/env python3
"""Run and audit the Day 11 Qwen3-0.6B tiny-overfit experiment.

The loop is intentionally explicit. It computes a supervised-token-weighted
loss across each accumulation window and records the state transition around
every optimizer step. It is not a general-purpose trainer.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
BOOTCAMP_ROOT = HERE.parent
DAY08_CONTRACT_PATH = BOOTCAMP_ROOT / "day-08-sft-data-contract/inspect_sft_sample.py"
DEFAULT_CONFIG_PATH = BOOTCAMP_ROOT / "artifacts/configs/day11-qwen3-0.6b-tiny-overfit.yaml"
IGNORE_INDEX = -100


class Day11TrainingError(ValueError):
    """A frozen contract or runtime invariant failed."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def semantic_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    if not path.is_file():
        raise Day11TrainingError(f"required file is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise Day11TrainingError(f"required JSON is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise Day11TrainingError(f"invalid JSON in {path}: {error}") from error


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    except FileNotFoundError as error:
        raise Day11TrainingError(f"required JSONL is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise Day11TrainingError(f"invalid JSONL in {path}: {error}") from error


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise Day11TrainingError(f"config is missing: {path}") from error
    except yaml.YAMLError as error:
        raise Day11TrainingError(f"invalid YAML in {path}: {error}") from error
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise Day11TrainingError("unsupported Day 11 config")
    return config


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else BOOTCAMP_ROOT / path


def load_day08_contract() -> Any:
    spec = importlib.util.spec_from_file_location("day08_sft_contract_for_day11", DAY08_CONTRACT_PATH)
    if spec is None or spec.loader is None:
        raise Day11TrainingError(f"cannot import Day 08 contract: {DAY08_CONTRACT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_manifest_hash(manifest: dict[str, Any]) -> None:
    expected = manifest.get("header", {}).get("manifest_hash")
    if not isinstance(expected, str):
        raise Day11TrainingError("tiny manifest has no manifest_hash")
    candidate = copy.deepcopy(manifest)
    candidate["header"].pop("manifest_hash", None)
    actual = semantic_hash(candidate)
    if actual != expected:
        raise Day11TrainingError(f"tiny manifest hash mismatch: expected {expected}, got {actual}")


def verify_model_files(model_path: Path, manifest: dict[str, Any]) -> None:
    expected_files = manifest["header"].get("model_files")
    if not isinstance(expected_files, dict) or not expected_files:
        raise Day11TrainingError("tiny manifest has no frozen model file hashes")
    for relative_path, expected_hash in expected_files.items():
        actual = file_sha256(model_path / relative_path)
        if actual != expected_hash:
            raise Day11TrainingError(
                f"base model file hash mismatch for {relative_path}: expected {expected_hash}, got {actual}"
            )


def verify_and_encode_inputs(
    config: dict[str, Any], tokenizer: Any, *, verify_base_model_path: Path | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data_path = resolve_project_path(config["data"]["path"])
    manifest_path = resolve_project_path(config["data"]["manifest_path"])
    manifest = load_json(manifest_path)
    header = manifest.get("header", {})
    if header.get("domain") != "day11.tiny_overfit_manifest":
        raise Day11TrainingError("unexpected tiny manifest domain")
    if header.get("status") != "frozen_for_day11_execution":
        raise Day11TrainingError("tiny manifest is not frozen for execution")
    if header.get("model_id") != config["model"]["id"]:
        raise Day11TrainingError("model ID differs between config and manifest")
    if header.get("model_revision") != config["model"]["revision"]:
        raise Day11TrainingError("model revision differs between config and manifest")
    if header.get("max_length") != config["data"]["max_length"]:
        raise Day11TrainingError("max_length differs between config and manifest")
    verify_manifest_hash(manifest)
    if file_sha256(data_path) != header.get("dataset_file_sha256"):
        raise Day11TrainingError("tiny dataset byte hash differs from frozen manifest")
    if verify_base_model_path is not None:
        verify_model_files(verify_base_model_path, manifest)

    template_hash = hashlib.sha256((tokenizer.chat_template or "").encode("utf-8")).hexdigest()
    if template_hash != header.get("chat_template_sha256"):
        raise Day11TrainingError("tokenizer chat template differs from frozen manifest")

    raw_records = load_jsonl(data_path)
    manifest_records = manifest.get("records")
    if len(raw_records) != header.get("record_count") or not isinstance(manifest_records, list):
        raise Day11TrainingError("tiny dataset record count differs from manifest")
    by_id = {record.get("id"): record for record in manifest_records}
    if len(by_id) != len(manifest_records):
        raise Day11TrainingError("tiny manifest IDs are not unique")

    contract = load_day08_contract()
    encoded: list[dict[str, Any]] = []
    for raw in raw_records:
        sample_id = raw.get("id")
        if sample_id not in by_id:
            raise Day11TrainingError(f"dataset ID is absent from manifest: {sample_id}")
        expected_record_hash = semantic_hash(
            {"id": raw["id"], "messages": raw["messages"], "metadata": raw["metadata"]}
        )
        if expected_record_hash != by_id[sample_id].get("record_hash"):
            raise Day11TrainingError(f"record hash mismatch: {sample_id}")
        audit = contract.encode_sample(tokenizer, raw, config["data"]["max_length"])
        input_ids = [row["token_id"] for row in audit["rows"]]
        labels = [row["label"] for row in audit["rows"]]
        supervised_tokens = sum(label != IGNORE_INDEX for label in labels[1:])
        expected = by_id[sample_id]
        if audit["encoded_length"] != expected.get("encoded_tokens"):
            raise Day11TrainingError(f"encoded token count drift: {sample_id}")
        if supervised_tokens != expected.get("supervised_tokens"):
            raise Day11TrainingError(f"supervised token count drift: {sample_id}")
        encoded.append(
            {
                "id": sample_id,
                "messages": raw["messages"],
                "input_ids": input_ids,
                "labels": labels,
                "supervised_tokens": supervised_tokens,
            }
        )
    return encoded, manifest


def deterministic_indices(record_count: int, cursor: int, count: int, seed: int) -> list[int]:
    if record_count <= 0 or cursor < 0 or count <= 0:
        raise Day11TrainingError("invalid sampler arguments")
    result: list[int] = []
    while len(result) < count:
        epoch = cursor // record_count
        offset = cursor % record_count
        order = list(range(record_count))
        random.Random(seed + epoch).shuffle(order)
        take = min(count - len(result), record_count - offset)
        result.extend(order[offset : offset + take])
        cursor += take
    return result


def make_window(
    records: list[dict[str, Any]], cursor: int, micro_batch_size: int, accumulation_steps: int, seed: int
) -> tuple[list[list[dict[str, Any]]], int]:
    sample_count = micro_batch_size * accumulation_steps
    indices = deterministic_indices(len(records), cursor, sample_count, seed)
    window_records = [records[index] for index in indices]
    microbatches = [
        window_records[index : index + micro_batch_size]
        for index in range(0, len(window_records), micro_batch_size)
    ]
    return microbatches, cursor + sample_count


def collate(records: list[dict[str, Any]], tokenizer: Any, device: Any) -> dict[str, Any]:
    import torch

    max_length = max(len(record["input_ids"]) for record in records)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise Day11TrainingError("tokenizer has neither pad_token_id nor eos_token_id")
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


def masked_causal_loss_sum(logits: Any, labels: Any) -> tuple[Any, int, int]:
    import torch.nn.functional as functional

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels.ne(IGNORE_INDEX)
    token_count = int(valid.sum().item())
    if token_count == 0:
        raise Day11TrainingError("microbatch has zero shifted supervised tokens")
    loss_sum = functional.cross_entropy(
        shift_logits.float().view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    correct = int((shift_logits.argmax(dim=-1)[valid] == shift_labels[valid]).sum().item())
    return loss_sum, token_count, correct


def _update_tensor_hash(digest: Any, name: str, tensor: Any) -> None:
    import torch

    digest.update(name.encode("utf-8") + b"\0")
    digest.update(str(tensor.dtype).encode("ascii") + b"\0")
    digest.update(canonical_json(list(tensor.shape)).encode("ascii") + b"\0")
    raw = tensor.detach().contiguous().view(-1).view(torch.uint8).cpu().numpy()
    digest.update(raw.tobytes())


def parameter_checksum(model: Any) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        _update_tensor_hash(digest, name, parameter)
    return "sha256:" + digest.hexdigest()


def _update_value_hash(digest: Any, value: Any, path: str) -> None:
    import torch

    if torch.is_tensor(value):
        _update_tensor_hash(digest, path, value)
    elif isinstance(value, dict):
        for key in sorted(value, key=lambda item: str(item)):
            _update_value_hash(digest, value[key], f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _update_value_hash(digest, item, f"{path}[{index}]")
    else:
        digest.update(path.encode("utf-8") + b"=")
        digest.update(repr(value).encode("utf-8") + b"\0")


def optimizer_checksum(optimizer: Any) -> str:
    digest = hashlib.sha256()
    _update_value_hash(digest, optimizer.state_dict(), "optimizer")
    return "sha256:" + digest.hexdigest()


def state_signature(
    model: Any, optimizer: Any, scheduler: Any, *, optimizer_step: int, sample_cursor: int,
    global_supervised_tokens: int
) -> dict[str, Any]:
    return {
        "optimizer_step": optimizer_step,
        "sample_cursor": sample_cursor,
        "global_supervised_tokens": global_supervised_tokens,
        "parameter_checksum": parameter_checksum(model),
        "optimizer_checksum": optimizer_checksum(optimizer),
        "scheduler_state_hash": semantic_hash(scheduler.state_dict()),
        "scheduler_last_epoch": scheduler.state_dict().get("last_epoch"),
        "lr": [group["lr"] for group in optimizer.param_groups],
    }


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


def make_optimizer_and_scheduler(model: Any, config: dict[str, Any]) -> tuple[Any, Any]:
    import torch

    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        eps=float(optimizer_config["epsilon"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        fused=False,
        foreach=False,
    )
    warmup_steps = int(config["scheduler"]["warmup_steps"])

    def lr_multiplier(scheduler_epoch: int) -> float:
        return 1.0 if warmup_steps <= 0 else min((scheduler_epoch + 1) / warmup_steps, 1.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
    return optimizer, scheduler


def save_checkpoint(
    path: Path, model: Any, tokenizer: Any, optimizer: Any, scheduler: Any, state: dict[str, Any]
) -> None:
    import torch

    if path.exists():
        raise Day11TrainingError(f"refusing to overwrite checkpoint: {path}")
    path.mkdir(parents=True)
    rng_state = {
        "python_random_state": random.getstate(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    model_path = path / "model"
    model.save_pretrained(model_path, safe_serialization=True)
    tokenizer.save_pretrained(model_path)
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    torch.save(scheduler.state_dict(), path / "scheduler.pt")
    torch.save(rng_state, path / "rng_state.pt")
    write_json(path / "training_state.json", state)
    files = {
        str(file.relative_to(path)): file_sha256(file)
        for file in sorted(path.rglob("*"))
        if file.is_file() and file.name not in {"checkpoint_manifest.json", "COMPLETE"}
    }
    write_json(
        path / "checkpoint_manifest.json",
        {
            "domain": "day11.training_checkpoint",
            "schema_version": 1,
            "created_at": utc_now(),
            "files": files,
            "training_state_hash": semantic_hash(state),
        },
    )
    (path / "COMPLETE").write_text("complete\n", encoding="utf-8")


def verify_checkpoint(path: Path, config_hash: str, dataset_hash: str) -> dict[str, Any]:
    if not (path / "COMPLETE").is_file():
        raise Day11TrainingError(f"checkpoint is incomplete: {path}")
    manifest = load_json(path / "checkpoint_manifest.json")
    if manifest.get("domain") != "day11.training_checkpoint" or manifest.get("schema_version") != 1:
        raise Day11TrainingError("unexpected checkpoint manifest contract")
    for relative_path, expected_hash in manifest.get("files", {}).items():
        actual = file_sha256(path / relative_path)
        if actual != expected_hash:
            raise Day11TrainingError(f"checkpoint file hash mismatch: {relative_path}")
    state = load_json(path / "training_state.json")
    if semantic_hash(state) != manifest.get("training_state_hash"):
        raise Day11TrainingError("checkpoint training_state hash mismatch")
    if state.get("config_hash") != config_hash or state.get("dataset_hash") != dataset_hash:
        raise Day11TrainingError("checkpoint config/data identity differs from current run")
    return state


def restore_checkpoint_state(
    checkpoint_path: Path, optimizer: Any, scheduler: Any, device: Any
) -> dict[str, Any]:
    import torch

    optimizer.load_state_dict(torch.load(checkpoint_path / "optimizer.pt", map_location="cpu", weights_only=False))
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(device)
    scheduler.load_state_dict(torch.load(checkpoint_path / "scheduler.pt", map_location="cpu", weights_only=False))
    rng_state = torch.load(checkpoint_path / "rng_state.pt", map_location="cpu", weights_only=False)
    random.setstate(rng_state["python_random_state"])
    torch.set_rng_state(rng_state["torch_cpu_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state["torch_cuda_rng_state_all"])
    return load_json(checkpoint_path / "training_state.json")


def teacher_forced_metrics(model: Any, records: list[dict[str, Any]], tokenizer: Any, device: Any) -> dict[str, Any]:
    import torch

    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    with torch.no_grad():
        for record in records:
            batch = collate([record], tokenizer, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                )
            loss_sum, token_count, correct = masked_causal_loss_sum(outputs.logits, batch["labels"])
            total_loss += float(loss_sum.detach().float().cpu().item())
            total_tokens += token_count
            total_correct += correct
    if was_training:
        model.train()
    return {
        "loss_sum": total_loss,
        "supervised_tokens": total_tokens,
        "mean_loss": total_loss / total_tokens,
        "correct_tokens": total_correct,
        "token_accuracy": total_correct / total_tokens,
    }


def generation_records(
    model: Any, records: list[dict[str, Any]], tokenizer: Any, device: Any, max_new_tokens: int,
    *, stage: str, optimizer_step: int
) -> list[dict[str, Any]]:
    import torch

    was_training = model.training
    model.eval()
    results: list[dict[str, Any]] = []
    with torch.no_grad():
        for record in records:
            prompt_messages = record["messages"][:-1]
            prompt_ids = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=(
                        tokenizer.pad_token_id
                        if tokenizer.pad_token_id is not None
                        else tokenizer.eos_token_id
                    ),
                    use_cache=True,
                )[0, input_ids.shape[1] :].detach().cpu().tolist()
            output = tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            results.append(
                {
                    "stage": stage,
                    "optimizer_step": optimizer_step,
                    "sample_id": record["id"],
                    "reference": record["messages"][-1]["content"],
                    "output": output,
                    "output_token_ids": output_ids,
                }
            )
    if was_training:
        model.train()
    return results


def write_run_artifacts(run_dir: Path, artifact_dir: Path, summary: dict[str, Any]) -> None:
    trace = load_jsonl(run_dir / "step-trace.jsonl")
    first_three = [record for record in trace if record.get("optimizer_step", 0) <= 3]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    first_three_path = artifact_dir.parent / "logs/day11-first-three-steps.jsonl"
    first_three_path.parent.mkdir(parents=True, exist_ok=True)
    first_three_path.write_text(
        "".join(canonical_json(record) + "\n" for record in first_three), encoding="utf-8"
    )
    final_metrics = summary["final_teacher_forced"]
    base_metrics = summary.get("base_teacher_forced")
    report = [
        "# Day 11 SFT Step Audit",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Status: `{summary['status']}`",
        f"- Final optimizer step: `{summary['optimizer_step']}`",
        f"- Global supervised tokens: `{summary['global_supervised_tokens']}`",
        f"- Threshold reached: `{summary['threshold_reached']}`",
    ]
    if base_metrics:
        report.extend(
            [
                f"- Base teacher-forced loss: `{base_metrics['mean_loss']:.8f}`",
                f"- Base teacher-forced token accuracy: `{base_metrics['token_accuracy']:.6f}`",
            ]
        )
    report.extend(
        [
            f"- Final teacher-forced loss: `{final_metrics['mean_loss']:.8f}`",
            f"- Final teacher-forced token accuracy: `{final_metrics['token_accuracy']:.6f}`",
            "",
            "## Claim boundary",
            "",
            "This run proves only that the frozen pipeline can memorize these supervised tokens. "
            "It does not establish generalization, data quality, or a useful SFT recipe.",
            "",
            "## Evidence locations",
            "",
            f"- Full step trace: `{run_dir / 'step-trace.jsonl'}`",
            f"- Evaluations: `{run_dir / 'evaluations.jsonl'}`",
            f"- Generations: `{run_dir / 'generations.jsonl'}`",
            f"- Checkpoints: `{run_dir / 'checkpoints'}`",
            "",
        ]
    )
    report_path = artifact_dir / "day11-sft-step-audit.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    write_json(artifact_dir.parent / "logs/day11-run-summary.json", summary)


def train(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = load_config(args.config)
    config_hash = semantic_hash(config)
    run_dir = args.run_dir.resolve()
    trace_path = run_dir / "step-trace.jsonl"
    evaluations_path = run_dir / "evaluations.jsonl"
    generations_path = run_dir / "generations.jsonl"
    if args.resume_from is None and run_dir.exists() and any(run_dir.iterdir()):
        raise Day11TrainingError(f"refusing to reuse non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise Day11TrainingError("CUDA GPU is required for Day 11 training")
    if (
        config["model"]["parameter_dtype"] != "float32"
        or config["model"]["compute_dtype"] != "bfloat16"
        or config["model"]["loss_compute_dtype"] != "float32"
        or not torch.cuda.is_bf16_supported()
    ):
        raise Day11TrainingError("the selected GPU/runtime does not support the frozen BF16 contract")
    device = torch.device("cuda:0")
    set_reproducibility(int(config["seed"]), bool(config["training"]["deterministic_algorithms"]))

    base_model_path = args.model_path.resolve()
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True, use_fast=True)
    records, manifest = verify_and_encode_inputs(config, tokenizer, verify_base_model_path=base_model_path)
    dataset_hash = manifest["header"]["manifest_hash"]

    resume_state: dict[str, Any] | None = None
    model_load_path = base_model_path
    if args.resume_from is not None:
        resume_state = verify_checkpoint(args.resume_from, config_hash, dataset_hash)
        model_load_path = args.resume_from / "model"
    model = AutoModelForCausalLM.from_pretrained(
        model_load_path,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation=config["model"]["attention_implementation"],
    ).to(device)
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    if parameter_dtypes != ["torch.float32"]:
        raise Day11TrainingError(f"unexpected trainable parameter dtypes: {parameter_dtypes}")
    model.config.use_cache = False
    optimizer, scheduler = make_optimizer_and_scheduler(model, config)

    optimizer_step = 0
    sample_cursor = 0
    global_supervised_tokens = 0
    base_metrics: dict[str, Any] | None = None
    if args.resume_from is not None:
        restored = restore_checkpoint_state(args.resume_from, optimizer, scheduler, device)
        optimizer_step = int(restored["optimizer_step"])
        sample_cursor = int(restored["sample_cursor"])
        global_supervised_tokens = int(restored["global_supervised_tokens"])
        loaded_signature = state_signature(
            model,
            optimizer,
            scheduler,
            optimizer_step=optimizer_step,
            sample_cursor=sample_cursor,
            global_supervised_tokens=global_supervised_tokens,
        )
        for key, value in loaded_signature.items():
            if restored.get(key) != value:
                raise Day11TrainingError(f"checkpoint restore mismatch for {key}")
        existing_trace = load_jsonl(trace_path)
        if not existing_trace or existing_trace[-1].get("optimizer_step") != optimizer_step:
            raise Day11TrainingError("trace/checkpoint optimizer-step continuity failed")
    elif args.evaluate:
        base_metrics = teacher_forced_metrics(model, records, tokenizer, device)
        append_jsonl(evaluations_path, {"stage": "base", "optimizer_step": 0, **base_metrics})
        for result in generation_records(
            model,
            records,
            tokenizer,
            device,
            int(config["training"]["generation_max_new_tokens"]),
            stage="base",
            optimizer_step=0,
        ):
            append_jsonl(generations_path, result)

    micro_batch_size = int(config["training"]["micro_batch_size"])
    accumulation_steps = int(config["training"]["gradient_accumulation_steps"])
    max_steps = int(args.max_steps or config["training"]["max_optimizer_steps"])
    stop_after_step = int(args.stop_after_step or max_steps)
    if stop_after_step > max_steps or optimizer_step >= stop_after_step:
        raise Day11TrainingError("invalid stop/max/resume step relationship")
    checksum_through = int(args.checksum_through_step)
    save_steps = {int(value) for value in args.save_steps.split(",") if value.strip()}
    max_grad_norm = float(config["optimizer"]["max_grad_norm"])
    optimizer.zero_grad(set_to_none=True)
    previous_checksum = parameter_checksum(model) if optimizer_step < checksum_through else None
    threshold_reached = False
    latest_metrics: dict[str, Any] | None = None
    early_step = int(config["training"]["early_checkpoint_step"])
    evaluation_interval = int(config["training"]["evaluation_interval_steps"])

    while optimizer_step < stop_after_step:
        next_step = optimizer_step + 1
        microbatches, next_cursor = make_window(
            records, sample_cursor, micro_batch_size, accumulation_steps, int(config["seed"])
        )
        window_supervised_tokens = sum(
            record["supervised_tokens"] for microbatch in microbatches for record in microbatch
        )
        if window_supervised_tokens <= 0:
            raise Day11TrainingError("accumulation window has zero supervised tokens")
        parameter_before = previous_checksum if next_step <= checksum_through else None
        microbatch_trace: list[dict[str, Any]] = []
        total_loss_sum = 0.0
        model.train()
        for microbatch in microbatches:
            batch = collate(microbatch, tokenizer, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                )
            loss_sum, token_count, _ = masked_causal_loss_sum(outputs.logits, batch["labels"])
            (loss_sum / window_supervised_tokens).backward()
            numeric_loss_sum = float(loss_sum.detach().float().cpu().item())
            total_loss_sum += numeric_loss_sum
            microbatch_trace.append(
                {
                    "sample_ids": [record["id"] for record in microbatch],
                    "supervised_tokens": token_count,
                    "loss_sum": numeric_loss_sum,
                    "mean_loss": numeric_loss_sum / token_count,
                }
            )
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad_norm, foreach=False
        )
        grad_norm = float(grad_norm_tensor.detach().float().cpu().item())
        if not math.isfinite(grad_norm) or grad_norm == 0.0:
            raise Day11TrainingError(f"invalid gradient norm at step {next_step}: {grad_norm}")
        gradient_dtypes = sorted(
            {str(parameter.grad.dtype) for parameter in model.parameters() if parameter.grad is not None}
        )
        if gradient_dtypes != ["torch.float32"]:
            raise Day11TrainingError(f"unexpected gradient dtypes: {gradient_dtypes}")
        clip_coefficient = min(1.0, max_grad_norm / (grad_norm + 1e-12))
        lr_before = [float(group["lr"]) for group in optimizer.param_groups]
        optimizer.step()
        optimizer_state_dtypes = sorted(
            {
                str(value.dtype)
                for state in optimizer.state.values()
                for key, value in state.items()
                if key in {"exp_avg", "exp_avg_sq"}
            }
        )
        if optimizer_state_dtypes != ["torch.float32"]:
            raise Day11TrainingError(f"unexpected Adam state dtypes: {optimizer_state_dtypes}")
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        gradients_cleared = all(parameter.grad is None for parameter in model.parameters())
        if not gradients_cleared:
            raise Day11TrainingError("zero_grad(set_to_none=True) did not clear all gradients")
        optimizer_step = next_step
        sample_cursor = next_cursor
        global_supervised_tokens += window_supervised_tokens
        parameter_after = parameter_checksum(model) if optimizer_step <= checksum_through else None
        previous_checksum = parameter_after
        if parameter_before is not None and parameter_before == parameter_after:
            raise Day11TrainingError(f"parameters did not change at optimizer step {optimizer_step}")
        step_record = {
            "event": "optimizer_step",
            "optimizer_step": optimizer_step,
            "sample_cursor": sample_cursor,
            "sample_ids": [record["id"] for batch_records in microbatches for record in batch_records],
            "microbatches": microbatch_trace,
            "window_supervised_tokens": window_supervised_tokens,
            "loss_sum": total_loss_sum,
            "mean_loss": total_loss_sum / window_supervised_tokens,
            "grad_norm_before_clip": grad_norm,
            "max_grad_norm": max_grad_norm,
            "clip_coefficient": clip_coefficient,
            "optimizer_lr": lr_before,
            "scheduler_lr_after_step": [float(group["lr"]) for group in optimizer.param_groups],
            "parameter_dtypes": parameter_dtypes,
            "gradient_dtypes": gradient_dtypes,
            "optimizer_state_dtypes": optimizer_state_dtypes,
            "grad_scaler": None,
            "gradients_cleared_after_zero_grad": gradients_cleared,
            "parameter_checksum_before": parameter_before,
            "parameter_checksum_after": parameter_after,
            "global_supervised_tokens": global_supervised_tokens,
            "operation_order": ["forward", "loss_sum/window_tokens", "backward", "clip", "optimizer", "scheduler", "zero_grad"],
        }
        append_jsonl(trace_path, step_record)

        should_evaluate = args.evaluate and (
            optimizer_step == early_step
            or optimizer_step % evaluation_interval == 0
            or optimizer_step == stop_after_step
        )
        if should_evaluate:
            latest_metrics = teacher_forced_metrics(model, records, tokenizer, device)
            append_jsonl(
                evaluations_path,
                {"stage": "train", "optimizer_step": optimizer_step, **latest_metrics},
            )
            if optimizer_step == early_step:
                for result in generation_records(
                    model,
                    records,
                    tokenizer,
                    device,
                    int(config["training"]["generation_max_new_tokens"]),
                    stage="early",
                    optimizer_step=optimizer_step,
                ):
                    append_jsonl(generations_path, result)
            threshold_reached = (
                optimizer_step >= int(config["training"]["early_stop_min_steps"])
                and latest_metrics["token_accuracy"]
                >= float(config["training"]["teacher_forced_accuracy_threshold"])
            )

        if optimizer_step in save_steps:
            signature = state_signature(
                model,
                optimizer,
                scheduler,
                optimizer_step=optimizer_step,
                sample_cursor=sample_cursor,
                global_supervised_tokens=global_supervised_tokens,
            )
            save_checkpoint(
                run_dir / f"checkpoints/checkpoint-step-{optimizer_step:06d}",
                model,
                tokenizer,
                optimizer,
                scheduler,
                {
                    "config_hash": config_hash,
                    "dataset_hash": dataset_hash,
                    "base_model_id": config["model"]["id"],
                    "base_model_revision": config["model"]["revision"],
                    **signature,
                },
            )
        if threshold_reached:
            break

    if latest_metrics is None:
        latest_metrics = teacher_forced_metrics(model, records, tokenizer, device)
    if args.evaluate:
        for result in generation_records(
            model,
            records,
            tokenizer,
            device,
            int(config["training"]["generation_max_new_tokens"]),
            stage="final",
            optimizer_step=optimizer_step,
        ):
            append_jsonl(generations_path, result)
    final_signature = state_signature(
        model,
        optimizer,
        scheduler,
        optimizer_step=optimizer_step,
        sample_cursor=sample_cursor,
        global_supervised_tokens=global_supervised_tokens,
    )
    write_json(run_dir / "final-state-signature.json", final_signature)

    final_checkpoint_path: str | None = None
    if args.save_final_checkpoint:
        final_path = run_dir / f"checkpoints/checkpoint-step-{optimizer_step:06d}-final"
        save_checkpoint(
            final_path,
            model,
            tokenizer,
            optimizer,
            scheduler,
            {
                "config_hash": config_hash,
                "dataset_hash": dataset_hash,
                "base_model_id": config["model"]["id"],
                "base_model_revision": config["model"]["revision"],
                **final_signature,
            },
        )
        final_checkpoint_path = str(final_path)

    summary = {
        "domain": "day11.training_run_summary",
        "schema_version": 1,
        "status": "threshold_reached" if threshold_reached else "completed_without_threshold",
        "created_at": utc_now(),
        "run_dir": str(run_dir),
        "config_hash": config_hash,
        "dataset_hash": dataset_hash,
        "optimizer_step": optimizer_step,
        "sample_cursor": sample_cursor,
        "global_supervised_tokens": global_supervised_tokens,
        "threshold_reached": threshold_reached,
        "threshold": config["training"]["teacher_forced_accuracy_threshold"],
        "base_teacher_forced": base_metrics,
        "final_teacher_forced": latest_metrics,
        "final_state_signature": final_signature,
        "final_checkpoint": final_checkpoint_path,
    }
    write_json(run_dir / "run-summary.json", summary)
    if args.artifact_report_dir is not None:
        write_run_artifacts(run_dir, args.artifact_report_dir, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def compare_resume(args: argparse.Namespace) -> None:
    reference_trace = load_jsonl(args.reference_run / "step-trace.jsonl")
    resumed_trace = load_jsonl(args.resumed_run / "step-trace.jsonl")
    reference_by_step = {record["optimizer_step"]: record for record in reference_trace}
    resumed_by_step = {record["optimizer_step"]: record for record in resumed_trace}
    compared_steps = sorted(set(reference_by_step) & set(resumed_by_step))
    if compared_steps != list(range(1, args.total_steps + 1)):
        raise Day11TrainingError(f"resume comparison has incomplete steps: {compared_steps}")
    exact_fields = (
        "sample_cursor",
        "sample_ids",
        "window_supervised_tokens",
        "global_supervised_tokens",
        "optimizer_lr",
        "scheduler_lr_after_step",
        "parameter_checksum_after",
    )
    float_fields = ("loss_sum", "mean_loss", "grad_norm_before_clip", "clip_coefficient")
    mismatches: list[str] = []
    for step in compared_steps:
        reference = reference_by_step[step]
        resumed = resumed_by_step[step]
        for field in exact_fields:
            if reference.get(field) != resumed.get(field):
                mismatches.append(f"step {step} field {field}")
        for field in float_fields:
            if not math.isclose(float(reference[field]), float(resumed[field]), rel_tol=0.0, abs_tol=1e-9):
                mismatches.append(f"step {step} field {field}")
    reference_state = load_json(args.reference_run / "final-state-signature.json")
    resumed_state = load_json(args.resumed_run / "final-state-signature.json")
    if reference_state != resumed_state:
        mismatches.append("final state signature")
    status = "pass" if not mismatches else "fail"
    result = {
        "domain": "day11.resume_comparison",
        "schema_version": 1,
        "status": status,
        "interruption_step": args.interruption_step,
        "total_steps": args.total_steps,
        "compared_steps": compared_steps,
        "mismatches": mismatches,
        "reference_final_state": reference_state,
        "resumed_final_state": resumed_state,
    }
    write_json(args.output_json, result)
    lines = [
        "# Day 11 Checkpoint Resume Audit",
        "",
        f"- Status: `{status}`",
        f"- Interrupted after optimizer step: `{args.interruption_step}`",
        f"- Compared through optimizer step: `{args.total_steps}`",
        f"- Exact final parameter checksum: `{reference_state.get('parameter_checksum')}`",
        f"- Exact final optimizer checksum: `{reference_state.get('optimizer_checksum')}`",
        f"- Restored sample cursor: `{resumed_state.get('sample_cursor')}`",
        f"- Restored scheduler last_epoch: `{resumed_state.get('scheduler_last_epoch')}`",
        "",
        "This comparison checks uninterrupted versus fresh-process resumed training. "
        "A weights-only load cannot pass the optimizer, scheduler, RNG, and sample-cursor checks.",
        "",
    ]
    if mismatches:
        lines.extend(["## Mismatches", "", *[f"- {item}" for item in mismatches], ""])
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text("\n".join(lines), encoding="utf-8")
    if mismatches:
        raise Day11TrainingError("resume comparison failed: " + ", ".join(mismatches))
    print(json.dumps(result, ensure_ascii=False, indent=2))


def preflight(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoTokenizer

    config = load_config(args.config)
    model_path = args.model_path.resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    records, manifest = verify_and_encode_inputs(config, tokenizer, verify_base_model_path=model_path)
    run_parent = args.run_root.resolve()
    run_parent.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(run_parent).free / 1024**3
    required_gib = float(config["storage"]["minimum_free_gib"])
    if free_gib < required_gib:
        raise Day11TrainingError(f"only {free_gib:.1f} GiB free; {required_gib:.1f} GiB required")
    if not torch.cuda.is_available():
        raise Day11TrainingError("CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise Day11TrainingError("GPU/runtime does not support BF16")
    properties = torch.cuda.get_device_properties(0)
    gpu_memory_gib = properties.total_memory / 1024**3
    minimum_gpu_memory_gib = float(config["hardware"]["minimum_gpu_memory_gib"])
    if gpu_memory_gib < minimum_gpu_memory_gib:
        raise Day11TrainingError(
            f"GPU has {gpu_memory_gib:.1f} GiB; {minimum_gpu_memory_gib:.1f} GiB is required"
        )
    result = {
        "status": "ready",
        "timestamp": utc_now(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": properties.name,
        "gpu_total_memory_gib": gpu_memory_gib,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "free_disk_gib": free_gib,
        "record_count": len(records),
        "total_supervised_tokens": manifest["header"]["total_supervised_tokens"],
        "dataset_hash": manifest["header"]["manifest_hash"],
        "model_revision": config["model"]["revision"],
    }
    if args.output is not None:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    preflight_parser.add_argument("--model-path", type=Path, required=True)
    preflight_parser.add_argument("--run-root", type=Path, required=True)
    preflight_parser.add_argument("--output", type=Path)
    preflight_parser.set_defaults(function=preflight)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    train_parser.add_argument("--model-path", type=Path, required=True)
    train_parser.add_argument("--run-dir", type=Path, required=True)
    train_parser.add_argument("--max-steps", type=int)
    train_parser.add_argument("--stop-after-step", type=int)
    train_parser.add_argument("--resume-from", type=Path)
    train_parser.add_argument("--save-steps", default="")
    train_parser.add_argument("--checksum-through-step", type=int, default=3)
    train_parser.add_argument("--evaluate", action="store_true")
    train_parser.add_argument("--save-final-checkpoint", action="store_true")
    train_parser.add_argument("--artifact-report-dir", type=Path)
    train_parser.set_defaults(function=train)

    compare_parser = subparsers.add_parser("compare-resume")
    compare_parser.add_argument("--reference-run", type=Path, required=True)
    compare_parser.add_argument("--resumed-run", type=Path, required=True)
    compare_parser.add_argument("--interruption-step", type=int, required=True)
    compare_parser.add_argument("--total-steps", type=int, required=True)
    compare_parser.add_argument("--output-json", type=Path, required=True)
    compare_parser.add_argument("--output-markdown", type=Path, required=True)
    compare_parser.set_defaults(function=compare_resume)
    return parser


def main() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = build_parser()
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
