#!/usr/bin/env python3
"""Compute the preregistered Transformers loss on the exact Day18 text batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


EXPECTED_MTP_TENSORS = {
    "mtp.fc.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
}


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 2:
        raise ValueError(f"expected exactly two fixture rows, got {len(rows)}")
    encoded = json.dumps(rows, ensure_ascii=False).lower()
    if any(marker in encoded for marker in ("<image>", "<video>", "<audio>")):
        raise ValueError("fixture is not text-only")
    return rows


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weight_tensor_names(model_dir: Path) -> set[str]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        return set(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"])
    from safetensors import safe_open

    names: set[str] = set()
    for path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            names.update(handle.keys())
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("USE_HF", "1")
    os.environ.setdefault("USE_MCORE_GDN", "1")

    import torch
    import torch.nn.functional as functional
    from swift import get_model_processor, get_template
    from swift.template import TemplateInputs
    from swift.utils import to_device

    rows = load_rows(args.fixture)
    tensor_names = weight_tensor_names(args.model)
    missing_mtp_tensors = sorted(EXPECTED_MTP_TENSORS - tensor_names)
    if missing_mtp_tensors:
        raise RuntimeError(f"model/export is missing Qwen3.5 MTP tensors: {missing_mtp_tensors}")
    model, processor = get_model_processor(
        str(args.model),
        model_type="qwen3_5",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        use_hf=True,
        download_model=False,
    )
    if model.__class__.__name__ != "Qwen3_5ForConditionalGeneration":
        raise RuntimeError(f"wrong conditional loader: {model.__class__.__name__}")

    template = get_template(
        processor,
        max_length=args.max_length,
        padding_free=False,
        loss_scale="default+ignore_empty_think",
        enable_thinking=False,
        add_non_thinking_prefix=True,
    )
    # Match Swift's SFT pipeline: Qwen3.5 needs the concrete model while the
    # collator computes multimodal RoPE positions, even for a text-only batch.
    if template.use_model:
        template.model = model
    template.set_mode("train")
    encoded = [template.encode(TemplateInputs.from_dict(row), return_length=True) for row in rows]
    batch = to_device(template.data_collator(encoded), "cuda:0")
    template.register_post_encode_hook([model])
    batch.pop("text_position_ids", None)

    forbidden_tensor_keys = sorted(
        key for key in batch if any(token in key.lower() for token in ("pixel", "image", "video", "audio"))
    )
    if forbidden_tensor_keys:
        raise RuntimeError(f"text batch unexpectedly contains multimodal tensors: {forbidden_tensor_keys}")

    forbidden_token_ids = {248053, 248054, 248056, 248057}
    input_ids = batch.get("input_ids")
    forbidden_token_hits = sorted(
        token_id for token_id in forbidden_token_ids if input_ids is not None and input_ids.eq(token_id).any().item()
    )
    if forbidden_token_hits:
        raise RuntimeError(f"text batch unexpectedly contains visual special token IDs: {forbidden_token_hits}")

    labels = batch["labels"]
    model_inputs = {key: value for key, value in batch.items() if key != "loss_scale"}
    with torch.inference_mode():
        outputs = model(**model_inputs, use_cache=False)
        logits = outputs.logits.float()

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels.ne(-100)
    if not valid.any():
        raise RuntimeError("fixture has zero supervised tokens")
    safe_labels = shift_labels.masked_fill(~valid, 0)
    per_token_loss = functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        safe_labels.view(-1),
        reduction="none",
    ).view_as(shift_labels)
    weights = valid.float()
    if batch.get("loss_scale") is not None:
        weights = batch["loss_scale"][:, 1:].float() * valid.float()
    loss = (per_token_loss * weights).sum() / weights.sum()
    if not math.isfinite(loss.item()):
        raise RuntimeError(f"non-finite reference loss: {loss.item()}")

    predictions = shift_logits.argmax(dim=-1)
    correct = predictions.eq(shift_labels) & valid
    per_row = []
    for row_index in range(shift_labels.shape[0]):
        row_weights = weights[row_index]
        row_weight_sum = row_weights.sum()
        row_supervised_tokens = valid[row_index].sum()
        if row_weight_sum.item() <= 0 or row_supervised_tokens.item() <= 0:
            raise RuntimeError(f"fixture row {row_index} has zero supervised loss weight or tokens")
        row_loss_numerator = (per_token_loss[row_index] * row_weights).sum()
        row_correct_tokens = correct[row_index].sum()
        per_row.append(
            {
                "row": row_index,
                "supervised_tokens": int(row_supervised_tokens.item()),
                "correct_tokens": int(row_correct_tokens.item()),
                "token_accuracy": float(row_correct_tokens.item() / row_supervised_tokens.item()),
                "loss_weight_sum": float(row_weight_sum.item()),
                "weighted_loss_sum": float(row_loss_numerator.item()),
                "mean_loss": float((row_loss_numerator / row_weight_sum).item()),
            }
        )

    correct_tokens = int(correct.sum().item())
    supervised_tokens = int(valid.sum().item())
    teacher_forced = {
        "supervised_tokens": supervised_tokens,
        "correct_tokens": correct_tokens,
        "token_accuracy": float(correct_tokens / supervised_tokens),
        "loss_weight_sum": float(weights.sum().item()),
        "weighted_loss_sum": float((per_token_loss * weights).sum().item()),
        "mean_loss": float(loss.item()),
        "rows": per_row,
    }

    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "process_id": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model_path": str(args.model.resolve()),
        "model_class": model.__class__.__name__,
        "fixture_sha256": file_sha256(args.fixture),
        "batch_shape": list(labels.shape),
        "supervised_tokens": supervised_tokens,
        "loss_weight_sum": float(weights.sum().item()),
        "reference_loss": float(loss.item()),
        "teacher_forced": teacher_forced,
        "forbidden_tensor_keys": forbidden_tensor_keys,
        "forbidden_visual_token_ids": forbidden_token_hits,
        "mtp_tensor_count": len(EXPECTED_MTP_TENSORS),
        "mtp_tensor_names_sha256": hashlib.sha256(
            "\n".join(sorted(EXPECTED_MTP_TENSORS)).encode("utf-8")
        ).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
