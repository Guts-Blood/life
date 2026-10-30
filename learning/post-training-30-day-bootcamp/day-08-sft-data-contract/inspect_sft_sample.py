#!/usr/bin/env python3
"""Inspect one Qwen-style SFT sample from messages through shifted targets.

This is deliberately narrow: text-only system/user/assistant messages, right
truncation, all-assistant-turn loss, and no packing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


IGNORE_INDEX = -100
MODEL_ID = "Qwen/Qwen3-0.6B-Base"
MODEL_REVISION = "ddc928429ed09d9ad603fd762053d0434c15e865"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
RESERVED_MARKERS = (IM_START, IM_END)


class ContractError(ValueError):
    def __init__(self, stage: str, code: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.code = code


def validate_sample_schema(sample: dict[str, Any]) -> None:
    sample_id = sample.get("id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ContractError("schema", "invalid_id", "sample id must be a non-empty string")

    messages = sample.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ContractError("schema", "missing_messages", "messages must be a non-empty list")
    metadata = sample.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ContractError("schema", "invalid_metadata", "metadata must be an object when present")

    roles: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ContractError("schema", "invalid_message", f"message {index} must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ContractError("schema", "invalid_role", f"message {index} has invalid role {role!r}")
        if not isinstance(content, str) or not content.strip():
            code = "empty_assistant" if role == "assistant" else "empty_content"
            raise ContractError("schema", code, f"message {index} has empty {role} content")
        if any(marker in content for marker in RESERVED_MARKERS):
            raise ContractError(
                "schema",
                "reserved_marker",
                f"message {index} contains a reserved chat-template marker",
            )
        roles.append(role)

    cursor = 1 if roles[0] == "system" else 0
    if "system" in roles[cursor:]:
        raise ContractError("schema", "system_position", "system is allowed only once at position 0")
    expected = "user"
    for index in range(cursor, len(roles)):
        if roles[index] != expected:
            raise ContractError(
                "schema",
                "role_order",
                f"message {index} must be {expected}, got {roles[index]}",
            )
        expected = "assistant" if expected == "user" else "user"
    if roles[-1] != "assistant":
        raise ContractError("schema", "missing_assistant", "final message must be assistant")


def validate_unique_ids(samples: Iterable[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for sample in samples:
        sample_id = sample.get("id")
        if sample_id in seen:
            raise ContractError("schema", "duplicate_id", f"duplicate sample id: {sample_id}")
        seen.add(sample_id)


def _rendered_role_spans(rendered: str, messages: list[dict[str, str]]) -> list[tuple[str, int, int]]:
    """Return role body spans, including each role's trailing <|im_end|>."""
    spans: list[tuple[str, int, int]] = []
    cursor = 0
    for index, message in enumerate(messages):
        role = message["role"]
        header = f"{IM_START}{role}\n"
        header_start = rendered.find(header, cursor)
        if header_start < 0:
            raise ContractError("encoded", "template_mismatch", f"cannot locate rendered header for message {index}")
        body_start = header_start + len(header)
        end_start = rendered.find(IM_END, body_start)
        if end_start < 0:
            raise ContractError("encoded", "missing_role_end", f"message {index} has no rendered {IM_END}")
        body_end = end_start + len(IM_END)
        spans.append((role, body_start, body_end))
        cursor = body_end
    return spans


def _char_to_token(encoding: Any, start: int, end: int, *, reverse: bool = False) -> int:
    positions = range(end - 1, start - 1, -1) if reverse else range(start, end)
    for char_position in positions:
        token_position = encoding.char_to_token(char_position)
        if token_position is not None:
            return token_position
    raise ContractError("encoded", "offset_mapping", f"cannot map character span {start}:{end} to tokens")


def encode_sample(tokenizer: Any, sample: dict[str, Any], max_length: int) -> dict[str, Any]:
    validate_sample_schema(sample)
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    messages = sample["messages"]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    role_spans = _rendered_role_spans(rendered, messages)
    encoding = tokenizer(
        rendered,
        add_special_tokens=False,
        return_attention_mask=True,
        return_offsets_mapping=True,
    )
    input_ids = list(encoding["input_ids"])
    attention_mask = list(encoding["attention_mask"])
    assistant_mask = [0] * len(input_ids)
    assistant_token_spans: list[tuple[int, int]] = []

    for role, char_start, char_end in role_spans:
        if role != "assistant":
            continue
        token_start = _char_to_token(encoding, char_start, char_end)
        token_end = _char_to_token(encoding, char_start, char_end, reverse=True)
        assistant_token_spans.append((token_start, token_end))
        for token_position in range(token_start, token_end + 1):
            assistant_mask[token_position] = 1

    labels = [token_id if supervised else IGNORE_INDEX for token_id, supervised in zip(input_ids, assistant_mask)]
    original_length = len(input_ids)
    input_ids = input_ids[:max_length]
    attention_mask = attention_mask[:max_length]
    assistant_mask = assistant_mask[:max_length]
    labels = labels[:max_length]

    supervised_positions = [index for index, label in enumerate(labels) if label != IGNORE_INDEX]
    if not supervised_positions:
        raise ContractError("encoded", "zero_supervision", "no supervised token remains after truncation")

    final_assistant_end = assistant_token_spans[-1][1]
    if final_assistant_end >= len(input_ids):
        raise ContractError(
            "encoded",
            "truncated_assistant",
            "truncation removed the final assistant turn boundary",
        )
    im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
    if input_ids[final_assistant_end] != im_end_id or labels[final_assistant_end] == IGNORE_INDEX:
        raise ContractError("encoded", "assistant_end", "final assistant end token is not supervised")

    shifted_targets = [
        {
            "logit_position": target_position - 1,
            "target_position": target_position,
            "target_id": labels[target_position],
        }
        for target_position in supervised_positions
        if target_position > 0
    ]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    rows = [
        {
            "position": index,
            "token_id": token_id,
            "token": tokens[index],
            "attention": attention_mask[index],
            "label": labels[index],
            "supervised": bool(assistant_mask[index]),
            "predicted_by_logit": index - 1 if labels[index] != IGNORE_INDEX and index > 0 else None,
        }
        for index, token_id in enumerate(input_ids)
    ]
    template = tokenizer.chat_template or ""
    return {
        "sample_id": sample["id"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "policy": {
            "loss": "all assistant bodies plus each <|im_end|>",
            "thinking": "non-thinking; frozen template has no enable_thinking branch",
            "truncation": "right; reject if final assistant boundary is removed",
            "padding": "not applied here; batch padding requires attention=0 and label=-100",
            "ignore_index": IGNORE_INDEX,
        },
        "raw_messages": messages,
        "metadata": sample.get("metadata", {}),
        "rendered_text": rendered,
        "original_length": original_length,
        "encoded_length": len(input_ids),
        "truncated": original_length > len(input_ids),
        "effective_label_tokens": len(supervised_positions),
        "rows": rows,
        "shifted_targets": shifted_targets,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def check_golden_cases(tokenizer: Any, cases: list[dict[str, Any]]) -> None:
    failures: list[str] = []
    for case in cases:
        expected = case["expected_stage"]
        try:
            encode_sample(tokenizer, case, case["max_length"])
            actual = "accept"
            actual_code = None
        except ContractError as error:
            actual = error.stage
            actual_code = error.code
        expected_code = case.get("expected_code")
        if actual != expected or actual_code != expected_code:
            failures.append(
                f"{case['id']}: expected {expected}/{expected_code}, got {actual}/{actual_code}"
            )
        suffix = "" if actual_code is None else f"/{actual_code}"
        print(f"{case['id']}: {actual}{suffix}")
    if failures:
        raise SystemExit("golden case failures:\n" + "\n".join(failures))


def print_markdown(audit: dict[str, Any]) -> None:
    print(f"# SFT token audit: {audit['sample_id']}")
    print(f"\n- template SHA-256: `{audit['chat_template_sha256']}`")
    print(f"- length: `{audit['encoded_length']}` / original `{audit['original_length']}`")
    print(f"- effective label tokens: `{audit['effective_label_tokens']}`")
    print("\n## Rendered text\n")
    print("```text")
    print(audit["rendered_text"], end="" if audit["rendered_text"].endswith("\n") else "\n")
    print("```")
    print("\n## Token / label trace\n")
    print("| pos | id | token | attn | label | supervised | predicted by logit |")
    print("|---:|---:|---|---:|---:|:---:|---:|")
    for row in audit["rows"]:
        token = row["token"].replace("|", "\\|")
        predicted_by = "" if row["predicted_by_logit"] is None else row["predicted_by_logit"]
        print(
            f"| {row['position']} | {row['token_id']} | `{token}` | {row['attention']} | "
            f"{row['label']} | {'yes' if row['supervised'] else 'no'} | {predicted_by} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "data"
            / "day08-template-golden-cases.jsonl"
        ),
    )
    parser.add_argument("--sample-id")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--check-golden", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    snapshot_path = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        allow_patterns=[
            "added_tokens.json",
            "chat_template.jinja",
            "merges.txt",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ],
    )
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_path,
        local_files_only=True,
        use_fast=True,
    )
    cases = load_jsonl(args.input)
    if args.check_golden:
        check_golden_cases(tokenizer, cases)
        return

    sample_id = args.sample_id or cases[0]["id"]
    try:
        sample = next(case for case in cases if case["id"] == sample_id)
    except StopIteration as error:
        raise SystemExit(f"unknown sample id: {sample_id}") from error
    audit = encode_sample(tokenizer, sample, args.max_length)
    if args.json:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
    else:
        print_markdown(audit)


if __name__ == "__main__":
    main()
