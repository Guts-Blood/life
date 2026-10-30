#!/usr/bin/env python3
"""Pure Day 20 data, LoRA, and training-configuration contracts.

This module intentionally depends only on the Python standard library.  Dataset
adapters are responsible for producing tokenizer-audited normalized JSONL rows;
this module validates and selects those rows without importing Transformers,
Swift, or a dataset client.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping, Sequence


SKILLS = ("general", "math", "finance", "code")

MAIN_SUPERVISED_TOKENS = 256_000
MAIN_TOKENS_PER_SKILL = 64_000
PROBE_SUPERVISED_TOKENS = 16_000
PROBE_TOKENS_PER_SKILL = 4_000
DIAGNOSTIC_RECORDS_PER_SKILL = 8
MIN_PROBE_RECORDS_BY_FORMAT = {
    "general_mcq": 5,
    "general_instruction": 5,
    "math_reasoning": 10,
    "finance_value_scale": 10,
    "code_continuation": 10,
}
MIN_MAIN_RECORDS_BY_FORMAT = {
    "general_mcq": 20,
    "general_instruction": 20,
    "math_reasoning": 40,
    "finance_value_scale": 40,
    "code_continuation": 40,
}

TARGET_FORMATS = {
    "general_mcq": "general",
    "general_instruction": "general",
    "math_reasoning": "math",
    "finance_value_scale": "finance",
    "code_continuation": "code",
}
MAIN_TOKENS_BY_FORMAT = {
    # Qwen3.5 encodes every canonical MMLU target as exactly six supervised
    # tokens, so 32,000 is arithmetically unreachable without duplication.
    "general_mcq": 31_998,
    "general_instruction": 32_002,
    "math_reasoning": 64_000,
    "finance_value_scale": 64_000,
    "code_continuation": 64_000,
}
PROBE_TOKENS_BY_FORMAT = {
    "general_mcq": 1_998,
    "general_instruction": 2_002,
    "math_reasoning": 4_000,
    "finance_value_scale": 4_000,
    "code_continuation": 4_000,
}

PROBE_LEARNING_RATES = (1e-5, 3e-5, 1e-4)
MAIN_CHECKPOINT_TOKENS = (64_000, 153_600, 256_000)

QWEN35_TEMPLATE_CONTRACT = {
    "model_type": "qwen3_5",
    "template": "qwen3_5",
    "enable_thinking": False,
    "add_non_thinking_prefix": True,
    "loss_scale": "default+ignore_empty_think",
    "padding_free": False,
    "packing": False,
    "max_length": 2304,
    "truncation_strategy": "raise",
}

LORA_TARGET_LEAVES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
# The required ``language_model.layers`` segment is the safety boundary that
# excludes the visual tower, aligner, embeddings, and lm_head.  Known PEFT
# wrappers may repeat ``model``/``base_model`` before that boundary.
LORA_TARGET_REGEX = (
    r"^(?:(?:base_model|model)\.)*language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)

FINANCE_SCALES = ("none", "percent", "thousand", "million", "billion")
SPECIAL_TARGET_MARKERS = (
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<image>",
    "<video>",
    "<audio>",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MCQ_RE = re.compile(r"^Final answer:\s*([A-Da-d])$")
_MATH_RE = re.compile(
    r"^(?P<reasoning>.+)\nFinal answer:\s*"
    r"(?P<answer>[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?)$",
    re.DOTALL,
)
_FINANCE_RE = re.compile(
    r"^Final answer:\s*"
    r"(?P<answer>[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"(?:(?P<percent>%)|\s+(?P<word_scale>thousand|million|billion))?$",
    re.IGNORECASE,
)
_LORA_PARAMETER_RE = re.compile(
    r"^(?P<module>.+)\.lora_(?P<side>A|B)(?:\.[^.]+)?\.weight$"
)


class Day20ContractError(ValueError):
    """A Day 20 data, configuration, or lineage invariant failed."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise Day20ContractError(f"{label} must be a lowercase bare SHA-256")
    return value


def _normalize_plain_text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise Day20ContractError(f"{label} must be text")
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise Day20ContractError(f"{label} must not be empty")
    if "\x00" in normalized:
        raise Day20ContractError(f"{label} contains a NUL byte")
    return normalized


def _normalize_code_text(value: Any) -> str:
    if not isinstance(value, str):
        raise Day20ContractError("code target must be text")
    lines = value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    normalized = "\n".join(line.rstrip() for line in lines)
    if not normalized:
        raise Day20ContractError("code target must not be empty")
    return normalized


def _reject_shared_target_failures(target: str) -> None:
    if any(marker in target for marker in SPECIAL_TARGET_MARKERS):
        raise Day20ContractError("assistant target contains a template/media marker")
    if "```" in target:
        raise Day20ContractError("assistant target must not contain Markdown fences")


def _canonical_number(value: str) -> str:
    compact = value.replace(",", "")
    try:
        number = Decimal(compact)
    except InvalidOperation as error:
        raise Day20ContractError(f"invalid canonical numeric answer: {value!r}") from error
    if not number.is_finite():
        raise Day20ContractError("numeric answer must be finite")
    rendered = format(number, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    if rendered in {"-0", "+0", ""}:
        rendered = "0"
    return rendered


def normalize_assistant_target(
    skill: str,
    target_format: str,
    target: Any,
    *,
    code_prefix: str | None = None,
    finance_scale: str | None = None,
) -> str:
    """Normalize one assistant target and enforce its task-specific grammar."""
    if skill not in SKILLS:
        raise Day20ContractError(f"unknown skill: {skill!r}")
    if TARGET_FORMATS.get(target_format) != skill:
        raise Day20ContractError(
            f"target format {target_format!r} does not belong to skill {skill!r}"
        )

    if target_format == "code_continuation":
        normalized = _normalize_code_text(target)
    else:
        normalized = _normalize_plain_text(target, "assistant target")
    _reject_shared_target_failures(normalized)

    if target_format == "general_mcq":
        match = _MCQ_RE.fullmatch(normalized)
        if not match:
            raise Day20ContractError(
                "general_mcq target must be exactly 'Final answer: <A|B|C|D>'"
            )
        return f"Final answer: {match.group(1).upper()}"

    if target_format == "general_instruction":
        if normalized.startswith(("Reasoning:", "Calculation:")):
            raise Day20ContractError(
                "general replay target carries a cross-task reasoning prefix"
            )
        return normalized

    if target_format == "math_reasoning":
        match = _MATH_RE.fullmatch(normalized)
        if not match or not match.group("reasoning").strip():
            raise Day20ContractError(
                "math target needs reasoning followed by 'Final answer: <number>'"
            )
        reasoning = match.group("reasoning").strip()
        if reasoning.startswith(("Reasoning:", "Calculation:")):
            raise Day20ContractError("math target uses a banned global style prefix")
        return f"{reasoning}\nFinal answer: {_canonical_number(match.group('answer'))}"

    if target_format == "finance_value_scale":
        match = _FINANCE_RE.fullmatch(normalized)
        if not match:
            raise Day20ContractError(
                "finance target must be one scorer-compatible final-answer line"
            )
        if not isinstance(finance_scale, str):
            raise Day20ContractError("finance_value_scale requires finance_scale")
        scale = finance_scale.strip().lower()
        if scale not in FINANCE_SCALES:
            raise Day20ContractError(
                f"finance scale must be one of {', '.join(FINANCE_SCALES)}"
            )
        observed_scale = (
            "percent"
            if match.group("percent")
            else (match.group("word_scale") or "none").lower()
        )
        if observed_scale != scale:
            raise Day20ContractError(
                f"finance target suffix {observed_scale!r} differs from "
                f"finance_scale {scale!r}"
            )
        answer = _canonical_number(match.group("answer"))
        suffix = "%" if scale == "percent" else (f" {scale}" if scale != "none" else "")
        return f"Final answer: {answer}{suffix}"

    if not isinstance(code_prefix, str) or not code_prefix.strip():
        raise Day20ContractError("code_continuation requires a non-empty code_prefix")
    prefix = code_prefix.replace("\r\n", "\n").replace("\r", "\n").rstrip()
    first_line = normalized.splitlines()[0]
    if re.match(r"^(?:async\s+def|def|class)\s+", first_line):
        raise Day20ContractError("code target redefines a function/class instead of continuing")
    if prefix in normalized or normalized in prefix:
        raise Day20ContractError("code target repeats or is already contained in its prefix")
    try:
        contained = ast.parse(f"def __day20_containment__():\n{normalized}\n")
    except SyntaxError as error:
        raise Day20ContractError(
            "code target is not a function-body continuation"
        ) from error
    if (
        len(contained.body) != 1
        or not isinstance(contained.body[0], ast.FunctionDef)
        or contained.body[0].name != "__day20_containment__"
        or not contained.body[0].body
    ):
        raise Day20ContractError("code target escapes the function body")
    try:
        ast.parse(f"{prefix}\n{normalized}\n")
    except SyntaxError as error:
        raise Day20ContractError(
            f"code prefix plus continuation is not valid Python: {error.msg}"
        ) from error
    return normalized


def normalize_and_validate_record(
    record: Mapping[str, Any],
    *,
    expected_skill: str,
    max_length: int = 2304,
) -> dict[str, Any]:
    """Validate a normalized adapter row and return its canonical representation.

    Required tokenization evidence is deliberately produced outside this module.
    The ``messages_sha256`` binding prevents this validator's normalization from
    silently invalidating those audited Qwen3.5 token counts.
    """
    if expected_skill not in SKILLS:
        raise Day20ContractError(f"unknown expected skill: {expected_skill!r}")
    if record.get("skill") != expected_skill:
        raise Day20ContractError(
            f"source skill mismatch: expected {expected_skill!r}, got {record.get('skill')!r}"
        )
    sample_id = _normalize_plain_text(record.get("sample_id"), "sample_id")
    target_format = record.get("target_format")
    if not isinstance(target_format, str):
        raise Day20ContractError(f"{sample_id}: target_format is required")

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) not in {2, 3}:
        raise Day20ContractError(
            f"{sample_id}: messages must be one user/assistant turn with optional system"
        )
    expected_roles = ["user", "assistant"] if len(messages) == 2 else ["system", "user", "assistant"]
    if [message.get("role") for message in messages if isinstance(message, dict)] != expected_roles:
        raise Day20ContractError(f"{sample_id}: invalid message role sequence")

    canonical_messages: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise Day20ContractError(f"{sample_id}: message {index} is not an object")
        role = expected_roles[index]
        if role == "assistant":
            content = normalize_assistant_target(
                expected_skill,
                target_format,
                message.get("content"),
                code_prefix=record.get("code_prefix"),
                finance_scale=record.get("finance_scale"),
            )
        else:
            content = _normalize_plain_text(message.get("content"), f"{role} content")
            if any(marker in content for marker in SPECIAL_TARGET_MARKERS):
                raise Day20ContractError(f"{sample_id}: message contains a template/media marker")
        canonical_messages.append({"role": role, "content": content})

    tokenization = record.get("qwen35_tokenization")
    if not isinstance(tokenization, Mapping):
        raise Day20ContractError(f"{sample_id}: qwen35_tokenization evidence is required")
    expected_tokenization = {
        "template": "qwen3_5",
        "enable_thinking": False,
        "add_non_thinking_prefix": True,
        "loss_scale": "default+ignore_empty_think",
        "truncated": False,
    }
    for key, expected in expected_tokenization.items():
        if tokenization.get(key) != expected:
            raise Day20ContractError(
                f"{sample_id}: qwen35_tokenization.{key} must be {expected!r}"
            )
    input_tokens = tokenization.get("input_tokens")
    supervised_tokens = tokenization.get("supervised_tokens")
    if not isinstance(input_tokens, int) or isinstance(input_tokens, bool) or input_tokens <= 0:
        raise Day20ContractError(f"{sample_id}: input_tokens must be a positive integer")
    if (
        not isinstance(supervised_tokens, int)
        or isinstance(supervised_tokens, bool)
        or supervised_tokens <= 0
        or supervised_tokens > input_tokens
    ):
        raise Day20ContractError(
            f"{sample_id}: supervised_tokens must be in [1, input_tokens]"
        )
    if input_tokens > max_length:
        raise Day20ContractError(f"{sample_id}: input would exceed max_length={max_length}")
    if tokenization.get("messages_sha256") != object_sha256(canonical_messages):
        raise Day20ContractError(
            f"{sample_id}: tokenization evidence is not bound to canonical messages"
        )
    render_sha256 = _require_sha256(tokenization.get("render_sha256"), "render_sha256")
    labels_sha256 = _require_sha256(tokenization.get("labels_sha256"), "labels_sha256")

    lineage = record.get("source_lineage")
    if not isinstance(lineage, Mapping):
        raise Day20ContractError(f"{sample_id}: source_lineage is required")
    for key in ("source", "revision", "split", "adapter", "license"):
        if not isinstance(lineage.get(key), str) or not lineage[key].strip():
            raise Day20ContractError(f"{sample_id}: source_lineage.{key} is required")
    _require_sha256(lineage.get("source_file_sha256"), "source_lineage.source_file_sha256")
    source_content_sha256 = _require_sha256(
        lineage.get("source_content_sha256"),
        "source_lineage.source_content_sha256",
    )
    reference_evidence = record.get("reference_evidence")
    if not isinstance(reference_evidence, Mapping):
        raise Day20ContractError(f"{sample_id}: reference_evidence is required")
    evidence_payload = dict(reference_evidence)
    evidence_sha256 = evidence_payload.pop("reference_evidence_sha256", None)
    checks = reference_evidence.get("checks")
    if (
        reference_evidence.get("schema_version") != 1
        or reference_evidence.get("verifier_version")
        != "day20_reference_evidence_v1"
        or reference_evidence.get("target_format") != target_format
        or reference_evidence.get("source_content_sha256")
        != source_content_sha256
        or evidence_sha256 != object_sha256(evidence_payload)
        or not isinstance(checks, Mapping)
    ):
        raise Day20ContractError(f"{sample_id}: reference evidence identity drifted")
    if target_format == "code_continuation":
        if (
            checks.get("source_rebuilt_match") is not True
            or checks.get("static_compile") is not True
        ):
            raise Day20ContractError(
                f"{sample_id}: code gold/recomposition verification failed"
            )
    elif checks.get("source_target_match") is not True:
        raise Day20ContractError(f"{sample_id}: source/target verification failed")

    prompt_messages = canonical_messages[:-1]
    prompt_sha256 = object_sha256(prompt_messages)
    content_sha256 = object_sha256(canonical_messages)
    result: dict[str, Any] = {
        "sample_id": sample_id,
        "skill": expected_skill,
        "target_format": target_format,
        "messages": canonical_messages,
        "qwen35_input_tokens": input_tokens,
        "qwen35_supervised_tokens": supervised_tokens,
        "qwen35_render_sha256": render_sha256,
        "qwen35_labels_sha256": labels_sha256,
        "prompt_sha256": prompt_sha256,
        "content_sha256": content_sha256,
        "source_content_sha256": source_content_sha256,
        "source_lineage": dict(lineage),
        "reference_evidence": dict(reference_evidence),
    }
    if target_format == "code_continuation":
        result["code_prefix"] = record["code_prefix"].replace("\r\n", "\n").replace("\r", "\n").rstrip()
    if target_format == "finance_value_scale":
        result["finance_scale"] = record["finance_scale"].strip().lower()
    for key in ("canonical_sample_id", "parent_id", "subskill"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            result[key] = value.strip()
    return result


def exact_subset_indices(weights: Sequence[int], target: int) -> set[int]:
    """Return one deterministic exact subset using bounded bitset DP."""
    if not isinstance(target, int) or isinstance(target, bool) or target < 0:
        raise Day20ContractError("subset target must be a non-negative integer")
    reachable = 1
    snapshots: list[int] = []
    mask = (1 << (target + 1)) - 1
    for weight in weights:
        if not isinstance(weight, int) or isinstance(weight, bool) or weight <= 0:
            raise Day20ContractError("subset weights must be positive integers")
        snapshots.append(reachable)
        reachable = (reachable | (reachable << weight)) & mask
    if not (reachable >> target) & 1:
        raise Day20ContractError(f"cannot construct exact supervised-token subset: {target}")

    chosen: set[int] = set()
    cursor = target
    for index in range(len(weights) - 1, -1, -1):
        if (snapshots[index] >> cursor) & 1:
            continue
        chosen.add(index)
        cursor -= weights[index]
    if cursor or sum(weights[index] for index in chosen) != target:
        raise Day20ContractError("exact subset reconstruction failed")
    return chosen


def _stable_identity(record: Mapping[str, Any], seed: str) -> tuple[str, str]:
    sample_id = record.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise Day20ContractError("stable selection requires a sample_id")
    return text_sha256(f"{seed}\0{sample_id}"), sample_id


def stable_hash_subset(
    records: Sequence[Mapping[str, Any]],
    count: int,
    *,
    seed: str,
) -> list[Mapping[str, Any]]:
    """Choose ``count`` records independently of their input ordering."""
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise Day20ContractError("stable subset count must be non-negative")
    sample_ids = [record.get("sample_id") for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise Day20ContractError("stable subset input has duplicate sample IDs")
    if count > len(records):
        raise Day20ContractError(
            f"stable subset needs {count} records but only {len(records)} are available"
        )
    return sorted(records, key=lambda record: _stable_identity(record, seed))[:count]


def validate_unique_training_records(records: Sequence[Mapping[str, Any]]) -> None:
    """Reject duplicate IDs, prompts, or complete message content."""
    for field in (
        "sample_id",
        "prompt_sha256",
        "content_sha256",
        "source_content_sha256",
    ):
        values = [record.get(field) for record in records]
        duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
        if duplicates:
            raise Day20ContractError(
                f"duplicate training {field}: {duplicates[:5]}"
            )


def eval_identity_sets(eval_records: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    sample_ids: set[str] = set()
    content_hashes: set[str] = set()
    prompt_hashes: set[str] = set()
    for record in eval_records:
        for value in (
            record.get("sample_id"),
            record.get("source_lineage", {}).get("parent_id")
            if isinstance(record.get("source_lineage"), Mapping)
            else None,
        ):
            if isinstance(value, str) and value:
                sample_ids.add(value)
        lineage = record.get("source_lineage")
        if isinstance(lineage, Mapping):
            value = lineage.get("candidate_content_hash")
            if isinstance(value, str) and value:
                content_hashes.add(value.removeprefix("sha256:"))
        for key in ("content_sha256", "content_hash", "reference_hash"):
            value = record.get(key)
            if isinstance(value, str) and value:
                content_hashes.add(value.removeprefix("sha256:"))
        messages = record.get("messages")
        if isinstance(messages, list) and messages:
            prompt_hashes.add(object_sha256(messages))
        for key in ("adapted_prompt", "raw_prompt"):
            value = record.get(key)
            if isinstance(value, str) and value:
                prompt_hashes.add(
                    object_sha256([{"role": "user", "content": value.strip()}])
                )
        for key in ("prompt_sha256", "adapted_prompt_hash", "raw_prompt_hash"):
            value = record.get(key)
            if isinstance(value, str) and value:
                prompt_hashes.add(value.removeprefix("sha256:"))
    return {
        "sample_ids": sample_ids,
        "content_hashes": content_hashes,
        "prompt_hashes": prompt_hashes,
    }


def assert_no_train_dev_leakage(
    train_records: Sequence[Mapping[str, Any]],
    eval_records: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    """Fail closed on sample-ID, prompt-hash, or content-hash overlap."""
    identities = eval_identity_sets(eval_records)
    train_ids: set[str] = set()
    for record in train_records:
        for key in ("sample_id", "canonical_sample_id", "parent_id"):
            value = record.get(key)
            if isinstance(value, str) and value:
                train_ids.add(value)
        lineage = record.get("source_lineage")
        if isinstance(lineage, Mapping):
            for key in ("source_id", "parent_id"):
                value = lineage.get(key)
                if isinstance(value, str) and value:
                    train_ids.add(value)
    train_content = {
        str(record[field])
        for record in train_records
        for field in ("content_sha256", "source_content_sha256")
        if record.get(field)
    }
    train_prompts = {
        str(record["prompt_sha256"])
        for record in train_records
        if record.get("prompt_sha256")
    }
    report = {
        "sample_id_matches": sorted(train_ids & identities["sample_ids"]),
        "content_hash_matches": sorted(train_content & identities["content_hashes"]),
        "prompt_hash_matches": sorted(train_prompts & identities["prompt_hashes"]),
    }
    if any(report.values()):
        raise Day20ContractError(f"train/dev leakage detected: {report}")
    return report


def validate_trainable_inventory(
    trainable_parameter_names: Iterable[str],
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate that every trainable parameter is one paired language LoRA tensor."""
    names = sorted(trainable_parameter_names)
    if not names:
        raise Day20ContractError("trainable inventory is empty")
    if len(names) != len(set(names)):
        raise Day20ContractError("trainable inventory contains duplicate names")
    module_sides: dict[str, set[str]] = defaultdict(set)
    leaf_counts: Counter[str] = Counter()
    target_pattern = re.compile(LORA_TARGET_REGEX)
    for name in names:
        match = _LORA_PARAMETER_RE.fullmatch(name)
        if not match:
            raise Day20ContractError(f"non-LoRA trainable parameter: {name}")
        module = match.group("module")
        if not target_pattern.fullmatch(module):
            raise Day20ContractError(f"LoRA parameter is outside language target scope: {name}")
        module_sides[module].add(match.group("side"))
        leaf_counts[module.rsplit(".", 1)[-1]] += 1
    unpaired = sorted(module for module, sides in module_sides.items() if sides != {"A", "B"})
    if unpaired:
        raise Day20ContractError(f"unpaired LoRA modules: {unpaired[:5]}")
    inventory_sha256 = object_sha256(names)
    if expected_sha256 is not None and inventory_sha256 != expected_sha256:
        raise Day20ContractError(
            f"trainable inventory hash drift: expected {expected_sha256}, got {inventory_sha256}"
        )
    return {
        "trainable_parameter_names": names,
        "trainable_parameter_count": len(names),
        "target_module_count": len(module_sides),
        "leaf_parameter_counts": dict(sorted(leaf_counts.items())),
        "trainable_names_sha256": inventory_sha256,
    }


def immutable_experiment_contract(model_snapshot: str) -> dict[str, Any]:
    model_snapshot = _normalize_plain_text(model_snapshot, "model_snapshot")
    return {
        "model": {
            "snapshot": model_snapshot,
            "model_type": "qwen3_5",
            "parent": "untouched_base",
        },
        "template": dict(QWEN35_TEMPLATE_CONTRACT),
        "data": {
            "skills": list(SKILLS),
            "main_supervised_tokens": MAIN_SUPERVISED_TOKENS,
            "main_tokens_per_skill": MAIN_TOKENS_PER_SKILL,
            "main_tokens_by_format": dict(MAIN_TOKENS_BY_FORMAT),
            "probe_supervised_tokens": PROBE_SUPERVISED_TOKENS,
            "probe_tokens_per_skill": PROBE_TOKENS_PER_SKILL,
            "probe_tokens_by_format": dict(PROBE_TOKENS_BY_FORMAT),
            "diagnostic_records_per_skill": DIAGNOSTIC_RECORDS_PER_SKILL,
            "minimum_probe_records_by_format": dict(MIN_PROBE_RECORDS_BY_FORMAT),
            "minimum_main_records_by_format": dict(MIN_MAIN_RECORDS_BY_FORMAT),
        },
        "lora": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
            "bias": "none",
            "target_regex": LORA_TARGET_REGEX,
            "target_leaves": list(LORA_TARGET_LEAVES),
            "freeze_vit": True,
            "freeze_aligner": True,
            "freeze_embeddings": True,
            "freeze_lm_head": True,
        },
        "optimizer": {
            "name": "adamw",
            "betas": [0.9, 0.95],
            "epsilon": 1e-8,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "warmup_fraction": 0.05,
            "lr_scheduler": "cosine",
        },
        "batch": {
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "global_batch_size": 8,
        },
        "runtime": {
            "torch_dtype": "bfloat16",
            "backend": "hf_transformers",
            "linear_attention": "model_native_qwen3_5_gdn",
            "packing": False,
            "padding_free": False,
            "seed": 20260809,
            "data_seed": 20260809,
        },
        "probe_learning_rates": list(PROBE_LEARNING_RATES),
        "main_checkpoint_supervised_tokens": list(MAIN_CHECKPOINT_TOKENS),
    }


def _training_config(
    *,
    run_kind: str,
    model_snapshot: str,
    dataset_path: str,
    dataset_sha256: str,
    dataset_supervised_tokens: int,
    learning_rate: float | str,
) -> dict[str, Any]:
    if run_kind not in {"probe", "main"}:
        raise Day20ContractError(f"invalid run kind: {run_kind!r}")
    _require_sha256(dataset_sha256, "dataset_sha256")
    expected_tokens = (
        PROBE_SUPERVISED_TOKENS if run_kind == "probe" else MAIN_SUPERVISED_TOKENS
    )
    if dataset_supervised_tokens != expected_tokens:
        raise Day20ContractError(
            f"{run_kind} dataset must have exactly {expected_tokens} supervised tokens"
        )
    if isinstance(learning_rate, float):
        if learning_rate not in PROBE_LEARNING_RATES:
            raise Day20ContractError("learning rate is outside the pre-registered probes")
    elif learning_rate != "__SELECT_FROM_PASSING_PROBE__" or run_kind != "main":
        raise Day20ContractError("invalid unresolved learning-rate placeholder")

    immutable = immutable_experiment_contract(model_snapshot)
    config: dict[str, Any] = {
        "schema_version": 1,
        "domain": "day20.qwen35_balanced_lora.training_config",
        "run_kind": run_kind,
        "model": immutable["model"],
        "data": {
            "path": _normalize_plain_text(dataset_path, "dataset_path"),
            "file_sha256": dataset_sha256,
            "supervised_tokens": dataset_supervised_tokens,
            "interleave": "round_robin_skill_v1",
        },
        "template": immutable["template"],
        "training": {
            "tuner_type": "lora",
            **immutable["lora"],
            **immutable["optimizer"],
            **immutable["batch"],
            **immutable["runtime"],
            "learning_rate": learning_rate,
            "safety_gate_steps": 5,
            "checkpoint_supervised_tokens": (
                list(MAIN_CHECKPOINT_TOKENS) if run_kind == "main" else []
            ),
            "checkpoint_policy": (
                "early_mid_final_resumable" if run_kind == "main" else "adapter_and_metrics_only"
            ),
        },
    }
    config["immutable_sha256"] = object_sha256(config)
    return config


def build_probe_config(
    *,
    model_snapshot: str,
    dataset_path: str,
    dataset_sha256: str,
    learning_rate: float,
) -> dict[str, Any]:
    return _training_config(
        run_kind="probe",
        model_snapshot=model_snapshot,
        dataset_path=dataset_path,
        dataset_sha256=dataset_sha256,
        dataset_supervised_tokens=PROBE_SUPERVISED_TOKENS,
        learning_rate=learning_rate,
    )


def build_main_config(
    *,
    model_snapshot: str,
    dataset_path: str,
    dataset_sha256: str,
    learning_rate: float | None = None,
) -> dict[str, Any]:
    return _training_config(
        run_kind="main",
        model_snapshot=model_snapshot,
        dataset_path=dataset_path,
        dataset_sha256=dataset_sha256,
        dataset_supervised_tokens=MAIN_SUPERVISED_TOKENS,
        learning_rate=(
            "__SELECT_FROM_PASSING_PROBE__" if learning_rate is None else learning_rate
        ),
    )


def verify_immutable_hash(value: Mapping[str, Any]) -> str:
    expected = value.get("immutable_sha256")
    if not isinstance(expected, str):
        raise Day20ContractError("immutable_sha256 is missing")
    payload = dict(value)
    payload.pop("immutable_sha256", None)
    actual = object_sha256(payload)
    if actual != expected:
        raise Day20ContractError(
            f"immutable config hash drift: expected {expected}, got {actual}"
        )
    return actual
