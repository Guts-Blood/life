#!/usr/bin/env python3
"""Qwen3.5 RLHF-only template preserving the coding response boundary.

Day 20 remains an immutable, training-only historical implementation.  This
module reuses its frozen boundary helpers but registers a separate alias whose
only allowed mode is ``rlhf`` so pinned ms-swift can encode both DPO branches.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


DAY23_DIR = Path(__file__).resolve().parent
DAY20_DIR = DAY23_DIR.parent / "day-20-qwen35-balanced-lora-sft"
if str(DAY20_DIR) not in sys.path:
    sys.path.insert(0, str(DAY20_DIR))

import day20_target_encoding_v3 as day20  # noqa: E402


TARGET_TEMPLATE_ALIAS = "day23_qwen3_5_dpo_target_v1"
TARGET_TEMPLATE_REVISION = "day23.qwen35_dpo_code_boundary_rlhf_v1"
SOURCE_TEMPLATE = "qwen3_5"
EXPECTED_DAY20_CONTRACT_SHA256 = (
    "7f835342861cc9aabfd6299228033df80437b183227e04db311450437cd07ded"
)
LOSS_SCALE = day20.LOSS_SCALE
DEFAULT_MAX_LENGTH = 512


class Day23RLHFTemplateError(ValueError):
    """The Day 23 template, response mask, or pinned identity drifted."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


TARGET_ENCODING_CONTRACT: dict[str, Any] = {
    "schema_name": "day23.qwen35_dpo_target_encoding_contract",
    "schema_version": 1,
    "contract_version": TARGET_TEMPLATE_REVISION,
    "template_alias": TARGET_TEMPLATE_ALIAS,
    "source_template": SOURCE_TEMPLATE,
    "mode": "rlhf_only",
    "branches": ["chosen", "rejected"],
    "source_day20_contract_sha256": EXPECTED_DAY20_CONTRACT_SHA256,
    "code_detection": day20.TARGET_ENCODING_CONTRACT["code_detection"],
    "representation": day20.TARGET_ENCODING_CONTRACT["representation"],
    "balanced_loss_mask_exchange": day20.TARGET_ENCODING_CONTRACT[
        "balanced_loss_mask_exchange"
    ],
    "pinned_tail_token_ids": [day20.IM_END_TOKEN_ID, day20.NEWLINE_TOKEN_ID],
    "loss_scale": LOSS_SCALE,
    "max_length": DEFAULT_MAX_LENGTH,
    "truncation_strategy": "raise",
}
TARGET_ENCODING_CONTRACT_SHA256 = object_sha256(TARGET_ENCODING_CONTRACT)


def _raise(error: Exception) -> Day23RLHFTemplateError:
    return Day23RLHFTemplateError(f"Day 20 boundary helper rejected input: {error}")


def _code_target(messages: Sequence[Mapping[str, Any]]) -> str | None:
    try:
        return day20._code_target(messages)
    except day20.Day20TargetEncodingV3Error as error:
        raise _raise(error) from error


def _messages(value: Any) -> list[Mapping[str, Any]]:
    try:
        return day20._messages(value)
    except day20.Day20TargetEncodingV3Error as error:
        raise _raise(error) from error


def apply_balanced_code_mask(
    encoded: Mapping[str, Any], *, target: str, template: Any
) -> dict[str, Any]:
    """Apply the frozen Day 20 balanced boundary exchange to one DPO branch."""
    try:
        input_ids = day20._integral_array(encoded, "input_ids")
        labels = day20._integral_array(encoded, "labels")
    except day20.Day20TargetEncodingV3Error as error:
        raise _raise(error) from error
    if not input_ids or len(input_ids) != len(labels):
        raise Day23RLHFTemplateError("Code token arrays are empty or misaligned")
    before = day20._supervised_tokens(labels)
    first_supervised = next(
        (index for index, label in enumerate(labels) if label != -100), None
    )
    if first_supervised is None or first_supervised <= 0:
        raise Day23RLHFTemplateError("Code labels have no supervised response boundary")
    candidates = [
        index
        for index in range(1, len(labels) - 1)
        if input_ids[index] == day20.INDENT_TOKEN_ID
        and labels[index] == -100
        and labels[index + 1] != -100
    ]
    indent_index = first_supervised - 1
    if candidates != [indent_index]:
        raise Day23RLHFTemplateError(
            "Code boundary must contain one masked four-space token"
        )
    if input_ids[-2:] != [day20.IM_END_TOKEN_ID, day20.NEWLINE_TOKEN_ID] or labels[
        -2:
    ] != [day20.IM_END_TOKEN_ID, day20.NEWLINE_TOKEN_ID]:
        raise Day23RLHFTemplateError(
            "Code target must end with supervised <|im_end|> and newline"
        )

    labels[indent_index] = day20.INDENT_TOKEN_ID
    labels[-1] = -100
    if day20._supervised_tokens(labels) != before:
        raise Day23RLHFTemplateError(
            "balanced Code mask exchange changed supervised-token count"
        )
    try:
        decoded = day20._decode_supervised(template, labels)
    except day20.Day20TargetEncodingV3Error as error:
        raise _raise(error) from error
    if decoded != target:
        raise Day23RLHFTemplateError(
            "Code supervised labels do not decode to the canonical target: "
            f"decoded={decoded!r}, expected={target!r}"
        )
    result = dict(encoded)
    result["input_ids"] = input_ids
    result["labels"] = labels
    return result


def _assert_template_contract(template: Any, *, max_length: int) -> None:
    meta = getattr(template, "template_meta", None)
    actual = {
        "template_type": getattr(meta, "template_type", None),
        "suffix": getattr(meta, "suffix", None),
        "mode": getattr(template, "mode", None),
        "template_backend": getattr(template, "template_backend", None),
        "enable_thinking": getattr(template, "enable_thinking", None),
        "add_non_thinking_prefix": getattr(template, "add_non_thinking_prefix", None),
        "padding_free": getattr(template, "padding_free", None),
        "packing": getattr(template, "packing", None),
        "loss_scale": getattr(template, "_loss_scale", None),
        "max_length": getattr(template, "max_length", None),
    }
    expected = {
        "template_type": TARGET_TEMPLATE_ALIAS,
        "suffix": ["<|im_end|>\n"],
        "mode": "rlhf",
        "template_backend": "swift",
        "enable_thinking": False,
        "add_non_thinking_prefix": True,
        "padding_free": False,
        "packing": False,
        "loss_scale": LOSS_SCALE,
        "max_length": max_length,
    }
    if actual != expected:
        raise Day23RLHFTemplateError(
            "DPO template drifted: "
            + json.dumps({"actual": actual, "expected": expected}, sort_keys=True)
        )
    if list(template._tokenize(day20.INDENT_TEXT)) != [day20.INDENT_TOKEN_ID]:
        raise Day23RLHFTemplateError("four-space token identity drifted")
    if list(template._tokenize("\n")) != [day20.NEWLINE_TOKEN_ID]:
        raise Day23RLHFTemplateError("newline token identity drifted")
    suffix_ids = list(template._encode_context_list(meta.suffix)[0])
    if suffix_ids != [day20.IM_END_TOKEN_ID, day20.NEWLINE_TOKEN_ID]:
        raise Day23RLHFTemplateError("ChatML suffix token identity drifted")


def register_dpo_template() -> None:
    """Register the isolated RLHF-only alias idempotently and fail closed."""
    if day20.TARGET_ENCODING_CONTRACT_SHA256 != EXPECTED_DAY20_CONTRACT_SHA256:
        raise Day23RLHFTemplateError("frozen Day 20 target contract drifted")
    try:
        from swift.template import TEMPLATE_MAPPING, register_template
        from swift.template.templates.qwen import Qwen3_5Template
    except ImportError as error:
        raise Day23RLHFTemplateError("pinned ms-swift is required") from error

    existing = TEMPLATE_MAPPING.get(TARGET_TEMPLATE_ALIAS)
    if existing is not None:
        cls = getattr(existing, "template_cls", None)
        if (
            getattr(existing, "template_type", None) != TARGET_TEMPLATE_ALIAS
            or getattr(cls, "day23_target_encoding_contract_sha256", None)
            != TARGET_ENCODING_CONTRACT_SHA256
            or not isinstance(cls, type)
            or not issubclass(cls, Qwen3_5Template)
        ):
            raise Day23RLHFTemplateError("DPO template alias collision")
        return

    source_meta = TEMPLATE_MAPPING.get(SOURCE_TEMPLATE)
    if source_meta is None or getattr(source_meta, "template_cls", None) is not Qwen3_5Template:
        raise Day23RLHFTemplateError("native qwen3_5 template identity drifted")

    class Day23Qwen35DPOTargetV1(Qwen3_5Template):
        day23_target_encoding_contract_sha256 = TARGET_ENCODING_CONTRACT_SHA256

        def _swift_prepare_inputs(self, inputs: Any) -> None:
            messages = _messages(inputs.messages)
            target = _code_target(messages)
            super()._swift_prepare_inputs(inputs)
            if target is None:
                return
            prepared = _messages(inputs.messages)
            if [message.get("role") for message in prepared] != ["user", "assistant"]:
                raise Day23RLHFTemplateError(
                    "native preprocessing changed the Code role sequence"
                )
            prepared[1]["content"] = target

        def _encode(self, inputs: Any) -> dict[str, Any]:
            messages = _messages(inputs.messages)
            target = _code_target(messages)
            if getattr(self, "mode", None) != "rlhf":
                raise Day23RLHFTemplateError("Day 23 DPO template is RLHF-only")
            encoded = super()._encode(inputs)
            if target is None:
                return encoded
            return apply_balanced_code_mask(encoded, target=target, template=self)

    Day23Qwen35DPOTargetV1.__name__ = "Day23Qwen35DPOTargetV1"
    Day23Qwen35DPOTargetV1.__qualname__ = "Day23Qwen35DPOTargetV1"
    Day23Qwen35DPOTargetV1.__module__ = __name__
    target_meta = copy.deepcopy(source_meta)
    target_meta.template_type = TARGET_TEMPLATE_ALIAS
    target_meta.template_cls = Day23Qwen35DPOTargetV1
    register_template(target_meta)


def build_dpo_template(model_path: Path, *, max_length: int = DEFAULT_MAX_LENGTH) -> Any:
    """Build the processor/template in RLHF mode without loading model weights."""
    resolved = model_path.expanduser().resolve()
    if not resolved.is_dir():
        raise Day23RLHFTemplateError(f"model snapshot is missing: {resolved}")
    if max_length <= 0:
        raise Day23RLHFTemplateError("max_length must be positive")
    register_dpo_template()
    try:
        from swift import get_model_processor, get_template
    except ImportError as error:
        raise Day23RLHFTemplateError("pinned ms-swift is required") from error
    model, processor = get_model_processor(
        str(resolved),
        model_type="qwen3_5",
        load_model=False,
        use_hf=True,
        download_model=False,
    )
    if model is not None:
        raise Day23RLHFTemplateError("load_model=False returned model weights")
    template = get_template(
        processor,
        template_type=TARGET_TEMPLATE_ALIAS,
        max_length=max_length,
        truncation_strategy="raise",
        padding_free=False,
        loss_scale=LOSS_SCALE,
        enable_thinking=False,
        add_non_thinking_prefix=True,
    )
    template.set_mode("rlhf")
    _assert_template_contract(template, max_length=max_length)
    return template


def encode_dpo_row(template: Any, row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        encoded = template.encode(copy.deepcopy(dict(row)), return_length=True)
    except Exception as error:
        if isinstance(error, Day23RLHFTemplateError):
            raise
        raise Day23RLHFTemplateError(
            f"DPO row encode failed: {type(error).__name__}: {error}"
        ) from error
    if not isinstance(encoded, Mapping):
        raise Day23RLHFTemplateError("DPO template returned a non-object")
    required = {
        "chosen_input_ids",
        "chosen_labels",
        "rejected_input_ids",
        "rejected_labels",
    }
    missing = required - set(encoded)
    if missing:
        raise Day23RLHFTemplateError(f"DPO template omitted branches: {sorted(missing)}")
    return dict(encoded)


def decode_supervised(template: Any, token_ids: Sequence[int]) -> str:
    try:
        return template.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return template.tokenizer.decode(list(token_ids), skip_special_tokens=True)

