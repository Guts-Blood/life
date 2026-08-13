#!/usr/bin/env python3
"""Day 20 v3 training-only Code boundary target encoding.

The native Qwen3.5 template trims assistant text and the configured
``ignore_empty_think`` loss scale masks whitespace immediately after the empty
thinking block.  For strict Python continuations that combination removes the
required four-space first token from the supervised labels.  This module
registers an isolated template alias which restores that token and exchanges
its loss mask with the semantically redundant newline after ``<|im_end|>``.

Inference must continue to use the native ``qwen3_5`` template.  The module is
lazy-imported so its pure identities and CLI help remain inspectable on hosts
without ms-swift installed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


TARGET_TEMPLATE_ALIAS = "day20_qwen3_5_target_v3"
TARGET_ENCODING_CONTRACT_VERSION = "day20.qwen35_code_boundary_target_v3"
SOURCE_TEMPLATE = "qwen3_5"
CODE_PROMPT_PREFIX = (
    "Complete the Python function below. Return only the indented Python "
    "continuation:"
)
INDENT_TEXT = "    "
INDENT_TOKEN_ID = 257
IM_END_TOKEN_ID = 248046
NEWLINE_TOKEN_ID = 198
LOSS_SCALE = "default+ignore_empty_think"
AUDIT_DOMAIN = "day20.target_encoding_audit.v3"


class Day20TargetEncodingV3Error(ValueError):
    """A target representation, loss mask, or identity invariant failed."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def token_sequence_sha256(values: Sequence[int]) -> str:
    if isinstance(values, (str, bytes, bytearray)):
        raise Day20TargetEncodingV3Error("token sequence must be an integer sequence")
    try:
        normalized = [int(value) for value in values]
    except (TypeError, ValueError) as error:
        raise Day20TargetEncodingV3Error("token sequence is not integral") from error
    return object_sha256(normalized)


TARGET_ENCODING_CONTRACT: dict[str, Any] = {
    "schema_name": "day20.target_encoding_contract",
    "schema_version": 3,
    "contract_version": TARGET_ENCODING_CONTRACT_VERSION,
    "template_alias": TARGET_TEMPLATE_ALIAS,
    "source_template": SOURCE_TEMPLATE,
    "mode": "train_only",
    "code_detection": {
        "prompt_prefix": CODE_PROMPT_PREFIX,
        "assistant_first_bytes": INDENT_TEXT,
        "assistant_first_line_indent": "exactly_four_ascii_spaces",
    },
    "representation": {
        "preserve_code_assistant_leading_whitespace": True,
        "non_code_native_identity_required": True,
    },
    "balanced_loss_mask_exchange": {
        "supervise": {
            "text": INDENT_TEXT,
            "token_ids": [INDENT_TOKEN_ID],
        },
        "mask": {
            "semantic_role": "newline_after_chatml_im_end",
            "token_ids": [NEWLINE_TOKEN_ID],
        },
        "preserve_supervised_token_count_per_row": True,
        "preserve_im_end_supervision": True,
    },
    "pinned_tail_token_ids": [IM_END_TOKEN_ID, NEWLINE_TOKEN_ID],
    "loss_scale": LOSS_SCALE,
}
TARGET_ENCODING_CONTRACT_SHA256 = object_sha256(TARGET_ENCODING_CONTRACT)


def _messages(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not value:
        raise Day20TargetEncodingV3Error("messages must be a non-empty list")
    if not all(isinstance(message, Mapping) for message in value):
        raise Day20TargetEncodingV3Error("every message must be an object")
    return value


def _code_target(messages: Sequence[Mapping[str, Any]]) -> str | None:
    """Return the exact Code target, failing closed on a partial match."""
    roles = [message.get("role") for message in messages]
    code_prompt_indexes = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and str(message["content"]).startswith(CODE_PROMPT_PREFIX)
    ]
    if not code_prompt_indexes:
        return None
    if roles != ["user", "assistant"] or code_prompt_indexes != [0]:
        raise Day20TargetEncodingV3Error(
            "Code target encoding requires exactly one user/assistant round"
        )
    target = messages[1].get("content")
    if not isinstance(target, str) or not target:
        raise Day20TargetEncodingV3Error("Code assistant target must be non-empty text")
    if not target.startswith(INDENT_TEXT):
        raise Day20TargetEncodingV3Error(
            "Code assistant target must begin at byte 0 with four ASCII spaces"
        )
    first_line = target.split("\n", 1)[0]
    leading = first_line[: len(first_line) - len(first_line.lstrip(" \t"))]
    if leading != INDENT_TEXT or "\t" in leading:
        raise Day20TargetEncodingV3Error(
            "Code assistant first line must have exactly four-space indentation"
        )
    return target


def _integral_array(encoded: Mapping[str, Any], key: str) -> list[int]:
    value = encoded.get(key)
    if not isinstance(value, (list, tuple)):
        raise Day20TargetEncodingV3Error(f"encoded {key} must be a token array")
    try:
        result = [int(item) for item in value]
    except (TypeError, ValueError) as error:
        raise Day20TargetEncodingV3Error(f"encoded {key} is not integral") from error
    return result


def _supervised_tokens(labels: Sequence[int]) -> int:
    # Causal-LM loss shifts labels left; position zero is never predicted.
    return sum(int(label) != -100 for label in labels[1:])


def _decode_supervised(template: Any, labels: Sequence[int]) -> str:
    """Decode semantic target bytes while discarding template special tokens."""
    supervised = [int(label) for label in labels if int(label) != -100]
    try:
        return template.tokenizer.decode(
            supervised,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return template.tokenizer.decode(supervised, skip_special_tokens=True)


def _assert_template_contract(template: Any) -> None:
    meta = getattr(template, "template_meta", None)
    drift = {
        "template_type": getattr(meta, "template_type", None),
        "suffix": getattr(meta, "suffix", None),
        "mode": getattr(template, "mode", None),
        "template_backend": getattr(template, "template_backend", None),
        "enable_thinking": getattr(template, "enable_thinking", None),
        "add_non_thinking_prefix": getattr(template, "add_non_thinking_prefix", None),
        "padding_free": getattr(template, "padding_free", None),
        "packing": getattr(template, "packing", None),
        "loss_scale": getattr(template, "_loss_scale", None),
    }
    expected = {
        "template_type": TARGET_TEMPLATE_ALIAS,
        "suffix": ["<|im_end|>\n"],
        "mode": "train",
        "template_backend": "swift",
        "enable_thinking": False,
        "add_non_thinking_prefix": True,
        "padding_free": False,
        "packing": False,
        "loss_scale": LOSS_SCALE,
    }
    if drift != expected:
        raise Day20TargetEncodingV3Error(
            "target template drifted: "
            + json.dumps({"actual": drift, "expected": expected}, sort_keys=True)
        )
    if list(template._tokenize(INDENT_TEXT)) != [INDENT_TOKEN_ID]:
        raise Day20TargetEncodingV3Error("four-space token identity drifted")
    if list(template._tokenize("\n")) != [NEWLINE_TOKEN_ID]:
        raise Day20TargetEncodingV3Error("newline token identity drifted")
    suffix_ids = list(template._encode_context_list(meta.suffix)[0])
    if suffix_ids != [IM_END_TOKEN_ID, NEWLINE_TOKEN_ID]:
        raise Day20TargetEncodingV3Error("ChatML suffix token identity drifted")


def register_target_template_v3() -> None:
    """Register the isolated training template alias, idempotently and strictly."""
    try:
        from swift.template import TEMPLATE_MAPPING, register_template
        from swift.template.templates.qwen import Qwen3_5Template
    except ImportError as error:
        raise Day20TargetEncodingV3Error("pinned ms-swift is required") from error

    existing = TEMPLATE_MAPPING.get(TARGET_TEMPLATE_ALIAS)
    if existing is not None:
        cls = getattr(existing, "template_cls", None)
        if (
            getattr(existing, "template_type", None) != TARGET_TEMPLATE_ALIAS
            or getattr(cls, "day20_target_encoding_contract_sha256", None)
            != TARGET_ENCODING_CONTRACT_SHA256
            or not isinstance(cls, type)
            or not issubclass(cls, Qwen3_5Template)
        ):
            raise Day20TargetEncodingV3Error("target template alias collision")
        return

    source_meta = TEMPLATE_MAPPING.get(SOURCE_TEMPLATE)
    if source_meta is None or getattr(source_meta, "template_cls", None) is not Qwen3_5Template:
        raise Day20TargetEncodingV3Error("native qwen3_5 template identity drifted")

    class Day20Qwen35TargetEncodingV3(Qwen3_5Template):
        day20_target_encoding_contract_sha256 = TARGET_ENCODING_CONTRACT_SHA256

        def _swift_prepare_inputs(self, inputs: Any) -> None:
            messages = _messages(inputs.messages)
            target = _code_target(messages)
            super()._swift_prepare_inputs(inputs)
            if target is None:
                return
            prepared = _messages(inputs.messages)
            if [message.get("role") for message in prepared] != ["user", "assistant"]:
                raise Day20TargetEncodingV3Error(
                    "native preprocessing changed the Code role sequence"
                )
            # Restore the exact canonical target after native Qwen3.5 trimming.
            prepared[1]["content"] = target

        def _encode(self, inputs: Any) -> dict[str, Any]:
            messages = _messages(inputs.messages)
            target = _code_target(messages)
            if getattr(self, "mode", None) != "train":
                raise Day20TargetEncodingV3Error(
                    "day20 target template alias is training-only"
                )
            encoded = super()._encode(inputs)
            if target is None:
                return encoded

            input_ids = _integral_array(encoded, "input_ids")
            labels = _integral_array(encoded, "labels")
            if not input_ids or len(input_ids) != len(labels):
                raise Day20TargetEncodingV3Error("Code token arrays are empty or misaligned")
            before = _supervised_tokens(labels)
            first_supervised = next(
                (index for index, label in enumerate(labels) if label != -100), None
            )
            if first_supervised is None or first_supervised <= 0:
                raise Day20TargetEncodingV3Error("Code labels have no supervised boundary")
            candidates = [
                index
                for index in range(1, len(labels) - 1)
                if input_ids[index] == INDENT_TOKEN_ID
                and labels[index] == -100
                and labels[index + 1] != -100
            ]
            indent_index = first_supervised - 1
            if candidates != [indent_index]:
                raise Day20TargetEncodingV3Error(
                    "Code boundary must contain one masked four-space token"
                )
            if input_ids[-2:] != [IM_END_TOKEN_ID, NEWLINE_TOKEN_ID] or labels[-2:] != [
                IM_END_TOKEN_ID,
                NEWLINE_TOKEN_ID,
            ]:
                raise Day20TargetEncodingV3Error(
                    "Code target must end with supervised <|im_end|> and newline"
                )

            labels[indent_index] = INDENT_TOKEN_ID
            labels[-1] = -100
            if _supervised_tokens(labels) != before:
                raise Day20TargetEncodingV3Error(
                    "balanced Code mask exchange changed supervised-token count"
                )
            decoded_target = _decode_supervised(self, labels)
            if decoded_target != target:
                raise Day20TargetEncodingV3Error(
                    "Code supervised labels do not decode to the canonical target: "
                    f"decoded={decoded_target!r}, expected={target!r}"
                )
            encoded["labels"] = labels
            return encoded

    Day20Qwen35TargetEncodingV3.__name__ = "Day20Qwen35TargetEncodingV3"
    Day20Qwen35TargetEncodingV3.__qualname__ = "Day20Qwen35TargetEncodingV3"
    Day20Qwen35TargetEncodingV3.__module__ = __name__

    target_meta = copy.deepcopy(source_meta)
    target_meta.template_type = TARGET_TEMPLATE_ALIAS
    target_meta.template_cls = Day20Qwen35TargetEncodingV3
    register_template(target_meta)


def build_target_template_v3(model_path: Path, *, max_length: int = 2304) -> Any:
    """Build the pinned v3 training template without loading model weights."""
    resolved = model_path.expanduser().resolve()
    if not resolved.is_dir():
        raise Day20TargetEncodingV3Error(f"model directory is missing: {resolved}")
    register_target_template_v3()
    try:
        from swift import get_model_processor, get_template
    except ImportError as error:
        raise Day20TargetEncodingV3Error("pinned ms-swift is required") from error
    model, processor = get_model_processor(
        str(resolved),
        model_type="qwen3_5",
        load_model=False,
        use_hf=True,
        download_model=False,
    )
    if model is not None:
        raise Day20TargetEncodingV3Error("load_model=False returned model weights")
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
    template.set_mode("train")
    _assert_template_contract(template)
    return template


def _encode(template: Any, messages: Sequence[Mapping[str, Any]]) -> tuple[list[int], list[int]]:
    try:
        encoded = template.encode(
            {"messages": copy.deepcopy(list(messages))}, return_length=True
        )
    except Exception as error:
        if isinstance(error, Day20TargetEncodingV3Error):
            raise
        raise Day20TargetEncodingV3Error(
            f"template encode failed: {type(error).__name__}: {error}"
        ) from error
    if not isinstance(encoded, Mapping):
        raise Day20TargetEncodingV3Error("template returned a non-object encoding")
    input_ids = _integral_array(encoded, "input_ids")
    labels = _integral_array(encoded, "labels")
    if not input_ids or len(input_ids) != len(labels):
        raise Day20TargetEncodingV3Error("template token arrays are empty or misaligned")
    return input_ids, labels


def encode_target_row_v3(
    row: Mapping[str, Any], *, native_template: Any, target_template: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Encode one row without mutation and return its v3 fields plus audit evidence."""
    messages = _messages(row.get("messages"))
    sample_id = row.get("sample_id")
    skill = row.get("skill")
    target_format = row.get("target_format")
    if not isinstance(sample_id, str) or not sample_id:
        raise Day20TargetEncodingV3Error("row sample_id is missing")
    is_code = target_format == "code_continuation"
    if is_code != (skill == "code"):
        raise Day20TargetEncodingV3Error(f"{sample_id}: Code skill/format drifted")
    prompt_detected = _code_target(messages) is not None
    if prompt_detected != is_code:
        raise Day20TargetEncodingV3Error(
            f"{sample_id}: Code prompt detection disagrees with target format"
        )

    native_ids, native_labels = _encode(native_template, messages)
    target_ids, target_labels = _encode(target_template, messages)
    native_count = _supervised_tokens(native_labels)
    target_count = _supervised_tokens(target_labels)
    native_pair_sha = object_sha256(
        {"input_ids": native_ids, "labels": native_labels}
    )
    target_pair_sha = object_sha256(
        {"input_ids": target_ids, "labels": target_labels}
    )
    first_target = next(
        (index for index, label in enumerate(target_labels) if label != -100), None
    )
    exact_four = bool(
        is_code
        and first_target is not None
        and target_ids[first_target] == INDENT_TOKEN_ID
        and target_labels[first_target] == INDENT_TOKEN_ID
    )
    arrays_identical = native_ids == target_ids and native_labels == target_labels
    if is_code:
        if target_count != native_count:
            raise Day20TargetEncodingV3Error(
                f"{sample_id}: Code supervised-token count changed"
            )
        if len(target_ids) != len(native_ids) + 1 or not exact_four:
            raise Day20TargetEncodingV3Error(
                f"{sample_id}: Code boundary encoding is not the balanced v3 form"
            )
    elif not arrays_identical:
        raise Day20TargetEncodingV3Error(
            f"{sample_id}: non-Code input_ids/labels changed from native"
        )

    updated = copy.deepcopy(dict(row))
    updated.update(
        {
            "qwen35_input_tokens": len(target_ids),
            "qwen35_supervised_tokens": target_count,
            "qwen35_render_sha256": token_sequence_sha256(target_ids),
            "qwen35_labels_sha256": token_sequence_sha256(target_labels),
        }
    )
    entry: dict[str, Any] = {
        "sample_id": sample_id,
        "skill": skill,
        "target_format": target_format,
        "native_input_tokens": len(native_ids),
        "target_input_tokens": len(target_ids),
        "native_supervised_tokens": native_count,
        "target_supervised_tokens": target_count,
        "native_input_ids_sha256": token_sequence_sha256(native_ids),
        "native_labels_sha256": token_sequence_sha256(native_labels),
        "target_input_ids_sha256": token_sequence_sha256(target_ids),
        "target_labels_sha256": token_sequence_sha256(target_labels),
        "native_pair_sha256": native_pair_sha,
        "target_pair_sha256": target_pair_sha,
        "code_exact_four_space_first_label": exact_four if is_code else None,
        "per_row_supervised_count_preserved": target_count == native_count,
        "non_code_input_ids_labels_identical": arrays_identical if not is_code else None,
    }
    entry["entry_sha256"] = object_sha256(entry)
    return updated, entry


def audit_target_encoding_v3(
    rows: Sequence[Mapping[str, Any]],
    *,
    native_template: Any,
    target_template: Any,
    expected_total_supervised_tokens: int | None = None,
) -> dict[str, Any]:
    """Audit a prepared v3 dataset without mutating any input row."""
    if isinstance(rows, (str, bytes, bytearray)) or not isinstance(rows, Sequence):
        raise Day20TargetEncodingV3Error("rows must be a sequence of objects")
    _assert_template_contract(target_template)
    entries: list[dict[str, Any]] = []
    by_skill: dict[str, int] = {}
    by_format: dict[str, int] = {}
    code_records = 0
    non_code_records = 0
    for row in rows:
        if not isinstance(row, Mapping):
            raise Day20TargetEncodingV3Error("dataset row is not an object")
        updated, entry = encode_target_row_v3(
            row, native_template=native_template, target_template=target_template
        )
        stored = {
            key: row.get(key)
            for key in (
                "qwen35_input_tokens",
                "qwen35_supervised_tokens",
                "qwen35_render_sha256",
                "qwen35_labels_sha256",
            )
        }
        expected = {key: updated[key] for key in stored}
        if stored != expected:
            raise Day20TargetEncodingV3Error(
                f"{row.get('sample_id')}: stored v3 token evidence drifted"
            )
        entries.append(entry)
        tokens = int(updated["qwen35_supervised_tokens"])
        skill = str(row.get("skill"))
        target_format = str(row.get("target_format"))
        by_skill[skill] = by_skill.get(skill, 0) + tokens
        by_format[target_format] = by_format.get(target_format, 0) + tokens
        if target_format == "code_continuation":
            code_records += 1
        else:
            non_code_records += 1
    total = sum(by_skill.values())
    if expected_total_supervised_tokens is not None and total != int(
        expected_total_supervised_tokens
    ):
        raise Day20TargetEncodingV3Error(
            "dataset supervised-token total differs from the expected budget"
        )
    audit: dict[str, Any] = {
        "schema_name": AUDIT_DOMAIN,
        "schema_version": 3,
        "status": "pass",
        "contract_version": TARGET_ENCODING_CONTRACT_VERSION,
        "contract_sha256": TARGET_ENCODING_CONTRACT_SHA256,
        "template_alias": TARGET_TEMPLATE_ALIAS,
        "records": len(entries),
        "code_records": code_records,
        "non_code_records": non_code_records,
        "native_supervised_tokens": sum(
            int(entry["native_supervised_tokens"]) for entry in entries
        ),
        "target_supervised_tokens": total,
        "target_supervised_tokens_by_skill": dict(sorted(by_skill.items())),
        "target_supervised_tokens_by_format": dict(sorted(by_format.items())),
        "code_exact_four_space_labels": sum(
            entry["code_exact_four_space_first_label"] is True for entry in entries
        ),
        "code_per_row_supervised_count_preserved": sum(
            entry["target_format"] == "code_continuation"
            and entry["per_row_supervised_count_preserved"]
            for entry in entries
        ),
        "non_code_input_ids_labels_identical": sum(
            entry["target_format"] != "code_continuation"
            and entry["non_code_input_ids_labels_identical"] is True
            for entry in entries
        ),
        "ordered_target_evidence_sha256": object_sha256(
            [entry["entry_sha256"] for entry in entries]
        ),
    }
    if (
        audit["code_exact_four_space_labels"] != code_records
        or audit["code_per_row_supervised_count_preserved"] != code_records
        or audit["non_code_input_ids_labels_identical"] != non_code_records
        or audit["native_supervised_tokens"] != total
    ):
        raise Day20TargetEncodingV3Error("aggregate target encoding audit failed")
    audit["audit_sha256"] = object_sha256(audit)
    return audit


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise Day20TargetEncodingV3Error(
                        f"blank JSONL row: {path}:{line_number}"
                    )
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise Day20TargetEncodingV3Error(
                        f"invalid JSONL: {path}:{line_number}"
                    ) from error
                if not isinstance(row, dict):
                    raise Day20TargetEncodingV3Error(
                        f"JSONL row is not an object: {path}:{line_number}"
                    )
                rows.append(row)
    except OSError as error:
        raise Day20TargetEncodingV3Error(f"cannot read dataset: {path}") from error
    if not rows:
        raise Day20TargetEncodingV3Error("dataset is empty")
    return rows


def audit_prepared_dataset_v3(
    dataset_path: Path,
    model_path: Path,
    *,
    expected_total_supervised_tokens: int | None = None,
) -> dict[str, Any]:
    """Build both templates and perform the complete dataset audit."""
    try:
        from day20_source_adapter import build_qwen35_template
    except ImportError as error:
        raise Day20TargetEncodingV3Error(
            "day20_source_adapter is required for the native audit"
        ) from error
    native = build_qwen35_template(model_path.expanduser().resolve())
    target = build_target_template_v3(model_path.expanduser().resolve())
    return audit_target_encoding_v3(
        _load_jsonl(dataset_path.expanduser().resolve()),
        native_template=native,
        target_template=target,
        expected_total_supervised_tokens=expected_total_supervised_tokens,
    )


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise Day20TargetEncodingV3Error(f"refusing to overwrite: {path}") from error
    except OSError as error:
        raise Day20TargetEncodingV3Error(f"cannot write audit: {path}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("contract")
    audit = subparsers.add_parser("audit")
    audit.add_argument("--model", required=True, type=Path)
    audit.add_argument("--dataset", required=True, type=Path)
    audit.add_argument("--output", required=True, type=Path)
    audit.add_argument("--expected-supervised-tokens", type=int)
    args = parser.parse_args()
    if args.command == "contract":
        print(
            json.dumps(
                {
                    "contract": TARGET_ENCODING_CONTRACT,
                    "contract_sha256": TARGET_ENCODING_CONTRACT_SHA256,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return
    result = audit_prepared_dataset_v3(
        args.dataset,
        args.model,
        expected_total_supervised_tokens=args.expected_supervised_tokens,
    )
    output = args.output.expanduser().resolve()
    if not output.parent.is_dir():
        raise Day20TargetEncodingV3Error(
            f"audit output parent is missing: {output.parent}"
        )
    _write_json_new(output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
