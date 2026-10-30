#!/usr/bin/env python3
"""Offline unit tests for the Day 20 v3 target-encoding boundary fix."""

from __future__ import annotations

import copy
import sys
import types
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest import mock

import day20_target_encoding_v3 as encoding


@dataclass
class FakeTemplateMeta:
    template_type: str
    template_cls: type
    suffix: list[str]


class FakeTokenizer:
    """Small reversible tokenizer with the pinned boundary/suffix token IDs."""

    _CHAR_OFFSET = 1_000

    def encode(self, text: str) -> list[int]:
        result: list[int] = []
        index = 0
        while index < len(text):
            if text.startswith(encoding.INDENT_TEXT, index):
                result.append(encoding.INDENT_TOKEN_ID)
                index += len(encoding.INDENT_TEXT)
            elif text[index] == "\n":
                result.append(encoding.NEWLINE_TOKEN_ID)
                index += 1
            else:
                result.append(self._CHAR_OFFSET + ord(text[index]))
                index += 1
        return result

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del clean_up_tokenization_spaces
        pieces: list[str] = []
        for token_id in token_ids:
            if token_id == encoding.INDENT_TOKEN_ID:
                pieces.append(encoding.INDENT_TEXT)
            elif token_id == encoding.NEWLINE_TOKEN_ID:
                pieces.append("\n")
            elif token_id == encoding.IM_END_TOKEN_ID:
                if not skip_special_tokens:
                    pieces.append("<|im_end|>")
            elif token_id >= self._CHAR_OFFSET:
                pieces.append(chr(token_id - self._CHAR_OFFSET))
            elif not skip_special_tokens:
                pieces.append(f"<{token_id}>")
        return "".join(pieces)


class FakeQwen3_5Template:
    """Native-template stub reproducing strip + empty-think indent masking."""

    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.mode = "train"
        self.template_backend = "swift"
        self.enable_thinking = False
        self.add_non_thinking_prefix = True
        self.padding_free = False
        self.packing = False
        self._loss_scale = encoding.LOSS_SCALE
        self.template_meta: FakeTemplateMeta | None = None
        self.fake_tail_ids = [encoding.IM_END_TOKEN_ID, encoding.NEWLINE_TOKEN_ID]
        self.fake_tail_labels: list[int] | None = None
        self.fake_mask_indent = True

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def _tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def _encode_context_list(self, context: list[str]) -> tuple[list[int], None]:
        if context != ["<|im_end|>\n"]:
            raise AssertionError(f"unexpected suffix context: {context!r}")
        return ([encoding.IM_END_TOKEN_ID, encoding.NEWLINE_TOKEN_ID], None)

    def _swift_prepare_inputs(self, inputs: SimpleNamespace) -> None:
        for message in inputs.messages:
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = content.strip()

    def _encode(self, inputs: SimpleNamespace) -> dict[str, list[int]]:
        # Deliberately polymorphic: the v3 override restores the Code target
        # after this native strip, just as pinned ms-swift does.
        self._swift_prepare_inputs(inputs)
        target = inputs.messages[-1]["content"]
        target_ids = self._tokenize(target)
        prefix_ids = [41, 42]
        tail_ids = list(self.fake_tail_ids)
        input_ids = prefix_ids + target_ids + tail_ids
        tail_labels = (
            list(self.fake_tail_labels)
            if self.fake_tail_labels is not None
            else list(tail_ids)
        )
        labels = [-100] * len(prefix_ids) + list(target_ids) + tail_labels
        if (
            self.fake_mask_indent
            and target_ids
            and target_ids[0] == encoding.INDENT_TOKEN_ID
        ):
            labels[len(prefix_ids)] = -100
        return {"input_ids": input_ids, "labels": labels}

    def encode(
        self, value: dict[str, object], *, return_length: bool = False
    ) -> dict[str, list[int]]:
        del return_length
        messages = value.get("messages")
        if not isinstance(messages, list):
            raise TypeError("messages must be a list")
        return self._encode(SimpleNamespace(messages=messages))


class Day20TargetEncodingV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.mapping: dict[str, FakeTemplateMeta] = {
            encoding.SOURCE_TEMPLATE: FakeTemplateMeta(
                template_type=encoding.SOURCE_TEMPLATE,
                template_cls=FakeQwen3_5Template,
                suffix=["<|im_end|>\n"],
            )
        }

        swift_module = types.ModuleType("swift")
        swift_module.__path__ = []  # type: ignore[attr-defined]
        template_module = types.ModuleType("swift.template")
        template_module.__path__ = []  # type: ignore[attr-defined]
        templates_module = types.ModuleType("swift.template.templates")
        templates_module.__path__ = []  # type: ignore[attr-defined]
        qwen_module = types.ModuleType("swift.template.templates.qwen")

        def register_template(meta: FakeTemplateMeta) -> None:
            if meta.template_type in self.mapping:
                raise RuntimeError(f"duplicate template: {meta.template_type}")
            self.mapping[meta.template_type] = meta

        template_module.TEMPLATE_MAPPING = self.mapping  # type: ignore[attr-defined]
        template_module.register_template = register_template  # type: ignore[attr-defined]
        qwen_module.Qwen3_5Template = FakeQwen3_5Template  # type: ignore[attr-defined]
        swift_module.template = template_module  # type: ignore[attr-defined]
        templates_module.qwen = qwen_module  # type: ignore[attr-defined]
        self.modules = mock.patch.dict(
            sys.modules,
            {
                "swift": swift_module,
                "swift.template": template_module,
                "swift.template.templates": templates_module,
                "swift.template.templates.qwen": qwen_module,
            },
        )
        self.modules.start()

    def tearDown(self) -> None:
        self.modules.stop()

    def templates(self) -> tuple[FakeQwen3_5Template, FakeQwen3_5Template]:
        encoding.register_target_template_v3()
        native = FakeQwen3_5Template()
        native.template_meta = self.mapping[encoding.SOURCE_TEMPLATE]
        target_cls = self.mapping[encoding.TARGET_TEMPLATE_ALIAS].template_cls
        target = target_cls()
        target.template_meta = self.mapping[encoding.TARGET_TEMPLATE_ALIAS]
        return native, target

    @staticmethod
    def code_row(
        target: str = "    return x + 1\n    # exact bytes"
    ) -> dict[str, object]:
        return {
            "sample_id": "code:fixture:0001",
            "skill": "code",
            "target_format": "code_continuation",
            "messages": [
                {
                    "role": "user",
                    "content": encoding.CODE_PROMPT_PREFIX + " fixture\n\ndef solve(x):",
                },
                {"role": "assistant", "content": target},
            ],
        }

    @staticmethod
    def non_code_row() -> dict[str, object]:
        return {
            "sample_id": "general:fixture:0001",
            "skill": "general",
            "target_format": "general_instruction",
            "messages": [
                {"role": "user", "content": "Give one concise fact."},
                {"role": "assistant", "content": "The sky appears blue."},
            ],
        }

    def test_code_balanced_exchange_preserves_count_and_exact_bytes(self) -> None:
        native, target = self.templates()
        row = self.code_row()
        original = copy.deepcopy(row)
        native_ids, native_labels = encoding._encode(native, row["messages"])
        target_ids, target_labels = encoding._encode(target, row["messages"])

        first_supervised = next(
            index for index, label in enumerate(target_labels) if label != -100
        )
        self.assertEqual(target_ids[first_supervised], encoding.INDENT_TOKEN_ID)
        self.assertEqual(target_labels[first_supervised], encoding.INDENT_TOKEN_ID)
        self.assertEqual(
            target_ids[-2:], [encoding.IM_END_TOKEN_ID, encoding.NEWLINE_TOKEN_ID]
        )
        self.assertEqual(target_labels[-2:], [encoding.IM_END_TOKEN_ID, -100])
        self.assertEqual(
            encoding._supervised_tokens(target_labels),
            encoding._supervised_tokens(native_labels),
        )
        self.assertEqual(len(target_ids), len(native_ids) + 1)
        self.assertEqual(
            encoding._decode_supervised(target, target_labels),
            row["messages"][-1]["content"],
        )

        updated, evidence = encoding.encode_target_row_v3(
            row, native_template=native, target_template=target
        )
        self.assertEqual(row, original)
        self.assertTrue(evidence["code_exact_four_space_first_label"])
        self.assertTrue(evidence["per_row_supervised_count_preserved"])
        self.assertEqual(
            updated["qwen35_supervised_tokens"],
            evidence["target_supervised_tokens"],
        )

    def test_non_code_arrays_are_native_identical(self) -> None:
        native, target = self.templates()
        row = self.non_code_row()
        native_arrays = encoding._encode(native, row["messages"])
        target_arrays = encoding._encode(target, row["messages"])
        self.assertEqual(target_arrays, native_arrays)

        updated, evidence = encoding.encode_target_row_v3(
            row, native_template=native, target_template=target
        )
        self.assertTrue(evidence["non_code_input_ids_labels_identical"])
        self.assertEqual(
            updated["qwen35_render_sha256"], evidence["target_input_ids_sha256"]
        )
        self.assertEqual(
            updated["qwen35_labels_sha256"], evidence["target_labels_sha256"]
        )

    def test_prepared_dataset_audit_covers_code_and_non_code(self) -> None:
        native, target = self.templates()
        rows = []
        for row in (self.code_row(), self.non_code_row()):
            updated, _ = encoding.encode_target_row_v3(
                row, native_template=native, target_template=target
            )
            rows.append(updated)
        expected = sum(int(row["qwen35_supervised_tokens"]) for row in rows)
        report = encoding.audit_target_encoding_v3(
            rows,
            native_template=native,
            target_template=target,
            expected_total_supervised_tokens=expected,
        )
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["records"], 2)
        self.assertEqual(report["code_exact_four_space_labels"], 1)
        self.assertEqual(report["non_code_input_ids_labels_identical"], 1)
        self.assertEqual(report["native_supervised_tokens"], expected)

    def test_alias_is_train_only(self) -> None:
        _, target = self.templates()
        target.set_mode("infer")
        with self.assertRaisesRegex(
            encoding.Day20TargetEncodingV3Error, "training-only"
        ):
            encoding._encode(target, self.code_row()["messages"])

    def test_five_space_target_fails_closed(self) -> None:
        native, target = self.templates()
        with self.assertRaisesRegex(
            encoding.Day20TargetEncodingV3Error, "exactly four-space indentation"
        ):
            encoding.encode_target_row_v3(
                self.code_row("     return x"),
                native_template=native,
                target_template=target,
            )

    def test_multiple_round_code_messages_fail_closed(self) -> None:
        native, target = self.templates()
        row = self.code_row()
        row["messages"].extend(
            [
                {"role": "user", "content": "Again"},
                {"role": "assistant", "content": "    return 2"},
            ]
        )
        with self.assertRaisesRegex(
            encoding.Day20TargetEncodingV3Error, "one user/assistant round"
        ):
            encoding.encode_target_row_v3(
                row, native_template=native, target_template=target
            )

    def test_code_prompt_prefix_drift_fails_closed(self) -> None:
        native, target = self.templates()
        row = self.code_row()
        row["messages"][0]["content"] = "Complete this function without the contract."
        with self.assertRaisesRegex(
            encoding.Day20TargetEncodingV3Error, "prompt detection disagrees"
        ):
            encoding.encode_target_row_v3(
                row, native_template=native, target_template=target
            )

    def test_tail_and_label_drift_fail_closed(self) -> None:
        for drift, pattern in (
            ({"fake_tail_ids": [encoding.IM_END_TOKEN_ID, 999]}, "must end"),
            ({"fake_tail_labels": [encoding.IM_END_TOKEN_ID, -100]}, "must end"),
            ({"fake_mask_indent": False}, "masked four-space token"),
        ):
            with self.subTest(drift=drift):
                _, target = self.templates()
                for name, value in drift.items():
                    setattr(target, name, value)
                with self.assertRaisesRegex(
                    encoding.Day20TargetEncodingV3Error, pattern
                ):
                    encoding._encode(target, self.code_row()["messages"])

    def test_registration_is_idempotent_and_collision_is_rejected(self) -> None:
        encoding.register_target_template_v3()
        registered = self.mapping[encoding.TARGET_TEMPLATE_ALIAS]
        encoding.register_target_template_v3()
        self.assertIs(self.mapping[encoding.TARGET_TEMPLATE_ALIAS], registered)

        self.mapping[encoding.TARGET_TEMPLATE_ALIAS] = FakeTemplateMeta(
            template_type=encoding.TARGET_TEMPLATE_ALIAS,
            template_cls=FakeQwen3_5Template,
            suffix=["<|im_end|>\n"],
        )
        with self.assertRaisesRegex(
            encoding.Day20TargetEncodingV3Error, "alias collision"
        ):
            encoding.register_target_template_v3()


if __name__ == "__main__":
    unittest.main()
