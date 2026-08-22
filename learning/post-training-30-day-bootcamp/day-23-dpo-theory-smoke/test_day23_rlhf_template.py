#!/usr/bin/env python3
"""Offline branch-level tests for the Day 23 RLHF-only template alias."""

from __future__ import annotations

import copy
import sys
import types
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest import mock

import day23_rlhf_template as template_module


day20 = template_module.day20


@dataclass
class FakeTemplateMeta:
    template_type: str
    template_cls: type
    suffix: list[str]


class FakeTokenizer:
    _OFFSET = 1_000

    def encode(self, text: str) -> list[int]:
        result: list[int] = []
        index = 0
        while index < len(text):
            if text.startswith(day20.INDENT_TEXT, index):
                result.append(day20.INDENT_TOKEN_ID)
                index += 4
            elif text[index] == "\n":
                result.append(day20.NEWLINE_TOKEN_ID)
                index += 1
            else:
                result.append(self._OFFSET + ord(text[index]))
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
            if token_id == day20.INDENT_TOKEN_ID:
                pieces.append(day20.INDENT_TEXT)
            elif token_id == day20.NEWLINE_TOKEN_ID:
                pieces.append("\n")
            elif token_id == day20.IM_END_TOKEN_ID:
                if not skip_special_tokens:
                    pieces.append("<|im_end|>")
            elif token_id >= self._OFFSET:
                pieces.append(chr(token_id - self._OFFSET))
        return "".join(pieces)


class FakeQwen3_5Template:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.mode = "rlhf"
        self.template_backend = "swift"
        self.enable_thinking = False
        self.add_non_thinking_prefix = True
        self.padding_free = False
        self.packing = False
        self._loss_scale = template_module.LOSS_SCALE
        self.max_length = 512
        self.template_meta: FakeTemplateMeta | None = None
        self.tail = [day20.IM_END_TOKEN_ID, day20.NEWLINE_TOKEN_ID]

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def _tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def _encode_context_list(self, context: list[str]) -> tuple[list[int], None]:
        if context != ["<|im_end|>\n"]:
            raise AssertionError(context)
        return ([day20.IM_END_TOKEN_ID, day20.NEWLINE_TOKEN_ID], None)

    def _swift_prepare_inputs(self, inputs: SimpleNamespace) -> None:
        for message in inputs.messages:
            if isinstance(message.get("content"), str):
                message["content"] = message["content"].strip()

    def _encode(self, inputs: SimpleNamespace) -> dict[str, list[int]]:
        self._swift_prepare_inputs(inputs)
        target = inputs.messages[-1]["content"]
        target_ids = self._tokenize(target)
        input_ids = [41, 42] + target_ids + list(self.tail)
        labels = [-100, -100] + target_ids + list(self.tail)
        if target_ids and target_ids[0] == day20.INDENT_TOKEN_ID:
            labels[2] = -100
        return {"input_ids": input_ids, "labels": labels}

    def encode(self, row: dict[str, object], *, return_length: bool = False):
        del return_length
        messages = copy.deepcopy(row["messages"])
        rejected = [
            copy.deepcopy(messages[0]),
            {"role": "assistant", "content": row["rejected_response"]},
        ]
        chosen_encoded = self._encode(SimpleNamespace(messages=messages))
        rejected_encoded = self._encode(SimpleNamespace(messages=rejected))
        return {
            **{f"chosen_{key}": value for key, value in chosen_encoded.items()},
            **{f"rejected_{key}": value for key, value in rejected_encoded.items()},
        }


class Day23RLHFTemplateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mapping: dict[str, FakeTemplateMeta] = {
            template_module.SOURCE_TEMPLATE: FakeTemplateMeta(
                template_module.SOURCE_TEMPLATE,
                FakeQwen3_5Template,
                ["<|im_end|>\n"],
            )
        }
        swift = types.ModuleType("swift")
        swift.__path__ = []  # type: ignore[attr-defined]
        swift_template = types.ModuleType("swift.template")
        swift_template.__path__ = []  # type: ignore[attr-defined]
        swift_templates = types.ModuleType("swift.template.templates")
        swift_templates.__path__ = []  # type: ignore[attr-defined]
        qwen = types.ModuleType("swift.template.templates.qwen")

        def register(meta: FakeTemplateMeta) -> None:
            if meta.template_type in self.mapping:
                raise RuntimeError("duplicate")
            self.mapping[meta.template_type] = meta

        swift_template.TEMPLATE_MAPPING = self.mapping  # type: ignore[attr-defined]
        swift_template.register_template = register  # type: ignore[attr-defined]
        qwen.Qwen3_5Template = FakeQwen3_5Template  # type: ignore[attr-defined]
        swift.template = swift_template  # type: ignore[attr-defined]
        swift_templates.qwen = qwen  # type: ignore[attr-defined]
        self.modules = mock.patch.dict(
            sys.modules,
            {
                "swift": swift,
                "swift.template": swift_template,
                "swift.template.templates": swift_templates,
                "swift.template.templates.qwen": qwen,
            },
        )
        self.modules.start()

    def tearDown(self) -> None:
        self.modules.stop()

    def target(self) -> FakeQwen3_5Template:
        template_module.register_dpo_template()
        cls = self.mapping[template_module.TARGET_TEMPLATE_ALIAS].template_cls
        result = cls()
        result.template_meta = self.mapping[template_module.TARGET_TEMPLATE_ALIAS]
        template_module._assert_template_contract(result, max_length=512)
        return result

    @staticmethod
    def row() -> dict[str, object]:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": day20.CODE_PROMPT_PREFIX + " fixture\n\ndef solve(x):",
                },
                {"role": "assistant", "content": "    return x + 1"},
            ],
            "rejected_response": "    return x - 1",
        }

    def test_both_rlhf_branches_preserve_four_space_boundary(self) -> None:
        target = self.target()
        encoded = template_module.encode_dpo_row(target, self.row())
        for branch, expected in (
            ("chosen", "    return x + 1"),
            ("rejected", "    return x - 1"),
        ):
            ids = encoded[f"{branch}_input_ids"]
            labels = encoded[f"{branch}_labels"]
            start = next(index for index, label in enumerate(labels) if label != -100)
            self.assertEqual(ids[start], day20.INDENT_TOKEN_ID)
            self.assertEqual(labels[start], day20.INDENT_TOKEN_ID)
            self.assertEqual(ids[-2:], [day20.IM_END_TOKEN_ID, day20.NEWLINE_TOKEN_ID])
            self.assertEqual(labels[-2:], [day20.IM_END_TOKEN_ID, -100])
            self.assertEqual(day20._decode_supervised(target, labels), expected)
        chosen_start = next(
            i for i, label in enumerate(encoded["chosen_labels"]) if label != -100
        )
        rejected_start = next(
            i for i, label in enumerate(encoded["rejected_labels"]) if label != -100
        )
        self.assertEqual(
            encoded["chosen_input_ids"][:chosen_start],
            encoded["rejected_input_ids"][:rejected_start],
        )

    def test_native_template_strips_the_required_indentation(self) -> None:
        native = FakeQwen3_5Template()
        native.template_meta = self.mapping[template_module.SOURCE_TEMPLATE]
        encoded = native.encode(self.row())
        start = next(i for i, label in enumerate(encoded["chosen_labels"]) if label != -100)
        self.assertNotEqual(encoded["chosen_input_ids"][start], day20.INDENT_TOKEN_ID)

    def test_alias_is_rlhf_only(self) -> None:
        target = self.target()
        target.set_mode("train")
        with self.assertRaisesRegex(template_module.Day23RLHFTemplateError, "RLHF-only"):
            template_module.encode_dpo_row(target, self.row())

    def test_five_space_target_fails_closed(self) -> None:
        target = self.target()
        row = self.row()
        row["messages"][1]["content"] = "     return x"
        with self.assertRaisesRegex(template_module.Day23RLHFTemplateError, "exactly four"):
            template_module.encode_dpo_row(target, row)

    def test_tail_drift_fails_closed(self) -> None:
        target = self.target()
        target.tail = [day20.IM_END_TOKEN_ID, 999]
        with self.assertRaisesRegex(template_module.Day23RLHFTemplateError, "must end"):
            template_module.encode_dpo_row(target, self.row())

    def test_registration_is_idempotent_and_collision_fails(self) -> None:
        template_module.register_dpo_template()
        registered = self.mapping[template_module.TARGET_TEMPLATE_ALIAS]
        template_module.register_dpo_template()
        self.assertIs(self.mapping[template_module.TARGET_TEMPLATE_ALIAS], registered)
        self.mapping[template_module.TARGET_TEMPLATE_ALIAS] = FakeTemplateMeta(
            template_module.TARGET_TEMPLATE_ALIAS,
            FakeQwen3_5Template,
            ["<|im_end|>\n"],
        )
        with self.assertRaisesRegex(template_module.Day23RLHFTemplateError, "collision"):
            template_module.register_dpo_template()


if __name__ == "__main__":
    unittest.main()
