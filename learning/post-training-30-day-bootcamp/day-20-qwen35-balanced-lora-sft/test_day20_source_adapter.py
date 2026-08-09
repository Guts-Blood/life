#!/usr/bin/env python3
"""Offline tests for the pinned Day 20 source adapter."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import day20_contract as contract
import day20_source_adapter as adapter


class FakeQwen35Template:
    """Small deterministic train-template double; it never loads a model."""

    def encode(self, value, return_length=False):
        assert return_length is True
        messages = value["messages"]
        seed = contract.object_sha256(messages)
        input_ids = [1, 2, int(seed[:4], 16), int(seed[4:8], 16)]
        return {
            "input_ids": input_ids,
            "labels": [-100, -100, input_ids[2], input_ids[3]],
            "length": len(input_ids),
        }


def lineage(source: str, source_hash: str) -> dict[str, object]:
    spec = adapter.SOURCE_SPECS[source]
    return {
        "source": spec.source,
        "revision": spec.revision,
        "split": "train",
        "adapter": spec.adapter,
        "license": spec.license,
        "source_file_sha256": "a" * 64,
        "source_content_sha256": source_hash,
        "source_id": f"fixture:{source}",
    }


def audited_row(candidate: dict[str, object], source: str) -> dict[str, object]:
    row = copy.deepcopy(candidate)
    row["qwen35_tokenization"] = adapter.audit_messages(
        row["messages"], FakeQwen35Template()
    )
    target = row["messages"][-1]["content"]
    if source == "mmlu":
        source_record = {"answer": "ABCD".index(target.rsplit(" ", 1)[-1])}
    elif source == "gsm8k":
        source_record = {"answer": f"fixture reasoning #### {target.rsplit(' ', 1)[-1]}"}
    elif source == "tatqa":
        value = target.removeprefix("Final answer:").strip()
        scale = row["finance_scale"]
        if scale == "percent":
            value = value.removesuffix("%")
        elif scale != "none":
            value = value.removesuffix(f" {scale}")
        source_record = {"question": {"answer": value, "scale": scale}}
    elif source == "mbpp":
        source_record = {
            "code": row["code_prefix"] + "\n" + target,
            "test_list": [],
            "challenge_test_list": [],
        }
    else:
        raise AssertionError(source)
    source_hash = contract.object_sha256(source_record)
    row["reference_evidence"] = adapter.build_reference_evidence(
        source_key=source,
        candidate=row,
        source_record=source_record,
        source_content_sha256=source_hash,
    )
    row["source_lineage"] = lineage(source, source_hash)
    return row


class Day20PureTransformTests(unittest.TestCase):
    def test_fixed_revisions_and_data_only_patterns(self) -> None:
        self.assertEqual(
            adapter.SOURCE_SPECS["mmlu"].revision,
            "c30699e8356da336a370243923dbaf21066bb9fe",
        )
        self.assertEqual(
            adapter.SOURCE_SPECS["tulu"].revision,
            "fe0c7d350c9b4542b8d829a6f1daa1c259f0ba0e",
        )
        self.assertEqual(
            adapter.SOURCE_SPECS["gsm8k"].revision,
            "740312add88f781978c0658806c59bc2815b9866",
        )
        self.assertEqual(
            adapter.SOURCE_SPECS["tatqa"].revision,
            "c96247f5077eac447f63527fd3dcfdc58bb56d6a",
        )
        self.assertEqual(
            adapter.SOURCE_SPECS["mbpp"].revision,
            "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
        )
        self.assertEqual(
            adapter.SOURCE_SPECS["tulu_code"].revision,
            "1412abe88dd2976af977260788e033013449f7b2",
        )
        self.assertTrue(
            all(
                not pattern.endswith(".py")
                for spec in adapter.SOURCE_SPECS.values()
                for pattern in spec.allow_patterns
            )
        )

    def test_mmlu_is_exact_letter_task(self) -> None:
        row = adapter.transform_mmlu(
            {
                "question": "Which value is even?",
                "choices": ["1", "2", "3", "5"],
                "answer": 1,
                "subject": "elementary_mathematics",
            },
            "mmlu:aux:1",
        )
        self.assertEqual(row["target_format"], "general_mcq")
        self.assertEqual(row["messages"][-1]["content"], "Final answer: B")
        self.assertIn("A. 1", row["messages"][0]["content"])
        with self.assertRaises(adapter.SourceRecordRejected):
            adapter.transform_mmlu(
                {"question": "q", "choices": ["a", "b"], "answer": 0}, "bad"
            )

    def test_tulu_filter_keeps_single_turn_general_only(self) -> None:
        kept = adapter.transform_tulu_general(
            {
                "messages": [
                    {"role": "user", "content": "Write a concise museum notice."},
                    {"role": "assistant", "content": "Please keep voices low."},
                ]
            },
            "tulu:1",
        )
        self.assertEqual(kept["target_format"], "general_instruction")
        with self.assertRaises(adapter.SourceRecordRejected):
            adapter.transform_tulu_general(
                {
                    "messages": [
                        {"role": "user", "content": "Write a Python function."},
                        {"role": "assistant", "content": "def f(): pass"},
                    ]
                },
                "tulu:code",
            )
        with self.assertRaises(adapter.SourceRecordRejected):
            adapter.transform_tulu_general(
                {
                    "messages": [
                        {"role": "user", "content": "Format this."},
                        {"role": "assistant", "content": "```text\nanswer\n```"},
                    ]
                },
                "tulu:fence",
            )

    def test_gsm8k_preserves_reasoning_and_canonical_final(self) -> None:
        row = adapter.transform_gsm8k(
            {
                "question": "Sam has 2 apples and gets 3. How many?",
                "answer": "Sam adds 2 and 3. <<2+3=5>>\nHe gets five. #### 5.0",
            },
            "gsm:1",
        )
        answer = row["messages"][-1]["content"]
        self.assertNotIn("<<", answer)
        self.assertEqual(answer.splitlines()[-1], "Final answer: 5")
        with self.assertRaises(adapter.SourceRecordRejected):
            adapter.transform_gsm8k(
                {"question": "q", "answer": "Only an answer"}, "gsm:bad"
            )

    @staticmethod
    def _tatqa_document() -> dict[str, object]:
        return {
            "table": {"uid": "table-1", "table": [["Year", "Revenue"], ["2024", "10"]]},
            "paragraphs": [
                {"order": 2, "text": "Second paragraph."},
                {"order": 1, "text": "First paragraph."},
            ],
            "questions": [
                {
                    "uid": "q-arithmetic",
                    "question": "What was the percentage increase?",
                    "answer_type": "arithmetic",
                    "answer": "12.50",
                    "scale": "percent",
                },
                {
                    "uid": "q-count",
                    "question": "How many years are listed?",
                    "answer_type": "count",
                    "answer": 1,
                    "scale": "",
                },
                {
                    "uid": "q-span",
                    "question": "Which year?",
                    "answer_type": "span",
                    "answer": "2024",
                    "scale": "",
                },
            ],
        }

    def test_tatqa_keeps_only_scalar_arithmetic_count_and_scale(self) -> None:
        rows = adapter.transform_tatqa_document(self._tatqa_document(), "doc:1")
        self.assertEqual(len(rows), 2)
        targets = [row[0]["messages"][-1]["content"] for row in rows]
        self.assertEqual(targets, ["Final answer: 12.5%", "Final answer: 1"])
        self.assertEqual(rows[0][0]["finance_scale"], "percent")
        self.assertIn("[1] First paragraph.", rows[0][0]["messages"][0]["content"])
        # Lineage content includes the table/paragraph context, not only the question.
        self.assertIn("table", rows[0][1])
        self.assertIn("question", rows[0][1])

    def test_mbpp_is_continuation_and_compile_only(self) -> None:
        prefix, continuation, entry_point = adapter.split_mbpp_continuation(
            'import math\n\ndef area(r):\n    """Return circle area."""\n    return math.pi * r * r'
        )
        self.assertEqual(entry_point, "area")
        self.assertIn("def area", prefix)
        self.assertEqual(continuation, "    return math.pi * r * r")
        row = adapter.transform_mbpp(
            {
                "task_id": 1,
                "text": "Return the area of a circle.",
                "code": "def area(r):\n    return 3.14 * r * r",
                "test_list": ["assert area(1) == 3.14"],
            },
            "mbpp:full:train:1",
            variant="full",
        )
        self.assertFalse(row["messages"][-1]["content"].lstrip().startswith("def "))
        compile(row["code_prefix"] + "\n" + row["messages"][-1]["content"], "<test>", "exec")
        helper_prefix, helper_continuation, _ = adapter.split_mbpp_continuation(
            "def outer(x):\n    def inner():\n        return x\n    return inner()"
        )
        self.assertIn("def inner", helper_prefix)
        self.assertEqual(helper_continuation, "    return inner()")

    def test_tulu_code_accepts_only_plain_static_function_body(self) -> None:
        row = adapter.transform_tulu_code(
            {
                "messages": [
                    {"role": "user", "content": "Write a function that doubles x."},
                    {
                        "role": "assistant",
                        "content": "import math\n\ndef double(x):\n    return x * 2",
                    },
                ]
            },
            "tulu-code:1",
        )
        self.assertIn("import math", row["code_prefix"])
        self.assertIn("def double", row["code_prefix"])
        self.assertEqual(row["messages"][-1]["content"], "    return x * 2")
        with self.assertRaises(adapter.SourceRecordRejected):
            adapter.transform_tulu_code(
                {
                    "messages": [
                        {"role": "user", "content": "Write f."},
                        {"role": "assistant", "content": "```python\ndef f():\n    return 1\n```"},
                    ]
                },
                "tulu-code:fenced",
            )
        with self.assertRaises(adapter.SourceRecordRejected):
            adapter.transform_tulu_code(
                {
                    "messages": [
                        {"role": "user", "content": "Write f."},
                        {
                            "role": "assistant",
                            "content": "def f(x):\n    value = x\n    import math\n    return value",
                        },
                    ]
                },
                "tulu-code:late-import",
            )

        source_record = {
            "messages": [
                {"role": "user", "content": "Keep x."},
                {
                    "role": "assistant",
                    "content": "def keep(x):\n    value = x  \n    return value",
                },
            ]
        }
        transformed = adapter.transform_tulu_code(
            source_record, "tulu-code:trailing-space"
        )
        canonical = adapter.canonicalize_candidate_for_audit(transformed)
        self.assertEqual(
            canonical["messages"][-1]["content"],
            "    value = x\n    return value",
        )
        canonical["qwen35_tokenization"] = adapter.audit_messages(
            canonical["messages"], FakeQwen35Template()
        )
        source_hash = contract.object_sha256(source_record)
        canonical["reference_evidence"] = adapter.build_reference_evidence(
            source_key="tulu_code",
            candidate=canonical,
            source_record=source_record,
            source_content_sha256=source_hash,
        )
        canonical["source_lineage"] = lineage("tulu_code", source_hash)
        normalized = contract.normalize_and_validate_record(
            canonical, expected_skill="code"
        )
        self.assertEqual(
            canonical["qwen35_tokenization"]["messages_sha256"],
            contract.object_sha256(normalized["messages"]),
        )

    def test_token_sequence_formula_and_fake_template_audit(self) -> None:
        expected = contract.object_sha256([1, 2, -100])
        self.assertEqual(adapter.token_sequence_sha256([1, 2, -100]), expected)
        messages = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]
        evidence = adapter.audit_messages(messages, FakeQwen35Template())
        self.assertEqual(evidence["input_tokens"], 4)
        self.assertEqual(evidence["supervised_tokens"], 2)
        self.assertEqual(evidence["messages_sha256"], contract.object_sha256(messages))
        self.assertEqual(
            evidence["render_sha256"],
            adapter.token_sequence_sha256(
                FakeQwen35Template().encode({"messages": messages}, return_length=True)["input_ids"]
            ),
        )

        class ExplicitShiftTemplate:
            def encode(self, value, return_length=False):
                return {"input_ids": [9, 8, 7], "labels": [9, -100, 7], "length": 3}

        # Position zero is excluded because causal LM predicts labels[1:].
        shifted = adapter.audit_messages(messages, ExplicitShiftTemplate())
        self.assertEqual(shifted["supervised_tokens"], 1)

    def test_reference_evidence_is_gold_bound_and_self_hashed(self) -> None:
        candidate = adapter.transform_mbpp(
            {
                "text": "Add two values.",
                "code": "def add(a, b):\n    return a + b",
                "test_list": ["assert add(1, 2) == 3"],
                "challenge_test_list": ["assert add(-1, 1) == 0"],
            },
            "mbpp:reference",
            variant="full",
        )
        source_record = {
            "code": "def add(a, b):\n    return a + b",
            "test_list": ["assert add(1, 2) == 3"],
            "challenge_test_list": ["assert add(-1, 1) == 0"],
        }
        source_hash = contract.object_sha256(source_record)
        evidence = adapter.build_reference_evidence(
            source_key="mbpp",
            candidate=candidate,
            source_record=source_record,
            source_content_sha256=source_hash,
        )
        self.assertEqual(
            adapter.verify_reference_evidence(evidence),
            evidence["reference_evidence_sha256"],
        )
        self.assertTrue(evidence["checks"]["source_rebuilt_match"])
        self.assertTrue(evidence["checks"]["static_compile"])
        self.assertEqual(evidence["checks"]["tests_count"], 1)
        self.assertEqual(evidence["checks"]["challenge_tests_count"], 1)
        tampered = copy.deepcopy(evidence)
        tampered["checks"]["tests_count"] = 99
        with self.assertRaisesRegex(adapter.Day20SourceAdapterError, "self-hash"):
            adapter.verify_reference_evidence(tampered)


class Day20SourcePipelineTests(unittest.TestCase):
    @staticmethod
    def _write_jsonl(path: Path, rows) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def _candidates(self) -> dict[str, dict[str, object]]:
        return {
            "general": adapter.transform_mmlu(
                {"question": "Pick B.", "choices": ["A", "B", "C", "D"], "answer": 1},
                "mmlu:1",
            ),
            "math": adapter.transform_gsm8k(
                {"question": "One plus one?", "answer": "One plus one is two. #### 2"},
                "gsm:1",
            ),
            "finance": adapter.transform_tatqa_document(
                self._tatqa_document(), "doc:1"
            )[0][0],
            "code": adapter.transform_mbpp(
                {
                    "text": "Add two values.",
                    "code": "def add(a, b):\n    return a + b",
                    "test_list": ["assert add(1, 2) == 3"],
                },
                "mbpp:1",
                variant="full",
            ),
        }

    @staticmethod
    def _tatqa_document() -> dict[str, object]:
        return {
            "table": {"table": [["Metric", "Value"], ["Growth", "5"]]},
            "paragraphs": [{"order": 1, "text": "Growth was reported."}],
            "questions": [
                {
                    "uid": "f1",
                    "question": "What was growth?",
                    "answer_type": "arithmetic",
                    "answer": "5",
                    "scale": "million",
                }
            ],
        }

    def test_independent_reaudit_supports_nested_and_flat_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "tokenizer.json").write_text("{}", encoding="utf-8")
            sources: dict[str, Path] = {}
            candidates = self._candidates()
            for skill, candidate in candidates.items():
                row = audited_row(candidate, {"general": "mmlu", "math": "gsm8k", "finance": "tatqa", "code": "mbpp"}[skill])
                # Exercise prepare_day20's flattened canonical representation too.
                if skill == "general":
                    row = contract.normalize_and_validate_record(row, expected_skill=skill)
                path = root / f"{skill}.jsonl"
                self._write_jsonl(path, [row])
                sources[skill] = path

            with mock.patch.object(
                adapter, "build_qwen35_template", return_value=FakeQwen35Template()
            ):
                report = adapter.audit_normalized_sources(
                    model_path=model, sources=sources
                )
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["records"], 4)
            self.assertEqual(report["supervised_tokens"], 8)

            finance_row = json.loads(sources["finance"].read_text(encoding="utf-8"))
            finance_row["qwen35_tokenization"]["supervised_tokens"] = 1
            self._write_jsonl(root / "tampered.jsonl", [finance_row])
            tampered_sources = dict(sources)
            tampered_sources["finance"] = root / "tampered.jsonl"
            with mock.patch.object(
                adapter, "build_qwen35_template", return_value=FakeQwen35Template()
            ), self.assertRaisesRegex(
                adapter.Day20SourceAdapterError, "token evidence drift"
            ):
                adapter.audit_normalized_sources(
                    model_path=model, sources=tampered_sources
                )

    def test_local_staging_requires_exact_revision_attestation(self) -> None:
        with self.assertRaisesRegex(adapter.Day20SourceAdapterError, "exact revision"):
            adapter._normalize_local_paths({"local_paths": {"mmlu": "/tmp/source.jsonl"}})
        value = adapter._normalize_local_paths(
            {
                "local_paths": {"mmlu": "/tmp/source.jsonl"},
                "local_revisions": {
                    "mmlu": adapter.SOURCE_SPECS["mmlu"].revision
                },
            }
        )
        self.assertIn("mmlu", value)

    def test_mbpp_capacity_fails_closed_without_duplication(self) -> None:
        pools = {
            "code": [
                {
                    "sample_id": f"code:{index}",
                    "target_format": "code_continuation",
                    "qwen35_tokenization": {"supervised_tokens": 2},
                }
                for index in range(10)
            ]
        }
        with self.assertRaisesRegex(
            adapter.Day20SourceAdapterError,
            r"never duplicated: code_continuation: available=20, required=21",
        ):
            adapter.validate_pool_capacity(
                pools,
                required_tokens_by_format={"code_continuation": 21},
                probe_tokens_by_format={},
            )
        report = adapter.validate_pool_capacity(
            pools,
            required_tokens_by_format={"code_continuation": 20},
            probe_tokens_by_format={},
        )
        self.assertTrue(report["code_continuation"]["exact_required_subset_reachable"])

    def test_tulu_code_is_only_the_minimum_stable_capacity_fill(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw: dict[str, list[dict[str, object]]] = {
                "mmlu": [
                    {"question": "Pick B.", "choices": ["A", "B", "C", "D"], "answer": 1}
                ],
                "tulu": [
                    {
                        "id": "general-1",
                        "messages": [
                            {"role": "user", "content": "Write a museum notice."},
                            {"role": "assistant", "content": "Please keep voices low."},
                        ],
                    }
                ],
                "gsm8k": [
                    {"question": "One plus one?", "answer": "One plus one is two. #### 2"}
                ],
                "tatqa": [self._tatqa_document()],
                "mbpp": [
                    {
                        "task_id": 1,
                        "text": "Add two values.",
                        "code": "def add(a, b):\n    return a + b",
                        "test_list": ["assert add(1, 2) == 3"],
                    }
                ],
                "tulu_code": [
                    {
                        "id": f"code-{index}",
                        "messages": [
                            {"role": "user", "content": f"Write function f_{index}."},
                            {
                                "role": "assistant",
                                "content": f"def f_{index}(x):\n    return x + {index}",
                            },
                        ],
                    }
                    for index in range(3)
                ],
            }
            source_files: dict[str, list[adapter.SourceFile]] = {}
            for key, rows in raw.items():
                path = root / f"{key}.jsonl"
                self._write_jsonl(path, rows)
                source_files[key] = [
                    adapter.SourceFile(
                        spec=adapter.SOURCE_SPECS[key],
                        path=path,
                        split="train",
                        variant="full" if key == "mbpp" else key,
                        relative_name=path.name,
                    )
                ]
            required = {
                "general_mcq": 2,
                "general_instruction": 2,
                "math_reasoning": 2,
                "finance_value_scale": 2,
                "code_continuation": 6,
            }
            probe_required = {target_format: 2 for target_format in required}
            with mock.patch.dict(
                adapter.MAIN_TOKENS_BY_FORMAT, required, clear=True
            ), mock.patch.dict(
                adapter.PROBE_TOKENS_BY_FORMAT, probe_required, clear=True
            ):
                pools, metrics = adapter.build_normalized_pools(
                    source_files=source_files,
                    template=FakeQwen35Template(),
                    config={},
                )
            code_rows = [
                row for row in pools["code"] if row["target_format"] == "code_continuation"
            ]
            self.assertEqual(len(code_rows), 3)  # one MBPP + exactly two supplement rows
            self.assertEqual(metrics["source_stats"]["mbpp"]["accepted_records"], 1)
            self.assertEqual(metrics["source_stats"]["tulu_code"]["accepted_records"], 2)
            self.assertEqual(
                metrics["rejections"]["tulu_code:capacity_fill_not_needed"], 1
            )
            for rows in pools.values():
                for row in rows:
                    self.assertEqual(
                        adapter.verify_reference_evidence(row["reference_evidence"]),
                        row["reference_evidence"]["reference_evidence_sha256"],
                    )


if __name__ == "__main__":
    unittest.main()
