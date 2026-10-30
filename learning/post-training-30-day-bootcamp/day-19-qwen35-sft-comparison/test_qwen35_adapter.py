#!/usr/bin/env python3
"""Regression tests for the Day 19 Qwen3.5 response and code adapters."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCORERS_PATH = HERE.parent / "day-10-frozen-eval-baseline" / "day10_scorers.py"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


QWEN = load("day19_qwen35_response_adapter_for_test", HERE / "qwen35_response_adapter.py")
CODE = load("day19_humaneval_adapter_for_test", HERE / "day19_humaneval_adapter.py")
RESCORE = load("day19_qwen35_rescore_for_test", HERE / "rescore_day19_qwen35_v2.py")
SCORERS = load("day19_frozen_scorers_for_adapter_test", SCORERS_PATH)
SCORE_V2 = load("day19_qwen35_code_e2b_v2_for_test", HERE / "score_day19_code_e2b_v2.py")


class Qwen35ResponseAdapterTests(unittest.TestCase):
    def response_capture(self):
        generated_text = "answer\n"
        boundary = QWEN.adapt_qwen35_text(
            QWEN.NON_THINKING_PREFIX + generated_text,
            generated_only_text=generated_text,
        )
        capture = {
            "method": "ms_swift_return_details",
            "prompt_token_ids": [10, 20],
            "prompt_token_ids_sha256": QWEN.token_ids_sha256([10, 20]),
            "generated_token_ids": [30, 40],
            "generated_token_ids_sha256": QWEN.token_ids_sha256([30, 40]),
            "generated_only_text": generated_text,
            "generated_only_text_sha256": QWEN.text_sha256(generated_text),
            "response_adapter_verification": {
                "adapter_version": QWEN.ADAPTER_VERSION,
                "prefix_status": boundary["prefix_status"],
                "final_text_sha256": boundary["final_text_sha256"],
            },
        }
        generation = {"prompt_tokens": 2, "completion_tokens": 2}
        return capture, generation

    def test_generated_token_slice_is_strict_and_does_not_mutate(self) -> None:
        prompt = [10, 20, 30]
        output = [10, 20, 30, 40, 50]
        prompt_before = list(prompt)
        output_before = list(output)
        self.assertEqual(QWEN.slice_generated_token_ids(prompt, output), (40, 50))
        self.assertEqual(prompt, prompt_before)
        self.assertEqual(output, output_before)
        with self.assertRaises(QWEN.Qwen35ResponseAdapterError):
            QWEN.slice_generated_token_ids(prompt, [10, 99, 30, 40])
        with self.assertRaises(QWEN.Qwen35ResponseAdapterError):
            QWEN.slice_generated_token_ids(prompt, [10, 20])

    def test_exact_empty_prefix_is_removed_once(self) -> None:
        raw = QWEN.NON_THINKING_PREFIX * 2 + "answer\n"
        adapted = QWEN.adapt_qwen35_text(raw)
        self.assertEqual(adapted["final_text"], QWEN.NON_THINKING_PREFIX + "answer\n")
        self.assertEqual(
            adapted["operations"], ["remove_one_exact_qwen35_non_thinking_prefix"]
        )

    def test_near_matches_and_embedded_tags_remain_byte_exact(self) -> None:
        values = (
            " " + QWEN.NON_THINKING_PREFIX + "answer",
            "<think>reason</think>\nanswer",
            "<think>\n\nnot closed",
            "answer " + QWEN.NON_THINKING_PREFIX,
            "    return '<think></think>'\n",
        )
        for value in values:
            with self.subTest(value=value):
                adapted = QWEN.adapt_qwen35_text(value)
                self.assertEqual(adapted["final_text"], value)
                self.assertEqual(adapted["operations"], [])

    def test_generated_token_decode_is_authoritative(self) -> None:
        generated = "answer\n"
        reconstructed = QWEN.NON_THINKING_PREFIX + generated
        adapted = QWEN.adapt_qwen35_text(
            reconstructed, generated_only_text=generated
        )
        self.assertEqual(adapted["final_text"], generated)
        self.assertEqual(adapted["prefix_status"], "verified_reconstructed")
        with self.assertRaises(QWEN.Qwen35ResponseAdapterError):
            QWEN.adapt_qwen35_text(reconstructed + "drift", generated_only_text=generated)

    def test_model_generated_prefix_is_not_silently_removed(self) -> None:
        generated = QWEN.NON_THINKING_PREFIX + "answer"
        message = QWEN.NON_THINKING_PREFIX + generated
        adapted = QWEN.adapt_qwen35_text(message, generated_only_text=generated)
        self.assertEqual(adapted["final_text"], generated)
        self.assertTrue(adapted["suspicious_thinking_prefix"])

    def test_response_capture_is_complete_and_fail_closed(self) -> None:
        capture, generation = self.response_capture()
        self.assertEqual(
            QWEN.validate_ms_swift_response_capture(
                capture,
                message_content=QWEN.NON_THINKING_PREFIX + "answer\n",
                generation=generation,
            ),
            "answer\n",
        )
        for mutation in ("generated_only_text", "generated_token_ids_sha256"):
            with self.subTest(mutation=mutation):
                broken = copy.deepcopy(capture)
                broken.pop(mutation)
                with self.assertRaises((QWEN.Qwen35ResponseAdapterError, TypeError)):
                    QWEN.validate_ms_swift_response_capture(
                        broken,
                        message_content=QWEN.NON_THINKING_PREFIX + "answer\n",
                        generation=generation,
                    )
        broken_generation = {"prompt_tokens": 2, "completion_tokens": 99}
        with self.assertRaises(QWEN.Qwen35ResponseAdapterError):
            QWEN.validate_ms_swift_response_capture(
                capture,
                message_content=QWEN.NON_THINKING_PREFIX + "answer\n",
                generation=broken_generation,
            )
        wrong_prefix_status = copy.deepcopy(capture)
        wrong_prefix_status["response_adapter_verification"]["prefix_status"] = "drift"
        with self.assertRaises(QWEN.Qwen35ResponseAdapterError):
            QWEN.validate_ms_swift_response_capture(
                wrong_prefix_status,
                message_content=QWEN.NON_THINKING_PREFIX + "answer\n",
                generation=generation,
            )
        boolean_counts = {"prompt_tokens": True, "completion_tokens": 2}
        with self.assertRaises(QWEN.Qwen35ResponseAdapterError):
            QWEN.validate_ms_swift_response_capture(
                capture,
                message_content=QWEN.NON_THINKING_PREFIX + "answer\n",
                generation=boolean_counts,
            )


class HumanEvalAdapterTests(unittest.TestCase):
    def classify(self, value: str, entry_point: str = "target"):
        return CODE.classify_code_candidate(
            value,
            entry_point=entry_point,
            extract_code_completion=SCORERS.extract_code_completion,
        )

    def test_markdown_solution_preserves_complete_program(self) -> None:
        raw = "Explanation\n```python\nimport math\n\ndef helper(x):\n    return x\n\ndef target(x):\n    return helper(x)\n```\n"
        result = self.classify(raw)
        self.assertEqual(result["candidate_mode"], "solution")
        self.assertTrue(result["markdown_fence_removed"])
        self.assertEqual(result["top_level_functions"], ["helper", "target"])
        self.assertEqual(result["entry_point_definition_count"], 1)
        self.assertTrue(result["execution_eligible"])
        self.assertNotIn("```", result["candidate"])

    def test_continuation_keeps_indentation(self) -> None:
        candidate = "    value = x + 1\n    return value"
        result = self.classify(candidate)
        self.assertEqual(result["candidate_mode"], "completion")
        self.assertEqual(result["candidate"], candidate)
        self.assertFalse(result["standalone_syntax_valid"])
        self.assertTrue(result["completion_contained"])
        self.assertTrue(result["execution_eligible"])
        self.assertEqual(result["anomalies"], [])

    def test_module_assignment_is_a_rejected_solution(self) -> None:
        result = self.classify("target = lambda x: x + 1\n")
        self.assertEqual(result["candidate_mode"], "solution")
        self.assertFalse(result["execution_eligible"])
        self.assertIn("solution_missing_expected_entry_point", result["anomalies"])

    def test_completion_cannot_dedent_into_module_scope(self) -> None:
        result = self.classify("    value = x + 1\ntarget = lambda x: value\n")
        self.assertEqual(result["candidate_mode"], "completion")
        self.assertFalse(result["completion_contained"])
        self.assertFalse(result["execution_eligible"])
        self.assertIn("completion_escapes_function_body", result["anomalies"])

    def test_wrong_function_name_stays_solution_and_is_not_repaired(self) -> None:
        candidate = "def wrong(x):\n    return x\n"
        result = self.classify(candidate)
        self.assertEqual(result["candidate_mode"], "solution")
        self.assertEqual(result["entry_point_definition_count"], 0)
        self.assertFalse(result["execution_eligible"])
        self.assertIn("solution_missing_expected_entry_point", result["anomalies"])
        self.assertNotIn("def target", result["candidate"])

    def test_repeated_entry_point_definition_is_not_execution_eligible(self) -> None:
        candidate = (
            "def target(x):\n"
            "    return x\n\n"
            "def target(x):\n"
            "    return x + 1\n"
        )
        result = self.classify(candidate)
        self.assertEqual(result["candidate_mode"], "solution")
        self.assertEqual(result["entry_point_definition_count"], 2)
        self.assertFalse(result["execution_eligible"])

    def test_malformed_full_function_stays_solution(self) -> None:
        result = self.classify("def target(x):\nreturn x")
        self.assertEqual(result["candidate_mode"], "solution")
        self.assertFalse(result["standalone_syntax_valid"])
        self.assertFalse(result["execution_eligible"])

    def test_solution_and_completion_have_distinct_exact_compositions(self) -> None:
        prompt = "def target(x):\n    \"\"\"doc\"\"\"\n"
        test = "def check(candidate):\n    assert candidate(1) == 2"
        completion = "    return x + 1"
        solution = "def target(x):\n    return x + 1"
        completion_program = CODE.compose_humaneval_program(
            source_prompt=prompt,
            source_test=test,
            entry_point="target",
            candidate=completion,
            candidate_mode="completion",
        )
        solution_program = CODE.compose_humaneval_program(
            source_prompt=prompt,
            source_test=test,
            entry_point="target",
            candidate=solution,
            candidate_mode="solution",
        )
        self.assertEqual(
            completion_program,
            prompt + completion + "\n" + test + "\ncheck(target)\n",
        )
        self.assertEqual(
            solution_program, solution + "\n" + test + "\ncheck(target)\n"
        )
        self.assertEqual(solution_program.count("def target"), 1)
        self.assertNotIn('"""doc"""', solution_program)

    def test_solution_preserves_import_and_helper_preamble(self) -> None:
        prompt = (
            "from typing import List\n\n"
            "def helper(x):\n"
            "    return x + 1\n\n"
            "def target(values: List[int]):\n"
            "    \"\"\"doc\"\"\"\n"
        )
        solution = (
            "def target(values: List[int]):\n"
            "    return [helper(value) for value in values]"
        )
        test = "def check(candidate):\n    assert candidate([1]) == [2]"
        program = CODE.compose_humaneval_program(
            source_prompt=prompt,
            source_test=test,
            entry_point="target",
            candidate=solution,
            candidate_mode="solution",
        )
        self.assertTrue(program.startswith("from typing import List\n\ndef helper"))
        self.assertEqual(program.count("def helper"), 1)
        self.assertEqual(program.count("def target"), 1)
        self.assertNotIn('"""doc"""', program)

    def test_static_completion_restores_manifest_prompt_newline(self) -> None:
        candidate = {
            "candidate_mode": "completion",
            "candidate": "    return x + 1",
        }
        status = RESCORE._code_static_status(
            'def target(x):\n    """doc"""', "target", candidate
        )
        self.assertEqual(status, {"valid": True, "error": None})

    def test_static_solution_checks_actual_preamble_composition(self) -> None:
        candidate = {
            "candidate_mode": "solution",
            "candidate": (
                "from __future__ import annotations\n\n"
                "def target(x):\n"
                "    return x\n"
            ),
        }
        status = RESCORE._code_static_status(
            "import math\n\ndef target(x):\n    \"\"\"doc\"\"\"",
            "target",
            candidate,
        )
        self.assertEqual(status, {"valid": False, "error": "SyntaxError"})
        raw = candidate["candidate"]
        record = {
            "sample_id": "future-import",
            "slice": "code",
            "reference": "    return x",
            "reference_hash": "reference",
            "messages_hash": "messages",
            "raw_prompt": "import math\n\ndef target(x):\n    \"\"\"doc\"\"\"",
            "metadata": {"entry_point": "target"},
        }
        sidecar = RESCORE.build_sidecar_rows(
            source_rows=[
                {
                    "schema_version": 1,
                    "ordinal": 1,
                    "raw_output": raw,
                    "raw_output_sha256": QWEN.text_sha256(raw),
                    "scorer_result": SCORERS.score_prediction(
                        "code", raw, record["reference"]
                    ),
                }
            ],
            records=[record],
            recipe="baseline-b",
            source_prediction_sha256="prediction",
            source_summary_sha256="summary",
            scorers=SCORERS,
        )[0]
        self.assertTrue(sidecar["code_candidate"]["execution_eligible"])
        self.assertFalse(sidecar["sandbox_execution_eligible"])
        self.assertFalse(sidecar["format_compliant"])

    def test_sidecar_builder_does_not_mutate_legacy_row(self) -> None:
        raw = QWEN.NON_THINKING_PREFIX + "```python\ndef target(x):\n    return x\n```"
        record = {
            "sample_id": "sample",
            "slice": "code",
            "reference": "    return x",
            "reference_hash": "reference-hash",
            "messages_hash": "messages-hash",
            "raw_prompt": 'def target(x):\n    """doc"""',
            "metadata": {"entry_point": "target"},
        }
        legacy_score = SCORERS.score_prediction("code", raw, record["reference"])
        source_row = {
            "schema_version": 1,
            "ordinal": 1,
            "raw_output": raw,
            "raw_output_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "scorer_result": legacy_score,
            "format_compliant": False,
            "anomalies": ["code_syntax_invalid"],
            "generation": {"completion_tokens": 10},
        }
        before = copy.deepcopy(source_row)
        rows = RESCORE.build_sidecar_rows(
            source_rows=[source_row],
            records=[record],
            recipe="baseline-b",
            source_prediction_sha256="source-prediction-hash",
            source_summary_sha256="source-summary-hash",
            scorers=SCORERS,
        )
        self.assertEqual(source_row, before)
        self.assertEqual(rows[0]["raw_output"], raw)
        self.assertEqual(rows[0]["source_prediction"]["legacy_scorer_result"], legacy_score)
        self.assertEqual(rows[0]["code_candidate"]["candidate_mode"], "solution")
        self.assertTrue(rows[0]["format_compliant"])

    def test_source_verifier_rejects_raw_output_tampering(self) -> None:
        manifest_path = HERE.parent / "artifacts" / "eval" / "day10-frozen-eval-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        order = manifest["header"]["evaluation_order"]["dev"]
        records = {row["sample_id"]: row for row in manifest["records"]}
        outputs = {
            "general": "A",
            "math": "Final answer: 0",
            "code": "    pass",
            "finance": "Final answer: 0",
        }
        rows = []
        for ordinal, sample_id in enumerate(order, 1):
            record = records[sample_id]
            raw = outputs[record["slice"]]
            rows.append(
                {
                    "schema_version": 1,
                    "ordinal": ordinal,
                    "recipe": "baseline-b",
                    "sample_id": sample_id,
                    "slice": record["slice"],
                    "reference_hash": record["reference_hash"],
                    "prompt_messages_hash": record["messages_hash"],
                    "raw_output": raw,
                    "raw_output_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                    "scorer_result": RESCORE.normalized_json(
                        SCORERS.score_prediction(
                            record["slice"], raw, record["reference"]
                        )
                    ),
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            predictions_path = Path(directory) / "predictions.jsonl"
            predictions_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            summary = {
                "schema_version": 1,
                "domain": "day19.qwen35_eval_summary",
                "status": "complete_with_code_sandbox_required",
                "recipe": "baseline-b",
                "checkpoint": "/checkpoint/b",
                "model_export": {
                    "model_class": "Qwen3_5ForConditionalGeneration",
                    "files": {
                        "config.json": {"bytes": 1, "sha256": "a" * 64},
                        "model.safetensors": {"bytes": 2, "sha256": "b" * 64},
                    },
                    "load_and_generation_verified": True,
                },
                "protocol": {
                    "eval_manifest_file_sha256": RESCORE.file_sha256(manifest_path),
                    "scorer_source_sha256": RESCORE.file_sha256(SCORERS_PATH),
                    "split": "dev",
                    "frozen_test_consumed": False,
                    "records": 112,
                    "sample_limit": None,
                    "template": "qwen3_5",
                    "enable_thinking": False,
                    "add_non_thinking_prefix": True,
                },
                "predictions": {
                    "records": 112,
                    "file_sha256": RESCORE.file_sha256(predictions_path),
                },
            }
            summary["model_export"]["snapshot_sha256"] = RESCORE.object_sha256(
                summary["model_export"]["files"]
            )
            summary["summary_sha256"] = RESCORE.object_sha256(summary)
            recipe, verified = RESCORE.verify_source_predictions(
                rows=rows,
                summary=summary,
                manifest=manifest,
                scorers=SCORERS,
                predictions_path=predictions_path,
                manifest_path=manifest_path,
                scorers_path=SCORERS_PATH,
            )
            self.assertEqual(recipe, "baseline-b")
            self.assertEqual(len(verified), 112)
            tampered = copy.deepcopy(rows)
            tampered[0]["raw_output_sha256"] = "0" * 64
            with self.assertRaises(RESCORE.Day19Qwen35RescoreError):
                RESCORE.verify_source_predictions(
                    rows=tampered,
                    summary=summary,
                    manifest=manifest,
                    scorers=SCORERS,
                    predictions_path=predictions_path,
                    manifest_path=manifest_path,
                    scorers_path=SCORERS_PATH,
                )
            partial_capture_claim = copy.deepcopy(summary)
            partial_capture_claim["protocol"]["response_capture"] = (
                "ms-swift RequestConfig(return_details=True)"
            )
            partial_capture_claim["summary_sha256"] = RESCORE.object_sha256(
                {
                    key: value
                    for key, value in partial_capture_claim.items()
                    if key != "summary_sha256"
                }
            )
            with self.assertRaises(RESCORE.Day19Qwen35RescoreError):
                RESCORE.verify_source_predictions(
                    rows=rows,
                    summary=partial_capture_claim,
                    manifest=manifest,
                    scorers=SCORERS,
                    predictions_path=predictions_path,
                    manifest_path=manifest_path,
                    scorers_path=SCORERS_PATH,
                )
            tampered_summary = copy.deepcopy(summary)
            tampered_summary["model_export"]["snapshot_sha256"] = "other-model"
            tampered_summary["summary_sha256"] = RESCORE.object_sha256(
                {
                    key: value
                    for key, value in tampered_summary.items()
                    if key != "summary_sha256"
                }
            )
            with self.assertRaises(RESCORE.Day19Qwen35RescoreError):
                RESCORE.verify_source_predictions(
                    rows=rows,
                    summary=tampered_summary,
                    manifest=manifest,
                    scorers=SCORERS,
                    predictions_path=predictions_path,
                    manifest_path=manifest_path,
                    scorers_path=SCORERS_PATH,
                )

    def test_adapter_run_hash_binds_model_identity(self) -> None:
        common = {
            "recipe": "baseline-b",
            "comparison_key": "comparison",
            "checkpoint": "/checkpoint/b",
            "source_prediction_file_sha256": "predictions",
            "source_evaluation_summary_file_sha256": "summary-file",
            "source_evaluation_summary_content_sha256": "summary-content",
        }
        first = RESCORE.adapter_run_hash(
            model_snapshot_sha256="model-one", **common
        )
        second = RESCORE.adapter_run_hash(
            model_snapshot_sha256="model-two", **common
        )
        self.assertNotEqual(first, second)

    def test_v2_verifier_rejects_resigned_model_tampering(self) -> None:
        manifest_path = (
            HERE.parent / "artifacts" / "eval" / "day10-frozen-eval-manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        order = manifest["header"]["evaluation_order"]["dev"]
        records_by_id = {row["sample_id"]: row for row in manifest["records"]}
        records = [records_by_id[sample_id] for sample_id in order]
        outputs = {
            "general": "A",
            "math": "Final answer: 0",
            "code": "    pass",
            "finance": "Final answer: 0",
        }
        source_rows = []
        for ordinal, record in enumerate(records, 1):
            raw = outputs[record["slice"]]
            source_rows.append(
                {
                    "schema_version": 1,
                    "ordinal": ordinal,
                    "raw_output": raw,
                    "raw_output_sha256": QWEN.text_sha256(raw),
                    "scorer_result": RESCORE.normalized_json(
                        SCORERS.score_prediction(
                            record["slice"], raw, record["reference"]
                        )
                    ),
                }
            )
        source_prediction_sha = "c" * 64
        source_summary_file_sha = "d" * 64
        source_summary_content_sha = "e" * 64
        rows = RESCORE.build_sidecar_rows(
            source_rows=source_rows,
            records=records,
            recipe="baseline-b",
            source_prediction_sha256=source_prediction_sha,
            source_summary_sha256=source_summary_file_sha,
            scorers=SCORERS,
        )
        manifest_sha = RESCORE.file_sha256(manifest_path)
        scorer_sha = RESCORE.file_sha256(SCORERS_PATH)
        comparison_key = RESCORE.adapter_comparison_key(
            manifest_file_sha256=manifest_sha,
            scorer_file_sha256=scorer_sha,
        )
        model_export = {
            "model_class": "Qwen3_5ForConditionalGeneration",
            "files": {
                "config.json": {"bytes": 1, "sha256": "a" * 64},
                "model.safetensors": {"bytes": 2, "sha256": "b" * 64},
            },
            "load_and_generation_verified": True,
        }
        model_export["snapshot_sha256"] = RESCORE.object_sha256(
            model_export["files"]
        )
        run_hash = RESCORE.adapter_run_hash(
            recipe="baseline-b",
            comparison_key=comparison_key,
            checkpoint="/checkpoint/b",
            model_snapshot_sha256=model_export["snapshot_sha256"],
            source_prediction_file_sha256=source_prediction_sha,
            source_evaluation_summary_file_sha256=source_summary_file_sha,
            source_evaluation_summary_content_sha256=source_summary_content_sha,
        )
        for row in rows:
            row["comparison_key"] = comparison_key
            row["run_hash"] = run_hash
            row["row_sha256"] = RESCORE.object_sha256(
                {key: value for key, value in row.items() if key != "row_sha256"}
            )
        with tempfile.TemporaryDirectory() as directory:
            predictions_path = Path(directory) / "v2.predictions.jsonl"
            predictions_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            summary = {
                "schema_version": 2,
                "domain": "day19.qwen35_adapter_summary",
                "status": "complete_with_code_sandbox_required",
                "recipe": "baseline-b",
                "checkpoint": "/checkpoint/b",
                "model_export": model_export,
                "comparison_key": comparison_key,
                "run_hash": run_hash,
                "protocol": {
                    "response_adapter_version": QWEN.ADAPTER_VERSION,
                    "response_adapter_file_sha256": RESCORE.file_sha256(
                        HERE / "qwen35_response_adapter.py"
                    ),
                    "code_parser_version": CODE.PARSER_VERSION,
                    "code_composer_version": CODE.COMPOSER_VERSION,
                    "code_adapter_file_sha256": RESCORE.file_sha256(
                        HERE / "day19_humaneval_adapter.py"
                    ),
                    "rescorer_file_sha256": RESCORE.file_sha256(
                        HERE / "rescore_day19_qwen35_v2.py"
                    ),
                    "manifest_file_sha256": manifest_sha,
                    "scorer_file_sha256": scorer_sha,
                    "source_prediction_file_sha256": source_prediction_sha,
                    "source_evaluation_summary_sha256": source_summary_file_sha,
                    "source_evaluation_summary_content_sha256": (
                        source_summary_content_sha
                    ),
                    "source_evaluation_protocol": {
                        "template": "qwen3_5",
                        "enable_thinking": False,
                        "add_non_thinking_prefix": True,
                        "response_capture": None,
                        "response_boundary_adapter": None,
                        "generated_token_ids_retained": None,
                        "ms_swift_version": None,
                        "non_thinking_prefix_sha256": None,
                    },
                },
                "predictions": {
                    "records": 112,
                    "file_sha256": RESCORE.file_sha256(predictions_path),
                },
            }
            summary["summary_sha256"] = SCORE_V2.object_sha256(summary)
            summary_path = Path(directory) / "v2.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            verified_rows, _ = SCORE_V2.verify_v2_predictions(
                predictions_path=predictions_path,
                summary_path=summary_path,
                manifest=manifest,
                manifest_file_sha256=manifest_sha,
                frozen_scorers=SCORERS,
            )
            self.assertEqual(len(verified_rows), 112)

            tampered = copy.deepcopy(summary)
            tampered["model_export"]["files"]["config.json"]["sha256"] = "f" * 64
            tampered["model_export"]["snapshot_sha256"] = RESCORE.object_sha256(
                tampered["model_export"]["files"]
            )
            tampered["summary_sha256"] = SCORE_V2.object_sha256(
                {
                    key: value
                    for key, value in tampered.items()
                    if key != "summary_sha256"
                }
            )
            tampered_path = Path(directory) / "v2-tampered.json"
            tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaises(SCORE_V2.Day19CodeSandboxV2Error):
                SCORE_V2.verify_v2_predictions(
                    predictions_path=predictions_path,
                    summary_path=tampered_path,
                    manifest=manifest,
                    manifest_file_sha256=manifest_sha,
                    frozen_scorers=SCORERS,
                )

    def test_model_export_identity_requires_real_model_files(self) -> None:
        with self.assertRaises(RESCORE.Day19Qwen35RescoreError):
            RESCORE.verify_model_export_identity(
                {
                    "model_class": "Qwen3_5ForConditionalGeneration",
                    "files": {},
                    "snapshot_sha256": RESCORE.object_sha256({}),
                    "load_and_generation_verified": True,
                }
            )

    def test_ineligible_candidate_is_never_sent_to_e2b(self) -> None:
        class FrozenStub:
            execute_calls = 0
            finalize_calls = 0

            @classmethod
            def execute_task(cls, task, context, bindings):
                cls.execute_calls += 1
                raise AssertionError("ineligible candidate reached executor")

            @classmethod
            def _finalize_result_row(cls, task, context, **kwargs):
                cls.finalize_calls += 1
                return {
                    "sample_id": task["sample_id"],
                    "execution_status": kwargs["execution_status"],
                    "score_status": kwargs["score_status"],
                    "score": kwargs["score"],
                }

        results = SCORE_V2.score_v2_tasks(
            tasks=[
                {
                    "sample_id": "code-sample",
                    "candidate_metadata": {"execution_eligible": False},
                    "execution_eligible": False,
                }
            ],
            context={},
            bindings={},
            frozen=FrozenStub,
        )
        self.assertEqual(FrozenStub.execute_calls, 0)
        self.assertEqual(FrozenStub.finalize_calls, 1)
        self.assertEqual(results[0]["execution_status"], "rejected_candidate_contract")
        self.assertEqual(results[0]["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
