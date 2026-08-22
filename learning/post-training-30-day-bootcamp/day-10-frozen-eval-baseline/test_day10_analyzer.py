import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("analyze_day10_baseline.py")
SPEC = importlib.util.spec_from_file_location("day10_analyzer", MODULE_PATH)
ANALYZER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYZER)


def make_manifest_record(
    sample_id, slice_name, ordinal, *, include_code_metadata=False
):
    rendered_prompt = f"rendered prompt {sample_id}"
    reference = f"reference {sample_id}"
    input_ids = [1, ordinal + 10]
    record = {
        "sample_id": sample_id,
        "evaluation_split": "dev",
        "slice": slice_name,
        "rendered_prompt": rendered_prompt,
        "rendered_prompt_hash": ANALYZER.exact_text_hash(rendered_prompt),
        "reference": reference,
        "reference_hash": ANALYZER.exact_text_hash(reference),
        "input_ids": input_ids,
        "input_ids_hash": ANALYZER.semantic_hash(
            {
                "domain": "day10.input_ids",
                "schema_version": 1,
                "input_ids": input_ids,
            }
        ),
        "input_token_count": len(input_ids),
        "generation_max_new_tokens": 2,
        "extractor_version": f"{slice_name}_extractor_v1",
        "scorer_version": f"{slice_name}_scorer_v1",
    }
    if slice_name == "code" and include_code_metadata:
        raw_prompt = f"def fixture_{ordinal}(value):\n"
        record.update(
            {
                "raw_prompt": raw_prompt,
                "raw_prompt_hash": ANALYZER.exact_text_hash(raw_prompt),
                "metadata": {
                    "entry_point": f"fixture_{ordinal}",
                    "test_sha256": hashlib.sha256(
                        f"test {ordinal}".encode("utf-8")
                    ).hexdigest(),
                },
                "source_lineage": {
                    "parent_id": f"HumanEval/{ordinal}",
                    "revision": "6d43fb980f9fee3c892a914eda09951f772ad10d",
                    "source": "openai/human-eval",
                    "source_file": (
                        "tmp/day09-eval-sources/human-eval/data/HumanEval.jsonl.gz"
                    ),
                    "source_file_sha256": (
                        "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"
                    ),
                    "source_split": "test",
                },
            }
        )
    return record


def make_manifest(*, include_code_metadata=False):
    records = []
    dev_order = []
    for slice_name in ANALYZER.SLICES:
        for index in range(ANALYZER.EXPECTED_RECORDS_PER_SLICE):
            sample_id = f"dev:{slice_name}:{index:02d}"
            dev_order.append(sample_id)
            records.append(
                make_manifest_record(
                    sample_id,
                    slice_name,
                    len(records),
                    include_code_metadata=include_code_metadata,
                )
            )
    manifest = {
        "header": {
            "domain": "day10.frozen_eval_manifest",
            "schema_version": 1,
            "status": "frozen_manifest_pre_baseline",
            "config_file_sha256": "1" * 64,
            "dataset_context_hash": "sha256:" + "2" * 64,
            "eval_suite_hash": "sha256:" + "3" * 64,
            "protocol_hash": "sha256:" + "4" * 64,
            "scorer_registry_hash": "sha256:" + "5" * 64,
            "environment_contract_sha256": "6" * 64,
            "environment_snapshot_sha256": "7" * 64,
            "model_and_rendering": {
                "model_id": "mock/base-model",
                "model_revision": "frozen-revision",
                "maximum_input_tokens": 64,
                "model_files": {
                    "config.json": "a" * 64,
                    "model.safetensors": "b" * 64,
                },
            },
            "generation": {
                "version": "day10_greedy_generation_v1",
                "do_sample": False,
                "num_beams": 1,
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "repetition_penalty": 1,
                "eos_token_id": 9,
                "pad_token_id": 0,
                "stop_token_ids": [9],
                "seed": 20260805,
                "max_new_tokens_by_slice": {
                    slice_name: 2 for slice_name in ANALYZER.SLICES
                },
            },
            "evaluation_order": {"dev": dev_order, "frozen_test": []},
            "counts": {
                "by_split": {"dev": 112, "frozen_test": 0},
                "by_slice_and_split": {
                    slice_name: {"dev": 28, "frozen_test": 0}
                    for slice_name in ANALYZER.SLICES
                },
                "total": 112,
            },
        },
        "records": records,
    }
    manifest["header"]["manifest_hash"] = ANALYZER.semantic_hash(manifest)
    return manifest


def scorer_result(slice_name, index):
    if slice_name == "code":
        return {
            "parse_status": "ok",
            "score_status": "sandbox_required",
            "score": None,
            "error_type": "sandbox_required",
            "parsed_answer": f"completion {index}",
        }
    if slice_name == "math" and index % 2:
        return {
            "parse_status": "parse_error",
            "score_status": "ok",
            "score": 0.0,
            "error_type": "parse_error",
        }
    correct = slice_name == "finance" or index % 2 == 0
    return {
        "parse_status": "ok",
        "score_status": "ok",
        "score": 1.0 if correct else 0.0,
        "error_type": None if correct else "wrong_answer",
    }


def build_fixture(
    root,
    *,
    run_started_at="2026-08-05T00:00:00+00:00",
    include_code_metadata=False,
):
    manifest = make_manifest(include_code_metadata=include_code_metadata)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    manifest_file_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    runtime = {
        "device": "cpu",
        "model_dtype": "torch.float32",
        "torch_version": "test",
        "transformers_version": "test",
        "python_version": "3.11",
        "python_implementation": "CPython",
        "platform": "test-platform",
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
    }
    execution_protocol = {
        "domain": "day10.execution_protocol",
        "schema_version": 1,
        "runner_source_sha256": "8" * 64,
        "environment_contract_sha256": manifest["header"][
            "environment_contract_sha256"
        ],
        "environment_snapshot_sha256": manifest["header"][
            "environment_snapshot_sha256"
        ],
        "local_files_only": True,
        "trust_remote_code": False,
        **runtime,
    }
    execution_protocol_hash = ANALYZER.semantic_hash(execution_protocol)
    comparison_key = ANALYZER.semantic_hash(
        {
            "domain": "day10.checkpoint_comparison",
            "schema_version": 1,
            "dataset_context_hash": manifest["header"]["dataset_context_hash"],
            "eval_suite_hash": manifest["header"]["eval_suite_hash"],
            "protocol_hash": manifest["header"]["protocol_hash"],
            "scorer_registry_hash": manifest["header"]["scorer_registry_hash"],
            "execution_protocol_hash": execution_protocol_hash,
        }
    )
    model_files = {
        "config.json": {"sha256": "a" * 64, "bytes": 10},
        "model.safetensors": {"sha256": "b" * 64, "bytes": 20},
    }
    model_snapshot_hash = ANALYZER.semantic_hash(
        {
            "domain": "day10.model_snapshot",
            "schema_version": 1,
            "files": model_files,
        }
    )
    selection_policy = {
        "version": "day10_frozen_dev_evaluation_order_v1",
        "evaluation_split": "dev",
        "sample_limit": None,
    }
    run_contract = {
        "domain": "day10.base_inference_run",
        "schema_version": 1,
        "manifest_hash": manifest["header"]["manifest_hash"],
        "config_file_sha256": manifest["header"]["config_file_sha256"],
        "protocol_hash": manifest["header"]["protocol_hash"],
        "scorer_registry_hash": manifest["header"]["scorer_registry_hash"],
        "execution_protocol_hash": execution_protocol_hash,
        "comparison_key": comparison_key,
        "model_id": "mock/base-model",
        "model_revision": "frozen-revision",
        "model_snapshot_source": "manifest",
        "model_snapshot_hash": model_snapshot_hash,
        "evaluation_split": "dev",
        "selection_policy": selection_policy,
        "sample_ids": manifest["header"]["evaluation_order"]["dev"],
        "generation": manifest["header"]["generation"],
        "runtime": runtime,
    }
    run_hash = ANALYZER.semantic_hash(run_contract)

    by_id = {record["sample_id"]: record for record in manifest["records"]}
    predictions = []
    for ordinal, sample_id in enumerate(manifest["header"]["evaluation_order"]["dev"]):
        source = by_id[sample_id]
        index = int(sample_id.rsplit(":", 1)[1])
        output_token_ids = [100 + ordinal]
        if index == 0:
            output_token_ids.append(200 + ordinal)
        raw_output = f"answer {sample_id}"
        predictions.append(
            {
                "schema_version": 1,
                "run_hash": run_hash,
                "run_started_at_utc": run_started_at,
                "run_ordinal": ordinal,
                "manifest_hash": manifest["header"]["manifest_hash"],
                "manifest_file_sha256": manifest_file_sha256,
                "config_file_sha256": manifest["header"]["config_file_sha256"],
                "protocol_hash": manifest["header"]["protocol_hash"],
                "scorer_registry_hash": manifest["header"]["scorer_registry_hash"],
                "execution_protocol": execution_protocol,
                "execution_protocol_hash": execution_protocol_hash,
                "comparison_key": comparison_key,
                "model_id": "mock/base-model",
                "model_revision": "frozen-revision",
                "model_snapshot_path": "/mock/model",
                "model_snapshot_source": "manifest",
                "model_snapshot_hash": model_snapshot_hash,
                "model_files": model_files,
                "runtime": runtime,
                "sample_id": sample_id,
                "evaluation_split": "dev",
                "selection_policy": selection_policy,
                "slice": source["slice"],
                "rendered_prompt_hash": source["rendered_prompt_hash"],
                "input_ids_hash": source["input_ids_hash"],
                "reference_hash": source["reference_hash"],
                "input_token_count": source["input_token_count"],
                "generation": {
                    "do_sample": False,
                    "num_beams": 1,
                    "max_new_tokens": 2,
                    "eos_token_id": 9,
                    "pad_token_id": 0,
                    "repetition_penalty": 1,
                },
                "raw_output": raw_output,
                "raw_output_hash": ANALYZER.exact_text_hash(raw_output),
                "output_token_ids": output_token_ids,
                "output_token_ids_hash": ANALYZER.semantic_hash(
                    {
                        "domain": "day10.output_token_ids",
                        "schema_version": 1,
                        "output_token_ids": output_token_ids,
                    }
                ),
                "output_token_count": len(output_token_ids),
                "latency_seconds": 1.0,
                "total_token_count": source["input_token_count"]
                + len(output_token_ids),
                "extractor_version": source["extractor_version"],
                "scorer_version": source["scorer_version"],
                "scorer_result": scorer_result(source["slice"], index),
            }
        )
    predictions_path = root / "predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
        encoding="utf-8",
    )
    return manifest_path, predictions_path, predictions


def rewrite_predictions(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def rewrite_code_results(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def build_code_results(
    root,
    manifest_path,
    predictions_path,
    predictions,
    *,
    correct=7,
    latency_seconds=1.25,
    name="code-results.jsonl",
):
    manifest = json.loads(manifest_path.read_text())
    by_id = {record["sample_id"]: record for record in manifest["records"]}
    dev_order = manifest["header"]["evaluation_order"]["dev"]
    code_order = [sample_id for sample_id in dev_order if by_id[sample_id]["slice"] == "code"]
    predictions_by_id = {row["sample_id"]: row for row in predictions}
    config_path = MODULE_PATH.with_name("day10_e2b_sandbox_config.json")
    config_payload = config_path.read_bytes()
    protocol = json.loads(config_payload)
    protocol_hash = ANALYZER.semantic_hash(protocol)
    comparison_key = predictions[0]["comparison_key"]
    run_hash = predictions[0]["run_hash"]
    complete_comparison_key = ANALYZER.semantic_hash(
        {
            "domain": "day10.complete_checkpoint_comparison",
            "schema_version": 1,
            "prediction_comparison_key": comparison_key,
            "code_execution_protocol_hash": protocol_hash,
        }
    )
    predictions_file_sha256 = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    manifest_file_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    code_run_hash = ANALYZER.semantic_hash(
        {
            "domain": "day10.code_sandbox_run",
            "schema_version": 1,
            "manifest_hash": manifest["header"]["manifest_hash"],
            "manifest_file_sha256": manifest_file_sha256,
            "predictions_file_sha256": predictions_file_sha256,
            "prediction_run_hash": run_hash,
            "code_execution_protocol_hash": protocol_hash,
            "sample_ids": code_order,
        }
    )
    empty_evidence = {
        "bytes": 0,
        "excerpt": "",
        "sha256": hashlib.sha256(b"").hexdigest(),
    }
    rows = []
    for code_ordinal, sample_id in enumerate(code_order):
        record = by_id[sample_id]
        prediction = predictions_by_id[sample_id]
        passed = code_ordinal < correct
        execution_status = "passed" if passed else "failed"
        score = 1.0 if passed else 0.0
        error_type = None if passed else "assertion_error"
        failure_message = None if passed else "fixture assertion failed"
        executor_result_sha256 = ANALYZER.semantic_hash(
            {
                "domain": "day10.code_executor_result",
                "schema_version": 1,
                "execution_status": execution_status,
                "score_status": "ok",
                "score": score,
                "error_type": error_type,
                "exit_code": 0,
                "stdout_sha256": empty_evidence["sha256"],
                "stderr_sha256": empty_evidence["sha256"],
                "failure_message": failure_message,
            }
        )
        completion_hash = ANALYZER.exact_text_hash(
            prediction["scorer_result"]["parsed_answer"]
        )
        row = {
            "domain": "day10.code_sandbox_result",
            "schema_version": 1,
            "sample_id": sample_id,
            "code_ordinal": code_ordinal,
            "prediction_run_ordinal": prediction["run_ordinal"],
            "evaluation_split": "dev",
            "slice": "code",
            "task_id": record["source_lineage"]["parent_id"],
            "entry_point": record["metadata"]["entry_point"],
            "test_sha256": record["metadata"]["test_sha256"],
            "manifest_hash": manifest["header"]["manifest_hash"],
            "manifest_file_sha256": manifest_file_sha256,
            "predictions_file_sha256": predictions_file_sha256,
            "run_hash": run_hash,
            "prediction_run_hash": run_hash,
            "comparison_key": comparison_key,
            "prediction_comparison_key": comparison_key,
            "complete_comparison_key": complete_comparison_key,
            "code_run_hash": code_run_hash,
            "model_id": prediction["model_id"],
            "model_revision": prediction["model_revision"],
            "model_snapshot_hash": prediction["model_snapshot_hash"],
            "extractor_version": prediction["extractor_version"],
            "scorer_version": prediction["scorer_version"],
            "harness_version": "day10_humaneval_e2b_v1",
            "raw_output_hash": prediction["raw_output_hash"],
            "raw_prompt_hash": record["raw_prompt_hash"],
            "reference_hash": record["reference_hash"],
            "output_token_ids_hash": prediction["output_token_ids_hash"],
            "parsed_completion_hash": completion_hash,
            "completion_hash": completion_hash,
            "composed_program_hash": ANALYZER.exact_text_hash(
                f"fixture program {sample_id}"
            ),
            "source_file_sha256": protocol["source"]["file_sha256"],
            "source_revision": protocol["source"]["revision"],
            "sandbox_contract_hash": protocol_hash,
            "code_execution_protocol": protocol,
            "code_execution_protocol_hash": protocol_hash,
            "sandbox_contract_file_sha256": hashlib.sha256(
                config_payload
            ).hexdigest(),
            "evaluator_source_sha256": protocol["evaluator_source_sha256"],
            "sandbox": {
                "backend": protocol["backend"],
                "sdk": protocol["sdk"],
                "template_id": protocol["template_id"],
                "template_runtime": protocol["template_runtime"],
                "fresh_sandbox_per_sample": True,
                "secure": True,
                "allow_internet_access": False,
                "allow_public_traffic": False,
                "deny_out": ["0.0.0.0/0"],
            },
            "latency_seconds": latency_seconds,
            "execution_status": execution_status,
            "score_status": "ok",
            "score": score,
            "passed": passed,
            "error_type": error_type,
            "exit_code": 0,
            "stdout": copy.deepcopy(empty_evidence),
            "stderr": copy.deepcopy(empty_evidence),
            "failure_message": failure_message,
            "scorer_result": {
                "parse_status": "ok",
                "score_status": "ok",
                "score": score,
                "passed": passed,
                "execution_outcome": execution_status,
                "executor_result_sha256": executor_result_sha256,
                "error_type": error_type,
            },
            "observed": {"elapsed_seconds": latency_seconds},
        }
        rows.append(row)
    output_path = root / name
    rewrite_code_results(output_path, rows)
    return output_path, rows


class Day10AnalyzerTests(unittest.TestCase):
    def test_complete_run_metrics_and_incomplete_code_macro(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path, predictions_path, _ = build_fixture(
                Path(temporary_directory)
            )
            summary = ANALYZER.analyze_baseline(manifest_path, predictions_path)

        self.assertEqual(summary["counts"], {"predictions": 112, "dev": 112, "frozen_test": 0})
        self.assertEqual(summary["per_slice"]["general"]["correct"], 14)
        self.assertEqual(summary["per_slice"]["general"]["wrong_answer"], 14)
        self.assertEqual(summary["per_slice"]["math"]["parse_error"], 14)
        self.assertEqual(summary["per_slice"]["finance"]["correct"], 28)
        self.assertEqual(summary["per_slice"]["code"]["sandbox_pending"], 28)
        self.assertEqual(summary["per_slice"]["code"]["scored"], 0)
        self.assertEqual(summary["per_slice"]["general"]["ceiling_hits"], 1)
        self.assertEqual(
            summary["aggregates"]["provisional_three_slice_macro"]["accuracy"],
            round(56 / 84, 12),
        )
        self.assertEqual(
            summary["aggregates"]["four_slice_macro"]["status"],
            "incomplete_due_to_code_sandbox",
        )
        self.assertIsNone(summary["aggregates"]["four_slice_macro"]["accuracy"])
        self.assertIn("Pilot-set sensitivity", summary["uncertainty"]["interpretation"])
        self.assertEqual(
            summary["summary_hash"],
            "sha256:bac623f6e7a35bebc62603150adf0334e71e77a80aa961cdbb46f404819befbc",
        )

    def test_complete_code_sidecar_updates_metrics_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path, predictions_path, predictions = build_fixture(
                root, include_code_metadata=True
            )
            code_results_path, _ = build_code_results(
                root,
                manifest_path,
                predictions_path,
                predictions,
                correct=7,
            )
            summary = ANALYZER.analyze_baseline(
                manifest_path, predictions_path, code_results_path
            )

        self.assertEqual(summary["status"], "complete_code_sandbox_scored")
        self.assertEqual(summary["counts"]["code_results"], 28)
        self.assertEqual(summary["per_slice"]["code"]["correct"], 7)
        self.assertEqual(summary["per_slice"]["code"]["scored"], 28)
        self.assertEqual(summary["per_slice"]["code"]["wrong_answer"], 21)
        self.assertEqual(summary["per_slice"]["code"]["sandbox_pending"], 0)
        self.assertEqual(summary["per_slice"]["code"]["accuracy"], 0.25)
        four_slice = summary["aggregates"]["four_slice_macro"]
        self.assertEqual(four_slice["status"], "complete")
        self.assertEqual(four_slice["correct"], 63)
        self.assertEqual(four_slice["scored"], 112)
        self.assertEqual(four_slice["accuracy"], 0.5625)
        self.assertIsNone(four_slice["wilson_95"])
        sandbox = summary["provenance"]["code_sandbox"]
        self.assertTrue(sandbox["results_semantic_hash"].startswith("sha256:"))
        self.assertTrue(sandbox["code_run_hash"].startswith("sha256:"))
        self.assertTrue(sandbox["complete_comparison_key"].startswith("sha256:"))
        self.assertEqual(sandbox["sdk"], {"package": "e2b", "version": "2.37.0"})

    def test_code_sidecar_rejects_incomplete_infra_and_tampering(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path, predictions_path, predictions = build_fixture(
                root, include_code_metadata=True
            )
            code_results_path, original_rows = build_code_results(
                root, manifest_path, predictions_path, predictions
            )
            cases = []

            missing = copy.deepcopy(original_rows[:-1])
            cases.append(("missing", missing, "expected 28 code results"))

            reordered = copy.deepcopy(original_rows)
            reordered[0], reordered[1] = reordered[1], reordered[0]
            cases.append(("reordered", reordered, "sample IDs/order"))

            infrastructure = copy.deepcopy(original_rows)
            infrastructure[0]["score_status"] = "infrastructure_error"
            infrastructure[0]["score"] = None
            infrastructure[0]["passed"] = None
            cases.append(("infrastructure", infrastructure, "not a valid score"))

            tampered = copy.deepcopy(original_rows)
            tampered[0]["raw_output_hash"] = "sha256:" + "0" * 64
            cases.append(("tampered", tampered, "raw_output_hash differs"))

            duplicate = copy.deepcopy(original_rows)
            duplicate[1] = copy.deepcopy(duplicate[0])
            cases.append(("duplicate", duplicate, "sample IDs/order"))

            for name, rows, error_pattern in cases:
                with self.subTest(name=name):
                    rewrite_code_results(code_results_path, rows)
                    with self.assertRaisesRegex(
                        ANALYZER.Day10AnalysisError, error_pattern
                    ):
                        ANALYZER.analyze_baseline(
                            manifest_path, predictions_path, code_results_path
                        )

    def test_code_summary_hash_excludes_sidecar_latency_and_exact_bytes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path, predictions_path, predictions = build_fixture(
                root, include_code_metadata=True
            )
            first_path, _ = build_code_results(
                root,
                manifest_path,
                predictions_path,
                predictions,
                latency_seconds=1.25,
                name="first.jsonl",
            )
            second_path, _ = build_code_results(
                root,
                manifest_path,
                predictions_path,
                predictions,
                latency_seconds=2.5,
                name="second.jsonl",
            )
            first = ANALYZER.analyze_baseline(
                manifest_path, predictions_path, first_path
            )
            second = ANALYZER.analyze_baseline(
                manifest_path, predictions_path, second_path
            )

        self.assertNotEqual(
            first["provenance"]["code_sandbox"]["results_file_sha256"],
            second["provenance"]["code_sandbox"]["results_file_sha256"],
        )
        self.assertEqual(
            first["provenance"]["code_sandbox"]["results_semantic_hash"],
            second["provenance"]["code_sandbox"]["results_semantic_hash"],
        )
        self.assertEqual(first["summary_hash"], second["summary_hash"])

    def test_summary_hash_excludes_observed_run_start(self):
        with tempfile.TemporaryDirectory() as first_directory, tempfile.TemporaryDirectory() as second_directory:
            first_manifest, first_predictions, _ = build_fixture(
                Path(first_directory), run_started_at="2026-08-05T00:00:00+00:00"
            )
            second_manifest, second_predictions, _ = build_fixture(
                Path(second_directory), run_started_at="2026-08-06T00:00:00+00:00"
            )
            first = ANALYZER.analyze_baseline(first_manifest, first_predictions)
            second = ANALYZER.analyze_baseline(second_manifest, second_predictions)

        self.assertNotEqual(
            first["provenance"]["observed_run_started_at_utc"],
            second["provenance"]["observed_run_started_at_utc"],
        )
        self.assertEqual(first["summary_hash"], second["summary_hash"])

    def test_rejects_reordered_or_frozen_predictions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path, predictions_path, rows = build_fixture(
                Path(temporary_directory)
            )
            rows[0], rows[1] = rows[1], rows[0]
            rewrite_predictions(predictions_path, rows)
            with self.assertRaisesRegex(
                ANALYZER.Day10AnalysisError, "sample IDs/order"
            ):
                ANALYZER.analyze_baseline(manifest_path, predictions_path)

            rows[0], rows[1] = rows[1], rows[0]
            rows[0]["evaluation_split"] = "frozen_test"
            rewrite_predictions(predictions_path, rows)
            with self.assertRaisesRegex(
                ANALYZER.Day10AnalysisError, "frozen_test prediction"
            ):
                ANALYZER.analyze_baseline(manifest_path, predictions_path)

    def test_rejects_hash_and_token_limit_tampering(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path, predictions_path, rows = build_fixture(
                Path(temporary_directory)
            )
            rows[0]["rendered_prompt_hash"] = "sha256:" + "0" * 64
            rewrite_predictions(predictions_path, rows)
            with self.assertRaisesRegex(
                ANALYZER.Day10AnalysisError, "rendered_prompt_hash differs"
            ):
                ANALYZER.analyze_baseline(manifest_path, predictions_path)

            _, _, rows = build_fixture(Path(temporary_directory))
            rows[0]["output_token_ids"] = [1, 2, 3]
            rows[0]["output_token_ids_hash"] = ANALYZER.semantic_hash(
                {
                    "domain": "day10.output_token_ids",
                    "schema_version": 1,
                    "output_token_ids": [1, 2, 3],
                }
            )
            rows[0]["output_token_count"] = 3
            rows[0]["total_token_count"] = rows[0]["input_token_count"] + 3
            rewrite_predictions(predictions_path, rows)
            with self.assertRaisesRegex(
                ANALYZER.Day10AnalysisError, "exceeds generation token limit"
            ):
                ANALYZER.analyze_baseline(manifest_path, predictions_path)

    def test_rejects_inconsistent_execution_hash(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path, predictions_path, rows = build_fixture(
                Path(temporary_directory)
            )
            rows[-1]["execution_protocol_hash"] = "sha256:" + "f" * 64
            rewrite_predictions(predictions_path, rows)
            with self.assertRaisesRegex(
                ANALYZER.Day10AnalysisError, "not constant.*execution_protocol_hash"
            ):
                ANALYZER.analyze_baseline(manifest_path, predictions_path)

    def test_output_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path, predictions_path, _ = build_fixture(root)
            summary = ANALYZER.analyze_baseline(manifest_path, predictions_path)
            output_path = root / "summary.json"
            ANALYZER.write_summary_atomic(summary, output_path, overwrite=False)
            original = output_path.read_bytes()
            with self.assertRaisesRegex(ANALYZER.Day10AnalysisError, "already exists"):
                ANALYZER.write_summary_atomic(summary, output_path, overwrite=False)
            self.assertEqual(output_path.read_bytes(), original)
            ANALYZER.write_summary_atomic(summary, output_path, overwrite=True)
            self.assertEqual(json.loads(output_path.read_text()), summary)

    def test_duplicate_json_keys_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path, predictions_path, _ = build_fixture(root)
            predictions_path.write_text('{"sample_id":"a","sample_id":"b"}\n')
            with self.assertRaisesRegex(
                ANALYZER.Day10AnalysisError, "duplicate JSON key"
            ):
                ANALYZER.analyze_baseline(manifest_path, predictions_path)


if __name__ == "__main__":
    unittest.main()
