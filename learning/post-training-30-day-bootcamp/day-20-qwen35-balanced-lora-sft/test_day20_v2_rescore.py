#!/usr/bin/env python3
"""Offline integration tests for the Day 20 normalized rescore v3 sidecar."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import day20_candidate_factory_v2 as factory
import qwen35_response_adapter_v3 as response_adapter
import rescore_day20_qwen35_v3 as rescore


HERE = Path(__file__).resolve().parent


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


class RescoreV3Tests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict[str, Path]:
        scorer = root / "frozen_scorer.py"
        scorer.write_text(
            "SCORER_REGISTRY_VERSION = 'fixture-scorer-v1'\n"
            "def score_prediction(skill, output, reference):\n"
            "    ok = bool(output)\n"
            "    return {'parsed_answer': output, 'canonical_answer': output, "
            "'canonical_reference': reference, 'parse_status': 'ok' if ok else "
            "'parse_error', 'score_status': 'ok', 'score': 1.0 if ok else 0.0, "
            "'error_type': None if ok else 'parse_error'}\n",
            encoding="utf-8",
        )
        evaluator = root / "evaluate_day20_v2_fixture.py"
        evaluator.write_text(
            "import json\n"
            "EVALUATOR_VERSION = 'day20-v2-raw-evaluator-v2'\n"
            "def verify_published_pair(predictions_path, summary_path, "
            "expected_scope=None, expected_candidate=None):\n"
            "    value = json.loads(summary_path.read_text(encoding='utf-8'))\n"
            "    assert value['eval_scope'] == expected_scope\n"
            "    assert value['candidate'] == expected_candidate\n"
            "    return value\n",
            encoding="utf-8",
        )

        skills = ("general", "math", "code", "finance")
        records: list[dict[str, object]] = []
        for index in range(112):
            skill = skills[index % 4]
            sample_id = f"eval:{skill}:{index:03d}"
            raw_prompt = (
                f"def solve_{index}(x):" if skill == "code" else f"Question {index}"
            )
            reference = "    return x" if skill == "code" else f"answer-{index}"
            records.append(
                {
                    "sample_id": sample_id,
                    "evaluation_split": "dev",
                    "slice": skill,
                    "raw_prompt": raw_prompt,
                    "reference": reference,
                    "reference_hash": f"sha256:{rescore.text_sha256(reference)}",
                    "messages_hash": f"sha256:{rescore.text_sha256(sample_id)}",
                    "source_lineage": {
                        "source": "fixture",
                        "revision": "fixed",
                        "parent_id": f"HumanEval/{index}",
                    },
                    "metadata": {
                        "entry_point": f"solve_{index}",
                        "test_sha256": rescore.text_sha256(f"test-{index}"),
                    },
                }
            )
        eval_manifest = {
            "header": {
                "evaluation_order": {
                    "dev": [record["sample_id"] for record in records]
                },
                "scorer_source_sha256": rescore.file_sha256(scorer),
                "manifest_hash": f"sha256:{'a' * 64}",
                "manifest_version": "day10_frozen_eval_manifest_v3",
            },
            "records": records,
        }
        eval_manifest_path = root / "eval-manifest.json"
        _write_json(eval_manifest_path, eval_manifest)

        diagnostic_ids = [record["sample_id"] for record in records[:32]]
        diagnostic = {
            "path": str(root / "diagnostic-v2.jsonl"),
            "file_sha256": "b" * 64,
            "records": 32,
            "selection_sha256": rescore.object_sha256(diagnostic_ids),
            "ordered_sample_ids": diagnostic_ids,
        }
        experiment = {
            "schema_version": 2,
            "domain": "day20.qwen35_balanced_lora.experiment_manifest.v2",
            "status": "prepared_stage_a_not_started",
            "contract": {"candidate_factory_version": factory.CONTRACT_VERSION},
            "datasets": {"diagnostic": diagnostic},
        }
        experiment["manifest_sha256"] = rescore.object_sha256(experiment)
        experiment_path = root / "DAY20-V2-MANIFEST.json"
        _write_json(experiment_path, experiment)

        candidate = factory.candidate_id(
            run_kind="probe",
            seed=factory.PRIMARY_SEED,
            learning_rate="1e-4",
            checkpoint="t6000",
        )
        candidate_identity = factory.parse_candidate_id(candidate).as_dict()
        model_identity = {"base": "c" * 64, "adapter": "d" * 64}
        checkpoint_package = {"kind": "peft_lora", "snapshot": "e" * 64}
        model_sha = rescore.object_sha256(model_identity)
        checkpoint_sha = rescore.object_sha256(checkpoint_package)
        raw_rows: list[dict[str, object]] = []
        for ordinal, record in enumerate(records[:32], 1):
            skill = str(record["slice"])
            output = (
                ("return x" if ordinal == 3 else "    return x")
                if skill == "code"
                else "Final answer: A"
            )
            generated_ids = [ordinal, ordinal + 100]
            message_content = (
                response_adapter.NON_THINKING_PREFIX + output
                if ordinal == 1
                else output
            )
            adapter_evidence = response_adapter.adapt_generated_token_decode(
                message_content,
                generated_token_ids=generated_ids,
                generated_only_text=output,
            )
            capture = {
                "method": "ms_swift_return_details",
                "message_content": message_content,
                "message_content_sha256": rescore.text_sha256(message_content),
                "prompt_token_ids": [ordinal],
                "prompt_token_ids_sha256": response_adapter.token_ids_sha256(
                    [ordinal]
                ),
                "generated_token_ids": generated_ids,
                "generated_token_ids_sha256": response_adapter.token_ids_sha256(
                    generated_ids
                ),
                "generated_only_text": output,
                "generated_only_text_sha256": rescore.text_sha256(output),
                "response_adapter": adapter_evidence,
            }
            raw_row: dict[str, object] = {
                "schema_version": 2,
                "domain": rescore.RAW_ROW_DOMAIN,
                "ordinal": ordinal,
                "candidate": candidate,
                "candidate_identity": candidate_identity,
                "eval_scope": "probe32",
                "sample_id": record["sample_id"],
                "skill": skill,
                "target_format": rescore.TARGET_FORMAT_BY_SKILL[skill],
                "raw_prompt": record["raw_prompt"],
                "raw_prompt_sha256": rescore.text_sha256(str(record["raw_prompt"])),
                "reference": record["reference"],
                "reference_sha256": rescore.text_sha256(str(record["reference"])),
                "raw_output": output,
                "raw_output_sha256": rescore.text_sha256(output),
                "response_capture": capture,
                "model_identity": {"model_identity_sha256": model_sha},
                "checkpoint_package": {
                    "checkpoint_package_sha256": checkpoint_sha
                },
                "generation": {
                    "prompt_tokens": 1,
                    "completion_tokens": len(generated_ids),
                },
                "scorer_result": {},
                "code_contract": None,
                "format_compliant": bool(output),
                "anomalies": [],
                "repeated_4gram_ratio": 0.0,
            }
            raw_row["row_sha256"] = rescore.object_sha256(raw_row)
            raw_rows.append(raw_row)
        raw_predictions = root / f"{candidate}-probe32-raw-predictions-v2.jsonl"
        _write_jsonl(raw_predictions, raw_rows)

        adapter_path = HERE / "qwen35_response_adapter_v3.py"
        protocol = {
            "candidate_factory": {
                "version": factory.CONTRACT_VERSION,
                "path": str(Path(factory.__file__).resolve()),
                "file_sha256": rescore.file_sha256(Path(factory.__file__).resolve()),
            },
            "data_contract": {
                "version": rescore.V2_CONTRACT_VERSION,
                "path": str((HERE / "day20_contract_v2.py").resolve()),
                "file_sha256": rescore.file_sha256(HERE / "day20_contract_v2.py"),
            },
            "eval_manifest": {
                "path": str(eval_manifest_path.resolve()),
                "file_sha256": rescore.file_sha256(eval_manifest_path),
                "content_sha256": "a" * 64,
                "manifest_version": "day10_frozen_eval_manifest_v3",
            },
            "experiment_manifest": {
                "path": str(experiment_path.resolve()),
                "file_sha256": rescore.file_sha256(experiment_path),
                "content_sha256": experiment["manifest_sha256"],
            },
            "diagnostic": {
                key: diagnostic[key]
                for key in ("path", "file_sha256", "records", "selection_sha256")
            },
            "scorer": {
                "path": str(scorer.resolve()),
                "file_sha256": rescore.file_sha256(scorer),
                "version": "fixture-scorer-v1",
            },
            "response_adapter": {
                "path": str(adapter_path.resolve()),
                "file_sha256": rescore.file_sha256(adapter_path),
                "version": response_adapter.ADAPTER_VERSION,
            },
            "evaluator": {
                "path": str(evaluator.resolve()),
                "file_sha256": rescore.file_sha256(evaluator),
                "version": "day20-v2-raw-evaluator-v2",
            },
            "generation": {
                "split": "dev",
                "eval_scope": "probe32",
                "records": 32,
                "template": "qwen3_5",
                "backend": "hf_transformers",
                "use_mcore_gdn": False,
                "enable_thinking": False,
                "add_non_thinking_prefix": True,
                "non_thinking_prefix_sha256": "f" * 64,
                "greedy": True,
                "seed": factory.PRIMARY_SEED,
                "response_capture": "ms-swift RequestConfig(return_details=True)",
                "generated_token_ids_retained": True,
                "adapter_loaded_unmerged": True,
                "code_execution": False,
            },
        }
        raw_summary: dict[str, object] = {
            "schema_version": 2,
            "domain": rescore.RAW_SUMMARY_DOMAIN,
            "status": "complete_with_code_sandbox_required",
            "created_at_utc": "2026-08-11T00:00:00+00:00",
            "candidate": candidate,
            "candidate_identity": candidate_identity,
            "eval_scope": "probe32",
            "checkpoint": "/fixture/checkpoint",
            "checkpoint_package": checkpoint_package,
            "checkpoint_package_sha256": checkpoint_sha,
            "model_identity": model_identity,
            "model_identity_sha256": model_sha,
            "training_identity": {},
            "protocol": protocol,
            "metrics": {},
            "runtime_metrics": {},
            "predictions": {
                "path": str(raw_predictions.resolve()),
                "records": 32,
                "file_sha256": rescore.file_sha256(raw_predictions),
                "content_sha256": rescore.object_sha256(raw_rows),
                "ordered_sample_ids_sha256": rescore.object_sha256(
                    [row["sample_id"] for row in raw_rows]
                ),
            },
        }
        raw_summary["summary_sha256"] = rescore.object_sha256(raw_summary)
        raw_summary_path = root / f"{candidate}-probe32-raw-summary-v2.json"
        _write_json(raw_summary_path, raw_summary)
        return {
            "raw_predictions": raw_predictions,
            "raw_summary": raw_summary_path,
            "eval_manifest": eval_manifest_path,
            "experiment_manifest": experiment_path,
            "scorers": scorer,
            "evaluator": evaluator,
            "adapter": adapter_path,
            "output": root / "normalized",
        }

    def test_rescore_round_trip_is_idempotent_and_never_repairs_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            kwargs = {
                "raw_predictions_path": paths["raw_predictions"],
                "raw_summary_path": paths["raw_summary"],
                "eval_manifest_path": paths["eval_manifest"],
                "experiment_manifest_path": paths["experiment_manifest"],
                "scorers_path": paths["scorers"],
                "output_dir": paths["output"],
                "scope": "probe32",
                "evaluator_path": paths["evaluator"],
                "response_adapter_path": paths["adapter"],
            }
            first = rescore.rescore_artifacts(**kwargs)
            second = rescore.rescore_artifacts(**kwargs)
            self.assertEqual(first, second)
            rows, verified = rescore.verify_normalized_pair(
                Path(first["predictions"]["path"]),
                paths["output"] / f"{first['candidate']}.qwen35-v3.json",
            )
            self.assertEqual(verified, first)
            invalid_code = next(
                row
                for row in rows
                if row["slice"] == "code"
                and row["normalized_output"] == "return x"
            )
            self.assertFalse(invalid_code["sandbox_execution_eligible"])
            self.assertEqual(invalid_code["code_candidate"]["candidate"], "return x")
            self.assertIsNone(invalid_code["code_candidate"]["canonical_sha256"])

    def test_tampered_existing_output_and_scope_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            kwargs = {
                "raw_predictions_path": paths["raw_predictions"],
                "raw_summary_path": paths["raw_summary"],
                "eval_manifest_path": paths["eval_manifest"],
                "experiment_manifest_path": paths["experiment_manifest"],
                "scorers_path": paths["scorers"],
                "output_dir": paths["output"],
                "scope": "probe32",
                "evaluator_path": paths["evaluator"],
                "response_adapter_path": paths["adapter"],
            }
            summary = rescore.rescore_artifacts(**kwargs)
            predictions = Path(summary["predictions"]["path"])
            predictions.write_text(
                predictions.read_text(encoding="utf-8") + "{}\n",
                encoding="utf-8",
            )
            with self.assertRaises(rescore.Day20Qwen35RescoreV3Error):
                rescore.rescore_artifacts(**kwargs)
            with self.assertRaises(rescore.Day20Qwen35RescoreV3Error):
                rescore.rescore_artifacts(**{**kwargs, "scope": "full112"})


if __name__ == "__main__":
    unittest.main()
