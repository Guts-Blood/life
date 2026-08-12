#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import day20_candidate_factory_v2 as factory
import evaluate_day20_v2 as evaluator
import qwen35_response_adapter_v3 as adapter


HERE = Path(__file__).resolve().parent


def _fixture_runtime_identity() -> dict:
    identity = {
        "schema_version": 2,
        "domain": "day20.qwen35_lora_runtime_identity.v2",
        "backend": "hf_transformers",
        "versions": dict(evaluator.EXPECTED_RUNTIME_VERSIONS),
        "ms_swift_root": "/fixture/ms-swift",
        "ms_swift_commit": evaluator.EXPECTED_MS_SWIFT_COMMIT,
        "ms_swift_worktree_clean": True,
        "implementation_file_sha256": {"fixture.py": "f" * 64},
    }
    identity["runtime_sha256"] = evaluator.object_sha256(identity)
    return identity


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


class _FrozenScorerStub:
    @staticmethod
    def score_prediction(skill: str, raw_output: str, reference: str) -> dict:
        if skill == "code":
            return {
                "parsed_answer": raw_output,
                "canonical_answer": None,
                "canonical_reference": None,
                "parse_status": "ok" if raw_output else "parse_error",
                "score_status": "sandbox_required",
                "score": None,
                "error_type": None if raw_output else "parse_error",
            }
        return {
            "parsed_answer": raw_output,
            "canonical_answer": raw_output,
            "canonical_reference": reference,
            "parse_status": "ok" if raw_output else "parse_error",
            "score_status": "ok",
            "score": 1.0 if raw_output == reference else 0.0,
            "error_type": None if raw_output == reference else "wrong_answer",
        }


def _record(skill: str = "code", index: int = 1) -> dict:
    return {
        "sample_id": f"eval:{skill}:fixture:{index}",
        "slice": skill,
        "raw_prompt": "def add_one(x):\n",
        "reference": "    return x + 1",
        "messages_hash": f"sha256:{'1' * 64}",
        "generation_max_new_tokens": 32,
    }


def _row(
    output: str = "    return x + 1", *, ordinal: int = 1, index: int = 1
) -> dict:
    identity = factory.parse_candidate_id("base-probe")
    return evaluator.build_prediction_row(
        ordinal=ordinal,
        candidate_identity=identity,
        eval_scope="probe32",
        record=_record(index=index),
        message_content=adapter.NON_THINKING_PREFIX + output,
        prompt_token_ids=[10, 11],
        generated_token_ids=[index + 11, index + 12, index + 13],
        generated_only_text=output,
        finish_reason="stop",
        prompt_tokens=2,
        completion_tokens=3,
        scorer=_FrozenScorerStub,
        compact_model_identity={
            "model_identity_sha256": "a" * 64,
            "base_snapshot_sha256": "b" * 64,
            "adapter_snapshot_sha256": None,
        },
        compact_checkpoint_package={
            "checkpoint_package_sha256": "c" * 64,
            "kind": "base_snapshot",
            "snapshot_sha256": "b" * 64,
            "integrity_sha256": None,
        },
    )


class ResponseAdapterV3Tests(unittest.TestCase):
    def test_generated_decode_is_exact_and_never_repaired(self) -> None:
        generated = "\n```python\n    return 1\n```\n"
        result = adapter.adapt_generated_token_decode(
            adapter.NON_THINKING_PREFIX + generated,
            generated_token_ids=[1, 2, 3],
            generated_only_text=generated,
        )
        self.assertEqual(result["final_text"], generated)
        self.assertEqual(result["operations"], [])
        self.assertFalse(result["trimming_applied"])
        self.assertFalse(result["fence_repair_applied"])
        self.assertEqual(
            result["message_content_relation"], "known_template_reconstruction"
        )

    def test_divergent_message_is_diagnostic_not_answer_boundary(self) -> None:
        result = adapter.adapt_generated_token_decode(
            "different reconstructed content",
            generated_token_ids=[7],
            generated_only_text="  exact token decode  ",
        )
        self.assertEqual(result["final_text"], "  exact token decode  ")
        self.assertEqual(
            result["message_content_relation"], "divergent_diagnostic_only"
        )

    def test_bool_or_negative_token_id_is_rejected(self) -> None:
        for token_ids in ([True], [-1]):
            with self.subTest(token_ids=token_ids), self.assertRaises(TypeError):
                adapter.adapt_generated_token_decode(
                    "x", generated_token_ids=token_ids, generated_only_text="x"
                )
        with self.assertRaises(adapter.Qwen35ResponseAdapterV3Error):
            adapter.adapt_generated_token_decode(
                "x", generated_token_ids=[], generated_only_text="x"
            )

    def test_capture_revalidation_detects_generated_text_drift(self) -> None:
        capture = evaluator.build_response_capture(
            message_content="answer",
            prompt_token_ids=[1],
            generated_token_ids=[2],
            generated_only_text="answer",
        )
        generation = {"prompt_tokens": 1, "completion_tokens": 1}
        self.assertEqual(
            adapter.validate_response_capture(
                capture, raw_output="answer", generation=generation
            ),
            "answer",
        )
        tampered = copy.deepcopy(capture)
        tampered["generated_only_text"] = "changed"
        with self.assertRaises(adapter.Qwen35ResponseAdapterV3Error):
            adapter.validate_response_capture(
                tampered, raw_output="changed", generation=generation
            )


class RawEvaluatorV2Tests(unittest.TestCase):
    def _published_base_fixture(
        self, root: Path
    ) -> tuple[Path, Path, dict[tuple[int, ...], str]]:
        run_root = root / "run"
        data_dir = run_root / "data"
        eval_dir = run_root / "eval"
        model_dir = root / "model"
        source_dir = root / "sources"
        for directory in (data_dir, eval_dir, model_dir, source_dir):
            directory.mkdir(parents=True)
        (run_root / ".day20-v2-run-root").write_text(
            "day20-qwen35-candidate-factory-v2\n", encoding="utf-8"
        )
        (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
        (model_dir / "tokenizer.json").write_text("{}\n", encoding="utf-8")
        (model_dir / "model.safetensors").write_bytes(b"fixture-weights")

        scorer_path = root / "scorer.py"
        scorer_path.write_text(
            "SCORER_REGISTRY_VERSION = 'fixture-scorer-v1'\n"
            "SCORER_REGISTRY = {'registry_version': SCORER_REGISTRY_VERSION, "
            "'slices': {"
            "'general': {'scorer_version': 'general-v1', 'extractor_version': 'general-x1'}, "
            "'math': {'scorer_version': 'math-v1', 'extractor_version': 'math-x1'}, "
            "'code': {'scorer_version': 'code-v1', 'extractor_version': 'code-x1'}, "
            "'finance': {'scorer_version': 'finance-v1', 'extractor_version': 'finance-x1'}}}\n"
            "def score_prediction(skill, output, reference):\n"
            "    if skill == 'code':\n"
            "        return {'parsed_answer': output, 'canonical_answer': None, "
            "'canonical_reference': None, 'parse_status': 'ok' if output else "
            "'parse_error', 'score_status': 'sandbox_required', 'score': None, "
            "'error_type': None if output else 'parse_error'}\n"
            "    ok = bool(output)\n"
            "    return {'parsed_answer': output, 'canonical_answer': output, "
            "'canonical_reference': reference, 'parse_status': 'ok' if ok else "
            "'parse_error', 'score_status': 'ok', 'score': 1.0 if output == "
            "reference else 0.0, 'error_type': None if output == reference else "
            "'wrong_answer'}\n",
            encoding="utf-8",
        )
        scorer_versions = {
            "general": ("general-v1", "general-x1"),
            "math": ("math-v1", "math-x1"),
            "code": ("code-v1", "code-x1"),
            "finance": ("finance-v1", "finance-x1"),
        }
        skills = ("general", "math", "code", "finance")
        records: list[dict] = []
        for index in range(160):
            skill = skills[index % 4]
            split = "dev" if index < 112 else "frozen_test"
            raw_prompt = (
                f"def solve_{index}(x):" if skill == "code" else f"Question {index}"
            )
            reference = "    return x" if skill == "code" else f"answer-{index}"
            messages = [{"role": "user", "content": raw_prompt}]
            input_ids = [index + 1, index + 2]
            scorer_version, extractor_version = scorer_versions[skill]
            records.append(
                {
                    "sample_id": f"eval:{skill}:fixture:{index:03d}",
                    "evaluation_split": split,
                    "slice": skill,
                    "raw_prompt": raw_prompt,
                    "raw_prompt_hash": f"sha256:{evaluator.text_sha256(raw_prompt)}",
                    "adapted_prompt": raw_prompt,
                    "adapted_prompt_hash": f"sha256:{evaluator.text_sha256(raw_prompt)}",
                    "messages": messages,
                    "messages_hash": "sha256:"
                    + evaluator.object_sha256(
                        {
                            "domain": "day10.messages",
                            "schema_version": 1,
                            "messages": messages,
                        }
                    ),
                    "rendered_prompt": raw_prompt,
                    "rendered_prompt_hash": f"sha256:{evaluator.text_sha256(raw_prompt)}",
                    "input_ids": input_ids,
                    "input_ids_hash": "sha256:"
                    + evaluator.object_sha256(
                        {
                            "domain": "day10.input_ids",
                            "schema_version": 1,
                            "input_ids": input_ids,
                        }
                    ),
                    "input_token_count": len(input_ids),
                    "reference": reference,
                    "reference_hash": f"sha256:{evaluator.text_sha256(reference)}",
                    "generation_max_new_tokens": 32,
                    "scorer_version": scorer_version,
                    "extractor_version": extractor_version,
                }
            )
        header: dict = {
            "schema_version": 1,
            "domain": "day10.frozen_eval_manifest",
            "manifest_version": "day10_frozen_eval_manifest_v3",
            "scorer_source_sha256": evaluator.file_sha256(scorer_path),
            "evaluation_order": {
                "dev": [row["sample_id"] for row in records[:112]]
            },
        }
        header["manifest_hash"] = "sha256:" + evaluator.object_sha256(
            {
                "header": header,
                "records": sorted(records, key=lambda row: row["sample_id"]),
            }
        )
        eval_manifest = {"header": header, "records": records}
        eval_manifest_path = root / "eval-manifest.json"
        _write_json(eval_manifest_path, eval_manifest)

        day09_path = source_dir / "day09.json"
        _write_json(day09_path, {"fixture": True})
        source_outputs: dict[str, dict] = {}
        base_sources: dict[str, dict] = {}
        for skill in skills:
            base_path = source_dir / f"{skill}.base.jsonl"
            expanded_path = source_dir / f"{skill}.expanded.jsonl"
            source_row = {
                "qwen35_tokenization": {"supervised_tokens": 1},
                "source_lineage": {"adapter": "fixture-adapter"},
            }
            _write_jsonl(base_path, [{"skill": skill}])
            _write_jsonl(expanded_path, [source_row])
            base_sources[skill] = {
                "path": str(base_path.resolve()),
                "file_sha256": evaluator.file_sha256(base_path),
            }
            source_outputs[skill] = {
                "path": str(expanded_path.resolve()),
                "file_sha256": evaluator.file_sha256(expanded_path),
                "records": 1,
                "supervised_tokens": 1,
                "records_by_adapter": {"fixture-adapter": 1},
            }
        source_manifest: dict = {
            "schema_version": 2,
            "domain": evaluator.SOURCE_EXPANSION_DOMAIN,
            "status": "pass",
            "adapter_version": "fixture-source-adapter-v2",
            "dataset_code_execution": False,
            "model_path": str(model_dir.resolve()),
            "tokenizer_files": {
                "config.json": evaluator.file_sha256(model_dir / "config.json"),
                "tokenizer.json": evaluator.file_sha256(
                    model_dir / "tokenizer.json"
                ),
            },
            "day09_manifest": {
                "path": str(day09_path.resolve()),
                "file_sha256": evaluator.file_sha256(day09_path),
            },
            "base_sources": base_sources,
            "outputs": source_outputs,
        }
        source_manifest["manifest_sha256"] = evaluator.object_sha256(
            source_manifest
        )
        source_manifest_path = source_dir / "SOURCE-EXPANSION-MANIFEST.json"
        _write_json(source_manifest_path, source_manifest)

        prepared_row = {
            "sample_id": "train:fixture:1",
            "skill": "general",
            "target_format": "general_mcq",
            "qwen35_supervised_tokens": 8,
        }

        def prepared_identity(path: Path) -> dict:
            _write_jsonl(path, [prepared_row])
            temporal_mix = {"status": "pass", "fixture": True}
            return {
                "path": str(path.resolve()),
                "file_sha256": evaluator.file_sha256(path),
                "records": 1,
                "supervised_tokens": 8,
                "supervised_tokens_by_skill": {
                    "general": 8,
                    "math": 0,
                    "code": 0,
                    "finance": 0,
                },
                "supervised_tokens_by_format": {"general_mcq": 8},
                "records_by_format": {"general_mcq": 1},
                "ordered_sample_ids": [prepared_row["sample_id"]],
                "ordered_supervised_tokens": [8],
                "selection_sha256": evaluator.object_sha256(
                    [prepared_row["sample_id"]]
                ),
                "temporal_mix_audit": temporal_mix,
                "temporal_mix_sha256": evaluator.object_sha256(temporal_mix),
            }

        diagnostic_rows = records[:32]
        diagnostic_path = data_dir / "diagnostic-v2.jsonl"
        _write_jsonl(diagnostic_path, diagnostic_rows)
        diagnostic_ids = [row["sample_id"] for row in diagnostic_rows]
        model_files = evaluator.file_manifest(model_dir)
        experiment: dict = {
            "schema_version": 2,
            "domain": evaluator.EXPERIMENT_MANIFEST_DOMAIN,
            "status": "prepared_stage_a_not_started",
            "run_root": str(run_root.resolve()),
            "contract": {
                "candidate_factory_version": factory.CONTRACT_VERSION,
                "data_contract_version": evaluator.V2_CONTRACT_VERSION,
            },
            "base_model_identity": {
                "path": str(model_dir.resolve()),
                "files": model_files,
                "snapshot_sha256": evaluator.object_sha256(model_files),
            },
            "source_expansion": {
                "path": str(source_manifest_path.resolve()),
                "file_sha256": evaluator.file_sha256(source_manifest_path),
                "content_sha256": source_manifest["manifest_sha256"],
                "adapter_version": source_manifest["adapter_version"],
            },
            "datasets": {
                "probe": prepared_identity(data_dir / "probe-v2.jsonl"),
                "main": prepared_identity(data_dir / "main-v2.jsonl"),
                "diagnostic": {
                    "path": str(diagnostic_path.resolve()),
                    "file_sha256": evaluator.file_sha256(diagnostic_path),
                    "records": 32,
                    "records_by_skill": {skill: 8 for skill in skills},
                    "ordered_sample_ids": diagnostic_ids,
                    "selection_sha256": evaluator.object_sha256(diagnostic_ids),
                    "eval_manifest_path": str(eval_manifest_path.resolve()),
                    "eval_manifest_file_sha256": evaluator.file_sha256(
                        eval_manifest_path
                    ),
                },
            },
        }
        experiment["manifest_sha256"] = evaluator.object_sha256(experiment)
        experiment_path = run_root / "DAY20-V2-MANIFEST.json"
        _write_json(experiment_path, experiment)

        frozen, scorer = evaluator.load_frozen_eval_manifest(
            eval_manifest_path, scorer_path=scorer_path
        )
        verified_experiment = evaluator.verify_experiment_manifest(
            experiment_path,
            eval_manifest_path=eval_manifest_path,
            frozen_manifest=frozen,
        )
        selected = evaluator.select_evaluation_records(
            frozen_manifest=frozen,
            experiment_manifest=verified_experiment,
            eval_scope="probe32",
        )
        identity = factory.parse_candidate_id("base-probe")
        provenance = evaluator.resolve_candidate_provenance(
            candidate_identity=identity,
            model_path=model_dir,
            adapter_path=None,
            training_summary_path=None,
            experiment_manifest=verified_experiment,
        )
        runtime_identity = _fixture_runtime_identity()
        protocol = evaluator.build_protocol(
            eval_scope="probe32",
            records=32,
            eval_manifest_path=eval_manifest_path,
            frozen_manifest=frozen,
            experiment_manifest_path=experiment_path,
            experiment_manifest=verified_experiment,
            scorer_path=scorer_path,
            scorer_version=scorer.SCORER_REGISTRY_VERSION,
            adapter_loaded=False,
            runtime_identity=runtime_identity,
        )
        rows = []
        decoded_by_ids: dict[tuple[int, ...], str] = {}
        for ordinal, record in enumerate(selected, 1):
            output = (
                "    return x" if record["slice"] == "code" else record["reference"]
            )
            decoded_by_ids[(ordinal + 2,)] = output
            rows.append(
                evaluator.build_prediction_row(
                    ordinal=ordinal,
                    candidate_identity=identity,
                    eval_scope="probe32",
                    record=record,
                    message_content=adapter.NON_THINKING_PREFIX + output,
                    prompt_token_ids=[ordinal, ordinal + 1],
                    generated_token_ids=[ordinal + 2],
                    generated_only_text=output,
                    finish_reason="stop",
                    prompt_tokens=2,
                    completion_tokens=1,
                    scorer=scorer,
                    compact_model_identity=provenance["compact_model_identity"],
                    compact_checkpoint_package=provenance[
                        "compact_checkpoint_package"
                    ],
                )
            )
        predictions_path, summary_path = evaluator.artifact_paths(
            eval_dir, candidate="base-probe", eval_scope="probe32"
        )
        _write_jsonl(predictions_path, rows)
        summary = evaluator.build_summary(
            candidate_identity=identity,
            eval_scope="probe32",
            provenance=provenance,
            protocol=protocol,
            rows=rows,
            predictions_path=predictions_path,
            predictions_file_sha256=evaluator.file_sha256(predictions_path),
            elapsed_seconds=1.0,
            peak_cuda_memory_gib=2.0,
            runtime_identity=runtime_identity,
        )
        _write_json(summary_path, summary)
        return predictions_path, summary_path, decoded_by_ids

    def test_scope_and_base_names_are_explicit(self) -> None:
        self.assertEqual(
            evaluator.validate_candidate_scope("base-probe", "probe32").scope,
            "probe32",
        )
        self.assertEqual(
            evaluator.validate_candidate_scope("base-full", "full112").scope,
            "full112",
        )
        for candidate, scope in (
            ("base", "probe32"),
            ("base-probe", "full112"),
            ("probe-1e-4", "probe32"),
        ):
            with self.subTest(candidate=candidate, scope=scope), self.assertRaises(
                evaluator.Day20EvaluationV2Error
            ):
                evaluator.validate_candidate_scope(candidate, scope)

    def test_runtime_identity_reuses_training_pin_and_fails_on_version_drift(
        self,
    ) -> None:
        runtime_identity = _fixture_runtime_identity()
        with mock.patch.object(
            evaluator,
            "training_runtime_identity",
            return_value=runtime_identity,
        ) as runtime_probe:
            self.assertEqual(evaluator.live_runtime_identity(), runtime_identity)
        runtime_probe.assert_called_once_with(evaluator.EXPECTED_MS_SWIFT_COMMIT)

        drifted = copy.deepcopy(runtime_identity)
        drifted["versions"]["torch"] = "unexpected"
        with mock.patch.object(
            evaluator,
            "training_runtime_identity",
            return_value=drifted,
        ), self.assertRaisesRegex(
            evaluator.Day20EvaluationV2Error,
            "runtime versions drifted",
        ):
            evaluator.live_runtime_identity()

    def test_code_row_uses_strict_raw_continuation_evidence(self) -> None:
        row = _row()
        self.assertEqual(row["domain"], evaluator.PREDICTION_DOMAIN)
        self.assertEqual(row["raw_output"], "    return x + 1")
        self.assertTrue(row["code_contract"]["valid"])
        self.assertTrue(row["code_contract"]["execution_eligible"])
        self.assertFalse(row["code_contract"]["execution_performed"])
        self.assertEqual(row["row_sha256"], evaluator.object_sha256({
            key: value for key, value in row.items() if key != "row_sha256"
        }))

    def test_code_fence_is_not_repaired_and_is_not_eligible(self) -> None:
        fenced = "```python\n    return x + 1\n```"
        row = _row(fenced)
        self.assertEqual(row["raw_output"], fenced)
        self.assertFalse(row["code_contract"]["valid"])
        self.assertFalse(row["code_contract"]["execution_eligible"])
        self.assertIn("strict_code_continuation_invalid", row["anomalies"])
        self.assertFalse(row["format_compliant"])

    def test_prediction_verifier_rebuilds_full_row(self) -> None:
        rows = [_row(ordinal=index, index=index) for index in range(1, 33)]
        records = [_record(index=index) for index in range(1, 33)]
        row = rows[0]
        identity = factory.parse_candidate_id("base-probe")
        evaluator.verify_prediction_rows(
            rows,
            selected_records=records,
            candidate_identity=identity,
            eval_scope="probe32",
            scorer=_FrozenScorerStub,
            compact_model_identity=row["model_identity"],
            compact_checkpoint_package=row["checkpoint_package"],
            generated_token_decoder=lambda token_ids: "    return x + 1",
        )
        tampered_rows = copy.deepcopy(rows)
        tampered_rows[0]["format_compliant"] = False
        tampered_rows[0]["row_sha256"] = evaluator.object_sha256(
            {
                key: value
                for key, value in tampered_rows[0].items()
                if key != "row_sha256"
            }
        )
        with self.assertRaises(evaluator.Day20EvaluationV2Error):
            evaluator.verify_prediction_rows(
                tampered_rows,
                selected_records=records,
                candidate_identity=identity,
                eval_scope="probe32",
                scorer=_FrozenScorerStub,
                compact_model_identity=row["model_identity"],
                compact_checkpoint_package=row["checkpoint_package"],
                generated_token_decoder=lambda token_ids: "    return x + 1",
            )

    def test_prediction_verifier_redecodes_generated_token_ids(self) -> None:
        rows = [_row(ordinal=index, index=index) for index in range(1, 33)]
        records = [_record(index=index) for index in range(1, 33)]
        row = rows[0]
        with self.assertRaisesRegex(
            evaluator.Day20EvaluationV2Error,
            "generated token/text boundary drifted",
        ):
            evaluator.verify_prediction_rows(
                rows,
                selected_records=records,
                candidate_identity=factory.parse_candidate_id("base-probe"),
                eval_scope="probe32",
                scorer=_FrozenScorerStub,
                compact_model_identity=row["model_identity"],
                compact_checkpoint_package=row["checkpoint_package"],
                generated_token_decoder=lambda token_ids: "different decode",
            )

    def test_repository_frozen_manifest_and_scorer_validate(self) -> None:
        manifest, scorer = evaluator.load_frozen_eval_manifest(
            HERE.parent / "artifacts" / "eval" / "day10-frozen-eval-manifest.json",
            scorer_path=HERE.parent
            / "day-10-frozen-eval-baseline"
            / "day10_scorers.py",
        )
        self.assertEqual(
            len(manifest["header"]["evaluation_order"]["dev"]), 112
        )
        self.assertEqual(scorer.SCORER_REGISTRY_VERSION, "day10-scorer-registry-v3")

    def test_artifact_names_bind_candidate_and_scope(self) -> None:
        predictions, summary = evaluator.artifact_paths(
            Path("/tmp/eval"), candidate="base-probe", eval_scope="probe32"
        )
        self.assertEqual(
            predictions.name, "base-probe-probe32-raw-predictions-v2.jsonl"
        )
        self.assertEqual(summary.name, "base-probe-probe32-raw-summary-v2.json")

    def test_published_pair_revalidates_all_live_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            predictions, summary, decoded_by_ids = self._published_base_fixture(
                Path(temporary)
            )
            runtime_identity = _fixture_runtime_identity()
            decoder = lambda token_ids: decoded_by_ids[tuple(token_ids)]
            with mock.patch.object(
                evaluator,
                "live_runtime_identity",
                return_value=runtime_identity,
            ), mock.patch.object(
                evaluator,
                "build_live_generated_token_decoder",
                return_value=decoder,
            ):
                verified = evaluator.verify_published_pair(
                    predictions,
                    summary,
                    expected_scope="probe32",
                    expected_candidate="base-probe",
                )
                self.assertEqual(verified["predictions"]["records"], 32)
                predictions.write_text(
                    predictions.read_text(encoding="utf-8") + "{}\n",
                    encoding="utf-8",
                )
                with self.assertRaises(evaluator.Day20EvaluationV2Error):
                    evaluator.verify_published_pair(predictions, summary)

    def test_published_pair_rejects_live_decoder_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            predictions, summary, _ = self._published_base_fixture(Path(temporary))
            with mock.patch.object(
                evaluator,
                "live_runtime_identity",
                return_value=_fixture_runtime_identity(),
            ), mock.patch.object(
                evaluator,
                "build_live_generated_token_decoder",
                return_value=lambda token_ids: "wrong decoded text",
            ), self.assertRaisesRegex(
                evaluator.Day20EvaluationV2Error,
                "generated token/text boundary drifted",
            ):
                evaluator.verify_published_pair(predictions, summary)


if __name__ == "__main__":
    unittest.main()
