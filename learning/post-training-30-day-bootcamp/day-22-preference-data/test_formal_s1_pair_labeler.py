from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import day22_contract as contract
import formal_s1_pair_labeler as labeler
import prepare_day22_s1_replay as replay_preparer
import rollout_day22_s1 as rollout_contract


def make_generator(*, checkpoint: str = "s1/main-s20260809-lr1e-4-final") -> dict:
    config = {
        "do_sample": True,
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.95,
    }
    return labeler.seal_generator(
        {
            "promoted_s1": True,
            "on_policy": True,
            "checkpoint": checkpoint,
            "downstream_key": "qwen35-s1-v1",
            "checkpoint_manifest_sha256": "a" * 64,
            "promotion_manifest_sha256": "b" * 64,
            "tokenizer_sha256": "c" * 64,
            "generator_name": "transformers.generate",
            "generator_version": "5.12.1",
            "generation_config": config,
            "generation_config_sha256": contract.object_sha256(config),
        }
    )


def make_tests() -> dict:
    normalized = {
        "test_list": ["assert f(1) == 2", "assert f(2) == 3", "assert f(3) == 4"],
        "challenge_test_list": [],
        "test_setup_code": "",
    }
    return {
        **normalized,
        "count": 3,
        "challenge_count": 0,
        "sha256": contract.object_sha256(normalized),
    }


def make_candidate(
    family_id: str,
    suffix: str,
    text: str,
    token_count: int,
    generator: dict,
    sample_index: int,
) -> dict:
    return labeler.seal_candidate(
        {
            "candidate_id": f"{family_id}:sample:{suffix}",
            "family_id": family_id,
            "origin": labeler.ORIGIN,
            "text": text,
            "sha256": contract.text_sha256(text),
            "response_token_count": token_count,
            "sample_index": sample_index,
            "sample_seed": 1000 + sample_index,
            "generation_run_id": f"gpu-shard-{sample_index % 2}",
            "generator_provenance_sha256": generator["provenance_sha256"],
        }
    )


def make_rollout(
    task_id: int,
    candidate_specs: list[tuple[str, str, int]],
    *,
    generator: dict | None = None,
) -> dict:
    generator = copy.deepcopy(generator or make_generator())
    family_id = f"mbpp:task:{task_id}"
    problem_text = f"Return x plus one for task {task_id}."
    prompt_text = f"Write f for task {task_id}.\ndef f(x):"
    tests = make_tests()
    candidates = [
        make_candidate(family_id, suffix, text, tokens, generator, index)
        for index, (suffix, text, tokens) in enumerate(candidate_specs)
    ]
    row = {
        "schema_name": labeler.ROLLOUT_SCHEMA,
        "schema_version": 1,
        "task_id": str(task_id),
        "family_id": family_id,
        "split": "train",
        "family_keys": {
            "problem": family_id,
            "prompt": f"sha256:{contract.text_sha256(prompt_text)}",
            "test": f"sha256:{tests['sha256']}",
            "source": family_id,
        },
        "source": {"dataset": "google-research-datasets/mbpp", "split": "train"},
        "problem": {"text": problem_text, "sha256": contract.text_sha256(problem_text)},
        "prompt": {"text": prompt_text, "sha256": contract.text_sha256(prompt_text)},
        "code_prefix": "def f(x):",
        "entry_point": "f",
        "tests": tests,
        "generator": generator,
        "candidates": candidates,
    }
    return labeler.seal_rollout(row)


def secure_sandbox() -> dict:
    return {
        "backend": "e2b_firecracker",
        "isolation": "fresh_security_sandbox",
        "secure": True,
        "allow_internet_access": False,
        "fresh_sandbox_per_run": True,
        "allow_public_traffic": False,
        "template_id": "rki5dems9wqfm4r03t7g",
    }


def make_evidence(
    rollout: dict,
    candidate: dict,
    statuses: tuple[str, str],
    *,
    pass_ratio: tuple[int, int] | None = None,
    sandbox: dict | None = None,
) -> dict:
    sandbox = copy.deepcopy(sandbox or secure_sandbox())
    sandbox_digest = contract.object_sha256(sandbox)
    runs = []
    for attempt, status in enumerate(statuses, 1):
        run = {
            "attempt": attempt,
            "run_id": f"{candidate['candidate_id']}:run:{attempt}",
            "candidate_id": candidate["candidate_id"],
            "task_id": rollout["task_id"],
            "family_id": rollout["family_id"],
            "status": status,
            "exit_code": 0 if status == "pass" else 1,
            "error_type": "assertion_error" if status == "wrong_answer" else None,
            "response_sha256": candidate["sha256"],
            "source_tests_sha256": rollout["tests"]["sha256"],
            "test_sha256": rollout["tests"]["sha256"],
            "sandbox_digest": sandbox_digest,
            "sandbox": sandbox,
            "stdout": "",
            "stdout_sha256": contract.text_sha256(""),
            "stderr": "",
            "stderr_sha256": contract.text_sha256(""),
        }
        runs.append(labeler.seal_run(run))
    evidence = {
        "schema_name": labeler.EVIDENCE_SCHEMA,
        "schema_version": 1,
        "backend": "e2b",
        "candidate_id": candidate["candidate_id"],
        "task_id": rollout["task_id"],
        "family_id": rollout["family_id"],
        "response_sha256": candidate["sha256"],
        "generator_provenance_sha256": rollout["generator"]["provenance_sha256"],
        "source_tests_sha256": rollout["tests"]["sha256"],
        "test_sha256": rollout["tests"]["sha256"],
        "sandbox_digest": sandbox_digest,
        "runs": runs,
    }
    if pass_ratio is not None:
        evidence["test_pass_ratio_basis"] = labeler.seal_pass_ratio_basis(
            {
                "method": "isolated_test_replay",
                "tests_passed": pass_ratio[0],
                "tests_total": pass_ratio[1],
            }
        )
    return labeler.seal_evidence(evidence)


def evidence_for_rollout(
    rollout: dict,
    statuses: list[tuple[str, str]],
    ratios: list[tuple[int, int] | None] | None = None,
) -> list[dict]:
    ratios = ratios or [None] * len(statuses)
    return [
        make_evidence(rollout, candidate, status, pass_ratio=ratio)
        for candidate, status, ratio in zip(rollout["candidates"], statuses, ratios)
    ]


class FormalS1PairLabelerTests(unittest.TestCase):
    def test_same_promoted_policy_allows_different_generation_configs_in_one_pair(self) -> None:
        first_generator = make_generator()
        second_generator = copy.deepcopy(first_generator)
        second_generator.pop("provenance_sha256")
        second_generator["generation_config"]["temperature"] = 1.1
        second_generator["generation_config_sha256"] = contract.object_sha256(
            second_generator["generation_config"]
        )
        second_generator = labeler.seal_generator(second_generator)
        rollout = make_rollout(
            799,
            [("pass", "    return x + 1", 8), ("wrong", "    return x - 1", 9)],
            generator=first_generator,
        )
        rejected = copy.deepcopy(rollout["candidates"][1])
        rejected["generator_provenance_sha256"] = second_generator["provenance_sha256"]
        rollout["candidates"][1] = labeler.seal_candidate(rejected)
        rollout["generators"] = [first_generator, second_generator]
        rollout = labeler.seal_rollout(rollout)
        evidence = evidence_for_rollout(
            rollout,
            [("pass", "pass"), ("wrong_answer", "wrong_answer")],
        )
        evidence[1]["generator_provenance_sha256"] = second_generator[
            "provenance_sha256"
        ]
        evidence[1] = labeler.seal_evidence(evidence[1])
        result = labeler.label_rollouts([rollout], evidence)
        self.assertEqual(len(result["replays"]), 1)
        self.assertEqual(
            result["replays"][0]["chosen"]["generator"]["provenance_sha256"],
            first_generator["provenance_sha256"],
        )
        self.assertEqual(
            result["replays"][0]["rejected"]["generator"]["provenance_sha256"],
            second_generator["provenance_sha256"],
        )
        self.assertEqual(
            set(result["summary"]["generator_provenance_sha256s"]),
            {first_generator["provenance_sha256"], second_generator["provenance_sha256"]},
        )
        self.assertNotIn("common_generator_provenance_sha256", result["summary"])

    def test_candidate_per_row_adapter_deduplicates_and_defines_evidence_coverage(self) -> None:
        task_id = 750
        family_id = f"mbpp:task:{task_id}"
        tests = make_tests()
        prompt_text = "Write f.\ndef f(x):"
        record = {
            "task_id": task_id,
            "task_family_id": family_id,
            "split": "train",
            "problem": {
                "text": "Return x plus one.",
                "sha256": contract.text_sha256("Return x plus one."),
            },
            "prompt": {"text": prompt_text, "sha256": contract.text_sha256(prompt_text)},
            "code_prefix": "def f(x):",
            "entry_point": "f",
            "tests": tests,
            "family_keys": {
                "problem": family_id,
                "prompt": f"sha256:{contract.text_sha256(prompt_text)}",
                "test": f"sha256:{tests['sha256']}",
                "source": family_id,
            },
            "source": {"dataset": "google-research-datasets/mbpp", "split": "train"},
        }
        seed = {"records": [record]}
        seed["manifest_sha256"] = rollout_contract.object_sha256(seed)
        frozen_contract = {
            "schema_name": rollout_contract.CONTRACT_SCHEMA,
            "schema_version": 1,
            "seed_pool": {"manifest_sha256": seed["manifest_sha256"]},
        }
        frozen_contract["contract_sha256"] = rollout_contract.object_sha256(frozen_contract)
        specs = [
            {
                "candidate_id": f"{family_id}:s1:{rollout_contract.WINNER}:sample:{index:02d}",
                "record": record,
                "sample_index": index,
                "seed": 100 + index,
                "shard_id": 0,
            }
            for index in range(6)
        ]
        texts = [
            "    return x + 1",
            "    return x + 1",  # exact duplicate of sample 0
            "    return x - 1",
            "    return x + 2",
            "not a continuation",
            "also invalid",
        ]
        rows = []
        for spec, text in zip(specs, texts):
            valid = spec["sample_index"] < 4
            response_ids = [10, 20 + spec["sample_index"]]
            row = {
                "schema_name": rollout_contract.CANDIDATE_SCHEMA,
                "schema_version": 1,
                "candidate_id": spec["candidate_id"],
                "task_id": task_id,
                "task_family_id": family_id,
                "split": "train",
                "sample_index": spec["sample_index"],
                "generation_seed": spec["seed"],
                "problem": record["problem"],
                "prompt": record["prompt"],
                "code_prefix": record["code_prefix"],
                "entry_point": record["entry_point"],
                "tests": tests,
                "family_keys": record["family_keys"],
                "source": record["source"],
                "response": {
                    "text": text,
                    "sha256": contract.text_sha256(text),
                    "message_content": text,
                    "message_content_sha256": contract.text_sha256(text),
                    "prompt_token_ids": [1, 2],
                    "prompt_token_count": 2,
                    "prompt_token_ids_sha256": contract.object_sha256([1, 2]),
                    "generated_token_ids": response_ids,
                    "generated_token_count": len(response_ids),
                    "generated_token_ids_sha256": contract.object_sha256(response_ids),
                    "finish_reason": "stop",
                    "response_adapter": {},
                },
                "format_contract": {
                    "valid": valid,
                    "execution_eligible": valid,
                    "validator": "validate_raw_code_continuation",
                    "evidence": {} if valid else None,
                    "error": None if valid else {"type": "format", "message": "invalid"},
                },
                "generator": {
                    "rollout_contract_sha256": frozen_contract["contract_sha256"],
                    "candidate": rollout_contract.WINNER,
                    "model_role": "merged_s1",
                    "merged_snapshot_sha256": "1" * 64,
                    "source_checkpoint_integrity_sha256": "2" * 64,
                    "source_adapter_sha256": "3" * 64,
                    "lineage_manifest_file_sha256": "4" * 64,
                    "runtime_sha256": "5" * 64,
                    "ms_swift_commit": rollout_contract.MS_SWIFT_COMMIT,
                    "downstream_key": "qwen35-s1-v1",
                    "promotion_manifest_sha256": "7" * 64,
                    "merged_export_manifest_sha256": "8" * 64,
                    "generation": {"do_sample": True, "samples_per_family": 6},
                    "generation_config_sha256": contract.object_sha256(
                        {"do_sample": True, "samples_per_family": 6}
                    ),
                    "shard_id": 0,
                    "shard_count": 2,
                    "elapsed_seconds": 0.5,
                },
            }
            row["candidate_sha256"] = rollout_contract.object_sha256(row)
            rows.append(row)

        with mock.patch.object(rollout_contract, "expected_specs", return_value=specs):
            prepared = labeler.prepare_candidate_replay_requests(
                list(reversed(rows)),
                rollout_contract=frozen_contract,
                seed_manifest=seed,
            )

        self.assertEqual(len(prepared["requests"]), 3)
        self.assertEqual(prepared["summary"]["counts"]["exact_response_duplicates"], 1)
        self.assertEqual(prepared["summary"]["counts"]["format_ineligible_candidates"], 2)
        retained = {row["candidate_id"] for row in prepared["requests"]}
        self.assertIn(specs[0]["candidate_id"], retained)
        self.assertNotIn(specs[1]["candidate_id"], retained)
        self.assertTrue(
            all(
                row["request_sha256"]
                == contract.object_sha256(
                    {key: value for key, value in row.items() if key != "request_sha256"}
                )
                for row in prepared["requests"]
            )
        )

    def test_prefers_minimum_length_delta_then_hardest_attested_fail(self) -> None:
        rollout = make_rollout(
            601,
            [
                ("pass", "    return x + 1", 100),
                ("easy", "    return x - 1", 90),
                ("hard", "    return x + 2", 90),
                ("far", "    return x * 2", 50),
            ],
        )
        evidence = evidence_for_rollout(
            rollout,
            [("pass", "pass"), ("wrong_answer", "wrong_answer"), ("wrong_answer", "wrong_answer"), ("wrong_answer", "wrong_answer")],
            [(3, 3), (1, 3), (2, 3), (2, 3)],
        )

        result = labeler.label_rollouts([rollout], list(reversed(evidence)))

        self.assertEqual(len(result["replays"]), 1)
        selection = result["selections"][0]
        self.assertEqual(selection["chosen"]["candidate_id"], "mbpp:task:601:sample:pass")
        self.assertEqual(selection["rejected"]["candidate_id"], "mbpp:task:601:sample:hard")
        self.assertEqual(selection["selection_metrics"]["relative_token_length_delta_fraction"], [1, 10])
        self.assertAlmostEqual(selection["selection_metrics"]["rejected_test_pass_ratio"], 2 / 3)
        self.assertEqual(selection["labeler"]["rubric_sha256"], labeler.RUBRIC_SHA256)
        self.assertEqual(
            selection["selection_sha256"],
            contract.object_sha256({key: value for key, value in selection.items() if key != "selection_sha256"}),
        )

    def test_length_delta_is_primary_over_hardness(self) -> None:
        rollout = make_rollout(
            602,
            [
                ("pass", "    return x + 1", 100),
                ("close", "    return x - 1", 99),
                ("harder", "    return x + 2", 90),
            ],
        )
        evidence = evidence_for_rollout(
            rollout,
            [("pass", "pass"), ("wrong_answer", "wrong_answer"), ("wrong_answer", "wrong_answer")],
            [(3, 3), (0, 3), (2, 3)],
        )

        result = labeler.label_rollouts([rollout], evidence)

        self.assertEqual(result["replays"][0]["rejected"]["candidate_id"], "mbpp:task:602:sample:close")

    def test_extra_error_candidate_is_quarantined_not_selected(self) -> None:
        rollout = make_rollout(
            603,
            [
                ("pass", "    return x + 1", 20),
                ("wrong", "    return x - 1", 20),
                ("runtime", "    return missing_name", 20),
            ],
        )
        evidence = evidence_for_rollout(
            rollout,
            [("pass", "pass"), ("wrong_answer", "wrong_answer"), ("runtime_error", "runtime_error")],
        )

        result = labeler.label_rollouts([rollout], evidence)

        self.assertEqual(len(result["replays"]), 1)
        excluded = result["selections"][0]["excluded_candidates"]
        self.assertEqual(excluded[0]["reason_code"], "runtime_error_not_preference_eligible")

    def test_quarantines_both_pass_both_fail_and_unstable_families(self) -> None:
        cases = [
            (604, [("pass", "pass"), ("pass", "pass")], "both_pass"),
            (605, [("wrong_answer", "wrong_answer"), ("wrong_answer", "wrong_answer")], "both_fail"),
            (606, [("pass", "timeout"), ("infra_error", "infra_error")], "unstable_execution"),
        ]
        for task_id, statuses, reason in cases:
            with self.subTest(reason=reason):
                rollout = make_rollout(
                    task_id,
                    [("a", "    return x + 1", 20), ("b", "    return x - 1", 20)],
                )
                result = labeler.label_rollouts([rollout], evidence_for_rollout(rollout, statuses))
                self.assertEqual(result["replays"], [])
                self.assertEqual(result["selections"][0]["selection_status"], "quarantine")
                self.assertEqual(result["selections"][0]["labeler"]["reason_code"], reason)

    def test_quarantines_ast_equivalent_tie(self) -> None:
        rollout = make_rollout(
            607,
            [("a", "    return x + 1", 20), ("b", "    return (x + 1)", 20)],
        )
        evidence = evidence_for_rollout(
            rollout,
            [("pass", "pass"), ("wrong_answer", "wrong_answer")],
        )

        result = labeler.label_rollouts([rollout], evidence)

        self.assertEqual(result["replays"], [])
        self.assertEqual(result["selections"][0]["labeler"]["reason_code"], "ast_equivalent_tie")

    def test_rejects_generator_provenance_drift_across_gpu_shards(self) -> None:
        first = make_rollout(608, [("a", "    return x + 1", 20), ("b", "    return x - 1", 20)])
        second = make_rollout(
            609,
            [("a", "    return x + 1", 20), ("b", "    return x - 1", 20)],
            generator=make_generator(checkpoint="s1/a-different-checkpoint"),
        )
        evidence = evidence_for_rollout(first, [("pass", "pass"), ("wrong_answer", "wrong_answer")])
        evidence += evidence_for_rollout(second, [("pass", "pass"), ("wrong_answer", "wrong_answer")])

        with self.assertRaisesRegex(labeler.FormalS1LabelerError, "one exact promoted-S1"):
            labeler.label_rollouts([first, second], evidence)

    def test_rejects_non_promoted_or_off_policy_generator(self) -> None:
        for field in ("promoted_s1", "on_policy"):
            with self.subTest(field=field):
                generator = make_generator()
                generator[field] = False
                generator = labeler.seal_generator(generator)
                rollout = make_rollout(
                    610,
                    [("a", "    return x + 1", 20), ("b", "    return x - 1", 20)],
                    generator=generator,
                )
                evidence = evidence_for_rollout(rollout, [("pass", "pass"), ("wrong_answer", "wrong_answer")])
                with self.assertRaisesRegex(labeler.FormalS1LabelerError, field):
                    labeler.label_rollouts([rollout], evidence)

    def test_rejects_insecure_or_tampered_evidence(self) -> None:
        rollout = make_rollout(611, [("a", "    return x + 1", 20), ("b", "    return x - 1", 20)])
        secure = evidence_for_rollout(rollout, [("pass", "pass"), ("wrong_answer", "wrong_answer")])

        tampered = copy.deepcopy(secure)
        tampered[0]["runs"][0]["status"] = "wrong_answer"
        tampered[0] = labeler.seal_evidence(tampered[0])
        with self.assertRaisesRegex(labeler.FormalS1LabelerError, "run_sha256"):
            labeler.label_rollouts([rollout], tampered)

        insecure_sandbox = secure_sandbox()
        insecure_sandbox["allow_internet_access"] = True
        insecure = [
            make_evidence(rollout, rollout["candidates"][0], ("pass", "pass"), sandbox=insecure_sandbox),
            make_evidence(rollout, rollout["candidates"][1], ("wrong_answer", "wrong_answer"), sandbox=insecure_sandbox),
        ]
        with self.assertRaisesRegex(labeler.FormalS1LabelerError, "allow_internet_access"):
            labeler.label_rollouts([rollout], insecure)

    def test_order_independent_and_one_pair_per_family(self) -> None:
        generator = make_generator()
        first = make_rollout(612, [("a", "    return x + 1", 21), ("b", "    return x - 1", 20)], generator=generator)
        second = make_rollout(613, [("a", "    return x + 1", 21), ("b", "    return x - 1", 20)], generator=generator)
        evidence = evidence_for_rollout(first, [("pass", "pass"), ("wrong_answer", "wrong_answer")])
        evidence += evidence_for_rollout(second, [("pass", "pass"), ("wrong_answer", "wrong_answer")])

        forward = labeler.label_rollouts([first, second], evidence)
        reverse = labeler.label_rollouts([second, first], list(reversed(evidence)))

        self.assertEqual(
            [row["pair_id"] for row in forward["replays"]],
            [row["pair_id"] for row in reverse["replays"]],
        )
        self.assertEqual(len(forward["replays"]), 2)
        self.assertEqual(len({row["family_id"] for row in forward["replays"]}), 2)

    def test_cli_writes_processor_and_scorer_compatible_replay(self) -> None:
        rollout = make_rollout(614, [("a", "    return x + 1", 20), ("b", "    return x - 1", 20)])
        evidence = evidence_for_rollout(rollout, [("pass", "pass"), ("wrong_answer", "wrong_answer")])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rollouts_path = root / "rollouts.jsonl"
            evidence_path = root / "evidence.jsonl"
            selections_path = root / "selections.jsonl"
            replay_path = root / "replay.jsonl"
            summary_path = root / "summary.json"
            rollouts_path.write_text(json.dumps(rollout) + "\n", encoding="utf-8")
            evidence_path.write_text("".join(json.dumps(row) + "\n" for row in evidence), encoding="utf-8")

            cli = labeler.run_cli(
                rollouts_path=rollouts_path,
                evidence_path=evidence_path,
                selections_output=selections_path,
                replay_output=replay_path,
                summary_output=summary_path,
                overwrite=False,
            )

            self.assertEqual(cli["eligible_pairs"], 1)
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
            self.assertEqual(replay["schema_name"], "day22.mbpp_replay_pair")
            self.assertEqual(replay["chosen"]["origin"], labeler.ORIGIN)
            self.assertEqual(replay["tests_sha256"], replay["tests"]["sha256"])
            self.assertEqual(
                replay["row_sha256"],
                contract.object_sha256({key: value for key, value in replay.items() if key != "row_sha256"}),
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertFalse(summary["formal_dpo_ready"])
            self.assertEqual(summary["counts"]["eligible_pairs"], 1)

            for path in (selections_path, replay_path, summary_path):
                path.write_text("stale\n", encoding="utf-8")
            labeler.run_cli(
                rollouts_path=rollouts_path,
                evidence_path=evidence_path,
                selections_output=selections_path,
                replay_output=replay_path,
                summary_output=summary_path,
                overwrite=True,
            )
            for path in (selections_path, replay_path, summary_path):
                self.assertNotEqual(path.read_text(encoding="utf-8"), "stale\n")

    def test_cli_preflights_all_outputs_before_creating_any(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selections_path = root / "selections.jsonl"
            replay_path = root / "replay.jsonl"
            summary_path = root / "summary.json"
            summary_path.write_text("existing summary\n", encoding="utf-8")

            with self.assertRaisesRegex(labeler.FormalS1LabelerError, "refusing to overwrite"):
                labeler.run_cli(
                    rollouts_path=root / "unused-rollouts.jsonl",
                    evidence_path=root / "unused-evidence.jsonl",
                    selections_output=selections_path,
                    replay_output=replay_path,
                    summary_output=summary_path,
                    overwrite=False,
                )

            self.assertFalse(selections_path.exists())
            self.assertFalse(replay_path.exists())
            self.assertEqual(summary_path.read_text(encoding="utf-8"), "existing summary\n")

    def test_replay_preparer_preflights_all_outputs_before_creating_any(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requests_path = root / "requests.jsonl"
            summary_path = root / "summary.json"
            summary_path.write_text("existing summary\n", encoding="utf-8")
            argv = [
                "prepare_day22_s1_replay.py",
                "--rollouts",
                str(root / "unused-rollouts.jsonl"),
                "--rollout-contract",
                str(root / "unused-contract.json"),
                "--seed-manifest",
                str(root / "unused-seed.json"),
                "--requests-output",
                str(requests_path),
                "--summary-output",
                str(summary_path),
            ]

            with (
                mock.patch("sys.argv", argv),
                mock.patch("sys.stderr"),
                self.assertRaises(SystemExit) as raised,
            ):
                replay_preparer.main()

            self.assertEqual(raised.exception.code, 2)
            self.assertFalse(requests_path.exists())
            self.assertEqual(summary_path.read_text(encoding="utf-8"), "existing summary\n")

    def test_replay_preparer_overwrite_replaces_all_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requests_path = root / "requests.jsonl"
            summary_path = root / "summary.json"
            requests_path.write_text("stale requests\n", encoding="utf-8")
            summary_path.write_text("stale summary\n", encoding="utf-8")
            prepared = {
                "requests": [{"candidate_id": "candidate-1"}],
                "summary": {"summary_sha256": "a" * 64},
            }
            argv = [
                "prepare_day22_s1_replay.py",
                "--rollouts",
                str(root / "rollouts.jsonl"),
                "--rollout-contract",
                str(root / "contract.json"),
                "--seed-manifest",
                str(root / "seed.json"),
                "--requests-output",
                str(requests_path),
                "--summary-output",
                str(summary_path),
                "--overwrite",
            ]

            with (
                mock.patch("sys.argv", argv),
                mock.patch.object(labeler, "load_jsonl", return_value=[]),
                mock.patch.object(labeler, "load_json", return_value={}),
                mock.patch.object(labeler, "prepare_candidate_replay_requests", return_value=prepared),
                mock.patch("builtins.print"),
            ):
                replay_preparer.main()

            self.assertEqual(
                requests_path.read_bytes(),
                labeler._jsonl_bytes(prepared["requests"]),
            )
            self.assertEqual(
                summary_path.read_bytes(),
                labeler._json_bytes(prepared["summary"]),
            )

    def test_output_preflight_rejects_paths_with_same_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(labeler.FormalS1LabelerError, "must be distinct"):
                labeler._preflight_output_paths(
                    [root / "output.json", root / "missing" / ".." / "output.json"],
                    overwrite=True,
                )


if __name__ == "__main__":
    unittest.main()
