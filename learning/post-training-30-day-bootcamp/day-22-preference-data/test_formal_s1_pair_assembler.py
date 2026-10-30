from __future__ import annotations

import copy
import re
import unittest
from collections import Counter
from typing import Any

import day22_contract as contract
import formal_s1_pair_assembler as assembler
import formal_s1_pair_labeler as labeler


def _generator() -> dict[str, Any]:
    config = {
        "do_sample": True,
        "max_new_tokens": 512,
        "temperature": 0.8,
        "top_p": 0.95,
    }
    return labeler.seal_generator(
        {
            "promoted_s1": True,
            "on_policy": True,
            "checkpoint": "s1/main-s20260809-lr1e-4-final",
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


def _tests(task_id: int) -> dict[str, Any]:
    normalized = {
        "test_list": [f"assert f(1) == 2  # task {task_id}"],
        "challenge_test_list": [],
        "test_setup_code": "",
    }
    return {
        **normalized,
        "count": 1,
        "challenge_count": 0,
        "sha256": contract.object_sha256(normalized),
    }


def _source(task_id: int, prompt_sha256: str) -> dict[str, Any]:
    variant = "full"
    pinned = contract.MBPP_TRAIN_FILES[variant]
    return {
        "dataset": contract.MBPP_SOURCE,
        "revision": contract.MBPP_REVISION,
        "license": contract.MBPP_LICENSE,
        "split": "train",
        "variant": variant,
        "source_file": pinned["source_file"],
        "source_file_sha256": pinned["source_file_sha256"],
        "source_row_index": task_id,
        "source_content_sha256": contract.text_sha256(f"source:{task_id}"),
        "day20_sample_id": f"train:code:mbpp:{task_id}",
        "day20_prompt_sha256": contract.text_sha256(f"day20-prompt:{task_id}"),
        "prompt_text_sha256": prompt_sha256,
        "task_id": task_id,
    }


def _candidate(
    family_id: str,
    suffix: str,
    text: str,
    sample_index: int,
    generator: dict[str, Any],
) -> dict[str, Any]:
    return labeler.seal_candidate(
        {
            "candidate_id": f"{family_id}:sample:{suffix}",
            "family_id": family_id,
            "origin": labeler.ORIGIN,
            "text": text,
            "sha256": contract.text_sha256(text),
            "response_token_count": 4,
            "sample_index": sample_index,
            "sample_seed": 2026081300 + sample_index,
            "generation_run_id": f"gpu-{sample_index}",
            "generator_provenance_sha256": generator["provenance_sha256"],
        }
    )


def _rollout(task_id: int, generator: dict[str, Any]) -> dict[str, Any]:
    family_id = f"mbpp:task:{task_id}"
    prompt_text = f"Complete task {task_id}.\n\ndef f(x):"
    problem_text = f"Return x plus one for task {task_id}."
    prompt = {"text": prompt_text, "sha256": contract.text_sha256(prompt_text)}
    problem = {"text": problem_text, "sha256": contract.text_sha256(problem_text)}
    tests = _tests(task_id)
    split = contract.stable_family_split(family_id)
    row = {
        "schema_name": labeler.ROLLOUT_SCHEMA,
        "schema_version": 1,
        "task_id": str(task_id),
        "family_id": family_id,
        "split": split,
        "family_keys": {
            "problem": family_id,
            "prompt": f"sha256:{prompt['sha256']}",
            "test": f"sha256:{tests['sha256']}",
            "source": family_id,
        },
        "source": _source(task_id, prompt["sha256"]),
        "problem": problem,
        "prompt": prompt,
        "code_prefix": "def f(x):",
        "entry_point": "f",
        "tests": tests,
        "generator": copy.deepcopy(generator),
        "candidates": [
            _candidate(family_id, "pass", "    return x + 1", 0, generator),
            _candidate(family_id, "wrong", "    return x - 1", 1, generator),
        ],
    }
    return labeler.seal_rollout(row)


def _sandbox() -> dict[str, Any]:
    return {
        "backend": "e2b_firecracker",
        "isolation": "fresh_security_sandbox",
        "secure": True,
        "allow_internet_access": False,
        "fresh_sandbox_per_run": True,
        "allow_public_traffic": False,
        "template_id": "rki5dems9wqfm4r03t7g",
        "wall_timeout_seconds": 4.0,
    }


def _evidence(
    rollout: dict[str, Any], candidate: dict[str, Any], status: str
) -> dict[str, Any]:
    sandbox = _sandbox()
    sandbox_digest = contract.object_sha256(sandbox)
    runs = []
    for attempt in (1, 2):
        stdout = ""
        stderr = "" if status == "pass" else "AssertionError\n"
        run = {
            "attempt": attempt,
            "run_id": f"{candidate['candidate_id']}:run:{attempt}",
            "candidate_id": candidate["candidate_id"],
            "task_id": rollout["task_id"],
            "family_id": rollout["family_id"],
            "status": status,
            "exit_code": 0 if status == "pass" else 1,
            "error_type": None if status == "pass" else "assertion_error",
            "response_sha256": candidate["sha256"],
            "source_tests_sha256": rollout["tests"]["sha256"],
            "test_sha256": rollout["tests"]["sha256"],
            "sandbox_digest": sandbox_digest,
            "sandbox": copy.deepcopy(sandbox),
            "stdout": stdout,
            "stdout_sha256": contract.text_sha256(stdout),
            "stderr": stderr,
            "stderr_sha256": contract.text_sha256(stderr),
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
        "source_rollout_candidate_sha256": candidate["candidate_sha256"],
        "source_tests_sha256": rollout["tests"]["sha256"],
        "test_sha256": rollout["tests"]["sha256"],
        "sandbox_digest": sandbox_digest,
        "runs": runs,
    }
    return labeler.seal_evidence(evidence)


def _processor_contract() -> dict[str, Any]:
    return assembler._seal(
        {
            "schema_name": "day22.qwen35_processor_contract",
            "schema_version": 1,
            "status": "frozen",
            "max_length": 2304,
            "processor_sha256": "1" * 64,
            "tokenizer_sha256": "2" * 64,
            "template_sha256": "3" * 64,
        },
        "contract_sha256",
    )


def _processor_audit(
    replay: dict[str, Any], processor_contract: dict[str, Any]
) -> dict[str, Any]:
    prompt_hash = "4" * 64

    def branch(response_hash: str) -> dict[str, Any]:
        return {
            "status": "pass",
            "prompt_prefix_token_ids_sha256": prompt_hash,
            "response_token_ids_sha256": response_hash,
            "input_token_count": 20,
            "response_token_count": 4,
            "response_span": [15, 19],
            "truncation": False,
            "response_only_mask": True,
            "causal_shift_status": "pass",
        }

    audit = {
        "schema_name": "day22.qwen35_pair_processor_audit",
        "schema_version": 1,
        "status": "pass",
        "pair_id": replay["pair_id"],
        "task_id": str(replay["task_id"]),
        "family_id": replay["family_id"],
        "chosen_candidate_id": replay["chosen"]["candidate_id"],
        "rejected_candidate_id": replay["rejected"]["candidate_id"],
        "replay_row_sha256": replay["row_sha256"],
        "retokenized_for_day22": True,
        "legacy_token_ids_reused": False,
        "prompt_text_sha256": replay["prompt"]["sha256"],
        "chosen_response_sha256": replay["chosen"]["sha256"],
        "rejected_response_sha256": replay["rejected"]["sha256"],
        "model_key": "Qwen/Qwen3.5-4B-Base",
        "model_revision": "1001bb4d826a52d1f399e183466143f4da7b741b",
        "processor_revision": "qwen35-v1",
        "tokenizer_revision": "qwen35-v1",
        "template_revision": "day20.qwen35_code_boundary_target_v3",
        "processor_sha256": processor_contract["processor_sha256"],
        "tokenizer_sha256": processor_contract["tokenizer_sha256"],
        "template_sha256": processor_contract["template_sha256"],
        "processor_contract_sha256": processor_contract["contract_sha256"],
        "rendered_prompt_sha256": prompt_hash,
        "chosen": branch(replay["chosen"]["sha256"]),
        "rejected": branch(replay["rejected"]["sha256"]),
    }
    return assembler._seal(audit, "audit_sha256")


def fixture(count: int) -> dict[str, Any]:
    generator = _generator()
    rollouts = [_rollout(task_id, generator) for task_id in range(1000, 1000 + count)]
    evidence = [
        _evidence(
            rollout,
            candidate,
            "pass" if candidate["candidate_id"].endswith(":pass") else "wrong_answer",
        )
        for rollout in rollouts
        for candidate in rollout["candidates"]
    ]
    labeled = labeler.label_rollouts(rollouts, evidence)
    processor_contract = _processor_contract()
    audits = [
        _processor_audit(replay, processor_contract) for replay in labeled["replays"]
    ]
    return {
        **labeled,
        "evidence": evidence,
        "processor_contract": processor_contract,
        "audits": audits,
    }


def fixture_with_excluded_candidate(count: int) -> dict[str, Any]:
    generator = _generator()
    rollouts = [_rollout(task_id, generator) for task_id in range(2000, 2000 + count)]
    first = copy.deepcopy(rollouts[0])
    first.pop("row_sha256")
    first["candidates"].append(
        _candidate(
            first["family_id"],
            "runtime",
            "    return missing_name",
            2,
            generator,
        )
    )
    rollouts[0] = labeler.seal_rollout(first)
    evidence = []
    for rollout in rollouts:
        for candidate in rollout["candidates"]:
            suffix = candidate["candidate_id"].rsplit(":", 1)[-1]
            status = {
                "pass": "pass",
                "wrong": "wrong_answer",
                "runtime": "runtime_error",
            }[suffix]
            evidence.append(_evidence(rollout, candidate, status))
    labeled = labeler.label_rollouts(rollouts, evidence)
    processor_contract = _processor_contract()
    audits = [
        _processor_audit(replay, processor_contract) for replay in labeled["replays"]
    ]
    return {
        **labeled,
        "evidence": evidence,
        "processor_contract": processor_contract,
        "audits": audits,
    }


def fixture_with_two_generation_configs(count: int) -> dict[str, Any]:
    first_generator = _generator()
    second_generator = copy.deepcopy(first_generator)
    second_generator.pop("provenance_sha256")
    second_generator["generation_config"]["temperature"] = 1.1
    second_generator["generation_config_sha256"] = contract.object_sha256(
        second_generator["generation_config"]
    )
    second_generator = labeler.seal_generator(second_generator)
    rollouts = [
        _rollout(task_id, first_generator) for task_id in range(3000, 3000 + count)
    ]
    first = copy.deepcopy(rollouts[0])
    rejected = copy.deepcopy(first["candidates"][1])
    rejected["generator_provenance_sha256"] = second_generator[
        "provenance_sha256"
    ]
    first["candidates"][1] = labeler.seal_candidate(rejected)
    first["generators"] = [first_generator, second_generator]
    rollouts[0] = labeler.seal_rollout(first)
    evidence = []
    for rollout in rollouts:
        for candidate in rollout["candidates"]:
            status = (
                "pass"
                if candidate["candidate_id"].endswith(":pass")
                else "wrong_answer"
            )
            row = _evidence(rollout, candidate, status)
            row["generator_provenance_sha256"] = candidate[
                "generator_provenance_sha256"
            ]
            evidence.append(labeler.seal_evidence(row))
    labeled = labeler.label_rollouts(rollouts, evidence)
    processor_contract = _processor_contract()
    audits = [
        _processor_audit(replay, processor_contract) for replay in labeled["replays"]
    ]
    return {
        **labeled,
        "evidence": evidence,
        "processor_contract": processor_contract,
        "audits": audits,
    }


def assemble(data: dict[str, Any]) -> dict[str, Any]:
    return assembler.assemble_formal_s1(
        data["selections"],
        data["replays"],
        data["summary"],
        data["evidence"],
        data["audits"],
        data["processor_contract"],
    )


class FormalS1PairAssemblerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.valid = fixture(assembler.MIN_FORMAL_PAIRS)

    def test_machine_ready_bundle_has_only_human_review_pending(self) -> None:
        result = assemble(copy.deepcopy(self.valid))
        self.assertEqual(len(result["pairs"]), 200)
        contract.validate_manifest(result["pairs"], mode="final")
        self.assertTrue(result["manifest"]["machine_ready"])
        self.assertFalse(result["manifest"]["formal_dpo_ready"])
        self.assertEqual(
            result["manifest"]["formal_dpo_blockers"],
            [assembler.PENDING_BLOCKER],
        )
        self.assertEqual(result["audit"]["pending_blockers"], [assembler.PENDING_BLOCKER])
        self.assertTrue(
            all(gate["passed"] for gate in result["audit"]["machine_gates"].values())
        )
        self.assertEqual(
            {pair["creation"]["synthetic"] for pair in result["pairs"]}, {False}
        )
        self.assertEqual(
            {pair["creation"]["promoted_s1_on_policy"] for pair in result["pairs"]},
            {True},
        )
        self.assertEqual(
            {tuple(pair["quality_flags"]) for pair in result["pairs"]},
            {(assembler.PENDING_BLOCKER,)},
        )

    def test_same_promoted_s1_with_two_generation_configs_is_machine_ready(self) -> None:
        data = fixture_with_two_generation_configs(assembler.MIN_FORMAL_PAIRS)
        result = assemble(data)
        self.assertEqual(len(result["pairs"]), assembler.MIN_FORMAL_PAIRS)
        mixed = next(
            pair
            for pair in result["pairs"]
            if len(pair["creation"]["generator_provenance_sha256s"]) == 2
        )
        self.assertNotEqual(
            mixed["chosen"]["generator"]["provenance_sha256"],
            mixed["rejected"]["generator"]["provenance_sha256"],
        )
        self.assertTrue(mixed["creation"]["promoted_s1_on_policy"])
        contract.validate_manifest(result["pairs"], mode="final")

    def test_blind_review_has_50_unique_pairs_and_concealed_swaps(self) -> None:
        result = assemble(copy.deepcopy(self.valid))
        worksheet = result["review_worksheet"]
        key = result["review_key"]
        self.assertEqual(key["unique_pairs"], 50)
        self.assertEqual(key["presentations"], 100)
        self.assertEqual(len(worksheet), 100)
        self.assertTrue(all("pair_id" not in row for row in worksheet))
        self.assertTrue(all(row["verdict"] == "" for row in worksheet))
        by_pair = Counter(item["pair_id"] for item in key["items"])
        self.assertEqual(set(by_pair.values()), {2})
        for pair_id in by_pair:
            items = [item for item in key["items"] if item["pair_id"] == pair_id]
            self.assertEqual({item["presentation"] for item in items}, {"primary", "swapped"})
            primary = next(item for item in items if item["presentation"] == "primary")
            swapped = next(item for item in items if item["presentation"] == "swapped")
            self.assertEqual((primary["a_side"], primary["b_side"]), (swapped["b_side"], swapped["a_side"]))

    def test_less_than_200_fails_closed(self) -> None:
        data = fixture(assembler.MIN_FORMAL_PAIRS - 1)
        with self.assertRaisesRegex(
            assembler.FormalS1AssemblyError,
            r"BLOCKED\[minimum_accepted_pairs\].*require 200",
        ):
            assemble(data)

    def test_non_s1_origin_is_rejected_even_when_hashes_are_resealed(self) -> None:
        data = copy.deepcopy(self.valid)
        pair_id = data["replays"][0]["pair_id"]
        data["replays"][0]["chosen"]["origin"] = "synthetic_mutation"
        data["replays"][0] = assembler._seal(data["replays"][0], "row_sha256")
        data["selections"][0]["chosen"]["origin"] = "synthetic_mutation"
        data["selections"][0]["replay_row_sha256"] = data["replays"][0]["row_sha256"]
        data["selections"][0] = assembler._seal(data["selections"][0], "selection_sha256")
        data["summary"]["content_identities"]["ordered_selection_sha256"] = contract.object_sha256(
            [row["selection_sha256"] for row in data["selections"]]
        )
        data["summary"]["content_identities"]["ordered_replay_sha256"] = contract.object_sha256(
            [row["row_sha256"] for row in data["replays"]]
        )
        data["summary"] = assembler._seal(data["summary"], "summary_sha256")
        with self.assertRaisesRegex(
            assembler.FormalS1AssemblyError, f"pair {re.escape(pair_id)} chosen origin"
        ):
            assemble(data)

    def test_processor_contract_join_is_content_bound(self) -> None:
        data = copy.deepcopy(self.valid)
        data["audits"][0]["processor_contract_sha256"] = "f" * 64
        data["audits"][0] = assembler._seal(data["audits"][0], "audit_sha256")
        with self.assertRaisesRegex(
            assembler.FormalS1AssemblyError, "processor_contract_sha256 drifted"
        ):
            assemble(data)

    def test_all_retained_candidate_evidence_is_bound_by_selection_hash(self) -> None:
        data = fixture_with_excluded_candidate(assembler.MIN_FORMAL_PAIRS)
        assemble(copy.deepcopy(data))
        excluded_id = next(
            item["candidate_id"]
            for item in data["selections"][0]["excluded_candidates"]
            if item.get("reason_code") == "runtime_error_not_preference_eligible"
        )
        excluded = next(
            item for item in data["evidence"] if item["candidate_id"] == excluded_id
        )
        excluded["generator_provenance_sha256"] = "f" * 64
        replaced = labeler.seal_evidence(excluded)
        data["evidence"] = [
            replaced if item["candidate_id"] == excluded_id else item
            for item in data["evidence"]
        ]
        with self.assertRaisesRegex(
            assembler.FormalS1AssemblyError,
            "selection evidence identity drifted",
        ):
            assemble(data)

    def test_candidate_evidence_file_is_bound_to_selection_summary_input(self) -> None:
        data = copy.deepcopy(self.valid)
        data["summary"]["inputs"] = {
            "candidate_evidence": {
                "path": "labeler-evidence.jsonl",
                "file_sha256": "a" * 64,
            }
        }
        data["summary"] = assembler._seal(data["summary"], "summary_sha256")
        with self.assertRaisesRegex(
            assembler.FormalS1AssemblyError,
            "candidate evidence input file identity drifted",
        ):
            assembler.assemble_formal_s1(
                data["selections"],
                data["replays"],
                data["summary"],
                data["evidence"],
                data["audits"],
                data["processor_contract"],
                input_identities={
                    "candidate_evidence": {
                        "path": "assembler-evidence.jsonl",
                        "file_sha256": "b" * 64,
                    }
                },
            )


if __name__ == "__main__":
    unittest.main()
