#!/usr/bin/env python3
"""Focused offline tests for the Day 22 MBPP preference-pair contract."""

from __future__ import annotations

import copy
import unittest
from typing import Any

import day22_contract as contract


def _seal(value: dict[str, Any], field: str) -> dict[str, Any]:
    value[field] = contract.object_sha256(value)
    return value


def _response(text: str) -> dict[str, Any]:
    ast_hash = contract._response_ast_sha256(text, "def add_one(n):")
    return {"text": text, "sha256": contract.text_sha256(text), "ast_sha256": ast_hash}


def _run(
    *,
    branch: str,
    attempt: int,
    status: str,
    response_sha256: str,
    test_sha256: str,
    sandbox_digest: str,
) -> dict[str, Any]:
    stdout = "" if status == "pass" else "assertion failed\n"
    stderr = "" if status == "pass" else "AssertionError\n"
    run = {
        "attempt": attempt,
        "run_id": f"{branch}-fresh-{attempt}",
        "status": status,
        "response_sha256": response_sha256,
        "test_sha256": test_sha256,
        "sandbox_digest": sandbox_digest,
        "exit_code": 0 if status == "pass" else 1,
        "error_type": "assertion_error" if status == "wrong_answer" else None,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_sha256": contract.text_sha256(stdout),
        "stderr_sha256": contract.text_sha256(stderr),
    }
    return _seal(run, "run_sha256")


def _execution(chosen_hash: str, rejected_hash: str, test_hash: str) -> dict[str, Any]:
    sandbox = "sha256:" + "4" * 64
    execution = {
        "test_sha256": test_hash,
        "sandbox_digest": sandbox,
        "timeout_seconds": 2.0,
        "verifier": {"name": "day22_mbpp_sandbox", "version": "v1"},
        "chosen": {
            "runs": [
                _run(
                    branch="chosen",
                    attempt=attempt,
                    status="pass",
                    response_sha256=chosen_hash,
                    test_sha256=test_hash,
                    sandbox_digest=sandbox,
                )
                for attempt in (1, 2)
            ]
        },
        "rejected": {
            "runs": [
                _run(
                    branch="rejected",
                    attempt=attempt,
                    status="wrong_answer",
                    response_sha256=rejected_hash,
                    test_sha256=test_hash,
                    sandbox_digest=sandbox,
                )
                for attempt in (1, 2)
            ]
        },
    }
    return _seal(execution, "evidence_sha256")


def _processor(prompt_hash: str) -> dict[str, Any]:
    shared_prompt_tokens = "5" * 64
    audit = {
        "status": "pass",
        "retokenized_for_day22": True,
        "legacy_token_ids_reused": False,
        "prompt_text_sha256": prompt_hash,
        "model_key": "Qwen/Qwen3.5-4B",
        "model_revision": "1001bb4",
        "processor_revision": "day22-processor-v1",
        "tokenizer_revision": "day22-tokenizer-v1",
        "template_revision": "qwen3_5-day20-v3",
        "processor_sha256": "6" * 64,
        "tokenizer_sha256": "7" * 64,
        "template_sha256": "8" * 64,
        "rendered_prompt_sha256": "9" * 64,
        "chosen": {
            "status": "pass",
            "prompt_prefix_token_ids_sha256": shared_prompt_tokens,
            "response_token_ids_sha256": "a" * 64,
            "input_token_count": 16,
            "response_token_count": 4,
            "response_span": [12, 16],
            "truncation": False,
            "response_only_mask": True,
            "causal_shift_status": "pass",
        },
        "rejected": {
            "status": "pass",
            "prompt_prefix_token_ids_sha256": shared_prompt_tokens,
            "response_token_ids_sha256": "b" * 64,
            "input_token_count": 17,
            "response_token_count": 5,
            "response_span": [12, 17],
            "truncation": False,
            "response_only_mask": True,
            "causal_shift_status": "pass",
        },
    }
    return _seal(audit, "audit_sha256")


def make_pair(task_id: int = 752, split: str = "train", *, final: bool = True) -> dict[str, Any]:
    prompt = "Complete the function.\n\ndef add_one(n):"
    problem = "Write a function that adds one."
    chosen = _response("    return n + 1")
    rejected = _response("    return n - 1")
    canonical_tests = {
        "test_list": ["assert add_one(1) == 2"],
        "challenge_test_list": [],
        "test_setup_code": "",
    }
    test_hash = contract.object_sha256(canonical_tests)
    native_family = f"mbpp:task:{task_id}"
    pair: dict[str, Any] = {
        "schema_name": contract.PAIR_SCHEMA_NAME,
        "schema_version": contract.PAIR_SCHEMA_VERSION,
        "pair_id": f"{native_family}:pair:0",
        "pair_status": "accepted" if final else "candidate",
        "split": split,
        "task_id": task_id,
        "language": "python",
        "family_keys": {
            "problem": native_family,
            "prompt": f"sha256:{contract.text_sha256(prompt)}",
            "test": f"sha256:{test_hash}",
            "source": native_family,
        },
        "source": {
            "dataset": contract.MBPP_SOURCE,
            "revision": contract.MBPP_REVISION,
            "license": contract.MBPP_LICENSE,
            "split": "train",
            "variant": "full",
            "source_file": contract.MBPP_TRAIN_FILES["full"]["source_file"],
            "source_file_sha256": contract.MBPP_TRAIN_FILES["full"]["source_file_sha256"],
            "source_row_index": task_id,
            "source_content_sha256": "c" * 64,
            "day20_sample_id": f"train:code:mbpp:{task_id}",
            "day20_prompt_sha256": "d" * 64,
            "prompt_text_sha256": "e" * 64,
        },
        "problem": {"text": problem, "sha256": contract.text_sha256(problem)},
        "prompt": {"text": prompt, "sha256": contract.text_sha256(prompt)},
        "code_prefix": "def add_one(n):",
        "entry_point": "add_one",
        "chosen": chosen,
        "rejected": rejected,
        "creation": {"method": "canonical_vs_ast_mutation", "version": "v1"},
        "tests": {**canonical_tests, "sha256": test_hash, "count": 1, "challenge_count": 0},
    }
    if final:
        pair["execution"] = _execution(chosen["sha256"], rejected["sha256"], test_hash)
        pair["processor_audit"] = _processor(pair["prompt"]["sha256"])
    else:
        pair["processor_audit"] = {"status": "pending"}
    return contract.seal_pair(pair)


class Day22ContractTests(unittest.TestCase):
    def test_hashes_and_family_split_are_stable(self) -> None:
        left = {"z": 1, "text": "中文"}
        right = {"text": "中文", "z": 1}
        self.assertEqual(contract.object_sha256(left), contract.object_sha256(right))
        first = contract.stable_family_split("mbpp:task:752")
        self.assertEqual(first, contract.stable_family_split("mbpp:task:752"))
        self.assertIn(first, contract.SPLITS)
        with self.assertRaises(contract.Day22ContractError):
            contract.stable_family_split("x", ratios=(80, 20))

    def test_candidate_is_valid_only_in_smoke_mode(self) -> None:
        pair = make_pair(final=False)
        contract.validate_pair(pair, mode="smoke")
        with self.assertRaisesRegex(contract.Day22ContractError, "final pair_status"):
            contract.validate_pair(pair, mode="final")

    def test_blocked_processor_pair_needs_valid_execution(self) -> None:
        pair = make_pair(final=True)
        pair["pair_status"] = "blocked_processor_audit"
        pair["processor_audit"] = {"status": "blocked", "reason": "processor unavailable"}
        pair = contract.seal_pair(pair)
        contract.validate_pair(pair, mode="smoke")

    def test_final_pair_passes_all_contracts(self) -> None:
        pair = make_pair()
        contract.validate_pair(pair)
        summary = contract.validate_manifest([pair])
        self.assertEqual(summary["records"], 1)
        self.assertEqual(summary["split_counts"], {"train": 1, "dev": 0, "heldout": 0})

    def test_verifier_caught_wrong_answer_may_have_zero_process_exit(self) -> None:
        pair = make_pair()
        execution = pair["execution"]
        for run in execution["rejected"]["runs"]:
            run["exit_code"] = 0
            run["run_sha256"] = contract.object_sha256(
                {key: value for key, value in run.items() if key != "run_sha256"}
            )
        execution["evidence_sha256"] = contract.object_sha256(
            {key: value for key, value in execution.items() if key != "evidence_sha256"}
        )
        pair = contract.seal_pair(pair)
        contract.validate_pair(pair)

    def test_processor_allows_one_trailing_masked_token(self) -> None:
        pair = make_pair()
        audit = pair["processor_audit"]
        audit["chosen"]["input_token_count"] += 1
        audit["rejected"]["input_token_count"] += 1
        audit["audit_sha256"] = contract.object_sha256(
            {key: value for key, value in audit.items() if key != "audit_sha256"}
        )
        pair = contract.seal_pair(pair)
        contract.validate_pair(pair)

    def test_identical_or_ast_equivalent_responses_are_rejected(self) -> None:
        pair = make_pair(final=False)
        pair["rejected"] = copy.deepcopy(pair["chosen"])
        pair = contract.seal_pair(pair)
        with self.assertRaisesRegex(contract.Day22ContractError, "identical"):
            contract.validate_pair(pair, mode="smoke")

        pair = make_pair(final=False)
        pair["rejected"] = _response("    return (n + 1)")
        pair = contract.seal_pair(pair)
        with self.assertRaisesRegex(contract.Day22ContractError, "AST-equivalent"):
            contract.validate_pair(pair, mode="smoke")

    def test_timeout_cannot_be_labeled_as_rejected(self) -> None:
        pair = make_pair()
        rejected_run = pair["execution"]["rejected"]["runs"][1]
        rejected_run["status"] = "timeout"
        rejected_run["exit_code"] = None
        rejected_run["run_sha256"] = contract.object_sha256(
            {key: value for key, value in rejected_run.items() if key != "run_sha256"}
        )
        execution = pair["execution"]
        execution["evidence_sha256"] = contract.object_sha256(
            {key: value for key, value in execution.items() if key != "evidence_sha256"}
        )
        pair = contract.seal_pair(pair)
        with self.assertRaisesRegex(contract.Day22ContractError, "wrong_answer"):
            contract.validate_pair(pair)

    def test_replay_attempts_must_be_distinct(self) -> None:
        pair = make_pair()
        second = pair["execution"]["chosen"]["runs"][1]
        second["run_id"] = pair["execution"]["chosen"]["runs"][0]["run_id"]
        second["run_sha256"] = contract.object_sha256(
            {key: value for key, value in second.items() if key != "run_sha256"}
        )
        execution = pair["execution"]
        execution["evidence_sha256"] = contract.object_sha256(
            {key: value for key, value in execution.items() if key != "evidence_sha256"}
        )
        pair = contract.seal_pair(pair)
        with self.assertRaisesRegex(contract.Day22ContractError, "distinct run_id"):
            contract.validate_pair(pair)

    def test_processor_prefix_or_response_span_drift_fails(self) -> None:
        pair = make_pair()
        audit = pair["processor_audit"]
        audit["rejected"]["prompt_prefix_token_ids_sha256"] = "d" * 64
        audit["audit_sha256"] = contract.object_sha256(
            {key: value for key, value in audit.items() if key != "audit_sha256"}
        )
        pair = contract.seal_pair(pair)
        with self.assertRaisesRegex(contract.Day22ContractError, "prompt token prefixes"):
            contract.validate_pair(pair)

        pair = make_pair()
        audit = pair["processor_audit"]
        audit["chosen"]["response_token_count"] = 3
        audit["audit_sha256"] = contract.object_sha256(
            {key: value for key, value in audit.items() if key != "audit_sha256"}
        )
        pair = contract.seal_pair(pair)
        with self.assertRaisesRegex(contract.Day22ContractError, "token count drifted"):
            contract.validate_pair(pair)

    def test_pair_self_hash_detects_tampering(self) -> None:
        pair = make_pair(final=False)
        pair["prompt"]["text"] += " changed"
        with self.assertRaises(contract.Day22ContractError):
            contract.validate_pair(pair, mode="smoke")

    def test_upstream_non_train_source_is_rejected(self) -> None:
        pair = make_pair(final=False)
        pair["source"]["split"] = "test"
        pair = contract.seal_pair(pair)
        with self.assertRaisesRegex(contract.Day22ContractError, "upstream MBPP train"):
            contract.validate_pair(pair, mode="smoke")

    def test_manifest_does_not_treat_dataset_name_as_one_family(self) -> None:
        first = make_pair(task_id=752, split="train")
        second = make_pair(task_id=753, split="dev")
        second["problem"] = {
            "text": "Write another small arithmetic function.",
            "sha256": contract.text_sha256("Write another small arithmetic function."),
        }
        second["prompt"] = {
            "text": "Complete another function.\n\ndef add_one(n):",
            "sha256": contract.text_sha256("Complete another function.\n\ndef add_one(n):"),
        }
        second["family_keys"]["prompt"] = f"sha256:{second['prompt']['sha256']}"
        second["processor_audit"] = _processor(second["prompt"]["sha256"])
        other_tests = {
            "test_list": ["assert add_one(2) == 3"],
            "challenge_test_list": [],
            "test_setup_code": "",
        }
        other_hash = contract.object_sha256(other_tests)
        second["tests"] = {**other_tests, "sha256": other_hash, "count": 1, "challenge_count": 0}
        second["family_keys"]["test"] = f"sha256:{other_hash}"
        second["execution"] = _execution(
            second["chosen"]["sha256"], second["rejected"]["sha256"], other_hash
        )
        second = contract.seal_pair(second)
        summary = contract.validate_manifest([first, second])
        self.assertEqual(summary["split_counts"], {"train": 1, "dev": 1, "heldout": 0})

    def test_manifest_rejects_test_family_leakage(self) -> None:
        first = make_pair(task_id=752, split="train")
        second = make_pair(task_id=753, split="dev")
        second["problem"] = {
            "text": "A separate problem.",
            "sha256": contract.text_sha256("A separate problem."),
        }
        second["prompt"] = {
            "text": "A separate prompt.\n\ndef add_one(n):",
            "sha256": contract.text_sha256("A separate prompt.\n\ndef add_one(n):"),
        }
        second["family_keys"]["prompt"] = f"sha256:{second['prompt']['sha256']}"
        second["processor_audit"] = _processor(second["prompt"]["sha256"])
        second = contract.seal_pair(second)
        with self.assertRaisesRegex(contract.Day22ContractError, "test family.*leaks"):
            contract.validate_manifest([first, second])


if __name__ == "__main__":
    unittest.main()
