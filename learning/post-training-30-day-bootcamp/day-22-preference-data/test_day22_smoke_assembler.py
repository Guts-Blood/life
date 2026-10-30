#!/usr/bin/env python3
"""Focused synthetic-fixture tests for the Day 22 smoke assembler."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import assemble_day22_smoke as assembler
import day22_contract as contract


def _seal(value: dict[str, Any], field: str) -> dict[str, Any]:
    value[field] = assembler.object_sha256(value)
    return value


def _response(text: str, prefix: str) -> dict[str, Any]:
    return {
        "text": text,
        "sha256": assembler.text_sha256(text),
        "ast_sha256": contract._response_ast_sha256(text, prefix),
    }


def fixture(*, secure: bool = False) -> dict[str, Any]:
    task_id = 11
    family = f"mbpp:task:{task_id}"
    pair_id = f"{family}:smoke:00"
    prefix = "def add_one(n):"
    prompt_text = "Complete the Python function.\n\nStarter code:\ndef add_one(n):"
    problem_text = "Write a function that adds one."
    tests_core = {
        "test_list": ["assert add_one(1) == 2"],
        "challenge_test_list": [],
        "test_setup_code": "",
    }
    tests_hash = assembler.object_sha256(tests_core)
    tests = {**tests_core, "sha256": tests_hash, "count": 1, "challenge_count": 0}
    chosen_response = _response("    return n + 1", prefix)
    rejected_response = _response("    return n - 1", prefix)
    source = {
        "dataset": contract.MBPP_SOURCE,
        "revision": contract.MBPP_REVISION,
        "license": contract.MBPP_LICENSE,
        "split": "train",
        "variant": "full",
        "source_file": contract.MBPP_TRAIN_FILES["full"]["source_file"],
        "source_file_sha256": contract.MBPP_TRAIN_FILES["full"]["source_file_sha256"],
        "source_row_index": 0,
        "source_content_sha256": "1" * 64,
        "day20_sample_id": "train:code:mbpp:fixture",
        "day20_prompt_sha256": "2" * 64,
        "prompt_text_sha256": assembler.text_sha256(prompt_text),
        "prompt_migration": "day20_prompt_identity",
    }
    prompt = {"text": prompt_text, "sha256": assembler.text_sha256(prompt_text)}
    problem = {"text": problem_text, "sha256": assembler.text_sha256(problem_text)}
    families = {
        "problem": family,
        "prompt": f"sha256:{prompt['sha256']}",
        "test": f"sha256:{tests_hash}",
        "source": family,
    }
    seed_record = {
        "schema_name": "day22.mbpp_seed",
        "schema_version": 1,
        "task_family_id": family,
        "task_id": task_id,
        "split": "train",
        "smoke_selected": True,
        "source": source,
        "problem": problem,
        "prompt": prompt,
        "code_prefix": prefix,
        "entry_point": "add_one",
        "canonical_chosen": {
            **chosen_response,
            "origin": "mbpp_canonical_solution",
        },
        "tests": tests,
        "family_keys": families,
    }
    _seal(seed_record, "record_sha256")
    seed = {
        "header": {
            "schema_name": "day22.mbpp_seed_manifest",
            "schema_version": 1,
            "status": "seed_pool_frozen",
        },
        "records": [seed_record],
    }
    _seal(seed, "manifest_sha256")

    chosen = {
        **chosen_response,
        "origin": "mbpp_canonical_solution",
        "candidate_id": f"{family}:canonical",
        "trust_tag": "canonical",
    }
    _seal(chosen, "candidate_sha256")
    mutation = {
        "version": "day22.mbpp_single_ast_bug_v1",
        "rule": "Add_to_Sub",
        "node_ordinal": 1,
    }
    rejected = {
        **rejected_response,
        "origin": "deterministic_ast_mutation",
        "candidate_id": f"{family}:mutation:00",
        "trust_tag": "deterministic_mutation",
        "mutation": mutation,
    }
    _seal(rejected, "candidate_sha256")
    candidate = {
        "schema_name": "day22.mbpp_replay_candidates",
        "schema_version": 1,
        "task_family_id": family,
        "task_id": task_id,
        "split": "train",
        "family_keys": families,
        "source": source,
        "problem": problem,
        "prompt": prompt,
        "code_prefix": prefix,
        "entry_point": "add_one",
        "tests": tests,
        "chosen": chosen,
        "mutants": [rejected],
    }
    _seal(candidate, "row_sha256")
    replay = {
        "schema_name": "day22.mbpp_replay_pair",
        "schema_version": 1,
        "pair_id": pair_id,
        "task_id": str(task_id),
        "family_id": family,
        "split": "train",
        "code_prefix": prefix,
        "tests": tests,
        "tests_sha256": tests_hash,
        "chosen": {
            "candidate_id": chosen["candidate_id"],
            "origin": "mbpp_canonical",
            "text": chosen["text"],
            "sha256": chosen["sha256"],
        },
        "rejected": {
            "candidate_id": rejected["candidate_id"],
            "origin": "deterministic_mutation",
            "text": rejected["text"],
            "sha256": rejected["sha256"],
            "mutation": mutation,
        },
    }
    _seal(replay, "row_sha256")

    if secure:
        backend = "e2b"
        sandbox = {
            "backend": "e2b_firecracker",
            "isolation": "fresh_security_sandbox",
            "secure": True,
            "allow_internet_access": False,
            "fresh_sandbox_per_run": True,
            "wall_timeout_seconds": 4.0,
        }
    else:
        backend = "trusted_local"
        sandbox = {
            "backend": "trusted_local_subprocess",
            "isolation": "trusted_local_not_security_sandbox",
            "network_isolation": False,
            "wall_timeout_seconds": 4.0,
        }
    sandbox_digest = assembler.object_sha256(sandbox)

    def make_run(side: str, attempt: int, status: str, response_hash: str) -> dict[str, Any]:
        stdout = ""
        stderr = "" if status == "pass" else "AssertionError\n"
        run = {
            "schema_version": 1,
            "domain": "day22.mbpp_sandbox_run",
            "pair_id": pair_id,
            "task_id": str(task_id),
            "family_id": family,
            "run_id": f"{pair_id}:{side}:{attempt}:fresh",
            "side": side,
            "attempt": attempt,
            "status": status,
            "exit_code": 0 if status == "pass" else 1,
            "response_sha256": response_hash,
            "program_sha256": "3" * 64,
            "source_tests_sha256": tests_hash,
            "test_sha256": tests_hash,
            "sandbox_digest": sandbox_digest,
            "sandbox": sandbox,
            "started_at_utc": "2026-08-13T00:00:00+00:00",
            "duration_ms": 1.0,
            "stdout": stdout,
            "stdout_sha256": assembler.text_sha256(stdout),
            "stdout_bytes": len(stdout),
            "stdout_truncated": False,
            "stderr": stderr,
            "stderr_sha256": assembler.text_sha256(stderr),
            "stderr_bytes": len(stderr),
            "stderr_truncated": False,
            "error_type": None if status == "pass" else "assertion_error",
            "failure_message": None,
        }
        return _seal(run, "run_sha256")

    execution = {
        "chosen": {
            "candidate_id": chosen["candidate_id"],
            "origin": "mbpp_canonical",
            "response_sha256": chosen["sha256"],
            "runs": [make_run("chosen", attempt, "pass", chosen["sha256"]) for attempt in (1, 2)],
        },
        "rejected": {
            "candidate_id": rejected["candidate_id"],
            "origin": "deterministic_mutation",
            "response_sha256": rejected["sha256"],
            "runs": [
                make_run("rejected", attempt, "wrong_answer", rejected["sha256"])
                for attempt in (1, 2)
            ],
        },
    }
    evidence = {
        "schema_version": 1,
        "domain": "day22.mbpp_sandbox_pair_evidence",
        "pair_id": pair_id,
        "task_id": str(task_id),
        "family_id": family,
        "backend": backend,
        "source_tests_sha256": tests_hash,
        "test_sha256": tests_hash,
        "include_challenge_tests": False,
        "sandbox_digest": sandbox_digest,
        "execution": execution,
        "pair_execution_status": "eligible",
    }
    _seal(evidence, "evidence_sha256")

    prompt_token_hash = "4" * 64
    audit = {
        "schema_name": "day22.qwen35_pair_processor_audit",
        "schema_version": 1,
        "pair_id": pair_id,
        "task_id": str(task_id),
        "family_id": family,
        "chosen_candidate_id": chosen["candidate_id"],
        "rejected_candidate_id": rejected["candidate_id"],
        "chosen_response_sha256": chosen["sha256"],
        "rejected_response_sha256": rejected["sha256"],
        "status": "pass",
        "retokenized_for_day22": True,
        "legacy_token_ids_reused": False,
        "prompt_text_sha256": prompt["sha256"],
        "model_key": "Qwen/Qwen3.5-4B",
        "model_revision": "1001bb4",
        "processor_revision": "day22-v1",
        "tokenizer_revision": "day22-v1",
        "template_revision": "qwen3_5-day20-v3",
        "processor_sha256": "5" * 64,
        "tokenizer_sha256": "6" * 64,
        "template_sha256": "7" * 64,
        "rendered_prompt_sha256": "8" * 64,
        "chosen": {
            "status": "pass",
            "prompt_prefix_token_ids_sha256": prompt_token_hash,
            "response_token_ids_sha256": "9" * 64,
            "input_token_count": 20,
            "response_token_count": 5,
            "response_span": [14, 19],
            "truncation": False,
            "response_only_mask": True,
            "causal_shift_status": "pass",
        },
        "rejected": {
            "status": "pass",
            "prompt_prefix_token_ids_sha256": prompt_token_hash,
            "response_token_ids_sha256": "a" * 64,
            "input_token_count": 20,
            "response_token_count": 5,
            "response_span": [14, 19],
            "truncation": False,
            "response_only_mask": True,
            "causal_shift_status": "pass",
        },
    }
    _seal(audit, "audit_sha256")
    return {
        "seed": seed,
        "candidates": [candidate],
        "replays": [replay],
        "evidence": [evidence],
        "audits": [audit],
        "pair_id": pair_id,
    }


def assemble(data: dict[str, Any]) -> dict[str, Any]:
    return assembler.assemble_smoke(
        data["seed"],
        data["candidates"],
        data["replays"],
        data["evidence"],
        data["audits"],
    )


class Day22SmokeAssemblerTests(unittest.TestCase):
    def test_happy_path_emits_audited_smoke_but_blocks_formal_readiness(self) -> None:
        data = fixture()
        result = assemble(data)
        self.assertEqual(len(result["pairs"]), 1)
        pair = result["pairs"][0]
        contract.validate_pair(pair, mode="final")
        self.assertEqual(pair["dataset_role"], "audited_smoke_not_formal_dpo_data")
        self.assertIn("synthetic_mutation", pair["quality_flags"])
        self.assertIn("insecure_smoke_backend", pair["quality_flags"])
        manifest = result["manifest"]
        self.assertEqual(manifest["readiness"], "BLOCKED")
        self.assertFalse(manifest["formal_dpo_ready"])
        self.assertEqual(
            manifest["readiness_gates"]["synthetic_pair_fraction"]["observed"],
            1.0,
        )
        self.assertFalse(manifest["readiness_gates"]["human_blind_review"]["passed"])
        self.assertEqual(result["split_ids"]["train"], [data["pair_id"]])

    def test_blind_review_conceals_labels_and_emits_swapped_presentations(self) -> None:
        data = fixture()
        result = assemble(data)
        worksheet = result["review_worksheet"]
        key = result["review_key"]
        self.assertEqual(len(worksheet), 2)
        self.assertEqual(key["unique_pairs"], 1)
        encoded = json.dumps(worksheet, sort_keys=True)
        self.assertNotIn(data["pair_id"], encoded)
        self.assertNotIn("a_side", encoded)
        by_id = {row["review_item_id"]: row for row in worksheet}
        key_rows = key["items"]
        first, second = key_rows
        self.assertEqual({first["presentation"], second["presentation"]}, {"primary", "swapped"})
        primary = next(row for row in key_rows if row["presentation"] == "primary")
        swapped = next(row for row in key_rows if row["presentation"] == "swapped")
        self.assertEqual(primary["a_side"], swapped["b_side"])
        self.assertEqual(primary["b_side"], swapped["a_side"])
        self.assertNotEqual(
            by_id[primary["review_item_id"]]["response_a"],
            by_id[swapped["review_item_id"]]["response_a"],
        )

    def test_secure_evidence_passes_only_the_secure_sandbox_gate(self) -> None:
        result = assemble(fixture(secure=True))
        gates = result["manifest"]["readiness_gates"]
        self.assertTrue(gates["secure_sandbox"]["passed"])
        self.assertEqual(result["manifest"]["readiness"], "BLOCKED")

    def test_ineligible_execution_is_quarantined_not_emitted(self) -> None:
        data = fixture()
        evidence = data["evidence"][0]
        run = evidence["execution"]["rejected"]["runs"][1]
        run["status"] = "runtime_error"
        run["error_type"] = "ValueError"
        run["run_sha256"] = assembler.object_sha256(
            {key: value for key, value in run.items() if key != "run_sha256"}
        )
        evidence["pair_execution_status"] = "quarantine"
        evidence["evidence_sha256"] = assembler.object_sha256(
            {key: value for key, value in evidence.items() if key != "evidence_sha256"}
        )
        result = assemble(data)
        self.assertEqual(result["pairs"], [])
        self.assertEqual(result["manifest"]["join_counts"]["quarantined_pairs"], 1)
        self.assertEqual(
            result["manifest"]["join_counts"]["quarantine_reason_counts"],
            {"execution_not_pass_vs_stable_wrong_answer": 1},
        )

    def test_processor_nonpass_is_quarantined(self) -> None:
        data = fixture()
        audit = data["audits"][0]
        audit["status"] = "blocked"
        audit["reason"] = "processor unavailable"
        audit["audit_sha256"] = assembler.object_sha256(
            {key: value for key, value in audit.items() if key != "audit_sha256"}
        )
        result = assemble(data)
        self.assertEqual(result["pairs"], [])
        self.assertEqual(
            result["manifest"]["join_counts"]["quarantine_reason_counts"],
            {"processor_audit_not_pass": 1},
        )

    def test_tampered_or_misjoined_evidence_fails_closed(self) -> None:
        data = fixture()
        evidence = data["evidence"][0]
        evidence["execution"]["chosen"]["response_sha256"] = "f" * 64
        evidence["evidence_sha256"] = assembler.object_sha256(
            {key: value for key, value in evidence.items() if key != "evidence_sha256"}
        )
        with self.assertRaisesRegex(assembler.Day22AssemblyError, "response.*drifted"):
            assemble(data)

        data = fixture()
        data["audits"][0]["pair_id"] = "unknown"
        data["audits"][0]["audit_sha256"] = assembler.object_sha256(
            {
                key: value
                for key, value in data["audits"][0].items()
                if key != "audit_sha256"
            }
        )
        with self.assertRaisesRegex(assembler.Day22AssemblyError, "pair IDs"):
            assemble(data)

    def test_multiple_pairs_for_one_problem_are_rejected(self) -> None:
        data = fixture()
        duplicate = copy.deepcopy(data["replays"][0])
        duplicate["pair_id"] += ":duplicate"
        duplicate["row_sha256"] = assembler.object_sha256(
            {key: value for key, value in duplicate.items() if key != "row_sha256"}
        )
        data["replays"].append(duplicate)
        with self.assertRaisesRegex(assembler.Day22AssemblyError, "multiple replay pairs"):
            assemble(data)

    def test_cli_writes_all_artifacts_and_file_hashes(self) -> None:
        data = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {
                "seed": root / "seed.json",
                "candidates": root / "candidates.jsonl",
                "replay": root / "replay.jsonl",
                "evidence": root / "evidence.jsonl",
                "audits": root / "audits.jsonl",
                "pairs": root / "pairs.jsonl",
                "splits": root / "splits.json",
                "manifest": root / "manifest.json",
                "worksheet": root / "worksheet.jsonl",
                "key": root / "key.json",
            }
            paths["seed"].write_text(json.dumps(data["seed"]), encoding="utf-8")
            for name, rows in (
                ("candidates", data["candidates"]),
                ("replay", data["replays"]),
                ("evidence", data["evidence"]),
                ("audits", data["audits"]),
            ):
                paths[name].write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
            exit_code = assembler.main(
                [
                    "--seed",
                    str(paths["seed"]),
                    "--candidates",
                    str(paths["candidates"]),
                    "--replay-input",
                    str(paths["replay"]),
                    "--sandbox-evidence",
                    str(paths["evidence"]),
                    "--processor-audit",
                    str(paths["audits"]),
                    "--pairs-output",
                    str(paths["pairs"]),
                    "--split-ids-output",
                    str(paths["splits"]),
                    "--manifest-output",
                    str(paths["manifest"]),
                    "--review-worksheet-output",
                    str(paths["worksheet"]),
                    "--review-key-output",
                    str(paths["key"]),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(all(paths[name].is_file() for name in ("pairs", "splits", "manifest", "worksheet", "key")))
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            expected_hash = assembler.object_sha256(
                {key: value for key, value in manifest.items() if key != "manifest_sha256"}
            )
            self.assertEqual(manifest["manifest_sha256"], expected_hash)
            pair_identity = manifest["output_files"][paths["pairs"].name]
            self.assertEqual(
                pair_identity["file_sha256"],
                hashlib.sha256(paths["pairs"].read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
