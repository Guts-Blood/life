#!/usr/bin/env python3
"""Offline tests for the Day 22 MBPP sandbox scorer."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import score_day22_mbpp_sandbox as scorer


def pair_fixture() -> dict:
    tests = {
        "test_list": [
            "assert add_one(1) == 2",
            "assert add_one(-1) == 0",
        ],
        "challenge_test_list": ["assert add_one(99) == 100"],
        "test_setup_code": "import math",
    }
    return {
        "pair_id": "day22:mbpp:11",
        "task_id": 11,
        "family_id": "mbpp:11",
        "code_prefix": "def add_one(x):",
        "chosen": {
            "response": "    return x + 1",
            "origin": "mbpp_canonical_solution",
            "candidate_id": "mbpp:11:canonical",
        },
        "rejected": {
            "response": "    return x - 1",
            "origin": "deterministic_mutation",
            "candidate_id": "mbpp:11:mutation:subtract",
        },
        "tests": {**tests, "sha256": scorer.object_sha256(tests)},
    }


class CompositionTests(unittest.TestCase):
    def test_composition_is_atomic_and_challenge_tests_are_opt_in(self):
        record = pair_fixture()
        program = scorer.compose_mbpp_program(
            record, record["chosen"]["response"]
        )
        self.assertEqual(
            program,
            "import math\n"
            "def add_one(x):\n"
            "    return x + 1\n"
            "assert add_one(1) == 2\n"
            "assert add_one(-1) == 0\n",
        )
        self.assertNotIn("add_one(99)", program)
        challenge = scorer.compose_mbpp_program(
            record,
            record["chosen"]["response"],
            include_challenge_tests=True,
        )
        self.assertIn("assert add_one(99) == 100", challenge)

    def test_full_and_execution_test_hashes_are_distinct_and_stable(self):
        record = pair_fixture()
        _, source_hash, construction_hash = scorer.execution_test_bundle(record)
        _, repeated_source, repeated_construction = scorer.execution_test_bundle(record)
        _, _, challenge_hash = scorer.execution_test_bundle(
            record, include_challenge_tests=True
        )
        self.assertEqual(source_hash, record["tests"]["sha256"])
        self.assertEqual((source_hash, construction_hash), (repeated_source, repeated_construction))
        self.assertNotEqual(source_hash, construction_hash)
        self.assertEqual(source_hash, challenge_hash)

    def test_test_hash_mismatch_fails_before_composition(self):
        record = pair_fixture()
        record["tests"]["test_list"][0] = "assert add_one(1) == 999"
        with self.assertRaisesRegex(scorer.Day22SandboxError, "tests_sha256"):
            scorer.compose_mbpp_program(record, record["chosen"]["response"])

    def test_runner_hides_candidate_bytes_from_shell_and_sets_all_limits(self):
        sentinel = "UNIQUE_CANDIDATE_SENTINEL"
        runner = scorer.build_runner_source(
            sentinel,
            result_path="/tmp/result.json",
            max_output_bytes=123,
        )
        compile(runner, "<runner>", "exec")
        self.assertNotIn(sentinel, runner)
        self.assertIn("base64.b64decode", runner)
        for name in (
            "RLIMIT_CPU",
            "RLIMIT_AS",
            "RLIMIT_FSIZE",
            "RLIMIT_NOFILE",
            "RLIMIT_NPROC",
            "RLIMIT_CORE",
        ):
            self.assertIn(name, runner)


class SafetyTests(unittest.TestCase):
    def test_score_pair_refuses_execution_without_explicit_backend(self):
        with self.assertRaisesRegex(scorer.Day22SandboxError, "disabled by default"):
            scorer.score_pair(pair_fixture())

    def test_trusted_local_refuses_generated_code(self):
        record = pair_fixture()
        record["rejected"]["origin"] = "s1_sample"
        with self.assertRaisesRegex(scorer.Day22SandboxError, "trusted_local"):
            scorer.score_pair(record, backend="trusted_local")

    def test_trusted_local_identity_discloses_missing_security_isolation(self):
        identity = scorer.trusted_local_sandbox_identity(
            wall_timeout_seconds=1.0, max_output_bytes=100
        )
        self.assertEqual(identity["isolation"], "trusted_local_not_security_sandbox")
        self.assertFalse(identity["network_isolation"])
        self.assertTrue(identity["fresh_temporary_cwd_per_run"])
        self.assertTrue(identity["process_group_kill_on_timeout"])

    def test_e2b_contract_denies_network_and_uses_no_mounts_env_or_mcp(self):
        kwargs = scorer.e2b_create_kwargs("not-a-real-key")
        self.assertTrue(kwargs["secure"])
        self.assertFalse(kwargs["allow_internet_access"])
        self.assertEqual(
            kwargs["network"],
            {"allow_public_traffic": False, "deny_out": ["0.0.0.0/0"]},
        )
        self.assertEqual(kwargs["envs"], {})
        self.assertEqual(kwargs["volume_mounts"], {})
        self.assertIsNone(kwargs["mcp"])

    def test_cli_requires_an_execution_backend(self):
        parser = scorer.build_parser()
        with mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--input", "in.jsonl", "--output", "out.jsonl"])

    def test_credential_reader_requires_owner_only_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "e2b.env"
            path.write_text("E2B_API_KEY=test-key\n", encoding="utf-8")
            path.chmod(0o644)
            with self.assertRaisesRegex(scorer.Day22SandboxError, "0600"):
                scorer.read_e2b_api_key(path)
            path.chmod(0o600)
            self.assertEqual(scorer.read_e2b_api_key(path), "test-key")


class TrustedLocalExecutionTests(unittest.TestCase):
    def test_pass_wrong_answer_syntax_runtime_and_timeout_statuses(self):
        cases = {
            "pass": "def f():\n    return 1\nassert f() == 1\n",
            "wrong_answer": "def f():\n    return 2\nassert f() == 1\n",
            "syntax_error": "def f(:\n    return 1\n",
            "runtime_error": "raise ValueError('bad')\n",
            "timeout": "while True:\n    pass\n",
        }
        for expected, program in cases.items():
            with self.subTest(expected=expected):
                result = scorer.execute_trusted_local_program(
                    program,
                    wall_timeout_seconds=0.15 if expected == "timeout" else 1.0,
                )
                self.assertEqual(result["status"], expected, result)

    def test_pair_evidence_replays_both_sides_twice_and_self_hashes_runs(self):
        evidence = scorer.score_pair(
            pair_fixture(), backend="trusted_local", wall_timeout_seconds=1.0
        )
        self.assertEqual(evidence["task_id"], "11")
        self.assertEqual(evidence["pair_execution_status"], "eligible")
        self.assertEqual(
            [run["status"] for run in evidence["execution"]["chosen"]["runs"]],
            ["pass", "pass"],
        )
        self.assertEqual(
            [run["status"] for run in evidence["execution"]["rejected"]["runs"]],
            ["wrong_answer", "wrong_answer"],
        )
        all_runs = (
            evidence["execution"]["chosen"]["runs"]
            + evidence["execution"]["rejected"]["runs"]
        )
        self.assertEqual(len({run["run_id"] for run in all_runs}), 4)
        for side in ("chosen", "rejected"):
            runs = evidence["execution"][side]["runs"]
            self.assertEqual([run["attempt"] for run in runs], [1, 2])
            for run in runs:
                expected = scorer.object_sha256(
                    {key: value for key, value in run.items() if key != "run_sha256"}
                )
                self.assertEqual(run["run_sha256"], expected)
                self.assertEqual(run["sandbox_digest"], evidence["sandbox_digest"])
                self.assertEqual(run["test_sha256"], evidence["test_sha256"])
        expected_evidence = scorer.object_sha256(
            {
                key: value
                for key, value in evidence.items()
                if key != "evidence_sha256"
            }
        )
        self.assertEqual(evidence["evidence_sha256"], expected_evidence)

    def test_runner_captures_output_and_uses_clean_temp_working_directory(self):
        program = (
            "import os\n"
            "print(os.getcwd())\n"
            "print(os.environ.get('HOME'))\n"
            "print(os.environ.get('E2B_API_KEY'))\n"
        )
        result = scorer.execute_trusted_local_program(program)
        self.assertEqual(result["status"], "pass")
        lines = result["stdout"].splitlines()
        self.assertIn("day22-mbpp-", lines[0])
        self.assertEqual(lines[1:], ["None", "None"])


if __name__ == "__main__":
    unittest.main()
