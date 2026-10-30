from __future__ import annotations

import copy
import unittest

import formal_s1_pair_labeler as labeler
import score_day22_s1_candidates as scorer
from test_formal_s1_pair_labeler import make_rollout


def make_request(status_suffix: str = "a") -> tuple[dict, dict, dict]:
    rollout = make_rollout(
        701,
        [
            (status_suffix, "    return x + 1", 20),
            ("unused", "    return x - 1", 20),
        ],
    )
    candidate = rollout["candidates"][0]
    request = {
        "schema_name": labeler.CANDIDATE_REPLAY_SCHEMA,
        "schema_version": 1,
        "candidate_id": candidate["candidate_id"],
        "task_id": rollout["task_id"],
        "family_id": rollout["family_id"],
        "split": rollout["split"],
        "code_prefix": rollout["code_prefix"],
        "entry_point": rollout["entry_point"],
        "tests": rollout["tests"],
        "tests_sha256": rollout["tests"]["sha256"],
        "candidate": {
            "candidate_id": candidate["candidate_id"],
            "origin": labeler.ORIGIN,
            "text": candidate["text"],
            "sha256": candidate["sha256"],
        },
        "generator_provenance_sha256": rollout["generator"]["provenance_sha256"],
        "source_rollout_candidate_sha256": candidate["candidate_sha256"],
    }
    return labeler._seal(request, "request_sha256"), rollout, candidate


def outcome(status: str) -> dict:
    return {
        "status": status,
        "exit_code": 0 if status == "pass" else 1,
        "error_type": "assertion_error" if status == "wrong_answer" else None,
        "failure_message": None,
        "duration_ms": 1.5,
        "stdout": "",
        "stdout_total_bytes": 0,
        "stderr": "",
        "stderr_total_bytes": 0,
    }


class Day22S1CandidateScorerTests(unittest.TestCase):
    def test_scores_exactly_two_fresh_runs_and_matches_labeler_contract(self) -> None:
        request, rollout, candidate = make_request()
        calls: list[str] = []

        def execute(program: str, **kwargs: object) -> dict:
            calls.append(program)
            return outcome("pass")

        evidence = scorer.score_candidate_request(
            request,
            e2b_api_key="secret",
            e2b_bindings={"fake": True},
            execute_program=execute,
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(len(evidence["runs"]), 2)
        self.assertEqual(evidence["source_tests_sha256"], request["tests_sha256"])
        self.assertEqual(evidence["test_sha256"], request["tests_sha256"])
        self.assertEqual(evidence["runs"][0]["sandbox"]["allow_internet_access"], False)
        self.assertNotEqual(evidence["runs"][0]["run_id"], evidence["runs"][1]["run_id"])
        validated = labeler._validate_evidence(
            evidence,
            candidate=candidate,
            rollout=rollout,
            index=0,
        )
        self.assertEqual(validated["status_class"], labeler.STABLE_PASS)

    def test_wrong_answer_evidence_is_stable_and_verifier_classified(self) -> None:
        request, rollout, candidate = make_request()
        evidence = scorer.score_candidate_request(
            request,
            e2b_api_key="secret",
            e2b_bindings={"fake": True},
            execute_program=lambda program, **kwargs: outcome("wrong_answer"),
        )

        validated = labeler._validate_evidence(
            evidence,
            candidate=candidate,
            rollout=rollout,
            index=0,
        )
        self.assertEqual(validated["status_class"], labeler.STABLE_WRONG)

    def test_request_self_hash_and_full_test_hash_fail_closed(self) -> None:
        request, _, _ = make_request()
        tampered = copy.deepcopy(request)
        tampered["candidate"]["text"] = "    return 0"
        with self.assertRaisesRegex(scorer.Day22S1CandidateScorerError, "request_sha256"):
            scorer.validate_request(tampered)

        wrong_tests = copy.deepcopy(request)
        wrong_tests["tests_sha256"] = "0" * 64
        wrong_tests = labeler._seal(wrong_tests, "request_sha256")
        with self.assertRaisesRegex(scorer.Day22S1CandidateScorerError, "tests_sha256"):
            scorer.validate_request(wrong_tests)

    def test_hash_sharding_is_deterministic_and_disjoint(self) -> None:
        ids = [f"candidate:{index}" for index in range(100)]
        partitions = [
            {candidate_id for candidate_id in ids if scorer.request_shard(candidate_id, 4) == shard}
            for shard in range(4)
        ]
        self.assertEqual(set.union(*partitions), set(ids))
        for left in range(4):
            for right in range(left + 1, 4):
                self.assertFalse(partitions[left] & partitions[right])


if __name__ == "__main__":
    unittest.main()
