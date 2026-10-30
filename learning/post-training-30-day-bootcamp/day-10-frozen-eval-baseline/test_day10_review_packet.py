from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import build_day10_review_packet as packet


class TestDay10ReviewPacket(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = packet.build_review_rows()
        analyzer = packet._load_analyzer()
        manifest, manifest_file_sha256 = analyzer.load_manifest(packet.DEFAULT_MANIFEST)
        cls.manifest_by_id, _, _ = analyzer.verify_manifest(
            manifest, manifest_file_sha256
        )
        cls.predictions = analyzer.load_predictions(packet.DEFAULT_PREDICTIONS)
        cls.code_results, _ = analyzer.load_code_results(
            packet.DEFAULT_CODE_RESULTS
        )
        cls.code_results_by_id = {
            row["sample_id"]: row for row in cls.code_results
        }
        cls.candidates = packet._prepare_candidates(
            cls.manifest_by_id, cls.predictions, cls.code_results_by_id
        )

    def test_packet_is_exactly_30_pending_dev_rows(self) -> None:
        self.assertEqual(len(self.rows), 30)
        self.assertEqual(len({row["sample_id"] for row in self.rows}), 30)
        self.assertTrue(all(row["evaluation_split"] == "dev" for row in self.rows))
        self.assertTrue(
            all(
                row["human_review"]
                == {"status": "pending", "judgment": None, "notes": None}
                for row in self.rows
            )
        )

    def test_all_automated_correct_rows_are_selected(self) -> None:
        expected = {
            candidate["sample_id"]
            for candidate in self.candidates
            if candidate["automated_category"]["primary"] == "correct"
        }
        actual = {row["sample_id"] for row in self.rows}
        self.assertEqual(len(expected), 15)
        self.assertLessEqual(expected, actual)

    def test_observed_error_classes_and_slices_are_covered(self) -> None:
        expected_pairs = {
            (candidate["slice"], candidate["automated_category"]["primary"])
            for candidate in self.candidates
            if candidate["automated_category"]["primary"]
            in packet.ERROR_CATEGORIES
        }
        actual_pairs = {
            (row["slice"], row["automated_category"])
            for row in self.rows
        }
        self.assertLessEqual(expected_pairs, actual_pairs)
        for slice_name in packet.SLICES:
            self.assertGreaterEqual(
                sum(row["slice"] == slice_name for row in self.rows),
                packet.MINIMUM_PER_SLICE,
            )

    def test_ceiling_degeneration_candidates_are_present(self) -> None:
        ceiling_rows = [
            row
            for row in self.rows
            if row["generation_ceiling_hit"]
        ]
        self.assertTrue(ceiling_rows)
        self.assertTrue(
            all(
                "ceiling_degeneration_candidate"
                in row["automated_flags"]
                for row in ceiling_rows
            )
        )

    def test_required_review_payload_is_copied_from_frozen_sources(self) -> None:
        predictions_by_id = {row["sample_id"]: row for row in self.predictions}
        for row in self.rows:
            manifest_record = self.manifest_by_id[row["sample_id"]]
            prediction = predictions_by_id[row["sample_id"]]
            self.assertEqual(row["raw_prompt"], manifest_record["raw_prompt"])
            self.assertEqual(row["reference"], manifest_record["reference"])
            self.assertEqual(row["raw_output"], prediction["raw_output"])
            self.assertEqual(row["scorer_result"], prediction["scorer_result"])
            code_result = self.code_results_by_id.get(row["sample_id"])
            if row["slice"] == "code":
                self.assertIsNotNone(code_result)
                self.assertIn(
                    row["automated_category"], {"syntax_error", "runtime_error"}
                )
                self.assertNotEqual(row["automated_category"], "sandbox_pending")
                self.assertEqual(
                    row["code_execution_result"],
                    {
                        "execution_status": code_result["execution_status"],
                        "score_status": code_result["score_status"],
                        "score": code_result["score"],
                        "error_type": code_result["error_type"],
                        "failure_message": code_result["failure_message"],
                        "scorer_result": code_result["scorer_result"],
                        "code_run_hash": code_result["code_run_hash"],
                        "complete_comparison_key": code_result[
                            "complete_comparison_key"
                        ],
                        "code_execution_protocol_hash": code_result[
                            "code_execution_protocol_hash"
                        ],
                    },
                )
            else:
                self.assertIsNone(row["code_execution_result"])
            self.assertEqual(
                row["token_counts"],
                {
                    "input": prediction["input_token_count"],
                    "output": prediction["output_token_count"],
                    "total": prediction["total_token_count"],
                    "generation_max_new_tokens": manifest_record[
                        "generation_max_new_tokens"
                    ],
                },
            )

    def test_selection_is_independent_of_candidate_input_order(self) -> None:
        selected_a, reasons_a = packet.select_candidates(self.candidates)
        selected_b, reasons_b = packet.select_candidates(
            list(reversed(self.candidates))
        )
        self.assertEqual(
            [row["sample_id"] for row in selected_a],
            [row["sample_id"] for row in selected_b],
        )
        self.assertEqual(reasons_a, reasons_b)

    def test_selection_ranks_and_source_hashes_are_bound(self) -> None:
        selection_hashes = {
            row["selection_policy"]["selection_hash"] for row in self.rows
        }
        self.assertEqual(len(selection_hashes), 1)
        for row in self.rows:
            self.assertEqual(
                row["selection"]["rank_sha256"],
                packet.selection_rank(row["sample_id"]),
            )
            self.assertEqual(
                row["source"]["manifest"]["file_sha256"],
                packet.file_sha256(packet.DEFAULT_MANIFEST),
            )
            self.assertEqual(
                row["source"]["predictions"]["file_sha256"],
                packet.file_sha256(packet.DEFAULT_PREDICTIONS),
            )
            self.assertEqual(
                row["source"]["code_results"]["file_sha256"],
                packet.file_sha256(packet.DEFAULT_CODE_RESULTS),
            )
            self.assertEqual(
                row["source"]["identities"]["complete_comparison_key"],
                self.code_results[0]["complete_comparison_key"],
            )
            self.assertEqual(
                row["source"]["identities"]["code_execution_protocol_hash"],
                self.code_results[0]["code_execution_protocol_hash"],
            )
            self.assertEqual(
                row["source"]["validation"]["source_sha256"],
                packet.file_sha256(packet.ANALYZER_PATH),
            )

    def test_committed_packet_is_a_byte_exact_rebuild(self) -> None:
        expected = packet.serialize_rows(self.rows)
        actual = packet.DEFAULT_OUTPUT.read_bytes()
        self.assertEqual(actual, expected)
        self.assertEqual(
            hashlib.sha256(actual).hexdigest(),
            "9b7da2940950a8ef8a34da4ef6bb404046e6d04f557df0241c9072dc6e5837df",
        )

    def test_truncated_predictions_fail_closed(self) -> None:
        lines = packet.DEFAULT_PREDICTIONS.read_text(encoding="utf-8").splitlines()
        repo_tmp = packet.REPO_ROOT / "tmp"
        repo_tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=repo_tmp) as temporary_directory:
            tampered = Path(temporary_directory) / "truncated-predictions.jsonl"
            tampered.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                packet.ReviewPacketError, "baseline source validation failed"
            ):
                packet.build_review_rows(packet.DEFAULT_MANIFEST, tampered)

    def test_truncated_code_results_fail_closed(self) -> None:
        lines = packet.DEFAULT_CODE_RESULTS.read_text(encoding="utf-8").splitlines()
        repo_tmp = packet.REPO_ROOT / "tmp"
        repo_tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=repo_tmp) as temporary_directory:
            tampered = Path(temporary_directory) / "truncated-code-results.jsonl"
            tampered.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                packet.ReviewPacketError, "baseline source validation failed"
            ):
                packet.build_review_rows(
                    packet.DEFAULT_MANIFEST,
                    packet.DEFAULT_PREDICTIONS,
                    tampered,
                )

    def test_day12_policy_pins_complete_code_comparison_identity(self) -> None:
        policy_path = (
            packet.REPO_ROOT
            / "learning/post-training-30-day-bootcamp/artifacts/configs/"
            "day12-checkpoint-selection-policy.json"
        )
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        contract = policy["evaluation_contract"]
        self.assertEqual(
            contract["complete_comparison_key"],
            self.code_results[0]["complete_comparison_key"],
        )
        self.assertEqual(
            contract["code_execution_protocol_hash"],
            self.code_results[0]["code_execution_protocol_hash"],
        )
        self.assertTrue(
            contract["required_same_complete_comparison_key_within_contrast"]
        )

    def test_writer_refuses_implicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "packet.jsonl"
            packet.write_packet_atomic(self.rows, output, overwrite=False)
            with self.assertRaisesRegex(packet.ReviewPacketError, "already exists"):
                packet.write_packet_atomic(self.rows, output, overwrite=False)
            packet.verify_packet(self.rows, output)


if __name__ == "__main__":
    unittest.main()
