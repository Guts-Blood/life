#!/usr/bin/env python3
"""Static safety checks for the standalone Day 20 v2 AutoDL runner."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_day20_v2_autodl.sh"


class Day20V2RunnerTests(unittest.TestCase):
    def test_shell_is_valid_and_v1_is_not_mutated_through_a_shim(self) -> None:
        subprocess.run(["bash", "-n", str(RUNNER)], check=True)
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("day20_train_plugin_v2.py", source)
        self.assertIn("prepare_day20_v2.py", source)
        self.assertNotIn("probe-1e-5", source)
        self.assertNotIn("DAY20_REGISTER_SWIFT_CALLBACK", source)

    def test_e2b_attestation_is_a_hard_gate_before_swift(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        body = source[source.index("run_training() {") : source.index("train_probe() {")]
        self.assertLess(body.index("require_training_gate"), body.index("swift.cli.sft"))
        self.assertIn("E2B-PREFLIGHT.json", source)
        self.assertIn("--save_only_model false", body)
        self.assertIn("--callbacks day20_v2_evidence", body)

    def test_probe_and_main_are_namespaced_by_seed_and_lr(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn('adapters/probe/s${seed}/lr${learning_rate}', source)
        self.assertIn('adapters/main/s${seed}/lr${learning_rate}', source)
        self.assertIn("1e-4|3e-5|6e-5|8e-5", source)

    def test_preflight_checks_gpu_runtime_data_and_live_e2b(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        body = source[
            source.index("system_preflight_unlocked() {") :
            source.index("require_training_gate() {")
        ]
        self.assertIn("nvidia-smi", body)
        self.assertIn("runtime_identity", body)
        self.assertIn("verify_prepared_run_unlocked", body)
        self.assertIn("validate-inputs", source)
        self.assertIn("run_e2b_preflight_unlocked", body)
        self.assertIn("preflight)", source)

    def test_eval_and_sandbox_use_v3_identity_and_secret_subshell(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("from day20_candidate_factory_v2 import parse_candidate_id", source)
        self.assertIn("evaluate_day20_v2.py", source)
        self.assertIn("rescore_day20_qwen35_v3.py", source)
        self.assertIn("score_day20_code_e2b_v3.py", source)
        self.assertIn(".qwen35-v3.predictions.jsonl", source)
        sandbox = source[
            source.index("sandbox_candidate() {") : source.index("status() {")
        ]
        self.assertIn("(\n    local key_line", sandbox)
        self.assertIn('exec "${SANDBOX_PYTHON}"', sandbox)

    def test_probe_funnel_brackets_only_a_stage_a_near_miss(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        body = source[source.index("probe_funnel() {") : source.index("status() {")]
        self.assertLess(
            body.index("run_probe_candidate_cohort 1e-4"),
            body.index('if [[ "${status}" == run_bracket ]]'),
        )
        conditional = body[body.index('if [[ "${status}" == run_bracket ]]') :]
        self.assertIn("for learning_rate in 3e-5 6e-5 8e-5", conditional)
        self.assertIn("select-probe bracket", conditional)

    def test_main_confirmation_and_finalize_are_evidence_driven(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("select_day20_main_v2.py", source)
        main = source[source.index("main_stage() {") : source.index("primary_selection_field() {")]
        self.assertIn("verified_probe_selection_field", main)
        self.assertIn("main_candidate_ids 20260809", main)
        confirm = source[source.index("confirmation_stage() {") : source.index("finalize_run() {")]
        self.assertIn("CONFIRMATION_SEED", confirm)
        self.assertIn("checkpoint_label", confirm)
        finalizer = source[source.index("finalize_run() {") : source.index("status() {")]
        self.assertIn("verify-final", finalizer)
        self.assertIn("merge_performed=false", finalizer)


if __name__ == "__main__":
    unittest.main()
