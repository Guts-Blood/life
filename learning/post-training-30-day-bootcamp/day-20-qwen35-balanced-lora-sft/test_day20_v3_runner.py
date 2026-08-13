#!/usr/bin/env python3
"""Static safety checks for the standalone Day 20 v3 AutoDL runner."""

from __future__ import annotations

import hashlib
import subprocess
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_day20_v3_autodl.sh"
FROZEN_SHA256 = {
    "day20_train_plugin_v2.py": "7129e5e3606756f242129539c740c303a13806ee636170df8ac8a86b070fbf5e",
    "day20_train_runtime_v2.py": "8473e5d4fe55f0d072f6f112418c0439809452c206d3a7b0f2c4cc7ae899de9d",
    "run_day20_v2_autodl.sh": "71560b68534fb15032aa562a63640c42baeedad93d730cfc9485e7a24e1efb95",
    "evaluate_day20_v2.py": "4da1e7bed95a628e6914711842001b491b25e821ba6711c7bafac8c08e837963",
    "rescore_day20_qwen35_v3.py": "0f8cffad3265fa85af920d5e0a8c2e88554b8c49c91d85f9a3885e899c7b06f2",
    "score_day20_code_e2b_v3.py": "cda49ee80e356ee4f419b6de512cca95a728998a7c88c3511bce373dd0a08784",
    "select_day20_v2.py": "99887e23150c7b0ff34185dfd89fd69e7cc5c8671ece699b53c75a469786a3fe",
    "select_day20_main_v2.py": "6e861507286bde99128d94f6b9f9c0ec04b229c656f99c68a79e03b3b2125187",
    "day20_candidate_factory_v2.py": "92b9b575b5ac80d6c896924de4e930badd26be98633dcea133347263dda144de",
    "day20_e2b_preflight_v2.py": "f3ed7045da293488ebb77c175ea3757253270dfb8c6217794ce423de1e0ee9d6",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Day20V3RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNNER.read_text(encoding="utf-8")

    def test_shell_and_independent_v3_run_namespace(self) -> None:
        subprocess.run(["bash", "-n", str(RUNNER)], check=True)
        self.assertIn("current-day20-v3-run-root", self.source)
        self.assertIn("day20-v3-qwen35-lora-", self.source)
        self.assertIn(".day20-v3-run-root", self.source)
        self.assertIn("day20-qwen35-target-encoding-v3", self.source)
        self.assertNotIn("run_day20_v2_autodl.sh\"", self.source)
        init = self.source[self.source.index("init_run() {") : self.source.index("prepare_run() {")]
        self.assertNotIn('mkdir -p "${run_root}"', init)
        prepare = self.source[
            self.source.index("prepare_run() {") :
            self.source.index("run_e2b_preflight_unlocked() {")
        ]
        self.assertIn("load_planned_run_root", prepare)
        self.assertLess(prepare.index('require_absent "${RUN_ROOT}"'), prepare.index('"${PYTHON_BIN}" "${PREPARER}"'))

    def test_training_uses_alias_and_both_callbacks_but_eval_is_native(self) -> None:
        training = self.source[
            self.source.index("run_training() {") : self.source.index("train_probe() {")
        ]
        evaluate = self.source[
            self.source.index("evaluate_candidate() {") :
            self.source.index("sandbox_candidate() {")
        ]
        self.assertIn("--template day20_qwen3_5_target_v3", training)
        self.assertIn('DAY20_V3_REGISTER_SWIFT_PLUGIN=1', training)
        self.assertIn('DAY20_V2_REGISTER_SWIFT_CALLBACK=1', training)
        self.assertIn(
            "--callbacks day20_v2_evidence day20_v3_target_binding", training
        )
        self.assertIn('--external_plugins "${TRAIN_PLUGIN}"', training)
        self.assertIn("evaluate_day20_v2.py", self.source)
        self.assertIn('--model "${BASE_MODEL}"', evaluate)
        self.assertNotIn("day20_qwen3_5_target_v3", evaluate)
        self.assertNotIn("--template", evaluate)
        self.assertIn(
            '--experiment-manifest "${RUN_ROOT}/DAY20-V2-MANIFEST.json"',
            evaluate,
        )
        checkpoint = self.source[
            self.source.index("checkpoint_for_candidate() {") :
            self.source.index("evaluate_candidate() {")
        ]
        self.assertIn("verify_sealed_attempt", checkpoint)

    def test_prepared_target_audit_and_e2b_are_hard_gates_before_swift(self) -> None:
        gate = self.source[
            self.source.index("require_training_gate() {") :
            self.source.index("json_field() {")
        ]
        self.assertLess(gate.index("verify_prepared_run_unlocked"), gate.index("E2B_PREFLIGHT"))
        self.assertLess(gate.index("E2B_PREFLIGHT"), gate.index("system_preflight_unlocked"))
        training = self.source[
            self.source.index("run_training() {") : self.source.index("train_probe() {")
        ]
        self.assertLess(training.index("require_training_gate"), training.index("swift.cli.sft"))
        prepared = self.source[
            self.source.index("verify_prepared_run_unlocked() {") :
            self.source.index("resolve_main_config() {")
        ]
        self.assertIn("DAY20-V3-MANIFEST.json", prepared)
        self.assertIn("DAY20-V2-MANIFEST.json", prepared)
        self.assertIn("TARGET-ENCODING-AUDIT.json", prepared)
        self.assertIn("validate-prepared", prepared)

    def test_probe_selector_and_eval_sandbox_pipeline_remain_frozen(self) -> None:
        funnel = self.source[
            self.source.index("probe_funnel() {") :
            self.source.index("verified_probe_selection_field() {")
        ]
        self.assertIn("eval base-probe", funnel)
        self.assertIn("run_probe_candidate_cohort 1e-4", funnel)
        self.assertIn('if [[ "${status}" == run_bracket ]]', funnel)
        conditional = funnel[funnel.index('if [[ "${status}" == run_bracket ]]') :]
        self.assertIn("for learning_rate in 3e-5 6e-5 8e-5", conditional)
        factory = (HERE / "day20_candidate_factory_v2.py").read_text(encoding="utf-8")
        self.assertIn('row["classification"] == "near_miss"', factory)
        self.assertIn('action = "run_bracket"', factory)
        self.assertIn("rescore_day20_qwen35_v3.py", self.source)
        self.assertIn("score_day20_code_e2b_v3.py", self.source)
        self.assertIn("select_day20_v2.py", self.source)

    def test_frozen_v2_and_eval_selection_files_are_byte_unchanged(self) -> None:
        actual = {name: sha256(HERE / name) for name in FROZEN_SHA256}
        self.assertEqual(actual, FROZEN_SHA256)


if __name__ == "__main__":
    unittest.main()
