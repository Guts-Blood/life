import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).with_name("run_day11_training.py")
SPEC = importlib.util.spec_from_file_location("run_day11_training", MODULE_PATH)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class FrozenInputTest(unittest.TestCase):
    def test_manifest_hash_and_dataset_hash_are_valid(self):
        config = RUNNER.load_config(RUNNER.DEFAULT_CONFIG_PATH)
        data_path = RUNNER.resolve_project_path(config["data"]["path"])
        manifest_path = RUNNER.resolve_project_path(config["data"]["manifest_path"])
        manifest = RUNNER.load_json(manifest_path)
        RUNNER.verify_manifest_hash(manifest)
        self.assertEqual(
            RUNNER.file_sha256(data_path), manifest["header"]["dataset_file_sha256"]
        )
        self.assertEqual(manifest["header"]["record_count"], 18)
        self.assertEqual(manifest["header"]["total_supervised_tokens"], 904)

    def test_manifest_tamper_is_rejected(self):
        config = RUNNER.load_config(RUNNER.DEFAULT_CONFIG_PATH)
        manifest = RUNNER.load_json(
            RUNNER.resolve_project_path(config["data"]["manifest_path"])
        )
        tampered = copy.deepcopy(manifest)
        tampered["records"][0]["supervised_tokens"] += 1
        with self.assertRaisesRegex(RUNNER.Day11TrainingError, "manifest hash mismatch"):
            RUNNER.verify_manifest_hash(tampered)


class TrainingSemanticsTest(unittest.TestCase):
    def test_sampler_cursor_resume_matches_uninterrupted_order(self):
        uninterrupted = RUNNER.deterministic_indices(18, 0, 24, 20260806)
        interrupted = RUNNER.deterministic_indices(18, 0, 12, 20260806)
        resumed = RUNNER.deterministic_indices(18, 12, 12, 20260806)
        self.assertEqual(uninterrupted, interrupted + resumed)

    def test_masked_loss_uses_only_shifted_nonignored_targets(self):
        logits = torch.tensor(
            [
                [
                    [4.0, 0.0, 0.0],
                    [0.0, 4.0, 0.0],
                    [0.0, 0.0, 4.0],
                    [4.0, 0.0, 0.0],
                ]
            ]
        )
        labels = torch.tensor([[RUNNER.IGNORE_INDEX, 0, RUNNER.IGNORE_INDEX, 2]])
        loss_sum, token_count, correct = RUNNER.masked_causal_loss_sum(logits, labels)
        expected = torch.nn.functional.cross_entropy(
            torch.stack((logits[0, 0], logits[0, 2])), torch.tensor([0, 2]), reduction="sum"
        )
        self.assertTrue(torch.equal(loss_sum, expected))
        self.assertEqual(token_count, 2)
        self.assertEqual(correct, 2)

    def test_warmup_first_optimizer_step_has_nonzero_lr_and_scheduler_resumes(self):
        config = RUNNER.load_config(RUNNER.DEFAULT_CONFIG_PATH)
        model = torch.nn.Linear(2, 2)
        optimizer, scheduler = RUNNER.make_optimizer_and_scheduler(model, config)
        first_lr = optimizer.param_groups[0]["lr"]
        self.assertGreater(first_lr, 0.0)
        self.assertLess(first_lr, config["optimizer"]["learning_rate"])
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        scheduler.step()
        state_tensors = [
            value
            for state in optimizer.state.values()
            for key, value in state.items()
            if key in {"exp_avg", "exp_avg_sq"}
        ]
        self.assertTrue(state_tensors)
        self.assertTrue(all(value.dtype == torch.float32 for value in state_tensors))
        optimizer_state = optimizer.state_dict()
        scheduler_state = scheduler.state_dict()

        restored_model = torch.nn.Linear(2, 2)
        restored_optimizer, restored_scheduler = RUNNER.make_optimizer_and_scheduler(
            restored_model, config
        )
        restored_optimizer.load_state_dict(optimizer_state)
        restored_scheduler.load_state_dict(scheduler_state)
        self.assertEqual(restored_scheduler.state_dict(), scheduler_state)
        self.assertEqual(restored_optimizer.param_groups[0]["lr"], optimizer.param_groups[0]["lr"])

    def test_parameter_checksum_changes_with_parameter(self):
        model = torch.nn.Linear(2, 1, bias=False)
        before = RUNNER.parameter_checksum(model)
        with torch.no_grad():
            model.weight[0, 0] += 1
        self.assertNotEqual(before, RUNNER.parameter_checksum(model))


class ResumeComparisonTest(unittest.TestCase):
    @staticmethod
    def step_record(step):
        return {
            "optimizer_step": step,
            "sample_cursor": step * 4,
            "sample_ids": [f"sample-{step}"],
            "window_supervised_tokens": step + 10,
            "global_supervised_tokens": step * 100,
            "optimizer_lr": [0.0001],
            "scheduler_lr_after_step": [0.0001],
            "parameter_checksum_after": f"sha256:{step}",
            "loss_sum": float(step),
            "mean_loss": float(step) / 10,
            "grad_norm_before_clip": 1.0,
            "clip_coefficient": 1.0,
        }

    def test_resume_comparison_passes_equal_runs_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference"
            resumed = root / "resumed"
            reference.mkdir()
            resumed.mkdir()
            records = [self.step_record(step) for step in range(1, 7)]
            payload = "".join(RUNNER.canonical_json(record) + "\n" for record in records)
            (reference / "step-trace.jsonl").write_text(payload, encoding="utf-8")
            (resumed / "step-trace.jsonl").write_text(payload, encoding="utf-8")
            state = {
                "optimizer_step": 6,
                "sample_cursor": 24,
                "global_supervised_tokens": 600,
                "parameter_checksum": "sha256:model",
                "optimizer_checksum": "sha256:optimizer",
                "scheduler_state_hash": "sha256:scheduler",
                "scheduler_last_epoch": 6,
                "lr": [0.0001],
            }
            RUNNER.write_json(reference / "final-state-signature.json", state)
            RUNNER.write_json(resumed / "final-state-signature.json", state)
            args = type(
                "Args",
                (),
                {
                    "reference_run": reference,
                    "resumed_run": resumed,
                    "interruption_step": 3,
                    "total_steps": 6,
                    "output_json": root / "result.json",
                    "output_markdown": root / "result.md",
                },
            )()
            RUNNER.compare_resume(args)
            self.assertEqual(json.loads((root / "result.json").read_text())["status"], "pass")

            drifted = copy.deepcopy(records)
            drifted[-1]["sample_cursor"] += 1
            (resumed / "step-trace.jsonl").write_text(
                "".join(RUNNER.canonical_json(record) + "\n" for record in drifted),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RUNNER.Day11TrainingError, "resume comparison failed"):
                RUNNER.compare_resume(args)


if __name__ == "__main__":
    unittest.main()
