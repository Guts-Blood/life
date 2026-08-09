#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


DAY18_DIR = Path(__file__).resolve().parent
BOOTCAMP_ROOT = DAY18_DIR.parent
CONFIG_DIR = BOOTCAMP_ROOT / "artifacts" / "configs" / "day18-qwen35-megatron"
FIXTURE = BOOTCAMP_ROOT / "artifacts" / "data" / "day18-qwen35-golden-text-2.jsonl"


class Day18PreparationTest(unittest.TestCase):
    def test_fixture_and_contract_are_frozen_together(self) -> None:
        rows = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
        contract = json.loads((CONFIG_DIR / "run-contract.json").read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 2)
        self.assertEqual(contract["fixture"]["rows"], 2)
        self.assertEqual(contract["fixture"]["max_length"], 512)
        rendered = json.dumps(rows, ensure_ascii=False).lower()
        for marker in ("<image>", "<video>", "<audio>"):
            self.assertNotIn(marker, rendered)
        for row in rows:
            self.assertIn("messages", row)
            self.assertEqual(row["messages"][-1]["role"], "assistant")

    def test_topologies_preserve_global_batch(self) -> None:
        contract = json.loads((CONFIG_DIR / "run-contract.json").read_text(encoding="utf-8"))
        gates = contract["gates"]
        self.assertEqual(gates["C2"]["tp"], 1)
        self.assertEqual(gates["C2"]["dp"], 2)
        self.assertEqual(gates["C3"]["tp"], 2)
        self.assertEqual(gates["C3"]["dp"], 1)
        self.assertEqual(contract["training"]["global_batch_size"], 2)
        self.assertEqual(gates["C4"]["resume_from_iteration"], 3)
        self.assertEqual(gates["C4"]["train_iters"], 5)
        self.assertEqual(gates["C5"]["tp"], 2)
        self.assertEqual(gates["C5"]["dp"], 1)
        self.assertEqual(gates["C5"]["train_iters"], 150)
        self.assertIn("model-only", gates["C5"]["checkpoint"])

    def test_candidate_dependencies_are_exact_direct_pins(self) -> None:
        requirements = (DAY18_DIR / "requirements-day18-candidate.txt").read_text(encoding="utf-8")
        required = (
            "transformers==5.12.1",
            "megatron-core==0.18.0",
            "mcore-bridge==1.6.0",
            "flash-linear-attention==0.4.2",
            "flash-attn==2.8.3",
            "qwen-vl-utils==0.0.14",
            "peft==0.19.1",
            "trl==0.29.1",
            "accelerate==1.14.0",
            "4f6ae4e26ae5fe8af9372f8d312ab25cc4595223",
        )
        for value in required:
            self.assertIn(value, requirements)
        self.assertNotIn("@main", requirements)
        self.assertNotIn(">=", "\n".join(line for line in requirements.splitlines() if not line.startswith("#")))

    def test_checkpoint_model_key_classifier_accepts_mcore_018_fqns(self) -> None:
        from audit_day18_checkpoint import is_model_state_key

        self.assertTrue(is_model_state_key("language_model.decoder.layers.0.mlp.linear_fc1.weight"))
        self.assertTrue(is_model_state_key("language_model.mtp.layers.0.eh_proj.weight"))
        self.assertTrue(is_model_state_key("visual.visual.blocks.0.attn.qkv.weight"))
        self.assertFalse(is_model_state_key("optimizer.distributed.dp_group_idx_0.exp_avg"))
        self.assertFalse(is_model_state_key("rng_state/shard_0.0_1.1"))

    def test_runner_saves_and_loads_optimizer_rng(self) -> None:
        runner = (DAY18_DIR / "run_day18_autodl.sh").read_text(encoding="utf-8")
        self.assertIn('--no_save_optim "${no_save_optim}"', runner)
        self.assertIn('--no_save_rng "${no_save_rng}"', runner)
        self.assertIn("build_training_common true true", runner)
        self.assertIn("--finetune false", runner)
        self.assertIn("--no_load_optim false", runner)
        self.assertIn("--no_load_rng false", runner)
        self.assertNotIn("--no_save_optim true", runner)
        self.assertNotIn("--no_save_rng true", runner)
        self.assertEqual(runner.count("-m torch.distributed.run"), 10)
        self.assertEqual(runner.count('--nproc_per_node=2 "${MEGATRON_SFT}"'), 5)
        self.assertIn("PIP_CACHE_DIR", runner)
        self.assertIn("TMPDIR", runner)
        self.assertIn("--expected-start 3 --expected-final 5 --expected-steps 2", runner)
        self.assertIn("require_gate c2-dp2", runner)
        self.assertIn("require_gate c3-tp2", runner)
        self.assertEqual(runner.count("--mtp_num_layers 1"), 5)
        self.assertEqual(runner.count("--model_type qwen3_5"), 5)
        hf_reference = (DAY18_DIR / "day18_hf_reference.py").read_text(encoding="utf-8")
        self.assertIn('model_type="qwen3_5"', hf_reference)
        self.assertLess(hf_reference.index("template.model = model"), hf_reference.index("template.data_collator(encoded)"))
        text_export = (DAY18_DIR / "day18_text_export.py").read_text(encoding="utf-8")
        sft_wrapper = (DAY18_DIR / "day18_megatron_sft.py").read_text(encoding="utf-8")
        compat = (DAY18_DIR / "day18_transformers_compat.py").read_text(encoding="utf-8")
        self.assertIn("install_hf_argparser_compat()", text_export)
        self.assertLess(text_export.index('template.set_mode("train")'), text_export.index("template.data_collator([encoded])"))
        self.assertIn("convert_utils._test_params_sum = original_params_sum", text_export)
        self.assertIn("megatron_sft_main()", sft_wrapper)
        self.assertIn('field.name == "cp_comm_type"', compat)
        self.assertIn("c1-single-rank-loss-parity.json", runner)
        self.assertIn("c2-dp2-loadcheck", runner)
        self.assertIn("c2-verify-existing", runner)
        self.assertIn("--checkpoint-kind model-only", runner)
        self.assertIn("require_gate c4-resume-export", runner)
        self.assertIn("finalize_day18_run.py", runner)
        self.assertIn("DAY18-PASS.json", runner)
        self.assertIn("build_day18_codepath_manifest.py", runner)
        self.assertLess(
            runner.index('"${PYTHON_BIN}" "${CODEPATH_MANIFEST}"'),
            runner.index('"${PYTHON_BIN}" "${FINALIZER}" finalize'),
        )

    def test_runtime_codepath_manifest_has_fail_closed_source_and_evidence_nodes(self) -> None:
        from build_day18_codepath_manifest import resolve_symbol, source_record
        from finalize_day18_run import REQUIRED_JSON_EVIDENCE

        manifest = (DAY18_DIR / "build_day18_codepath_manifest.py").read_text(encoding="utf-8")
        for node_id in (
            "conditional_loader",
            "processor_template",
            "conversion_provider",
            "gdn",
            "batch_labels_loss",
            "backward_optimizer",
            "distributed_save_load",
            "hf_export",
        ):
            self.assertIn(f'"{node_id}"', manifest)
        self.assertEqual(REQUIRED_JSON_EVIDENCE["codepath-runtime-evidence.json"], "pass")
        self.assertIs(resolve_symbol("json:loads"), json.loads)
        source = source_record("json:loads")
        self.assertTrue(Path(source["source_file"]).is_file())
        with self.assertRaises((AttributeError, ModuleNotFoundError)):
            resolve_symbol("json:missing_day18_symbol")

    def test_upload_consumes_both_internal_manifests(self) -> None:
        uploader = (DAY18_DIR / "upload_day18_bundles.sh").read_text(encoding="utf-8")
        self.assertIn("cd '${REMOTE_SOURCE}' && sha256sum -c MANIFEST.sha256", uploader)
        self.assertIn("cd '${REMOTE_MODEL}' && sha256sum -c MANIFEST.sha256", uploader)
        self.assertIn("refusing non-empty extraction target", uploader)
        packager = (DAY18_DIR / "package_qwen35_snapshot.sh").read_text(encoding="utf-8")
        self.assertIn('print(v["sha256"], "  ", k, sep="")', packager)

    def test_resource_and_autodl_runtime_contract_are_fail_closed(self) -> None:
        contract = json.loads((CONFIG_DIR / "run-contract.json").read_text(encoding="utf-8"))
        self.assertEqual(contract["resource_envelope"]["minimum_remote_data_disk_gb"], 350)
        self.assertEqual(contract["resource_envelope"]["minimum_free_gib_after_upload_and_extract"], 280)
        self.assertEqual(contract["resource_envelope"]["allowed_gpu_models"], ["H800", "H100"])
        self.assertEqual(contract["resource_envelope"]["required_compute_capability"], [9, 0])
        self.assertEqual(
            contract["candidate_container"]["boot_image_reference"],
            "autodl-observed-runtime:ubuntu22.04.4-py3.12.3-torch2.5.1+cu124-cuda12.4",
        )
        self.assertFalse(contract["candidate_container"]["oci_digest_exposed_by_provider"])
        preflight = (DAY18_DIR / "preflight_day18.py").read_text(encoding="utf-8")
        self.assertIn("AutoDL boot image reference drift", preflight)
        self.assertIn("AutoDL boot runtime drift", preflight)
        self.assertIn('SUPPORTED_GPU_MODELS = ("H800", "H100")', preflight)
        self.assertIn("EXPECTED_COMPUTE_CAPABILITY = (9, 0)", preflight)
        self.assertIn("--query-gpu=index,uuid,mig.mode.current", preflight)
        runner = (DAY18_DIR / "run_day18_autodl.sh").read_text(encoding="utf-8")
        self.assertIn("at least 280 GiB free", runner)
        self.assertIn("audit_day18_checkpoint.py", runner)
        self.assertIn("/root/autodl-tmp/Qwen--Qwen3.5-4B-Base/", runner)
        self.assertNotIn("python3.12 -m venv --system-site-packages", runner)
        self.assertIn("missing isolated Day18 conda prefix", runner)
        self.assertIn("import nvidia.cudnn", runner)
        requirements = (DAY18_DIR / "requirements-day18-candidate.txt").read_text(encoding="utf-8")
        self.assertIn("datasets==4.8.4", requirements)
        self.assertNotIn("decord==", requirements)
        self.assertNotIn("transformer-engine[pytorch]", requirements)

    def test_rank_gate_accepts_complete_pass_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ranks = root / "ranks"
            ranks.mkdir()
            for rank in range(2):
                payload = {
                    "status": "pass",
                    "failures": [],
                    "rank": rank,
                    "world_size": 2,
                    "tp_world_size": 1,
                    "dp_world_size": 2,
                    "start_iteration": 0,
                    "final_iteration": 1,
                    "gdn_modules": [{"class": "mcore.GatedDeltaNet", "source": "/wheel/gdn.py"}],
                    "expected_mcore_bridge_gdn_modules": [{
                        "class": "mcore_bridge.model.modules.gated_delta_net.GatedDeltaNet",
                        "source": "/wheel/mcore_bridge/model/modules/gated_delta_net.py",
                    }],
                    "visual_parameter_tensors_local": 1,
                    "trainable_visual_parameter_tensors_local": 0,
                    "mtp_parameter_tensors_local": 1,
                    "trainable_mtp_parameter_tensors_local": 1,
                    "steps": [{
                        "iteration": 1,
                        "update_successful": True,
                        "trainable_tensors_with_gradient": 1,
                        "visual_tensors_with_gradient": 0,
                        "cuda_free_fraction": 0.5,
                        "mtp_gradient_stats": {
                            "parameter_tensors": 1,
                            "missing_gradient_tensors": 0,
                            "nonfinite_gradient_tensors": 0,
                            "nonzero_gradient_tensors": 1,
                        },
                    }],
                }
                (ranks / f"rank-{rank}.json").write_text(json.dumps(payload), encoding="utf-8")
            output = root / "summary.json"
            command = [
                sys.executable,
                str(DAY18_DIR / "verify_day18_logs.py"),
                "ranks",
                "--evidence-dir", str(ranks),
                "--expected-ranks", "2",
                "--expected-tp", "1",
                "--expected-dp", "2",
                "--expected-start", "0",
                "--expected-final", "1",
                "--expected-steps", "1",
                "--output", str(output),
            ]
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["status"], "pass")

            rank_zero_path = ranks / "rank-0.json"
            rank_zero = json.loads(rank_zero_path.read_text(encoding="utf-8"))
            rank_zero["steps"][0]["update_successful"] = False
            rank_zero_path.write_text(json.dumps(rank_zero), encoding="utf-8")
            failed = subprocess.run(command, check=False, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(any("optimizer update was not successful" in failure for failure in result["failures"]))

            rank_zero["steps"][0]["update_successful"] = True
            rank_zero["steps"][0]["iteration"] = 2
            rank_zero_path.write_text(json.dumps(rank_zero), encoding="utf-8")
            failed = subprocess.run(command, check=False, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(any("expected step iterations" in failure for failure in result["failures"]))

    def test_conversion_log_gate_passes_known_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            log = root / "conversion.log"
            output = root / "result.json"
            log.write_text(
                "mean_diff (with loss): 0.02, max_diff (with loss): 0.5\n"
                "token_diff (with loss): 0\n",
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    str(DAY18_DIR / "verify_day18_logs.py"),
                    "conversion",
                    "--log",
                    str(log),
                    "--thresholds",
                    str(CONFIG_DIR / "parity-thresholds.json"),
                    "--threshold-key",
                    "conversion_text_logits",
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["status"], "pass")

    def test_single_rank_mcore_loss_gate_aggregates_both_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            log = root / "conversion.log"
            reference = root / "hf.json"
            output = root / "loss.json"
            log.write_text(
                "DAY18_MCORE_MAIN_LOSS row=0 numerator=10 denominator=2 loss=5\n"
                "DAY18_MCORE_MAIN_LOSS row=1 numerator=20 denominator=4 loss=5\n",
                encoding="utf-8",
            )
            reference.write_text(json.dumps({"reference_loss": 5.0}), encoding="utf-8")
            subprocess.run(
                [
                    sys.executable,
                    str(DAY18_DIR / "verify_day18_logs.py"),
                    "mcore-loss",
                    "--log", str(log),
                    "--hf-reference", str(reference),
                    "--thresholds", str(CONFIG_DIR / "parity-thresholds.json"),
                    "--threshold-key", "hf_reference_vs_c1_mcore_loss",
                    "--expected-records", "2",
                    "--output", str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["status"], "pass")

    def test_c5_thresholds_are_preregistered_as_a_learnability_nonclaim(self) -> None:
        thresholds = json.loads((CONFIG_DIR / "parity-thresholds.json").read_text(encoding="utf-8"))
        c5 = thresholds["c5_tiny_overfit"]
        self.assertEqual(c5["minimum_optimizer_steps"], 50)
        self.assertEqual(c5["maximum_optimizer_steps"], 150)
        self.assertEqual(c5["teacher_forced_token_accuracy_min"], 0.95)
        self.assertGreaterEqual(c5["minimum_relative_loss_reduction"], 0.8)
        self.assertIn("not a generalization", thresholds["policy"])

    def test_parameter_scope_classifier_separates_trainable_and_frozen_weights(self) -> None:
        from compare_day18_parameter_scopes import parameter_scope

        self.assertEqual(parameter_scope("mtp.fc.weight"), "mtp")
        self.assertEqual(
            parameter_scope("model.visual.merger.linear_fc1.weight"),
            "visual_or_aligner",
        )
        self.assertEqual(
            parameter_scope("model.language_model.layers.0.mlp.down_proj.weight"),
            "main_language_model",
        )

    def test_tiny_overfit_gate_accepts_preregistered_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_path = root / "base.json"
            final_path = root / "final.json"
            log_path = root / "train.log"
            output = root / "result.json"

            def evaluation(model_path: str, loss: float, correct: int) -> dict:
                return {
                    "fixture_sha256": "fixture-hash",
                    "model_class": "Qwen3_5ForConditionalGeneration",
                    "model_path": model_path,
                    "forbidden_tensor_keys": [],
                    "forbidden_visual_token_ids": [],
                    "teacher_forced": {
                        "supervised_tokens": 100,
                        "correct_tokens": correct,
                        "token_accuracy": correct / 100,
                        "loss_weight_sum": 100.0,
                        "mean_loss": loss,
                        "rows": [
                            {"row": 0, "supervised_tokens": 40},
                            {"row": 1, "supervised_tokens": 60},
                        ],
                    },
                }

            base_path.write_text(json.dumps(evaluation("/base", 5.0, 10)), encoding="utf-8")
            final_path.write_text(json.dumps(evaluation("/exported", 0.5, 95)), encoding="utf-8")
            log_path.write_text(
                "\n".join("{'loss': 5.0, 'grad_norm': 1.0, 'mtp_0_loss': 0.5}" for _ in range(150)),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    str(DAY18_DIR / "verify_day18_logs.py"),
                    "tiny-overfit",
                    "--base", str(base_path),
                    "--final", str(final_path),
                    "--log", str(log_path),
                    "--thresholds", str(CONFIG_DIR / "parity-thresholds.json"),
                    "--expected-steps", "150",
                    "--output", str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "pass")
            self.assertIn("not generalization", result["claim_scope"])

    def test_finalizer_requires_every_gate_and_preserves_resolved_problems(self) -> None:
        from finalize_day18_run import REQUIRED_FILES, REQUIRED_GATES, REQUIRED_JSON_EVIDENCE

        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir) / "run"
            evidence = run_root / "evidence"
            gates = evidence / "gates"
            gates.mkdir(parents=True)
            (run_root / ".day18-run-root").write_text("fixture", encoding="utf-8")
            for gate in REQUIRED_GATES:
                (gates / f"{gate}.pass.json").write_text(
                    json.dumps({"gate": gate, "status": "pass"}),
                    encoding="utf-8",
                )
            for relative, expected_status in REQUIRED_JSON_EVIDENCE.items():
                path = evidence / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = {"status": expected_status} if expected_status is not None else {"schema_version": 1}
                path.write_text(json.dumps(payload), encoding="utf-8")
            for relative in REQUIRED_FILES:
                (evidence / relative).write_text("evidence", encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(DAY18_DIR / "finalize_day18_run.py"),
                    "problem",
                    "--run-root", str(run_root),
                    "--gate", "bootstrap",
                    "--category", "runtime_amendment",
                    "--status", "resolved",
                    "--summary", "Used the isolated CUDA 12.6 prefix.",
                    "--evidence", "evidence/runtime-preflight.json",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(DAY18_DIR / "finalize_day18_run.py"),
                    "finalize",
                    "--run-root", str(run_root),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads((run_root / "DAY18-PASS.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "day18_pass")
            self.assertEqual(result["open_problem_count"], 0)
            self.assertEqual(result["problems"][0]["status"], "resolved")

    def test_finalizer_rejects_problem_evidence_outside_run_root(self) -> None:
        from finalize_day18_run import validate_problem_evidence

        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = (Path(temp_dir) / "run").resolve()
            log = run_root / "logs" / "host-inventory.log"
            log.parent.mkdir(parents=True)
            log.write_text("inventory", encoding="utf-8")
            failures, paths = validate_problem_evidence(
                [{"evidence": ["logs/host-inventory.log", "../outside.log", "evidence/missing.json"]}],
                run_root,
            )
            self.assertEqual(paths, [log])
            self.assertTrue(any("escapes run root" in failure for failure in failures))
            self.assertTrue(any("does not exist" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
