import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).with_name("run_day10_baseline.py")
SPEC = importlib.util.spec_from_file_location("run_day10_baseline", MODULE_PATH)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class MockModel:
    dtype = torch.float32

    def __init__(self, continuations):
        self.continuations = continuations
        self.calls = []

    def eval(self):
        return self

    def generate(self, *, input_ids, attention_mask, **kwargs):
        self.calls.append(kwargs)
        self.assert_attention_mask = attention_mask.detach().cpu().tolist()
        marker = input_ids[0, -1].item()
        continuation = torch.tensor(
            [self.continuations[marker]], dtype=torch.long, device=input_ids.device
        )
        return torch.cat((input_ids, continuation), dim=1)


class MockTokenizer:
    def __init__(self, decoded):
        self.decoded = decoded

    def decode(self, token_ids, **kwargs):
        if kwargs != {
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
        }:
            raise AssertionError(f"unexpected decode kwargs: {kwargs}")
        return self.decoded[tuple(token_ids)]


def make_record(sample_id, split, slice_name, input_ids, reference):
    rendered_prompt = f"rendered {sample_id}"
    scorer = RUNNER._load_scorers(hashlib.sha256(RUNNER.SCORER_PATH.read_bytes()).hexdigest())
    scorer_spec = scorer.SCORER_REGISTRY["slices"][slice_name]
    return {
        "sample_id": sample_id,
        "evaluation_split": split,
        "slice": slice_name,
        "rendered_prompt": rendered_prompt,
        "rendered_prompt_hash": RUNNER.exact_text_hash(rendered_prompt),
        "input_ids": input_ids,
        "input_ids_hash": RUNNER.semantic_hash(
            {
                "domain": "day10.input_ids",
                "schema_version": 1,
                "input_ids": input_ids,
            }
        ),
        "input_token_count": len(input_ids),
        "reference": reference,
        "reference_hash": RUNNER.exact_text_hash(reference),
        "generation_max_new_tokens": 8,
        "extractor_version": scorer_spec["extractor_version"],
        "scorer_version": scorer_spec["scorer_version"],
    }


def make_manifest(records):
    dev_ids = [
        record["sample_id"] for record in records if record["evaluation_split"] == "dev"
    ]
    frozen_ids = [
        record["sample_id"]
        for record in records
        if record["evaluation_split"] == "frozen_test"
    ]
    scorer_sha = hashlib.sha256(RUNNER.SCORER_PATH.read_bytes()).hexdigest()
    manifest = {
        "header": {
            "domain": "day10.frozen_eval_manifest",
            "schema_version": 1,
            "status": "frozen_manifest_pre_baseline",
            "config_file_sha256": "1" * 64,
            "dataset_context_hash": "sha256:" + "6" * 64,
            "eval_suite_hash": "sha256:" + "7" * 64,
            "protocol_hash": "sha256:" + "2" * 64,
            "scorer_source_sha256": scorer_sha,
            "scorer_registry_hash": "sha256:" + "3" * 64,
            "environment_contract_sha256": "4" * 64,
            "environment_snapshot_sha256": "5" * 64,
            "model_and_rendering": {
                "model_id": "mock/model",
                "model_revision": "frozen-revision",
            },
            "generation": {
                "version": "day10_greedy_generation_v1",
                "do_sample": False,
                "num_beams": 1,
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "repetition_penalty": 1,
                "eos_token_id": 9,
                "pad_token_id": 0,
                "stop_token_ids": [9],
                "seed": 20260805,
            },
            "evaluation_order": {"dev": dev_ids, "frozen_test": frozen_ids},
        },
        "records": records,
    }
    manifest["header"]["manifest_hash"] = RUNNER.semantic_hash(manifest)
    return manifest


def write_manifest(path, manifest):
    path.write_text(json.dumps(manifest), encoding="utf-8")


def make_model_snapshot(path):
    path.mkdir()
    (path / "config.json").write_text('{"model_type":"mock"}', encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"mock weights")


class ManifestPolicyTest(unittest.TestCase):
    def test_selection_uses_frozen_dev_order_and_rejects_frozen_test(self):
        records = [
            make_record("dev-b", "dev", "general", [1, 11], "A. yes"),
            make_record("frozen-a", "frozen_test", "general", [1, 12], "A. yes"),
            make_record("dev-a", "dev", "general", [1, 13], "A. yes"),
        ]
        manifest = make_manifest(records)
        manifest["header"]["evaluation_order"]["dev"] = ["dev-a", "dev-b"]
        manifest["header"]["manifest_hash"] = RUNNER.semantic_hash(
            {"header": {k: v for k, v in manifest["header"].items() if k != "manifest_hash"}, "records": records}
        )
        RUNNER.verify_manifest(manifest)
        selected = RUNNER.select_records(manifest, sample_limit=1)
        self.assertEqual([record["sample_id"] for record in selected], ["dev-a"])
        with self.assertRaisesRegex(RUNNER.Day10RunnerError, "frozen_test access"):
            RUNNER.select_records(manifest, split="frozen_test")

    def test_five_sample_dry_run_covers_all_four_slices(self):
        records = []
        marker = 20
        references = {
            "general": "A. yes",
            "math": "work #### 1",
            "code": "unused",
            "finance": json.dumps(
                {"answer": 1, "answer_type": "count", "derivation": "", "scale": ""}
            ),
        }
        for slice_name in ("general", "math", "code", "finance"):
            for index in range(2):
                marker += 1
                records.append(
                    make_record(
                        f"dev-{slice_name}-{index}",
                        "dev",
                        slice_name,
                        [1, marker],
                        references[slice_name],
                    )
                )
        manifest = make_manifest(records)
        selected = RUNNER.select_records(manifest, sample_limit=5)
        self.assertEqual(
            [record["slice"] for record in selected],
            ["general", "math", "code", "finance", "general"],
        )
        self.assertEqual(
            RUNNER.selection_policy(selected, 5)["version"],
            "day10_dev_stratified_round_robin_v1",
        )

    def test_tampered_manifest_and_duplicate_json_keys_fail_closed(self):
        manifest = make_manifest(
            [make_record("dev-a", "dev", "general", [1, 13], "A. yes")]
        )
        manifest["records"][0]["input_ids"].append(99)
        with self.assertRaisesRegex(RUNNER.Day10RunnerError, "semantic hash mismatch"):
            RUNNER.verify_manifest(manifest)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "manifest.json"
            path.write_text('{"header": {}, "header": {}}', encoding="utf-8")
            with self.assertRaisesRegex(RUNNER.Day10RunnerError, "duplicate JSON key"):
                RUNNER.load_json(path)


class GenerationTest(unittest.TestCase):
    def test_generation_is_continuation_only_and_uses_greedy_contract(self):
        model = MockModel({7: [21, 22]})
        tokenizer = MockTokenizer({(21, 22): "Final answer: A"})
        kwargs = {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": 8,
            "eos_token_id": 9,
            "pad_token_id": 0,
            "repetition_penalty": 1,
        }
        result = RUNNER.generate_prediction(
            model, tokenizer, [5, 7], kwargs, device="cpu"
        )
        self.assertEqual(result["raw_output"], "Final answer: A")
        self.assertEqual(result["output_token_ids"], [21, 22])
        self.assertEqual(result["output_token_count"], 2)
        self.assertEqual(model.calls, [kwargs])
        self.assertEqual(model.assert_attention_mask, [[1, 1]])

    def test_manifest_model_file_hashes_are_a_required_subset(self):
        manifest = make_manifest(
            [make_record("dev-a", "dev", "general", [1, 13], "A. yes")]
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "model"
            make_model_snapshot(model_path)
            (model_path / "README.md").write_text("not runtime-critical", encoding="utf-8")
            config_hash = hashlib.sha256((model_path / "config.json").read_bytes()).hexdigest()
            manifest["header"]["model_and_rendering"]["model_files"] = {
                "config.json": config_hash
            }
            identity = RUNNER.model_snapshot_identity(manifest, model_path, "manifest")
            self.assertIn("README.md", identity["model_files"])
            manifest["header"]["model_and_rendering"]["model_files"]["config.json"] = "0" * 64
            with self.assertRaisesRegex(RUNNER.Day10RunnerError, "hash mismatch"):
                RUNNER.model_snapshot_identity(manifest, model_path, "manifest")

    def test_cli_model_path_requires_actual_identity(self):
        manifest = make_manifest(
            [make_record("dev-a", "dev", "general", [1, 13], "A. yes")]
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "model"
            make_model_snapshot(model_path)
            with self.assertRaisesRegex(RUNNER.Day10RunnerError, "explicit --model-id"):
                RUNNER.model_snapshot_identity(manifest, model_path, "cli_required")
            identity = RUNNER.model_snapshot_identity(
                manifest,
                model_path,
                "cli_override",
                cli_model_id="trained/sft-a",
                cli_model_revision="checkpoint-100",
            )
            self.assertEqual(identity["model_id"], "trained/sft-a")
            self.assertEqual(identity["model_revision"], "checkpoint-100")

    def test_comparison_key_excludes_checkpoint_but_run_hash_includes_it(self):
        records = [make_record("dev-a", "dev", "general", [1, 13], "A. yes")]
        manifest = make_manifest(records)
        runtime = {
            "device": "cpu",
            "model_dtype": "torch.float32",
            "torch_version": "test-torch",
            "transformers_version": "test-transformers",
            "python_version": "3.11.0",
            "python_implementation": "CPython",
            "platform": "test-platform",
            "torch_num_threads": 4,
            "torch_num_interop_threads": 2,
        }
        execution_hash = RUNNER.semantic_hash(
            RUNNER.build_execution_protocol(manifest, runtime)
        )
        comparison_key = RUNNER.build_comparison_key(manifest, execution_hash)
        policy = RUNNER.selection_policy(records, 1)
        first_model = {
            "model_id": "mock/model",
            "model_revision": "checkpoint-a",
            "model_snapshot_source": "cli_override",
            "model_snapshot_hash": "sha256:" + "a" * 64,
        }
        second_model = {
            **first_model,
            "model_revision": "checkpoint-b",
            "model_snapshot_hash": "sha256:" + "b" * 64,
        }
        first_contract = RUNNER.build_run_contract(
            manifest,
            first_model,
            records,
            policy,
            runtime,
            execution_hash,
            comparison_key,
        )
        second_contract = RUNNER.build_run_contract(
            manifest,
            second_model,
            records,
            policy,
            runtime,
            execution_hash,
            comparison_key,
        )
        self.assertEqual(first_contract["comparison_key"], second_contract["comparison_key"])
        self.assertNotEqual(
            RUNNER.semantic_hash(first_contract), RUNNER.semantic_hash(second_contract)
        )

    def test_reference_base_fields_do_not_change_comparison_key(self):
        records = [make_record("dev-a", "dev", "general", [1, 13], "A. yes")]
        first = make_manifest(records)
        second = json.loads(json.dumps(first))
        second["header"]["model_and_rendering"]["model_id"] = "other/reference-base"
        second["header"]["model_and_rendering"]["model_revision"] = "other-revision"
        second["header"].pop("manifest_hash")
        second["header"]["manifest_hash"] = RUNNER.semantic_hash(second)
        self.assertNotEqual(first["header"]["manifest_hash"], second["header"]["manifest_hash"])
        execution_hash = "sha256:" + "8" * 64
        self.assertEqual(
            RUNNER.build_comparison_key(first, execution_hash),
            RUNNER.build_comparison_key(second, execution_hash),
        )


class FullRunnerTest(unittest.TestCase):
    def test_mock_run_writes_only_dev_and_never_executes_code(self):
        records = [
            make_record("dev-general", "dev", "general", [1, 11], "A. yes"),
            make_record("dev-code", "dev", "code", [1, 12], "unused"),
            make_record(
                "frozen-general", "frozen_test", "general", [1, 13], "B. no"
            ),
        ]
        manifest = make_manifest(records)
        model = MockModel({11: [21], 12: [22]})
        tokenizer = MockTokenizer(
            {
                (21,): "Final answer: A",
                (22,): "raise RuntimeError('must never execute')",
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "manifest.json"
            output_path = root / "results.jsonl"
            model_path = root / "model"
            write_manifest(manifest_path, manifest)
            make_model_snapshot(model_path)

            result = RUNNER.run_baseline(
                manifest_path,
                output_path,
                model_path=model_path,
                model_id="mock/model",
                model_revision="mock-revision",
                sample_limit=2,
                requested_device="cpu",
                model=model,
                tokenizer=tokenizer,
            )
            written = [json.loads(line) for line in output_path.read_text().splitlines()]
            self.assertEqual(result["record_count"], 2)
            self.assertEqual(
                [record["sample_id"] for record in written],
                ["dev-general", "dev-code"],
            )
            self.assertTrue(all(record["evaluation_split"] == "dev" for record in written))
            self.assertEqual(written[0]["scorer_result"]["score"], 1.0)
            self.assertEqual(
                written[1]["scorer_result"]["score_status"], "sandbox_required"
            )
            self.assertIsNone(written[1]["scorer_result"]["score"])
            self.assertIn("config.json", written[0]["model_files"])
            self.assertEqual(written[0]["run_hash"], written[1]["run_hash"])
            self.assertEqual(
                written[0]["model_snapshot_hash"], written[1]["model_snapshot_hash"]
            )
            self.assertEqual(
                written[0]["selection_policy"]["version"],
                "day10_dev_stratified_round_robin_v1",
            )
            self.assertEqual(
                written[0]["execution_protocol_hash"],
                written[1]["execution_protocol_hash"],
            )
            self.assertEqual(written[0]["comparison_key"], written[1]["comparison_key"])

            with self.assertRaisesRegex(RUNNER.Day10RunnerError, "already exists"):
                RUNNER.run_baseline(
                    manifest_path,
                    output_path,
                    model_path=model_path,
                    model_id="mock/model",
                    model_revision="mock-revision",
                    sample_limit=1,
                    requested_device="cpu",
                    model=model,
                    tokenizer=tokenizer,
                )
            self.assertEqual(len(output_path.read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
