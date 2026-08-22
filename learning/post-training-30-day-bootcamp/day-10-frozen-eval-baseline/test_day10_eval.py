import importlib.util
import json
import unittest
from collections import Counter
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("compile_day10_eval.py")
SPEC = importlib.util.spec_from_file_location("compile_day10_eval", MODULE_PATH)
assert SPEC and SPEC.loader
COMPILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPILE)

SCORER_PATH = Path(__file__).with_name("day10_scorers.py")
SCORER_SPEC = importlib.util.spec_from_file_location("day10_scorers_for_eval", SCORER_PATH)
assert SCORER_SPEC and SCORER_SPEC.loader
SCORERS = importlib.util.module_from_spec(SCORER_SPEC)
SCORER_SPEC.loader.exec_module(SCORERS)


class Day10EvalCompilerTest(unittest.TestCase):
    def test_rfc8785_hash_is_key_order_independent(self):
        left = {"domain": "test", "schema_version": 1, "b": 2, "a": 1}
        right = {"a": 1, "b": 2, "schema_version": 1, "domain": "test"}
        self.assertEqual(COMPILE.canonical_bytes(left), COMPILE.canonical_bytes(right))
        self.assertEqual(COMPILE.semantic_hash(left), COMPILE.semantic_hash(right))

    def test_duplicate_json_keys_are_rejected(self):
        with self.assertRaisesRegex(COMPILE.Day10CompileError, "duplicate JSON key"):
            json.loads('{"a": 1, "a": 2}', object_pairs_hook=COMPILE.reject_duplicate_keys)

    def test_split_assignment_is_input_order_independent_and_balanced(self):
        rows = []
        for subject in ("s1", "s2", "s3", "s4"):
            rows.extend(
                {
                    "eval_sample_id": f"eval:general:{subject}:{index}",
                    "skill": "general",
                    "metadata": {"subject": subject},
                }
                for index in range(10)
            )
        for slice_name in ("math", "code", "finance"):
            rows.extend(
                {
                    "eval_sample_id": f"eval:{slice_name}:{index}",
                    "skill": slice_name,
                    "metadata": {},
                }
                for index in range(40)
            )
        config = {
            "version": "day10_eval_split_v1",
            "slices": ["general", "math", "code", "finance"],
            "candidates_per_slice": 40,
            "dev_per_slice": 28,
            "frozen_test_per_slice": 12,
        }
        forward = COMPILE.assign_evaluation_splits(rows, config)
        reversed_result = COMPILE.assign_evaluation_splits(list(reversed(rows)), config)
        self.assertEqual(forward, reversed_result)
        counts = Counter(
            (row["skill"], forward[row["eval_sample_id"]]) for row in rows
        )
        for slice_name in config["slices"]:
            self.assertEqual(counts[(slice_name, "dev")], 28)
            self.assertEqual(counts[(slice_name, "frozen_test")], 12)

    def test_prompt_adapters_freeze_answer_format(self):
        self.assertIn("Final answer: <A|B|C|D>", COMPILE.adapt_prompt("general", "Q"))
        self.assertIn("Final answer: <answer>", COMPILE.adapt_prompt("math", "Q"))
        self.assertIn("required scale", COMPILE.adapt_prompt("finance", "Q"))
        code = COMPILE.adapt_prompt("code", "def f():")
        self.assertIn("Do not repeat the prompt", code)
        self.assertTrue(code.endswith("def f():"))

    def test_rendering_contract_excludes_checkpoint_and_cache_identity(self):
        config = COMPILE.load_json(COMPILE.DEFAULT_CONFIG)
        full = config["model_and_rendering"]
        contract = COMPILE.checkpoint_independent_rendering_contract(full)
        for checkpoint_key in (
            "model_id",
            "model_revision",
            "model_snapshot_path",
            "model_files",
            "tokenizer_snapshot_path",
        ):
            self.assertNotIn(checkpoint_key, contract)
        changed = dict(full)
        changed["model_revision"] = "different-checkpoint"
        changed["model_files"] = {"model.safetensors": "0" * 64}
        self.assertEqual(
            COMPILE.semantic_hash(contract),
            COMPILE.semantic_hash(
                COMPILE.checkpoint_independent_rendering_contract(changed)
            ),
        )

    def test_config_matches_executable_scorer_registry(self):
        config = COMPILE.load_json(COMPILE.DEFAULT_CONFIG)
        self.assertTrue(COMPILE.verify_model_snapshot(config).is_dir())
        registry, source_hash = COMPILE.load_scorer_registry(config)
        self.assertEqual(
            registry["registry_version"], config["scoring"]["registry_version"]
        )
        self.assertEqual(len(source_hash), 64)
        self.assertEqual(
            registry["slices"]["code"]["execution_policy"], "sandbox_required"
        )

    def test_full_manifest_compiles_with_frozen_invariants(self):
        first = COMPILE.build_manifest()
        second = COMPILE.build_manifest()
        self.assertEqual(COMPILE.manifest_bytes(first), COMPILE.manifest_bytes(second))
        COMPILE.verify_manifest_hash(first)

        header = first["header"]
        records = first["records"]
        self.assertEqual(header["counts"]["total"], 160)
        self.assertEqual(header["counts"]["by_split"], {"dev": 112, "frozen_test": 48})
        self.assertTrue(header["rendered_inputs_hash"].startswith("sha256:"))
        self.assertEqual(len(records), 160)
        self.assertEqual(len({row["sample_id"] for row in records}), 160)
        self.assertEqual(
            {row["slice"] for row in records},
            {"general", "math", "code", "finance"},
        )
        for row in records:
            expected = COMPILE.semantic_hash(
                {
                    "domain": "day10.input_ids",
                    "schema_version": 1,
                    "input_ids": row["input_ids"],
                }
            )
            self.assertEqual(row["input_ids_hash"], expected)
            self.assertEqual(row["input_token_count"], len(row["input_ids"]))

        records_by_id = {row["sample_id"]: row for row in records}
        for split, ordered_ids in header["evaluation_order"].items():
            self.assertEqual(len(ordered_ids), len(set(ordered_ids)))
            self.assertTrue(
                all(records_by_id[sample_id]["evaluation_split"] == split for sample_id in ordered_ids)
            )

    def test_manifest_hash_detects_tampering(self):
        manifest = COMPILE.build_manifest()
        manifest["records"][0]["reference"] += "tampered"
        with self.assertRaisesRegex(COMPILE.Day10CompileError, "semantic hash mismatch"):
            COMPILE.verify_manifest_hash(manifest)

    def test_every_frozen_reference_is_accepted_by_its_registered_scorer(self):
        manifest = COMPILE.build_manifest()
        for record in manifest["records"]:
            result = SCORERS.score_prediction(
                record["slice"], "", record["reference"]
            )
            expected_status = (
                "sandbox_required" if record["slice"] == "code" else "ok"
            )
            self.assertEqual(result["score_status"], expected_status)


if __name__ == "__main__":
    unittest.main()
