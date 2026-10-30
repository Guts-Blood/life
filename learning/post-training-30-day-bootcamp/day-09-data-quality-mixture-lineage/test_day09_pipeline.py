import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("prepare_day09_data.py")
SPEC = importlib.util.spec_from_file_location("prepare_day09_data", MODULE_PATH)
assert SPEC and SPEC.loader
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


class Day09PipelineTest(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"id": "r1", "messages": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]},
            {"id": "r2", "messages": [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]},
            {"id": "r3", "messages": [{"role": "user", "content": "q3"}, {"role": "assistant", "content": "a3"}]},
            {"id": "r4", "messages": [{"role": "user", "content": "q4"}, {"role": "assistant", "content": "a4"}]},
            {"id": "r5", "messages": [{"role": "user", "content": "q5"}, {"role": "assistant", "content": "a5"}]},
        ]

    def test_selection_is_independent_of_input_order(self):
        kwargs = {"source_id": "example/source", "revision": "a" * 40, "count": 3}
        forward = PREPARE.select_records(self.records, **kwargs)
        reverse = PREPARE.select_records(reversed(self.records), **kwargs)
        self.assertEqual(
            [(item["parent_id"], item["selection_hash"]) for item in forward],
            [(item["parent_id"], item["selection_hash"]) for item in reverse],
        )

    def test_duplicate_parent_id_is_rejected(self):
        with self.assertRaisesRegex(PREPARE.DataPreparationError, "duplicate parent id"):
            PREPARE.select_records(
                [self.records[0], self.records[0]],
                source_id="example/source",
                revision="a" * 40,
                count=1,
            )

    def test_tulu_adapter_preserves_messages(self):
        messages = PREPARE.canonicalize_tulu(self.records[0])
        self.assertEqual(messages, self.records[0]["messages"])

    def test_finqa_adapter_uses_gold_evidence_and_program(self):
        messages = PREPARE.canonicalize_finqa(
            {
                "id": "report-1",
                "question": "What is the change?",
                "answer": "20%",
                "exe_ans": "0.2",
                "program": "divide(20, 100)",
                "gold_inds": '{"table_1": "The change is 20 out of 100."}',
            }
        )
        self.assertIn("The change is 20 out of 100.", messages[0]["content"])
        self.assertEqual(
            messages[1]["content"],
            "Calculation: divide(20, 100)\nFinal answer: 20%",
        )

    def test_day08_contract_is_reused_from_frozen_module(self):
        contract = PREPARE.load_day08_contract()
        self.assertEqual(Path(contract.__file__).resolve(), PREPARE.DAY08_MODULE.resolve())
        self.assertTrue(callable(contract.encode_sample))

    def test_content_hash_uses_canonical_object_key_order(self):
        left = [{"role": "user", "content": "hello"}]
        right = [{"content": "hello", "role": "user"}]
        self.assertEqual(PREPARE.content_hash(left), PREPARE.content_hash(right))

    def test_preprocess_candidate_records_all_token_units(self):
        class FakeTokenizer:
            def __call__(self, text, *, add_special_tokens):
                if add_special_tokens:
                    raise AssertionError("raw token count must not add special tokens")
                return {"input_ids": text.split()}

        class FakeContract:
            @staticmethod
            def encode_sample(tokenizer, sample, max_length):
                return {
                    "original_length": 12,
                    "encoded_length": 12,
                    "effective_label_tokens": 4,
                    "truncated": False,
                }

        candidate = {
            "sample_id": "general:r1",
            "parent_id": "r1",
            "source": "example/source",
            "revision": "a" * 40,
            "license": "test",
            "messages": self.records[0]["messages"],
            "transform_chain": ["fixture_v1"],
        }
        accepted, _ = PREPARE.preprocess_candidate(
            FakeTokenizer(), FakeContract(), candidate, 256
        )
        self.assertEqual(accepted["raw_token_count"], 2)
        self.assertEqual(accepted["rendered_token_count"], 12)
        self.assertEqual(accepted["input_token_count"], 12)
        self.assertEqual(accepted["supervised_token_count"], 4)
        self.assertFalse(accepted["truncated"])


if __name__ == "__main__":
    unittest.main()
