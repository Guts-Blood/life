import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("inspect_sft_sample.py")
SPEC = importlib.util.spec_from_file_location("inspect_sft_sample", MODULE_PATH)
assert SPEC and SPEC.loader
INSPECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSPECTOR)


class SchemaValidatorTest(unittest.TestCase):
    def test_valid_system_user_assistant(self):
        INSPECTOR.validate_sample_schema(
            {
                "id": "valid",
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                ],
            }
        )

    def test_empty_assistant_is_rejected(self):
        with self.assertRaisesRegex(INSPECTOR.ContractError, "empty assistant"):
            INSPECTOR.validate_sample_schema(
                {
                    "id": "empty",
                    "messages": [
                        {"role": "user", "content": "Question"},
                        {"role": "assistant", "content": "  "},
                    ],
                }
            )

    def test_illegal_role_order_is_rejected(self):
        with self.assertRaisesRegex(INSPECTOR.ContractError, "must be user"):
            INSPECTOR.validate_sample_schema(
                {"id": "order", "messages": [{"role": "assistant", "content": "Answer"}]}
            )

    def test_duplicate_ids_are_rejected(self):
        samples = [
            {"id": "same", "messages": []},
            {"id": "same", "messages": []},
        ]
        with self.assertRaisesRegex(INSPECTOR.ContractError, "duplicate sample id"):
            INSPECTOR.validate_unique_ids(samples)


if __name__ == "__main__":
    unittest.main()
