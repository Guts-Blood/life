import importlib.util
import json
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("day10_scorers.py")
SPEC = importlib.util.spec_from_file_location("day10_scorers", MODULE_PATH)
assert SPEC and SPEC.loader
SCORERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORERS)


class RegistryTest(unittest.TestCase):
    def test_registry_is_versioned_serializable_and_covers_four_slices(self):
        self.assertEqual(
            SCORERS.SCORER_REGISTRY["registry_version"],
            SCORERS.SCORER_REGISTRY_VERSION,
        )
        self.assertEqual(
            set(SCORERS.SCORER_REGISTRY["slices"]),
            {"general", "math", "finance", "code"},
        )
        json.dumps(SCORERS.SCORER_REGISTRY, sort_keys=True)
        self.assertEqual(
            SCORERS.SCORER_REGISTRY["slices"]["code"]["execution_policy"],
            "sandbox_required",
        )
        expected_versions = {
            "general": ("mmlu_option_extractor_v2", "mmlu_exact_option_v2"),
            "math": ("gsm8k_final_number_extractor_v1", "gsm8k_numeric_exact_v1"),
            "finance": ("tatqa_final_answer_extractor_v3", "tatqa_normalized_exact_v3"),
            "code": (
                "human_eval_completion_extractor_v1",
                "human_eval_pass_at_1_sandbox_v1",
            ),
        }
        for slice_name, (extractor, scorer) in expected_versions.items():
            spec = SCORERS.SCORER_REGISTRY["slices"][slice_name]
            self.assertEqual(spec["extractor_version"], extractor)
            self.assertEqual(spec["scorer_version"], scorer)


class GeneralScorerTest(unittest.TestCase):
    def test_extracts_explicit_mmlu_option_and_scores_exactly(self):
        result = SCORERS.score_mmlu(
            "Reasoning first. The final answer is **C**.",
            "C. the reference choice",
        )
        self.assertEqual(result["parsed_answer"], "C")
        self.assertEqual(result["score"], 1.0)
        self.assertIsNone(result["error_type"])
        self.assertEqual(SCORERS.extract_mmlu_option("Final answer: <B>"), "B")

    def test_wrong_option_is_not_a_parse_error(self):
        result = SCORERS.score_mmlu("Answer: D", "A. first choice")
        self.assertEqual(result["parse_status"], "ok")
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["error_type"], "wrong_answer")

    def test_unstructured_text_is_parse_error(self):
        result = SCORERS.score_mmlu("I cannot determine it.", "B. choice")
        self.assertEqual(result["parse_status"], "parse_error")
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["error_type"], "parse_error")

    def test_copied_option_text_is_not_mistaken_for_a_prediction(self):
        result = SCORERS.score_mmlu("C. copied option text", "C. reference text")
        self.assertEqual(result["parse_status"], "parse_error")
        self.assertEqual(result["score"], 0.0)


class MathScorerTest(unittest.TestCase):
    def test_gsm8k_uses_final_marker_and_exact_numeric_canonicalization(self):
        result = SCORERS.score_gsm8k(
            "We first get 20. Final answer: $1,200.50",
            "A rationale with 12 and 100. #### 1200.5",
        )
        self.assertEqual(result["parsed_answer"], "$1,200.50")
        self.assertEqual(result["score"], 1.0)

    def test_decimal_and_fraction_are_canonically_equal(self):
        result = SCORERS.score_gsm8k("The answer is 0.5", "work #### 1/2")
        self.assertEqual(result["canonical_answer"], "1/2")
        self.assertEqual(result["score"], 1.0)

    def test_wrong_number_and_parse_error_are_distinct(self):
        wrong = SCORERS.score_gsm8k("Final answer: 41", "work #### 42")
        malformed = SCORERS.score_gsm8k("No numeric answer.", "work #### 42")
        self.assertEqual(wrong["error_type"], "wrong_answer")
        self.assertEqual(wrong["parse_status"], "ok")
        self.assertEqual(malformed["error_type"], "parse_error")
        self.assertEqual(malformed["parse_status"], "parse_error")


class FinanceScorerTest(unittest.TestCase):
    def test_parses_day09_tatqa_reference(self):
        reference = json.dumps(
            {
                "answer": ["fixed-price type", "cost-plus type"],
                "answer_type": "multi-span",
                "derivation": "",
                "scale": "",
            }
        )
        parsed = SCORERS.parse_tatqa_reference(reference)
        self.assertEqual(parsed["answer_type"], "multi-span")
        self.assertEqual(parsed["answers"], ["fixed-price type", "cost-plus type"])

    def test_arithmetic_accepts_declared_scale_and_scores_exactly(self):
        reference = {
            "answer": -22.22,
            "answer_type": "arithmetic",
            "derivation": "",
            "scale": "percent",
        }
        result = SCORERS.score_tatqa("Final answer: -22.22%", reference)
        self.assertEqual(result["score"], 1.0)
        missing_scale = SCORERS.score_tatqa("Final answer: -22.22", reference)
        self.assertEqual(missing_scale["error_type"], "wrong_answer")

    def test_finance_wrong_and_parse_error_are_distinct(self):
        reference = {
            "answer": 4,
            "answer_type": "count",
            "derivation": "",
            "scale": "",
        }
        wrong = SCORERS.score_tatqa("Answer: 5", reference)
        malformed = SCORERS.score_tatqa("Answer: several", reference)
        self.assertEqual(wrong["error_type"], "wrong_answer")
        self.assertEqual(wrong["parse_status"], "ok")
        self.assertEqual(malformed["error_type"], "parse_error")
        self.assertEqual(malformed["parse_status"], "parse_error")

    def test_span_normalization_is_conservative(self):
        reference = {
            "answer": ["Annual basis"],
            "answer_type": "span",
            "derivation": "",
            "scale": "",
        }
        same = SCORERS.score_tatqa("Answer:  ANNUAL   BASIS. ", reference)
        different = SCORERS.score_tatqa("Answer: yearly basis", reference)
        self.assertEqual(same["score"], 1.0)
        self.assertEqual(different["error_type"], "wrong_answer")

    def test_multi_span_json_array_is_order_independent_but_exact(self):
        reference = {
            "answer": ["2019", "2018"],
            "answer_type": "multi-span",
            "derivation": "",
            "scale": "",
        }
        result = SCORERS.score_tatqa('["2018", "2019"]', reference)
        self.assertEqual(result["score"], 1.0)

    def test_multi_span_numeric_final_answer_can_follow_prompt_format(self):
        reference = {
            "answer": ["27.1%", "28.7%"],
            "answer_type": "multi-span",
            "derivation": "",
            "scale": "",
        }
        result = SCORERS.score_tatqa(
            "Final answer: 28.7% and 27.1%", reference
        )
        self.assertEqual(result["score"], 1.0)

    def test_undeclared_percent_is_not_silently_dropped(self):
        reference = {
            "answer": ["36%"],
            "answer_type": "span",
            "derivation": "",
            "scale": "",
        }
        result = SCORERS.score_tatqa("Answer: 36", reference)
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["error_type"], "wrong_answer")

    def test_invalid_reference_fails_loudly(self):
        with self.assertRaises(ValueError):
            SCORERS.parse_tatqa_reference("not json")

    def test_copied_answer_format_instruction_is_not_the_final_segment(self):
        reference = {
            "answer": ["20"],
            "answer_type": "span",
            "derivation": "",
            "scale": "",
        }
        result = SCORERS.score_tatqa(
            "End with exactly: Final answer: <answer>, including scale.\nTable:",
            reference,
        )
        self.assertEqual(result["parsed_answer"], "Table:")
        self.assertEqual(result["score"], 0.0)


class CodeScorerTest(unittest.TestCase):
    def test_extracts_python_markdown_block_without_executing_it(self):
        raw = "Before\n```python\nraise RuntimeError('must not run')\n```\nAfter"
        self.assertEqual(
            SCORERS.extract_code_completion(raw),
            "raise RuntimeError('must not run')",
        )
        result = SCORERS.score_code(raw, "unused reference")
        self.assertEqual(result["score_status"], "sandbox_required")
        self.assertIsNone(result["score"])
        self.assertEqual(result["error_type"], "sandbox_required")

    def test_plain_completion_is_preserved_for_sandbox_handoff(self):
        completion = "    return x + 1\n"
        result = SCORERS.score_code(completion)
        self.assertEqual(result["parsed_answer"], "    return x + 1")
        self.assertEqual(result["parse_status"], "ok")


class DispatcherTest(unittest.TestCase):
    def test_dispatches_known_slice_and_rejects_unknown_slice(self):
        self.assertEqual(
            SCORERS.score_prediction("general", "A", "A. correct")["score"],
            1.0,
        )
        with self.assertRaises(ValueError):
            SCORERS.score_prediction("unknown", "", "")


if __name__ == "__main__":
    unittest.main()
