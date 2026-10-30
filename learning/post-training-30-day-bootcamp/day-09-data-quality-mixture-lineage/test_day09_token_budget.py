import importlib.util
import unittest
from fractions import Fraction
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("plan_day09_token_budget.py")
SPEC = importlib.util.spec_from_file_location("plan_day09_token_budget", MODULE_PATH)
assert SPEC and SPEC.loader
BUDGET = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUDGET)


class Day09TokenBudgetTest(unittest.TestCase):
    def test_bitset_reachability_and_reconstruction_use_whole_examples(self):
        rows = [
            {"sample_id": "a", "supervised_token_count": 3},
            {"sample_id": "b", "supervised_token_count": 5},
            {"sample_id": "c", "supervised_token_count": 7},
        ]
        bits = BUDGET.reachable_bitset([3, 5, 7], 10)
        self.assertTrue(BUDGET.is_reachable(bits, 10))
        selected = BUDGET.reconstruct_subset(rows, 10)
        self.assertEqual(sum(row["supervised_token_count"] for row in selected), 10)
        self.assertEqual(len({row["sample_id"] for row in selected}), len(selected))

    def test_theoretical_upper_and_common_budget_respect_both_mixes(self):
        ratios = {
            "mix_A_balanced": {
                "general": Fraction(1, 4),
                "math": Fraction(1, 4),
                "code": Fraction(1, 4),
                "finance": Fraction(1, 4),
            },
            "mix_B_targeted": {
                "general": Fraction(1, 8),
                "math": Fraction(1, 8),
                "code": Fraction(1, 2),
                "finance": Fraction(1, 4),
            },
        }
        available = {"general": 10, "math": 10, "code": 10, "finance": 4}
        upper, constraints = BUDGET.theoretical_upper_total(available, ratios, 8)
        self.assertEqual(upper, 16)
        self.assertTrue(any(item["skill"] == "finance" for item in constraints))
        reachability = {
            "general": BUDGET.reachable_bitset([2, 4, 4], 4),
            "math": BUDGET.reachable_bitset([2, 4, 4], 4),
            "code": BUDGET.reachable_bitset([2, 4, 4], 8),
            "finance": BUDGET.reachable_bitset([4], 4),
        }
        total, targets = BUDGET.find_common_budget(
            available, reachability, ratios, 8, upper
        )
        self.assertEqual(total, 16)
        self.assertEqual(targets["mix_A_balanced"]["finance"], 4)
        self.assertEqual(targets["mix_B_targeted"]["code"], 8)

    def test_stable_order_is_input_order_independent(self):
        rows = [
            {"sample_id": "b", "supervised_token_count": 1},
            {"sample_id": "a", "supervised_token_count": 1},
        ]
        first = BUDGET.stable_order(rows, "v1", "mix", "general")
        second = BUDGET.stable_order(list(reversed(rows)), "v1", "mix", "general")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
