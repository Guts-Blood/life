#!/usr/bin/env python3
"""Offline tests for the isolated Day 20 v2 Code target contract."""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import day20_contract as v1
import day20_contract_v2 as v2


PREFIX = 'def solve(x):\n    """Return a deterministic result."""'


class Day20V2ContractTests(unittest.TestCase):
    def test_v2_budgets_and_v1_isolation(self) -> None:
        self.assertEqual(v2.PROBE_SUPERVISED_TOKENS, 24_000)
        self.assertEqual(v2.PROBE_TOKENS_PER_SKILL, 6_000)
        self.assertEqual(sum(v2.PROBE_TOKENS_BY_FORMAT.values()), 24_000)
        self.assertEqual(v2.PROBE_TOKENS_BY_FORMAT["general_mcq"], 3_000)
        self.assertEqual(v2.PROBE_TOKENS_BY_FORMAT["general_instruction"], 3_000)
        self.assertEqual(v2.MAIN_SUPERVISED_TOKENS, 320_000)
        self.assertEqual(v2.MAIN_TOKENS_PER_SKILL, 80_000)
        self.assertEqual(sum(v2.MAIN_TOKENS_BY_FORMAT.values()), 320_000)
        self.assertEqual(v2.MAIN_TOKENS_BY_FORMAT["general_mcq"], 31_998)
        self.assertEqual(v2.MAIN_TOKENS_BY_FORMAT["general_instruction"], 48_002)

        self.assertEqual(v1.PROBE_SUPERVISED_TOKENS, 16_000)
        self.assertEqual(v1.MAIN_SUPERVISED_TOKENS, 256_000)

    def test_ast_canonicalizes_training_indentation(self) -> None:
        result = v2.canonicalize_code_continuation(
            "if x > 0:\n    return x\nreturn 0", PREFIX
        )
        self.assertEqual(
            result.canonical,
            "    if x > 0:\n        return x\n    return 0",
        )
        self.assertEqual(result.raw, "if x > 0:\n    return x\nreturn 0")
        self.assertEqual(result.as_evidence()["canonical"], result.canonical)
        for digest in (
            result.raw_sha256,
            result.canonical_sha256,
            result.ast_sha256,
        ):
            self.assertRegex(digest, re.compile(r"^[0-9a-f]{64}$"))

    def test_hashes_distinguish_bytes_but_bind_same_ast(self) -> None:
        canonical = v2.canonicalize_code_continuation("    return x + 1", PREFIX)
        wider = v2.canonicalize_code_continuation("        return x + 1\n", PREFIX)
        crlf = v2.canonicalize_code_continuation("return x + 1\r\n", PREFIX)
        self.assertNotEqual(canonical.raw_sha256, wider.raw_sha256)
        self.assertNotEqual(canonical.raw_sha256, crlf.raw_sha256)
        self.assertEqual(canonical.canonical_sha256, wider.canonical_sha256)
        self.assertEqual(canonical.canonical_sha256, crlf.canonical_sha256)
        self.assertEqual(canonical.ast_sha256, wider.ast_sha256)
        self.assertEqual(canonical.ast_sha256, crlf.ast_sha256)

    def test_multiline_string_and_nested_blocks_compile(self) -> None:
        result = v2.canonicalize_code_continuation(
            '    value = """hello\n    world"""\n'
            "    if x:\n"
            "        value += str(x)\n"
            "    return value",
            PREFIX,
        )
        compile(f"{PREFIX}\n{result.canonical}\n", "<test>", "exec")
        self.assertTrue(result.canonical.startswith("    "))

    def test_strict_raw_contract_does_not_repair(self) -> None:
        accepted = v2.validate_raw_code_continuation("    return x", PREFIX)
        self.assertEqual(accepted.canonical, "    return x")
        invalid = (
            "return x",
            "        return x",
            "\treturn x",
            "\n    return x",
            "    \n    return x",
            "    return x\r\n",
        )
        for target in invalid:
            with self.subTest(target=repr(target)):
                with self.assertRaises(v2.Day20V2ContractError):
                    v2.validate_raw_code_continuation(target, PREFIX)

    def test_rejects_imports_definitions_classes_and_namespace_escape(self) -> None:
        invalid = (
            "import os\nreturn os.getcwd()",
            "from pathlib import Path\nreturn Path('.')",
            "def helper():\n    return 1\nreturn helper()",
            "async def helper():\n    return 1\nreturn 1",
            "class Helper:\n    pass\nreturn Helper()",
            "if x:\n    import os\nreturn x",
            "global x\nx = 1\nreturn x",
            "nonlocal x\nreturn x",
        )
        for target in invalid:
            with self.subTest(target=target):
                with self.assertRaises(v2.Day20V2ContractError):
                    v2.canonicalize_code_continuation(target, PREFIX)

    def test_rejects_fences_prose_and_module_escape(self) -> None:
        invalid = (
            "```python\nreturn x\n```",
            "Here is the implementation:\nreturn x",
            '"This is an explanation."',
            "explanation_only",
            "    if x:\n        return x\nreturn 0",
        )
        for target in invalid:
            with self.subTest(target=target):
                with self.assertRaises(v2.Day20V2ContractError):
                    v2.canonicalize_code_continuation(target, PREFIX)

    def test_prefix_binding_must_compile_and_end_in_target_function(self) -> None:
        with self.assertRaises(v2.Day20V2ContractError):
            v2.canonicalize_code_continuation("return x", "x = 1")
        with self.assertRaises(v2.Day20V2ContractError):
            v2.canonicalize_code_continuation("return x", "def solve(x):\n    pass\nx = 1")


if __name__ == "__main__":
    unittest.main()
