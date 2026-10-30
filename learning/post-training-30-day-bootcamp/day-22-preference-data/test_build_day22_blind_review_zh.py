#!/usr/bin/env python3
"""Tests for the Chinese reviewer-facing blind-review page."""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

import build_day22_blind_review_zh as builder


class ChineseBlindReviewPageTest(unittest.TestCase):
    def test_embedded_cases_are_byte_source_equivalent_and_no_key_is_present(self) -> None:
        source_sha_before = builder.file_sha256(builder.DEFAULT_INPUT)
        cases = builder.load_cases(builder.DEFAULT_INPUT)
        html = builder.render_html(cases, source_sha_before)
        match = re.search(
            r'<script id="case-data" type="application/json">(.*?)</script>',
            html,
            re.DOTALL,
        )
        self.assertIsNotNone(match)
        embedded = json.loads(match.group(1).replace("<\\/", "</"))
        self.assertEqual(embedded, cases)
        self.assertEqual(builder.file_sha256(builder.DEFAULT_INPUT), source_sha_before)
        self.assertEqual(source_sha_before, builder.FROZEN_SOURCE_SHA256)
        self.assertNotIn('"pair_id"', html)
        self.assertNotIn('"a_side"', html)
        self.assertNotIn('"b_side"', html)
        self.assertIn("connect-src 'none'", html)

    def test_chinese_ui_exports_finalizer_enums(self) -> None:
        html = builder.render_html(
            builder.load_cases(builder.DEFAULT_INPUT),
            builder.file_sha256(builder.DEFAULT_INPUT),
        )
        for text in ("A 更优", "B 更优", "基本相同", "无法判断", "样本有问题", "置信度"):
            self.assertIn(text, html)
        for value in ("A", "B", "tie", "ambiguous", "reject"):
            self.assertIn(f'value="{value}"', html)
        for value in ("low", "medium", "high"):
            self.assertIn(f'value="{value}"', html)
        self.assertIn('id="export-draft"', html)
        self.assertIn('id="export-completed"', html)

    def test_cli_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "review.html"
            self.assertEqual(builder.main(["--output", str(output)]), 0)
            self.assertEqual(builder.main(["--output", str(output)]), 2)


if __name__ == "__main__":
    unittest.main()
