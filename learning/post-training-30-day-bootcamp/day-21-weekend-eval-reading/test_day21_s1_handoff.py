#!/usr/bin/env python3
"""Focused tests for the append-only Day21 S1 handoff."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).with_name("day21_s1_handoff.py")
SPEC = importlib.util.spec_from_file_location("day21_s1_handoff", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
handoff = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(handoff)


class IsolatedCollectorTests(unittest.TestCase):
    def test_collector_uses_fresh_interpreter_and_final_json_line(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bootcamp = Path(temporary) / "bootcamp"
            collector = bootcamp / "rsi-control/rsi_day20_collect.py"
            collector.parent.mkdir(parents=True)
            collector.write_text("# frozen collector\n", encoding="utf-8")
            run_root = Path(temporary) / "run"
            run_root.mkdir()
            expected = {
                "status": "pass",
                "inventory_sha256": handoff.COLLECTOR_INVENTORY_SHA256,
            }
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="diagnostic prefix\n" + json.dumps(expected) + "\n",
                stderr="template warnings stay on stderr\n",
            )
            with mock.patch.object(
                handoff.subprocess, "run", return_value=completed
            ) as run:
                actual = handoff.collect_rsi_inventory_isolated(
                    run_root=run_root, bootcamp_root=bootcamp
                )

            self.assertEqual(actual, expected)
            run.assert_called_once_with(
                [
                    sys.executable,
                    str(collector.resolve()),
                    "--run-root",
                    str(run_root.resolve()),
                ],
                cwd=str(bootcamp.resolve()),
                check=False,
                capture_output=True,
                text=True,
                timeout=600,
            )

    def test_collector_failure_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bootcamp = Path(temporary) / "bootcamp"
            collector = bootcamp / "rsi-control/rsi_day20_collect.py"
            collector.parent.mkdir(parents=True)
            collector.write_text("# frozen collector\n", encoding="utf-8")
            run_root = Path(temporary) / "run"
            run_root.mkdir()
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=2,
                stdout=json.dumps(
                    {"status": "verification_failed", "message": "drift"}
                ),
                stderr="",
            )
            with mock.patch.object(handoff.subprocess, "run", return_value=completed):
                with self.assertRaisesRegex(
                    handoff.HandoffError, "isolated RSI collector rejected"
                ):
                    handoff.collect_rsi_inventory_isolated(
                        run_root=run_root, bootcamp_root=bootcamp
                    )

    def test_recovery_collects_before_parent_process_import_setup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            run_root.mkdir()
            (run_root / ".day20-v3-run-root").write_text(
                "day20-qwen35-target-encoding-v3\n", encoding="utf-8"
            )
            calls: list[str] = []

            def reject_after_recording(**_: object) -> dict[str, object]:
                calls.append("isolated_collector")
                raise handoff.HandoffError("sentinel")

            def record_import_setup(_: Path) -> None:
                calls.append("parent_import_setup")

            with (
                mock.patch.object(
                    handoff,
                    "collect_rsi_inventory_isolated",
                    side_effect=reject_after_recording,
                ),
                mock.patch.object(
                    handoff, "add_day20_imports", side_effect=record_import_setup
                ),
            ):
                with self.assertRaisesRegex(handoff.HandoffError, "sentinel"):
                    handoff.build_recovery(
                        run_root=run_root,
                        bootcamp_root=Path(temporary) / "bootcamp",
                        archive_dir=Path(temporary) / "archive",
                        copy_archive=False,
                    )

            self.assertEqual(calls, ["isolated_collector"])


class ContractSmokeTests(unittest.TestCase):
    def test_parser_constructs_every_command(self) -> None:
        parser = handoff.parser()
        self.assertEqual(
            parser.parse_args(
                [
                    "recover",
                    "--run-root",
                    "run",
                    "--bootcamp-root",
                    "bootcamp",
                    "--archive-dir",
                    "archive",
                    "--output",
                    "evidence.json",
                ]
            ).command,
            "recover",
        )

    def test_self_hash_rejects_tampering(self) -> None:
        artifact_dir = MODULE_PATH.parent.parent / "artifacts/checkpoints"
        expected = {
            "day21-qwen35-s1-checkpoint-archive-evidence.json": (
                "057806704e53ab7d19aef61a2f9a49b2991071d1aafcbba6c004edb9c1e88bd0",
                "evidence_sha256",
                "c781ed2404f7ab8cbb10d51b1faf711358f60c552534038779e5ea528254f88e",
            ),
            "day21-qwen35-s1-merged-export-manifest.json": (
                "9c196e43f633116d7fc804870dcfe961db5ae0d216a15f5e21d679d8c1f14b5a",
                "manifest_sha256",
                "660eed4af7f76796631561275f0190c402952520a8ccce358289e6269fb8f8d3",
            ),
            "day21-qwen35-s1-adapter-fresh-capture.json": (
                "8d5c34b556b05d57f7679b8c067a89050f68620eebf89a6a1ca9222803cc7c24",
                "capture_sha256",
                "bdcd04e0779869ca8f5b59726e298d7694a83d1631e125f0a8ee640b20d440f4",
            ),
            "day21-qwen35-s1-merged-fresh-capture.json": (
                "db2af58635f35a165d0386677e64751f70da529f04590f00c0f4b230b3b7082a",
                "capture_sha256",
                "6ded91a67250decf48c5c66aa64bd2069defe62bc5bed09e3f73066399dee3cf",
            ),
            "day21-qwen35-s1-adapter-merged-parity.json": (
                "ded244ef1813d898fb08eb64e773a0390d1c42e252a1fcc70e45a76292ee6bea",
                "parity_sha256",
                "93299b61322ecc45eba86e995f63216ef76264ac239c4894d374bd9515a07dc7",
            ),
            "day21-qwen35-s1-promotion-manifest.json": (
                "40c76f690dcb73805250e0036f8e124ef07b8d2d32b1723637656ae4304d94f8",
                "promotion_manifest_sha256",
                "64e6a6bd61951d1f10eb4def520273cb056625c1b041b68ba861c6d1e15d7a6c",
            ),
            "day21-qwen35-s1-downstream-key.json": (
                "979e599aa6c9c05bec9a14144b6ebae689dbdf8315a9112caaec09b5469bd545",
                "key_sha256",
                "4a3a467232288feb675eb7d15cc00ff95bc7255a1c4dc2d1debdd55da5831241",
            ),
        }
        for filename, (file_hash, self_field, self_hash) in expected.items():
            path = artifact_dir / filename
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), file_hash)
            artifact = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(handoff.verify_self_hash(artifact, self_field), self_hash)

        verify_path = artifact_dir / "day21-qwen35-s1-handoff-verify.json"
        self.assertEqual(
            hashlib.sha256(verify_path.read_bytes()).hexdigest(),
            "249f7f5837477b31059ea1eb56b5620f7ae44ab4ed6cbb11c2f64c7a8fb1d9a6",
        )
        verify = json.loads(verify_path.read_text(encoding="utf-8"))
        self.assertTrue(verify["manifest"]["downstream_ready"])
        self.assertEqual(
            verify["manifest"]["promotion_manifest_sha256"],
            verify["manifest"]["_verified_promotion_manifest_sha256"],
        )

        value = handoff.seal({"status": "pass"}, "sha256")
        handoff.verify_self_hash(value, "sha256")
        value["status"] = "changed"
        with self.assertRaises(handoff.HandoffError):
            handoff.verify_self_hash(value, "sha256")


if __name__ == "__main__":
    unittest.main()
