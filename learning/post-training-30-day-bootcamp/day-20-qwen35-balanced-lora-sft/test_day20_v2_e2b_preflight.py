#!/usr/bin/env python3
"""Offline tests for the train-before-E2B Day 20 v2 hard gate."""

from __future__ import annotations

import json
import os
import tempfile
import types
import unittest
from pathlib import Path

import day20_e2b_preflight_v2 as preflight


class FakeSandbox:
    def __init__(self) -> None:
        self.killed = False

    def get_info(self) -> object:
        return object()

    def kill(self) -> bool:
        self.killed = True
        return True


class Day20E2BPreflightV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "run"
        (self.root / "evidence").mkdir(parents=True)
        self.scorer = Path(self.temporary.name) / "frozen.py"
        self.scorer.write_text("# frozen fixture\n", encoding="utf-8")
        self.source = Path(self.temporary.name) / "HumanEval.jsonl.gz"
        self.source.write_bytes(b"pinned-humaneval")
        self.config = Path(self.temporary.name) / "sandbox.json"
        self.config.write_text(
            json.dumps(
                {
                    "sdk": {"package": "e2b", "version": "2.37.0"},
                    "evaluator_source_sha256": preflight.file_sha256(self.scorer),
                    "source": {"file_sha256": preflight.file_sha256(self.source)},
                }
            ),
            encoding="utf-8",
        )
        self.credential = Path(self.temporary.name) / "day20-v2-e2b.env"
        self.credential.write_text("E2B_API_KEY=test-secret-value\n", encoding="utf-8")
        self.credential.chmod(0o600)
        self.attestation = Path(self.temporary.name) / "day20-v2-e2b-attestation.json"
        value: dict[str, object] = {
            "schema_version": 2,
            "domain": preflight.ATTESTATION_DOMAIN,
            "status": "rotated",
            "created_at_utc": "2026-08-11T00:00:00Z",
            "credential_file_sha256": preflight.file_sha256(self.credential),
        }
        value["attestation_sha256"] = preflight.object_sha256(value)
        self.attestation.write_text(json.dumps(value), encoding="utf-8")
        self.attestation.chmod(0o600)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def fake_module(self, _: Path) -> types.SimpleNamespace:
        sandbox = FakeSandbox()
        return types.SimpleNamespace(
            verify_config=lambda config: None,
            load_e2b_bindings=lambda config: {"create": lambda **kwargs: sandbox},
            sandbox_create_kwargs=lambda config: {"template": "fixture"},
            sandbox_info_mismatch=lambda info, config: None,
        )

    def test_live_smoke_is_attested_killed_and_reverifiable(self) -> None:
        previous = os.environ.pop("E2B_API_KEY", None)
        try:
            result = preflight.run_preflight(
                run_root=self.root,
                frozen_scorer=self.scorer,
                sandbox_config=self.config,
                humaneval_source=self.source,
                credential_file=self.credential,
                credential_attestation=self.attestation,
                module_loader=self.fake_module,
                installed_e2b_version="2.37.0",
            )
            self.assertEqual(result["status"], "pass")
            self.assertTrue(result["live_smoke"]["sandbox_killed"])
            self.assertNotIn("test-secret-value", json.dumps(result))
            self.assertNotIn("E2B_API_KEY", os.environ)
            verified = preflight.verify_preflight(
                self.root / "evidence/E2B-PREFLIGHT.json", run_root=self.root
            )
            self.assertEqual(verified, result)

            self.credential.write_text(
                "E2B_API_KEY=rotated-after-smoke\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                preflight.Day20E2BPreflightV2Error,
                "credential changed after live smoke",
            ):
                preflight.verify_preflight(
                    self.root / "evidence/E2B-PREFLIGHT.json", run_root=self.root
                )
        finally:
            if previous is not None:
                os.environ["E2B_API_KEY"] = previous

    def test_credential_or_attestation_drift_fails_before_smoke(self) -> None:
        self.credential.write_text("E2B_API_KEY=changed\n", encoding="utf-8")
        with self.assertRaisesRegex(
            preflight.Day20E2BPreflightV2Error, "attestation drifted"
        ):
            preflight.run_preflight(
                run_root=self.root,
                frozen_scorer=self.scorer,
                sandbox_config=self.config,
                humaneval_source=self.source,
                credential_file=self.credential,
                credential_attestation=self.attestation,
                module_loader=self.fake_module,
                installed_e2b_version="2.37.0",
            )

    def test_wrong_sdk_version_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            preflight.Day20E2BPreflightV2Error, "expected e2b"
        ):
            preflight.run_preflight(
                run_root=self.root,
                frozen_scorer=self.scorer,
                sandbox_config=self.config,
                humaneval_source=self.source,
                credential_file=self.credential,
                credential_attestation=self.attestation,
                module_loader=self.fake_module,
                installed_e2b_version="2.36.0",
            )

    def test_attestation_helper_never_persists_the_secret(self) -> None:
        output = Path(self.temporary.name) / "generated-attestation.json"
        result = preflight.attest_credential(
            self.credential,
            output,
            created_at_utc="2026-08-11T00:00:00Z",
        )
        self.assertEqual(output.stat().st_mode & 0o777, 0o600)
        self.assertEqual(
            result["credential_file_sha256"],
            preflight.file_sha256(self.credential),
        )
        self.assertNotIn("test-secret-value", output.read_text(encoding="utf-8"))
        _, verified = preflight.verify_credential(self.credential, output)
        self.assertEqual(verified["content_sha256"], result["attestation_sha256"])


if __name__ == "__main__":
    unittest.main()
