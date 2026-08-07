#!/usr/bin/env python3
"""Validate the Day 12 AutoDL binding and optionally attest one live sandbox."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import stat
from pathlib import Path
from types import ModuleType
from typing import Any


HERE = Path(__file__).resolve().parent
BOOTCAMP_ROOT = HERE.parent
DEFAULT_BINDING = (
    BOOTCAMP_ROOT / "artifacts/configs/day12-autodl-e2b-binding.json"
)


class PreflightError(RuntimeError):
    """A Day 12 AutoDL E2B binding invariant failed."""


def file_sha256(path: Path) -> str:
    if not path.is_file():
        raise PreflightError(f"required file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file_hash(path: Path, expected: str) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise PreflightError(
            f"SHA-256 mismatch for {path}: expected {expected}, got {actual}"
        )


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PreflightError(f"binding is missing: {path}") from error
    if not isinstance(value, dict):
        raise PreflightError("binding root must be a JSON object")
    return value


def resolve_bootcamp_path(value: str) -> Path:
    path = (BOOTCAMP_ROOT / value).resolve()
    try:
        path.relative_to(BOOTCAMP_ROOT.resolve())
    except ValueError as error:
        raise PreflightError(f"path escapes the bootcamp root: {value}") from error
    return path


def load_frozen_scorer(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("day12_frozen_e2b_scorer", path)
    if spec is None or spec.loader is None:
        raise PreflightError(f"cannot load frozen scorer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_package_versions(binding: dict[str, Any]) -> None:
    required = binding["runtime"]["required_packages"]
    for package, expected in required.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise PreflightError(f"required package is missing: {package}") from error
        if actual != expected:
            raise PreflightError(
                f"package mismatch for {package}: expected {expected}, got {actual}"
            )


def verify_credential(binding: dict[str, Any]) -> None:
    credential = binding["credential"]
    path = Path(credential["env_file"])
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError as error:
        raise PreflightError(f"credential file is missing: {path}") from error
    expected_mode = int(credential["required_file_mode"], 8)
    if mode != expected_mode:
        raise PreflightError(
            f"credential file mode must be {expected_mode:04o}, got {mode:04o}"
        )
    variable = credential["environment_variable"]
    if not os.environ.get(variable):
        raise PreflightError(
            f"{variable} is not exported; source the credential file with set -a"
        )


def verify_frozen_inputs(binding: dict[str, Any]) -> tuple[ModuleType, dict[str, Any]]:
    frozen = binding["frozen_inputs"]
    scorer_spec = frozen["scorer"]
    scorer_path = resolve_bootcamp_path(scorer_spec["path"])
    require_file_hash(scorer_path, scorer_spec["file_sha256"])

    contract_spec = frozen["sandbox_contract"]
    contract_path = resolve_bootcamp_path(contract_spec["path"])
    require_file_hash(contract_path, contract_spec["file_sha256"])
    contract = load_json(contract_path)

    manifest_spec = frozen["manifest"]
    require_file_hash(
        resolve_bootcamp_path(manifest_spec["path"]),
        manifest_spec["file_sha256"],
    )

    source_spec = frozen["humaneval_source"]
    require_file_hash(Path(source_spec["path"]), source_spec["file_sha256"])

    tokenizer_spec = frozen["tokenizer_snapshot"]
    tokenizer_root = Path(tokenizer_spec["path"])
    for filename, expected in tokenizer_spec["required_file_sha256"].items():
        require_file_hash(tokenizer_root / filename, expected)

    scorer = load_frozen_scorer(scorer_path)
    scorer.verify_config(contract)
    if scorer.semantic_hash(contract) != contract_spec["semantic_hash"]:
        raise PreflightError("frozen sandbox contract semantic hash mismatch")
    return scorer, contract


def run_live_smoke(scorer: ModuleType, contract: dict[str, Any]) -> None:
    bindings = scorer.load_e2b_bindings(contract)
    sandbox = bindings["create"](**scorer.sandbox_create_kwargs(contract))
    mismatch: str | None = None
    try:
        mismatch = scorer.sandbox_info_mismatch(sandbox.get_info(), contract)
    finally:
        killed = sandbox.kill()
    if mismatch is not None:
        raise PreflightError(f"live sandbox attestation mismatch: {mismatch}")
    if killed is False:
        raise PreflightError("live sandbox kill returned false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, default=DEFAULT_BINDING)
    parser.add_argument("--live-smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    binding = load_json(args.binding.resolve())
    if binding.get("domain") != "day12.autodl_e2b_binding":
        raise PreflightError("unexpected binding domain")
    verify_package_versions(binding)
    verify_credential(binding)
    scorer, contract = verify_frozen_inputs(binding)
    if args.live_smoke:
        run_live_smoke(scorer, contract)
    print(
        json.dumps(
            {
                "status": "day12_e2b_preflight_passed",
                "binding_file_sha256": file_sha256(args.binding.resolve()),
                "sdk": binding["runtime"]["required_packages"],
                "sandbox_contract_hash": binding["frozen_inputs"]
                ["sandbox_contract"]["semantic_hash"],
                "live_sandbox_attested_and_killed": args.live_smoke,
                "credential_value_logged": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
