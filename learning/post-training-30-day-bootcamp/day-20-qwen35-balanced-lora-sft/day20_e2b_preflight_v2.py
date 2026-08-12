#!/usr/bin/env python3
"""Run and verify the Day 20 v2 E2B gate before any GPU training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping


DOMAIN = "day20.v2.e2b_preflight"
ATTESTATION_DOMAIN = "day20.v2.e2b_credential_attestation"
EXPECTED_E2B_VERSION = "2.37.0"
MARKER_NAME = "E2B-PREFLIGHT.json"


class Day20E2BPreflightV2Error(ValueError):
    """The credential, runtime, sandbox, or preflight marker failed closed."""


def object_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    if not path.is_file():
        raise Day20E2BPreflightV2Error(f"required file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Day20E2BPreflightV2Error(f"cannot load JSON: {path}") from error
    if not isinstance(value, dict):
        raise Day20E2BPreflightV2Error(f"JSON root must be an object: {path}")
    return value


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    expected = value.get(field)
    actual = object_sha256({key: item for key, item in value.items() if key != field})
    if not isinstance(expected, str) or expected != actual:
        raise Day20E2BPreflightV2Error(f"{field} mismatch")
    return expected


def _require_private_file(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise Day20E2BPreflightV2Error(f"{label} is missing or symbolic")
    metadata = path.stat()
    if stat.S_IMODE(metadata.st_mode) != 0o600 or metadata.st_uid != os.getuid():
        raise Day20E2BPreflightV2Error(
            f"{label} must be mode 0600 and owned by the current user"
        )


def _read_credential_secret(credential: Path) -> str:
    _require_private_file(credential, "credential file")
    try:
        lines = [
            line.strip()
            for line in credential.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError) as error:
        raise Day20E2BPreflightV2Error("cannot read credential file") from error
    if len(lines) != 1 or not lines[0].startswith("E2B_API_KEY="):
        raise Day20E2BPreflightV2Error(
            "credential file must contain exactly one E2B_API_KEY assignment"
        )
    secret = lines[0].removeprefix("E2B_API_KEY=")
    if not secret or any(character.isspace() for character in secret):
        raise Day20E2BPreflightV2Error("E2B_API_KEY value is empty or malformed")
    return secret


def verify_credential(
    credential_file: Path, attestation_file: Path
) -> tuple[str, dict[str, Any]]:
    credential = credential_file.resolve()
    attestation = attestation_file.resolve()
    _require_private_file(attestation, "credential attestation")
    secret = _read_credential_secret(credential)
    value = load_json(attestation)
    content_hash = _self_hash(value, "attestation_sha256")
    if (
        value.get("schema_version") != 2
        or value.get("domain") != ATTESTATION_DOMAIN
        or value.get("status") != "rotated"
        or not isinstance(value.get("created_at_utc"), str)
        or not value["created_at_utc"]
        or value.get("credential_file_sha256") != file_sha256(credential)
    ):
        raise Day20E2BPreflightV2Error("credential attestation drifted")
    identity = {
        "path": str(attestation),
        "file_sha256": file_sha256(attestation),
        "content_sha256": content_hash,
        "status": "rotated",
    }
    return secret, identity


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("day20_v2_frozen_e2b", path)
    if spec is None or spec.loader is None:
        raise Day20E2BPreflightV2Error("cannot import frozen E2B scorer")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise Day20E2BPreflightV2Error(
            f"cannot load frozen E2B scorer: {type(error).__name__}"
        ) from error
    return module


def _atomic_write_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise Day20E2BPreflightV2Error(f"refusing to overwrite marker: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except FileExistsError as error:
        raise Day20E2BPreflightV2Error("preflight marker already exists") from error
    finally:
        if temporary.exists():
            temporary.unlink()


def attest_credential(
    credential_file: Path,
    output_path: Path,
    *,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Create the non-secret, mode-0600 identity required by preflight."""
    credential = credential_file.resolve()
    _read_credential_secret(credential)
    timestamp = created_at_utc or datetime.now(timezone.utc).isoformat()
    if not isinstance(timestamp, str) or not timestamp:
        raise Day20E2BPreflightV2Error("created_at_utc must be non-empty text")
    result: dict[str, Any] = {
        "schema_version": 2,
        "domain": ATTESTATION_DOMAIN,
        "status": "rotated",
        "created_at_utc": timestamp,
        "credential_file_sha256": file_sha256(credential),
    }
    result["attestation_sha256"] = object_sha256(result)
    _atomic_write_new(output_path.resolve(), result)
    _require_private_file(output_path.resolve(), "credential attestation")
    return result


def run_preflight(
    *,
    run_root: Path,
    frozen_scorer: Path,
    sandbox_config: Path,
    humaneval_source: Path,
    credential_file: Path,
    credential_attestation: Path,
    marker_path: Path | None = None,
    module_loader: Callable[[Path], ModuleType] = _load_module,
    installed_e2b_version: str | None = None,
) -> dict[str, Any]:
    root = run_root.resolve()
    marker = (marker_path or root / "evidence" / MARKER_NAME).resolve()
    expected_parent = (root / "evidence").resolve()
    if marker.parent != expected_parent:
        raise Day20E2BPreflightV2Error("preflight marker must be under run_root/evidence")
    scorer_path = frozen_scorer.resolve()
    config_path = sandbox_config.resolve()
    source_path = humaneval_source.resolve()
    config = load_json(config_path)
    if config.get("evaluator_source_sha256") != file_sha256(scorer_path):
        raise Day20E2BPreflightV2Error("sandbox config/scorer identity drifted")
    source = config.get("source")
    if not isinstance(source, Mapping) or source.get("file_sha256") != file_sha256(source_path):
        raise Day20E2BPreflightV2Error("HumanEval source identity drifted")
    sdk = config.get("sdk")
    if sdk != {"package": "e2b", "version": EXPECTED_E2B_VERSION}:
        raise Day20E2BPreflightV2Error("sandbox SDK contract drifted")
    version = installed_e2b_version
    if version is None:
        try:
            version = importlib.metadata.version("e2b")
        except importlib.metadata.PackageNotFoundError as error:
            raise Day20E2BPreflightV2Error("pinned e2b SDK is not installed") from error
    if version != EXPECTED_E2B_VERSION:
        raise Day20E2BPreflightV2Error(
            f"expected e2b {EXPECTED_E2B_VERSION}, found {version}"
        )
    secret, credential_identity = verify_credential(
        credential_file, credential_attestation
    )
    frozen = module_loader(scorer_path)
    if not callable(getattr(frozen, "verify_config", None)) or not callable(
        getattr(frozen, "load_e2b_bindings", None)
    ):
        raise Day20E2BPreflightV2Error("frozen scorer API is incomplete")
    frozen.verify_config(config)
    previous = os.environ.get("E2B_API_KEY")
    os.environ["E2B_API_KEY"] = secret
    sandbox = None
    killed = False
    try:
        bindings = frozen.load_e2b_bindings(config)
        create = bindings.get("create") if isinstance(bindings, Mapping) else None
        if not callable(create):
            raise Day20E2BPreflightV2Error("E2B create binding is unavailable")
        sandbox = create(**frozen.sandbox_create_kwargs(config))
        mismatch = frozen.sandbox_info_mismatch(sandbox.get_info(), config)
        if mismatch is not None:
            raise Day20E2BPreflightV2Error(
                f"live sandbox attestation mismatch: {mismatch}"
            )
    finally:
        if sandbox is not None:
            killed = sandbox.kill() is not False
        if previous is None:
            os.environ.pop("E2B_API_KEY", None)
        else:
            os.environ["E2B_API_KEY"] = previous
    if not killed:
        raise Day20E2BPreflightV2Error("live sandbox was not killed")
    result: dict[str, Any] = {
        "schema_version": 2,
        "domain": DOMAIN,
        "status": "pass",
        "run_root": str(root),
        "sandbox_python": str(Path(os.sys.executable).resolve()),
        "e2b_sdk": {"package": "e2b", "version": version},
        "frozen_scorer": {
            "path": str(scorer_path),
            "file_sha256": file_sha256(scorer_path),
        },
        "sandbox_config": {
            "path": str(config_path),
            "file_sha256": file_sha256(config_path),
            "content_sha256": object_sha256(config),
        },
        "humaneval_source": {
            "path": str(source_path),
            "file_sha256": file_sha256(source_path),
        },
        # The path and digest are safe to persist; the secret bytes never are.
        # Revalidating this identity immediately before training prevents a
        # valid smoke marker from outliving a rotated or removed credential.
        "credential_file": {
            "path": str(credential_file.resolve()),
            "file_sha256": file_sha256(credential_file.resolve()),
        },
        "credential_attestation": credential_identity,
        "live_smoke": {
            "sandbox_created": True,
            "runtime_attested": True,
            "sandbox_killed": True,
            "candidate_code_executed": False,
        },
    }
    result["preflight_sha256"] = object_sha256(result)
    _atomic_write_new(marker, result)
    return result


def verify_preflight(
    marker_path: Path, *, run_root: Path, require_live_credential: bool = True
) -> dict[str, Any]:
    marker = marker_path.resolve()
    root = run_root.resolve()
    if marker.parent != (root / "evidence").resolve():
        raise Day20E2BPreflightV2Error("preflight marker is outside run_root/evidence")
    value = load_json(marker)
    _self_hash(value, "preflight_sha256")
    if (
        value.get("schema_version") != 2
        or value.get("domain") != DOMAIN
        or value.get("status") != "pass"
        or value.get("run_root") != str(root)
        or value.get("e2b_sdk")
        != {"package": "e2b", "version": EXPECTED_E2B_VERSION}
        or value.get("live_smoke")
        != {
            "sandbox_created": True,
            "runtime_attested": True,
            "sandbox_killed": True,
            "candidate_code_executed": False,
        }
    ):
        raise Day20E2BPreflightV2Error("E2B preflight marker identity drifted")
    for key in ("frozen_scorer", "sandbox_config", "humaneval_source"):
        identity = value.get(key)
        if not isinstance(identity, Mapping):
            raise Day20E2BPreflightV2Error(f"{key} identity is missing")
        path = Path(str(identity.get("path", ""))).resolve()
        if identity.get("file_sha256") != file_sha256(path):
            raise Day20E2BPreflightV2Error(f"{key} file identity drifted")
    config_identity = value["sandbox_config"]
    if config_identity.get("content_sha256") != object_sha256(
        load_json(Path(config_identity["path"]))
    ):
        raise Day20E2BPreflightV2Error("sandbox config content identity drifted")
    attestation = value.get("credential_attestation")
    credential = value.get("credential_file")
    if not isinstance(attestation, Mapping) or not isinstance(credential, Mapping):
        raise Day20E2BPreflightV2Error("credential identity is missing")
    attestation_path = Path(str(attestation.get("path", ""))).resolve()
    _require_private_file(attestation_path, "credential attestation")
    attestation_value = load_json(attestation_path)
    if (
        attestation.get("file_sha256") != file_sha256(attestation_path)
        or attestation.get("content_sha256")
        != _self_hash(attestation_value, "attestation_sha256")
    ):
        raise Day20E2BPreflightV2Error("credential attestation changed after smoke")
    if require_live_credential:
        credential_path = Path(str(credential.get("path", ""))).resolve()
        _require_private_file(credential_path, "credential file")
        credential_sha256 = file_sha256(credential_path)
        if (
            credential.get("file_sha256") != credential_sha256
            or attestation_value.get("credential_file_sha256") != credential_sha256
        ):
            raise Day20E2BPreflightV2Error("credential changed after live smoke")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--run-root", required=True, type=Path)
    run.add_argument("--frozen-scorer", required=True, type=Path)
    run.add_argument("--sandbox-config", required=True, type=Path)
    run.add_argument("--humaneval-source", required=True, type=Path)
    run.add_argument("--credential-file", required=True, type=Path)
    run.add_argument("--credential-attestation", required=True, type=Path)
    run.add_argument("--marker", type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--run-root", required=True, type=Path)
    verify.add_argument("--marker", required=True, type=Path)
    attest = subparsers.add_parser("attest-credential")
    attest.add_argument("--credential-file", required=True, type=Path)
    attest.add_argument("--output", required=True, type=Path)
    attest.add_argument("--created-at-utc")
    args = parser.parse_args()
    if args.command == "run":
        result = run_preflight(
            run_root=args.run_root,
            frozen_scorer=args.frozen_scorer,
            sandbox_config=args.sandbox_config,
            humaneval_source=args.humaneval_source,
            credential_file=args.credential_file,
            credential_attestation=args.credential_attestation,
            marker_path=args.marker,
        )
    elif args.command == "verify":
        result = verify_preflight(args.marker, run_root=args.run_root)
    else:
        result = attest_credential(
            args.credential_file,
            args.output,
            created_at_utc=args.created_at_utc,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
