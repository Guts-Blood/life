#!/usr/bin/env python3
"""Shared deterministic contracts for the Day 25 Qwen3.5 coding GRPO lab."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


DAY25_DIR = Path(__file__).resolve().parent
BOOTCAMP_ROOT = DAY25_DIR.parent
REPO_ROOT = BOOTCAMP_ROOT.parents[1]
ARTIFACTS = BOOTCAMP_ROOT / "artifacts"

MS_SWIFT_COMMIT = "565a1ad586a21d24b23931c52d2c62b49c39bee8"
MODEL_KEY = "Qwen/Qwen3.5-4B-Base"
MODEL_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
GROUP_SIZE = 4
TRAIN_FAMILIES = 32
EVAL_FAMILIES = 40
TRAIN_SELECTION_DOMAIN = "day25.grpo.train32.v1"
EVAL_DEV_SELECTION_DOMAIN = "day25.grpo.eval20.dev.v1"
EVAL_HELDOUT_SELECTION_DOMAIN = "day25.grpo.eval20.heldout.v1"
REWARD_POLICY = "tests_only"
POLICY_PARENT_ROLE = "day21_promoted_s1"
TARGET_REGEX = (
    r"^(?:(?:base_model|model)\.)*language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)


class Day25ContractError(ValueError):
    """A frozen Day 25 input, output, or runtime assumption drifted."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Day25ContractError(message)


def seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result.pop(field, None)
    result[field] = object_sha256(result)
    return result


def verify_seal(value: Mapping[str, Any], field: str, label: str) -> str:
    expected = value.get(field)
    actual = object_sha256({key: item for key, item in value.items() if key != field})
    require(isinstance(expected, str) and expected == actual, f"{label} {field} mismatch")
    return expected


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Day25ContractError(f"cannot read JSON: {path}") from error
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, 1):
                require(bool(line.strip()), f"blank JSONL row: {path}:{line_number}")
                value = json.loads(line)
                require(isinstance(value, dict), f"non-object JSONL row: {path}:{line_number}")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise Day25ContractError(f"cannot read JSONL: {path}") from error
    require(bool(rows), f"empty JSONL: {path}")
    return rows


def json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json(row) + b"\n" for row in rows)


def write_atomic(path: Path, payload: bytes, *, overwrite: bool) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise Day25ContractError(f"refusing to overwrite: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def stable_rank(domain: str, key: str) -> tuple[str, str]:
    return text_sha256(f"{domain}\0{key}"), key


def relative_to_bootcamp(path: Path) -> str:
    try:
        return path.resolve().relative_to(BOOTCAMP_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise Day25ContractError(f"path escaped bootcamp root: {path}") from error
