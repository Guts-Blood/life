#!/usr/bin/env python3
"""Prepare the auditable MBPP seed pool and deterministic Day 22 smoke candidates.

This module never executes candidate code.  It binds Day 20 normalized rows back
to the exact pinned MBPP train Parquet, removes probe overlap, freezes family
splits, and creates syntax-valid one-node AST mutations for sandbox replay.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
BOOTCAMP_ROOT = HERE.parent
DAY20_DIR = BOOTCAMP_ROOT / "day-20-qwen35-balanced-lora-sft"

import sys

if str(DAY20_DIR) not in sys.path:
    sys.path.insert(0, str(DAY20_DIR))

from day20_contract import object_sha256 as day20_object_sha256  # noqa: E402
from day20_contract_v2 import (  # noqa: E402
    Day20V2ContractError,
    canonicalize_code_continuation,
    validate_raw_code_continuation,
)
from day20_source_adapter import split_mbpp_continuation  # noqa: E402


MBPP_SOURCE = "google-research-datasets/mbpp"
MBPP_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
MBPP_LICENSE = "CC-BY-4.0"
EXPECTED_TRAIN_FILES = {
    "full": {
        "relative_path": "full/train-00000-of-00001.parquet",
        "sha256": "09d125ca31edacb7800be8c67c45abff618faf0214ff551291817d06bdb914ae",
    },
    "sanitized": {
        "relative_path": "sanitized/train-00000-of-00001.parquet",
        "sha256": "d95f8ad6d2fff08fe4826122d6e3e31f75716825d0c5c340d297aca5e9e0de0e",
    },
}
SPLIT_COUNTS = {"train": 264, "dev": 33, "heldout": 33}
SMOKE_COUNTS = {"train": 60, "dev": 10, "heldout": 10}
SPLIT_SEED = "day22-mbpp-family-v1"
SMOKE_SEED = "day22-mbpp-smoke-v1"
MUTATION_VERSION = "day22.mbpp_single_ast_bug_v1"
TASK_ID_RE = re.compile(r"^mbpp:(?:full|sanitized):(?:train|validation|test):(\d+)$")


class Day22BuildError(ValueError):
    """A source, lineage, split, or mutation invariant failed."""


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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Day22BuildError(message)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise Day22BuildError(f"blank JSONL row: {path}:{line_number}")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise Day22BuildError(
                        f"invalid JSONL row: {path}:{line_number}"
                    ) from error
                if not isinstance(row, dict):
                    raise Day22BuildError(
                        f"JSONL row is not an object: {path}:{line_number}"
                    )
                rows.append(row)
    except OSError as error:
        raise Day22BuildError(f"cannot read JSONL: {path}") from error
    _require(bool(rows), f"JSONL is empty: {path}")
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Day22BuildError(f"cannot read JSON: {path}") from error
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _write_bytes_atomic(path: Path, payload: bytes, *, overwrite: bool) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise Day22BuildError(f"refusing to overwrite: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise Day22BuildError(f"output appeared while writing: {path}") from error
            temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path: Path, value: Mapping[str, Any], *, overwrite: bool) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    _write_bytes_atomic(path, payload, overwrite=overwrite)


def write_jsonl(
    path: Path, rows: Sequence[Mapping[str, Any]], *, overwrite: bool
) -> None:
    payload = b"".join(canonical_json(row) + b"\n" for row in rows)
    _write_bytes_atomic(path, payload, overwrite=overwrite)


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise Day22BuildError(
            "pyarrow is required to read pinned MBPP Parquet; refusing to install it"
        ) from error
    try:
        rows = parquet.read_table(path).to_pylist()
    except Exception as error:
        raise Day22BuildError(f"cannot read MBPP Parquet {path}: {error}") from error
    _require(all(isinstance(row, dict) for row in rows), f"invalid rows in {path}")
    return rows


def resolve_raw_train(mbpp_root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for variant, identity in EXPECTED_TRAIN_FILES.items():
        path = mbpp_root / identity["relative_path"]
        _require(path.is_file(), f"missing pinned MBPP {variant} train file: {path}")
        actual = file_sha256(path)
        _require(
            actual == identity["sha256"],
            f"MBPP {variant} train SHA-256 mismatch: {actual}",
        )
        rows = _read_parquet(path)
        by_task: dict[int, tuple[int, dict[str, Any]]] = {}
        for row_index, row in enumerate(rows):
            task_id = row.get("task_id")
            _require(
                isinstance(task_id, int) and not isinstance(task_id, bool),
                f"MBPP {variant} task_id is invalid at row {row_index}",
            )
            _require(task_id not in by_task, f"duplicate MBPP {variant} task_id {task_id}")
            by_task[task_id] = (row_index, row)
        result[variant] = {
            "path": path.resolve(),
            "relative_path": identity["relative_path"],
            "sha256": actual,
            "rows": rows,
            "by_task": by_task,
        }
    return result


def _lineage(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("source_lineage")
    if not isinstance(value, Mapping):
        raise Day22BuildError(f"{row.get('sample_id')}: missing source lineage")
    return value


def _task_id(row: Mapping[str, Any]) -> int:
    lineage = _lineage(row)
    source_id = lineage.get("source_id", row.get("parent_id"))
    match = TASK_ID_RE.fullmatch(str(source_id))
    if match is None:
        raise Day22BuildError(f"{row.get('sample_id')}: invalid MBPP source id")
    return int(match.group(1))


def _is_mbpp(row: Mapping[str, Any]) -> bool:
    lineage = row.get("source_lineage")
    return isinstance(lineage, Mapping) and lineage.get("source") == MBPP_SOURCE


def _canonical_tests(record: Mapping[str, Any]) -> dict[str, Any]:
    tests: dict[str, Any] = {}
    for key in ("test_list", "challenge_test_list"):
        value = record.get(key) or []
        _require(
            isinstance(value, (list, tuple))
            and all(isinstance(item, str) and item.strip() for item in value),
            f"malformed MBPP {key}",
        )
        tests[key] = [str(item).strip() for item in value]
    setup = record.get("test_setup_code") or record.get("test_imports") or ""
    if isinstance(setup, list):
        setup = "\n".join(str(item).strip() for item in setup if str(item).strip())
    _require(isinstance(setup, str), "malformed MBPP test setup")
    tests["test_setup_code"] = setup.strip()
    return tests


def _prompt_text(row: Mapping[str, Any]) -> str:
    messages = row.get("messages")
    _require(
        isinstance(messages, list)
        and len(messages) == 2
        and all(isinstance(message, Mapping) for message in messages),
        f"{row.get('sample_id')}: messages are invalid",
    )
    _require(
        messages[0].get("role") == "user" and messages[1].get("role") == "assistant",
        f"{row.get('sample_id')}: unexpected message roles",
    )
    prompt = messages[0].get("content")
    _require(isinstance(prompt, str) and prompt, f"{row.get('sample_id')}: empty prompt")
    return prompt


def _assistant_text(row: Mapping[str, Any]) -> str:
    value = row["messages"][1].get("content")
    _require(isinstance(value, str) and value, f"{row.get('sample_id')}: empty answer")
    return value


def _problem_text(record: Mapping[str, Any], variant: str) -> str:
    value = record.get("prompt") if variant == "sanitized" else record.get("text")
    if value is None:
        value = record.get("text") if variant == "sanitized" else record.get("prompt")
    _require(isinstance(value, str) and value.strip(), "MBPP problem text is empty")
    return value.strip()


def _rank(seed: str, key: str) -> str:
    return text_sha256(f"{seed}\0{key}")


def _assign_exact_splits(task_ids: Iterable[int]) -> dict[int, str]:
    ordered = sorted(task_ids, key=lambda task_id: (_rank(SPLIT_SEED, str(task_id)), task_id))
    _require(len(ordered) == sum(SPLIT_COUNTS.values()), "unexpected family count for split")
    result: dict[int, str] = {}
    cursor = 0
    for split in ("train", "dev", "heldout"):
        count = SPLIT_COUNTS[split]
        for task_id in ordered[cursor : cursor + count]:
            result[task_id] = split
        cursor += count
    return result


def _smoke_ids(split_by_task: Mapping[int, str]) -> set[int]:
    result: set[int] = set()
    for split, count in SMOKE_COUNTS.items():
        members = [task_id for task_id, value in split_by_task.items() if value == split]
        members.sort(key=lambda task_id: (_rank(SMOKE_SEED, str(task_id)), task_id))
        result.update(members[:count])
    return result


def _verify_day20_row(
    row: Mapping[str, Any],
    raw: Mapping[str, Any],
    *,
    raw_index: int,
    variant: str,
    source_file_sha256: str,
) -> tuple[dict[str, Any], Any, str, str, str]:
    lineage = _lineage(row)
    task_id = _task_id(row)
    _require(lineage.get("revision") == MBPP_REVISION, f"task {task_id}: revision drift")
    _require(lineage.get("license") == MBPP_LICENSE, f"task {task_id}: license drift")
    _require(lineage.get("split") == "train", f"task {task_id}: non-train row")
    _require(lineage.get("variant") == variant, f"task {task_id}: variant drift")
    _require(
        lineage.get("source_file_sha256") == source_file_sha256,
        f"task {task_id}: source file hash drift",
    )
    _require(
        lineage.get("source_row_index") == raw_index,
        f"task {task_id}: source row index drift",
    )
    raw_hash = day20_object_sha256(raw)
    _require(
        raw_hash == lineage.get("source_content_sha256") == row.get("source_content_sha256"),
        f"task {task_id}: source content hash drift",
    )
    evidence = row.get("reference_evidence")
    _require(isinstance(evidence, Mapping), f"task {task_id}: missing reference evidence")
    checks = evidence.get("checks")
    _require(isinstance(checks, Mapping), f"task {task_id}: missing reference checks")
    tests = _canonical_tests(raw)
    tests_sha256 = day20_object_sha256(tests)
    _require(checks.get("tests_sha256") == tests_sha256, f"task {task_id}: tests drift")
    _require(checks.get("tests_count") == len(tests["test_list"]), f"task {task_id}: test count drift")
    _require(
        checks.get("challenge_tests_count") == len(tests["challenge_test_list"]),
        f"task {task_id}: challenge test count drift",
    )
    messages = row.get("messages")
    _require(isinstance(messages, list), f"task {task_id}: messages missing")
    _require(
        row.get("prompt_sha256") == day20_object_sha256(messages[:-1]),
        f"task {task_id}: Day20 prompt hash drift",
    )
    prefix = row.get("code_prefix")
    _require(isinstance(prefix, str) and prefix.strip(), f"task {task_id}: prefix missing")
    prompt = _prompt_text(row)
    prompt_migration = "day20_prompt_identity"
    try:
        chosen = canonicalize_code_continuation(_assistant_text(row), prefix)
        validate_raw_code_continuation(chosen.canonical, prefix)
    except Day20V2ContractError as error:
        source_code = raw.get("code")
        _require(isinstance(source_code, str), f"task {task_id}: raw code missing")
        try:
            source_tree = ast.parse(source_code)
            canonical_source = ast.unparse(source_tree)
            _require(
                ast.dump(ast.parse(canonical_source), include_attributes=False)
                == ast.dump(source_tree, include_attributes=False),
                f"task {task_id}: source AST normalization drifted",
            )
            prefix, continuation, _ = split_mbpp_continuation(canonical_source)
            chosen = canonicalize_code_continuation(continuation, prefix)
            validate_raw_code_continuation(chosen.canonical, prefix)
        except (SyntaxError, Day20V2ContractError, ValueError) as migration_error:
            raise Day22BuildError(
                f"task {task_id}: canonical chosen invalid: {error}; "
                f"AST migration failed: {migration_error}"
            ) from migration_error
        marker = "Starter code:\n"
        _require(marker in prompt, f"task {task_id}: prompt starter marker missing")
        prompt = prompt.split(marker, 1)[0] + marker + prefix
        prompt_migration = "ast_equivalent_full_source_indent_normalization_v1"
    return tests, chosen, prefix, prompt, prompt_migration


def build_seed_manifest(
    *, main_path: Path, probe_path: Path, mbpp_root: Path
) -> dict[str, Any]:
    raw_sources = resolve_raw_train(mbpp_root)
    main_rows = _load_jsonl(main_path)
    probe_rows = _load_jsonl(probe_path)

    mbpp_main = [row for row in main_rows if _is_mbpp(row)]
    mbpp_train = [row for row in mbpp_main if _lineage(row).get("split") == "train"]
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in mbpp_train:
        grouped[_task_id(row)].append(row)
    probe_ids = {_task_id(row) for row in probe_rows if _is_mbpp(row)}
    excluded = sorted(set(grouped) & probe_ids)
    retained_ids = sorted(set(grouped) - set(excluded))

    _require(len(main_rows) == 13_197, f"unexpected Day20 main size: {len(main_rows)}")
    _require(len(mbpp_main) == 1_134, f"unexpected Day20 MBPP size: {len(mbpp_main)}")
    _require(len(mbpp_train) == 422, f"unexpected Day20 MBPP train size: {len(mbpp_train)}")
    _require(len(grouped) == 351, f"unexpected train family count: {len(grouped)}")
    _require(len(excluded) == 21, f"unexpected probe overlap count: {len(excluded)}")
    _require(len(retained_ids) == 330, f"unexpected retained family count: {len(retained_ids)}")

    split_by_task = _assign_exact_splits(retained_ids)
    smoke_ids = _smoke_ids(split_by_task)
    records: list[dict[str, Any]] = []
    selected_variants: Counter[str] = Counter()
    for task_id in retained_ids:
        variants: dict[str, dict[str, Any]] = {}
        for row in grouped[task_id]:
            variant = str(_lineage(row).get("variant"))
            _require(variant in EXPECTED_TRAIN_FILES, f"task {task_id}: bad variant")
            _require(variant not in variants, f"task {task_id}: duplicate {variant} row")
            variants[variant] = row
        variant = "sanitized" if "sanitized" in variants else "full"
        row = variants[variant]
        selected_variants[variant] += 1
        raw_index, raw = raw_sources[variant]["by_task"].get(task_id, (None, None))
        _require(
            isinstance(raw_index, int) and isinstance(raw, dict),
            f"task {task_id}: absent from pinned {variant} train",
        )
        tests, chosen, prefix, prompt, prompt_migration = _verify_day20_row(
            row,
            raw,
            raw_index=raw_index,
            variant=variant,
            source_file_sha256=str(raw_sources[variant]["sha256"]),
        )
        problem = _problem_text(raw, variant)
        tests_sha256 = day20_object_sha256(tests)
        family_id = f"mbpp:task:{task_id}"
        record: dict[str, Any] = {
            "schema_name": "day22.mbpp_seed",
            "schema_version": 1,
            "task_family_id": family_id,
            "task_id": task_id,
            "split": split_by_task[task_id],
            "smoke_selected": task_id in smoke_ids,
            "source": {
                "dataset": MBPP_SOURCE,
                "revision": MBPP_REVISION,
                "license": MBPP_LICENSE,
                "split": "train",
                "variant": variant,
                "available_variants": sorted(variants),
                "source_file": raw_sources[variant]["relative_path"],
                "source_file_sha256": raw_sources[variant]["sha256"],
                "source_row_index": raw_index,
                "source_content_sha256": day20_object_sha256(raw),
                "day20_sample_id": row.get("sample_id"),
                "day20_prompt_sha256": row.get("prompt_sha256"),
                "prompt_text_sha256": text_sha256(_prompt_text(row)),
                "prompt_migration": prompt_migration,
            },
            "problem": {"text": problem, "sha256": text_sha256(problem)},
            "prompt": {"text": prompt, "sha256": text_sha256(prompt)},
            "code_prefix": prefix,
            "entry_point": prefix.split("def ", 1)[1].split("(", 1)[0].strip(),
            "canonical_chosen": {
                "text": chosen.canonical,
                "sha256": chosen.canonical_sha256,
                "ast_sha256": chosen.ast_sha256,
                "origin": "mbpp_canonical_solution",
            },
            "tests": {
                **tests,
                "sha256": tests_sha256,
                "count": len(tests["test_list"]),
                "challenge_count": len(tests["challenge_test_list"]),
            },
            "family_keys": {
                "problem": family_id,
                "prompt": f"sha256:{text_sha256(prompt)}",
                "test": f"sha256:{tests_sha256}",
                "source": family_id,
            },
        }
        record["record_sha256"] = object_sha256(record)
        records.append(record)

    records.sort(key=lambda record: int(record["task_id"]))
    split_counts = Counter(str(record["split"]) for record in records)
    smoke_counts = Counter(
        str(record["split"]) for record in records if record["smoke_selected"]
    )
    _require(dict(split_counts) == SPLIT_COUNTS, f"split counts drifted: {split_counts}")
    _require(dict(smoke_counts) == SMOKE_COUNTS, f"smoke counts drifted: {smoke_counts}")

    header: dict[str, Any] = {
        "schema_name": "day22.mbpp_seed_manifest",
        "schema_version": 1,
        "status": "seed_pool_frozen",
        "source_policy": {
            "dataset": MBPP_SOURCE,
            "revision": MBPP_REVISION,
            "license": MBPP_LICENSE,
            "allowed_upstream_split": "train",
            "variant_preference": ["sanitized", "full"],
            "validation_test_prompt_excluded": True,
            "source_heldout": "not_applicable_single_source",
        },
        "source_files": {
            variant: {
                "relative_path": value["relative_path"],
                "sha256": value["sha256"],
                "records": len(value["rows"]),
            }
            for variant, value in sorted(raw_sources.items())
        },
        "day20_inputs": {
            "main_file_sha256": file_sha256(main_path),
            "probe_file_sha256": file_sha256(probe_path),
            "main_records": len(main_rows),
            "main_mbpp_records": len(mbpp_main),
            "main_mbpp_train_records": len(mbpp_train),
            "main_mbpp_train_families": len(grouped),
            "probe_overlap_task_ids": excluded,
            "probe_overlap_count": len(excluded),
        },
        "selection": {
            "records": len(records),
            "clean_family_ids_sha256": object_sha256(retained_ids),
            "variant_counts": dict(sorted(selected_variants.items())),
            "split_seed": SPLIT_SEED,
            "split_counts": dict(split_counts),
            "smoke_seed": SMOKE_SEED,
            "smoke_counts": dict(smoke_counts),
            "family_unit": "native_mbpp_task_id",
        },
    }
    manifest = {"header": header, "records": records}
    manifest["manifest_sha256"] = object_sha256(manifest)
    return manifest


_COMPARE_SWAPS: dict[type[ast.cmpop], type[ast.cmpop]] = {
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
}
_BINOP_SWAPS: dict[type[ast.operator], type[ast.operator]] = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.FloorDiv,
    ast.FloorDiv: ast.Mult,
    ast.Mod: ast.FloorDiv,
}


def _mutation_specs(function: ast.FunctionDef, family_id: str) -> list[dict[str, Any]]:
    nodes = list(ast.walk(function))
    specs: list[dict[str, Any]] = []
    for ordinal, node in enumerate(nodes):
        location = {"line": getattr(node, "lineno", None), "column": getattr(node, "col_offset", None)}
        if isinstance(node, ast.Compare):
            for op_index, operator in enumerate(node.ops):
                replacement = _COMPARE_SWAPS.get(type(operator))
                if replacement is not None:
                    specs.append(
                        {
                            "kind": "compare",
                            "node_ordinal": ordinal,
                            "op_index": op_index,
                            "replacement": replacement.__name__,
                            "rule": f"{type(operator).__name__}_to_{replacement.__name__}",
                            "before": type(operator).__name__,
                            "after": replacement.__name__,
                            "location": location,
                            "priority": 0,
                        }
                    )
        elif isinstance(node, ast.BinOp):
            replacement = _BINOP_SWAPS.get(type(node.op))
            if replacement is not None:
                specs.append(
                    {
                        "kind": "binop",
                        "node_ordinal": ordinal,
                        "replacement": replacement.__name__,
                        "rule": f"{type(node.op).__name__}_to_{replacement.__name__}",
                        "before": type(node.op).__name__,
                        "after": replacement.__name__,
                        "location": location,
                        "priority": 1,
                    }
                )
        elif isinstance(node, ast.BoolOp):
            replacement: type[ast.boolop] | None = None
            if isinstance(node.op, ast.And):
                replacement = ast.Or
            elif isinstance(node.op, ast.Or):
                replacement = ast.And
            if replacement is not None:
                specs.append(
                    {
                        "kind": "boolop",
                        "node_ordinal": ordinal,
                        "replacement": replacement.__name__,
                        "rule": f"{type(node.op).__name__}_to_{replacement.__name__}",
                        "before": type(node.op).__name__,
                        "after": replacement.__name__,
                        "location": location,
                        "priority": 1,
                    }
                )
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, int)
            and not isinstance(node.value, bool)
        ):
            for delta in (-1, 1):
                replacement_value = int(node.value) + delta
                specs.append(
                    {
                        "kind": "integer",
                        "node_ordinal": ordinal,
                        "replacement": replacement_value,
                        "rule": "integer_minus_one" if delta < 0 else "integer_plus_one",
                        "before": int(node.value),
                        "after": replacement_value,
                        "location": location,
                        "priority": 2,
                    }
                )
        elif isinstance(node, (ast.If, ast.While)):
            specs.append(
                {
                    "kind": "condition_not",
                    "node_ordinal": ordinal,
                    "replacement": "Not",
                    "rule": "condition_negated",
                    "before": ast.unparse(node.test),
                    "after": f"not ({ast.unparse(node.test)})",
                    "location": location,
                    "priority": 3,
                }
            )
        elif isinstance(node, ast.Return) and node.value is not None:
            specs.append(
                {
                    "kind": "return_none",
                    "node_ordinal": ordinal,
                    "replacement": None,
                    "rule": "return_value_to_none",
                    "before": ast.unparse(node.value),
                    "after": "None",
                    "location": location,
                    "priority": 4,
                }
            )
    for spec in specs:
        spec["rank"] = _rank(
            MUTATION_VERSION,
            f"{family_id}:{spec['node_ordinal']}:{spec['rule']}:{spec.get('op_index', '')}",
        )
    return sorted(specs, key=lambda spec: (int(spec["priority"]), str(spec["rank"])))


def _operator_class(name: str) -> type[Any]:
    value = getattr(ast, name, None)
    if not isinstance(value, type):
        raise Day22BuildError(f"unknown AST operator: {name}")
    return value


def _apply_mutation(function: ast.FunctionDef, spec: Mapping[str, Any]) -> ast.FunctionDef:
    mutated = copy.deepcopy(function)
    nodes = list(ast.walk(mutated))
    ordinal = int(spec["node_ordinal"])
    _require(0 <= ordinal < len(nodes), "mutation node ordinal is out of range")
    target = nodes[ordinal]
    kind = spec["kind"]
    if kind == "compare":
        _require(isinstance(target, ast.Compare), "mutation Compare target drifted")
        op_index = int(spec["op_index"])
        target.ops[op_index] = _operator_class(str(spec["replacement"]))()
    elif kind == "binop":
        _require(isinstance(target, ast.BinOp), "mutation BinOp target drifted")
        target.op = _operator_class(str(spec["replacement"]))()
    elif kind == "boolop":
        _require(isinstance(target, ast.BoolOp), "mutation BoolOp target drifted")
        target.op = _operator_class(str(spec["replacement"]))()
    elif kind == "integer":
        _require(isinstance(target, ast.Constant), "mutation Constant target drifted")
        target.value = int(spec["replacement"])
    elif kind == "condition_not":
        _require(isinstance(target, (ast.If, ast.While)), "mutation condition target drifted")
        target.test = ast.UnaryOp(op=ast.Not(), operand=target.test)
    elif kind == "return_none":
        _require(isinstance(target, ast.Return), "mutation Return target drifted")
        target.value = ast.Constant(value=None)
    else:
        raise Day22BuildError(f"unknown mutation kind: {kind}")
    ast.fix_missing_locations(mutated)
    return mutated


def _continuation_function(continuation: str) -> ast.FunctionDef:
    try:
        module = ast.parse(f"def __day22_mutation_target__():\n{continuation}\n")
    except SyntaxError as error:
        raise Day22BuildError(f"cannot parse canonical continuation: {error.msg}") from error
    _require(
        len(module.body) == 1 and isinstance(module.body[0], ast.FunctionDef),
        "canonical continuation wrapper is invalid",
    )
    return module.body[0]


def generate_mutants(record: Mapping[str, Any], *, limit: int = 12) -> list[dict[str, Any]]:
    chosen = record.get("canonical_chosen")
    _require(isinstance(chosen, Mapping), "seed chosen is missing")
    chosen_text = chosen.get("text")
    prefix = record.get("code_prefix")
    family_id = record.get("task_family_id")
    _require(isinstance(chosen_text, str), "seed chosen text is missing")
    _require(isinstance(prefix, str), "seed prefix is missing")
    _require(isinstance(family_id, str), "seed family id is missing")
    function = _continuation_function(chosen_text)
    specs = _mutation_specs(function, family_id)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = {text_sha256(chosen_text)}
    for spec in specs:
        if len(candidates) >= limit:
            break
        mutated = _apply_mutation(function, spec)
        body = "\n".join(ast.unparse(statement) for statement in mutated.body)
        response = "\n".join(f"    {line}" if line else "" for line in body.splitlines())
        try:
            evidence = validate_raw_code_continuation(response, prefix)
        except Day20V2ContractError:
            continue
        response_hash = text_sha256(evidence.canonical)
        if response_hash in seen or evidence.ast_sha256 == chosen.get("ast_sha256"):
            continue
        seen.add(response_hash)
        metadata = {
            "version": MUTATION_VERSION,
            "rule": spec["rule"],
            "node_ordinal": spec["node_ordinal"],
            "source_location": spec["location"],
            "before": spec["before"],
            "after": spec["after"],
            "rank": spec["rank"],
        }
        candidate: dict[str, Any] = {
            "candidate_id": f"{family_id}:mutation:{len(candidates):02d}",
            "origin": "deterministic_ast_mutation",
            "trust_tag": "deterministic_mutation",
            "text": evidence.canonical,
            "sha256": evidence.canonical_sha256,
            "ast_sha256": evidence.ast_sha256,
            "mutation": metadata,
        }
        candidate["candidate_sha256"] = object_sha256(candidate)
        candidates.append(candidate)
    return candidates


def build_candidate_rows(
    manifest: Mapping[str, Any], *, scope: str, mutation_limit: int
) -> list[dict[str, Any]]:
    expected = manifest.get("manifest_sha256")
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    _require(expected == object_sha256(payload), "seed manifest self-hash is invalid")
    records = manifest.get("records")
    _require(isinstance(records, list), "seed manifest records are missing")
    rows: list[dict[str, Any]] = []
    for record in records:
        _require(isinstance(record, Mapping), "seed record is not an object")
        if scope == "smoke" and record.get("smoke_selected") is not True:
            continue
        chosen = dict(record["canonical_chosen"])
        chosen["candidate_id"] = f"{record['task_family_id']}:canonical"
        chosen["trust_tag"] = "canonical"
        chosen["candidate_sha256"] = object_sha256(chosen)
        mutants = generate_mutants(record, limit=mutation_limit)
        row: dict[str, Any] = {
            "schema_name": "day22.mbpp_replay_candidates",
            "schema_version": 1,
            "task_family_id": record["task_family_id"],
            "task_id": record["task_id"],
            "split": record["split"],
            "family_keys": record["family_keys"],
            "source": record["source"],
            "problem": record["problem"],
            "prompt": record["prompt"],
            "code_prefix": record["code_prefix"],
            "entry_point": record["entry_point"],
            "tests": record["tests"],
            "chosen": chosen,
            "mutants": mutants,
        }
        row["row_sha256"] = object_sha256(row)
        rows.append(row)
    _require(bool(rows), "candidate selection is empty")
    return rows


def build_replay_rows(
    candidate_rows: Sequence[Mapping[str, Any]], *, mutant_index: int = 0
) -> list[dict[str, Any]]:
    _require(mutant_index >= 0, "mutant index must be non-negative")
    rows: list[dict[str, Any]] = []
    for candidate_row in candidate_rows:
        mutants = candidate_row.get("mutants")
        _require(isinstance(mutants, list), "candidate mutants are missing")
        if len(mutants) <= mutant_index:
            continue
        chosen = candidate_row.get("chosen")
        rejected = mutants[mutant_index]
        _require(isinstance(chosen, Mapping), "candidate chosen is missing")
        _require(isinstance(rejected, Mapping), "candidate rejected is invalid")
        family_id = str(candidate_row["task_family_id"])
        replay: dict[str, Any] = {
            "schema_name": "day22.mbpp_replay_pair",
            "schema_version": 1,
            "pair_id": f"{family_id}:smoke:{mutant_index:02d}",
            "task_id": str(candidate_row["task_id"]),
            "family_id": family_id,
            "split": candidate_row["split"],
            "family_keys": candidate_row["family_keys"],
            "source": candidate_row["source"],
            "problem": candidate_row["problem"],
            "prompt": candidate_row["prompt"],
            "code_prefix": candidate_row["code_prefix"],
            "entry_point": candidate_row["entry_point"],
            "tests": candidate_row["tests"],
            "tests_sha256": candidate_row["tests"]["sha256"],
            "chosen": {
                "candidate_id": chosen["candidate_id"],
                "origin": "mbpp_canonical",
                "text": chosen["text"],
                "sha256": chosen["sha256"],
                "ast_sha256": chosen["ast_sha256"],
            },
            "rejected": {
                "candidate_id": rejected["candidate_id"],
                "origin": "deterministic_mutation",
                "text": rejected["text"],
                "sha256": rejected["sha256"],
                "ast_sha256": rejected["ast_sha256"],
                "mutation": rejected["mutation"],
            },
            "creation": {
                "method": "canonical_vs_deterministic_ast_mutation",
                "version": MUTATION_VERSION,
            },
        }
        replay["row_sha256"] = object_sha256(replay)
        rows.append(replay)
    return rows


def _seed_command(args: argparse.Namespace) -> dict[str, Any]:
    manifest = build_seed_manifest(
        main_path=args.main.resolve(),
        probe_path=args.probe.resolve(),
        mbpp_root=args.mbpp_root.resolve(),
    )
    write_json(args.output, manifest, overwrite=args.overwrite)
    return {
        "output": str(args.output.resolve()),
        "file_sha256": file_sha256(args.output.resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "records": len(manifest["records"]),
        "split_counts": manifest["header"]["selection"]["split_counts"],
        "smoke_counts": manifest["header"]["selection"]["smoke_counts"],
    }


def _mutate_command(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _load_json(args.seed.resolve())
    rows = build_candidate_rows(
        manifest, scope=args.scope, mutation_limit=args.mutation_limit
    )
    write_jsonl(args.output, rows, overwrite=args.overwrite)
    mutant_counts = [len(row["mutants"]) for row in rows]
    return {
        "output": str(args.output.resolve()),
        "file_sha256": file_sha256(args.output.resolve()),
        "records": len(rows),
        "mutants": sum(mutant_counts),
        "tasks_without_mutants": sum(count == 0 for count in mutant_counts),
        "minimum_mutants_per_task": min(mutant_counts),
        "maximum_mutants_per_task": max(mutant_counts),
    }


def _replay_command(args: argparse.Namespace) -> dict[str, Any]:
    candidate_rows = _load_jsonl(args.candidates.resolve())
    rows = build_replay_rows(candidate_rows, mutant_index=args.mutant_index)
    _require(bool(rows), "replay pair selection is empty")
    write_jsonl(args.output, rows, overwrite=args.overwrite)
    return {
        "output": str(args.output.resolve()),
        "file_sha256": file_sha256(args.output.resolve()),
        "records": len(rows),
        "mutant_index": args.mutant_index,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    seed = subparsers.add_parser("seed", help="freeze the 330-family MBPP train pool")
    seed.add_argument("--main", required=True, type=Path)
    seed.add_argument("--probe", required=True, type=Path)
    seed.add_argument("--mbpp-root", required=True, type=Path)
    seed.add_argument("--output", required=True, type=Path)
    seed.add_argument("--overwrite", action="store_true")
    mutate = subparsers.add_parser("mutate", help="create deterministic AST smoke candidates")
    mutate.add_argument("--seed", required=True, type=Path)
    mutate.add_argument("--output", required=True, type=Path)
    mutate.add_argument("--scope", choices=("smoke", "all"), default="smoke")
    mutate.add_argument("--mutation-limit", type=int, default=12)
    mutate.add_argument("--overwrite", action="store_true")
    replay = subparsers.add_parser("replay", help="select one mutant per task for sandbox replay")
    replay.add_argument("--candidates", required=True, type=Path)
    replay.add_argument("--output", required=True, type=Path)
    replay.add_argument("--mutant-index", type=int, default=0)
    replay.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "seed":
        result = _seed_command(args)
    elif args.command == "mutate":
        _require(args.mutation_limit > 0, "mutation limit must be positive")
        result = _mutate_command(args)
    else:
        _require(args.mutant_index >= 0, "mutant index must be non-negative")
        result = _replay_command(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Day22BuildError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
