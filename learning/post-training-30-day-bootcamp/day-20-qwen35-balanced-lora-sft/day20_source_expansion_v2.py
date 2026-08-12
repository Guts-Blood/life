#!/usr/bin/env python3
"""Build the expanded, Qwen3.5-token-audited Day 20 v2 source pools.

The v1 normalized pools remain immutable inputs.  This adapter re-encodes
their exact messages after applying the v2 Code continuation contract and adds
compact FinQA/Tulu-Code candidates from the already pinned Day 09 manifest.
No dataset repository code is imported or executed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from day20_contract import (
    QWEN35_TEMPLATE_CONTRACT,
    SKILLS,
    TARGET_FORMATS,
    normalize_assistant_target,
    object_sha256,
    text_sha256,
)
from day20_contract_v2 import (
    Day20V2ContractError,
    canonicalize_code_continuation,
)
from day20_source_adapter import (
    SourceRecordRejected,
    audit_messages,
    build_qwen35_template,
    transform_tulu_code,
)


ADAPTER_VERSION = "day20_qwen35_source_expansion_v2"
FINQA_ADAPTER = "finqa_day09_value_scale_v2"
TULU_CODE_ADAPTER = "tulu_code_plain_python_continuation_v2"
REFERENCE_VERIFIER = "day20_reference_evidence_v2"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FINAL_ANSWER_RE = re.compile(r"(?m)^Final answer:\s*(?P<answer>[^\n]+?)\s*$")


class Day20SourceExpansionV2Error(ValueError):
    """A pinned input, normalized row, or output identity failed closed."""


def file_sha256(path: Path) -> str:
    if not path.is_file():
        raise Day20SourceExpansionV2Error(f"required file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Day20SourceExpansionV2Error(f"cannot load JSON: {path}") from error
    if not isinstance(value, dict):
        raise Day20SourceExpansionV2Error(f"JSON root must be an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise Day20SourceExpansionV2Error(
                        f"blank JSONL row: {path}:{line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise Day20SourceExpansionV2Error(
                        f"JSONL row is not an object: {path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise Day20SourceExpansionV2Error(f"cannot load JSONL: {path}") from error
    if not rows:
        raise Day20SourceExpansionV2Error(f"JSONL source is empty: {path}")
    return rows


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise Day20SourceExpansionV2Error(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl_new(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise Day20SourceExpansionV2Error(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise Day20SourceExpansionV2Error(f"stale temporary file exists: {temporary}")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def tokenizer_file_identity(model_path: Path) -> dict[str, str]:
    resolved = model_path.expanduser().resolve()
    names = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "added_tokens.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    )
    files = {
        name: file_sha256(resolved / name)
        for name in names
        if (resolved / name).is_file()
    }
    if "config.json" not in files or not any(
        name in files for name in ("tokenizer.json", "vocab.json")
    ):
        raise Day20SourceExpansionV2Error(
            "model path lacks config.json or tokenizer identity files"
        )
    return dict(sorted(files.items()))


def parse_source_specs(specs: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for spec in specs:
        skill, separator, raw_path = spec.partition("=")
        if not separator or skill not in SKILLS or not raw_path or skill in result:
            raise Day20SourceExpansionV2Error(
                "--base-source must provide each of general/math/finance/code once"
            )
        result[skill] = Path(raw_path).expanduser().resolve()
    if set(result) != set(SKILLS):
        raise Day20SourceExpansionV2Error("four --base-source inputs are required")
    return result


def verify_day09_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _load_json(path)
    header = manifest.get("header")
    rows = manifest.get("records")
    if not isinstance(header, dict) or not isinstance(rows, list):
        raise Day20SourceExpansionV2Error("Day 09 manifest shape is invalid")
    if header.get("manifest_version") != "day09_clean_parent_pool_v1":
        raise Day20SourceExpansionV2Error("Day 09 manifest version drifted")
    sample_ids = [row.get("sample_id") for row in rows if isinstance(row, dict)]
    if (
        len(sample_ids) != len(rows)
        or any(not isinstance(value, str) or not value for value in sample_ids)
        or sample_ids != sorted(sample_ids)
        or len(sample_ids) != len(set(sample_ids))
    ):
        raise Day20SourceExpansionV2Error("Day 09 records are not unique/stably ordered")
    declared = header.get("manifest_hash")
    payload_header = {key: value for key, value in header.items() if key != "manifest_hash"}
    recomputed = object_sha256({"header": payload_header, "records": rows})
    if declared != recomputed:
        raise Day20SourceExpansionV2Error("Day 09 manifest content hash drifted")
    return header, [dict(row) for row in rows]


def _reference_evidence(
    target_format: str, source_content_sha256: str, checks: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 2,
        "verifier_version": REFERENCE_VERIFIER,
        "target_format": target_format,
        "source_content_sha256": source_content_sha256,
        "checks": dict(checks),
    }
    result["reference_evidence_sha256"] = object_sha256(result)
    return result


def _source_lineage(
    row: Mapping[str, Any], *, adapter: str, source_file_sha256: str
) -> dict[str, Any]:
    source_content = object_sha256(row)
    return {
        "source": str(row.get("source")),
        "revision": str(row.get("revision")),
        "split": str(row.get("split")),
        "adapter": adapter,
        "license": str(row.get("license")),
        "source_file": "day09-dataset-manifest.json",
        "source_file_sha256": source_file_sha256,
        "source_content_sha256": source_content,
        "source_id": str(row.get("parent_id")),
        "dataset_code_execution": False,
        "day09_record_content_hash": str(row.get("content_hash")),
    }


def _finqa_candidate(row: Mapping[str, Any], source_file_sha256: str) -> dict[str, Any]:
    messages = row.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or [message.get("role") for message in messages if isinstance(message, dict)]
        != ["user", "assistant"]
    ):
        raise Day20SourceExpansionV2Error("FinQA row is not one user/assistant turn")
    prompt = messages[0].get("content")
    response = messages[1].get("content")
    if not isinstance(prompt, str) or not prompt.strip() or not isinstance(response, str):
        raise Day20SourceExpansionV2Error("FinQA row messages are malformed")
    matches = list(_FINAL_ANSWER_RE.finditer(response))
    if len(matches) != 1:
        raise Day20SourceExpansionV2Error("FinQA response needs exactly one final answer")
    raw_answer = matches[0].group("answer").strip()
    lowered = raw_answer.lower()
    if raw_answer.endswith("%"):
        scale = "percent"
    else:
        scale = next(
            (name for name in ("thousand", "million", "billion") if lowered.endswith(f" {name}")),
            "none",
        )
    target = normalize_assistant_target(
        "finance",
        "finance_value_scale",
        f"Final answer: {raw_answer}",
        finance_scale=scale,
    )
    prompt = (
        prompt.strip()
        + "\n\nReturn exactly one line: 'Final answer: <value>', preserving any "
        "required percent sign or scale word."
    )
    lineage = _source_lineage(
        row, adapter=FINQA_ADAPTER, source_file_sha256=source_file_sha256
    )
    checks = {
        "source_final_answer": raw_answer,
        "target_final_answer": target.removeprefix("Final answer:").strip(),
        "source_target_match": target.removeprefix("Final answer:").strip() == raw_answer,
    }
    return {
        "sample_id": f"finance:finqa-v2:{row['parent_id']}",
        "parent_id": str(row["parent_id"]),
        "skill": "finance",
        "target_format": "finance_value_scale",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ],
        "finance_scale": scale,
        "subskill": "financial_numerical_reasoning_finqa_v2",
        "source_lineage": lineage,
        "reference_evidence": _reference_evidence(
            "finance_value_scale", lineage["source_content_sha256"], checks
        ),
    }


def _ast_sha256(source: str) -> str:
    try:
        tree = ast.parse(source)
        compile(tree, "<day20-v2-source>", "exec")
    except (SyntaxError, ValueError, TypeError) as error:
        raise Day20SourceExpansionV2Error("source code does not compile") from error
    return text_sha256(ast.dump(tree, annotate_fields=True, include_attributes=False))


def _tulu_code_candidate(
    row: Mapping[str, Any], source_file_sha256: str
) -> dict[str, Any]:
    try:
        candidate = transform_tulu_code(row, str(row["parent_id"]))
        continuation = canonicalize_code_continuation(
            candidate["messages"][-1]["content"], candidate["code_prefix"]
        )
    except (KeyError, SourceRecordRejected, Day20V2ContractError) as error:
        raise Day20SourceExpansionV2Error(f"Tulu Code row rejected: {error}") from error
    candidate["sample_id"] = f"code:tulu-code-v2:{row['parent_id']}"
    candidate["messages"][-1]["content"] = continuation.canonical
    candidate["code_continuation_v2"] = continuation.as_evidence()
    lineage = _source_lineage(
        row, adapter=TULU_CODE_ADAPTER, source_file_sha256=source_file_sha256
    )
    raw_messages = row.get("messages")
    raw_source = raw_messages[-1]["content"] if isinstance(raw_messages, list) else None
    if not isinstance(raw_source, str):
        raise Day20SourceExpansionV2Error("Tulu Code source response is missing")
    rebuilt = f"{candidate['code_prefix']}\n{continuation.canonical}\n"
    source_ast = _ast_sha256(raw_source)
    rebuilt_ast = _ast_sha256(rebuilt)
    checks = {
        "source_ast_sha256": source_ast,
        "rebuilt_ast_sha256": rebuilt_ast,
        "source_rebuilt_ast_match": source_ast == rebuilt_ast,
        "static_compile": True,
    }
    if not checks["source_rebuilt_ast_match"]:
        raise Day20SourceExpansionV2Error("Tulu Code AST reconstruction drifted")
    candidate["source_lineage"] = lineage
    candidate["reference_evidence"] = _reference_evidence(
        "code_continuation", lineage["source_content_sha256"], checks
    )
    return candidate


def _stored_messages(row: Mapping[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) not in {2, 3}:
        raise Day20SourceExpansionV2Error("normalized row messages are malformed")
    result: list[dict[str, str]] = []
    for message in messages:
        if (
            not isinstance(message, Mapping)
            or message.get("role") not in {"system", "user", "assistant"}
            or not isinstance(message.get("content"), str)
            or not message["content"].strip()
        ):
            raise Day20SourceExpansionV2Error("normalized message is malformed")
        result.append({"role": str(message["role"]), "content": message["content"]})
    roles = [message["role"] for message in result]
    if roles not in (["user", "assistant"], ["system", "user", "assistant"]):
        raise Day20SourceExpansionV2Error("normalized message role sequence is invalid")
    return result


def _upgrade_base_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    messages = _stored_messages(row)
    if row.get("target_format") == "code_continuation":
        try:
            continuation = canonicalize_code_continuation(
                messages[-1]["content"], row.get("code_prefix")
            )
        except Day20V2ContractError as error:
            raise Day20SourceExpansionV2Error(
                f"base Code row {row.get('sample_id')} violates v2: {error}"
            ) from error
        messages[-1]["content"] = continuation.canonical
        result["code_continuation_v2"] = continuation.as_evidence()
    result["messages"] = messages
    return result


def _retokenize(row: Mapping[str, Any], template: Any) -> dict[str, Any]:
    result = dict(row)
    result["qwen35_tokenization"] = audit_messages(result["messages"], template)
    for key in (
        "qwen35_input_tokens",
        "qwen35_supervised_tokens",
        "qwen35_render_sha256",
        "qwen35_labels_sha256",
        "prompt_sha256",
        "content_sha256",
    ):
        result.pop(key, None)
    validate_v2_normalized_record(result, expected_skill=str(result.get("skill")))
    return result


def validate_v2_normalized_record(
    row: Mapping[str, Any], *, expected_skill: str, max_length: int = 2304
) -> dict[str, Any]:
    """Validate the GPU-materialized row shape used by v2 preparation."""
    if expected_skill not in SKILLS or row.get("skill") != expected_skill:
        raise Day20SourceExpansionV2Error("normalized row skill drifted")
    sample_id = row.get("sample_id")
    target_format = row.get("target_format")
    messages = _stored_messages(row)
    tokenization = row.get("qwen35_tokenization")
    lineage = row.get("source_lineage")
    reference = row.get("reference_evidence")
    if (
        not isinstance(sample_id, str)
        or not sample_id
        or TARGET_FORMATS.get(target_format) != expected_skill
        or not isinstance(tokenization, Mapping)
        or not isinstance(lineage, Mapping)
        or not isinstance(reference, Mapping)
    ):
        raise Day20SourceExpansionV2Error("normalized row identity is incomplete")
    evidence_payload = dict(reference)
    evidence_hash = evidence_payload.pop("reference_evidence_sha256", None)
    if (
        evidence_hash != object_sha256(evidence_payload)
        or reference.get("target_format") != target_format
        or reference.get("source_content_sha256")
        != lineage.get("source_content_sha256")
        or not _SHA256_RE.fullmatch(str(lineage.get("source_file_sha256", "")))
        or not _SHA256_RE.fullmatch(str(lineage.get("source_content_sha256", "")))
    ):
        raise Day20SourceExpansionV2Error(f"{sample_id}: reference lineage drifted")
    required_token_contract = {
        "template": "qwen3_5",
        "enable_thinking": False,
        "add_non_thinking_prefix": True,
        "loss_scale": "default+ignore_empty_think",
        "truncated": False,
    }
    if any(tokenization.get(key) != value for key, value in required_token_contract.items()):
        raise Day20SourceExpansionV2Error(f"{sample_id}: tokenization contract drifted")
    input_tokens = tokenization.get("input_tokens")
    supervised = tokenization.get("supervised_tokens")
    if (
        isinstance(input_tokens, bool)
        or not isinstance(input_tokens, int)
        or input_tokens <= 0
        or input_tokens > max_length
        or isinstance(supervised, bool)
        or not isinstance(supervised, int)
        or not 0 < supervised <= input_tokens
        or tokenization.get("messages_sha256") != object_sha256(messages)
        or not _SHA256_RE.fullmatch(str(tokenization.get("render_sha256", "")))
        or not _SHA256_RE.fullmatch(str(tokenization.get("labels_sha256", "")))
    ):
        raise Day20SourceExpansionV2Error(f"{sample_id}: token evidence is invalid")
    for key in ("source", "revision", "split", "adapter", "license"):
        if not isinstance(lineage.get(key), str) or not lineage[key]:
            raise Day20SourceExpansionV2Error(f"{sample_id}: source lineage is incomplete")
    if target_format == "code_continuation":
        try:
            evidence = canonicalize_code_continuation(
                messages[-1]["content"], row.get("code_prefix"), require_raw_contract=True
            )
        except Day20V2ContractError as error:
            raise Day20SourceExpansionV2Error(f"{sample_id}: {error}") from error
        stored_code = row.get("code_continuation_v2")
        if (
            not isinstance(stored_code, Mapping)
            or stored_code.get("canonical") != evidence.canonical
            or stored_code.get("canonical_sha256") != evidence.canonical_sha256
            or stored_code.get("ast_sha256") != evidence.ast_sha256
            or not isinstance(stored_code.get("raw"), str)
            or stored_code.get("raw_sha256")
            != text_sha256(str(stored_code.get("raw")))
        ):
            raise Day20SourceExpansionV2Error(f"{sample_id}: Code v2 evidence drifted")
    else:
        normalized = normalize_assistant_target(
            expected_skill,
            str(target_format),
            messages[-1]["content"],
            finance_scale=(
                row.get("finance_scale")
                if target_format == "finance_value_scale"
                else None
            ),
        )
        if normalized != messages[-1]["content"]:
            raise Day20SourceExpansionV2Error(f"{sample_id}: target is not canonical")
    return dict(row)


def build_expanded_sources(
    *,
    base_sources: Mapping[str, Path],
    day09_manifest_path: Path,
    output_dir: Path,
    template: Any,
    model_path: Path,
) -> dict[str, Any]:
    """Materialize append-only v2 pools; caller supplies the live Qwen template."""
    destination = output_dir.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise Day20SourceExpansionV2Error(f"output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    resolved_model = model_path.expanduser().resolve()
    tokenizer_files = tokenizer_file_identity(resolved_model)
    _, day09_rows = verify_day09_manifest(day09_manifest_path.resolve())
    day09_file_sha = file_sha256(day09_manifest_path.resolve())
    pools: dict[str, list[dict[str, Any]]] = {skill: [] for skill in SKILLS}
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    seen_messages: set[str] = set()
    rejections: Counter[str] = Counter()

    def admit(candidate: Mapping[str, Any], *, source_label: str) -> None:
        sample_id = str(candidate.get("sample_id", ""))
        messages = _stored_messages(candidate)
        prompt_hash = object_sha256(messages[:-1])
        messages_hash = object_sha256(messages)
        reason = None
        if sample_id in seen_ids:
            reason = "duplicate_sample_id"
        elif prompt_hash in seen_prompts:
            reason = "duplicate_prompt"
        elif messages_hash in seen_messages:
            reason = "duplicate_messages"
        if reason:
            rejections[f"{source_label}:{reason}"] += 1
            return
        normalized = _retokenize(candidate, template)
        skill = str(normalized["skill"])
        pools[skill].append(normalized)
        seen_ids.add(sample_id)
        seen_prompts.add(prompt_hash)
        seen_messages.add(messages_hash)

    for skill in SKILLS:
        source_path = Path(base_sources[skill]).resolve()
        for row in _load_jsonl(source_path):
            if row.get("skill") != skill:
                raise Day20SourceExpansionV2Error(f"base source skill drifted: {source_path}")
            admit(_upgrade_base_row(row), source_label=f"base_{skill}")

    for row in day09_rows:
        skill = row.get("skill")
        if skill not in {"finance", "code"}:
            continue
        try:
            candidate = (
                _finqa_candidate(row, day09_file_sha)
                if skill == "finance"
                else _tulu_code_candidate(row, day09_file_sha)
            )
            admit(candidate, source_label=f"day09_{skill}")
        except (Day20SourceExpansionV2Error, SourceRecordRejected, ValueError) as error:
            rejections[f"day09_{skill}:{type(error).__name__}"] += 1

    output_paths: dict[str, Path] = {}
    outputs: dict[str, Any] = {}
    for skill in SKILLS:
        rows = sorted(
            pools[skill],
            key=lambda row: (text_sha256(f"{ADAPTER_VERSION}\0{row['sample_id']}"), row["sample_id"]),
        )
        if not rows:
            raise Day20SourceExpansionV2Error(f"expanded {skill} pool is empty")
        path = destination / f"{skill}.normalized.qwen35-v2.jsonl"
        _write_jsonl_new(path, rows)
        output_paths[skill] = path
        outputs[skill] = {
            "path": str(path),
            "file_sha256": file_sha256(path),
            "records": len(rows),
            "supervised_tokens": sum(
                int(row["qwen35_tokenization"]["supervised_tokens"]) for row in rows
            ),
            "records_by_adapter": dict(
                sorted(Counter(str(row["source_lineage"]["adapter"]) for row in rows).items())
            ),
        }
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "domain": "day20.qwen35_source_expansion.v2",
        "status": "pass",
        "adapter_version": ADAPTER_VERSION,
        "dataset_code_execution": False,
        "model_path": str(resolved_model),
        "tokenizer_files": tokenizer_files,
        "template": dict(QWEN35_TEMPLATE_CONTRACT),
        "day09_manifest": {
            "path": str(day09_manifest_path.resolve()),
            "file_sha256": day09_file_sha,
        },
        "base_sources": {
            skill: {
                "path": str(Path(base_sources[skill]).resolve()),
                "file_sha256": file_sha256(Path(base_sources[skill]).resolve()),
            }
            for skill in SKILLS
        },
        "outputs": outputs,
        "rejections": dict(sorted(rejections.items())),
    }
    manifest["manifest_sha256"] = object_sha256(manifest)
    _write_json_new(destination / "SOURCE-EXPANSION-MANIFEST.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--day09-manifest", required=True, type=Path)
    parser.add_argument("--base-source", action="append", default=[])
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    sources = parse_source_specs(args.base_source)
    template = build_qwen35_template(args.model.expanduser().resolve())
    result = build_expanded_sources(
        base_sources=sources,
        day09_manifest_path=args.day09_manifest,
        output_dir=args.output_dir,
        template=template,
        model_path=args.model,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
