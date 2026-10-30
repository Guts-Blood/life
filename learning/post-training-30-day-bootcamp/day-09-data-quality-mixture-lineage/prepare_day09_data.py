#!/usr/bin/env python3
"""Download, select, canonicalize, and preprocess Day 09 source records."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import importlib.util
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "day09_assignment.json"
DEFAULT_CACHE = HERE.parents[2] / "tmp" / "day09-hf-cache"
DEFAULT_WORK = HERE.parents[2] / "tmp" / "day09-work"
DEFAULT_TOKENIZER_CACHE = DEFAULT_CACHE / "tokenizer"
DAY08_MODULE = HERE.parent / "day-08-sft-data-contract" / "inspect_sft_sample.py"


class DataPreparationError(ValueError):
    """A deterministic source or transformation invariant failed."""


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def selection_hash(source_id: str, revision: str, parent_id: str) -> str:
    payload = "\0".join((source_id, revision, parent_id)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_records(
    records: Iterable[dict[str, Any]],
    *,
    source_id: str,
    revision: str,
    count: int,
    parent_id_field: str = "id",
) -> list[dict[str, Any]]:
    """Return the deterministic hash-ranked subset after validating all IDs."""
    if count <= 0:
        raise ValueError("count must be positive")
    seen: set[str] = set()

    def ranked() -> Iterator[tuple[str, str, dict[str, Any]]]:
        for record in records:
            raw_parent_id = record.get(parent_id_field)
            if raw_parent_id is None or not str(raw_parent_id).strip():
                raise DataPreparationError(f"missing parent id field {parent_id_field!r}")
            parent_id = str(raw_parent_id).strip()
            if parent_id in seen:
                raise DataPreparationError(f"duplicate parent id: {parent_id}")
            seen.add(parent_id)
            yield (
                selection_hash(source_id, revision, parent_id),
                parent_id,
                record,
            )

    selected = heapq.nsmallest(count, ranked(), key=lambda item: (item[0], item[1]))
    if len(selected) < count:
        raise DataPreparationError(
            f"source {source_id} has {len(selected)} records, fewer than requested {count}"
        )
    return [
        {"selection_hash": digest, "parent_id": parent_id, "record": record}
        for digest, parent_id, record in selected
    ]


def canonicalize_tulu(record: dict[str, Any]) -> list[dict[str, str]]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise DataPreparationError("Tulu record has no messages")
    canonical: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise DataPreparationError(f"Tulu message {index} is not an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise DataPreparationError(f"Tulu message {index} lacks string role/content")
        canonical.append({"role": role, "content": content})
    return canonical


def _finqa_evidence(record: dict[str, Any]) -> list[str]:
    raw_evidence = record.get("gold_inds")
    if isinstance(raw_evidence, str):
        try:
            raw_evidence = json.loads(raw_evidence)
        except json.JSONDecodeError as error:
            raise DataPreparationError("FinQA gold_inds is invalid JSON") from error
    if not isinstance(raw_evidence, dict):
        raise DataPreparationError("FinQA gold_inds must be an object")
    evidence = [str(value).strip() for value in raw_evidence.values() if str(value).strip()]
    if not evidence:
        raise DataPreparationError("FinQA record has no gold supporting evidence")
    return evidence


def canonicalize_finqa(record: dict[str, Any]) -> list[dict[str, str]]:
    question = str(record.get("question") or "").strip()
    program = str(record.get("program") or "").strip()
    answer = str(record.get("answer") or record.get("exe_ans") or "").strip()
    if not question or not program or not answer:
        raise DataPreparationError("FinQA record requires question, program, and answer")
    evidence_lines = "\n".join(f"- {item}" for item in _finqa_evidence(record))
    prompt = (
        "Use the evidence below to answer the financial question.\n\n"
        f"Evidence:\n{evidence_lines}\n\nQuestion: {question}"
    )
    response = f"Calculation: {program}\nFinal answer: {answer}"
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def canonicalize_record(adapter: str, record: dict[str, Any]) -> list[dict[str, str]]:
    if adapter == "tulu_messages_v1":
        return canonicalize_tulu(record)
    if adapter == "finqa_gold_evidence_v1":
        return canonicalize_finqa(record)
    raise DataPreparationError(f"unknown adapter: {adapter}")


def load_day08_contract() -> Any:
    """Load the frozen Day 08 implementation without copying its mask logic."""
    spec = importlib.util.spec_from_file_location("day08_inspect_sft_sample", DAY08_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Day 08 contract from {DAY08_MODULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def content_hash(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(
        messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def raw_token_count(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    return sum(
        len(tokenizer(message["content"], add_special_tokens=False)["input_ids"])
        for message in messages
    )


def preprocess_candidate(
    tokenizer: Any,
    contract: Any,
    candidate: dict[str, Any],
    max_length: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = candidate["messages"]
    sample = {
        "id": candidate["sample_id"],
        "messages": messages,
        "metadata": {
            "source": candidate["source"],
            "revision": candidate["revision"],
            "parent_id": candidate["parent_id"],
            "license": candidate["license"],
        },
    }
    audit = contract.encode_sample(tokenizer, sample, max_length)
    accepted = {
        **candidate,
        "content_hash": content_hash(messages),
        "transform_chain": [
            *candidate["transform_chain"],
            f"day08_sft_data_contract_v1:max_length={max_length}",
        ],
        "raw_token_count": raw_token_count(tokenizer, messages),
        "rendered_token_count": audit["original_length"],
        "input_token_count": audit["encoded_length"],
        "supervised_token_count": audit["effective_label_tokens"],
        "truncated": audit["truncated"],
    }
    return accepted, audit


def load_frozen_tokenizer(
    config: dict[str, Any], cache_dir: Path, *, local_files_only: bool
) -> tuple[Any, dict[str, Any]]:
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    preprocessing = config["preprocessing"]
    expected_version = preprocessing["transformers_version"]
    if transformers.__version__ != expected_version:
        raise DataPreparationError(
            f"transformers version mismatch: expected {expected_version}, "
            f"got {transformers.__version__}"
        )
    snapshot_path = Path(
        snapshot_download(
            repo_id=preprocessing["model_id"],
            revision=preprocessing["model_revision"],
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            allow_patterns=[
                "added_tokens.json",
                "chat_template.jinja",
                "merges.txt",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
            ],
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_path,
        local_files_only=True,
        use_fast=True,
    )
    template_sha256 = hashlib.sha256(
        (tokenizer.chat_template or "").encode("utf-8")
    ).hexdigest()
    tokenizer_config_sha256 = _sha256_file(snapshot_path / "tokenizer_config.json")
    if template_sha256 != preprocessing["chat_template_sha256"]:
        raise DataPreparationError("chat template hash does not match Day 08")
    if tokenizer_config_sha256 != preprocessing["tokenizer_config_sha256"]:
        raise DataPreparationError("tokenizer config hash does not match Day 08")
    return tokenizer, {
        "model_id": preprocessing["model_id"],
        "model_revision": preprocessing["model_revision"],
        "transformers_version": transformers.__version__,
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_sha256": template_sha256,
        "tokenizer_config_sha256": tokenizer_config_sha256,
    }


def _source_dir(cache_dir: Path, source: dict[str, Any]) -> Path:
    safe_id = source["source_id"].replace("/", "--")
    return cache_dir / "snapshots" / f"{safe_id}--{source['revision']}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_source(source: dict[str, Any], cache_dir: Path) -> dict[str, Any]:
    if source.get("source_type") != "huggingface_dataset":
        raise DataPreparationError(f"unsupported source type: {source.get('source_type')}")

    from huggingface_hub import HfApi, snapshot_download

    source_id = source["source_id"]
    revision = source["revision"]
    resolved_revision = HfApi().dataset_info(source_id, revision=revision).sha
    if resolved_revision != revision:
        raise DataPreparationError(
            f"revision mismatch for {source_id}: expected {revision}, got {resolved_revision}"
        )

    local_dir = _source_dir(cache_dir, source)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=source_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=source["allow_patterns"],
        local_dir=local_dir,
    )
    files = []
    for path in sorted(local_dir.rglob("*")):
        if path.is_file() and ".cache" not in path.parts:
            files.append(
                {
                    "path": str(path.relative_to(local_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    return {
        "source_id": source_id,
        "slice": source["slice"],
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "local_dir": str(local_dir),
        "files": files,
    }


def download_all(config: dict[str, Any], cache_dir: Path, source_slices: set[str]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = cache_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for source in config["sources"]:
        if source_slices and source["slice"] not in source_slices:
            continue
        metadata = download_source(source, cache_dir)
        metadata_path = metadata_dir / f"{source['slice']}-snapshot.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        total_bytes = sum(item["bytes"] for item in metadata["files"])
        print(
            f"{source['slice']}: {metadata['resolved_revision']} "
            f"{len(metadata['files'])} files {total_bytes} bytes"
        )


def iter_parquet_records(paths: list[Path]) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as parquet

    for path in paths:
        parquet_file = parquet.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=2048):
            yield from batch.to_pylist()


def select_source(
    source: dict[str, Any], cache_dir: Path, work_dir: Path, count: int
) -> dict[str, Any]:
    source_dir = _source_dir(cache_dir, source)
    snapshot_metadata_path = cache_dir / "metadata" / f"{source['slice']}-snapshot.json"
    if not snapshot_metadata_path.is_file():
        raise DataPreparationError(f"missing snapshot metadata for {source['slice']}")
    with snapshot_metadata_path.open(encoding="utf-8") as handle:
        snapshot_metadata = json.load(handle)
    if snapshot_metadata.get("resolved_revision") != source["revision"]:
        raise DataPreparationError(
            f"snapshot metadata revision mismatch for {source['slice']}"
        )
    paths = sorted(source_dir.glob("data/train-*.parquet"))
    if not paths:
        raise DataPreparationError(f"no train parquet files found for {source['slice']}")
    selected = select_records(
        iter_parquet_records(paths),
        source_id=source["source_id"],
        revision=source["revision"],
        count=count,
        parent_id_field=source["parent_id_field"],
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path = work_dir / f"{source['slice']}-candidates.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for item in selected:
            messages = canonicalize_record(source["adapter"], item["record"])
            candidate = {
                "sample_id": f"{source['slice']}:{item['parent_id']}",
                "source": source["source_id"],
                "revision": source["revision"],
                "split": source["split"],
                "parent_id": item["parent_id"],
                "selection_hash": item["selection_hash"],
                "skill": source["slice"],
                "language": "en",
                "license": source["license"],
                "retrieved_at": snapshot_metadata["retrieved_at"],
                "messages": messages,
                "transform_chain": [source["adapter"]],
            }
            if "upstream" in source:
                candidate["upstream"] = source["upstream"]
            handle.write(json.dumps(candidate, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "slice": source["slice"],
        "candidate_count": len(selected),
        "duplicate_parent_ids": 0,
        "output": str(output_path),
        "selection_sha256": _sha256_file(output_path),
    }


def select_all(config: dict[str, Any], cache_dir: Path, work_dir: Path) -> None:
    summaries = [
        select_source(
            source,
            cache_dir,
            work_dir,
            config["candidate_samples_per_slice"],
        )
        for source in config["sources"]
    ]
    summary_path = work_dir / "selection-summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    for summary in summaries:
        print(
            f"{summary['slice']}: {summary['candidate_count']} candidates "
            f"sha256={summary['selection_sha256']}"
        )


def _candidate_rejection(
    candidate: dict[str, Any], tokenizer: Any, error: Any, max_length: int
) -> dict[str, Any]:
    messages = candidate.get("messages")
    before_state = {
        "message_count": len(messages) if isinstance(messages, list) else None,
        "raw_token_count": None,
    }
    if isinstance(messages, list):
        try:
            before_state["raw_token_count"] = raw_token_count(tokenizer, messages)
        except (KeyError, TypeError):
            pass
    return {
        "sample_id": candidate.get("sample_id"),
        "source": candidate.get("source"),
        "revision": candidate.get("revision"),
        "split": candidate.get("split"),
        "parent_id": candidate.get("parent_id"),
        "selection_hash": candidate.get("selection_hash"),
        "content_hash": content_hash(messages) if isinstance(messages, list) else None,
        "transform_chain": [
            *candidate.get("transform_chain", []),
            f"day08_sft_data_contract_v1:max_length={max_length}:rejected",
        ],
        "rejected_stage": error.stage,
        "reason_code": error.code,
        "detail": str(error),
        "before_state": before_state,
    }


def preprocess_source(
    source: dict[str, Any],
    candidate_dir: Path,
    output_dir: Path,
    tokenizer: Any,
    contract: Any,
    max_length: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    candidate_path = candidate_dir / f"{source['slice']}-candidates.jsonl"
    if not candidate_path.is_file():
        raise DataPreparationError(f"missing candidate file: {candidate_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = output_dir / f"{source['slice']}-accepted.jsonl"
    rejected_path = output_dir / f"{source['slice']}-rejected.jsonl"
    candidate_count = 0
    accepted_count = 0
    accepted_truncated_count = 0
    rejected_count = 0
    rejected_reasons: Counter[str] = Counter()
    raw_tokens = 0
    input_tokens = 0
    supervised_tokens = 0
    first_audit = None

    with (
        candidate_path.open(encoding="utf-8") as candidates,
        accepted_path.open("w", encoding="utf-8") as accepted_output,
        rejected_path.open("w", encoding="utf-8") as rejected_output,
    ):
        for line in candidates:
            if not line.strip():
                continue
            candidate_count += 1
            candidate = json.loads(line)
            candidate["subskill"] = source["subskill"]
            try:
                accepted, audit = preprocess_candidate(
                    tokenizer,
                    contract,
                    candidate,
                    max_length,
                )
            except contract.ContractError as error:
                rejected = _candidate_rejection(candidate, tokenizer, error, max_length)
                rejected_output.write(
                    json.dumps(rejected, ensure_ascii=False, sort_keys=True) + "\n"
                )
                rejected_count += 1
                rejected_reasons[error.code] += 1
                continue
            accepted_output.write(
                json.dumps(accepted, ensure_ascii=False, sort_keys=True) + "\n"
            )
            accepted_count += 1
            accepted_truncated_count += int(accepted["truncated"])
            raw_tokens += accepted["raw_token_count"]
            input_tokens += accepted["input_token_count"]
            supervised_tokens += accepted["supervised_token_count"]
            if first_audit is None:
                first_audit = audit

    expected = source.get("candidate_count")
    if expected is not None and candidate_count != expected:
        raise DataPreparationError(
            f"candidate count mismatch for {source['slice']}: "
            f"expected {expected}, got {candidate_count}"
        )
    if candidate_count != accepted_count + rejected_count:
        raise DataPreparationError(f"unaccounted candidates for {source['slice']}")
    summary = {
        "slice": source["slice"],
        "candidate_count": candidate_count,
        "accepted_count": accepted_count,
        "accepted_truncated_count": accepted_truncated_count,
        "rejected_count": rejected_count,
        "acceptance_rate": accepted_count / candidate_count,
        "rejected_by_reason": dict(sorted(rejected_reasons.items())),
        "accepted_token_totals": {
            "raw": raw_tokens,
            "input": input_tokens,
            "supervised": supervised_tokens,
        },
        "accepted_file": accepted_path.name,
        "accepted_sha256": _sha256_file(accepted_path),
        "rejected_file": rejected_path.name,
        "rejected_sha256": _sha256_file(rejected_path),
    }
    return summary, first_audit


def preprocess_all(
    config: dict[str, Any],
    candidate_dir: Path,
    output_dir: Path,
    tokenizer_cache_dir: Path,
    *,
    local_files_only: bool,
) -> None:
    tokenizer, tokenizer_metadata = load_frozen_tokenizer(
        config,
        tokenizer_cache_dir,
        local_files_only=local_files_only,
    )
    contract = load_day08_contract()
    max_length = config["preprocessing"]["max_length"]
    sources = [
        {**source, "candidate_count": config["candidate_samples_per_slice"]}
        for source in config["sources"]
    ]
    summaries = []
    boundary_trace = None
    for source in sources:
        summary, first_audit = preprocess_source(
            source,
            candidate_dir,
            output_dir,
            tokenizer,
            contract,
            max_length,
        )
        summaries.append(summary)
        if boundary_trace is None and first_audit is not None:
            boundary_trace = first_audit
        print(
            f"{source['slice']}: candidates={summary['candidate_count']} "
            f"accepted={summary['accepted_count']} rejected={summary['rejected_count']}"
        )
    if boundary_trace is None:
        raise DataPreparationError("no accepted sample available for boundary trace")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "boundary-trace.json").open("w", encoding="utf-8") as handle:
        json.dump(boundary_trace, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    payload = {
        "contract_version": config["preprocessing"]["contract_version"],
        "tokenizer": tokenizer_metadata,
        "max_length": max_length,
        "token_count_definitions": config["token_count_definitions"],
        "sources": summaries,
        "totals": {
            "candidate_count": sum(item["candidate_count"] for item in summaries),
            "accepted_count": sum(item["accepted_count"] for item in summaries),
            "rejected_count": sum(item["rejected_count"] for item in summaries),
        },
        "boundary_trace_file": "boundary-trace.json",
        "boundary_trace_sha256": _sha256_file(output_dir / "boundary-trace.json"),
    }
    with (output_dir / "preprocess-summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="download pinned raw snapshots")
    download_parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    download_parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="download only this slice; may be repeated",
    )

    select_parser = subparsers.add_parser("select", help="build deterministic candidate JSONL")
    select_parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    select_parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)

    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="apply the frozen Day 08 contract to candidate JSONL",
    )
    preprocess_parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_WORK)
    preprocess_parser.add_argument(
        "--output-dir",
        type=Path,
        help="defaults to tmp/day09-work/step3-max{configured max_length}",
    )
    preprocess_parser.add_argument(
        "--tokenizer-cache-dir",
        type=Path,
        default=DEFAULT_TOKENIZER_CACHE,
    )
    preprocess_parser.add_argument("--local-files-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "download":
        download_all(config, args.cache_dir, set(args.source))
    elif args.command == "select":
        select_all(config, args.cache_dir, args.work_dir)
    elif args.command == "preprocess":
        output_dir = args.output_dir or (
            DEFAULT_WORK / f"step3-max{config['preprocessing']['max_length']}"
        )
        preprocess_all(
            config,
            args.candidate_dir,
            output_dir,
            args.tokenizer_cache_dir,
            local_files_only=args.local_files_only,
        )


if __name__ == "__main__":
    main()
