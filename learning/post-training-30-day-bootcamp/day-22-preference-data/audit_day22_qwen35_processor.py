#!/usr/bin/env python3
"""Retokenize Day 22 replay pairs with the frozen Qwen3.5 target template.

The module deliberately keeps ms-swift and transformers imports behind the CLI
runtime boundary.  Its encoding checks are pure Python so they can be tested on
machines that do not have the Qwen runtime installed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SCHEMA_NAME = "day22.qwen35_pair_processor_audit"
SCHEMA_VERSION = 1
CONTRACT_SCHEMA_NAME = "day22.qwen35_processor_contract"
CONTRACT_SCHEMA_VERSION = 1
MODEL_KEY = "Qwen/Qwen3.5-4B-Base"
MODEL_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = (
    REPO_ROOT.parent
    / f"{REPO_ROOT.name}-checkpoints"
    / "Qwen--Qwen3.5-4B-Base"
    / MODEL_REVISION
)
TARGET_TEMPLATE_ALIAS = "day20_qwen3_5_target_v3"
TARGET_TEMPLATE_REVISION = "day20.qwen35_code_boundary_target_v3"
TARGET_TEMPLATE_SHA256 = (
    "7f835342861cc9aabfd6299228033df80437b183227e04db311450437cd07ded"
)
INDENT_TEXT = "    "
INDENT_TOKEN_ID = 257
IM_END_TOKEN_ID = 248046
NEWLINE_TOKEN_ID = 198

# Only the files needed to construct the processor/template are in scope.  The
# model shards are intentionally neither read nor hashed by this audit.
EXPECTED_SNAPSHOT_SHA256 = {
    "config.json": "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
    "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    "video_preprocessor_config.json": "d039cd7d88b3502a99edd455496b75b44ecf7dd3b3669748bafac37e6cecc085",
    "tokenizer.json": "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927",
    "tokenizer_config.json": "3891e840d7dc5fca0af33d3a25083a735e36fe06214e3f707024820cb6b9f89c",
    "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
    "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
}
PROCESSOR_FILES = (
    "config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)


class Day22ProcessorAuditError(ValueError):
    """A replay, snapshot, encoding, or evidence invariant failed."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def text_sha256(value: str) -> str:
    if not isinstance(value, str):
        raise Day22ProcessorAuditError("text hash input must be text")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise Day22ProcessorAuditError(f"cannot hash file: {path}") from error
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Day22ProcessorAuditError(message)


def _require_text(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} must be non-empty text")
    return str(value)


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _integral_tokens(value: Any, label: str) -> list[int]:
    _require(
        not isinstance(value, (str, bytes, bytearray)),
        f"{label} must be an integer token sequence",
    )
    try:
        tokens = [int(item) for item in value]
    except (TypeError, ValueError) as error:
        raise Day22ProcessorAuditError(
            f"{label} must be an integer token sequence"
        ) from error
    _require(bool(tokens), f"{label} must not be empty")
    return tokens


def _validate_four_space_target(text: str, label: str) -> None:
    _require(bool(text), f"{label} response must not be empty")
    _require(text.startswith(INDENT_TEXT), f"{label} must start with four ASCII spaces")
    first_line = text.split("\n", 1)[0]
    leading = first_line[: len(first_line) - len(first_line.lstrip(" \t"))]
    _require(
        leading == INDENT_TEXT and "\t" not in leading,
        f"{label} first line must have exactly four-space indentation",
    )


def _decode_target(
    decode_supervised: Callable[[Sequence[int]], str], labels: Sequence[int]
) -> str:
    supervised = [int(label) for label in labels if int(label) != -100]
    try:
        decoded = decode_supervised(supervised)
    except Exception as error:
        raise Day22ProcessorAuditError(
            f"cannot decode supervised target: {type(error).__name__}: {error}"
        ) from error
    _require(isinstance(decoded, str), "decoded supervised target must be text")
    return decoded


def analyze_branch_encoding(
    encoded: Mapping[str, Any],
    *,
    response_text: str,
    decode_supervised: Callable[[Sequence[int]], str],
    max_length: int,
    label: str,
) -> tuple[dict[str, Any], tuple[int, ...]]:
    """Validate one encoded branch and return sealed evidence plus prompt IDs."""
    _validate_four_space_target(response_text, label)
    input_ids = _integral_tokens(encoded.get("input_ids"), f"{label}.input_ids")
    labels = _integral_tokens(encoded.get("labels"), f"{label}.labels")
    _require(len(input_ids) == len(labels), f"{label} input_ids/labels are misaligned")
    _require(len(input_ids) <= max_length, f"{label} encoding exceeds max_length")

    positions = [index for index, token in enumerate(labels) if token != -100]
    _require(bool(positions), f"{label} has no supervised response labels")
    start = positions[0]
    end = positions[-1] + 1
    _require(start > 0, f"{label} has no causal prompt token before the response")
    _require(
        positions == list(range(start, end)),
        f"{label} supervised response labels are not contiguous",
    )
    _require(
        all(token == -100 for token in labels[:start])
        and all(token == -100 for token in labels[end:]),
        f"{label} mask is not response-only",
    )
    _require(
        labels[start:end] == input_ids[start:end],
        f"{label} labels do not align with causal-LM target tokens",
    )
    _require(
        input_ids[start] == INDENT_TOKEN_ID and labels[start] == INDENT_TOKEN_ID,
        f"{label} four-space boundary token is not supervised",
    )
    _require(
        input_ids[-2:] == [IM_END_TOKEN_ID, NEWLINE_TOKEN_ID]
        and labels[-2:] == [IM_END_TOKEN_ID, -100],
        f"{label} does not use the balanced v3 ChatML tail mask",
    )
    _require(
        end == len(input_ids) - 1,
        f"{label} must leave exactly the trailing ChatML newline masked",
    )
    decoded = _decode_target(decode_supervised, labels)
    _require(
        decoded == response_text,
        f"{label} supervised labels decode to {decoded!r}, expected {response_text!r}",
    )

    prompt_prefix = tuple(input_ids[:start])
    supervised_ids = input_ids[start:end]
    evidence: dict[str, Any] = {
        "status": "pass",
        "prompt_prefix_token_ids_sha256": object_sha256(list(prompt_prefix)),
        "response_token_ids_sha256": object_sha256(supervised_ids),
        "input_ids_sha256": object_sha256(input_ids),
        "labels_sha256": object_sha256(labels),
        "response_text_sha256": text_sha256(response_text),
        "input_token_count": len(input_ids),
        "response_token_count": len(supervised_ids),
        "response_span": [start, end],
        "causal_shift_prediction_span": [start - 1, end - 1],
        "trailing_masked_token_count": len(input_ids) - end,
        "truncation": False,
        "response_only_mask": True,
        "four_space_boundary": True,
        "causal_shift_status": "pass",
    }
    evidence["branch_sha256"] = object_sha256(evidence)
    return evidence, prompt_prefix


def _validated_replay_text(row: Mapping[str, Any], key: str) -> str:
    record = _require_mapping(row.get(key), key)
    text = _require_text(record.get("text"), f"{key}.text")
    _require(record.get("sha256") == text_sha256(text), f"{key}.sha256 drifted")
    return text


def _validated_prompt(row: Mapping[str, Any]) -> str:
    prompt = _require_mapping(row.get("prompt"), "prompt")
    text = _require_text(prompt.get("text"), "prompt.text")
    _require(prompt.get("sha256") == text_sha256(text), "prompt.sha256 drifted")
    return text


def _validate_replay_self_hash(row: Mapping[str, Any]) -> str:
    digest = _require_text(row.get("row_sha256"), "row_sha256")
    payload = {key: value for key, value in row.items() if key != "row_sha256"}
    _require(digest == object_sha256(payload), "replay row self-hash drifted")
    return digest


def audit_replay_pair(
    row: Mapping[str, Any],
    *,
    encode_messages: Callable[[Sequence[Mapping[str, str]]], Mapping[str, Any]],
    decode_supervised: Callable[[Sequence[int]], str],
    processor_contract: Mapping[str, Any],
    max_length: int,
) -> dict[str, Any]:
    """Retokenize and audit one replay pair using injected runtime functions."""
    _require(row.get("schema_name") == "day22.mbpp_replay_pair", "unexpected replay schema")
    _require(row.get("schema_version") == 1, "unexpected replay schema version")
    replay_sha256 = _validate_replay_self_hash(row)
    pair_id = _require_text(row.get("pair_id"), "pair_id")
    family_id = _require_text(row.get("family_id"), "family_id")
    prompt_text = _validated_prompt(row)
    chosen_text = _validated_replay_text(row, "chosen")
    rejected_text = _validated_replay_text(row, "rejected")
    _require(chosen_text != rejected_text, f"{pair_id}: chosen/rejected text is identical")

    branch_results: dict[str, dict[str, Any]] = {}
    prefixes: dict[str, tuple[int, ...]] = {}
    for branch, response_text in (("chosen", chosen_text), ("rejected", rejected_text)):
        messages = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": response_text},
        ]
        try:
            encoded = encode_messages(copy.deepcopy(messages))
        except Exception as error:
            if isinstance(error, Day22ProcessorAuditError):
                raise
            raise Day22ProcessorAuditError(
                f"{pair_id}:{branch} encode failed: {type(error).__name__}: {error}"
            ) from error
        _require(isinstance(encoded, Mapping), f"{pair_id}:{branch} encoding is not an object")
        evidence, prefix = analyze_branch_encoding(
            encoded,
            response_text=response_text,
            decode_supervised=decode_supervised,
            max_length=max_length,
            label=f"{pair_id}:{branch}",
        )
        branch_results[branch] = evidence
        prefixes[branch] = prefix

    _require(
        prefixes["chosen"] == prefixes["rejected"],
        f"{pair_id}: chosen/rejected rendered prompt token prefixes differ",
    )
    _require(
        branch_results["chosen"]["response_span"][0]
        == branch_results["rejected"]["response_span"][0],
        f"{pair_id}: chosen/rejected response boundaries differ",
    )

    contract_sha256 = _require_text(
        processor_contract.get("contract_sha256"), "processor_contract.contract_sha256"
    )
    contract_payload = {
        key: value for key, value in processor_contract.items() if key != "contract_sha256"
    }
    _require(
        contract_sha256 == object_sha256(contract_payload),
        "processor contract self-hash drifted",
    )
    audit: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "pair_id": pair_id,
        "task_id": str(row.get("task_id")),
        "family_id": family_id,
        "chosen_candidate_id": _require_text(
            row.get("chosen", {}).get("candidate_id"), "chosen.candidate_id"
        ),
        "rejected_candidate_id": _require_text(
            row.get("rejected", {}).get("candidate_id"), "rejected.candidate_id"
        ),
        "replay_row_sha256": replay_sha256,
        "retokenized_for_day22": True,
        "legacy_token_ids_reused": False,
        "prompt_text_sha256": text_sha256(prompt_text),
        "chosen_response_sha256": text_sha256(chosen_text),
        "rejected_response_sha256": text_sha256(rejected_text),
        "model_key": processor_contract["model_key"],
        "model_revision": processor_contract["model_revision"],
        "processor_revision": processor_contract["processor_revision"],
        "tokenizer_revision": processor_contract["tokenizer_revision"],
        "template_revision": processor_contract["template_revision"],
        "processor_sha256": processor_contract["processor_sha256"],
        "tokenizer_sha256": processor_contract["tokenizer_sha256"],
        "template_sha256": processor_contract["template_sha256"],
        "processor_contract_sha256": contract_sha256,
        "rendered_prompt_sha256": branch_results["chosen"][
            "prompt_prefix_token_ids_sha256"
        ],
        "chosen": branch_results["chosen"],
        "rejected": branch_results["rejected"],
    }
    audit["audit_sha256"] = object_sha256(audit)
    return audit


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def build_processor_contract(
    model_path: Path,
    *,
    day20_module_path: Path,
    day20_contract_sha256: str,
    max_length: int,
    expected_snapshot_sha256: Mapping[str, str] = EXPECTED_SNAPSHOT_SHA256,
) -> dict[str, Any]:
    """Verify the pinned snapshot and construct its deterministic contract."""
    resolved = model_path.expanduser().resolve()
    _require(resolved.is_dir(), f"model snapshot is missing: {resolved}")
    _require(resolved.name == MODEL_REVISION, "model snapshot revision directory drifted")
    _require(max_length > 0, "max_length must be positive")
    _require(
        day20_contract_sha256 == TARGET_TEMPLATE_SHA256,
        "Day 20 target template contract drifted",
    )

    files: dict[str, dict[str, Any]] = {}
    for relative_path, expected_digest in sorted(expected_snapshot_sha256.items()):
        path = resolved / relative_path
        _require(path.is_file(), f"snapshot file is missing: {path}")
        actual_digest = file_sha256(path)
        _require(
            actual_digest == expected_digest,
            f"snapshot file hash drifted: {relative_path}",
        )
        files[relative_path] = {
            "sha256": actual_digest,
            "size_bytes": path.stat().st_size,
        }

    processor_files = {name: files[name] for name in PROCESSOR_FILES}
    tokenizer_files = {name: files[name] for name in TOKENIZER_FILES}
    resolved_day20_module = day20_module_path.resolve()
    _require(resolved_day20_module.is_file(), "Day 20 target encoding module is missing")
    contract: dict[str, Any] = {
        "schema_name": CONTRACT_SCHEMA_NAME,
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "status": "frozen",
        "scope": "processor_tokenizer_template_only_no_model_weights",
        "model_key": MODEL_KEY,
        "model_revision": MODEL_REVISION,
        # Keep the committed contract portable; the verified file identities,
        # model key, and revision bind the snapshot without a user-specific
        # absolute cache path.
        "snapshot_path": f"{resolved.parent.name}/{resolved.name}",
        "snapshot_files": files,
        "processor_revision": f"{MODEL_KEY}@{MODEL_REVISION}",
        "tokenizer_revision": f"{MODEL_KEY}@{MODEL_REVISION}",
        "template_revision": TARGET_TEMPLATE_REVISION,
        "processor_sha256": object_sha256(processor_files),
        "tokenizer_sha256": object_sha256(tokenizer_files),
        "template_sha256": day20_contract_sha256,
        "day20_target_encoding_module_sha256": file_sha256(resolved_day20_module),
        "template_alias": TARGET_TEMPLATE_ALIAS,
        "max_length": max_length,
        "truncation_strategy": "raise",
        "runtime": {
            "python": platform.python_version(),
            "ms_swift": _package_version("ms-swift"),
            "transformers": _package_version("transformers"),
            "huggingface_hub": _package_version("huggingface-hub"),
        },
    }
    contract["contract_sha256"] = object_sha256(contract)
    return contract


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise Day22ProcessorAuditError(f"blank JSONL row: {path}:{line_number}")
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise Day22ProcessorAuditError(
                        f"invalid JSONL row: {path}:{line_number}"
                    ) from error
                _require(isinstance(value, dict), f"JSONL row is not an object: {path}:{line_number}")
                rows.append(value)
    except OSError as error:
        raise Day22ProcessorAuditError(f"cannot read JSONL: {path}") from error
    _require(bool(rows), f"JSONL is empty: {path}")
    return rows


def _preflight_outputs(paths: Sequence[Path], *, overwrite: bool) -> None:
    for path in paths:
        _require(path.parent.is_dir(), f"output directory is missing: {path.parent}")
        if path.exists() and not overwrite:
            raise Day22ProcessorAuditError(f"refusing to overwrite: {path}")
        _require(not path.exists() or path.is_file(), f"output path is not a file: {path}")


def write_json(path: Path, value: Mapping[str, Any], *, overwrite: bool) -> None:
    mode = "w" if overwrite else "x"
    try:
        with path.open(mode, encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except OSError as error:
        raise Day22ProcessorAuditError(f"cannot write JSON: {path}") from error


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]], *, overwrite: bool) -> None:
    mode = "w" if overwrite else "x"
    try:
        with path.open(mode, encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(canonical_json(row).decode("utf-8"))
                handle.write("\n")
    except OSError as error:
        raise Day22ProcessorAuditError(f"cannot write JSONL: {path}") from error


def _load_day20_module() -> tuple[Any, Path]:
    module_dir = Path(__file__).resolve().parent.parent / "day-20-qwen35-balanced-lora-sft"
    module_path = module_dir / "day20_target_encoding_v3.py"
    _require(module_path.is_file(), f"Day 20 target encoding module is missing: {module_path}")
    sys.path.insert(0, str(module_dir))
    try:
        import day20_target_encoding_v3 as target_v3
    except ImportError as error:
        raise Day22ProcessorAuditError("cannot import Day 20 target encoding module") from error
    _require(
        Path(target_v3.__file__).resolve() == module_path.resolve(),
        "imported Day 20 target encoding module from an unexpected path",
    )
    return target_v3, module_path


def run_audit(
    *,
    input_path: Path,
    output_path: Path,
    contract_output_path: Path,
    model_path: Path,
    max_length: int,
    overwrite: bool,
) -> dict[str, Any]:
    rows = load_jsonl(input_path.resolve())
    pair_ids = [_require_text(row.get("pair_id"), "pair_id") for row in rows]
    _require(len(pair_ids) == len(set(pair_ids)), "replay pair_id values are not unique")
    target_v3, day20_module_path = _load_day20_module()
    contract = build_processor_contract(
        model_path,
        day20_module_path=day20_module_path,
        day20_contract_sha256=target_v3.TARGET_ENCODING_CONTRACT_SHA256,
        max_length=max_length,
    )
    template = target_v3.build_target_template_v3(model_path, max_length=max_length)

    def encode_messages(messages: Sequence[Mapping[str, str]]) -> Mapping[str, Any]:
        encoded = template.encode(
            {"messages": copy.deepcopy(list(messages))}, return_length=True
        )
        _require(isinstance(encoded, Mapping), "target template returned a non-object")
        return encoded

    def decode_supervised(token_ids: Sequence[int]) -> str:
        try:
            return template.tokenizer.decode(
                list(token_ids),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return template.tokenizer.decode(list(token_ids), skip_special_tokens=True)

    audits = [
        audit_replay_pair(
            row,
            encode_messages=encode_messages,
            decode_supervised=decode_supervised,
            processor_contract=contract,
            max_length=max_length,
        )
        for row in rows
    ]
    resolved_output = output_path.resolve()
    resolved_contract_output = contract_output_path.resolve()
    _require(resolved_output != resolved_contract_output, "output paths must differ")
    _preflight_outputs(
        (resolved_output, resolved_contract_output), overwrite=overwrite
    )
    write_jsonl(resolved_output, audits, overwrite=overwrite)
    write_json(resolved_contract_output, contract, overwrite=overwrite)
    return {
        "status": "pass",
        "records": len(audits),
        "input": str(input_path.resolve()),
        "output": str(resolved_output),
        "output_sha256": file_sha256(resolved_output),
        "contract_output": str(resolved_contract_output),
        "contract_output_sha256": file_sha256(resolved_contract_output),
        "processor_contract_sha256": contract["contract_sha256"],
        "ordered_audit_sha256": object_sha256(
            [audit["audit_sha256"] for audit in audits]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--contract-output", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-length", type=int, default=2304)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        result = run_audit(
            input_path=args.input,
            output_path=args.output,
            contract_output_path=args.contract_output,
            model_path=args.model,
            max_length=args.max_length,
            overwrite=args.overwrite,
        )
    except Day22ProcessorAuditError as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
