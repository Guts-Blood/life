#!/usr/bin/env python3
"""Pure-standard-library contracts for Day 22 MBPP preference pairs.

The contract deliberately separates a CPU smoke candidate from a final DPO
record.  A smoke candidate can prove source and pair identities before the
Qwen3.5 processor is available; only a final record may claim ``accepted``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence


PAIR_SCHEMA_NAME = "day22.coding_preference_pair"
PAIR_SCHEMA_VERSION = 1

MBPP_SOURCE = "google-research-datasets/mbpp"
MBPP_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
MBPP_LICENSE = "CC-BY-4.0"
MBPP_TRAIN_FILES = {
    "full": {
        "source_file": "full/train-00000-of-00001.parquet",
        "source_file_sha256": (
            "09d125ca31edacb7800be8c67c45abff"
            "618faf0214ff551291817d06bdb914ae"
        ),
    },
    "sanitized": {
        "source_file": "sanitized/train-00000-of-00001.parquet",
        "source_file_sha256": (
            "d95f8ad6d2fff08fe4826122d6e3e31"
            "f75716825d0c5c340d297aca5e9e0de0e"
        ),
    },
}

SPLITS = ("train", "dev", "heldout")
FAMILY_KEY_NAMES = ("problem", "prompt", "test", "source")
EXECUTION_STATUSES = frozenset(
    {"pass", "wrong_answer", "syntax_error", "runtime_error", "timeout", "infra_error"}
)
SMOKE_PAIR_STATUSES = frozenset({"candidate", "blocked_processor_audit"})
FINAL_PAIR_STATUS = "accepted"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class Day22ContractError(ValueError):
    """A Day 22 schema, evidence, or split invariant failed."""


def canonical_json(value: Any) -> bytes:
    """Return the single canonical JSON representation used by Day 22 hashes."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def text_sha256(value: str) -> str:
    if not isinstance(value, str):
        raise Day22ContractError("text_sha256 input must be text")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def seal_pair(pair: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with its content-bound ``pair_sha256`` populated."""
    sealed = dict(pair)
    sealed.pop("pair_sha256", None)
    sealed["pair_sha256"] = object_sha256(sealed)
    return sealed


def stable_family_split(
    family_key: str,
    *,
    seed: str = "day22-mbpp-family-v1",
    ratios: Sequence[int] = (80, 10, 10),
) -> str:
    """Assign one family deterministically without depending on input row order."""
    key = _require_text(family_key, "family_key")
    split_seed = _require_text(seed, "split seed")
    if (
        len(ratios) != len(SPLITS)
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in ratios
        )
    ):
        raise Day22ContractError("split ratios must be three positive integers")
    bucket = int(text_sha256(f"{split_seed}\0{key}"), 16) % sum(ratios)
    boundary = 0
    for split, ratio in zip(SPLITS, ratios):
        boundary += ratio
        if bucket < boundary:
            return split
    raise AssertionError("unreachable split bucket")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Day22ContractError(message)


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Day22ContractError(f"{label} must be an object")
    return value


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Day22ContractError(f"{label} must be non-empty text")
    if "\x00" in value:
        raise Day22ContractError(f"{label} contains a NUL byte")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Day22ContractError(f"{label} must be a lowercase bare SHA-256")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise Day22ContractError(f"{label} must be an integer >= {minimum}")
    return value


def _verify_self_hash(value: Mapping[str, Any], field: str, label: str) -> None:
    expected = _require_sha256(value.get(field), f"{label}.{field}")
    actual = object_sha256({key: item for key, item in value.items() if key != field})
    _require(expected == actual, f"{label}.{field} does not bind its contents")


def _validate_hashed_text(value: Any, label: str) -> tuple[str, str]:
    record = _require_mapping(value, label)
    text = _require_text(record.get("text"), f"{label}.text")
    digest = _require_sha256(record.get("sha256"), f"{label}.sha256")
    _require(digest == text_sha256(text), f"{label}.sha256 does not match text")
    return text, digest


def _response_ast_sha256(response: str, code_prefix: str) -> str:
    try:
        body_wrapper = ast.parse(f"def __day22_response__():\n{response}\n")
        combined = ast.parse(f"{code_prefix}\n{response}\n")
        compile(combined, "<day22-pair>", "exec")
    except (IndentationError, SyntaxError) as error:
        raise Day22ContractError(
            f"response is not a valid code continuation: {error.msg}"
        ) from error
    function = body_wrapper.body[0]
    if not isinstance(function, ast.FunctionDef) or not function.body:
        raise Day22ContractError("response continuation has no executable body")
    body = ast.Module(body=function.body, type_ignores=[])
    identity = ast.dump(body, annotate_fields=True, include_attributes=False)
    return text_sha256(identity)


def _validate_response(value: Any, label: str, code_prefix: str) -> tuple[str, str, str]:
    record = _require_mapping(value, label)
    text, digest = _validate_hashed_text(record, label)
    _require("\r" not in text, f"{label}.text must use LF newlines")
    _require("```" not in text, f"{label}.text contains a Markdown fence")
    ast_digest = _response_ast_sha256(text, code_prefix)
    stored_ast_digest = _require_sha256(record.get("ast_sha256"), f"{label}.ast_sha256")
    _require(stored_ast_digest == ast_digest, f"{label}.ast_sha256 does not match code")
    return text, digest, ast_digest


def _validate_source(pair: Mapping[str, Any], family_keys: Mapping[str, Any]) -> None:
    source = _require_mapping(pair.get("source"), "source")
    _require(source.get("dataset") == MBPP_SOURCE, "source.dataset must be pinned MBPP")
    _require(source.get("revision") == MBPP_REVISION, "source.revision drifted")
    _require(source.get("license") == MBPP_LICENSE, "source.license drifted")
    _require(source.get("split") == "train", "only the upstream MBPP train split is allowed")
    variant = source.get("variant")
    _require(variant in MBPP_TRAIN_FILES, "source.variant must be full or sanitized")
    expected = MBPP_TRAIN_FILES[str(variant)]
    _require(source.get("source_file") == expected["source_file"], "source.source_file drifted")
    source_file_sha256 = _require_sha256(
        source.get("source_file_sha256"), "source.source_file_sha256"
    )
    _require(
        source_file_sha256 == expected["source_file_sha256"],
        "source.source_file_sha256 drifted",
    )
    _require_int(source.get("source_row_index"), "source.source_row_index")
    _require_sha256(source.get("source_content_sha256"), "source.source_content_sha256")
    _require_text(source.get("day20_sample_id"), "source.day20_sample_id")
    # ``day20_prompt_sha256`` is the upstream messages[:-1] object identity;
    # ``prompt_text_sha256`` is the original user-message text identity.  The
    # latter may differ from pair.prompt after an audited indentation migration.
    _require_sha256(source.get("day20_prompt_sha256"), "source.day20_prompt_sha256")
    _require_sha256(source.get("prompt_text_sha256"), "source.prompt_text_sha256")

    task_id = _require_int(pair.get("task_id"), "task_id")
    if "task_id" in source:
        _require(source.get("task_id") == task_id, "source.task_id differs from pair task_id")
    native_family = f"mbpp:task:{task_id}"
    _require(
        family_keys.get("problem") == native_family,
        "problem family is not native MBPP task ID",
    )
    _require(family_keys.get("source") == native_family, "source family is not task-scoped")


def _validate_tests(value: Any) -> tuple[Mapping[str, Any], str]:
    tests = _require_mapping(value, "tests")
    normalized: dict[str, Any] = {}
    for key in ("test_list", "challenge_test_list"):
        items = tests.get(key)
        _require(isinstance(items, list), f"tests.{key} must be a list")
        _require(
            all(isinstance(item, str) and item.strip() for item in items),
            f"tests.{key} contains an empty or non-text test",
        )
        normalized[key] = [item.strip() for item in items]
    _require(bool(normalized["test_list"]), "tests.test_list must not be empty")
    setup = tests.get("test_setup_code")
    _require(isinstance(setup, str), "tests.test_setup_code must be text")
    normalized["test_setup_code"] = setup.strip()
    digest = _require_sha256(tests.get("sha256"), "tests.sha256")
    _require(digest == object_sha256(normalized), "tests.sha256 does not bind canonical tests")
    _require_int(tests.get("count"), "tests.count")
    _require_int(tests.get("challenge_count"), "tests.challenge_count")
    _require(tests.get("count") == len(normalized["test_list"]), "tests.count drifted")
    _require(
        tests.get("challenge_count") == len(normalized["challenge_test_list"]),
        "tests.challenge_count drifted",
    )
    return tests, digest


def _validate_run(
    value: Any,
    *,
    branch: str,
    attempt: int,
    response_sha256: str,
    test_sha256: str,
    sandbox_digest: str,
) -> tuple[str, str]:
    label = f"execution.{branch}.runs[{attempt - 1}]"
    run = _require_mapping(value, label)
    _require(run.get("attempt") == attempt, f"{label}.attempt must be {attempt}")
    run_id = _require_text(run.get("run_id"), f"{label}.run_id")
    status = run.get("status")
    _require(status in EXECUTION_STATUSES, f"{label}.status is invalid")
    _require(run.get("response_sha256") == response_sha256, f"{label} response drifted")
    _require(run.get("test_sha256") == test_sha256, f"{label} tests drifted")
    _require(run.get("sandbox_digest") == sandbox_digest, f"{label} sandbox drifted")
    exit_code = run.get("exit_code")
    _require(
        exit_code is None or (isinstance(exit_code, int) and not isinstance(exit_code, bool)),
        f"{label}.exit_code must be an integer or null",
    )
    stdout = run.get("stdout")
    stderr = run.get("stderr")
    _require(isinstance(stdout, str), f"{label}.stdout must be text")
    _require(isinstance(stderr, str), f"{label}.stderr must be text")
    _require(run.get("stdout_sha256") == text_sha256(stdout), f"{label}.stdout hash drifted")
    _require(run.get("stderr_sha256") == text_sha256(stderr), f"{label}.stderr hash drifted")
    if status == "pass":
        _require(exit_code == 0, f"{label} pass must have exit_code 0")
    if status == "wrong_answer":
        _require(
            isinstance(exit_code, int) and not isinstance(exit_code, bool),
            f"{label} wrong_answer must have an integer exit code",
        )
        _require(
            run.get("error_type") == "assertion_error",
            f"{label} wrong_answer must be verifier-classified assertion_error",
        )
    _verify_self_hash(run, "run_sha256", label)
    return str(status), run_id


def _validate_execution(
    value: Any,
    *,
    chosen_sha256: str,
    rejected_sha256: str,
    test_sha256: str,
) -> None:
    execution = _require_mapping(value, "execution")
    _require(execution.get("test_sha256") == test_sha256, "execution tests drifted")
    sandbox_digest = _require_text(execution.get("sandbox_digest"), "execution.sandbox_digest")
    timeout = execution.get("timeout_seconds")
    _require(
        isinstance(timeout, (int, float))
        and not isinstance(timeout, bool)
        and timeout > 0,
        "execution.timeout_seconds must be positive",
    )
    verifier = _require_mapping(execution.get("verifier"), "execution.verifier")
    _require_text(verifier.get("name"), "execution.verifier.name")
    _require_text(verifier.get("version"), "execution.verifier.version")

    branch_statuses: dict[str, list[str]] = {}
    run_ids: list[str] = []
    for branch, response_digest in (
        ("chosen", chosen_sha256),
        ("rejected", rejected_sha256),
    ):
        branch_record = _require_mapping(execution.get(branch), f"execution.{branch}")
        runs = branch_record.get("runs")
        _require(isinstance(runs, list) and len(runs) == 2, f"execution.{branch} needs two runs")
        statuses: list[str] = []
        for attempt, run in enumerate(runs, 1):
            status, run_id = _validate_run(
                run,
                branch=branch,
                attempt=attempt,
                response_sha256=response_digest,
                test_sha256=test_sha256,
                sandbox_digest=sandbox_digest,
            )
            statuses.append(status)
            run_ids.append(run_id)
        branch_statuses[branch] = statuses

    _require(len(set(run_ids)) == 4, "execution runs must have four distinct run_id values")
    _require(branch_statuses["chosen"] == ["pass", "pass"], "chosen must pass twice")
    _require(
        branch_statuses["rejected"] == ["wrong_answer", "wrong_answer"],
        "rejected must fail twice as wrong_answer; timeout/runtime/infra are not preferences",
    )
    _verify_self_hash(execution, "evidence_sha256", "execution")


def _validate_blocked_processor_audit(value: Any) -> None:
    audit = _require_mapping(value, "processor_audit")
    _require(audit.get("status") == "blocked", "blocked pair needs processor_audit.status=blocked")
    _require_text(audit.get("reason"), "processor_audit.reason")


def _validate_processor_branch(value: Any, label: str) -> tuple[int, str]:
    branch = _require_mapping(value, label)
    _require(branch.get("status") == "pass", f"{label}.status must be pass")
    prompt_hash = _require_sha256(
        branch.get("prompt_prefix_token_ids_sha256"),
        f"{label}.prompt_prefix_token_ids_sha256",
    )
    _require_sha256(branch.get("response_token_ids_sha256"), f"{label}.response_token_ids_sha256")
    input_count = _require_int(
        branch.get("input_token_count"), f"{label}.input_token_count", minimum=1
    )
    response_count = _require_int(
        branch.get("response_token_count"), f"{label}.response_token_count", minimum=1
    )
    span = branch.get("response_span")
    _require(
        isinstance(span, list)
        and len(span) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in span),
        f"{label}.response_span must be [start, end]",
    )
    start, end = span
    # Qwen3.5's frozen Day 20 v3 target contract deliberately masks the
    # rendered trailing newline after ``<|im_end|>``.  The supervised response
    # span therefore need not consume every input token.
    _require(0 < start < end <= input_count, f"{label}.response_span is out of bounds")
    _require(end - start == response_count, f"{label}.response token count drifted")
    _require(branch.get("truncation") is False, f"{label} must not be truncated")
    _require(branch.get("response_only_mask") is True, f"{label} mask is not response-only")
    _require(branch.get("causal_shift_status") == "pass", f"{label} causal-shift audit failed")
    return start, prompt_hash


def _validate_final_processor_audit(value: Any, prompt_sha256: str) -> None:
    audit = _require_mapping(value, "processor_audit")
    _require(audit.get("status") == "pass", "final pair needs processor_audit.status=pass")
    _require(audit.get("retokenized_for_day22") is True, "pair was not retokenized for Day 22")
    _require(audit.get("legacy_token_ids_reused") is False, "legacy token IDs were reused")
    _require(audit.get("prompt_text_sha256") == prompt_sha256, "processor prompt text drifted")
    for key in (
        "model_key",
        "model_revision",
        "processor_revision",
        "tokenizer_revision",
        "template_revision",
    ):
        _require_text(audit.get(key), f"processor_audit.{key}")
    for key in (
        "processor_sha256",
        "tokenizer_sha256",
        "template_sha256",
        "rendered_prompt_sha256",
    ):
        _require_sha256(audit.get(key), f"processor_audit.{key}")
    chosen_start, chosen_prompt_hash = _validate_processor_branch(
        audit.get("chosen"), "processor_audit.chosen"
    )
    rejected_start, rejected_prompt_hash = _validate_processor_branch(
        audit.get("rejected"), "processor_audit.rejected"
    )
    _require(chosen_start == rejected_start, "chosen/rejected response boundaries differ")
    _require(
        chosen_prompt_hash == rejected_prompt_hash,
        "chosen/rejected prompt token prefixes differ",
    )
    _verify_self_hash(audit, "audit_sha256", "processor_audit")


def validate_pair(pair: Mapping[str, Any], *, mode: str = "final") -> None:
    """Validate one preference pair in ``smoke`` or ``final`` lifecycle mode."""
    pair = _require_mapping(pair, "pair")
    _require(mode in {"smoke", "final"}, "mode must be 'smoke' or 'final'")
    _require(pair.get("schema_name") == PAIR_SCHEMA_NAME, "pair schema_name drifted")
    _require(pair.get("schema_version") == PAIR_SCHEMA_VERSION, "pair schema_version drifted")
    _require_text(pair.get("pair_id"), "pair_id")
    split = pair.get("split")
    _require(split in SPLITS, "split must be train, dev, or heldout")
    _require(pair.get("language") == "python", "language must be python")

    family_keys = _require_mapping(pair.get("family_keys"), "family_keys")
    for key in FAMILY_KEY_NAMES:
        _require_text(family_keys.get(key), f"family_keys.{key}")
    _validate_source(pair, family_keys)
    problem_text, problem_sha256 = _validate_hashed_text(pair.get("problem"), "problem")
    del problem_text
    prompt_text, prompt_sha256 = _validate_hashed_text(pair.get("prompt"), "prompt")
    del prompt_text
    _require(family_keys.get("prompt") == f"sha256:{prompt_sha256}", "prompt family hash drifted")
    _require(pair.get("problem", {}).get("sha256") == problem_sha256, "problem hash drifted")

    code_prefix = _require_text(pair.get("code_prefix"), "code_prefix")
    _require_text(pair.get("entry_point"), "entry_point")
    chosen_text, chosen_sha256, chosen_ast = _validate_response(
        pair.get("chosen"), "chosen", code_prefix
    )
    rejected_text, rejected_sha256, rejected_ast = _validate_response(
        pair.get("rejected"), "rejected", code_prefix
    )
    _require(chosen_text != rejected_text, "chosen and rejected are identical")
    _require(chosen_sha256 != rejected_sha256, "chosen and rejected hashes are identical")
    _require(chosen_ast != rejected_ast, "chosen and rejected are AST-equivalent")

    creation = _require_mapping(pair.get("creation"), "creation")
    _require_text(creation.get("method"), "creation.method")
    _require_text(creation.get("version"), "creation.version")
    _, test_sha256 = _validate_tests(pair.get("tests"))
    _require(family_keys.get("test") == f"sha256:{test_sha256}", "test family hash drifted")

    status = pair.get("pair_status")
    if mode == "final":
        _require(status == FINAL_PAIR_STATUS, "final pair_status must be accepted")
        _validate_execution(
            pair.get("execution"),
            chosen_sha256=chosen_sha256,
            rejected_sha256=rejected_sha256,
            test_sha256=test_sha256,
        )
        _validate_final_processor_audit(pair.get("processor_audit"), prompt_sha256)
    else:
        _require(status in SMOKE_PAIR_STATUSES, "smoke pair_status is invalid")
        if status == "candidate":
            _require(
                "execution" not in pair,
                "unscored candidate must not carry execution evidence",
            )
            if "processor_audit" in pair:
                pending = _require_mapping(pair["processor_audit"], "processor_audit")
                _require(
                    pending.get("status") == "pending",
                    "candidate processor audit must be pending",
                )
        else:
            _validate_execution(
                pair.get("execution"),
                chosen_sha256=chosen_sha256,
                rejected_sha256=rejected_sha256,
                test_sha256=test_sha256,
            )
            _validate_blocked_processor_audit(pair.get("processor_audit"))
    _verify_self_hash(pair, "pair_sha256", "pair")


def validate_manifest(
    pairs: Iterable[Mapping[str, Any]], *, mode: str = "final"
) -> dict[str, Any]:
    """Validate pair uniqueness and problem/prompt/test/source split isolation."""
    rows = list(pairs)
    _require(bool(rows), "pair manifest must not be empty")
    pair_ids: set[str] = set()
    problem_families: set[str] = set()
    family_splits: dict[str, dict[str, str]] = {key: {} for key in FAMILY_KEY_NAMES}
    split_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    pair_hashes: list[str] = []

    for index, pair in enumerate(rows):
        try:
            validate_pair(pair, mode=mode)
        except Day22ContractError as error:
            raise Day22ContractError(f"pair manifest row {index}: {error}") from error
        pair_id = str(pair["pair_id"])
        _require(pair_id not in pair_ids, f"duplicate pair_id: {pair_id}")
        pair_ids.add(pair_id)
        split = str(pair["split"])
        families = pair["family_keys"]
        problem_family = str(families["problem"])
        _require(problem_family not in problem_families, f"multiple pairs for {problem_family}")
        problem_families.add(problem_family)
        for family_name in FAMILY_KEY_NAMES:
            family_key = str(families[family_name])
            prior = family_splits[family_name].get(family_key)
            _require(
                prior is None or prior == split,
                f"{family_name} family {family_key!r} leaks across {prior}/{split}",
            )
            family_splits[family_name][family_key] = split
        split_counts[split] += 1
        status_counts[str(pair["pair_status"])] += 1
        pair_hashes.append(str(pair["pair_sha256"]))

    return {
        "records": len(rows),
        "split_counts": {split: split_counts[split] for split in SPLITS},
        "pair_status_counts": dict(sorted(status_counts.items())),
        "ordered_pair_hashes_sha256": object_sha256(pair_hashes),
    }
