#!/usr/bin/env python3
"""Pinned, non-executable source adapter for the Day 20 training pools.

``prepare_day20.py`` calls :func:`fetch_sources` through
``--fetch-adapter day20_source_adapter:fetch_sources``.  This module downloads
only data files from exact Hugging Face revisions (or reads explicitly staged
local JSON/JSONL/Parquet files), converts task-native targets, and audits every
accepted row with the real Qwen3.5 ms-swift train template.

It deliberately does not import or call ``datasets.load_dataset``.  Dataset
repository Python files are neither downloaded nor executed.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from day20_contract import (
    MAIN_TOKENS_BY_FORMAT,
    PROBE_TOKENS_BY_FORMAT,
    QWEN35_TEMPLATE_CONTRACT,
    Day20ContractError,
    exact_subset_indices,
    normalize_assistant_target,
    normalize_and_validate_record,
    object_sha256,
    text_sha256,
)


class Day20SourceAdapterError(ValueError):
    """Pinned source resolution, transformation, or capacity failed."""


class SourceRecordRejected(ValueError):
    """One source row is unsuitable under the preregistered filters."""


@dataclass(frozen=True)
class SourceSpec:
    key: str
    source: str
    revision: str
    license: str
    adapter: str
    allow_patterns: tuple[str, ...]
    data_patterns: tuple[str, ...]
    default_split: str


@dataclass(frozen=True)
class SourceFile:
    spec: SourceSpec
    path: Path
    split: str
    variant: str
    relative_name: str


SOURCE_SPECS: dict[str, SourceSpec] = {
    "mmlu": SourceSpec(
        key="mmlu",
        source="cais/mmlu",
        revision="c30699e8356da336a370243923dbaf21066bb9fe",
        license="MIT",
        adapter="mmlu_auxiliary_train_mcq_v1",
        allow_patterns=("all/auxiliary_train-*.parquet",),
        data_patterns=("all/auxiliary_train-*.parquet",),
        default_split="auxiliary_train",
    ),
    "tulu": SourceSpec(
        key="tulu",
        source="allenai/tulu-3-sft-personas-instruction-following",
        revision="fe0c7d350c9b4542b8d829a6f1daa1c259f0ba0e",
        license="ODC-BY-1.0",
        adapter="tulu_general_replay_filtered_v1",
        allow_patterns=("data/train-*.parquet",),
        data_patterns=("data/train-*.parquet",),
        default_split="train",
    ),
    "gsm8k": SourceSpec(
        key="gsm8k",
        source="openai/gsm8k",
        revision="740312add88f781978c0658806c59bc2815b9866",
        license="MIT",
        adapter="gsm8k_reasoning_final_v1",
        allow_patterns=("main/train-*.parquet",),
        data_patterns=("main/train-*.parquet",),
        default_split="train",
    ),
    "tatqa": SourceSpec(
        key="tatqa",
        source="next-tat/TAT-QA",
        revision="c96247f5077eac447f63527fd3dcfdc58bb56d6a",
        license="CC-BY-4.0",
        adapter="tatqa_train_scalar_value_scale_v1",
        allow_patterns=("tatqa_dataset_train.json",),
        data_patterns=("tatqa_dataset_train.json",),
        default_split="train",
    ),
    "mbpp": SourceSpec(
        key="mbpp",
        source="google-research-datasets/mbpp",
        revision="4bb6404fdc6cacfda99d4ac4205087b89d32030c",
        license="CC-BY-4.0",
        adapter="mbpp_sanitized_full_static_continuation_v1",
        allow_patterns=(
            "sanitized/train-*.parquet",
            "sanitized/validation-*.parquet",
            "sanitized/test-*.parquet",
            "full/train-*.parquet",
            "full/validation-*.parquet",
            "full/test-*.parquet",
        ),
        data_patterns=(
            "sanitized/train-*.parquet",
            "sanitized/validation-*.parquet",
            "sanitized/test-*.parquet",
            "full/train-*.parquet",
            "full/validation-*.parquet",
            "full/test-*.parquet",
        ),
        default_split="train",
    ),
    "tulu_code": SourceSpec(
        key="tulu_code",
        source="allenai/tulu-3-sft-personas-code",
        revision="1412abe88dd2976af977260788e033013449f7b2",
        license="ODC-BY-1.0",
        adapter="tulu_code_plain_python_continuation_v1",
        allow_patterns=("data/train-*.parquet",),
        data_patterns=("data/train-*.parquet",),
        default_split="train",
    ),
}

SOURCE_ALIASES = {
    "tulu_general": "tulu",
    "tulu-code": "tulu_code",
    "tat-qa": "tatqa",
}
ALLOWED_DATA_SUFFIXES = {".json", ".jsonl", ".parquet"}
SPECIAL_MARKERS = (
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<image>",
    "<video>",
    "<audio>",
)
REFERENCE_EVIDENCE_VERIFIER_VERSION = "day20_reference_evidence_v1"
CHOICE_LABELS = "ABCD"
DEFAULT_CANDIDATE_LIMITS = {"mmlu": 6_000, "tulu": 1_500, "gsm8k": 1_500}
_GENERAL_REPLAY_EXCLUSIONS = (
    re.compile(
        r"\b(?:python|javascript|typescript|c\+\+|source\s+code|debug(?:ging)?|"
        r"implement\s+(?:a|the)\s+(?:function|class|algorithm)|"
        r"write\s+(?:a|the)?\s*(?:function|program|script|code))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:solve\s+(?:the|this)|calculate|derivative|integral|quadratic\s+equation|"
        r"mathematical\s+proof|prove\s+that)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:balance\s+sheet|income\s+statement|financial\s+ratio|earnings\s+per\s+share|"
        r"stock\s+valuation)\b",
        re.IGNORECASE,
    ),
)
_CALCULATOR_ANNOTATION_RE = re.compile(r"<<[^<>\n]*>>")
_CODE_DEFINITION_RE = re.compile(r"(?m)^\s*(?:async\s+def|def|class)\s+")


def _plain_text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise SourceRecordRejected(f"{label}_not_text")
    text = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text or "\x00" in text:
        raise SourceRecordRejected(f"{label}_empty_or_nul")
    return text


def canonicalize_candidate_for_audit(
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Canonicalize messages before deriving any tokenizer or gold evidence."""
    result = dict(candidate)
    messages = candidate.get("messages")
    if not isinstance(messages, list) or not messages:
        raise Day20SourceAdapterError("candidate messages are missing")
    canonical_messages: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise Day20SourceAdapterError("candidate message is not an object")
        role = message.get("role")
        if role == "assistant":
            try:
                content = normalize_assistant_target(
                    str(candidate.get("skill")),
                    str(candidate.get("target_format")),
                    message.get("content"),
                    code_prefix=(
                        str(candidate["code_prefix"])
                        if candidate.get("code_prefix") is not None
                        else None
                    ),
                    finance_scale=(
                        str(candidate["finance_scale"])
                        if candidate.get("finance_scale") is not None
                        else None
                    ),
                )
            except Day20ContractError as error:
                raise Day20SourceAdapterError(
                    "transformed assistant target is not canonicalizable"
                ) from error
        elif role in {"system", "user"}:
            content = _plain_text(message.get("content"), f"{role}_message")
        else:
            raise Day20SourceAdapterError(f"invalid candidate role: {role!r}")
        canonical_messages.append({"role": str(role), "content": content})
    result["messages"] = canonical_messages
    return result


def _canonical_number(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise SourceRecordRejected("answer_not_scalar_numeric")
    text = str(value).strip().replace(",", "")
    try:
        number = Decimal(text)
    except InvalidOperation as error:
        raise SourceRecordRejected("answer_not_decimal") from error
    if not number.is_finite():
        raise SourceRecordRejected("answer_not_finite")
    rendered = format(number, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return "0" if rendered in {"", "+0", "-0"} else rendered


def _reject_markers(text: str, *, allow_markdown: bool = False) -> None:
    if any(marker in text for marker in SPECIAL_MARKERS):
        raise SourceRecordRejected("template_or_media_marker")
    if not allow_markdown and "```" in text:
        raise SourceRecordRejected("markdown_fence")


def _parent_id(prefix: str, raw_identity: str) -> str:
    return f"{prefix}:{raw_identity}"


def _sample_id(skill: str, source_key: str, parent_id: str) -> str:
    return f"train:{skill}:{source_key}:{text_sha256(parent_id)[:24]}"


def transform_mmlu(record: Mapping[str, Any], parent_id: str) -> dict[str, Any]:
    question = _plain_text(record.get("question"), "mmlu_question")
    choices = record.get("choices")
    answer = record.get("answer")
    if not isinstance(choices, (list, tuple)) or len(choices) != 4:
        raise SourceRecordRejected("mmlu_requires_four_choices")
    normalized_choices = [_plain_text(choice, "mmlu_choice") for choice in choices]
    if isinstance(answer, bool):
        raise SourceRecordRejected("mmlu_answer_invalid")
    try:
        answer_index = int(answer)
    except (TypeError, ValueError) as error:
        raise SourceRecordRejected("mmlu_answer_invalid") from error
    if answer_index not in range(4) or str(answer).strip() not in {
        str(answer_index),
        f"{answer_index}.0",
    }:
        raise SourceRecordRejected("mmlu_answer_out_of_range")
    prompt = (
        "Answer the multiple-choice question. Respond exactly with "
        "'Final answer: <A|B|C|D>'.\n\n"
        f"Question: {question}\n\nChoices:\n"
        + "\n".join(
            f"{CHOICE_LABELS[index]}. {choice}"
            for index, choice in enumerate(normalized_choices)
        )
    )
    subject = record.get("subject")
    result = {
        "sample_id": _sample_id("general", "mmlu", parent_id),
        "parent_id": parent_id,
        "skill": "general",
        "target_format": "general_mcq",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"Final answer: {CHOICE_LABELS[answer_index]}"},
        ],
        "subskill": "multiple_choice_knowledge",
    }
    if isinstance(subject, str) and subject.strip():
        result["mmlu_subject"] = subject.strip()
    return result


def transform_tulu_general(
    record: Mapping[str, Any], parent_id: str
) -> dict[str, Any]:
    raw_messages = record.get("messages")
    if not isinstance(raw_messages, list) or len(raw_messages) not in {2, 3}:
        raise SourceRecordRejected("tulu_not_single_turn")
    roles = ["user", "assistant"] if len(raw_messages) == 2 else ["system", "user", "assistant"]
    messages: list[dict[str, str]] = []
    for index, expected_role in enumerate(roles):
        message = raw_messages[index]
        if not isinstance(message, Mapping) or message.get("role") != expected_role:
            raise SourceRecordRejected("tulu_role_sequence")
        content = _plain_text(message.get("content"), f"tulu_{expected_role}")
        _reject_markers(content, allow_markdown=expected_role != "assistant")
        messages.append({"role": expected_role, "content": content})
    prompt = messages[-2]["content"]
    response = messages[-1]["content"]
    if response.startswith(("Reasoning:", "Calculation:")):
        raise SourceRecordRejected("tulu_cross_task_prefix")
    if any(pattern.search(prompt) for pattern in _GENERAL_REPLAY_EXCLUSIONS):
        raise SourceRecordRejected("tulu_cross_skill_prompt")
    return {
        "sample_id": _sample_id("general", "tulu", parent_id),
        "parent_id": parent_id,
        "skill": "general",
        "target_format": "general_instruction",
        "messages": messages,
        "subskill": "constraint_following_replay",
    }


def transform_gsm8k(record: Mapping[str, Any], parent_id: str) -> dict[str, Any]:
    question = _plain_text(record.get("question"), "gsm8k_question")
    raw_answer = _plain_text(record.get("answer"), "gsm8k_answer")
    reasoning, marker, final = raw_answer.rpartition("####")
    if not marker:
        raise SourceRecordRejected("gsm8k_missing_final_marker")
    reasoning = _CALCULATOR_ANNOTATION_RE.sub("", reasoning).strip()
    if not reasoning or reasoning.startswith(("Reasoning:", "Calculation:")):
        raise SourceRecordRejected("gsm8k_invalid_reasoning")
    final_number = _canonical_number(final.strip())
    prompt = (
        "Solve the problem and show the necessary reasoning. End with exactly "
        "'Final answer: <number>'.\n\n"
        f"Problem: {question}"
    )
    return {
        "sample_id": _sample_id("math", "gsm8k", parent_id),
        "parent_id": parent_id,
        "skill": "math",
        "target_format": "math_reasoning",
        "messages": [
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": f"{reasoning}\nFinal answer: {final_number}",
            },
        ],
        "subskill": "grade_school_reasoning",
    }


def render_tatqa_context(document: Mapping[str, Any]) -> str:
    table = document.get("table")
    if not isinstance(table, Mapping) or not isinstance(table.get("table"), list):
        raise SourceRecordRejected("tatqa_table_missing")
    table_lines: list[str] = []
    for row in table["table"]:
        if not isinstance(row, list):
            raise SourceRecordRejected("tatqa_table_row_invalid")
        table_lines.append("\t".join(str(cell).strip() for cell in row))
    paragraphs = document.get("paragraphs")
    if not isinstance(paragraphs, list):
        raise SourceRecordRejected("tatqa_paragraphs_missing")
    normalized_paragraphs: list[tuple[int, str]] = []
    for paragraph in paragraphs:
        if not isinstance(paragraph, Mapping):
            raise SourceRecordRejected("tatqa_paragraph_invalid")
        try:
            order = int(paragraph.get("order"))
        except (TypeError, ValueError) as error:
            raise SourceRecordRejected("tatqa_paragraph_order_invalid") from error
        normalized_paragraphs.append(
            (order, _plain_text(paragraph.get("text"), "tatqa_paragraph_text"))
        )
    paragraph_lines = [
        f"[{order}] {text}" for order, text in sorted(normalized_paragraphs)
    ]
    return "Table:\n" + "\n".join(table_lines) + "\n\nParagraphs:\n" + "\n".join(paragraph_lines)


def transform_tatqa_document(
    document: Mapping[str, Any], document_parent_id: str
) -> list[tuple[dict[str, Any], Mapping[str, Any]]]:
    context = render_tatqa_context(document)
    questions = document.get("questions")
    if not isinstance(questions, list):
        raise SourceRecordRejected("tatqa_questions_missing")
    transformed: list[tuple[dict[str, Any], Mapping[str, Any]]] = []
    for question_index, question in enumerate(questions):
        if not isinstance(question, Mapping):
            continue
        answer_type = str(question.get("answer_type") or "").strip().lower()
        if answer_type not in {"arithmetic", "count"}:
            continue
        raw_answer = question.get("answer")
        if isinstance(raw_answer, (list, tuple, dict)):
            continue
        try:
            answer = _canonical_number(raw_answer)
        except SourceRecordRejected:
            continue
        raw_scale = str(question.get("scale") or "").strip().lower()
        scale = "none" if not raw_scale else raw_scale
        if scale not in {"none", "percent", "thousand", "million", "billion"}:
            continue
        suffix = "%" if scale == "percent" else (f" {scale}" if scale != "none" else "")
        question_text = _plain_text(question.get("question"), "tatqa_question")
        question_uid = question.get("uid")
        identity = (
            str(question_uid).strip()
            if question_uid is not None and str(question_uid).strip()
            else f"{document_parent_id}:question:{question_index}"
        )
        parent_id = _parent_id("tatqa", identity)
        prompt = (
            "Use the financial table and paragraphs to answer the question. "
            "Return exactly one line in the form 'Final answer: <value>' and "
            "include the percent sign or scale word when required.\n\n"
            f"{context}\n\nQuestion:\n{question_text}"
        )
        transformed.append(
            (
                {
                    "sample_id": _sample_id("finance", "tatqa", parent_id),
                    "parent_id": parent_id,
                    "skill": "finance",
                    "target_format": "finance_value_scale",
                    "messages": [
                        {"role": "user", "content": prompt},
                        {
                            "role": "assistant",
                            "content": f"Final answer: {answer}{suffix}",
                        },
                    ],
                    "finance_scale": scale,
                    "subskill": answer_type,
                },
                {
                    "table": document.get("table"),
                    "paragraphs": document.get("paragraphs"),
                    "question": question,
                },
            )
        )
    return transformed


def _compile_only(source: str, label: str) -> None:
    try:
        compile(source, f"<{label}>", "exec")
    except (SyntaxError, ValueError, TypeError) as error:
        raise SourceRecordRejected(f"{label}_static_compile_failed") from error


def split_mbpp_continuation(code: Any) -> tuple[str, str, str]:
    source = _plain_text(code, "mbpp_code")
    _reject_markers(source)
    try:
        module = ast.parse(source)
    except SyntaxError as error:
        raise SourceRecordRejected("mbpp_solution_parse_failed") from error
    functions = [
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    allowed_nodes = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef)
    if len(functions) != 1 or any(not isinstance(node, allowed_nodes) for node in module.body):
        raise SourceRecordRejected("mbpp_requires_one_top_level_function")
    function = functions[0]
    lines = source.splitlines()
    body_start = function.body[0].lineno - 1 if function.body else -1
    if body_start <= 0:
        raise SourceRecordRejected("mbpp_function_body_missing")
    prefix_end = body_start
    leading_prefix_nodes = 0
    for body_node in function.body:
        is_docstring = (
            isinstance(body_node, ast.Expr)
            and isinstance(body_node.value, ast.Constant)
            and isinstance(body_node.value.value, str)
        )
        if is_docstring or isinstance(
            body_node,
            (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            prefix_end = int(body_node.end_lineno or body_node.lineno)
            leading_prefix_nodes += 1
            continue
        break
    continuation_nodes = function.body[leading_prefix_nodes:]
    if any(
        isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for body_node in continuation_nodes
        for node in ast.walk(body_node)
    ):
        raise SourceRecordRejected("mbpp_import_or_helper_not_prefixable")
    prefix = "\n".join(line.rstrip() for line in lines[:prefix_end]).rstrip()
    continuation = "\n".join(line.rstrip() for line in lines[prefix_end:]).rstrip()
    if not prefix or not continuation:
        raise SourceRecordRejected("mbpp_empty_prefix_or_continuation")
    if _CODE_DEFINITION_RE.search(continuation):
        raise SourceRecordRejected("mbpp_continuation_contains_definition")
    rebuilt = f"{prefix}\n{continuation}\n"
    _compile_only(rebuilt, "mbpp_solution")
    return prefix, continuation, function.name


def transform_mbpp(
    record: Mapping[str, Any], parent_id: str, *, variant: str
) -> dict[str, Any]:
    problem = record.get("prompt") if variant == "sanitized" else record.get("text")
    if problem is None:
        problem = record.get("text") if variant == "sanitized" else record.get("prompt")
    problem_text = _plain_text(problem, "mbpp_problem")
    _reject_markers(problem_text, allow_markdown=True)
    prefix, continuation, function_name = split_mbpp_continuation(record.get("code"))

    setup = record.get("test_setup_code") or record.get("test_imports") or ""
    if isinstance(setup, list):
        setup = "\n".join(str(item) for item in setup)
    if setup:
        _compile_only(str(setup), "mbpp_test_setup")
    for tests_key in ("test_list", "challenge_test_list"):
        tests = record.get(tests_key)
        if tests is not None:
            if not isinstance(tests, (list, tuple)) or not all(
                isinstance(test, str) and test.strip() for test in tests
            ):
                raise SourceRecordRejected(f"mbpp_{tests_key}_invalid")
            if tests:
                _compile_only(
                    "\n".join(test.strip() for test in tests),
                    f"mbpp_{tests_key}",
                )

    prompt = (
        "Complete the Python function below. Return only the indented Python "
        "continuation: no Markdown fence, explanation, or repeated function definition.\n\n"
        f"Task: {problem_text}\n\nStarter code:\n{prefix}"
    )
    return {
        "sample_id": _sample_id("code", "mbpp", parent_id),
        "parent_id": parent_id,
        "skill": "code",
        "target_format": "code_continuation",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": continuation},
        ],
        "code_prefix": prefix,
        "subskill": f"python_continuation_{variant}",
        "entry_point": function_name,
    }


def transform_tulu_code(
    record: Mapping[str, Any], parent_id: str
) -> dict[str, Any]:
    """Convert one plain-Python Tulu response into a body continuation.

    Fenced answers and prose wrappers are rejected rather than stripped.  The
    response itself must parse as a module containing exactly one top-level
    Python function plus optional imports.  Consequently all module imports
    and the full function header live in ``code_prefix`` while the assistant
    target contains only the indented function body.
    """
    raw_messages = record.get("messages")
    if not isinstance(raw_messages, list) or len(raw_messages) != 2:
        raise SourceRecordRejected("tulu_code_not_single_turn")
    if [message.get("role") for message in raw_messages if isinstance(message, Mapping)] != [
        "user",
        "assistant",
    ]:
        raise SourceRecordRejected("tulu_code_role_sequence")
    prompt_text = _plain_text(raw_messages[0].get("content"), "tulu_code_prompt")
    source_code = _plain_text(raw_messages[1].get("content"), "tulu_code_response")
    _reject_markers(prompt_text, allow_markdown=True)
    # No fence removal and no prose extraction: accepted bytes are Python.
    _reject_markers(source_code)
    prefix, continuation, function_name = split_mbpp_continuation(source_code)
    if _CODE_DEFINITION_RE.search(continuation):
        raise SourceRecordRejected("tulu_code_continuation_contains_definition")
    prompt = (
        "Complete the Python function below. Return only the indented Python "
        "continuation: no Markdown fence, explanation, import preamble, or "
        "repeated function definition.\n\n"
        f"Task: {prompt_text}\n\nStarter code:\n{prefix}"
    )
    return {
        "sample_id": _sample_id("code", "tulu-code", parent_id),
        "parent_id": parent_id,
        "skill": "code",
        "target_format": "code_continuation",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": continuation},
        ],
        "code_prefix": prefix,
        "subskill": "python_continuation_tulu_code",
        "entry_point": function_name,
    }


def _canonical_code_bytes(value: Any) -> str:
    source = _plain_text(value, "reference_code")
    return "\n".join(line.rstrip() for line in source.splitlines()).strip()


def _canonical_tests(record: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("test_list", "challenge_test_list"):
        value = record.get(key) or []
        if not isinstance(value, (list, tuple)) or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            raise Day20SourceAdapterError(f"cannot bind malformed MBPP {key}")
        result[key] = [item.strip() for item in value]
    setup = record.get("test_setup_code") or record.get("test_imports") or ""
    if isinstance(setup, list):
        setup = "\n".join(str(item).strip() for item in setup if str(item).strip())
    result["test_setup_code"] = str(setup).strip()
    return result


def _finance_target_parts(candidate: Mapping[str, Any]) -> tuple[str, str]:
    scale = str(candidate.get("finance_scale") or "").strip().lower()
    target = str(candidate["messages"][-1]["content"])
    value = target.removeprefix("Final answer:").strip()
    if scale == "percent":
        if not value.endswith("%"):
            raise Day20SourceAdapterError("finance target lost its percent suffix")
        value = value[:-1].strip()
    elif scale != "none":
        suffix = f" {scale}"
        if not value.lower().endswith(suffix):
            raise Day20SourceAdapterError("finance target lost its scale suffix")
        value = value[: -len(suffix)].strip()
    return _canonical_number(value), scale


def build_reference_evidence(
    *,
    source_key: str,
    candidate: Mapping[str, Any],
    source_record: Mapping[str, Any],
    source_content_sha256: str,
) -> dict[str, Any]:
    """Bind a normalized target to the exact gold fields that produced it."""
    target_format = str(candidate["target_format"])
    target = str(candidate["messages"][-1]["content"])
    checks: dict[str, Any]
    if source_key == "mmlu":
        answer_index = int(source_record["answer"])
        source_letter = CHOICE_LABELS[answer_index]
        target_letter = target.removeprefix("Final answer:").strip()
        checks = {
            "source_answer_index": answer_index,
            "source_answer_letter": source_letter,
            "target_answer_letter": target_letter,
            "source_target_match": source_letter == target_letter,
        }
    elif source_key == "tulu":
        raw_messages = source_record.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise Day20SourceAdapterError("cannot bind Tulu replay gold response")
        source_assistant = _plain_text(
            raw_messages[-1].get("content"), "tulu_reference_assistant"
        )
        checks = {
            "source_assistant_sha256": text_sha256(source_assistant),
            "target_assistant_sha256": text_sha256(target),
            "source_target_match": source_assistant == target,
        }
    elif source_key == "gsm8k":
        raw_answer = _plain_text(source_record.get("answer"), "gsm8k_reference")
        _, marker, raw_gold = raw_answer.rpartition("####")
        if not marker:
            raise Day20SourceAdapterError("cannot bind GSM8K final gold")
        source_gold = _canonical_number(raw_gold.strip())
        target_gold = _canonical_number(target.rsplit("Final answer:", 1)[-1].strip())
        checks = {
            "source_parsed_final_number": source_gold,
            "target_final_number": target_gold,
            "source_target_match": source_gold == target_gold,
        }
    elif source_key == "tatqa":
        question = source_record.get("question")
        if not isinstance(question, Mapping):
            raise Day20SourceAdapterError("cannot bind TAT-QA source question")
        source_answer = _canonical_number(question.get("answer"))
        source_scale = str(question.get("scale") or "none").strip().lower() or "none"
        target_answer, target_scale = _finance_target_parts(candidate)
        checks = {
            "source_numeric_answer": source_answer,
            "source_scale": source_scale,
            "target_numeric_answer": target_answer,
            "target_scale": target_scale,
            "source_target_match": (
                source_answer == target_answer and source_scale == target_scale
            ),
        }
    elif source_key in {"mbpp", "tulu_code"}:
        if source_key == "mbpp":
            source_code_value = source_record.get("code")
        else:
            raw_messages = source_record.get("messages")
            if not isinstance(raw_messages, list) or not raw_messages:
                raise Day20SourceAdapterError("cannot bind Tulu Code source response")
            source_code_value = raw_messages[-1].get("content")
        raw_source_code = _canonical_code_bytes(source_code_value)
        reference_prefix, reference_continuation, _ = split_mbpp_continuation(
            source_code_value
        )
        try:
            canonical_reference_continuation = normalize_assistant_target(
                "code",
                "code_continuation",
                reference_continuation,
                code_prefix=reference_prefix,
            )
        except Day20ContractError as error:
            raise Day20SourceAdapterError(
                "cannot canonicalize the source code continuation"
            ) from error
        source_code = _canonical_code_bytes(
            f"{reference_prefix}\n{canonical_reference_continuation}"
        )
        rebuilt_code = _canonical_code_bytes(
            f"{candidate['code_prefix']}\n{target}"
        )
        _compile_only(raw_source_code, "reference_raw_source_code")
        _compile_only(source_code, "reference_source_code")
        _compile_only(rebuilt_code, "reference_rebuilt_code")
        checks = {
            "raw_source_code_sha256": text_sha256(raw_source_code),
            "source_code_sha256": text_sha256(source_code),
            "rebuilt_code_sha256": text_sha256(rebuilt_code),
            "source_rebuilt_match": source_code == rebuilt_code,
            "static_compile": True,
        }
        if source_key == "mbpp":
            tests = _canonical_tests(source_record)
            checks.update(
                {
                    "tests_count": len(tests["test_list"]),
                    "challenge_tests_count": len(tests["challenge_test_list"]),
                    "tests_sha256": object_sha256(tests),
                }
            )
    else:
        raise Day20SourceAdapterError(f"no reference verifier for source {source_key}")
    match_key = "source_rebuilt_match" if source_key in {"mbpp", "tulu_code"} else "source_target_match"
    if checks.get(match_key) is not True:
        raise Day20SourceAdapterError(
            f"reference evidence target mismatch for {candidate.get('sample_id')}"
        )
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "verifier_version": REFERENCE_EVIDENCE_VERIFIER_VERSION,
        "target_format": target_format,
        "source_content_sha256": source_content_sha256,
        "checks": checks,
    }
    evidence["reference_evidence_sha256"] = object_sha256(evidence)
    return evidence


def verify_reference_evidence(evidence: Mapping[str, Any]) -> str:
    payload = dict(evidence)
    expected = payload.pop("reference_evidence_sha256", None)
    actual = object_sha256(payload)
    if (
        evidence.get("schema_version") != 1
        or evidence.get("verifier_version") != REFERENCE_EVIDENCE_VERIFIER_VERSION
        or expected != actual
    ):
        raise Day20SourceAdapterError("reference evidence self-hash is invalid")
    return actual


def token_sequence_sha256(values: Sequence[int]) -> str:
    """Hash a token sequence as its canonical compact JSON integer array.

    Formula: ``sha256(json.dumps([int...], sort_keys=True,
    separators=(",", ":"), ensure_ascii=False).encode("utf-8"))``.  This is
    exactly :func:`day20_contract.object_sha256` applied to the integer list and
    is shared by adapter generation, independent audits, and Trainer callbacks.
    """
    if isinstance(values, (str, bytes, bytearray)):
        raise Day20SourceAdapterError("token sequence must be an integer sequence")
    try:
        normalized = [int(value) for value in values]
    except (TypeError, ValueError) as error:
        raise Day20SourceAdapterError("token sequence is not integral") from error
    return object_sha256(normalized)


def audit_messages(messages: Sequence[Mapping[str, str]], template: Any) -> dict[str, Any]:
    """Create token evidence using a configured ms-swift train template."""
    canonical_messages = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages
    ]
    try:
        encoded = template.encode({"messages": canonical_messages}, return_length=True)
    except Exception as error:
        if error.__class__.__name__ == "MaxLengthError":
            raise SourceRecordRejected("qwen35_over_max_length") from error
        raise Day20SourceAdapterError(
            f"Qwen3.5 template encoding failed: {type(error).__name__}: {error}"
        ) from error
    if not isinstance(encoded, Mapping):
        raise Day20SourceAdapterError("Qwen3.5 template returned a non-mapping")
    input_ids = encoded.get("input_ids")
    labels = encoded.get("labels")
    if not isinstance(input_ids, (list, tuple)) or not isinstance(labels, (list, tuple)):
        raise Day20SourceAdapterError("Qwen3.5 template omitted input_ids or labels")
    try:
        token_ids = [int(value) for value in input_ids]
        label_ids = [int(value) for value in labels]
    except (TypeError, ValueError) as error:
        raise Day20SourceAdapterError("Qwen3.5 token evidence is not integral") from error
    if not token_ids or len(token_ids) != len(label_ids):
        raise Day20SourceAdapterError("Qwen3.5 input/label lengths are invalid")
    # Causal-LM loss shifts labels left; position zero is never predicted.
    # Keep this formula explicit even though the pinned template also masks it.
    supervised = sum(label != -100 for label in label_ids[1:])
    if supervised <= 0:
        raise SourceRecordRejected("qwen35_zero_supervised_tokens")
    if len(token_ids) > int(QWEN35_TEMPLATE_CONTRACT["max_length"]):
        raise Day20SourceAdapterError("Qwen3.5 template silently exceeded max_length")
    return {
        "template": "qwen3_5",
        "enable_thinking": False,
        "add_non_thinking_prefix": True,
        "loss_scale": "default+ignore_empty_think",
        "truncated": False,
        "input_tokens": len(token_ids),
        "supervised_tokens": supervised,
        "messages_sha256": object_sha256(canonical_messages),
        # Hash exact token arrays rather than a lossy decode/re-encode round trip.
        "render_sha256": token_sequence_sha256(token_ids),
        "labels_sha256": token_sequence_sha256(label_ids),
        "evidence_version": "qwen35_swift_train_token_arrays_v1",
    }


def build_qwen35_template(model_path: Path) -> Any:
    if not model_path.is_dir():
        raise Day20SourceAdapterError(f"local Qwen3.5 model directory is missing: {model_path}")
    os.environ.setdefault("USE_HF", "1")
    try:
        from swift import get_model_processor, get_template
    except ImportError as error:
        raise Day20SourceAdapterError(
            "ms-swift is required in the existing environment; refusing to install it"
        ) from error
    model, processor = get_model_processor(
        str(model_path),
        model_type="qwen3_5",
        load_model=False,
        use_hf=True,
        download_model=False,
    )
    if model is not None:
        raise Day20SourceAdapterError("load_model=False unexpectedly returned model weights")
    template = get_template(
        processor,
        template_type="qwen3_5",
        max_length=2304,
        truncation_strategy="raise",
        padding_free=False,
        loss_scale="default+ignore_empty_think",
        enable_thinking=False,
        add_non_thinking_prefix=True,
    )
    template.set_mode("train")
    meta = getattr(template, "template_meta", None)
    if (
        getattr(meta, "template_type", None) != "qwen3_5"
        or getattr(template, "max_length", None) != 2304
        or getattr(template, "truncation_strategy", None) != "raise"
        or getattr(template, "enable_thinking", None) is not False
        or getattr(template, "add_non_thinking_prefix", None) is not True
        or getattr(template, "padding_free", None) is not False
        or getattr(template, "packing", None) is not False
    ):
        raise Day20SourceAdapterError("constructed Qwen3.5 template drifted from contract")
    return template


def _stored_tokenization(row: Mapping[str, Any]) -> dict[str, Any]:
    nested = row.get("qwen35_tokenization")
    if isinstance(nested, Mapping):
        return dict(nested)
    flat_keys = (
        "qwen35_input_tokens",
        "qwen35_supervised_tokens",
        "qwen35_render_sha256",
        "qwen35_labels_sha256",
    )
    if not all(key in row for key in flat_keys):
        raise Day20SourceAdapterError(
            f"{row.get('sample_id', '<unknown>')}: stored Qwen3.5 evidence is missing"
        )
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise Day20SourceAdapterError(
            f"{row.get('sample_id', '<unknown>')}: messages are missing"
        )
    return {
        "template": "qwen3_5",
        "enable_thinking": False,
        "add_non_thinking_prefix": True,
        "loss_scale": "default+ignore_empty_think",
        "truncated": False,
        "input_tokens": row["qwen35_input_tokens"],
        "supervised_tokens": row["qwen35_supervised_tokens"],
        "messages_sha256": object_sha256(messages),
        "render_sha256": row["qwen35_render_sha256"],
        "labels_sha256": row["qwen35_labels_sha256"],
    }


def _raw_contract_row(row: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(row)
    raw["qwen35_tokenization"] = dict(evidence)
    return raw


def _tokenizer_file_identity(model_path: Path) -> dict[str, str]:
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
        name: file_sha256(model_path / name)
        for name in names
        if (model_path / name).is_file()
    }
    if "config.json" not in files or not any(
        name in files for name in ("tokenizer.json", "vocab.json")
    ):
        raise Day20SourceAdapterError(
            "model path lacks config.json or tokenizer identity files"
        )
    return dict(sorted(files.items()))


def audit_normalized_sources(
    *,
    model_path: str | Path,
    sources: Mapping[str, str | Path],
    max_length: int = 2304,
) -> dict[str, Any]:
    """Re-encode stored rows and prove their token evidence has not drifted.

    Both adapter JSONL (nested ``qwen35_tokenization``) and the flattened rows
    emitted by ``prepare_day20.py`` are accepted.  No stored token count or hash
    is trusted: canonical messages are reconstructed by the Day 20 contract,
    encoded again, and every evidence field is compared to the recomputation.
    """
    if max_length != int(QWEN35_TEMPLATE_CONTRACT["max_length"]):
        raise Day20SourceAdapterError("Day 20 token audit max_length is frozen at 2304")
    expected_skills = {"general", "math", "finance", "code"}
    if not isinstance(sources, Mapping) or set(sources) != expected_skills:
        raise Day20SourceAdapterError(
            "token audit sources must contain exactly general, math, finance, and code"
        )
    resolved_model = Path(model_path).expanduser().resolve()
    template = build_qwen35_template(resolved_model)
    report_sources: dict[str, Any] = {}
    evidence_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    total_tokens = 0
    for skill in ("general", "math", "finance", "code"):
        path = Path(sources[skill]).expanduser().resolve()
        if not path.is_file():
            raise Day20SourceAdapterError(f"token audit source is missing: {path}")
        before_hash = file_sha256(path)
        records = 0
        skill_tokens = 0
        for line_index, row in _iter_json(path):
            stored = _stored_tokenization(row)
            try:
                canonical = normalize_and_validate_record(
                    _raw_contract_row(row, stored),
                    expected_skill=skill,
                    max_length=max_length,
                )
            except Day20ContractError as error:
                raise Day20SourceAdapterError(
                    f"{path}:{line_index + 1}: stored normalized row is invalid: {error}"
                ) from error
            sample_id = canonical["sample_id"]
            if sample_id in seen_ids:
                raise Day20SourceAdapterError(f"duplicate token-audit sample_id: {sample_id}")
            seen_ids.add(sample_id)
            recomputed = audit_messages(canonical["messages"], template)
            comparison_keys = (
                "template",
                "enable_thinking",
                "add_non_thinking_prefix",
                "loss_scale",
                "truncated",
                "input_tokens",
                "supervised_tokens",
                "messages_sha256",
                "render_sha256",
                "labels_sha256",
            )
            drift = {
                key: {"stored": stored.get(key), "recomputed": recomputed[key]}
                for key in comparison_keys
                if stored.get(key) != recomputed[key]
            }
            if drift:
                raise Day20SourceAdapterError(
                    f"{path}:{line_index + 1}: Qwen3.5 token evidence drift: {drift}"
                )
            if "evidence_version" in stored and (
                stored["evidence_version"] != recomputed["evidence_version"]
            ):
                raise Day20SourceAdapterError(
                    f"{path}:{line_index + 1}: token evidence version drifted"
                )
            records += 1
            row_tokens = int(recomputed["supervised_tokens"])
            skill_tokens += row_tokens
            total_tokens += row_tokens
            evidence_rows.append(
                {
                    "sample_id": sample_id,
                    "skill": skill,
                    "input_tokens": recomputed["input_tokens"],
                    "supervised_tokens": row_tokens,
                    "render_sha256": recomputed["render_sha256"],
                    "labels_sha256": recomputed["labels_sha256"],
                }
            )
        if not records:
            raise Day20SourceAdapterError(f"token audit source is empty: {path}")
        after_hash = file_sha256(path)
        if after_hash != before_hash:
            raise Day20SourceAdapterError(f"token audit source changed while reading: {path}")
        report_sources[skill] = {
            "path": str(path),
            "file_sha256": before_hash,
            "records": records,
            "supervised_tokens": skill_tokens,
        }
    result: dict[str, Any] = {
        "schema_version": 1,
        "domain": "day20.qwen35_source_token_reaudit",
        "status": "pass",
        "model_path": str(resolved_model),
        "tokenizer_files": _tokenizer_file_identity(resolved_model),
        "template": dict(QWEN35_TEMPLATE_CONTRACT),
        "sources": report_sources,
        "records": len(evidence_rows),
        "supervised_tokens": total_tokens,
        "ordered_evidence_sha256": object_sha256(evidence_rows),
    }
    result["audit_sha256"] = object_sha256(result)
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_json(path: Path) -> Iterator[tuple[int, Mapping[str, Any]]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise Day20SourceAdapterError(f"invalid JSONL at {path}:{index + 1}") from error
                if not isinstance(value, Mapping):
                    raise Day20SourceAdapterError(f"JSONL row is not an object: {path}:{index + 1}")
                yield index, value
        return
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise Day20SourceAdapterError(f"invalid JSON source: {path}") from error
    if isinstance(value, Mapping) and isinstance(value.get("records"), list):
        value = value["records"]
    elif isinstance(value, Mapping) and isinstance(value.get("data"), list):
        value = value["data"]
    elif isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, list):
        raise Day20SourceAdapterError(f"JSON source root is not records: {path}")
    for index, record in enumerate(value):
        if not isinstance(record, Mapping):
            raise Day20SourceAdapterError(f"JSON row is not an object: {path}:{index}")
        yield index, record


def iter_source_records(path: Path) -> Iterator[tuple[int, Mapping[str, Any]]]:
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        yield from _iter_json(path)
        return
    if suffix != ".parquet":
        raise Day20SourceAdapterError(f"unsupported source extension: {path}")
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise Day20SourceAdapterError(
            "pyarrow is required to read staged Parquet; refusing to install it"
        ) from error
    try:
        rows = parquet.read_table(path).to_pylist()
    except Exception as error:
        raise Day20SourceAdapterError(f"cannot read Parquet source {path}: {error}") from error
    for index, record in enumerate(rows):
        if not isinstance(record, Mapping):
            raise Day20SourceAdapterError(f"Parquet row is not an object: {path}:{index}")
        yield index, record


def _normalize_local_paths(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config.get("local_paths", {})
    if not isinstance(value, Mapping):
        raise Day20SourceAdapterError("local_paths must be an object")
    revisions = config.get("local_revisions", {})
    if not isinstance(revisions, Mapping):
        raise Day20SourceAdapterError("local_revisions must be an object")
    normalized: dict[str, Any] = {}
    for raw_key, paths in value.items():
        key = SOURCE_ALIASES.get(str(raw_key), str(raw_key))
        if key not in SOURCE_SPECS:
            raise Day20SourceAdapterError(f"unknown local source key: {raw_key!r}")
        if key in normalized:
            raise Day20SourceAdapterError(f"duplicate local source key: {key!r}")
        declared_revision = revisions.get(raw_key, revisions.get(key))
        if declared_revision != SOURCE_SPECS[key].revision:
            raise Day20SourceAdapterError(
                f"staged local source {key!r} must declare exact revision "
                f"{SOURCE_SPECS[key].revision} in local_revisions"
            )
        normalized[key] = paths
    return normalized


def _flatten_path_value(value: Any, *, label: str = "") -> list[tuple[str, Path]]:
    if isinstance(value, (str, os.PathLike)):
        return [(label, Path(value).expanduser().resolve())]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result: list[tuple[str, Path]] = []
        for item in value:
            result.extend(_flatten_path_value(item, label=label))
        return result
    if isinstance(value, Mapping):
        result = []
        for child_label, child in value.items():
            result.extend(_flatten_path_value(child, label=str(child_label)))
        return result
    raise Day20SourceAdapterError("local source path must be a path, list, or label-to-path object")


def _variant_split(spec: SourceSpec, path: Path, label: str) -> tuple[str, str]:
    combined = f"{label}/{path.parent.name}/{path.name}".lower()
    if spec.key == "mbpp":
        variant = "sanitized" if "sanitized" in combined else ("full" if "full" in combined else "")
        split = next((name for name in ("train", "validation", "test") if name in combined), "")
        if not variant or not split:
            raise Day20SourceAdapterError(
                "flat MBPP local files require labels such as sanitized/train or full/test"
            )
        return variant, split
    if spec.key == "mmlu":
        return "all", "auxiliary_train"
    if spec.key == "gsm8k":
        return "main", "train"
    return spec.key, spec.default_split


def _files_from_local(spec: SourceSpec, value: Any) -> list[SourceFile]:
    files: list[SourceFile] = []
    for label, path in _flatten_path_value(value):
        candidates: list[Path]
        if path.is_dir():
            candidates = sorted(
                candidate
                for pattern in spec.data_patterns
                for candidate in path.glob(pattern)
                if candidate.is_file()
            )
        elif path.is_file():
            candidates = [path]
        else:
            raise Day20SourceAdapterError(f"staged local source is missing: {path}")
        for candidate in candidates:
            if candidate.suffix.lower() not in ALLOWED_DATA_SUFFIXES:
                raise Day20SourceAdapterError(f"local source is not JSON/JSONL/Parquet: {candidate}")
            variant, split = _variant_split(spec, candidate, label)
            files.append(
                SourceFile(
                    spec=spec,
                    path=candidate.resolve(),
                    split=split,
                    variant=variant,
                    relative_name=candidate.name if not label else f"{label}/{candidate.name}",
                )
            )
    if not files:
        raise Day20SourceAdapterError(f"no staged files matched pinned source {spec.key}")
    return sorted(files, key=lambda item: (item.variant, item.split, item.relative_name))


def _files_from_snapshot(
    spec: SourceSpec, *, cache_dir: Path, local_files_only: bool
) -> list[SourceFile]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise Day20SourceAdapterError(
            "huggingface_hub is required in the existing environment; refusing to install it"
        ) from error
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=spec.source,
                repo_type="dataset",
                revision=spec.revision,
                cache_dir=str(cache_dir),
                allow_patterns=list(spec.allow_patterns),
                local_files_only=local_files_only,
            )
        )
    except Exception as error:
        raise Day20SourceAdapterError(
            f"cannot resolve pinned source {spec.source}@{spec.revision}: {error}"
        ) from error
    if snapshot.name != spec.revision:
        raise Day20SourceAdapterError(
            f"snapshot revision drift for {spec.source}: {snapshot.name}"
        )
    files: list[SourceFile] = []
    for pattern in spec.data_patterns:
        for path in sorted(snapshot.glob(pattern)):
            if not path.is_file() or path.suffix.lower() not in ALLOWED_DATA_SUFFIXES:
                continue
            relative = path.relative_to(snapshot).as_posix()
            variant, split = _variant_split(spec, path, relative)
            files.append(
                SourceFile(
                    spec=spec,
                    path=path,
                    split=split,
                    variant=variant,
                    relative_name=relative,
                )
            )
    if not files:
        raise Day20SourceAdapterError(
            f"pinned snapshot has no allowed data files: {spec.source}@{spec.revision}"
        )
    return sorted(files, key=lambda item: (item.variant != "sanitized", item.variant, item.split, item.relative_name))


def resolve_source_files(config: Mapping[str, Any]) -> dict[str, list[SourceFile]]:
    local_paths = _normalize_local_paths(config)
    missing_local = set(SOURCE_SPECS) - set(local_paths)
    cache_value = config.get("cache_dir")
    if missing_local and not isinstance(cache_value, (str, os.PathLike)):
        raise Day20SourceAdapterError(
            f"cache_dir is required to resolve unstaged sources: {sorted(missing_local)}"
        )
    cache_dir = Path(cache_value).expanduser().resolve() if cache_value else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    local_files_only = bool(config.get("local_files_only", False))
    resolved: dict[str, list[SourceFile]] = {}
    for key, spec in SOURCE_SPECS.items():
        if key in local_paths:
            resolved[key] = _files_from_local(spec, local_paths[key])
        else:
            assert cache_dir is not None
            resolved[key] = _files_from_snapshot(
                spec, cache_dir=cache_dir, local_files_only=local_files_only
            )
    return resolved


def _raw_parent_id(source_file: SourceFile, row_index: int, record: Mapping[str, Any]) -> str:
    key = source_file.spec.key
    native = None
    for field in ("id", "task_id", "uid"):
        value = record.get(field)
        if value is not None and str(value).strip():
            native = str(value).strip()
            break
    if native is None:
        native = f"{source_file.relative_name}:row:{row_index}"
    return _parent_id(key, f"{source_file.variant}:{source_file.split}:{native}")


def _source_content_hash(record: Mapping[str, Any]) -> str:
    try:
        return object_sha256(record)
    except (TypeError, ValueError) as error:
        raise Day20SourceAdapterError("source row is not canonical-JSON serializable") from error


def _candidate_rank(candidate: Mapping[str, Any], source_hash: str) -> tuple[str, str]:
    sample_id = str(candidate["sample_id"])
    return text_sha256(f"day20-source-candidate-v1\0{source_hash}\0{sample_id}"), sample_id


def _candidate_limit(config: Mapping[str, Any], key: str) -> int | None:
    limits = config.get("candidate_limits", {})
    if not isinstance(limits, Mapping):
        raise Day20SourceAdapterError("candidate_limits must be an object")
    value = limits.get(key, DEFAULT_CANDIDATE_LIMITS.get(key))
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Day20SourceAdapterError(f"candidate limit for {key} must be a positive integer")
    return value


def _transform_file(
    source_file: SourceFile,
    rejection_counts: Counter[str],
) -> list[tuple[dict[str, Any], Mapping[str, Any], str, int]]:
    output: list[tuple[dict[str, Any], Mapping[str, Any], str, int]] = []
    for row_index, record in iter_source_records(source_file.path):
        parent_id = _raw_parent_id(source_file, row_index, record)
        try:
            if source_file.spec.key == "mmlu":
                candidates = [(transform_mmlu(record, parent_id), record)]
            elif source_file.spec.key == "tulu":
                candidates = [(transform_tulu_general(record, parent_id), record)]
            elif source_file.spec.key == "gsm8k":
                candidates = [(transform_gsm8k(record, parent_id), record)]
            elif source_file.spec.key == "tatqa":
                candidates = transform_tatqa_document(record, parent_id)
            elif source_file.spec.key == "mbpp":
                candidates = [
                    (
                        transform_mbpp(record, parent_id, variant=source_file.variant),
                        record,
                    )
                ]
            elif source_file.spec.key == "tulu_code":
                candidates = [(transform_tulu_code(record, parent_id), record)]
            else:
                raise Day20SourceAdapterError(f"unhandled pinned source: {source_file.spec.key}")
        except SourceRecordRejected as error:
            rejection_counts[f"{source_file.spec.key}:{error}"] += 1
            continue
        if not candidates:
            rejection_counts[f"{source_file.spec.key}:task_filter"] += 1
        for candidate, source_record in candidates:
            output.append((candidate, source_record, source_file.relative_name, row_index))
    return output


def validate_pool_capacity(
    pools: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    required_tokens_by_format: Mapping[str, int] = MAIN_TOKENS_BY_FORMAT,
    probe_tokens_by_format: Mapping[str, int] = PROBE_TOKENS_BY_FORMAT,
) -> dict[str, dict[str, Any]]:
    """Prove the deterministic probe plus main extension without replication."""
    formats: dict[str, dict[str, Any]] = {}
    capacity_failures: list[str] = []
    all_rows = [row for skill_rows in pools.values() for row in skill_rows]
    for target_format, required_tokens in required_tokens_by_format.items():
        if (
            isinstance(required_tokens, bool)
            or not isinstance(required_tokens, int)
            or required_tokens <= 0
        ):
            raise Day20SourceAdapterError(
                f"invalid required token capacity for {target_format!r}"
            )
        rows = sorted(
            (row for row in all_rows if row.get("target_format") == target_format),
            key=lambda row: (
                text_sha256(f"day20:{target_format}:v1\0{row['sample_id']}"),
                row["sample_id"],
            ),
        )
        weights: list[int] = []
        for row in rows:
            tokenization = row.get("qwen35_tokenization")
            value = tokenization.get("supervised_tokens") if isinstance(tokenization, Mapping) else None
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise Day20SourceAdapterError(
                    f"{row.get('sample_id', '<unknown>')}: invalid capacity token weight"
                )
            weights.append(value)
        available = sum(weights)
        main_exact_reachable = False
        if available >= required_tokens:
            try:
                exact_subset_indices(weights, required_tokens)
                main_exact_reachable = True
            except Day20ContractError:
                pass
        probe_tokens = probe_tokens_by_format.get(target_format)
        nested_exact_reachable = main_exact_reachable
        if probe_tokens is not None:
            if (
                isinstance(probe_tokens, bool)
                or not isinstance(probe_tokens, int)
                or probe_tokens <= 0
                or probe_tokens > required_tokens
            ):
                raise Day20SourceAdapterError(
                    f"invalid probe token capacity for {target_format!r}"
                )
            nested_exact_reachable = False
            try:
                probe_indices = exact_subset_indices(weights, probe_tokens)
                remaining = [
                    weight for index, weight in enumerate(weights) if index not in probe_indices
                ]
                exact_subset_indices(remaining, required_tokens - probe_tokens)
                nested_exact_reachable = True
            except Day20ContractError:
                pass
        formats[target_format] = {
            "records": len(rows),
            "available_supervised_tokens": available,
            "required_supervised_tokens": required_tokens,
            "probe_required_supervised_tokens": probe_tokens,
            "main_exact_subset_reachable": main_exact_reachable,
            "exact_required_subset_reachable": nested_exact_reachable,
        }
        if not nested_exact_reachable:
            capacity_failures.append(
                f"{target_format}: available={available}, required={required_tokens}, "
                f"records={len(rows)}, exact_subset_reachable={nested_exact_reachable}"
            )
    if capacity_failures:
        raise Day20SourceAdapterError(
            "normalized source capacity failed; records are never duplicated: "
            + "; ".join(capacity_failures)
        )
    return formats


def _exact_format_capacity_reachable(
    pools: Mapping[str, Sequence[Mapping[str, Any]]],
    target_format: str,
    required_tokens: int,
    probe_tokens: int | None,
) -> bool:
    rows = sorted(
        (
            row
            for skill_rows in pools.values()
            for row in skill_rows
            if row.get("target_format") == target_format
        ),
        key=lambda row: (
            text_sha256(f"day20:{target_format}:v1\0{row['sample_id']}"),
            row["sample_id"],
        ),
    )
    weights = [int(row["qwen35_tokenization"]["supervised_tokens"]) for row in rows]
    if sum(weights) < required_tokens:
        return False
    try:
        if probe_tokens is None:
            exact_subset_indices(weights, required_tokens)
        else:
            probe_indices = exact_subset_indices(weights, probe_tokens)
            remaining = [
                weight for index, weight in enumerate(weights) if index not in probe_indices
            ]
            exact_subset_indices(remaining, required_tokens - probe_tokens)
    except Day20ContractError:
        return False
    return True


def build_normalized_pools(
    *,
    source_files: Mapping[str, Sequence[SourceFile]],
    template: Any,
    config: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rejection_counts: Counter[str] = Counter()
    file_hashes: dict[Path, str] = {}
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    seen_messages: set[str] = set()
    seen_prompts: set[str] = set()
    seen_source_content: set[str] = set()
    source_stats: dict[str, dict[str, Any]] = {}

    for key in SOURCE_SPECS:
        files = list(source_files.get(key, ()))
        if not files:
            raise Day20SourceAdapterError(f"no resolved files for pinned source {key}")
        candidates: list[tuple[dict[str, Any], Mapping[str, Any], SourceFile, int]] = []
        for source_file in files:
            file_hashes[source_file.path] = file_sha256(source_file.path)
            transformed = _transform_file(source_file, rejection_counts)
            if file_sha256(source_file.path) != file_hashes[source_file.path]:
                raise Day20SourceAdapterError(
                    f"source file changed while it was being read: {source_file.path}"
                )
            for candidate, source_record, _, row_index in transformed:
                candidates.append((candidate, source_record, source_file, row_index))
        candidates.sort(
            key=lambda item: (
                item[2].variant != "sanitized" if key == "mbpp" else False,
                *_candidate_rank(item[0], _source_content_hash(item[1])),
            )
        )
        limit = _candidate_limit(config, key)
        if limit is not None and len(candidates) > limit:
            rejection_counts[f"{key}:stable_candidate_limit"] += len(candidates) - limit
            candidates = candidates[:limit]

        accepted_before = sum(len(rows) for rows in pools.values())
        tokens_before = sum(
            int(row["qwen35_tokenization"]["supervised_tokens"])
            for rows in pools.values()
            for row in rows
        )
        supplement_already_unneeded = key == "tulu_code" and _exact_format_capacity_reachable(
            pools,
            "code_continuation",
            MAIN_TOKENS_BY_FORMAT["code_continuation"],
            PROBE_TOKENS_BY_FORMAT["code_continuation"],
        )
        if supplement_already_unneeded:
            rejection_counts["tulu_code:capacity_fill_not_needed"] += len(candidates)
            candidates = []
        for candidate_index, (candidate, source_record, source_file, row_index) in enumerate(candidates):
            candidate = canonicalize_candidate_for_audit(candidate)
            source_hash = _source_content_hash(source_record)
            prompt_hash = object_sha256(candidate["messages"][:-1])
            messages_hash = object_sha256(candidate["messages"])
            duplicate_reason = None
            if candidate["sample_id"] in seen_ids:
                duplicate_reason = "sample_id"
            elif messages_hash in seen_messages:
                duplicate_reason = "messages"
            elif prompt_hash in seen_prompts:
                duplicate_reason = "prompt"
            elif source_hash in seen_source_content:
                duplicate_reason = "source_content"
            if duplicate_reason:
                rejection_counts[f"{key}:duplicate_{duplicate_reason}"] += 1
                continue
            try:
                tokenization = audit_messages(candidate["messages"], template)
            except SourceRecordRejected as error:
                rejection_counts[f"{key}:{error}"] += 1
                continue
            row = dict(candidate)
            row["qwen35_tokenization"] = tokenization
            row["reference_evidence"] = build_reference_evidence(
                source_key=key,
                candidate=candidate,
                source_record=source_record,
                source_content_sha256=source_hash,
            )
            row["source_lineage"] = {
                "source": source_file.spec.source,
                "revision": source_file.spec.revision,
                "split": source_file.split,
                "variant": source_file.variant,
                "adapter": source_file.spec.adapter,
                "license": source_file.spec.license,
                "source_file": source_file.relative_name,
                "source_file_sha256": file_hashes[source_file.path],
                "source_content_sha256": source_hash,
                "source_id": candidate["parent_id"],
                "source_row_index": row_index,
                "dataset_code_execution": False,
            }
            try:
                normalize_and_validate_record(
                    row,
                    expected_skill=str(row["skill"]),
                    max_length=int(QWEN35_TEMPLATE_CONTRACT["max_length"]),
                )
            except Day20ContractError as error:
                raise Day20SourceAdapterError(
                    f"adapter emitted invalid row {row['sample_id']}: {error}"
                ) from error
            seen_ids.add(str(row["sample_id"]))
            seen_messages.add(messages_hash)
            seen_prompts.add(prompt_hash)
            seen_source_content.add(source_hash)
            pools[str(row["skill"])].append(row)
            if key == "tulu_code" and _exact_format_capacity_reachable(
                pools,
                "code_continuation",
                MAIN_TOKENS_BY_FORMAT["code_continuation"],
                PROBE_TOKENS_BY_FORMAT["code_continuation"],
            ):
                rejection_counts["tulu_code:capacity_fill_not_needed"] += (
                    len(candidates) - candidate_index - 1
                )
                break

        accepted_after = sum(len(rows) for rows in pools.values())
        tokens_after = sum(
            int(row["qwen35_tokenization"]["supervised_tokens"])
            for rows in pools.values()
            for row in rows
        )
        source_stats[key] = {
            "source": SOURCE_SPECS[key].source,
            "revision": SOURCE_SPECS[key].revision,
            "files": [
                {
                    "path": source_file.relative_name,
                    "sha256": file_hashes[source_file.path],
                    "split": source_file.split,
                    "variant": source_file.variant,
                }
                for source_file in files
            ],
            "accepted_records": accepted_after - accepted_before,
            "accepted_supervised_tokens": tokens_after - tokens_before,
        }

    expected_skills = {"general", "math", "finance", "code"}
    if set(pools) != expected_skills:
        raise Day20SourceAdapterError(
            f"normalized pools do not cover four skills: {sorted(pools)}"
        )
    for rows in pools.values():
        rows.sort(key=lambda row: (text_sha256(f"day20-source-output-v1\0{row['sample_id']}"), row["sample_id"]))

    formats = validate_pool_capacity(pools)
    metrics = {
        "source_stats": source_stats,
        "formats": formats,
        "rejections": dict(sorted(rejection_counts.items())),
    }
    return dict(pools), metrics


def _write_jsonl_new(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise Day20SourceAdapterError(f"refusing to overwrite adapter artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise Day20SourceAdapterError(f"stale adapter temporary file exists: {temporary}")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise Day20SourceAdapterError(f"refusing to overwrite adapter artifact: {path}")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def fetch_sources(*, config: Mapping[str, Any], output_dir: str | Path) -> Mapping[str, Path]:
    """Resolve pinned bytes and emit four normalized, token-audited JSONL files.

    Minimal config::

        {
          "model": "/local/Qwen3.5-4B-Base/fixed-snapshot",
          "cache_dir": "/local/hf-dataset-cache",
          "local_files_only": false,
          "local_paths": {"tatqa": "/optional/staged/tatqa_dataset_train.json"},
          "local_revisions": {
            "tatqa": "c96247f5077eac447f63527fd3dcfdc58bb56d6a"
          }
        }

    ``local_paths`` may cover all six keys (``mmlu``, ``tulu``, ``gsm8k``,
    ``tatqa``, ``mbpp``, ``tulu_code``) for a completely offline run.  MBPP flat staged files
    use labels such as ``{"sanitized/train": "/path/file.jsonl"}`` so their
    immutable variant and split remain explicit.  Every staged key must also
    attest its exact pinned commit in ``local_revisions``; actual file SHA-256
    values are recomputed and retained in every output row.
    """
    if not isinstance(config, Mapping):
        raise Day20SourceAdapterError("fetch adapter config must be an object")
    model_value = config.get("model", config.get("model_path"))
    if not isinstance(model_value, (str, os.PathLike)):
        raise Day20SourceAdapterError("fetch config requires a local model path")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise Day20SourceAdapterError(f"adapter output directory is not empty: {destination}")

    source_files = resolve_source_files(config)
    resolved_model = Path(model_value).expanduser().resolve()
    template = build_qwen35_template(resolved_model)
    pools, metrics = build_normalized_pools(
        source_files=source_files,
        template=template,
        config=config,
    )
    paths: dict[str, Path] = {}
    for skill in ("general", "math", "finance", "code"):
        path = destination / f"{skill}.normalized.qwen35.jsonl"
        _write_jsonl_new(path, pools[skill])
        paths[skill] = path

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "domain": "day20.qwen35_pinned_source_adapter_manifest",
        "status": "pass",
        "dataset_code_execution": False,
        "implementation": {
            "source_adapter_file_sha256": file_sha256(Path(__file__).resolve()),
            "contract_file_sha256": file_sha256(
                Path(__file__).resolve().with_name("day20_contract.py")
            ),
            "ms_swift_version": importlib.metadata.version("ms-swift"),
            "reference_evidence_verifier_version": REFERENCE_EVIDENCE_VERIFIER_VERSION,
            "reference_evidence_verifier_file_sha256": file_sha256(
                Path(__file__).resolve()
            ),
        },
        "model_path": str(resolved_model),
        "tokenizer_files": _tokenizer_file_identity(resolved_model),
        "template": dict(QWEN35_TEMPLATE_CONTRACT),
        "fixed_sources": {
            key: {
                "source": spec.source,
                "revision": spec.revision,
                "license": spec.license,
                "adapter": spec.adapter,
            }
            for key, spec in SOURCE_SPECS.items()
        },
        "metrics": metrics,
        "outputs": {
            skill: {
                "path": str(path),
                "file_sha256": file_sha256(path),
                "records": len(pools[skill]),
                "supervised_tokens": sum(
                    int(row["qwen35_tokenization"]["supervised_tokens"])
                    for row in pools[skill]
                ),
            }
            for skill, path in paths.items()
        },
    }
    manifest["manifest_sha256"] = object_sha256(manifest)
    _write_json_new(destination / "SOURCE-ADAPTER-MANIFEST.json", manifest)
    return paths


__all__ = [
    "Day20SourceAdapterError",
    "SOURCE_SPECS",
    "audit_messages",
    "audit_normalized_sources",
    "build_reference_evidence",
    "canonicalize_candidate_for_audit",
    "build_normalized_pools",
    "build_qwen35_template",
    "fetch_sources",
    "render_tatqa_context",
    "resolve_source_files",
    "split_mbpp_continuation",
    "token_sequence_sha256",
    "transform_gsm8k",
    "transform_mbpp",
    "transform_mmlu",
    "transform_tatqa_document",
    "transform_tulu_general",
    "transform_tulu_code",
    "validate_pool_capacity",
    "verify_reference_evidence",
]
