#!/usr/bin/env python3
"""Day 19 HumanEval parser supporting complete solutions and continuations."""

from __future__ import annotations

import ast
import re
from collections.abc import Callable
from typing import Any

from qwen35_response_adapter import text_sha256


PARSER_VERSION = "day19-humaneval-solution-or-completion-v3"
COMPOSER_VERSION = "day19-humaneval-dual-shape-composer-v3"
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FULL_FUNCTION_PREFIX_RE = re.compile(
    r"^(?:async[ \t]+def|def)[ \t]+([A-Za-z_][A-Za-z0-9_]*)[ \t]*\("
)


class Day19HumanEvalAdapterError(ValueError):
    """A HumanEval adapter input violates the v2 contract."""


def extract_source_preamble(source_prompt: str, *, entry_point: str) -> str:
    """Return top-level source text before the HumanEval entry-point definition."""

    if not isinstance(source_prompt, str) or not source_prompt.endswith("\n"):
        raise Day19HumanEvalAdapterError("HumanEval source prompt must end with a newline")
    if not isinstance(entry_point, str) or not _IDENTIFIER_RE.fullmatch(entry_point):
        raise Day19HumanEvalAdapterError(f"invalid HumanEval entry point: {entry_point!r}")
    try:
        tree = ast.parse(source_prompt)
    except (SyntaxError, ValueError, TypeError) as error:
        raise Day19HumanEvalAdapterError("HumanEval source prompt is not valid Python") from error
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == entry_point
    ]
    if len(definitions) != 1:
        raise Day19HumanEvalAdapterError(
            f"expected one top-level {entry_point!r} definition, found {len(definitions)}"
        )
    definition = definitions[0]
    start_lines = [definition.lineno]
    start_lines.extend(decorator.lineno for decorator in definition.decorator_list)
    first_line = min(start_lines)
    return "".join(source_prompt.splitlines(keepends=True)[: first_line - 1])


def classify_code_candidate(
    final_text: str,
    *,
    entry_point: str,
    extract_code_completion: Callable[[str], str],
) -> dict[str, Any]:
    """Extract fenced code and classify it without repairing candidate code."""

    if not isinstance(final_text, str):
        raise TypeError("final_text must be a string")
    if not isinstance(entry_point, str) or not _IDENTIFIER_RE.fullmatch(entry_point):
        raise Day19HumanEvalAdapterError(f"invalid HumanEval entry point: {entry_point!r}")
    if not callable(extract_code_completion):
        raise TypeError("extract_code_completion must be callable")

    candidate = extract_code_completion(final_text)
    if not isinstance(candidate, str):
        raise Day19HumanEvalAdapterError("code extractor did not return text")
    fence_removed = candidate != final_text.strip("\r\n")
    syntax_error = None
    standalone_tree = None
    top_level_functions: list[str] = []
    try:
        standalone_tree = ast.parse(candidate)
    except (SyntaxError, ValueError, TypeError) as error:
        syntax_error = type(error).__name__
    else:
        top_level_functions = [
            node.name
            for node in standalone_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

    lexical_function = _FULL_FUNCTION_PREFIX_RE.match(candidate)
    lexical_function_name = lexical_function.group(1) if lexical_function else None
    # A generated continuation is a function body and therefore cannot parse as
    # a standalone Python module.  Treat every standalone-valid module as a
    # solution so assignments/aliases cannot escape the original function and
    # rebind the benchmark entry point.
    candidate_mode = (
        "solution"
        if standalone_tree is not None or lexical_function is not None
        else "completion"
    )
    entry_point_definition_count = top_level_functions.count(entry_point)
    completion_contained = None
    completion_containment_error = None
    if candidate_mode == "completion":
        try:
            wrapper_tree = ast.parse("def __day19_candidate_wrapper__():\n" + candidate)
        except (SyntaxError, ValueError, TypeError) as error:
            completion_contained = False
            completion_containment_error = type(error).__name__
        else:
            completion_contained = (
                len(wrapper_tree.body) == 1
                and isinstance(wrapper_tree.body[0], ast.FunctionDef)
                and wrapper_tree.body[0].name == "__day19_candidate_wrapper__"
                and bool(wrapper_tree.body[0].body)
            )
            if not completion_contained:
                completion_containment_error = "ModuleEscape"
        execution_eligible = bool(completion_contained)
    else:
        execution_eligible = syntax_error is None and entry_point_definition_count == 1
    anomalies: list[str] = []
    if syntax_error is not None and candidate_mode == "solution":
        anomalies.append("candidate_syntax_invalid")
    if candidate_mode == "solution" and entry_point_definition_count == 0:
        anomalies.append("solution_missing_expected_entry_point")
    if entry_point_definition_count > 1:
        anomalies.append("solution_redefines_expected_entry_point")
    if candidate_mode == "completion" and not completion_contained:
        anomalies.append("completion_escapes_function_body")

    return {
        "parser_version": PARSER_VERSION,
        "candidate_mode": candidate_mode,
        "candidate": candidate,
        "candidate_sha256": text_sha256(candidate),
        "markdown_fence_removed": fence_removed,
        "top_level_functions": top_level_functions,
        "lexical_function_name": lexical_function_name,
        "entry_point": entry_point,
        "entry_point_definition_count": entry_point_definition_count,
        "execution_eligible": execution_eligible,
        "completion_contained": completion_contained,
        "completion_containment_error": completion_containment_error,
        "standalone_syntax_valid": syntax_error is None,
        "standalone_syntax_error": syntax_error,
        "anomalies": anomalies,
    }


def compose_humaneval_program(
    *,
    source_prompt: str,
    source_test: str,
    entry_point: str,
    candidate: str,
    candidate_mode: str,
) -> str:
    """Compose the exact program sent to the sandbox; never execute it here."""

    if not isinstance(source_prompt, str) or not source_prompt.endswith("\n"):
        raise Day19HumanEvalAdapterError("HumanEval source prompt must end with a newline")
    if not isinstance(source_test, str):
        raise TypeError("source_test must be a string")
    if not isinstance(candidate, str):
        raise TypeError("candidate must be a string")
    if not isinstance(entry_point, str) or not _IDENTIFIER_RE.fullmatch(entry_point):
        raise Day19HumanEvalAdapterError(f"invalid HumanEval entry point: {entry_point!r}")
    if candidate_mode == "completion":
        candidate_program = source_prompt + candidate
    elif candidate_mode == "solution":
        candidate_program = extract_source_preamble(
            source_prompt, entry_point=entry_point
        ) + candidate
    else:
        raise Day19HumanEvalAdapterError(f"unknown candidate mode: {candidate_mode!r}")
    return candidate_program + "\n" + source_test + f"\ncheck({entry_point})\n"


def composed_syntax_status(program: str) -> dict[str, Any]:
    """Compile for syntax diagnostics without executing candidate code."""

    try:
        compile(program, "<day19-humaneval-candidate>", "exec", dont_inherit=True)
    except (SyntaxError, ValueError, TypeError) as error:
        return {"valid": False, "error": type(error).__name__}
    return {"valid": True, "error": None}
