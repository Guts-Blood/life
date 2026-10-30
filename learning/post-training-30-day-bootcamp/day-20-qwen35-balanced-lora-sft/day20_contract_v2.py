#!/usr/bin/env python3
"""Pure Day 20 v2 target and token-budget contracts.

This module is deliberately separate from :mod:`day20_contract`: v1 artifacts
remain governed by their frozen contract while v2 preparation opts into the
stricter Code continuation grammar defined here.
"""

from __future__ import annotations

import ast
import hashlib
import textwrap
from dataclasses import asdict, dataclass
from typing import Any


V2_CONTRACT_VERSION = "day20.qwen35_candidate_factory_v2"

MAIN_SUPERVISED_TOKENS = 320_000
MAIN_TOKENS_PER_SKILL = 80_000
PROBE_SUPERVISED_TOKENS = 24_000
PROBE_TOKENS_PER_SKILL = 6_000

MAIN_TOKENS_BY_FORMAT = {
    "general_mcq": 31_998,
    "general_instruction": 48_002,
    "math_reasoning": 80_000,
    "finance_value_scale": 80_000,
    "code_continuation": 80_000,
}
PROBE_TOKENS_BY_FORMAT = {
    "general_mcq": 3_000,
    "general_instruction": 3_000,
    "math_reasoning": 6_000,
    "finance_value_scale": 6_000,
    "code_continuation": 6_000,
}

_SPECIAL_MARKERS = (
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<image>",
    "<video>",
    "<audio>",
)
_FORBIDDEN_CONTINUATION_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Global,
    ast.Nonlocal,
)
_ALLOWED_EXPRESSION_STATEMENTS = (
    ast.Await,
    ast.Call,
    ast.NamedExpr,
    ast.Yield,
    ast.YieldFrom,
)


class Day20V2ContractError(ValueError):
    """A Day 20 v2 target or budget invariant failed."""


@dataclass(frozen=True)
class CodeContinuationV2:
    """Canonical Code target and the identities needed for lineage audits."""

    raw: str
    canonical: str
    raw_sha256: str
    canonical_sha256: str
    ast_sha256: str

    def as_evidence(self) -> dict[str, str]:
        """Return a JSON-serializable evidence object."""
        return asdict(self)


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise Day20V2ContractError(f"{label} must be text")
    if not value or not value.strip():
        raise Day20V2ContractError(f"{label} must not be empty")
    if "\x00" in value:
        raise Day20V2ContractError(f"{label} contains a NUL byte")
    if "```" in value:
        raise Day20V2ContractError(f"{label} must not contain Markdown fences")
    if any(marker in value for marker in _SPECIAL_MARKERS):
        raise Day20V2ContractError(f"{label} contains a template/media marker")
    return value


def _trim_blank_edges(value: str) -> str:
    lines = value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(line.rstrip() for line in lines)


def _first_line_indent(value: str) -> str:
    first_line = value.split("\n", 1)[0]
    return first_line[: len(first_line) - len(first_line.lstrip(" \t"))]


def _parse_continuation(value: str) -> tuple[list[ast.stmt], str]:
    """Parse a body fragment without repairing a later module-level escape."""
    first_indent = _first_line_indent(value)
    candidates: list[str] = []
    if first_indent:
        # Canonical four-space input takes this path and preserves multiline
        # string contents.  Wider but internally consistent source indentation
        # gets one deterministic dedent attempt below.
        candidates.append(value)
        dedented = textwrap.dedent(value)
        if dedented != value:
            candidates.append(textwrap.indent(dedented, "    "))
    else:
        # Source adapters may supply an otherwise valid unindented body.  This
        # is a training-time normalization only; strict inference validation
        # calls ``validate_raw_code_continuation``.
        candidates.append(textwrap.indent(value, "    "))

    last_error: SyntaxError | None = None
    for indented in candidates:
        wrapper = f"def __day20_v2_containment__():\n{indented}\n"
        try:
            module = ast.parse(wrapper)
            compile(module, "<day20-v2-continuation>", "exec")
        except SyntaxError as error:
            last_error = error
            continue
        function = module.body[0]
        if not isinstance(function, ast.FunctionDef) or not function.body:
            continue
        return list(function.body), wrapper
    detail = f": {last_error.msg}" if last_error is not None else ""
    raise Day20V2ContractError(
        f"code target is not a contained function-body continuation{detail}"
    ) from last_error


def _validate_body_ast(statements: list[ast.stmt]) -> None:
    for statement in statements:
        for node in ast.walk(statement):
            if isinstance(node, _FORBIDDEN_CONTINUATION_NODES):
                raise Day20V2ContractError(
                    "code continuation contains an import, definition, class, "
                    "or namespace escape"
                )
        if isinstance(statement, ast.Expr) and not isinstance(
            statement.value, _ALLOWED_EXPRESSION_STATEMENTS
        ):
            raise Day20V2ContractError(
                "code continuation contains prose or a non-executable expression"
            )


def _ast_identity(statements: list[ast.stmt]) -> str:
    body = ast.Module(body=statements, type_ignores=[])
    return _text_sha256(ast.dump(body, annotate_fields=True, include_attributes=False))


def _validate_prefix_binding(code_prefix: str, canonical: str, body: list[ast.stmt]) -> None:
    prefix = _trim_blank_edges(_require_text(code_prefix, "code prefix"))
    combined = f"{prefix}\n{canonical}\n"
    try:
        module = ast.parse(combined)
        compile(module, "<day20-v2-prefix-plus-continuation>", "exec")
    except SyntaxError as error:
        raise Day20V2ContractError(
            f"code prefix plus continuation is not valid Python: {error.msg}"
        ) from error
    if not module.body or not isinstance(
        module.body[-1], (ast.FunctionDef, ast.AsyncFunctionDef)
    ):
        raise Day20V2ContractError(
            "code continuation is not bound to the final top-level function"
        )
    target_body = module.body[-1].body
    if len(target_body) < len(body):
        raise Day20V2ContractError("code continuation escaped its target function")
    expected = [ast.dump(node, include_attributes=False) for node in body]
    observed = [
        ast.dump(node, include_attributes=False) for node in target_body[-len(body) :]
    ]
    if observed != expected:
        raise Day20V2ContractError("code continuation escaped its target function")


def _require_raw_four_space_contract(raw: str) -> None:
    if "\r" in raw or raw.startswith("\n"):
        raise Day20V2ContractError(
            "raw code continuation must begin at byte 0 with exactly four spaces"
        )
    first_line = raw.split("\n", 1)[0]
    if not first_line.strip() or _first_line_indent(raw) != "    ":
        raise Day20V2ContractError(
            "raw code continuation must begin at byte 0 with exactly four spaces"
        )
    if any("\t" in line[: len(line) - len(line.lstrip(" \t"))] for line in raw.splitlines()):
        raise Day20V2ContractError("raw code continuation indentation must use spaces")


def canonicalize_code_continuation(
    target: Any,
    code_prefix: Any,
    *,
    require_raw_contract: bool = False,
) -> CodeContinuationV2:
    """Return the v2 AST-canonical continuation and its three identities.

    Training preparation may normalize a valid zero- or wider-indented source
    fragment.  Evaluation must set ``require_raw_contract=True`` (or call
    :func:`validate_raw_code_continuation`) so a model output is never repaired
    before its primary validity decision.
    """
    raw = _require_text(target, "code target")
    if require_raw_contract:
        _require_raw_four_space_contract(raw)
    normalized = _trim_blank_edges(raw)
    if not normalized:
        raise Day20V2ContractError("code target must not be empty")

    body, _ = _parse_continuation(normalized)
    _validate_body_ast(body)
    canonical_body = "\n".join(ast.unparse(statement) for statement in body)
    canonical = textwrap.indent(canonical_body, "    ")
    if _first_line_indent(canonical) != "    ":
        raise Day20V2ContractError("canonical continuation lost its four-space contract")
    _validate_prefix_binding(code_prefix, canonical, body)

    return CodeContinuationV2(
        raw=raw,
        canonical=canonical,
        raw_sha256=_text_sha256(raw),
        canonical_sha256=_text_sha256(canonical),
        ast_sha256=_ast_identity(body),
    )


def validate_raw_code_continuation(
    target: Any, code_prefix: Any
) -> CodeContinuationV2:
    """Validate an inference output without applying indentation repair."""
    return canonicalize_code_continuation(
        target, code_prefix, require_raw_contract=True
    )
