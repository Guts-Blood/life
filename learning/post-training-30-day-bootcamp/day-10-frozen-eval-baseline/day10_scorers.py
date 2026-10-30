#!/usr/bin/env python3
"""Deterministic extractors and scorers for the four Day 10 eval slices.

This module deliberately does not execute generated code.  HumanEval completions
are only extracted and marked for later execution in a pinned sandbox.
"""

from __future__ import annotations

import json
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Mapping


SCORER_REGISTRY_VERSION = "day10-scorer-registry-v3"

# Keep this object JSON-serializable: it is protocol metadata, not a dispatch table.
SCORER_REGISTRY = {
    "registry_version": SCORER_REGISTRY_VERSION,
    "slices": {
        "general": {
            "dataset": "MMLU",
            "extractor_version": "mmlu_option_extractor_v2",
            "scorer_version": "mmlu_exact_option_v2",
        },
        "math": {
            "dataset": "GSM8K",
            "extractor_version": "gsm8k_final_number_extractor_v1",
            "canonicalizer_version": "exact-rational-number-v1",
            "scorer_version": "gsm8k_numeric_exact_v1",
        },
        "finance": {
            "dataset": "TAT-QA",
            "extractor_version": "tatqa_final_answer_extractor_v3",
            "reference_parser_version": "tatqa-reference-parser-v1",
            "canonicalizer_version": "tatqa-conservative-normalizer-v1",
            "scorer_version": "tatqa_normalized_exact_v3",
        },
        "code": {
            "dataset": "HumanEval",
            "extractor_version": "human_eval_completion_extractor_v1",
            "scorer_version": "human_eval_pass_at_1_sandbox_v1",
            "execution_policy": "sandbox_required",
        },
    },
}


_MMLU_EXPLICIT_RE = re.compile(
    r"(?i:\b(?:final\s+answer|answer|choice|option)\b)"
    r"\s*(?:(?i:is)\s*)?(?:[:=\-]\s*)?"
    r"(?:\*\*)?\s*[\(\[<]?([A-Z])(?:[\)\]>]|\*\*)?"
    r"(?=\s|[.,;:!?*]|$)"
)
_MMLU_BOXED_RE = re.compile(r"\\boxed\{\s*([A-Z])\s*\}")
_MMLU_LINE_RE = re.compile(
    r"^(?:[-*>]\s*)?(?:\*\*)?\s*[\(\[<]?([A-Z])"
    r"(?:[\)\]>]|\*\*)?\s*[.:]?\s*$"
)
_MMLU_REFERENCE_RE = re.compile(
    r"^\s*[\(\[<]?([A-Z])(?:[\)\]>]|\s*[.:\-])(?=\s|$)"
)


def extract_mmlu_option(text: str) -> str | None:
    """Extract an explicitly stated uppercase MMLU option label."""

    if not isinstance(text, str) or not text.strip():
        return None
    matches: list[tuple[int, str]] = []
    matches.extend((match.start(), match.group(1)) for match in _MMLU_BOXED_RE.finditer(text))
    matches.extend(
        (match.start(), match.group(1)) for match in _MMLU_EXPLICIT_RE.finditer(text)
    )
    if matches:
        return max(matches, key=lambda item: item[0])[1]

    for line in reversed(text.splitlines()):
        if not line.strip():
            continue
        match = _MMLU_LINE_RE.fullmatch(line.strip())
        return match.group(1) if match else None
    return None


def parse_mmlu_reference(text: str) -> str | None:
    """Parse a reference label without loosening prediction extraction."""

    if not isinstance(text, str) or not text.strip():
        return None
    match = _MMLU_REFERENCE_RE.match(text)
    return match.group(1) if match else extract_mmlu_option(text)


_UNSIGNED_DECIMAL = r"(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?"
_SIGNED_DECIMAL = rf"[+\-]?{_UNSIGNED_DECIMAL}"
_NUMBER_TOKEN_RE = re.compile(
    rf"(?<![\w.])(?:[$£€]\s*)?"
    rf"(?:\(\s*{_UNSIGNED_DECIMAL}\s*\)|{_SIGNED_DECIMAL})"
    rf"(?:\s*/\s*{_SIGNED_DECIMAL})?\s*%?(?![\w.])"
)
_BOXED_VALUE_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_FINAL_ANSWER_RE = re.compile(
    r"(?i:\b(?:final\s+answer|answer)\b)\s*(?:(?i:is)\s*)?(?:[:=\-]\s*)?([^\n]+)"
)


def _decimal_fraction(value: str) -> Fraction:
    try:
        return Fraction(Decimal(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"invalid decimal: {value!r}") from error


def canonicalize_number(value: Any) -> str | None:
    """Return an exact rational representation for a standalone number."""

    if isinstance(value, bool) or value is None:
        return None
    text = unicodedata.normalize("NFKC", str(value)).strip()
    if not text:
        return None
    text = re.sub(r"^[\$£€]\s*", "", text)
    text = text.strip()
    negative_parentheses = text.startswith("(") and text.endswith(")")
    if negative_parentheses:
        text = text[1:-1].strip()
        text = re.sub(r"^[\$£€]\s*", "", text)
    if text.endswith("%"):
        text = text[:-1].strip()
    text = text.replace(",", "").replace(" ", "")
    if not text:
        return None

    try:
        if "/" in text:
            numerator_text, denominator_text = text.split("/", 1)
            numerator = _decimal_fraction(numerator_text)
            denominator = _decimal_fraction(denominator_text)
            if denominator == 0:
                return None
            number = numerator / denominator
        else:
            number = _decimal_fraction(text)
    except ValueError:
        return None
    if negative_parentheses:
        number = -number
    if number == 0:
        number = Fraction(0)
    if number.denominator == 1:
        return str(number.numerator)
    return f"{number.numerator}/{number.denominator}"


def _numbers_in(text: str) -> list[str]:
    return [match.group(0).strip() for match in _NUMBER_TOKEN_RE.finditer(text)]


def extract_gsm8k_final_number(text: str) -> str | None:
    """Extract the final numeric answer using frozen, ordered fallbacks."""

    if not isinstance(text, str) or not text.strip():
        return None

    if "####" in text:
        candidates = _numbers_in(text.rsplit("####", 1)[1])
        if candidates:
            return candidates[-1]

    boxed_candidates = [
        match.group(1).strip()
        for match in _BOXED_VALUE_RE.finditer(text)
        if canonicalize_number(match.group(1)) is not None
    ]
    if boxed_candidates:
        return boxed_candidates[-1]

    explicit_candidates: list[str] = []
    for match in _FINAL_ANSWER_RE.finditer(text):
        explicit_candidates.extend(_numbers_in(match.group(1)))
    if explicit_candidates:
        return explicit_candidates[-1]

    candidates = _numbers_in(text)
    return candidates[-1] if candidates else None


def _exact_result(
    *,
    parsed_answer: Any,
    canonical_answer: Any,
    canonical_reference: Any,
) -> dict[str, Any]:
    if canonical_answer is None:
        return {
            "parsed_answer": parsed_answer,
            "canonical_answer": None,
            "canonical_reference": canonical_reference,
            "parse_status": "parse_error",
            "score_status": "ok",
            "score": 0.0,
            "error_type": "parse_error",
        }
    correct = canonical_answer == canonical_reference
    return {
        "parsed_answer": parsed_answer,
        "canonical_answer": canonical_answer,
        "canonical_reference": canonical_reference,
        "parse_status": "ok",
        "score_status": "ok",
        "score": 1.0 if correct else 0.0,
        "error_type": None if correct else "wrong_answer",
    }


def score_mmlu(raw_output: str, reference: str) -> dict[str, Any]:
    """Score an MMLU response by exact option-label match."""

    canonical_reference = parse_mmlu_reference(reference)
    if canonical_reference is None:
        raise ValueError(f"invalid MMLU reference: {reference!r}")
    parsed_answer = extract_mmlu_option(raw_output)
    return _exact_result(
        parsed_answer=parsed_answer,
        canonical_answer=parsed_answer,
        canonical_reference=canonical_reference,
    )


def score_gsm8k(raw_output: str, reference: str) -> dict[str, Any]:
    """Score a GSM8K response by exact canonical final-number match."""

    reference_number = extract_gsm8k_final_number(reference)
    canonical_reference = canonicalize_number(reference_number)
    if canonical_reference is None:
        raise ValueError("GSM8K reference has no valid final number")
    parsed_answer = extract_gsm8k_final_number(raw_output)
    return _exact_result(
        parsed_answer=parsed_answer,
        canonical_answer=canonicalize_number(parsed_answer),
        canonical_reference=canonical_reference,
    )


_TATQA_ANSWER_TYPES = {"span", "multi-span", "arithmetic", "count"}
_TATQA_SCALES = {"", "thousand", "million", "billion", "percent"}
_SCALE_SUFFIXES = (
    ("percent", re.compile(r"(?i)\s*(?:%|percent|percentage)\s*$")),
    ("thousand", re.compile(r"(?i)\s*(?:thousand|thousands)\s*$")),
    ("million", re.compile(r"(?i)\s*(?:million|millions)\s*$")),
    ("billion", re.compile(r"(?i)\s*(?:billion|billions)\s*$")),
)
_FINAL_SEGMENT_RE = re.compile(
    r"(?i:\b(?:final\s+answer|answer)\b)(?:\s+(?i:is)\s*|[:=\-]\s*)"
)
_MARKDOWN_PAIR_RE = re.compile(r"^(?:\*\*|__|`)(.*)(?:\*\*|__|`)$", re.DOTALL)


def parse_tatqa_reference(reference: str | Mapping[str, Any]) -> dict[str, Any]:
    """Parse and validate the structured TAT-QA reference used by Day 09."""

    if isinstance(reference, str):
        try:
            payload = json.loads(reference)
        except json.JSONDecodeError as error:
            raise ValueError("TAT-QA reference is not valid JSON") from error
    elif isinstance(reference, Mapping):
        payload = dict(reference)
    else:
        raise ValueError("TAT-QA reference must be a JSON object or mapping")
    if not isinstance(payload, dict):
        raise ValueError("TAT-QA reference must decode to an object")

    answer_type = payload.get("answer_type")
    scale = payload.get("scale")
    if answer_type not in _TATQA_ANSWER_TYPES:
        raise ValueError(f"unsupported TAT-QA answer_type: {answer_type!r}")
    if not isinstance(scale, str) or scale.casefold().strip() not in _TATQA_SCALES:
        raise ValueError(f"unsupported TAT-QA scale: {scale!r}")
    if "answer" not in payload:
        raise ValueError("TAT-QA reference is missing answer")

    answer = payload["answer"]
    if isinstance(answer, list):
        answers = answer
    else:
        answers = [answer]
    if not answers or any(
        isinstance(item, (dict, list, bool)) or item is None for item in answers
    ):
        raise ValueError("TAT-QA answer must contain scalar values")
    if answer_type == "multi-span" and not isinstance(answer, list):
        raise ValueError("TAT-QA multi-span answer must be a list")
    if answer_type in {"arithmetic", "count"} and len(answers) != 1:
        raise ValueError(f"TAT-QA {answer_type} answer must be scalar")

    return {
        "answer": answer,
        "answers": list(answers),
        "answer_type": answer_type,
        "scale": scale.casefold().strip(),
        "derivation": payload.get("derivation", ""),
    }


def _unwrap_markdown(text: str) -> str:
    value = text.strip()
    match = _MARKDOWN_PAIR_RE.fullmatch(value)
    if match:
        value = match.group(1).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()
    return value


def _tatqa_final_segment(raw_output: str) -> str | None:
    if not isinstance(raw_output, str) or not raw_output.strip():
        return None
    text = raw_output.strip()
    markers = list(_FINAL_SEGMENT_RE.finditer(text))
    for marker in reversed(markers):
        prefix = text[max(0, marker.start() - 40) : marker.start()].casefold()
        if "end with exactly" in prefix:
            continue
        return _unwrap_markdown(text[marker.end() :].strip()) or None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return _unwrap_markdown(lines[-1]) if lines else None


def extract_tatqa_answer(
    raw_output: str, reference: str | Mapping[str, Any]
) -> Any | None:
    """Extract a scalar answer, or a JSON array for multi-span references."""

    spec = parse_tatqa_reference(reference)
    text = raw_output.strip() if isinstance(raw_output, str) else ""
    if not text:
        return None

    if spec["answer_type"] == "multi-span":
        candidates = [text]
        segment = _tatqa_final_segment(text)
        if segment and segment != text:
            candidates.append(segment)
        for candidate in reversed(candidates):
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(value, list):
                return value
        if segment:
            bullet_values = [
                re.sub(r"^\s*(?:[-*]|\d+[.)])\s+", "", line).strip()
                for line in segment.splitlines()
                if re.match(r"^\s*(?:[-*]|\d+[.)])\s+", line)
            ]
            if len(bullet_values) == len(spec["answers"]):
                return bullet_values
            semicolon_values = [item.strip() for item in segment.split(";")]
            if len(semicolon_values) == len(spec["answers"]):
                return semicolon_values
            references_are_numeric = all(
                canonicalize_number(_split_scale(str(item))[0]) is not None
                for item in spec["answers"]
            )
            numeric_values = _numbers_in(segment)
            if references_are_numeric and len(numeric_values) == len(spec["answers"]):
                return numeric_values
        return None

    segment = _tatqa_final_segment(text)
    if segment is None:
        return None
    try:
        value = json.loads(segment)
    except json.JSONDecodeError:
        return segment
    return value if not isinstance(value, (dict, list, bool)) else None


def _split_scale(value: str) -> tuple[str, str]:
    for scale, pattern in _SCALE_SUFFIXES:
        match = pattern.search(value)
        if match:
            return value[: match.start()].strip(), scale
    return value.strip(), ""


def _normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    text = _unwrap_markdown(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" \t\r\n.,;:!?")


def _canonicalize_tatqa_scalar(
    value: Any,
    *,
    declared_scale: str,
    numeric_required: bool,
    is_reference: bool,
) -> str | None:
    if isinstance(value, bool) or value is None or isinstance(value, (dict, list)):
        return None
    text = unicodedata.normalize("NFKC", str(value)).strip()
    if not text and not numeric_required:
        return "text:"
    unscaled, explicit_scale = _split_scale(_unwrap_markdown(text))
    canonical_number = canonicalize_number(unscaled)
    if canonical_number is not None:
        effective_scale = explicit_scale or (declared_scale if is_reference else "")
        return f"number:{canonical_number}|scale:{effective_scale}"
    if numeric_required:
        return None
    return f"text:{_normalize_text(text)}"


def canonicalize_tatqa_answers(
    values: list[Any], *, answer_type: str, scale: str, is_reference: bool = False
) -> tuple[str, ...] | None:
    """Canonicalize TAT-QA values without fuzzy or partial-credit matching."""

    numeric_required = answer_type in {"arithmetic", "count"}
    normalized = [
        _canonicalize_tatqa_scalar(
            value,
            declared_scale=scale,
            numeric_required=numeric_required,
            is_reference=is_reference,
        )
        for value in values
    ]
    if any(item is None for item in normalized):
        return None
    result = tuple(item for item in normalized if item is not None)
    return tuple(sorted(result)) if answer_type == "multi-span" else result


def score_tatqa(
    raw_output: str, reference: str | Mapping[str, Any]
) -> dict[str, Any]:
    """Apply conservative normalized exact match to a TAT-QA response."""

    spec = parse_tatqa_reference(reference)
    canonical_reference = canonicalize_tatqa_answers(
        spec["answers"],
        answer_type=spec["answer_type"],
        scale=spec["scale"],
        is_reference=True,
    )
    if canonical_reference is None:
        raise ValueError("TAT-QA reference cannot be canonicalized")

    parsed_answer = extract_tatqa_answer(raw_output, spec)
    if parsed_answer is None:
        prediction_values: list[Any] = []
        canonical_answer = None
    elif isinstance(parsed_answer, list):
        prediction_values = parsed_answer
        canonical_answer = canonicalize_tatqa_answers(
            prediction_values,
            answer_type=spec["answer_type"],
            scale=spec["scale"],
        )
    else:
        prediction_values = [parsed_answer]
        canonical_answer = canonicalize_tatqa_answers(
            prediction_values,
            answer_type=spec["answer_type"],
            scale=spec["scale"],
        )
    return _exact_result(
        parsed_answer=parsed_answer,
        canonical_answer=canonical_answer,
        canonical_reference=canonical_reference,
    )


_FENCED_CODE_RE = re.compile(
    r"```[ \t]*([A-Za-z0-9_+\-]*)[ \t]*\r?\n(.*?)```", re.DOTALL
)


def extract_code_completion(raw_output: str) -> str:
    """Extract a fenced code block or return the raw completion unchanged."""

    if not isinstance(raw_output, str):
        raise TypeError("raw_output must be a string")
    blocks = list(_FENCED_CODE_RE.finditer(raw_output))
    if blocks:
        python_blocks = [
            match for match in blocks if match.group(1).casefold() in {"python", "py"}
        ]
        selected = (python_blocks or blocks)[0]
        return selected.group(2).strip("\r\n")
    return raw_output.strip("\r\n")


def score_code(raw_output: str, reference: Any = None) -> dict[str, Any]:
    """Return a sandbox handoff; generated code is never executed here."""

    del reference
    completion = extract_code_completion(raw_output)
    return {
        "parsed_answer": completion,
        "canonical_answer": None,
        "canonical_reference": None,
        "parse_status": "ok",
        "score_status": "sandbox_required",
        "score": None,
        "error_type": "sandbox_required",
    }


_SCORERS = {
    "general": score_mmlu,
    "math": score_gsm8k,
    "finance": score_tatqa,
    "code": score_code,
}


def score_prediction(slice_name: str, raw_output: str, reference: Any) -> dict[str, Any]:
    """Dispatch one raw prediction through the frozen four-slice registry."""

    try:
        scorer = _SCORERS[slice_name]
    except KeyError as error:
        raise ValueError(f"unknown Day 10 slice: {slice_name!r}") from error
    return scorer(raw_output, reference)
