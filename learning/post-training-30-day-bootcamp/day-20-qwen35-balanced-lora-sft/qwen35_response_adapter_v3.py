#!/usr/bin/env python3
"""Token-boundary-only response capture for Day 20 v2 Qwen3.5 eval.

The decoded generated token IDs are the sole answer boundary.  Message content
is retained only as diagnostic evidence; it is never trimmed, prefix-stripped,
or used to repair Markdown fences.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


ADAPTER_VERSION = "qwen35-response-boundary-v3"
NON_THINKING_PREFIX = "<think>\n\n</think>\n\n"
BOUNDARY_SOURCE = "generated_token_ids_decode"


class Qwen35ResponseAdapterV3Error(ValueError):
    """Response evidence is malformed or internally inconsistent."""


def text_sha256(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("value must be a string")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validated_token_ids(token_ids: Sequence[int], name: str) -> tuple[int, ...]:
    if isinstance(token_ids, (str, bytes)) or not isinstance(token_ids, Sequence):
        raise TypeError(f"{name} must be a sequence of integers")
    values = tuple(token_ids)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values
    ):
        raise TypeError(f"{name} must contain non-negative integers")
    return values


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    values = _validated_token_ids(token_ids, "token_ids")
    payload = json.dumps(values, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _message_relation(message_content: str, generated_only_text: str) -> str:
    if message_content == generated_only_text:
        return "exact_generated_decode"
    if message_content == NON_THINKING_PREFIX + generated_only_text:
        return "known_template_reconstruction"
    return "divergent_diagnostic_only"


def adapt_generated_token_decode(
    message_content: str,
    *,
    generated_token_ids: Sequence[int],
    generated_only_text: str,
) -> dict[str, Any]:
    """Return an auditable answer boundary without changing decoded text.

    ``generated_only_text`` must be the direct decode of
    ``generated_token_ids`` performed by the caller's frozen template.  Token
    IDs cannot be decoded in this pure module, so both are retained and bound
    by hashes.  A message-content mismatch is diagnostic, not an alternative
    answer boundary.
    """

    if not isinstance(message_content, str):
        raise TypeError("message_content must be a string")
    if not isinstance(generated_only_text, str):
        raise TypeError("generated_only_text must be a string")
    ids = _validated_token_ids(generated_token_ids, "generated_token_ids")
    if not ids and generated_only_text:
        raise Qwen35ResponseAdapterV3Error(
            "an empty generated-token sequence cannot decode to non-empty text"
        )
    return {
        "adapter_version": ADAPTER_VERSION,
        "boundary_source": BOUNDARY_SOURCE,
        "message_content_sha256": text_sha256(message_content),
        "message_content_relation": _message_relation(
            message_content, generated_only_text
        ),
        "generated_token_ids_sha256": token_ids_sha256(ids),
        "generated_token_count": len(ids),
        "generated_only_text_sha256": text_sha256(generated_only_text),
        "operations": [],
        "trimming_applied": False,
        "fence_repair_applied": False,
        "final_text": generated_only_text,
        "final_text_sha256": text_sha256(generated_only_text),
    }


def validate_response_capture(
    capture: Mapping[str, Any],
    *,
    raw_output: str,
    generation: Mapping[str, Any],
) -> str:
    """Recompute a complete v3 capture and return the authoritative answer."""

    if not isinstance(capture, Mapping):
        raise Qwen35ResponseAdapterV3Error("response capture must be an object")
    if capture.get("method") != "ms_swift_return_details":
        raise Qwen35ResponseAdapterV3Error("unexpected response capture method")
    message_content = capture.get("message_content")
    generated_text = capture.get("generated_only_text")
    prompt_ids = capture.get("prompt_token_ids")
    generated_ids = capture.get("generated_token_ids")
    if not isinstance(message_content, str) or not isinstance(generated_text, str):
        raise Qwen35ResponseAdapterV3Error("response text evidence is missing")
    try:
        validated_prompt_ids = _validated_token_ids(
            prompt_ids, "prompt_token_ids"  # type: ignore[arg-type]
        )
        validated_generated_ids = _validated_token_ids(
            generated_ids, "generated_token_ids"  # type: ignore[arg-type]
        )
    except TypeError as error:
        raise Qwen35ResponseAdapterV3Error(str(error)) from error
    if capture.get("message_content_sha256") != text_sha256(message_content):
        raise Qwen35ResponseAdapterV3Error("message content hash mismatch")
    if capture.get("prompt_token_ids_sha256") != token_ids_sha256(
        validated_prompt_ids
    ):
        raise Qwen35ResponseAdapterV3Error("prompt token ID hash mismatch")
    if capture.get("generated_token_ids_sha256") != token_ids_sha256(
        validated_generated_ids
    ):
        raise Qwen35ResponseAdapterV3Error("generated token ID hash mismatch")
    if capture.get("generated_only_text_sha256") != text_sha256(generated_text):
        raise Qwen35ResponseAdapterV3Error("generated decode hash mismatch")
    expected_adapter = adapt_generated_token_decode(
        message_content,
        generated_token_ids=validated_generated_ids,
        generated_only_text=generated_text,
    )
    if capture.get("response_adapter") != expected_adapter:
        raise Qwen35ResponseAdapterV3Error("response adapter evidence drifted")
    if not isinstance(raw_output, str) or raw_output != generated_text:
        raise Qwen35ResponseAdapterV3Error(
            "raw output is not the exact generated-token decode"
        )
    if not isinstance(generation, Mapping):
        raise Qwen35ResponseAdapterV3Error("generation metadata must be an object")
    prompt_tokens = generation.get("prompt_tokens")
    completion_tokens = generation.get("completion_tokens")
    if (
        isinstance(prompt_tokens, bool)
        or not isinstance(prompt_tokens, int)
        or prompt_tokens != len(validated_prompt_ids)
    ):
        raise Qwen35ResponseAdapterV3Error("prompt token count mismatch")
    if (
        isinstance(completion_tokens, bool)
        or not isinstance(completion_tokens, int)
        or completion_tokens != len(validated_generated_ids)
    ):
        raise Qwen35ResponseAdapterV3Error("completion token count mismatch")
    return generated_text
