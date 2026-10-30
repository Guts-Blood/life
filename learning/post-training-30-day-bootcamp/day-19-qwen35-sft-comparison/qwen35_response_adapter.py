#!/usr/bin/env python3
"""Auditable Qwen3.5 response-boundary handling for Day 19 evaluation.

The ms-swift Qwen3.5 template prepends its response prefix when decoding
generated token IDs.  That reconstructed message content is useful for chat,
but the prefix is not part of the generated answer.  This module removes only
the one exact non-thinking prefix used by the pinned Day 19 template.  It does
not parse Markdown, benchmark answers, or arbitrary thinking blocks.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any


ADAPTER_VERSION = "qwen35-response-boundary-v2"
NON_THINKING_PREFIX = "<think>\n\n</think>\n\n"


class Qwen35ResponseAdapterError(ValueError):
    """The supplied response evidence is internally inconsistent."""


def text_sha256(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("value must be a string")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    values = _validated_token_ids(token_ids, "token_ids")
    payload = json.dumps(values, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _validated_token_ids(token_ids: Sequence[int], name: str) -> tuple[int, ...]:
    if isinstance(token_ids, (str, bytes)) or not isinstance(token_ids, Sequence):
        raise TypeError(f"{name} must be a sequence of integers")
    values = tuple(token_ids)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise TypeError(f"{name} must contain non-negative integers")
    return values


def slice_generated_token_ids(
    prompt_token_ids: Sequence[int], output_token_ids: Sequence[int]
) -> tuple[int, ...]:
    """Return generated-only IDs after proving the output contains the prompt.

    This is for direct ``model.generate`` outputs.  ms-swift's
    ``ChatCompletionResponseChoice.token_ids`` are already generated-only and
    should be recorded directly instead.
    """

    prompt = _validated_token_ids(prompt_token_ids, "prompt_token_ids")
    output = _validated_token_ids(output_token_ids, "output_token_ids")
    if len(output) < len(prompt):
        raise Qwen35ResponseAdapterError("output token IDs are shorter than the prompt")
    if output[: len(prompt)] != prompt:
        raise Qwen35ResponseAdapterError("output token IDs do not start with the prompt")
    return output[len(prompt) :]


def validate_ms_swift_response_capture(
    capture: dict[str, Any], *, message_content: str, generation: dict[str, Any]
) -> str:
    """Validate a complete ms-swift ``return_details`` evidence bundle."""

    if not isinstance(capture, dict):
        raise Qwen35ResponseAdapterError("response capture must be an object")
    if capture.get("method") != "ms_swift_return_details":
        raise Qwen35ResponseAdapterError("unexpected response capture method")
    prompt_ids = _validated_token_ids(
        capture.get("prompt_token_ids"), "prompt_token_ids"
    )
    generated_ids = _validated_token_ids(
        capture.get("generated_token_ids"), "generated_token_ids"
    )
    if capture.get("prompt_token_ids_sha256") != token_ids_sha256(prompt_ids):
        raise Qwen35ResponseAdapterError("prompt token ID hash mismatch")
    if capture.get("generated_token_ids_sha256") != token_ids_sha256(generated_ids):
        raise Qwen35ResponseAdapterError("generated token ID hash mismatch")
    generated_text = capture.get("generated_only_text")
    if not isinstance(generated_text, str):
        raise Qwen35ResponseAdapterError("generated-only decode is missing")
    if capture.get("generated_only_text_sha256") != text_sha256(generated_text):
        raise Qwen35ResponseAdapterError("generated-only decode hash mismatch")
    if not isinstance(message_content, str):
        raise Qwen35ResponseAdapterError("message content must be text")
    adapted = adapt_qwen35_text(
        message_content, generated_only_text=generated_text
    )
    expected_verification = {
        "adapter_version": ADAPTER_VERSION,
        "prefix_status": adapted["prefix_status"],
        "final_text_sha256": adapted["final_text_sha256"],
    }
    if capture.get("response_adapter_verification") != expected_verification:
        raise Qwen35ResponseAdapterError("response adapter verification drifted")
    if not isinstance(generation, dict):
        raise Qwen35ResponseAdapterError("generation metadata must be an object")
    prompt_tokens = generation.get("prompt_tokens")
    completion_tokens = generation.get("completion_tokens")
    if (
        isinstance(prompt_tokens, bool)
        or not isinstance(prompt_tokens, int)
        or prompt_tokens < 0
        or prompt_tokens != len(prompt_ids)
    ):
        raise Qwen35ResponseAdapterError("prompt token count mismatch")
    if (
        isinstance(completion_tokens, bool)
        or not isinstance(completion_tokens, int)
        or completion_tokens < 0
        or completion_tokens != len(generated_ids)
    ):
        raise Qwen35ResponseAdapterError("completion token count mismatch")
    return generated_text


def adapt_qwen35_text(
    message_content: str, *, generated_only_text: str | None = None
) -> dict[str, Any]:
    """Separate Qwen3.5's reconstructed prefix from final answer content.

    When token-derived ``generated_only_text`` is available it is authoritative
    and must exactly match either the message content or the message content
    after one known prefix.  Without token evidence, one exact prefix at byte
    zero is removed.  Non-empty, malformed, indented, or embedded thinking tags
    are never removed.
    """

    if not isinstance(message_content, str):
        raise TypeError("message_content must be a string")
    if generated_only_text is not None and not isinstance(generated_only_text, str):
        raise TypeError("generated_only_text must be a string or None")

    operations: list[str] = []
    if generated_only_text is None:
        evidence_source = "ms_swift_message_content"
        if message_content.startswith(NON_THINKING_PREFIX):
            final_text = message_content[len(NON_THINKING_PREFIX) :]
            operations.append("remove_one_exact_qwen35_non_thinking_prefix")
            prefix_status = "removed_without_token_evidence"
        else:
            final_text = message_content
            prefix_status = "not_present"
    else:
        evidence_source = "generated_token_ids_decode"
        final_text = generated_only_text
        if message_content == NON_THINKING_PREFIX + generated_only_text:
            operations.append("verify_ms_swift_reconstructed_non_thinking_prefix")
            prefix_status = "verified_reconstructed"
        elif message_content == generated_only_text:
            prefix_status = "message_content_already_generated_only"
        else:
            raise Qwen35ResponseAdapterError(
                "message content does not match generated-only token decode"
            )

    suspicious_thinking_prefix = final_text.startswith("<think>")
    return {
        "adapter_version": ADAPTER_VERSION,
        "evidence_source": evidence_source,
        "message_content_sha256": text_sha256(message_content),
        "generated_only_text_sha256": (
            text_sha256(generated_only_text) if generated_only_text is not None else None
        ),
        "known_response_prefix": NON_THINKING_PREFIX,
        "known_response_prefix_sha256": text_sha256(NON_THINKING_PREFIX),
        "prefix_status": prefix_status,
        "operations": operations,
        "final_text": final_text,
        "final_text_sha256": text_sha256(final_text),
        "suspicious_thinking_prefix": suspicious_thinking_prefix,
    }
