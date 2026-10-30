#!/usr/bin/env python3
"""Offline tests for the Day 22 Qwen3.5 pair processor auditor."""

from __future__ import annotations

import copy
import unittest
from typing import Any, Mapping, Sequence

import audit_day22_qwen35_processor as audit


CHOSEN_TEXT = "    return n + 1"
REJECTED_TEXT = "    return n - 1"
CHOSEN_TOKENS = [audit.INDENT_TOKEN_ID, 1001, audit.IM_END_TOKEN_ID]
REJECTED_TOKENS = [audit.INDENT_TOKEN_ID, 1002, audit.IM_END_TOKEN_ID]


def _encoding(prefix: Sequence[int], response_tokens: Sequence[int]) -> dict[str, Any]:
    input_ids = list(prefix) + list(response_tokens) + [audit.NEWLINE_TOKEN_ID]
    labels = [-100] * len(prefix) + list(response_tokens) + [-100]
    return {"input_ids": input_ids, "labels": labels}


def _decoder(token_ids: Sequence[int]) -> str:
    values = list(token_ids)
    if values == CHOSEN_TOKENS:
        return CHOSEN_TEXT
    if values == REJECTED_TOKENS:
        return REJECTED_TEXT
    raise AssertionError(f"unexpected tokens: {values}")


def _contract() -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_name": audit.CONTRACT_SCHEMA_NAME,
        "schema_version": audit.CONTRACT_SCHEMA_VERSION,
        "status": "frozen",
        "model_key": audit.MODEL_KEY,
        "model_revision": audit.MODEL_REVISION,
        "processor_revision": f"processor@{audit.MODEL_REVISION}",
        "tokenizer_revision": f"tokenizer@{audit.MODEL_REVISION}",
        "template_revision": audit.TARGET_TEMPLATE_REVISION,
        "processor_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "template_sha256": audit.TARGET_TEMPLATE_SHA256,
    }
    value["contract_sha256"] = audit.object_sha256(value)
    return value


def _replay() -> dict[str, Any]:
    prompt = (
        "Complete the Python function below. Return only the indented Python "
        "continuation: no Markdown fence, explanation, or repeated function definition."
        "\n\nTask: Add one.\n\nStarter code:\ndef add_one(n):"
    )
    row: dict[str, Any] = {
        "schema_name": "day22.mbpp_replay_pair",
        "schema_version": 1,
        "pair_id": "mbpp:task:1:smoke:00",
        "family_id": "mbpp:task:1",
        "task_id": "1",
        "split": "train",
        "prompt": {"text": prompt, "sha256": audit.text_sha256(prompt)},
        "chosen": {
            "candidate_id": "chosen",
            "origin": "mbpp_canonical",
            "text": CHOSEN_TEXT,
            "sha256": audit.text_sha256(CHOSEN_TEXT),
        },
        "rejected": {
            "candidate_id": "rejected",
            "origin": "deterministic_mutation",
            "text": REJECTED_TEXT,
            "sha256": audit.text_sha256(REJECTED_TEXT),
        },
    }
    row["row_sha256"] = audit.object_sha256(row)
    return row


class BranchEncodingTests(unittest.TestCase):
    def test_valid_branch_has_trailing_masked_newline(self) -> None:
        evidence, prefix = audit.analyze_branch_encoding(
            _encoding([10, 20], CHOSEN_TOKENS),
            response_text=CHOSEN_TEXT,
            decode_supervised=_decoder,
            max_length=32,
            label="chosen",
        )
        self.assertEqual(prefix, (10, 20))
        self.assertEqual(evidence["response_span"], [2, 5])
        self.assertEqual(evidence["input_token_count"], 6)
        self.assertEqual(evidence["trailing_masked_token_count"], 1)
        sealed = {key: value for key, value in evidence.items() if key != "branch_sha256"}
        self.assertEqual(evidence["branch_sha256"], audit.object_sha256(sealed))

    def test_noncontiguous_response_labels_fail(self) -> None:
        encoded = _encoding([10, 20], CHOSEN_TOKENS)
        encoded["labels"][3] = -100
        with self.assertRaisesRegex(audit.Day22ProcessorAuditError, "not contiguous"):
            audit.analyze_branch_encoding(
                encoded,
                response_text=CHOSEN_TEXT,
                decode_supervised=_decoder,
                max_length=32,
                label="chosen",
            )

    def test_unsupervised_four_space_boundary_fails(self) -> None:
        encoded = _encoding([10, 20], CHOSEN_TOKENS)
        encoded["labels"][2] = -100
        with self.assertRaisesRegex(audit.Day22ProcessorAuditError, "four-space boundary"):
            audit.analyze_branch_encoding(
                encoded,
                response_text=CHOSEN_TEXT,
                decode_supervised=_decoder,
                max_length=32,
                label="chosen",
            )

    def test_more_than_four_leading_spaces_fail(self) -> None:
        with self.assertRaisesRegex(audit.Day22ProcessorAuditError, "exactly four-space"):
            audit.analyze_branch_encoding(
                _encoding([10, 20], CHOSEN_TOKENS),
                response_text="        return n + 1",
                decode_supervised=lambda _: "        return n + 1",
                max_length=32,
                label="chosen",
            )


class PairAuditTests(unittest.TestCase):
    @staticmethod
    def _encoder(messages: Sequence[Mapping[str, str]]) -> Mapping[str, Any]:
        response = messages[1]["content"]
        tokens = CHOSEN_TOKENS if response == CHOSEN_TEXT else REJECTED_TOKENS
        return _encoding([10, 20], tokens)

    def test_pair_audit_has_shared_prompt_and_self_hash(self) -> None:
        result = audit.audit_replay_pair(
            _replay(),
            encode_messages=self._encoder,
            decode_supervised=_decoder,
            processor_contract=_contract(),
            max_length=32,
        )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["task_id"], "1")
        self.assertEqual(result["chosen_candidate_id"], "chosen")
        self.assertEqual(result["rejected_candidate_id"], "rejected")
        self.assertTrue(result["retokenized_for_day22"])
        self.assertFalse(result["legacy_token_ids_reused"])
        self.assertEqual(
            result["chosen"]["prompt_prefix_token_ids_sha256"],
            result["rejected"]["prompt_prefix_token_ids_sha256"],
        )
        sealed = {key: value for key, value in result.items() if key != "audit_sha256"}
        self.assertEqual(result["audit_sha256"], audit.object_sha256(sealed))

    def test_different_rendered_prompt_prefix_fails(self) -> None:
        def drifted_encoder(messages: Sequence[Mapping[str, str]]) -> Mapping[str, Any]:
            response = messages[1]["content"]
            if response == CHOSEN_TEXT:
                return _encoding([10, 20], CHOSEN_TOKENS)
            return _encoding([10, 21], REJECTED_TOKENS)

        with self.assertRaisesRegex(audit.Day22ProcessorAuditError, "prefixes differ"):
            audit.audit_replay_pair(
                _replay(),
                encode_messages=drifted_encoder,
                decode_supervised=_decoder,
                processor_contract=_contract(),
                max_length=32,
            )

    def test_replay_self_hash_drift_fails_before_encoding(self) -> None:
        row = copy.deepcopy(_replay())
        row["split"] = "heldout"
        with self.assertRaisesRegex(audit.Day22ProcessorAuditError, "self-hash drifted"):
            audit.audit_replay_pair(
                row,
                encode_messages=self._encoder,
                decode_supervised=_decoder,
                processor_contract=_contract(),
                max_length=32,
            )

    def test_processor_contract_self_hash_drift_fails(self) -> None:
        contract = _contract()
        contract["model_key"] = "drifted"
        with self.assertRaisesRegex(audit.Day22ProcessorAuditError, "contract self-hash"):
            audit.audit_replay_pair(
                _replay(),
                encode_messages=self._encoder,
                decode_supervised=_decoder,
                processor_contract=contract,
                max_length=32,
            )


if __name__ == "__main__":
    unittest.main()
