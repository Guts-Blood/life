#!/usr/bin/env python3
"""Run Megatron export while forcing its built-in parity probe to use a fixed text fixture."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def load_text_examples(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != 2:
        raise ValueError(f"fixture must contain exactly two rows, got {len(rows)}")
    for row in rows:
        if any(key in row for key in ("images", "videos", "audios")):
            raise ValueError("Day18 parity fixture must be text-only")
        rendered = json.dumps(row, ensure_ascii=False).lower()
        if any(marker in rendered for marker in ("<image>", "<video>", "<audio>")):
            raise ValueError("Day18 parity fixture contains a multimodal placeholder")
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("every fixture row must have non-empty messages")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--fixture", required=True, type=Path)
    known, megatron_args = parser.parse_known_args()
    examples = load_text_examples(known.fixture)

    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    os.environ.setdefault("USE_MCORE_GDN", "1")
    if os.environ["USE_MCORE_GDN"] != "1":
        raise RuntimeError("Day18 requires USE_MCORE_GDN=1")

    from day18_transformers_compat import install_hf_argparser_compat

    install_hf_argparser_compat()

    # The pinned ms-swift helper defaults to an internet-hosted image for any
    # multimodal model. Replace only its example provider; all conversion and
    # HF/MCore forward logic remains the pinned implementation.
    import swift.megatron.utils as megatron_utils
    from swift.megatron.utils import convert_utils

    original_test = convert_utils.test_convert_precision
    original_examples = convert_utils.get_examples
    original_forward = convert_utils.forward_step_helper
    original_params_sum = convert_utils._test_params_sum
    captured_mcore_logits = []

    def capture_mcore_logits(*args, **kwargs):
        logits = original_forward(*args, **kwargs)
        if logits is not None:
            captured_mcore_logits.append(logits)
        return logits

    def test_both_text_rows(args, hf_model, mg_model, template, test_convert_dtype=None) -> None:
        try:
            template.set_mode("train")
            for row_index, example in enumerate(examples):
                print(f"DAY18_PARITY_ROW={row_index}", flush=True)
                # Parameter inventory is model-level evidence and only needs
                # one pass.  The pinned helper is otherwise reusable, but a
                # second scan after its module streaming trips autograd on the
                # restored MCore parameters.
                if row_index:
                    convert_utils._test_params_sum = lambda *_args, **_kwargs: None
                convert_utils.get_examples = lambda _mm_type, current=example: current
                template.use_megatron = False
                encoded = template.encode(example, return_length=True)
                loss_batch = template.data_collator([encoded])
                labels = loss_batch["labels"].to("cuda")
                loss_scale = loss_batch.get("loss_scale")
                if loss_scale is not None:
                    loss_scale = loss_scale.to("cuda")
                captured_mcore_logits.clear()
                if hf_model is not None:
                    # The pinned helper streams modules back to CPU after each
                    # comparison. Put the small 4B reference back on this H100
                    # before the next fixed row.
                    hf_model.to("cuda")
                original_test(
                    args,
                    hf_model,
                    mg_model,
                    template,
                    test_convert_dtype=test_convert_dtype,
                )
                if len(captured_mcore_logits) != 1:
                    raise RuntimeError(
                        f"expected one MCore logits tensor for parity row {row_index}, got {len(captured_mcore_logits)}"
                    )
                import torch.nn.functional as functional

                logits = captured_mcore_logits[0].float()
                sequence_length = min(logits.shape[1], labels.shape[1])
                shift_logits = logits[:, :sequence_length - 1, :].contiguous()
                shift_labels = labels[:, 1:sequence_length].contiguous()
                valid = shift_labels.ne(-100)
                safe_labels = shift_labels.masked_fill(~valid, 0)
                token_losses = functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.shape[-1]),
                    safe_labels.view(-1),
                    reduction="none",
                ).view_as(shift_labels)
                weights = valid.float()
                if loss_scale is not None:
                    weights = loss_scale[:, 1:sequence_length].float() * valid.float()
                denominator = weights.sum()
                if denominator.item() <= 0:
                    raise RuntimeError(f"parity row {row_index} has zero supervised loss weight")
                numerator = (token_losses * weights).sum()
                loss = numerator / denominator
                print(
                    "DAY18_MCORE_MAIN_LOSS "
                    f"row={row_index} numerator={numerator.item():.12g} "
                    f"denominator={denominator.item():.12g} loss={loss.item():.12g}",
                    flush=True,
                )
        finally:
            convert_utils.get_examples = original_examples
            convert_utils.forward_step_helper = original_forward
            convert_utils._test_params_sum = original_params_sum

    # The lazy pipeline imports this symbol from swift.megatron.utils.
    convert_utils.test_convert_precision = test_both_text_rows
    convert_utils.forward_step_helper = capture_mcore_logits
    megatron_utils.test_convert_precision = test_both_text_rows

    from swift.megatron import megatron_export_main

    megatron_export_main(megatron_args)


if __name__ == "__main__":
    main()
