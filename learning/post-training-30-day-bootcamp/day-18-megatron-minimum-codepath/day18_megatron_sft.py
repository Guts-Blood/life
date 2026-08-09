#!/usr/bin/env python3
"""Launch the pinned Megatron-SWIFT SFT entrypoint with Day18 compatibility checks."""

import os

from day18_transformers_compat import install_hf_argparser_compat


def main() -> None:
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    install_hf_argparser_compat()
    from swift.megatron import megatron_sft_main

    megatron_sft_main()


if __name__ == "__main__":
    main()
