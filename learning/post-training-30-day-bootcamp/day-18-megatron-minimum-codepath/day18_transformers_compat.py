#!/usr/bin/env python3
"""Narrow compatibility shim for the pinned Transformers argument parser."""

from __future__ import annotations

import inspect
from typing import get_args, get_origin


def install_hf_argparser_compat() -> None:
    import transformers
    from transformers.hf_argparser import HfArgumentParser

    if transformers.__version__ != "5.12.1":
        raise RuntimeError(f"Day18 compatibility shim only covers transformers 5.12.1, got {transformers.__version__}")

    original = HfArgumentParser._parse_dataclass_field
    if getattr(original, "_day18_cp_comm_type_compat", False):
        return
    if "isinstance(None, field.type.__args__[1])" not in inspect.getsource(original):
        raise RuntimeError("pinned HfArgumentParser no longer matches the audited cp_comm_type failure")

    def patched(parser, field):
        if field.name == "cp_comm_type" and field.type is not str:
            args = get_args(field.type)
            list_of_str = any(get_origin(arg) is list and get_args(arg) == (str,) for arg in args)
            if str not in args or type(None) not in args or not list_of_str:
                raise RuntimeError(f"unexpected cp_comm_type annotation: {field.type!r}")
            # HfArgumentParser intends to select `str` from Union[str,
            # List[str], None].  Its 5.12.1 implementation instead calls
            # isinstance(None, List[str]) when typing's union cache changes
            # argument order, which raises on Python 3.12.
            field.type = str
        return original(parser, field)

    patched._day18_cp_comm_type_compat = True
    HfArgumentParser._parse_dataclass_field = staticmethod(patched)
