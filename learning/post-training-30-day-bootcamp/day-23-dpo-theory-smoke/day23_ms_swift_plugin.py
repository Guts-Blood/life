#!/usr/bin/env python3
"""External ms-swift plugin entry point for the Day 23 RLHF template alias."""

from __future__ import annotations

import sys
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parent
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from day23_rlhf_template import register_dpo_template  # noqa: E402


register_dpo_template()
