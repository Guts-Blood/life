#!/usr/bin/env python3
"""Register the Day 25 tests-only E2B reward with pinned ms-swift."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List

from swift.rewards import ORM, orms

from day25_reward_adapter import Day25RewardAdapter, Day25RewardAdapterError, day24


class Day25MBPPTestsOnly(ORM):
    """Production ms-swift wrapper; all domain logic remains in Day 24."""

    def __init__(self, args: Any = None, **kwargs: Any) -> None:
        super().__init__(args, **kwargs)
        if args is not None and int(getattr(args, "num_generations", 0)) != 4:
            raise Day25RewardAdapterError("pinned Day 25 reward requires num_generations=4")
        run_id = os.environ.get("DAY25_RUN_ID", "")
        policy_version = os.environ.get("DAY25_POLICY_VERSION", "")
        ledger_value = os.environ.get("DAY25_TRAJECTORY_LEDGER", "")
        if not ledger_value:
            raise Day25RewardAdapterError("DAY25_TRAJECTORY_LEDGER is required")
        try:
            workers = int(os.environ.get("DAY25_SANDBOX_WORKERS", "4"))
        except ValueError as error:
            raise Day25RewardAdapterError("DAY25_SANDBOX_WORKERS must be an integer") from error
        rollout_only = os.environ.get("DAY25_ABORT_AFTER_REWARD_BATCH", "0")
        if rollout_only not in {"0", "1"}:
            raise Day25RewardAdapterError(
                "DAY25_ABORT_AFTER_REWARD_BATCH must be exactly 0 or 1"
            )
        self.adapter = Day25RewardAdapter(
            executor=day24.execute_e2b_request,
            run_id=run_id,
            policy_version=policy_version,
            ledger_path=Path(ledger_value),
            workers=workers,
            abort_after_complete_batch=rollout_only == "1",
        )

    def __call__(self, completions: List[str], **kwargs: Any) -> List[float]:
        return self.adapter.score_batch(completions, **kwargs)


orms["day25_mbpp_tests_only"] = Day25MBPPTestsOnly
