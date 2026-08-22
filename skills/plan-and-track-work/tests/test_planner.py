from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
FIXTURE = Path(__file__).parent / "fixtures" / "av2-plan.json"
sys.path.insert(0, str(SCRIPTS))

from planner_core import (  # noqa: E402
    PlanError,
    compute_inherited_urgency,
    conflict_matrix,
    dispatch_queue,
    human_focus_chain,
    load_events,
    new_event,
    parallel_waves,
    render_views,
    replay_events,
    review_batches,
    trace_task,
    validate_event,
    validate_plan,
)


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class PlannerCoreTests(unittest.TestCase):
    def test_fixture_is_valid(self) -> None:
        result = validate_plan(load_fixture())
        self.assertTrue(result.ok, result.errors)

    def test_cycle_is_rejected(self) -> None:
        plan = load_fixture()
        plan["tasks"][0]["dependencies"] = ["AV2-RELEASE"]
        result = validate_plan(plan)
        self.assertFalse(result.ok)
        self.assertTrue(any("cycle" in error for error in result.errors))

    def test_priority_is_inherited_without_mutating_declared_priority(self) -> None:
        plan = load_fixture()
        inherited = compute_inherited_urgency(plan)
        contract = next(task for task in plan["tasks"] if task["id"] == "AV2-CONTRACT")
        self.assertEqual(contract["declared_priority"], "P1")
        self.assertEqual(inherited["AV2-CONTRACT"], "P0")

    def test_priority_change_requires_user_authority(self) -> None:
        current = load_fixture()
        candidate = copy.deepcopy(current)
        candidate["tasks"][0]["declared_priority"] = "P0"
        denied = validate_plan(candidate, against=current)
        allowed = validate_plan(candidate, against=current, allow_priority_changes=True)
        self.assertFalse(denied.ok)
        self.assertTrue(allowed.ok, allowed.errors)

    def test_conflicting_tasks_are_never_in_same_wave(self) -> None:
        plan = load_fixture()
        conflicts = conflict_matrix(plan)
        key = tuple(sorted(("AV2-BACKEND", "AV2-SCHEMA-MIGRATION")))
        self.assertIn(key, conflicts)
        for wave in parallel_waves(plan):
            self.assertFalse({"AV2-BACKEND", "AV2-SCHEMA-MIGRATION"} <= set(wave["tasks"]))

    def test_pre_execution_gate_blocks_dispatch_and_enters_human_focus(self) -> None:
        plan = load_fixture()
        events = [new_event(plan, "plan.approved", "user")]
        events.append(new_event(plan, "task.ready", "coordinator", task_id="AV2-CONTRACT"))
        state = replay_events(plan, events)
        queue = dispatch_queue(plan, state)
        self.assertNotIn("AV2-CONTRACT", queue["dispatch_now"])
        self.assertTrue(
            any(item["task_id"] == "AV2-CONTRACT" and item["kind"] == "review" for item in human_focus_chain(plan, state))
        )

    def test_draft_never_dispatches(self) -> None:
        plan = load_fixture()
        state = replay_events(plan, [])
        state["task_statuses"]["AV2-CONTRACT"] = "ready"
        queue = dispatch_queue(plan, state)
        self.assertFalse(queue["dispatch_now"])
        self.assertEqual(queue["reason"], "plan_draft")

    def test_approved_decision_gate_unblocks_dispatch(self) -> None:
        plan = load_fixture()
        events = [
            new_event(plan, "plan.approved", "user"),
            new_event(plan, "task.ready", "coordinator", task_id="AV2-CONTRACT"),
            new_event(
                plan,
                "review.approved",
                "user",
                task_id="AV2-CONTRACT",
                data={"gate_id": "contract-decision"},
            ),
        ]
        state = replay_events(plan, events)
        self.assertEqual(dispatch_queue(plan, state)["dispatch_now"], ["AV2-CONTRACT"])

    def test_dirty_repo_is_plannable_but_not_dispatchable(self) -> None:
        plan = load_fixture()
        plan["repos"]["recent-master"]["dirty"] = True
        result = validate_plan(plan)
        self.assertTrue(result.ok)
        events = [
            new_event(plan, "plan.approved", "user"),
            new_event(plan, "task.ready", "coordinator", task_id="AV2-CONTRACT"),
            new_event(
                plan,
                "review.approved",
                "user",
                task_id="AV2-CONTRACT",
                data={"gate_id": "contract-decision"},
            ),
        ]
        queue = dispatch_queue(plan, replay_events(plan, events))
        self.assertFalse(queue["dispatch_now"])
        self.assertTrue(any(item["reason"] == "snapshot_dirty" for item in queue["waiting"]))

    def test_stale_task_snapshot_is_not_dispatchable(self) -> None:
        plan = load_fixture()
        contract = next(task for task in plan["tasks"] if task["id"] == "AV2-CONTRACT")
        contract["snapshot_sha"] = "stale-sha"
        result = validate_plan(plan)
        self.assertTrue(result.ok)
        self.assertTrue(any("differs from repo" in warning for warning in result.warnings))
        events = [
            new_event(plan, "plan.approved", "user"),
            new_event(plan, "task.ready", "coordinator", task_id="AV2-CONTRACT"),
            new_event(
                plan,
                "review.approved",
                "user",
                task_id="AV2-CONTRACT",
                data={"gate_id": "contract-decision"},
            ),
        ]
        queue = dispatch_queue(plan, replay_events(plan, events))
        self.assertFalse(queue["dispatch_now"])
        self.assertTrue(any(item["reason"] == "snapshot_stale" for item in queue["waiting"]))

    def test_event_replay_and_trace(self) -> None:
        plan = load_fixture()
        events = [
            new_event(plan, "plan.approved", "user"),
            new_event(plan, "task.ready", "coordinator", task_id="AV2-CONTRACT"),
            new_event(plan, "task.claimed", "worker", task_id="AV2-CONTRACT", run_id="run-1"),
            new_event(plan, "task.started", "worker", task_id="AV2-CONTRACT", run_id="run-1"),
            new_event(plan, "task.completed", "worker", task_id="AV2-CONTRACT", run_id="run-1"),
        ]
        state = replay_events(plan, events)
        self.assertEqual(state["task_statuses"]["AV2-CONTRACT"], "completed")
        self.assertEqual(len(trace_task(events, plan["plan_id"], "AV2-CONTRACT")), 4)

    def test_invalid_state_transition_is_rejected(self) -> None:
        plan = load_fixture()
        state = replay_events(plan, [new_event(plan, "plan.approved", "user")])
        with self.assertRaises(PlanError):
            validate_event(plan, state, "task.completed", task_id="AV2-CONTRACT")

    def test_ready_event_cannot_bypass_dependencies(self) -> None:
        plan = load_fixture()
        state = replay_events(plan, [new_event(plan, "plan.approved", "user")])
        with self.assertRaises(PlanError):
            validate_event(plan, state, "task.ready", task_id="AV2-BACKEND")

    def test_manual_verification_cannot_bypass_review_gate(self) -> None:
        plan = load_fixture()
        events = [
            new_event(plan, "plan.approved", "user"),
            new_event(plan, "task.ready", "coordinator", task_id="AV2-CONTRACT"),
            new_event(plan, "task.started", "worker", task_id="AV2-CONTRACT"),
            new_event(plan, "task.completed", "worker", task_id="AV2-CONTRACT"),
            new_event(
                plan,
                "review.requested",
                "coordinator",
                task_id="AV2-CONTRACT",
                data={"gate_id": "cross-repo-contract"},
            ),
        ]
        state = replay_events(plan, events)
        with self.assertRaises(PlanError):
            validate_event(plan, state, "task.verified", task_id="AV2-CONTRACT")

    def test_review_batch_waits_for_all_blocking_gates(self) -> None:
        plan = load_fixture()
        events = [
            new_event(plan, "plan.approved", "user"),
            new_event(plan, "task.ready", "coordinator", task_id="AV2-RELEASE"),
            new_event(plan, "task.started", "human", task_id="AV2-RELEASE"),
            new_event(plan, "task.completed", "human", task_id="AV2-RELEASE"),
            new_event(
                plan,
                "review.requested",
                "coordinator",
                task_id="AV2-RELEASE",
                data={"gate_id": "release-authority"},
            ),
            new_event(
                plan,
                "review.requested",
                "coordinator",
                task_id="AV2-RELEASE",
                data={"gate_id": "cross-repo-release"},
            ),
            new_event(
                plan,
                "review.approved",
                "user",
                task_id="AV2-RELEASE",
                data={"gate_id": "release-authority"},
            ),
        ]
        state = replay_events(plan, events)
        self.assertEqual(state["task_statuses"]["AV2-RELEASE"], "review")
        production_batch = next(
            batch for batch in review_batches(plan, state) if batch["milestone"] == "production-release"
        )
        self.assertEqual(sum(1 for item in production_batch["items"] if item["ready"]), 1)
        events.append(
            new_event(
                plan,
                "review.approved",
                "user",
                task_id="AV2-RELEASE",
                data={"gate_id": "cross-repo-release"},
            )
        )
        state = replay_events(plan, events)
        self.assertEqual(state["task_statuses"]["AV2-RELEASE"], "verified")
        self.assertFalse(any(batch["milestone"] == "production-release" for batch in review_batches(plan, state)))

    def test_render_contains_all_core_views(self) -> None:
        plan = load_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            files = render_views(plan, [new_event(plan, "plan.created", "coordinator")], temporary)
            dashboard = Path(files["dashboard"]).read_text(encoding="utf-8")
            mermaid = Path(files["mermaid"]).read_text(encoding="utf-8")
            self.assertIn("## Human Focus Chain", dashboard)
            self.assertIn("## Proposed Codex Queue (after approval)", dashboard)
            self.assertIn("## Insights", dashboard)
            self.assertIn("AV2-CONTRACT", dashboard)
            self.assertIn("flowchart LR", mermaid)


class PlannerCliTests(unittest.TestCase):
    def run_cli(self, *arguments: str, expected: int = 0) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, str(SCRIPTS / "work_planner.py"), *arguments]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != expected:
            self.fail(
                f"CLI returned {result.returncode}, expected {expected}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        return result

    def test_approve_record_render_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = root / "state" / "plan.json"
            events_path = root / "traces" / "events.jsonl"
            views_path = root / "views"
            self.run_cli("init", "--root", str(root))
            plan_path.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")

            self.run_cli(
                "approve",
                "--plan",
                str(plan_path),
                "--events",
                str(events_path),
            )
            self.run_cli(
                "record",
                "--plan",
                str(plan_path),
                "--events",
                str(events_path),
                "--event",
                "review.approved",
                "--task",
                "AV2-CONTRACT",
                "--gate-id",
                "contract-decision",
                "--actor",
                "user",
            )
            self.run_cli(
                "render",
                "--plan",
                str(plan_path),
                "--events",
                str(events_path),
                "--output-dir",
                str(views_path),
            )
            state = replay_events(load_fixture(), load_events(events_path))
            self.assertEqual(state["plan_status"], "approved")
            self.assertTrue((views_path / "dashboard.md").exists())
            self.assertTrue((views_path / "runtime-state.json").exists())

    def test_verification_auto_readies_downstream_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = root / "state" / "plan.json"
            events_path = root / "traces" / "events.jsonl"
            self.run_cli("init", "--root", str(root))
            plan_path.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
            self.run_cli("approve", "--plan", str(plan_path), "--events", str(events_path))
            commands = [
                ("review.approved", ["--gate-id", "contract-decision"]),
                ("task.claimed", []),
                ("task.started", []),
                ("task.completed", []),
                ("review.requested", ["--gate-id", "cross-repo-contract"]),
                ("review.approved", ["--gate-id", "cross-repo-contract"]),
                ("task.done", []),
            ]
            for event_type, extra in commands:
                self.run_cli(
                    "record",
                    "--plan",
                    str(plan_path),
                    "--events",
                    str(events_path),
                    "--event",
                    event_type,
                    "--task",
                    "AV2-CONTRACT",
                    "--actor",
                    "test",
                    *extra,
                )
            state = replay_events(load_fixture(), load_events(events_path))
            for task_id in ("AV2-BACKEND", "AV2-FRONTEND", "AV2-BENCHMARK", "AV2-SCHEMA-MIGRATION"):
                self.assertEqual(state["task_statuses"][task_id], "ready")

    def test_replan_preserves_priority_without_user_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            current_path = root / "current.json"
            candidate_path = root / "candidate.json"
            output_path = root / "output.json"
            events_path = root / "events.jsonl"
            current = load_fixture()
            candidate = copy.deepcopy(current)
            candidate["tasks"][0]["declared_priority"] = "P0"
            current_path.write_text(json.dumps(current), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            self.run_cli(
                "replan",
                "--current",
                str(current_path),
                "--candidate",
                str(candidate_path),
                "--output",
                str(output_path),
                "--events",
                str(events_path),
                expected=2,
            )
            self.run_cli(
                "replan",
                "--current",
                str(current_path),
                "--candidate",
                str(candidate_path),
                "--output",
                str(output_path),
                "--events",
                str(events_path),
                "--priority-change-authority",
                "user",
            )
            replanned = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(replanned["version"], 2)
            self.assertEqual(replanned["status"], "draft")


if __name__ == "__main__":
    unittest.main()
