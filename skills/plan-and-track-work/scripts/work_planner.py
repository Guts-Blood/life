#!/usr/bin/env python3
"""CLI for the plan-and-track-work skill.

Users normally invoke this through Codex conversation.  The CLI exists so all
state changes and rendered views use deterministic, testable operations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from planner_core import (
    PlanError,
    append_event,
    atomic_write_json,
    atomic_write_text,
    human_focus_chain,
    inspect_repo,
    load_events,
    new_event,
    newly_ready_task_ids,
    read_json,
    render_views,
    replay_events,
    task_map,
    trace_task,
    utc_now,
    validate_event,
    validate_plan,
)


def _print_json(value: Any) -> None:
    json.dump(value, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


def _load_data_json(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise PlanError("--data-json must decode to an object")
    return value


def _append_ready_events(plan: dict[str, Any], events_path: Path, actor: str) -> list[dict[str, Any]]:
    events = load_events(events_path)
    state = replay_events(plan, events)
    appended: list[dict[str, Any]] = []
    for task_id in newly_ready_task_ids(plan, state):
        event = new_event(
            plan,
            "task.ready",
            actor,
            task_id=task_id,
            data={"reason": "dependencies_satisfied"},
        )
        append_event(events_path, event)
        appended.append(event)
        events.append(event)
        state = replay_events(plan, events)
    return appended


def command_init(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    state_dir = root / "state"
    traces_dir = root / "traces"
    views_dir = root / "views"
    runs_dir = root / "runs"
    for directory in (state_dir, traces_dir, views_dir, runs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.json"
    if not config_path.exists():
        atomic_write_json(
            config_path,
            {
                "schema_version": 1,
                "max_parallelism": 3,
                "priority_order": ["P0", "P1", "P2", "P3"],
                "review_cadence": "milestone_batch",
                "autonomy": "balanced",
                "repos": {},
            },
        )
    events_path = traces_dir / "events.jsonl"
    events_path.touch(exist_ok=True)
    _print_json(
        {
            "root": str(root),
            "config": str(config_path),
            "plan": str(state_dir / "plan.json"),
            "events": str(events_path),
            "views": str(views_dir),
        }
    )


def command_validate(args: argparse.Namespace) -> None:
    plan = read_json(args.plan)
    against = read_json(args.against) if args.against else None
    result = validate_plan(
        plan,
        against=against,
        allow_priority_changes=args.priority_change_authority == "user",
    )
    _print_json(result.to_dict())
    if not result.ok:
        raise SystemExit(2)


def command_inspect_repos(args: argparse.Namespace) -> None:
    repos: dict[str, Any] = {}
    for item in args.repo:
        if "=" not in item:
            raise PlanError("--repo values must use name=/absolute/path")
        name, raw_path = item.split("=", 1)
        if not name or not raw_path:
            raise PlanError("--repo values must use name=/absolute/path")
        repos[name] = inspect_repo(raw_path)
    _print_json({"repos": repos})


def command_snapshot(args: argparse.Namespace) -> None:
    plan = read_json(args.plan)
    if plan.get("status", "draft") != "draft":
        raise PlanError(
            "snapshot only updates a draft; create a replan candidate before refreshing an approved plan"
        )
    for item in args.repo:
        if "=" not in item:
            raise PlanError("--repo values must use name=/absolute/path")
        name, raw_path = item.split("=", 1)
        if name not in plan.get("repos", {}):
            raise PlanError(f"cannot override unknown repo {name!r}")
        plan["repos"][name]["path"] = str(Path(raw_path).expanduser().resolve())
    drift: list[dict[str, Any]] = []
    for name, previous in plan.get("repos", {}).items():
        if not isinstance(previous, dict) or not previous.get("path"):
            raise PlanError(f"repo {name!r} needs an absolute path before snapshot refresh")
        current = inspect_repo(previous["path"])
        if (
            current.get("snapshot_sha") != previous.get("snapshot_sha")
            or current.get("dirty") != previous.get("dirty")
            or current.get("changed_paths", []) != previous.get("changed_paths", [])
        ):
            drift.append(
                {
                    "repo": name,
                    "from_sha": previous.get("snapshot_sha"),
                    "to_sha": current.get("snapshot_sha"),
                    "dirty": current.get("dirty"),
                    "changed_paths": current.get("changed_paths", []),
                }
            )
        plan["repos"][name] = {**previous, **current}
    for task in plan.get("tasks", []):
        repo = plan.get("repos", {}).get(task.get("repo"), {})
        if repo.get("snapshot_sha"):
            task["snapshot_sha"] = repo["snapshot_sha"]
    plan["updated_at"] = utc_now()
    atomic_write_json(args.output, plan)
    _print_json({"plan": str(Path(args.output)), "drift": drift})


def command_render(args: argparse.Namespace) -> None:
    plan = read_json(args.plan)
    result = validate_plan(plan)
    if not result.ok:
        _print_json(result.to_dict())
        raise SystemExit(2)
    events = load_events(args.events)
    files = render_views(plan, events, args.output_dir)
    _print_json({"files": files, "warnings": list(result.warnings)})


def command_create(args: argparse.Namespace) -> None:
    plan = read_json(args.plan)
    result = validate_plan(plan)
    if not result.ok:
        _print_json(result.to_dict())
        raise SystemExit(2)
    events_path = Path(args.events)
    events = load_events(events_path)
    exists = any(
        event.get("plan_id") == plan["plan_id"]
        and event.get("plan_version") == plan["version"]
        and event.get("event_type") == "plan.created"
        for event in events
    )
    if exists:
        raise PlanError("plan.created already exists for this plan version")
    event = new_event(
        plan,
        "plan.created",
        args.actor,
        data={"warnings": list(result.warnings)},
    )
    append_event(events_path, event)
    _print_json({"event": event, "warnings": list(result.warnings)})


def command_approve(args: argparse.Namespace) -> None:
    plan_path = Path(args.plan)
    events_path = Path(args.events)
    plan = read_json(plan_path)
    result = validate_plan(plan)
    if not result.ok:
        _print_json(result.to_dict())
        raise SystemExit(2)
    events = load_events(events_path)
    if not any(
        event.get("plan_id") == plan["plan_id"]
        and event.get("plan_version") == plan["version"]
        and event.get("event_type") in {"plan.created", "plan.revised"}
        for event in events
    ):
        created = new_event(
            plan,
            "plan.created",
            "codex-coordinator",
            data={"reason": "backfilled_before_approval", "warnings": list(result.warnings)},
        )
        append_event(events_path, created)
        events.append(created)
    state = replay_events(plan, events)
    validate_event(plan, state, "plan.approved")
    approved = new_event(
        plan,
        "plan.approved",
        args.actor,
        data={"approval": "explicit", "warnings_acknowledged": list(result.warnings)},
    )
    append_event(events_path, approved)
    events.append(approved)
    plan["status"] = "approved"
    atomic_write_json(plan_path, plan)
    ready_events = _append_ready_events(plan, events_path, args.actor)

    state = replay_events(plan, load_events(events_path))
    focus = human_focus_chain(plan, state)
    pinned_event = None
    if focus:
        bundle = focus[0]["context_bundle"]
        pinned_event = new_event(
            plan,
            "focus.pinned",
            args.actor,
            data={"context_bundle": bundle, "reason": "first_approved_human_focus"},
        )
        append_event(events_path, pinned_event)
    output = {"approved_event": approved, "ready_events": ready_events}
    if pinned_event:
        output["focus_event"] = pinned_event
    _print_json(output)


def command_record(args: argparse.Namespace) -> None:
    plan = read_json(args.plan)
    events_path = Path(args.events)
    events = load_events(events_path)
    state = replay_events(plan, events)
    data = _load_data_json(args.data_json)
    if args.reason:
        data["reason"] = args.reason
    if args.gate_id:
        data["gate_id"] = args.gate_id
    validate_event(
        plan,
        state,
        args.event,
        task_id=args.task,
        data=data,
    )
    event = new_event(
        plan,
        args.event,
        args.actor,
        task_id=args.task,
        run_id=args.run_id,
        data=data,
    )
    append_event(events_path, event)
    ready_events = _append_ready_events(plan, events_path, args.actor)
    _print_json({"event": event, "ready_events": ready_events})


def command_state(args: argparse.Namespace) -> None:
    plan = read_json(args.plan)
    state = replay_events(plan, load_events(args.events))
    _print_json(state)


def command_trace(args: argparse.Namespace) -> None:
    plan = read_json(args.plan)
    events = load_events(args.events)
    if args.task not in task_map(plan):
        raise PlanError(f"unknown task {args.task!r}")
    _print_json({"task_id": args.task, "events": trace_task(events, plan["plan_id"], args.task)})


def command_replan(args: argparse.Namespace) -> None:
    current = read_json(args.current)
    candidate = read_json(args.candidate)
    allow_priority = args.priority_change_authority == "user"
    result = validate_plan(
        candidate,
        against=current,
        allow_priority_changes=allow_priority,
    )
    if not result.ok:
        _print_json(result.to_dict())
        raise SystemExit(2)
    if candidate.get("plan_id") != current.get("plan_id"):
        raise PlanError("candidate plan_id must match the current plan_id")
    candidate["version"] = int(current.get("version", 0)) + 1
    candidate["status"] = "draft"
    atomic_write_json(args.output, candidate)
    event = new_event(
        candidate,
        "plan.revised",
        args.actor,
        data={
            "previous_version": current.get("version"),
            "priority_change_authority": args.priority_change_authority,
            "warnings": list(result.warnings),
        },
    )
    append_event(args.events, event)
    _print_json({"plan": str(Path(args.output)), "event": event, "warnings": list(result.warnings)})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan, schedule, render, and trace multi-repo work."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize work-planning runtime directories")
    init_parser.add_argument("--root", required=True)
    init_parser.set_defaults(func=command_init)

    validate_parser = subparsers.add_parser("validate", help="Validate a plan")
    validate_parser.add_argument("--plan", required=True)
    validate_parser.add_argument("--against")
    validate_parser.add_argument(
        "--priority-change-authority", choices=("none", "user"), default="none"
    )
    validate_parser.set_defaults(func=command_validate)

    inspect_parser = subparsers.add_parser("inspect-repos", help="Capture Git snapshots")
    inspect_parser.add_argument("--repo", action="append", default=[], required=True)
    inspect_parser.set_defaults(func=command_inspect_repos)

    snapshot_parser = subparsers.add_parser(
        "snapshot", help="Refresh Git snapshots in a draft plan"
    )
    snapshot_parser.add_argument("--plan", required=True)
    snapshot_parser.add_argument("--output", required=True)
    snapshot_parser.add_argument("--repo", action="append", default=[])
    snapshot_parser.set_defaults(func=command_snapshot)

    render_parser = subparsers.add_parser("render", help="Generate Markdown and Mermaid views")
    render_parser.add_argument("--plan", required=True)
    render_parser.add_argument("--events", required=True)
    render_parser.add_argument("--output-dir", required=True)
    render_parser.set_defaults(func=command_render)

    create_parser = subparsers.add_parser("create", help="Record creation of a validated draft")
    create_parser.add_argument("--plan", required=True)
    create_parser.add_argument("--events", required=True)
    create_parser.add_argument("--actor", default="codex-coordinator")
    create_parser.set_defaults(func=command_create)

    approve_parser = subparsers.add_parser("approve", help="Approve a draft plan")
    approve_parser.add_argument("--plan", required=True)
    approve_parser.add_argument("--events", required=True)
    approve_parser.add_argument("--actor", default="user")
    approve_parser.set_defaults(func=command_approve)

    record_parser = subparsers.add_parser("record", help="Append a validated trace event")
    record_parser.add_argument("--plan", required=True)
    record_parser.add_argument("--events", required=True)
    record_parser.add_argument("--event", required=True)
    record_parser.add_argument("--task")
    record_parser.add_argument("--actor", required=True)
    record_parser.add_argument("--run-id")
    record_parser.add_argument("--gate-id")
    record_parser.add_argument("--reason")
    record_parser.add_argument("--data-json")
    record_parser.set_defaults(func=command_record)

    state_parser = subparsers.add_parser("state", help="Replay the event ledger")
    state_parser.add_argument("--plan", required=True)
    state_parser.add_argument("--events", required=True)
    state_parser.set_defaults(func=command_state)

    trace_parser = subparsers.add_parser("trace", help="Show one task's event history")
    trace_parser.add_argument("--plan", required=True)
    trace_parser.add_argument("--events", required=True)
    trace_parser.add_argument("--task", required=True)
    trace_parser.set_defaults(func=command_trace)

    replan_parser = subparsers.add_parser("replan", help="Validate and create a new plan version")
    replan_parser.add_argument("--current", required=True)
    replan_parser.add_argument("--candidate", required=True)
    replan_parser.add_argument("--output", required=True)
    replan_parser.add_argument("--events", required=True)
    replan_parser.add_argument("--actor", default="codex-coordinator")
    replan_parser.add_argument(
        "--priority-change-authority", choices=("none", "user"), default="none"
    )
    replan_parser.set_defaults(func=command_replan)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (PlanError, json.JSONDecodeError, OSError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
