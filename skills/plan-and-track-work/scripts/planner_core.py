#!/usr/bin/env python3
"""Deterministic core for the plan-and-track-work Codex skill.

The language model enriches tasks with semantic context.  This module owns the
parts that must be deterministic: validation, graph scheduling, conflict
detection, event replay, and generated views.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
import uuid
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
ACTORS = {"codex", "hybrid", "human"}
CONFIDENCE_LEVELS = {"high", "medium", "low"}
PLAN_STATUSES = {"draft", "approved"}
TASK_STATUSES = {
    "draft",
    "ready",
    "claimed",
    "running",
    "blocked",
    "completed",
    "review",
    "verified",
    "done",
}
SATISFIED_STATUSES = {"verified", "done"}
REVIEW_TYPES = {
    "decision_gate",
    "artifact_gate",
    "integration_gate",
    "release_gate",
}
REVIEW_STAGES = {
    "pre_execution",
    "post_artifact",
    "pre_integration",
    "pre_release",
}

TASK_EVENT_TRANSITIONS: dict[str, tuple[set[str], str | None]] = {
    "task.ready": ({"draft", "blocked"}, "ready"),
    "task.claimed": ({"ready"}, "claimed"),
    "task.started": ({"ready", "claimed"}, "running"),
    "task.progress": ({"claimed", "running"}, None),
    "task.blocked": ({"ready", "claimed", "running"}, "blocked"),
    "task.completed": ({"running"}, "completed"),
    "task.verified": ({"completed", "review"}, "verified"),
    "task.done": ({"verified"}, "done"),
    "task.reopened": (
        {"blocked", "completed", "review", "verified", "done"},
        "ready",
    ),
    "review.requested": ({"completed", "review"}, "review"),
}
NON_TRANSITION_EVENTS = {
    "plan.created",
    "plan.approved",
    "plan.revised",
    "review.approved",
    "review.rejected",
    "handoff.created",
    "handoff.accepted",
    "artifact.created",
    "focus.pinned",
    "focus.released",
}


class PlanError(ValueError):
    """Raised when a plan or event violates a deterministic invariant."""


@dataclass(frozen=True)
class ValidationResult:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: os.PathLike[str] | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise PlanError(f"Expected a JSON object in {path}")
    return value


def atomic_write_json(path: os.PathLike[str] | str, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: os.PathLike[str] | str, value: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def append_event(path: os.PathLike[str] | str, event: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(event), ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def load_events(path: os.PathLike[str] | str | None) -> list[dict[str, Any]]:
    if path is None or not Path(path).exists():
        return []
    events: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PlanError(f"Invalid JSONL event at line {line_number}: {exc}") from exc
            if not isinstance(event, dict):
                raise PlanError(f"Event at line {line_number} is not an object")
            events.append(event)
    return events


def _is_string_list(value: Any, *, nonempty: bool = False) -> bool:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return False
    return bool(value) if nonempty else True


def task_map(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(task.get("id")): task
        for task in plan.get("tasks", [])
        if isinstance(task, dict) and task.get("id")
    }


def repo_names_for_task(task: Mapping[str, Any]) -> set[str]:
    names = {str(task.get("repo", ""))}
    names.update(str(repo) for repo in task.get("affected_repos", []))
    names.discard("")
    return names


def _default_review_stage(review_type: str) -> str:
    return {
        "decision_gate": "pre_execution",
        "artifact_gate": "post_artifact",
        "integration_gate": "pre_integration",
        "release_gate": "pre_release",
    }.get(review_type, "post_artifact")


def validate_plan(
    plan: Mapping[str, Any],
    *,
    against: Mapping[str, Any] | None = None,
    allow_priority_changes: bool = False,
) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if plan.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if not isinstance(plan.get("plan_id"), str) or not plan.get("plan_id"):
        errors.append("plan_id must be a non-empty string")
    if not isinstance(plan.get("version"), int) or int(plan.get("version", 0)) < 1:
        errors.append("version must be an integer >= 1")
    if plan.get("status", "draft") not in PLAN_STATUSES:
        errors.append("plan status must be draft or approved")

    priorities = plan.get("priority_order", ["P0", "P1", "P2", "P3"])
    if not _is_string_list(priorities, nonempty=True) or len(set(priorities)) != len(priorities):
        errors.append("priority_order must contain unique priority labels")
        priorities = ["P0", "P1", "P2", "P3"]
    max_parallelism = plan.get("max_parallelism", 3)
    if not isinstance(max_parallelism, int) or not 1 <= max_parallelism <= 32:
        errors.append("max_parallelism must be an integer between 1 and 32")

    repos = plan.get("repos", {})
    if not isinstance(repos, dict):
        errors.append("repos must be an object keyed by repo name")
        repos = {}
    for name, repo in repos.items():
        if not isinstance(repo, dict):
            errors.append(f"repo {name!r} must be an object")
            continue
        if not isinstance(repo.get("snapshot_sha"), str) or not repo.get("snapshot_sha"):
            errors.append(f"repo {name!r} must declare snapshot_sha")
        if repo.get("dirty"):
            warnings.append(f"repo {name!r} is dirty; dependent tasks cannot be auto-dispatched")

    tasks = plan.get("tasks")
    if not isinstance(tasks, list):
        errors.append("tasks must be a list")
        tasks = []
    if not tasks:
        warnings.append("plan has no tasks")

    seen_ids: set[str] = set()
    all_ids = {
        task.get("id")
        for task in tasks
        if isinstance(task, dict) and isinstance(task.get("id"), str)
    }
    for index, task in enumerate(tasks):
        prefix = f"tasks[{index}]"
        if not isinstance(task, dict):
            errors.append(f"{prefix} must be an object")
            continue
        task_id = task.get("id")
        if not isinstance(task_id, str) or not task_id:
            errors.append(f"{prefix}.id must be a non-empty string")
            continue
        prefix = f"task {task_id!r}"
        if task_id in seen_ids:
            errors.append(f"duplicate task id {task_id!r}")
        seen_ids.add(task_id)

        for field in ("title", "goal", "repo", "context_bundle"):
            if not isinstance(task.get(field), str) or not task.get(field):
                errors.append(f"{prefix} must declare non-empty {field}")
        priority = task.get("declared_priority")
        if priority not in priorities:
            errors.append(f"{prefix} has unknown declared_priority {priority!r}")
        if task.get("actor") not in ACTORS:
            errors.append(f"{prefix} actor must be one of {sorted(ACTORS)}")
        if task.get("delegation_confidence") not in CONFIDENCE_LEVELS:
            errors.append(
                f"{prefix} delegation_confidence must be one of {sorted(CONFIDENCE_LEVELS)}"
            )
        if not _is_string_list(task.get("delegation_reasons"), nonempty=True):
            errors.append(f"{prefix} must include delegation_reasons")

        repo_name = task.get("repo")
        if repo_name not in repos:
            errors.append(f"{prefix} references unknown repo {repo_name!r}")
        repo_snapshot = repos.get(repo_name, {}).get("snapshot_sha") if isinstance(repos, dict) else None
        snapshot = task.get("snapshot_sha")
        if not isinstance(snapshot, str) or not snapshot:
            errors.append(f"{prefix} must declare snapshot_sha")
        elif snapshot in {"unknown", "unresolved"}:
            warnings.append(f"{prefix} has unresolved snapshot_sha")
        elif repo_snapshot and snapshot != repo_snapshot:
            warnings.append(
                f"{prefix} snapshot_sha differs from repo {repo_name!r}; refresh before dispatch"
            )

        dependencies = task.get("dependencies", [])
        if not _is_string_list(dependencies):
            errors.append(f"{prefix}.dependencies must be a list of task ids")
            dependencies = []
        for dependency in dependencies:
            if dependency == task_id:
                errors.append(f"{prefix} cannot depend on itself")
            elif dependency not in all_ids:
                errors.append(f"{prefix} depends on unknown task {dependency!r}")

        conflicts = task.get("conflicts_with", [])
        if not _is_string_list(conflicts):
            errors.append(f"{prefix}.conflicts_with must be a list of task ids")
        else:
            for conflict in conflicts:
                if conflict == task_id:
                    errors.append(f"{prefix} cannot conflict with itself")
                elif conflict not in all_ids:
                    errors.append(f"{prefix} conflicts with unknown task {conflict!r}")

        for field in (
            "affected_repos",
            "context_tags",
            "touch_scope",
            "shared_resources",
            "acceptance_criteria",
            "verification_commands",
            "expected_evidence",
        ):
            if not _is_string_list(task.get(field, [])):
                errors.append(f"{prefix}.{field} must be a list of strings")
        if not task.get("acceptance_criteria"):
            warnings.append(f"{prefix} has no acceptance criteria")
        if not task.get("expected_evidence"):
            warnings.append(f"{prefix} has no expected evidence")
        if task.get("actor") in {"codex", "hybrid"} and not task.get("verification_commands"):
            warnings.append(f"{prefix} is delegable but has no automatic verification command")
        if task.get("delegation_confidence") == "low":
            warnings.append(f"{prefix} has low delegation confidence and needs approval attention")

        affected_repos = task.get("affected_repos", [])
        for affected in affected_repos:
            if affected not in repos:
                errors.append(f"{prefix} references unknown affected repo {affected!r}")

        gates = task.get("review_gates", [])
        if not isinstance(gates, list):
            errors.append(f"{prefix}.review_gates must be a list")
            gates = []
        gate_ids: set[str] = set()
        gate_types: set[str] = set()
        for gate_index, gate in enumerate(gates):
            gate_prefix = f"{prefix}.review_gates[{gate_index}]"
            if not isinstance(gate, dict):
                errors.append(f"{gate_prefix} must be an object")
                continue
            gate_id = gate.get("id")
            if not isinstance(gate_id, str) or not gate_id:
                errors.append(f"{gate_prefix}.id must be a non-empty string")
            elif gate_id in gate_ids:
                errors.append(f"{prefix} has duplicate gate id {gate_id!r}")
            gate_ids.add(str(gate_id))
            gate_type = gate.get("type")
            gate_types.add(str(gate_type))
            if gate_type not in REVIEW_TYPES:
                errors.append(f"{gate_prefix}.type must be one of {sorted(REVIEW_TYPES)}")
            stage = gate.get("stage", _default_review_stage(str(gate_type)))
            if stage not in REVIEW_STAGES:
                errors.append(f"{gate_prefix}.stage must be one of {sorted(REVIEW_STAGES)}")
            if not isinstance(gate.get("question"), str) or not gate.get("question"):
                errors.append(f"{gate_prefix} must include the decision question")
            if not isinstance(gate.get("milestone"), str) or not gate.get("milestone"):
                errors.append(f"{gate_prefix} must include milestone")
            if not _is_string_list(gate.get("criteria", [])):
                errors.append(f"{gate_prefix}.criteria must be a list of strings")

        if len(repo_names_for_task(task)) > 1 and "integration_gate" not in gate_types:
            warnings.append(f"{prefix} spans repos but has no integration_gate")
        if "release_gate" in gate_types and task.get("actor") == "codex":
            warnings.append(f"{prefix} has a release_gate and should normally be hybrid or human")

        handoff = task.get("handoff_contract")
        if not isinstance(handoff, dict):
            errors.append(f"{prefix}.handoff_contract must be an object")
        else:
            for field in ("inputs", "outputs", "stop_conditions"):
                if not _is_string_list(handoff.get(field), nonempty=True):
                    errors.append(f"{prefix}.handoff_contract.{field} must be a non-empty list")

        effort = task.get("effort", 1)
        if not isinstance(effort, (int, float)) or effort <= 0:
            errors.append(f"{prefix}.effort must be a positive number")

    if not errors:
        try:
            topological_order(plan)
        except PlanError as exc:
            errors.append(str(exc))

    if against is not None:
        old_tasks = task_map(against)
        for task_id, new_task in task_map(plan).items():
            if task_id not in old_tasks:
                continue
            old_priority = old_tasks[task_id].get("declared_priority")
            new_priority = new_task.get("declared_priority")
            if old_priority != new_priority and not allow_priority_changes:
                errors.append(
                    f"task {task_id!r} changed declared_priority from {old_priority!r} "
                    f"to {new_priority!r}; require explicit user priority authority"
                )

    return ValidationResult(tuple(errors), tuple(warnings))


def topological_order(plan: Mapping[str, Any]) -> list[str]:
    tasks = task_map(plan)
    indegree = {task_id: 0 for task_id in tasks}
    children: dict[str, list[str]] = {task_id: [] for task_id in tasks}
    for task_id, task in tasks.items():
        for dependency in task.get("dependencies", []):
            if dependency not in tasks:
                raise PlanError(f"task {task_id!r} depends on unknown task {dependency!r}")
            indegree[task_id] += 1
            children[dependency].append(task_id)
    queue = deque(sorted(task_id for task_id, degree in indegree.items() if degree == 0))
    order: list[str] = []
    while queue:
        task_id = queue.popleft()
        order.append(task_id)
        for child in sorted(children[task_id]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(order) != len(tasks):
        cycle_nodes = sorted(task_id for task_id, degree in indegree.items() if degree > 0)
        raise PlanError(f"dependency graph contains a cycle involving: {', '.join(cycle_nodes)}")
    return order


def graph_children(plan: Mapping[str, Any]) -> dict[str, list[str]]:
    children: dict[str, list[str]] = {task_id: [] for task_id in task_map(plan)}
    for task_id, task in task_map(plan).items():
        for dependency in task.get("dependencies", []):
            children[dependency].append(task_id)
    for values in children.values():
        values.sort()
    return children


def compute_inherited_urgency(plan: Mapping[str, Any]) -> dict[str, str]:
    priorities = list(plan.get("priority_order", ["P0", "P1", "P2", "P3"]))
    rank = {priority: index for index, priority in enumerate(priorities)}
    tasks = task_map(plan)
    children = graph_children(plan)
    best = {task_id: rank[tasks[task_id]["declared_priority"]] for task_id in tasks}
    for task_id in reversed(topological_order(plan)):
        for child in children[task_id]:
            best[task_id] = min(best[task_id], best[child])
    return {task_id: priorities[value] for task_id, value in best.items()}


def compute_critical_path_scores(plan: Mapping[str, Any]) -> dict[str, float]:
    tasks = task_map(plan)
    children = graph_children(plan)
    score: dict[str, float] = {}
    for task_id in reversed(topological_order(plan)):
        tail = max((score[child] for child in children[task_id]), default=0.0)
        score[task_id] = float(tasks[task_id].get("effort", 1)) + tail
    return score


def critical_path(plan: Mapping[str, Any]) -> list[str]:
    tasks = task_map(plan)
    if not tasks:
        return []
    scores = compute_critical_path_scores(plan)
    children = graph_children(plan)
    roots = [task_id for task_id, task in tasks.items() if not task.get("dependencies")]
    current = max(roots, key=lambda item: (scores[item], item))
    path = [current]
    while children[current]:
        current = max(children[current], key=lambda item: (scores[item], item))
        path.append(current)
    return path


def _scope_tokens_conflict(left: str, right: str) -> bool:
    left = left.strip().rstrip("/")
    right = right.strip().rstrip("/")
    if not left or not right:
        return False
    if left == "*" or right == "*" or left == right:
        return True
    if ":" in left or ":" in right:
        return False
    left_prefix = left.removesuffix("/**").rstrip("/")
    right_prefix = right.removesuffix("/**").rstrip("/")
    return left_prefix.startswith(f"{right_prefix}/") or right_prefix.startswith(f"{left_prefix}/")


def conflict_matrix(plan: Mapping[str, Any]) -> dict[tuple[str, str], list[str]]:
    tasks = task_map(plan)
    task_ids = sorted(tasks)
    ancestors: dict[str, set[str]] = {task_id: set() for task_id in tasks}
    for task_id in topological_order(plan):
        for dependency in tasks[task_id].get("dependencies", []):
            ancestors[task_id].add(dependency)
            ancestors[task_id].update(ancestors[dependency])
    conflicts: dict[tuple[str, str], list[str]] = {}
    for left_index, left_id in enumerate(task_ids):
        left = tasks[left_id]
        for right_id in task_ids[left_index + 1 :]:
            right = tasks[right_id]
            if left_id in ancestors[right_id] or right_id in ancestors[left_id]:
                # A dependency already serializes the pair; conflict insight is redundant.
                continue
            reasons: list[str] = []
            if right_id in left.get("conflicts_with", []) or left_id in right.get("conflicts_with", []):
                reasons.append("explicit conflict")
            shared_resources = sorted(
                set(left.get("shared_resources", [])) & set(right.get("shared_resources", []))
            )
            if shared_resources:
                reasons.append(f"shared resource: {', '.join(shared_resources)}")
            if repo_names_for_task(left) & repo_names_for_task(right):
                scope_pairs = [
                    (left_scope, right_scope)
                    for left_scope in left.get("touch_scope", [])
                    for right_scope in right.get("touch_scope", [])
                    if _scope_tokens_conflict(left_scope, right_scope)
                ]
                if scope_pairs:
                    rendered = sorted({left_scope for left_scope, _ in scope_pairs} | {right for _, right in scope_pairs})
                    reasons.append(f"overlapping touch scope: {', '.join(rendered)}")
            if reasons:
                conflicts[(left_id, right_id)] = reasons
    return conflicts


def tasks_conflict(
    left_id: str,
    right_id: str,
    conflicts: Mapping[tuple[str, str], Sequence[str]],
) -> bool:
    return tuple(sorted((left_id, right_id))) in conflicts


def _task_sort_key(
    task_id: str,
    plan: Mapping[str, Any],
    inherited: Mapping[str, str],
    critical: Mapping[str, float],
) -> tuple[Any, ...]:
    tasks = task_map(plan)
    priority_order = list(plan.get("priority_order", ["P0", "P1", "P2", "P3"]))
    rank = {priority: index for index, priority in enumerate(priority_order)}
    task = tasks[task_id]
    return (
        rank[inherited[task_id]],
        rank[task["declared_priority"]],
        -critical[task_id],
        task.get("context_bundle", ""),
        task_id,
    )


def parallel_waves(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks = task_map(plan)
    if not tasks:
        return []
    inherited = compute_inherited_urgency(plan)
    critical = compute_critical_path_scores(plan)
    conflicts = conflict_matrix(plan)
    remaining = set(tasks)
    completed: set[str] = set()
    max_agents = int(plan.get("max_parallelism", 3))
    waves: list[dict[str, Any]] = []

    while remaining:
        ready = [
            task_id
            for task_id in remaining
            if set(tasks[task_id].get("dependencies", [])) <= completed
        ]
        if not ready:
            raise PlanError("unable to schedule tasks; dependency graph may be cyclic")
        ready.sort(key=lambda item: _task_sort_key(item, plan, inherited, critical))
        selected: list[str] = []
        human_used = False
        agent_count = 0
        for task_id in ready:
            task = tasks[task_id]
            is_human = task.get("actor") == "human"
            if is_human and human_used:
                continue
            if not is_human and agent_count >= max_agents:
                continue
            if any(tasks_conflict(task_id, other, conflicts) for other in selected):
                continue
            selected.append(task_id)
            if is_human:
                human_used = True
            else:
                agent_count += 1
        if not selected:
            selected = [ready[0]]

        lane_assignments: dict[str, str] = {}
        agent_lane = 1
        for task_id in selected:
            if tasks[task_id].get("actor") == "human":
                lane_assignments[task_id] = "human"
            else:
                lane_assignments[task_id] = f"codex-{agent_lane}"
                agent_lane += 1
        waves.append(
            {
                "wave": len(waves) + 1,
                "tasks": selected,
                "lanes": lane_assignments,
            }
        )
        remaining.difference_update(selected)
        completed.update(selected)
    return waves


def new_event(
    plan: Mapping[str, Any],
    event_type: str,
    actor: str,
    *,
    task_id: str | None = None,
    run_id: str | None = None,
    data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "event_id": str(uuid.uuid4()),
        "timestamp": utc_now(),
        "plan_id": plan["plan_id"],
        "plan_version": plan["version"],
        "event_type": event_type,
        "actor": actor,
        "data": dict(data or {}),
    }
    if task_id:
        event["task_id"] = task_id
    if run_id:
        event["run_id"] = run_id
    return event


def _gate_lookup(plan: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    gates: dict[tuple[str, str], dict[str, Any]] = {}
    for task_id, task in task_map(plan).items():
        for gate in task.get("review_gates", []):
            if isinstance(gate, dict) and gate.get("id"):
                normalized = dict(gate)
                normalized.setdefault("stage", _default_review_stage(str(gate.get("type"))))
                gates[(task_id, str(gate["id"]))] = normalized
    return gates


def replay_events(
    plan: Mapping[str, Any], events: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    tasks = task_map(plan)
    state: dict[str, Any] = {
        "plan_status": "draft",
        "preview": False,
        "task_statuses": {task_id: "draft" for task_id in tasks},
        "approved_gates": [],
        "rejected_gates": [],
        "requested_gates": [],
        "artifacts": defaultdict(list),
        "handoffs": defaultdict(list),
        "pinned_human_bundle": None,
        "last_event_at": None,
    }
    approved_gates: set[tuple[str, str]] = set()
    rejected_gates: set[tuple[str, str]] = set()
    requested_gates: set[tuple[str, str]] = set()
    gates = _gate_lookup(plan)

    for event in events:
        if event.get("plan_id") != plan.get("plan_id"):
            continue
        if int(event.get("plan_version", 0)) > int(plan.get("version", 0)):
            continue
        event_type = event.get("event_type")
        task_id = event.get("task_id")
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        state["last_event_at"] = event.get("timestamp")

        if event_type == "plan.created":
            state["plan_status"] = "draft"
        elif event_type == "plan.approved":
            state["plan_status"] = "approved"
        elif event_type == "plan.revised":
            state["plan_status"] = "draft"
        elif event_type in TASK_EVENT_TRANSITIONS and task_id in tasks:
            target_status = TASK_EVENT_TRANSITIONS[event_type][1]
            if target_status:
                state["task_statuses"][task_id] = target_status
            if event_type == "task.reopened":
                for (candidate_task_id, candidate_gate_id), candidate in gates.items():
                    if candidate_task_id == task_id and candidate.get("stage") != "pre_execution":
                        approved_gates.discard((candidate_task_id, candidate_gate_id))
                        requested_gates.discard((candidate_task_id, candidate_gate_id))
            if event_type == "review.requested":
                gate_id = str(data.get("gate_id", ""))
                if gate_id:
                    requested_gates.add((task_id, gate_id))
        elif event_type == "review.approved" and task_id in tasks:
            gate_id = str(data.get("gate_id", ""))
            if gate_id:
                approved_gates.add((task_id, gate_id))
                rejected_gates.discard((task_id, gate_id))
                requested_gates.discard((task_id, gate_id))
                gate = gates.get((task_id, gate_id), {})
                if gate.get("stage") != "pre_execution" and state["task_statuses"][task_id] == "review":
                    blocking_post_gates = {
                        (candidate_task_id, candidate_gate_id)
                        for (candidate_task_id, candidate_gate_id), candidate in gates.items()
                        if candidate_task_id == task_id
                        and candidate.get("stage") != "pre_execution"
                        and candidate.get("blocking", True)
                    }
                    if blocking_post_gates <= approved_gates:
                        state["task_statuses"][task_id] = "verified"
        elif event_type == "review.rejected" and task_id in tasks:
            gate_id = str(data.get("gate_id", ""))
            if gate_id:
                rejected_gates.add((task_id, gate_id))
                approved_gates.discard((task_id, gate_id))
                requested_gates.discard((task_id, gate_id))
                for (candidate_task_id, candidate_gate_id), candidate in gates.items():
                    if candidate_task_id == task_id and candidate.get("stage") != "pre_execution":
                        approved_gates.discard((candidate_task_id, candidate_gate_id))
                        requested_gates.discard((candidate_task_id, candidate_gate_id))
            if state["task_statuses"][task_id] == "review":
                state["task_statuses"][task_id] = "ready"
        elif event_type == "artifact.created" and task_id in tasks:
            state["artifacts"][task_id].append(data)
        elif event_type in {"handoff.created", "handoff.accepted"} and task_id in tasks:
            state["handoffs"][task_id].append({"event_type": event_type, **data})
        elif event_type == "focus.pinned":
            state["pinned_human_bundle"] = data.get("context_bundle")
        elif event_type == "focus.released":
            state["pinned_human_bundle"] = None

    state["approved_gates"] = sorted([list(item) for item in approved_gates])
    state["rejected_gates"] = sorted([list(item) for item in rejected_gates])
    state["requested_gates"] = sorted([list(item) for item in requested_gates])
    state["artifacts"] = dict(state["artifacts"])
    state["handoffs"] = dict(state["handoffs"])
    return state


def validate_event(
    plan: Mapping[str, Any],
    state: Mapping[str, Any],
    event_type: str,
    *,
    task_id: str | None = None,
    data: Mapping[str, Any] | None = None,
) -> None:
    tasks = task_map(plan)
    data = data or {}
    if event_type not in TASK_EVENT_TRANSITIONS and event_type not in NON_TRANSITION_EVENTS:
        raise PlanError(f"unsupported event type {event_type!r}")
    if event_type.startswith(("task.", "review.", "handoff.", "artifact.")):
        if not task_id or task_id not in tasks:
            raise PlanError(f"event {event_type!r} requires a known task_id")
    if event_type in TASK_EVENT_TRANSITIONS:
        current = state["task_statuses"][task_id]
        allowed, _ = TASK_EVENT_TRANSITIONS[event_type]
        if current not in allowed:
            raise PlanError(
                f"event {event_type!r} is invalid for task {task_id!r} in state {current!r}; "
                f"allowed states: {sorted(allowed)}"
            )
        if event_type == "task.ready" and not dependencies_satisfied(tasks[task_id], state):
            raise PlanError(f"task {task_id!r} cannot become ready before its dependencies are verified")
        if event_type == "task.verified":
            unresolved = []
            for gate in tasks[task_id].get("review_gates", []):
                stage = gate.get("stage", _default_review_stage(str(gate.get("type"))))
                if stage == "pre_execution" or not gate.get("blocking", True):
                    continue
                if [task_id, str(gate.get("id"))] not in state.get("approved_gates", []):
                    unresolved.append(str(gate.get("id")))
            if unresolved:
                raise PlanError(
                    f"task {task_id!r} cannot be verified before blocking review gates: "
                    f"{', '.join(unresolved)}"
                )
    if event_type.startswith("review."):
        gate_id = data.get("gate_id")
        if not gate_id or (str(task_id), str(gate_id)) not in _gate_lookup(plan):
            raise PlanError(f"event {event_type!r} requires a valid gate_id")
        gate = _gate_lookup(plan)[(str(task_id), str(gate_id))]
        current = state["task_statuses"][task_id]
        if event_type == "review.requested" and gate.get("stage") == "pre_execution":
            raise PlanError("pre-execution gates are ready by task state and do not use review.requested")
        if event_type in {"review.approved", "review.rejected"}:
            if gate.get("stage") == "pre_execution" and current != "ready":
                raise PlanError(
                    f"pre-execution gate {gate_id!r} requires task {task_id!r} to be ready"
                )
            if gate.get("stage") != "pre_execution" and current != "review":
                raise PlanError(
                    f"post-execution gate {gate_id!r} requires task {task_id!r} to be in review"
                )
    if event_type == "plan.approved" and state.get("plan_status") == "approved":
        raise PlanError("plan is already approved")


def dependencies_satisfied(
    task: Mapping[str, Any], state: Mapping[str, Any]
) -> bool:
    statuses = state["task_statuses"]
    return all(statuses.get(dependency) in SATISFIED_STATUSES for dependency in task.get("dependencies", []))


def newly_ready_task_ids(plan: Mapping[str, Any], state: Mapping[str, Any]) -> list[str]:
    if state.get("plan_status") != "approved":
        return []
    inherited = compute_inherited_urgency(plan)
    critical = compute_critical_path_scores(plan)
    tasks = task_map(plan)
    ready = [
        task_id
        for task_id, task in tasks.items()
        if state["task_statuses"].get(task_id) == "draft" and dependencies_satisfied(task, state)
    ]
    return sorted(ready, key=lambda item: _task_sort_key(item, plan, inherited, critical))


def _gate_is_approved(state: Mapping[str, Any], task_id: str, gate_id: str) -> bool:
    return [task_id, gate_id] in state.get("approved_gates", [])


def unresolved_pre_execution_gates(
    task_id: str, task: Mapping[str, Any], state: Mapping[str, Any]
) -> list[dict[str, Any]]:
    gates = []
    for raw_gate in task.get("review_gates", []):
        gate = dict(raw_gate)
        gate.setdefault("stage", _default_review_stage(str(gate.get("type"))))
        if (
            gate.get("stage") == "pre_execution"
            and gate.get("blocking", True)
            and not _gate_is_approved(state, task_id, str(gate.get("id")))
        ):
            gates.append(gate)
    return gates


def _repo_is_dirty(plan: Mapping[str, Any], task: Mapping[str, Any]) -> bool:
    repos = plan.get("repos", {})
    return any(bool(repos.get(repo_name, {}).get("dirty")) for repo_name in repo_names_for_task(task))


def _task_snapshot_is_stale(plan: Mapping[str, Any], task: Mapping[str, Any]) -> bool:
    repo = plan.get("repos", {}).get(task.get("repo"), {})
    return bool(repo.get("snapshot_sha")) and task.get("snapshot_sha") != repo.get("snapshot_sha")


def dispatch_queue(plan: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    tasks = task_map(plan)
    inherited = compute_inherited_urgency(plan)
    critical = compute_critical_path_scores(plan)
    conflicts = conflict_matrix(plan)
    candidates: list[str] = []
    waiting: list[dict[str, Any]] = []

    for task_id, task in tasks.items():
        status = state["task_statuses"].get(task_id, "draft")
        if task.get("actor") not in {"codex", "hybrid"}:
            continue
        if status != "ready":
            waiting.append({"task_id": task_id, "reason": f"status={status}"})
            continue
        gates = unresolved_pre_execution_gates(task_id, task, state)
        if gates:
            waiting.append(
                {
                    "task_id": task_id,
                    "reason": "blocking pre-execution review",
                    "gates": [gate["id"] for gate in gates],
                }
            )
            continue
        if _repo_is_dirty(plan, task):
            waiting.append({"task_id": task_id, "reason": "snapshot_dirty"})
            continue
        if _task_snapshot_is_stale(plan, task):
            waiting.append({"task_id": task_id, "reason": "snapshot_stale"})
            continue
        candidates.append(task_id)

    candidates.sort(key=lambda item: _task_sort_key(item, plan, inherited, critical))
    selected: list[str] = []
    for task_id in candidates:
        if len(selected) >= int(plan.get("max_parallelism", 3)):
            waiting.append({"task_id": task_id, "reason": "parallelism_limit"})
            continue
        blockers = [other for other in selected if tasks_conflict(task_id, other, conflicts)]
        if blockers:
            waiting.append(
                {
                    "task_id": task_id,
                    "reason": "conflict",
                    "conflicts_with": blockers,
                }
            )
            continue
        selected.append(task_id)
    if state.get("plan_status") != "approved":
        return {
            "dispatch_now": [],
            "proposed_after_approval": selected,
            "waiting": waiting,
            "reason": "plan_draft",
        }
    return {"dispatch_now": selected, "proposed_after_approval": [], "waiting": waiting}


def _static_state(
    plan: Mapping[str, Any], base_state: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    state = json.loads(json.dumps(base_state or replay_events(plan, [])))
    state["plan_status"] = "draft"
    state["preview"] = True
    tasks = task_map(plan)
    for task_id, task in tasks.items():
        if state["task_statuses"].get(task_id, "draft") != "draft":
            continue
        if dependencies_satisfied(task, state):
            state["task_statuses"][task_id] = "ready"
    return state


def review_batches(plan: Mapping[str, Any], state: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks = task_map(plan)
    batches: dict[str, dict[str, Any]] = {}
    requested = {tuple(item) for item in state.get("requested_gates", [])}
    for task_id, task in tasks.items():
        for raw_gate in task.get("review_gates", []):
            gate = dict(raw_gate)
            gate.setdefault("stage", _default_review_stage(str(gate.get("type"))))
            gate_id = str(gate["id"])
            if _gate_is_approved(state, task_id, gate_id):
                continue
            milestone = str(gate.get("milestone") or task.get("context_bundle"))
            batch = batches.setdefault(
                milestone,
                {
                    "milestone": milestone,
                    "context_bundles": set(),
                    "items": [],
                    "ready": False,
                },
            )
            status = state["task_statuses"].get(task_id, "draft")
            gate_ready = False
            if gate["stage"] == "pre_execution":
                gate_ready = status == "ready"
            elif (task_id, gate_id) in requested:
                gate_ready = True
            elif status in {"completed", "review"}:
                gate_ready = True
            batch["ready"] = batch["ready"] or gate_ready
            batch["context_bundles"].add(task.get("context_bundle"))
            batch["items"].append(
                {
                    "task_id": task_id,
                    "gate_id": gate_id,
                    "type": gate["type"],
                    "stage": gate["stage"],
                    "question": gate["question"],
                    "criteria": gate.get("criteria", []),
                    "ready": gate_ready,
                }
            )
    result = []
    for batch in batches.values():
        batch["context_bundles"] = sorted(item for item in batch["context_bundles"] if item)
        batch["items"].sort(key=lambda item: (not item["ready"], item["task_id"], item["gate_id"]))
        result.append(batch)
    return sorted(result, key=lambda item: (not item["ready"], item["milestone"]))


def human_focus_chain(plan: Mapping[str, Any], state: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks = task_map(plan)
    inherited = compute_inherited_urgency(plan)
    critical = compute_critical_path_scores(plan)
    priority_order = list(plan.get("priority_order", ["P0", "P1", "P2", "P3"]))
    rank = {priority: index for index, priority in enumerate(priority_order)}
    pinned = state.get("pinned_human_bundle")
    items: list[dict[str, Any]] = []

    for task_id, task in tasks.items():
        status = state["task_statuses"].get(task_id, "draft")
        if task.get("actor") == "human" and status in {"ready", "running", "blocked"}:
            items.append(
                {
                    "kind": "task",
                    "task_id": task_id,
                    "title": task["title"],
                    "context_bundle": task["context_bundle"],
                    "status": status,
                    "question": task["goal"],
                }
            )
        if status == "ready":
            for gate in unresolved_pre_execution_gates(task_id, task, state):
                items.append(
                    {
                        "kind": "review",
                        "task_id": task_id,
                        "gate_id": gate["id"],
                        "title": task["title"],
                        "context_bundle": task["context_bundle"],
                        "status": "review-ready",
                        "question": gate["question"],
                    }
                )
    for batch in review_batches(plan, state):
        for item in batch["items"]:
            if not item["ready"] or item["stage"] == "pre_execution":
                continue
            task = tasks[item["task_id"]]
            items.append(
                {
                    "kind": "review",
                    "task_id": item["task_id"],
                    "gate_id": item["gate_id"],
                    "title": task["title"],
                    "context_bundle": task["context_bundle"],
                    "status": "review-ready",
                    "question": item["question"],
                }
            )

    def sort_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
        task_id = str(item["task_id"])
        return (
            0 if pinned and item.get("context_bundle") == pinned else 1,
            rank[inherited[task_id]],
            -critical[task_id],
            item.get("context_bundle", ""),
            0 if item.get("kind") == "review" else 1,
            task_id,
        )

    items.sort(key=sort_key)
    return items


def plan_insights(plan: Mapping[str, Any], state: Mapping[str, Any]) -> list[dict[str, str]]:
    tasks = task_map(plan)
    children = graph_children(plan)
    inherited = compute_inherited_urgency(plan)
    conflicts = conflict_matrix(plan)
    insights: list[dict[str, str]] = []

    for task_id, task in tasks.items():
        descendants: set[str] = set()
        queue = deque(children[task_id])
        while queue:
            child = queue.popleft()
            if child in descendants:
                continue
            descendants.add(child)
            queue.extend(children[child])
        if len(descendants) >= 2 and (
            task.get("actor") in {"human", "hybrid"}
            or unresolved_pre_execution_gates(task_id, task, state)
        ):
            insights.append(
                {
                    "kind": "parallel-unlocker",
                    "message": f"{task_id} is a decision/unlocker for {len(descendants)} downstream tasks.",
                }
            )
        if inherited[task_id] != task.get("declared_priority"):
            insights.append(
                {
                    "kind": "inherited-urgency",
                    "message": (
                        f"{task_id} keeps declared priority {task.get('declared_priority')} but inherits "
                        f"urgency {inherited[task_id]} from a higher-priority descendant."
                    ),
                }
            )
        if not task.get("acceptance_criteria") or not task.get("expected_evidence"):
            insights.append(
                {
                    "kind": "weak-verification",
                    "message": f"{task_id} needs stronger acceptance criteria or expected evidence.",
                }
            )
        if task.get("delegation_confidence") == "low":
            insights.append(
                {
                    "kind": "low-confidence-delegation",
                    "message": f"{task_id} actor assignment needs explicit approval attention.",
                }
            )

    for repo_name, repo in plan.get("repos", {}).items():
        affected = sorted(
            task_id
            for task_id, task in tasks.items()
            if repo_name in repo_names_for_task(task)
        )
        if repo.get("dirty"):
            changed = repo.get("changed_paths", [])
            detail = f"; changed paths: {', '.join(changed[:5])}" if changed else ""
            insights.append(
                {
                    "kind": "snapshot-risk",
                    "message": (
                        f"Repo {repo_name} is dirty, blocking safe auto-dispatch for "
                        f"{', '.join(affected)}{detail}."
                    ),
                }
            )
        stale = sorted(
            task_id
            for task_id, task in tasks.items()
            if task.get("repo") == repo_name and _task_snapshot_is_stale(plan, task)
        )
        if stale:
            insights.append(
                {
                    "kind": "snapshot-drift",
                    "message": f"Refresh task snapshots for repo {repo_name}: {', '.join(stale)}.",
                }
            )

    for (left, right), reasons in conflicts.items():
        insights.append(
            {
                "kind": "parallel-conflict",
                "message": f"{left} and {right} must not share a wave: {'; '.join(reasons)}.",
            }
        )
    for batch in review_batches(plan, state):
        if len(batch["items"]) >= 3:
            insights.append(
                {
                    "kind": "review-pressure",
                    "message": (
                        f"Milestone {batch['milestone']} batches {len(batch['items'])} review items; "
                        "keep one context-loaded checkpoint instead of individual interruptions."
                    ),
                }
            )
    if not insights:
        insights.append({"kind": "healthy-plan", "message": "No structural planning risks detected."})
    return insights


def _safe_mermaid_id(prefix: str, raw: str) -> str:
    return f"{prefix}_{re.sub(r'[^A-Za-z0-9_]', '_', raw)}"


def _escape_mermaid_label(value: Any) -> str:
    return str(value).replace('"', "'").replace("\n", " ")


def render_mermaid(plan: Mapping[str, Any]) -> str:
    tasks = task_map(plan)
    waves = parallel_waves(plan)
    lane_by_task: dict[str, str] = {}
    for wave in waves:
        lane_by_task.update(wave["lanes"])
    max_agents = int(plan.get("max_parallelism", 3))
    lines = ["flowchart LR"]

    lines.append('  subgraph HUMAN["Human Focus / Review Lane"]')
    for task_id, task in tasks.items():
        if task.get("actor") == "human":
            node = _safe_mermaid_id("task", task_id)
            label = _escape_mermaid_label(f"{task_id} · {task['title']}")
            lines.append(f'    {node}["{label}"]')
    for task_id, task in tasks.items():
        for raw_gate in task.get("review_gates", []):
            gate = dict(raw_gate)
            gate_id = _safe_mermaid_id("gate", f"{task_id}_{gate['id']}")
            label = _escape_mermaid_label(f"{gate['type']}: {gate['question']}")
            lines.append(f'    {gate_id}{{"{label}"}}')
    lines.append("  end")

    for lane_number in range(1, max_agents + 1):
        lane_name = f"codex-{lane_number}"
        lines.append(f'  subgraph C{lane_number}["Codex Lane {lane_number}"]')
        for task_id, task in tasks.items():
            if lane_by_task.get(task_id) != lane_name:
                continue
            node = _safe_mermaid_id("task", task_id)
            label = _escape_mermaid_label(f"{task_id} · {task['title']}")
            lines.append(f'    {node}["{label}"]')
        lines.append("  end")

    for task_id, task in tasks.items():
        node = _safe_mermaid_id("task", task_id)
        for dependency in task.get("dependencies", []):
            dependency_node = _safe_mermaid_id("task", dependency)
            lines.append(f"  {dependency_node} --> {node}")
        for raw_gate in task.get("review_gates", []):
            gate = dict(raw_gate)
            gate.setdefault("stage", _default_review_stage(str(gate.get("type"))))
            gate_node = _safe_mermaid_id("gate", f"{task_id}_{gate['id']}")
            if gate["stage"] == "pre_execution":
                lines.append(f"  {gate_node} --> {node}")
            else:
                lines.append(f"  {node} --> {gate_node}")

    for (left, right), _reasons in conflict_matrix(plan).items():
        left_node = _safe_mermaid_id("task", left)
        right_node = _safe_mermaid_id("task", right)
        lines.append(f'  {left_node} -. "conflict" .- {right_node}')

    lines.extend(
        [
            "  classDef codex fill:#E8F1FF,stroke:#2563EB,color:#111827",
            "  classDef hybrid fill:#FEF3C7,stroke:#D97706,color:#111827",
            "  classDef human fill:#FCE7F3,stroke:#DB2777,color:#111827",
            "  classDef review fill:#FFF7ED,stroke:#EA580C,color:#111827",
        ]
    )
    for task_id, task in tasks.items():
        lines.append(f"  class {_safe_mermaid_id('task', task_id)} {task.get('actor')}")
        for gate in task.get("review_gates", []):
            gate_key = f"{task_id}_{gate['id']}"
            lines.append(f"  class {_safe_mermaid_id('gate', gate_key)} review")
    return "\n".join(lines) + "\n"


def _markdown_list(values: Sequence[str], empty: str = "None") -> str:
    return ", ".join(values) if values else empty


def render_dashboard(plan: Mapping[str, Any], state: Mapping[str, Any]) -> str:
    tasks = task_map(plan)
    inherited = compute_inherited_urgency(plan)
    path = critical_path(plan)
    queue = dispatch_queue(plan, state)
    focus = human_focus_chain(plan, state)
    batches = review_batches(plan, state)
    insights = plan_insights(plan, state)
    lines = [
        f"# Work Plan: {plan.get('title', plan['plan_id'])}",
        "",
        f"- Plan: `{plan['plan_id']}` v{plan['version']} ({state['plan_status']})",
        f"- Critical path: `{_markdown_list(path)}`",
        f"- Safe Codex parallelism: {plan.get('max_parallelism', 3)}",
        f"- {'Proposed' if state.get('preview') else 'Ready'} review batches: {sum(1 for batch in batches if batch['ready'])}",
        "",
        "## Flow",
        "",
        "```mermaid",
        render_mermaid(plan).rstrip(),
        "```",
        "",
        "## Human Focus Chain",
        "",
    ]
    if focus:
        for index, item in enumerate(focus, start=1):
            lines.append(
                f"{index}. **{item['task_id']} · {item['title']}** "
                f"— `{item['context_bundle']}` — {item['question']}"
            )
    else:
        lines.append("No human task or review is ready.")

    queue_title = "Codex Dispatch Queue" if state["plan_status"] == "approved" else "Proposed Codex Queue (after approval)"
    lines.extend(["", f"## {queue_title}", ""])
    displayed_queue = queue["dispatch_now"] or queue.get("proposed_after_approval", [])
    if displayed_queue:
        for task_id in displayed_queue:
            task = tasks[task_id]
            lines.append(
                f"- **{task_id} · {task['title']}** — {task['actor']} / "
                f"{task['delegation_confidence']} confidence — `{task['context_bundle']}`"
            )
    else:
        lines.append("No Codex task is dispatchable now.")
    if queue["waiting"]:
        lines.append("")
        lines.append("Waiting:")
        for item in queue["waiting"]:
            lines.append(f"- `{item['task_id']}` — {item['reason']}")

    lines.extend(["", "## Review Checkpoints", ""])
    if batches:
        for batch in batches:
            status = "ready" if batch["ready"] else "planned"
            lines.append(f"### {batch['milestone']} ({status})")
            lines.append("")
            for item in batch["items"]:
                ready = "READY" if item["ready"] else "later"
                lines.append(
                    f"- [{ready}] `{item['task_id']}/{item['gate_id']}` "
                    f"{item['type']}: {item['question']}"
                )
            lines.append("")
    else:
        lines.append("No review gates.")

    lines.extend(["", "## Insights", ""])
    for insight in insights:
        lines.append(f"- **{insight['kind']}** — {insight['message']}")

    lines.extend(["", "## Delegation, Review & Handoff", ""])
    for task_id in topological_order(plan):
        task = tasks[task_id]
        lines.extend(
            [
                f"### {task_id} · {task['title']}",
                "",
                f"- Assignment: **{task['actor']}** ({task['delegation_confidence']} confidence)",
                f"- Why: {_markdown_list(task.get('delegation_reasons', []))}",
                f"- Automatic evidence: {_markdown_list(task.get('verification_commands', []), 'No command; human evidence required')}",
            ]
        )
        gates = task.get("review_gates", [])
        if gates:
            lines.append(
                "- Human review: "
                + "; ".join(f"{gate['type']} — {gate['question']}" for gate in gates)
            )
        else:
            lines.append("- Human review: None before automatic verification")
        handoff = task.get("handoff_contract", {})
        lines.extend(
            [
                f"- Handoff outputs: {_markdown_list(handoff.get('outputs', []))}",
                f"- Stop and hand back if: {_markdown_list(handoff.get('stop_conditions', []))}",
                "",
            ]
        )

    lines.extend(
        [
            "",
            "## Tasks",
            "",
            "| Task | Priority | Urgency | Actor | State | Context | Dependencies | Review |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for task_id in topological_order(plan):
        task = tasks[task_id]
        gates = [gate["type"] for gate in task.get("review_gates", [])]
        task_status = state["task_statuses"].get(task_id, "draft")
        if state.get("preview"):
            task_status = "proposed-ready" if task_status == "ready" else "proposed-waiting"
        lines.append(
            f"| {task_id} · {task['title']} | {task['declared_priority']} | "
            f"{inherited[task_id]} | {task['actor']} | "
            f"{task_status} | {task['context_bundle']} | "
            f"{_markdown_list(task.get('dependencies', []))} | {_markdown_list(gates)} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def render_review_queue(plan: Mapping[str, Any], state: Mapping[str, Any]) -> str:
    tasks = task_map(plan)
    lines = ["# Review Queue", ""]
    batches = review_batches(plan, state)
    ready_batches = [batch for batch in batches if batch["ready"]]
    if not ready_batches:
        lines.append("No review checkpoint is ready.")
    for batch in ready_batches:
        lines.extend(
            [
                f"## {batch['milestone']}",
                "",
                f"Context: `{_markdown_list(batch['context_bundles'])}`",
                "",
            ]
        )
        for item in batch["items"]:
            if not item["ready"]:
                continue
            lines.append(f"- **{item['task_id']} / {item['gate_id']}**: {item['question']}")
            for criterion in item["criteria"]:
                lines.append(f"  - {criterion}")
            task = tasks[item["task_id"]]
            lines.append(
                f"  - Expected evidence: {_markdown_list(task.get('expected_evidence', []))}"
            )
            artifacts = state.get("artifacts", {}).get(item["task_id"], [])
            if artifacts:
                lines.append(f"  - Recorded artifacts: {json.dumps(artifacts, ensure_ascii=False)}")
            handoffs = state.get("handoffs", {}).get(item["task_id"], [])
            if handoffs:
                lines.append(f"  - Recorded handoffs: {json.dumps(handoffs, ensure_ascii=False)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_blocked(plan: Mapping[str, Any], state: Mapping[str, Any]) -> str:
    tasks = task_map(plan)
    lines = ["# Blocked Work", ""]
    blocked = [
        task_id
        for task_id, status in state["task_statuses"].items()
        if status == "blocked"
    ]
    if not blocked:
        lines.append("No task is blocked.")
    for task_id in blocked:
        lines.append(f"- **{task_id} · {tasks[task_id]['title']}**")
    dirty_tasks = [
        task_id for task_id, task in tasks.items() if _repo_is_dirty(plan, task)
    ]
    if dirty_tasks:
        lines.extend(["", "## Snapshot Risks", ""])
        for task_id in dirty_tasks:
            lines.append(f"- `{task_id}` targets a dirty repo and requires a clean execution base.")
    return "\n".join(lines).rstrip() + "\n"


def render_views(
    plan: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    output_dir: os.PathLike[str] | str,
) -> dict[str, str]:
    state = replay_events(plan, events)
    if plan.get("status") == "draft" and state.get("plan_status") == "draft":
        state = _static_state(plan, state)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "dashboard": output / "dashboard.md",
        "mermaid": output / "plan.mmd",
        "review_queue": output / "review-queue.md",
        "blocked": output / "blocked.md",
        "runtime_state": output / "runtime-state.json",
    }
    atomic_write_text(files["dashboard"], render_dashboard(plan, state))
    atomic_write_text(files["mermaid"], render_mermaid(plan))
    atomic_write_text(files["review_queue"], render_review_queue(plan, state))
    atomic_write_text(files["blocked"], render_blocked(plan, state))
    atomic_write_json(files["runtime_state"], state)
    return {key: str(path) for key, path in files.items()}


def trace_task(events: Sequence[Mapping[str, Any]], plan_id: str, task_id: str) -> list[dict[str, Any]]:
    return [
        dict(event)
        for event in events
        if event.get("plan_id") == plan_id and event.get("task_id") == task_id
    ]


def inspect_repo(path: os.PathLike[str] | str) -> dict[str, Any]:
    """Inspect a Git repo without mutating it; imported lazily to keep core lightweight."""
    import subprocess

    repo = Path(path).expanduser().resolve()
    if not repo.exists():
        return {"path": str(repo), "exists": False, "snapshot_sha": "unresolved", "dirty": True}
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"path": str(repo), "exists": True, "snapshot_sha": "unresolved", "dirty": True}
    changed_paths = []
    for line in status.splitlines():
        if len(line) >= 4:
            changed_paths.append(line[3:])
    return {
        "path": str(repo),
        "exists": True,
        "snapshot_sha": sha,
        "dirty": bool(status.strip()),
        "changed_paths": changed_paths,
    }
