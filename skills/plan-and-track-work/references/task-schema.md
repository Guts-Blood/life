# Task and plan schema

Use JSON as the canonical machine state and Markdown/Mermaid only as generated views.

## Plan object

```json
{
  "schema_version": 1,
  "plan_id": "stable-plan-id",
  "title": "Human-readable title",
  "version": 1,
  "status": "draft",
  "created_at": "ISO-8601 timestamp",
  "updated_at": "ISO-8601 timestamp",
  "max_parallelism": 3,
  "priority_order": ["P0", "P1", "P2", "P3"],
  "repos": {
    "repo-name": {
      "path": "/absolute/path",
      "snapshot_sha": "git SHA",
      "dirty": false,
      "changed_paths": []
    }
  },
  "tasks": []
}
```

- Keep `plan_id` stable across revisions and increment `version`.
- Keep `status=draft` until explicit approval.
- Preserve `priority_order`; lower array indexes are more urgent.
- Capture dirty paths when present. Planning is allowed; automatic dispatch is not.

## Task object

```json
{
  "id": "TASK-01",
  "title": "Implement backend contract",
  "goal": "One outcome, not a list of activities",
  "declared_priority": "P0",
  "actor": "codex",
  "delegation_confidence": "high",
  "delegation_reasons": ["Clear contract", "Automated tests exist"],
  "repo": "recent-master",
  "affected_repos": [],
  "snapshot_sha": "git SHA",
  "context_bundle": "auto-skill-contract",
  "context_tags": ["backend", "service-api"],
  "mental_mode": "implementation",
  "dependencies": ["TASK-00"],
  "conflicts_with": [],
  "touch_scope": ["src/service/**", "contract:auto-skill-v1"],
  "shared_resources": ["integration-test-env"],
  "acceptance_criteria": ["Endpoint satisfies the approved contract"],
  "verification_commands": ["pytest tests/service -q"],
  "expected_evidence": ["Passing test output", "Commit SHA"],
  "review_gates": [],
  "handoff_contract": {
    "inputs": ["Approved API contract"],
    "outputs": ["Commit", "Test evidence", "Residual risks"],
    "stop_conditions": ["Contract is ambiguous", "Test environment is unavailable"]
  },
  "risk": "low",
  "reversible": true,
  "requires_external_authority": false,
  "effort": 1
}
```

## Derived fields

Do not ask the user to maintain these fields manually:

- `inherited_urgency`: highest downstream urgency propagated to prerequisites.
- critical-path score and path.
- ready/blocked state reconstructed from events.
- parallel wave and Codex lane.
- automatic conflict reasons from overlapping `touch_scope` or `shared_resources`.

Never replace `declared_priority` with `inherited_urgency`.

## Review gate object

```json
{
  "id": "contract-decision",
  "type": "decision_gate",
  "stage": "pre_execution",
  "blocking": true,
  "milestone": "contract-approved",
  "question": "Is this compatibility boundary acceptable?",
  "criteria": ["Old clients keep working", "Rollback is explicit"]
}
```

Allowed types and default stages:

- `decision_gate` → `pre_execution`
- `artifact_gate` → `post_artifact`
- `integration_gate` → `pre_integration`
- `release_gate` → `pre_release`

## Event object

Each JSONL line contains:

```json
{
  "event_id": "UUID",
  "timestamp": "ISO-8601 timestamp",
  "plan_id": "stable-plan-id",
  "plan_version": 1,
  "event_type": "task.completed",
  "actor": "codex-worker",
  "task_id": "TASK-01",
  "run_id": "optional run id",
  "data": {"reason": "optional structured context"}
}
```

The coordinator is the single writer. Workers return handoff artifacts instead of appending events.

Task lifecycle:

`draft → ready → claimed → running → blocked/completed → review → verified → done`

Use `task.reopened` to return blocked, completed, review, verified, or done work to `ready`.
