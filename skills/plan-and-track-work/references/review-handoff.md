# Review and handoff contracts

## Review gates

Use the smallest gate that protects the real boundary.

### Decision gate

Place before execution when an unresolved architecture, product, contract, or safety choice would make
downstream work wasteful. Ask one precise question and state what each answer unlocks.

### Artifact gate

Use for subjective output quality that automatic checks cannot prove, such as research synthesis,
copy, visuals, or user-facing behavior. Attach the artifact and a short rubric.

### Integration gate

Batch related backend/frontend/schema changes into one context-loaded checkpoint. Include the combined
diff, compatibility evidence, test results, and rollback boundary.

### Release gate

Require explicit human authority before merge, push, release, migration, production configuration, or
another consequential external mutation. Provide a rollout and rollback checklist.

## Review batching

- Group gates by milestone and context bundle.
- Default to one checkpoint after a dependency layer or coherent artifact bundle.
- Do not notify on every worker completion.
- Raise an early checkpoint only when every higher-priority path is waiting on the same decision.
- Present the exact decision first, then evidence and residual risk.

## Worker context pack

Give a worker only:

```json
{
  "task_id": "TASK-01",
  "goal": "...",
  "acceptance_criteria": ["..."],
  "repo": {"path": "/absolute/path", "snapshot_sha": "...", "worktree": "..."},
  "relevant_files": ["..."],
  "approved_decisions": ["..."],
  "constraints": ["..."],
  "upstream_artifacts": ["..."],
  "verification_commands": ["..."],
  "allowed_actions": ["local edits", "tests", "local commit"],
  "stop_conditions": ["..."]
}
```

Do not include unrelated workstreams or the full parent conversation.

## Worker return packet

Require this shape:

```json
{
  "task_id": "TASK-01",
  "run_id": "...",
  "outcome": "completed|blocked|needs-review",
  "summary": "...",
  "artifacts": ["commit/diff/file paths"],
  "verification": [{"command": "...", "result": "pass|fail", "evidence": "..."}],
  "deviations": ["..."],
  "residual_risks": ["..."],
  "unblocked_tasks": ["..."],
  "decision_needed": null
}
```

The coordinator validates the packet, records events serially, and updates generated views. A worker
must not claim completion without the declared evidence.

## Worktree boundary

- Create one worktree and `codex/` branch per concurrent coding task.
- Base it on the task's recorded snapshot or an explicitly approved predecessor commit.
- Do not include uncommitted user changes implicitly.
- Do not let two workers edit a shared contract or file scope in the same wave.
- Allow workers to commit locally for a reviewable handoff; do not push, merge, or release without the
  corresponding gate.
