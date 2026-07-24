---
name: plan-and-track-work
description: Plan and trace complex work from repository context and user-prioritized tasks. Use when Codex needs to create or revise a dependency DAG and Mermaid workflow, minimize human context switching, decide which tasks Codex can execute versus which need human judgment, batch review checkpoints, prepare handoffs, dispatch approved parallel workers, report the next focus, record blockers or completions, or reconstruct a task trace across one or more repositories.
---

# Plan and Track Work

Turn repo context and user-owned priorities into a versioned plan with one human focus lane,
up to three safe Codex lanes, explicit review gates, and an append-only trace.

## Operating contract

- Preserve every `declared_priority`; change it only when the user explicitly changes it.
- Separate proposed plans from approved execution. Never dispatch from a draft.
- Keep the current human context bundle pinned until it finishes or blocks.
- Batch model outputs at milestone review checkpoints. Interrupt early only when all higher-priority
  work is blocked by one human decision.
- Treat the coordinator as the only writer to `events.jsonl`. Workers return handoffs; they do not
  edit shared planning state.
- Never push, merge, release, migrate, or perform another externally consequential action without
  the matching human gate.

## Workflow

### 1. Ground the plan

1. Find the control repo. Default to the current `life` repo when it contains `work-planning/`.
2. Read the target repos' instructions, architecture, Git status, current SHA, tests, and relevant
   workstream material. Do not silently exclude dirty worktree context.
3. Read [task-schema.md](references/task-schema.md) before creating or changing plan JSON.
4. Read [delegation-policy.md](references/delegation-policy.md) before assigning an actor or writing
   plan insights.
5. Read [review-handoff.md](references/review-handoff.md) before creating gates or dispatch packets.

Inspect repo snapshots deterministically:

```bash
python3 <skill-dir>/scripts/work_planner.py inspect-repos \
  --repo recent-master=/absolute/path/to/recent-master \
  --repo tab-web=/absolute/path/to/tab-web
```

After writing repo paths into a draft, refresh its SHA and dirty-path snapshot before validation:

```bash
python3 <skill-dir>/scripts/work_planner.py snapshot \
  --plan /path/to/draft.json --output work-planning/state/plan.json
```

Never run `snapshot` directly on an approved plan. Create a replan candidate so repo drift remains an
explicit versioned decision.

### 2. Create a draft

Translate the user's tasks into `work-planning/state/plan.json`. Mark semantic inferences clearly:

- hard dependencies versus soft ordering;
- context bundle and touch scope;
- `codex`, `hybrid`, or `human`, with confidence and reasons;
- automatic evidence and human review gates;
- a minimal handoff contract and stop conditions.

Use explicit task and resource conflicts. Do not encode preferences as fake hard dependencies.
Validate and render the draft:

```bash
python3 <skill-dir>/scripts/work_planner.py validate \
  --plan work-planning/state/plan.json
python3 <skill-dir>/scripts/work_planner.py create \
  --plan work-planning/state/plan.json \
  --events work-planning/traces/events.jsonl
python3 <skill-dir>/scripts/work_planner.py render \
  --plan work-planning/state/plan.json \
  --events work-planning/traces/events.jsonl \
  --output-dir work-planning/views
```

Record `plan.created` once per draft version. If a draft has already been recorded, do not append a
duplicate event.

Present the generated dashboard's execution summary, Mermaid graph, human focus chain, Codex queue,
review batches, and insights. Surface every validation warning and low-confidence inference before
approval.

### 3. Approve and dispatch

Run approval only after explicit user approval:

```bash
python3 <skill-dir>/scripts/work_planner.py approve \
  --plan work-planning/state/plan.json \
  --events work-planning/traces/events.jsonl \
  --actor user
```

Then replay state and dispatch only `dispatch_now` tasks from the generated runtime view.

- Use at most `max_parallelism` workers; default to three.
- Give each coding worker its own Git worktree and `codex/` branch.
- Never co-schedule tasks listed as conflicting.
- Give each worker only its context pack and handoff contract.
- Ask workers to stop at their declared stop conditions and return structured results.
- Record `claimed`, `started`, progress, artifact, handoff, completion, verification, and done events
  through the coordinator.

Record an event:

```bash
python3 <skill-dir>/scripts/work_planner.py record \
  --plan work-planning/state/plan.json \
  --events work-planning/traces/events.jsonl \
  --event task.blocked --task TASK-ID --actor codex-worker \
  --reason "blocking reason" --run-id RUN-ID
```

After every material event, render the views again. Treat `views/dashboard.md` and related files as
materialized views, never as independently editable state.

### 4. Review and hand off

When a review batch becomes ready, give the user one compact packet per milestone:

- the exact decision requested;
- related tasks and context bundle;
- diffs, artifacts, tests, and deviations;
- residual risks and what the decision unlocks.

Record the selected gate with `review.approved` or `review.rejected`. A rejected post-execution review
reopens the task. Do not make the user reconstruct context from worker chat.

### 5. Replan or trace

Create a candidate plan and validate it against the current plan. Preserve priorities unless the user
explicitly authorized a priority edit:

```bash
python3 <skill-dir>/scripts/work_planner.py replan \
  --current work-planning/state/plan.json \
  --candidate /path/to/candidate.json \
  --output work-planning/state/plan.json \
  --events work-planning/traces/events.jsonl
```

Use `--priority-change-authority user` only after an explicit user change. A replan creates a draft
version and preserves the event-pinned human bundle until it finishes or blocks.

Show one task's causal history with:

```bash
python3 <skill-dir>/scripts/work_planner.py trace \
  --plan work-planning/state/plan.json \
  --events work-planning/traces/events.jsonl --task TASK-ID
```

## Failure rules

- Reject cycles, unknown tasks, invalid actors, invalid transitions, and undeclared priorities.
- Allow planning against a dirty repo, but block automatic dispatch until the execution base is
  explicit and clean.
- Ask for a decision when a low-confidence dependency, actor, or gate would change the critical path.
- If collaboration workers or worktrees are unavailable, return the dispatch queue and handoff packs
  without pretending execution occurred.
- Import old Markdown tracking once as context; never overwrite it or maintain two editable sources of
  truth.
