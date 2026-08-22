# Delegation and insight policy

Assign work by explainable evidence. Do not use a hidden aggregate score.

## Actor decision

Choose `codex` when all material conditions hold:

- the goal and acceptance criteria are concrete;
- required repo context and tools are accessible;
- the operation is local, reversible, or isolated in a worktree;
- tests or deterministic evidence can verify the result;
- the task does not require human authority or stakeholder judgment.

Choose `hybrid` when Codex can do most of the work but a human decision or subjective review controls
the boundary. Typical cases include architecture, public API/schema changes, UX quality, cross-repo
contracts, ambiguous migrations, and high-impact integration.

Choose `human` when the task's primary value is judgment or authority: priority changes, product
tradeoffs, personnel communication, credentials, security/privacy decisions, production rollout,
legal/compliance decisions, or irreversible actions. Codex may still prepare analysis and checklists.

## Confidence

- `high`: repo evidence, acceptance, verification, and authority boundary are explicit.
- `medium`: one bounded assumption needs approval but the execution path is otherwise clear.
- `low`: missing context could change the actor, dependency graph, or critical path.

Low confidence must appear in the approval summary. Ask before dispatch when it affects a hard
dependency, blocking gate, or critical path.

## Plan insights

Generate insights at three levels.

### Task

- Why the actor is appropriate.
- What Codex can self-verify.
- What remains a human judgment.
- What would force an early stop or handoff.

### Bundle

- Tasks that should stay continuous because they share repo/module/tools/mental mode.
- Tasks that appear parallel but share files, contracts, environments, or reviewers.
- A decision task that should move earlier because it unlocks several model tasks.
- Review amplification: several workers producing more output than one human can absorb.

### Plan

- Critical path and highest-value unlocker.
- Safe parallelism ceiling and conflict bottlenecks.
- Human focus chain and expected review checkpoints.
- Weak acceptance criteria, dirty snapshots, low-confidence inferences, and priority inheritance.

## Context continuity

Build `context_bundle` primarily from explicit properties:

1. repo and workstream;
2. module, files, schema, or contract;
3. tools and runtime;
4. mental mode: research, design, implementation, debugging, review, or communication;
5. unresolved decisions that should be closed before switching.

Prefer finishing one bundle before switching. Do not continuously optimize a running human task.
Replan it only after completion, blocking, explicit priority change, failed verification, or repo drift.

## Priority policy

- The user owns `declared_priority`.
- Dependencies may inherit a higher downstream urgency without changing their declared priority.
- Prefer ready work in the highest inherited-urgency band.
- Within a band, prefer the critical-path unlocker, then context continuity.
- Never invent a hard dependency solely to force preferred ordering.
