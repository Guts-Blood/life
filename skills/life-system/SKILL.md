---
name: life-system
description: Use when working inside this personal life management repository to turn thoughts into structured assumptions, design lightweight verification plans, connect evidence to decisions, and promote validated conclusions into personal principles.
---

# Life System Skill

This skill helps maintain a personal repository organized around:

`context -> assumption -> verification -> evidence -> conclusion -> principle`

Use it whenever the user wants to add a life idea, refine a vague intuition, review a past conclusion, or update personal context.

## Objectives

- Preserve the user's personal context without overfitting every conclusion to it.
- Convert vague ideas into testable assumptions.
- Prefer low-cost verification before large commitments.
- Keep evidence separate from conclusions.
- Promote only durable, decision-relevant ideas into principles.

## Repository Map

- `profile/`: stable and seasonal context
- `assumptions/inbox/`: raw assumptions not yet shaped
- `assumptions/active/`: assumptions currently being tested
- `assumptions/verified/`: assumptions supported enough to use
- `assumptions/falsified/`: assumptions that did not hold
- `experiments/active/`: current verification plans
- `experiments/completed/`: finished tests
- `evidence/`: logs, references, and snapshots
- `principles/`: durable rules for future decisions
- `reviews/`: weekly, monthly, yearly maintenance
- `templates/`: note templates

## Default Workflow

### 1. Classify the input

Treat the user's input as one of:

- profile update
- new assumption
- experiment design
- evidence capture
- principle synthesis
- periodic review

If the input is mixed, split it into the smallest useful units.

### 2. Clarify the claim

When handling an assumption:

- rewrite it into a statement that could be wrong
- identify why the user currently believes it
- identify what observation would weaken it
- note what decision depends on it

If the idea cannot affect any decision or behavior, keep it as a note instead of an assumption.

### 3. Design the smallest useful test

Prefer:

- short time windows
- low emotional cost
- low financial cost
- observable signals
- reversible experiments

Avoid turning life questions into fake precision. If exact metrics are unrealistic, use directional signals and reflection prompts.

### 4. Separate evidence from conclusion

- evidence files store observations and source material
- assumption files store the claim and current status
- principle files store reusable conclusions

Do not upgrade a single observation into a principle unless it is unusually strong and decision-relevant.

### 5. Record time explicitly

When possible, include:

- `created_at`
- `start_at`
- `end_at`
- `verified_at`
- `falsified_at`
- `recheck_at`

Time matters because many life conclusions are season-dependent.

### 6. Track scope

When something appears true, identify the likely scope:

- always true
- true in the current season
- true under certain constraints
- true only when a specific habit or environment is present

Use `validity_scope` in principles to prevent false universalization.

## Operating Heuristics

- Prefer one crisp assumption over a page of journaling.
- Prefer one small experiment over a high-effort life redesign.
- Prefer one grounded principle over a motivational slogan.
- Surface tradeoffs explicitly when two values conflict.
- Keep the repository honest: uncertainty should stay visible.

## File Creation Rules

- Use the templates in `templates/`.
- Place new ideas in `assumptions/inbox/` by default.
- Move an assumption to `active/` only when a real verification plan exists.
- Move to `verified/` or `falsified/` only after enough evidence is linked.
- Create a principle only when the conclusion is reusable across future decisions.

## Review Rules

During weekly or monthly reviews:

- find active assumptions with no recent evidence
- find verified assumptions that may deserve promotion into principles
- find principles whose scope may be too broad
- find stale conclusions that should be re-checked

## Collaboration Style

When helping the user:

- be concise but structured
- challenge assumptions gently
- distinguish fact, interpretation, and recommendation
- make uncertainty explicit
- optimize for usefulness, not rhetorical perfection

## When To Read More

Read [repo-schema.md](references/repo-schema.md) when you need the intended role of each directory.

Read [verification-rules.md](references/verification-rules.md) when you need guidance on choosing a suitable test or deciding whether a conclusion is strong enough to promote.

