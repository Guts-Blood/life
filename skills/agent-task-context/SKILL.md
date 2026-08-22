---
name: agent-task-context
description: Use before ambiguous, complex, multi-step, high-stakes, or context-heavy tasks. Map known knowns, known unknowns, unknown knowns, and unknown unknowns before planning, asking, researching, coding, or executing.
---

# Agent Task Context Skill

Context is any condition that changes what should be done.

Before acting, map what is known, missing, implicit, and hidden. Then choose the smallest next move that makes the task safe enough to continue.

## The Four Quadrants

| Quadrant | Meaning | Typical Signals | Agent Move |
|---|---|---|---|
| Known knowns | Explicit, trusted, usable facts | Goal, constraints, inputs, preferences, success criteria | Anchor |
| Known unknowns | Visible gaps that may change the answer | Ambiguous terms, missing data, unresolved tradeoffs, time-sensitive facts | Ask, assume, search, or verify |
| Unknown knowns | Relevant context that exists but was not stated | Repo patterns, user habits, domain norms, implicit audience, deeper intent | Discover |
| Unknown unknowns | Blind spots that have not surfaced yet | Wrong frame, edge cases, hidden risks, failure modes | Stress-test |

## 1. Known Knowns: Anchors

Record only what is explicit, reliable, and decision-relevant.

Look for:

- the user's real goal
- the requested output
- provided facts, files, data, and tools
- hard constraints and stated preferences
- agreed definitions and success criteria

Rules:

- Do not ask again for information already given.
- Do not silently rewrite the user's goal.
- Turn anchors into boundaries and acceptance criteria.

## 2. Known Unknowns: Gaps

Make uncertainty visible, but do not let questioning replace progress.

For each important gap, choose one action:

- **Ask** when the answer changes the plan and a wrong guess is costly.
- **Assume** when the risk is small, reversible, or easy to correct.
- **Search** when the answer should exist in files, code, docs, data, or current sources.
- **Verify** when the answer can be tested, run, computed, or cross-checked.

Rules:

- Do not turn guesses into facts.
- Label assumptions when using them.
- Ask only for answers that truly block the next move.

## 3. Unknown Knowns: Silent Context

Look for what the user did not say, but the task already carries.

Common sources:

- current repository structure and style
- the user's prior preferences
- domain conventions
- implicit audience, stakes, and use case
- the deeper problem behind the surface request

Rules:

- Read available context before inventing context.
- Separate observed facts from inferred background.
- Use silent context to improve fit, not to over-interpret.

## 4. Unknown Unknowns: Blind Spots

Search for what could make the answer fail.

Prioritize:

- a misframed problem
- missing stakeholders
- permission, privacy, safety, legal, financial, or operational risk
- edge cases, abnormal inputs, and scale limits
- stale facts
- missing validation

Methods:

- Ask: "What would make this answer wrong?"
- Generate counterexamples.
- Check boundaries.
- Run the smallest useful test.
- Validate against real output.

Rules:

- Do not chase every possible risk.
- Focus on blind spots that would change the plan, cost, or safety profile.

## Protocol

For non-trivial tasks, follow this sequence:

1. **Restate the task**: Confirm the goal and output in one or two sentences.
2. **Map the quadrants**: Keep only context that affects decisions.
3. **Tag the unknowns**: Mark each important unknown as Ask, Assume, Search, or Verify.
4. **Choose the next move**: Ask if blocked; otherwise proceed with stated assumptions.
5. **Plan**: The plan must respond to anchors, gaps, silent context, and major blind spots.
6. **Execute and update**: Revise the context map when new facts appear.
7. **Verify**: Check the result against success criteria, key assumptions, and failure modes.

## Output Formats

Use the full format for ambiguous, strategic, or risky tasks:

```md
## Task Understanding

...

## Context Map

### Known Knowns
- ...

### Known Unknowns
- ...

### Unknown Knowns
- ...

### Unknown Unknowns
- ...

## Key Assumptions
- ...

## Plan
1. ...
2. ...
3. ...

## Verification
- ...
```

Use the compressed format for ordinary tasks:

```md
Known: ...
Gaps: ...
Assumptions: ...
Next: ...
```

For small tasks, do the context pass silently and answer directly.

## Operating Rules

- Ask less, but ask sharper.
- Inspect available context before requesting more from the user.
- Separate facts, assumptions, and inferences before proposing a solution.
- Show uncertainty only when it changes the decision.
- Verify information that is current, technical, high-stakes, or easy to test.
- Let analysis serve action, not replace it.

## Reusable Prompt

```text
Before answering, use the Agent Task Context Skill.

Map the task into:

1. Known knowns: What facts, goals, constraints, and success criteria can be used as anchors?
2. Known unknowns: What is missing or uncertain? Tag each important item as Ask, Assume, Search, or Verify.
3. Unknown knowns: What silent context can be discovered from the environment, history, code, documents, or domain norms?
4. Unknown unknowns: What blind spots, edge cases, or failure modes could change the plan?

Then give a concise plan and proceed unless a missing answer truly blocks progress.
```
