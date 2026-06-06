---
id: A-20260426-agent-systems-high-level-mapping-hashnode-en
title: A High-Level Mapping Between Agent Systems and State Transitions
status: inbox
domain: AI systems
created_at: 2026-04-26
updated_at: 2026-04-26
confidence: 0.5
source: derived from A-20260426-agent-systems-high-level-mapping
related_profile:
related_experiment:
review_at:
verified_at:
falsified_at:
supersedes:
---

# A High-Level Mapping Between Agent Systems and State Transitions

## Suggested Hashnode Metadata

**Title:** A High-Level Mapping Between Agent Systems and State Transitions

**Subtitle:** Thinking about agents as controlled stochastic processes, not just prompt-response machines.

**Tags:** AI Agents, LLM, System Design, Markov Chains, Software Engineering

## Article Draft

I have been trying to understand agent systems through the lens of state machines, transition probabilities, and convergence time.

The high-level intuition is this:

> An agent is not just a model producing one response. An agent system is a controlled stochastic state transition process.

This framing is not meant to be a perfect academic reduction. Real agent systems are partially observable, non-stationary, and heavily shaped by tools, prompts, memory, users, and external environments. But as a design lens, it is useful because it forces us to ask a better question:

> How does the system move from the current state to a successful final state, and what makes it drift into failure states?

## From Agent Execution to State Transitions

At a high level, we can describe an agent system as moving from one state to another:

```text
S_t -> S_{t+1}
```

Here, `S_t` is the state of the whole agent system at step `t`.

This state is not just the prompt. It is not just the model's context window. It is an abstraction of everything the system currently knows and can act on.

For example, `S_t` may include:

- the user's goal
- the current task progress
- known constraints
- tool results
- uncertainty
- previous failed attempts
- the current control phase
- budget, permission, and latency constraints

If we collect all possible states, we get a state space:

```text
S = {s_1, s_2, ..., s_n}
```

In real agent systems, this state space is huge and cannot be explicitly enumerated. But the abstraction is still useful: every agent run can be viewed as a trajectory through some implicit state space.

## The Transition Matrix Is Induced by the Whole System

If we temporarily ignore actions and only look at state-to-state movement, we can write:

```text
P(s' | s)
```

This means: given the system is currently in state `s`, what is the probability that it moves to state `s'` next?

If the state space were finite and explicit, we could imagine a transition matrix:

```text
P =
[
  P(s_1 | s_1)  P(s_2 | s_1)  ...  P(s_n | s_1)
  P(s_1 | s_2)  P(s_2 | s_2)  ...  P(s_n | s_2)
  ...
  P(s_1 | s_n)  P(s_2 | s_n)  ...  P(s_n | s_n)
]
```

Of course, real agent systems do not come with a clean transition matrix. The matrix is implicit. It is induced by the behavior of the whole system.

That behavior is shaped by:

- the base model
- the system prompt
- the current context
- the action space
- available tools
- tool schemas
- the planner
- the harness
- the verifier
- memory
- user feedback
- the external environment

So the important design question is not:

> Can we explicitly write down the full transition matrix?

The more useful question is:

> How does our agent design change the probability of moving from bad states to good states?

## Goal States and Failure States

Most agent tasks do not have a single final state. They have a set of acceptable goal states:

```text
G = goal states
```

Examples:

- The user's question is answered correctly.
- The code change is complete and tests pass.
- The requested file is generated correctly.
- The system asks the right clarification question when information is missing.
- The output satisfies the user's constraints and quality bar.

There is also a set of failure states:

```text
F = failure states
```

Examples:

- wrong answers
- wrong tool calls
- loops
- premature completion
- context pollution
- permission or safety violations
- continuing in the wrong direction

So a good agent system should not merely wander toward some stable behavior. It should maximize the probability of reaching the goal set before the failure set:

```text
Pr(reach G before F)
```

This is one reason why evaluating agents by single-turn output quality is insufficient. The real object of evaluation is the whole transition process.

## Mixing Time vs. Hitting Time

From the system theory side, we may be tempted to talk about mixing time: how quickly a process converges toward its stationary distribution.

That intuition is useful, but for agent systems the more direct concept is usually hitting time:

```text
tau_G = first time reaching G
```

In other words, how long does it take the agent system to reach the goal state set for the first time?

In engineering terms, this can correspond to:

- number of model turns
- number of tool calls
- token cost
- wall-clock latency
- planner expansions
- user interventions
- repair attempts

So the high-level objective becomes:

```text
maximize   Pr(tau_G < tau_F)
minimize   E[tau_G]
minimize   cost(tau_G)
```

Where:

```text
tau_G = first time reaching the goal state set G
tau_F = first time reaching the failure state set F
```

This is the agent-system version of "converging faster": not converging to an arbitrary stationary distribution, but reaching the target region more reliably and with lower cost.

## Mapping System Theory to Agent Design

Here is the high-level mapping:

| System concept | Agent system counterpart | Role |
| --- | --- | --- |
| State space `S` | All possible task, context, control, and environment states | Defines where the system can be |
| Current state `S_t` | Current task representation, context, tool results, plan progress | Defines where the next step starts |
| Transition probability `P(s' | s)` | The system's tendency to move from one state to another | Describes where the agent naturally goes |
| Action `a_t` | Model response, tool call, clarification, planning, verification, repair | Changes the transition path |
| Policy `pi(a | s)` | The model and prompt's action distribution under the current state | Decides what the system does next |
| Goal set `G` | States satisfying the user's objective and quality bar | Defines success |
| Failure set `F` | Wrong answers, loops, false completion, violations, polluted states | Defines failure |
| Hitting time | Steps, latency, tool calls, token cost, repair count | Measures how fast the system reaches the goal |
| Bottleneck | A narrow bridge between useful and harmful state regions | Explains why correction can be hard |
| Recurrent class | A cluster of states the system keeps revisiting | Explains stuck behavior and loops |
| Absorbing state | Done, false done, fatal error | Explains why termination quality matters |

This gives us a compact design objective:

> Make the system assign more probability mass to paths that reach `G`, and less probability mass to paths that reach `F`, loops, or false completion.

## Where Planning Fits

A plan is not the transition itself. A plan is a hypothesis about a future path:

```text
S_0 -> S_1 -> S_2 -> ... -> G
```

A good plan increases the probability that the system follows a useful trajectory. It does this by proposing intermediate states and actions that are likely to lead to the goal set.

Planning answers:

> From the current state, which intermediate states are likely to lead to success?

This also explains why planning can fail. A bad plan does not just waste time. It changes the trajectory and can push the system toward a wrong region of the state space.

## Where the Harness Fits

The harness is not just a wrapper around the model. It is a control layer for the transition process.

The harness can define:

- allowed actions
- allowed state transitions
- validation rules
- retries
- rollback behavior
- termination conditions
- user handoff points
- escape paths from bad states

It answers:

> Which transitions are allowed, which should be blocked, which require verification, and when should the system stop?

So at a high level:

```text
Plan    = proposes a likely path to G
Harness = constrains and corrects the actual transitions
```

This is why agent engineering is not just prompt engineering. Prompting changes the action distribution, but the harness changes the shape of the transition process itself.

## A Working Definition of a Good Agent System

Using this framing, we can define a good agent system as:

> A controlled stochastic state transition system that can move from an initial state `S_0` to a goal state set `G` with high probability, low cost, and recoverability under a given task distribution.

Such a system should have:

- high probability of reaching successful states
- low probability of reaching failure states
- low expected hitting time
- high probability of escaping bad states
- robustness to initial-state perturbations
- recovery paths for tool failure and missing information
- resistance to false completion
- traces and evaluations that estimate transition quality

This is the key shift:

> We should not evaluate an agent only by the quality of one response. We should evaluate the quality of the state transition process it induces.

## What This Framing Does Not Solve Yet

This is only the high-level mapping. It does not yet answer several important questions:

- How should `S_t` be represented?
- How should the action space be layered?
- How exactly does planning reshape trajectory probability?
- How should a harness define finite control states?
- How should a verifier define `G` and `F`?
- How does memory change transition probabilities across episodes?
- How can traces estimate `P(s' | s, a)` empirically?
- How do we detect bottlenecks, stuck states, and false completion?

Those questions require a more detailed breakdown.

But this high-level frame already changes how we think about agent design.

The core idea is:

> Agent engineering is the work of shaping state transitions. The goal is to move probability mass away from bad transitions, false completion, and loops, and toward trajectories that reliably reach the user's goal.

