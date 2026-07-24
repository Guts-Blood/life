# Life System

This repository is a personal life management system built around:

`context -> assumption -> verification -> evidence -> conclusion -> principle`

The goal is not only to record ideas, but to turn intuitions into testable assumptions,
collect evidence, and gradually build a set of validated personal principles.

## Top-Level Purpose

The repository exists to make daily life compound.

The highest-level goals are:

1. Improve cognition and use that improved cognition to reach financial freedom.
2. Maintain a healthy life.

Both goals require life to be operated as a system: daily actions should generate
feedback, evidence, better assumptions, stronger principles, and eventually more
reliable behavior. AI agents are currently an important domain and leverage point,
but they are not the final goal. They are one part of the broader life system.

## Core Areas

- `profile/`: stable and seasonal personal context
- `assumptions/`: ideas waiting to be tested
- `experiments/`: concrete verification plans
- `evidence/`: observations, logs, and source material
- `principles/`: ideas that have earned enough support to guide decisions
- `reviews/`: periodic reflection and system maintenance
- `templates/`: reusable note templates
- `skill/`: workflow instructions for collaborating with Codex on this system
- `skills/`: versioned personal Codex skills, including context-aware work planning
- `work-planning/`: approved plans, event traces, and generated focus/review views
- `work-tracking/`: concrete workstreams, launch tracking, and weekly execution logs

## Recommended Flow

1. Update your context in `profile/`.
2. Capture a new idea as an assumption in `assumptions/inbox/`.
3. Promote it to `assumptions/active/` when you want to test it.
4. Create a matching experiment in `experiments/active/`.
5. Store observations and proof in `evidence/`.
6. Move the assumption to `verified/` or `falsified/`.
7. If it generalizes, promote the conclusion into `principles/`.

## Naming Suggestions

- assumptions: `A-YYYYMMDD-short-title.md`
- experiments: `E-YYYYMMDD-short-title.md`
- evidence: `EV-YYYYMMDD-short-title.md`
- principles: `P-YYYYMMDD-short-title.md`
- work tracking: `work-tracking/<area>/<dashboard-or-workstream>.md`

## First Files To Fill In

- `profile/user-profile.md`
- `profile/current-season.md`
- `assumptions/inbox/`
