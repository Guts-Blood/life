# Day 22 Formal S1 Blind Review Protocol v1

This protocol applies only to
`artifacts/data/day22-qwen35-formal-s1-blind-review.jsonl`.
The frozen worksheet contains 50 unique pairs, each shown twice with A/B order
reversed, for 100 independent judgments.

## Reviewer isolation

- Review a copy of the worksheet. Do not edit the frozen blank worksheet.
- Do not open or receive the concealed key before all 100 rows are complete.
- Judge rows in file order. Do not search for the matching swapped presentation.
- Use only the prompt and the two displayed responses. Do not run the hidden
  tests or consult the verifier labels.

## Fields to fill

For every row, fill `verdict` and `confidence`. `notes` is optional for `A` or
`B` and required for `tie`, `ambiguous`, or `reject`. Do not change any other
field.

`verdict` must be exactly one of:

- `A`: response A is clearly preferable for correctness and instruction
  compliance.
- `B`: response B is clearly preferable for correctness and instruction
  compliance.
- `tie`: both responses appear equally good or equally bad.
- `ambiguous`: the prompt or visible evidence is insufficient to decide.
- `reject`: the review item is malformed, contaminated, or otherwise unsuitable.

`confidence` must be exactly `low`, `medium`, or `high`.

Prefer functional correctness first, then instruction compliance. Do not prefer
an answer merely because it is longer, more polished, or appears in position A
or B.

## Fail-closed readiness policy

After annotation, the finalizer maps A/B through the concealed key and compares
the primary and swapped presentations for each unique pair.

- All 100 rows must be completed.
- `tie`, `ambiguous`, and `reject` exclude that reviewed pair.
- A primary/swapped semantic disagreement excludes that reviewed pair.
- A position-consistent human preference for the verifier's rejected side
  excludes that reviewed pair.
- Position consistency and verifier-chosen agreement must each be at least 90%.
- At least 200 pairs must remain after exclusions.
- A ready manifest binds the unchanged pending pair file, so this v1 handoff
  emits READY only when there are no exclusions. A future larger bundle would
  need a separately sealed filtered-pairs artifact before it could retain a
  subset.

The current bundle contains exactly 200 pairs, so any exclusion keeps formal
DPO readiness blocked. The finalizer records the result; it never fabricates or
silently repairs a human judgment.

## Handoff commands

From the bootcamp repository root, first make the reviewer-owned copy:

```bash
cp artifacts/data/day22-qwen35-formal-s1-blind-review.jsonl \
  artifacts/data/day22-qwen35-formal-s1-blind-review-completed.jsonl
```

After all 100 rows are filled, finalize once into new files:

```bash
python3 day-22-preference-data/finalize_day22_formal_s1_review.py \
  --pending-manifest artifacts/data/day22-qwen35-formal-s1-manifest.json \
  --completed-worksheet artifacts/data/day22-qwen35-formal-s1-blind-review-completed.jsonl \
  --review-audit-output artifacts/reports/day22-qwen35-formal-s1-human-review-audit.json \
  --ready-manifest-output artifacts/data/day22-qwen35-formal-s1-ready-manifest.json
```

Exit code `0` means a ready manifest was emitted. Exit code `3` means the
review was complete but a readiness gate failed; the review audit records why
and no ready manifest is created. Exit code `2` means the inputs were invalid.
The finalizer never overwrites an existing output.

Independently verify a successful result before Day 23 consumes it:

```bash
python3 artifacts/scripts/validate_day22_formal_s1_bundle.py \
  --manifest artifacts/data/day22-qwen35-formal-s1-ready-manifest.json \
  --require-formal-ready
```
