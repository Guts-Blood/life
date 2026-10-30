# Day 19 root-cause report source notes

## Reporting job

- Question: Is the 4B result primarily a template/contract problem or a model instruction-following problem, and what should Day 20 do?
- Audience: technical.
- Decision: whether to repair evaluation, retrain, or both.
- Scope: Day 19 frozen development evaluation, 112 examples per checkpoint (28 per slice), plus the selected Day 19 SFT records.
- Comparison basis: Baseline A, Baseline B, and Best-E under the matched Day 19 protocol; historical 0.6B scores are context, not a causal control.

## Required technical-report structure mapping

1. Title → `title`
2. Technical summary → `technical_summary`
3. Key findings with visual evidence → `code_contract_finding`, `code_contract_chart`, `noncode_finding`, `baseline_b_chart`, `best_e_regression`
4. Scope, data, and metric definitions → `scope_definitions`
5. Methodology → `training_contract_finding`, `methodology`
6. Limitations, uncertainty, and robustness checks → `limitations`
7. Recommended next steps → `next_steps`
8. Further questions → `further_questions`

## Source inventory

- `DAY19-RESULTS.json`: canonical Day 19 aggregate metrics and decision record.
- `eval/{baseline-a,baseline-b,best-e}.predictions.jsonl`: retained raw model outputs and scorer results.
- `data/{baseline-a,baseline-b,best-e}.jsonl`: exact selected SFT messages.
- `day10-frozen-eval-manifest.json`: frozen prompts, task metadata, and continuation-only HumanEval contract.
- `day19_diagnostic_metrics.sql`: reviewed report-layer projection of the extracted counts.

## Diagnostic transformations

- A leading empty `<think>...</think>` wrapper was removed only for diagnostic classification.
- If the remaining code parsed as one complete function matching the requested entry point, its function body was tested for syntactic composability after the frozen HumanEval prompt.
- No candidate code was executed during this diagnostic pass.
- “Diagnostic syntax salvageable” is not an accuracy score and does not imply semantic correctness.

## Chart map

| Section | Question | Family / type | Fields | Supported claim | Palette | Delivery |
|---|---|---|---|---|---|---|
| Code contract | How much of the 0/84 syntax result is explained by output-shape mismatch? | Comparison / bar | recipe, diagnostic_syntax_salvageable, denominator | 63/84 outputs become syntactically composable after bounded normalization | sequential blue, direct values, reference at 28 | portable HTML report |
| Baseline B slices | Is low accuracy explained by parser/format failure? | Comparison / grouped bar | slice, metric, response_count | Math and finance are mostly parseable but still wrong | categorical blue/gold, legend plus values | portable HTML report |

Repeated bar charts are intentional: both questions are discrete category comparisons, but they use different datasets, measures, and claims.

## Validation notes

- Recomputed totals: 15 + 25 + 23 = 63 salvageable code outputs out of 84.
- Baseline B slice counts each use denominator 28.
- Baseline B correct counts sum to 15 across the four slices, matching `DAY19-RESULTS.json`.
- The report separates verified observations from hypotheses and does not claim that normalized code passes HumanEval.
- The untouched Qwen3.5 C0/Base was not evaluated in Day 19; this remains the principal causal gap.
