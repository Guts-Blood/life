-- Reproducible report-layer projection of reviewed Day 19 evidence.
-- Inputs reviewed on 2026-08-08:
--   DAY19-RESULTS.json
--   eval/{baseline-a,baseline-b,best-e}.predictions.jsonl
--   data/{baseline-a,baseline-b,best-e}.jsonl
--   day10-frozen-eval-manifest.json

WITH baseline_b_by_slice(slice, metric, response_count, denominator) AS (
    VALUES
        ('General', 'Format valid', 21, 28),
        ('General', 'Correct',      15, 28),
        ('Math',    'Format valid', 28, 28),
        ('Math',    'Correct',       0, 28),
        ('Code (legacy contract)', 'Format valid', 0, 28),
        ('Code (legacy contract)', 'Correct',      0, 28),
        ('Finance', 'Format valid', 24, 28),
        ('Finance', 'Correct',       0, 28)
)
SELECT slice, metric, response_count, denominator
FROM baseline_b_by_slice
ORDER BY
    CASE slice
        WHEN 'General' THEN 1
        WHEN 'Math' THEN 2
        WHEN 'Code (legacy contract)' THEN 3
        WHEN 'Finance' THEN 4
    END,
    CASE metric WHEN 'Format valid' THEN 1 ELSE 2 END;

WITH code_contract(recipe, raw_syntax_valid, diagnostic_syntax_salvageable, denominator) AS (
    VALUES
        ('Baseline A', 0, 15, 28),
        ('Baseline B', 0, 25, 28),
        ('Best-E',     0, 23, 28)
)
SELECT recipe, raw_syntax_valid, diagnostic_syntax_salvageable, denominator
FROM code_contract
ORDER BY diagnostic_syntax_salvageable DESC;

WITH code_shape_summary(metric, response_count, denominator) AS (
    VALUES
        ('Leading empty thinking wrapper', 84, 84),
        ('Complete function after wrapper removal', 84, 84),
        ('Diagnostic syntax salvageable', 63, 84)
)
SELECT metric, response_count, denominator
FROM code_shape_summary;

WITH selected_code_training(recipe, selected_records, continuation_only_prompts, indented_body_targets) AS (
    VALUES
        ('Baseline A', 115, 0, 0),
        ('Baseline B', 221, 0, 0),
        ('Best-E',     142, 0, 0)
)
SELECT recipe, selected_records, continuation_only_prompts, indented_body_targets
FROM selected_code_training;
