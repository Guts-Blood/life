SELECT *
FROM read_json_auto('evidence/code_boundary.json', format='array')
ORDER BY record_type, candidate_order NULLS LAST, leading_spaces NULLS LAST;
