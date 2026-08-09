SELECT *
FROM read_json_auto('evidence/probe_metrics.json', format='array')
ORDER BY candidate_order;
