SELECT *
FROM read_json_auto('evidence/run_provenance.json', format='array')
ORDER BY artifact;
