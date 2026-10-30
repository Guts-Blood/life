SELECT *
FROM read_json_auto('evidence/engineering_incidents.json', format='array')
ORDER BY stage;
