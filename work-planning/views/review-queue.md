# Review Queue

## contract-approved

Context: `auto-skill-contract`

- **AV2-CONTRACT / contract-decision**: Are the compatibility and rollback boundaries acceptable?
  - Old clients have a migration path
  - Rollback is explicit
  - Expected evidence: Approved contract artifact, Decision record

## image-file-id-contract

Context: `image-artifact-contract`

- **AV2-IMAGE-FILE-ID / file-id-decision**: Approve the file-id lifecycle, permission, and legacy compatibility boundary?
  - Old image_url references have a migration path
  - Permission failures are user-visible
  - Expected evidence: Migration contract, Compatibility matrix, Decision record
