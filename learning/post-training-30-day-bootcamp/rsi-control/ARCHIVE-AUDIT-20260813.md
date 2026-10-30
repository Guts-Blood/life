# RSI archive audit — 2026-08-13

## Current claim

The committed RSI state is internally valid and records `rsi-v0002` as
`complete_qualified`, with `main-s20260809-lr1e-4-final` promoted after an
independent confirmation run. The result summary has file SHA-256
`66d6f1682f1ced3f5125333fb2b0272af92c4440cbefcf2ff7f931955850c1be`.

Commit `4d09ae95632b0b21dba9f3d7cc2b55209d02ab41` first placed the Day20 v3
implementation and the complete compact v0001/v0002 ledger in Git. The archive
includes failed launches, retries, run contracts, operation hash chains,
training/evaluation summaries, selections, costs, and final promotion evidence.

Local verification before publication passed:

- 153 Day20 unit tests;
- 17 RSI controller/collector tests;
- `rsi_control.py validate` and `status`;
- Python compilation and both runner shell syntax checks;
- staged secret-pattern and whitespace scans.

## Deliberate archive boundary

Git contains compact evidence and immutable identities, not model weights,
checkpoint directories, raw remote logs, credentials, or the full AutoDL run
root. The promoted checkpoint remains identified by:

- remote URI:
  `/root/autodl-tmp/runs/day20-v3-qwen35-lora-20260812T105141Z/adapters/main/s20260809/lr1e-4/attempt-001/checkpoint-1904`;
- integrity SHA-256:
  `d0f72be9751628c9073cc8e4104f16d8620bd598dbb8a1e97f1df9bae51170a3`;
- snapshot SHA-256:
  `c169e0bb55b20951d27a889e4ef0deae3a01fe10acabee804b212d3d8170db0a`;
- adapter SHA-256:
  `29dc1a7d676b7f2f88675521bc44cf81dd43b700423df3cf4adf8d6a1c4487c3`.

Those identities are preserved in the Main training completion payload and its
hash-chained operation event. The checkpoint bytes themselves were not copied
into Git.

## Known durability limitations

1. The final read-only collector verified 44 items and published inventory
   SHA-256
   `9d6a78e9954e37e74fae9698ad6729dc253cb3c8156f3008d86ad24ed7ccab5b`,
   but the full inventory JSON was not pulled before the AutoDL endpoint went
   offline. Git retains its compact summary and every decision-critical item,
   but cannot reconstruct that 44-item document byte-for-byte offline.
2. `RSI-V0002-RESULT.md` was not added to the pre-existing self-hashed
   `artifacts.json`; Git now supplies its durable version identity. The archived
   artifacts record should not be rewritten after completion merely to add this
   link.
3. The v0001 run contract preserves launch-time evaluator/E2B wrapper hashes,
   while v0001 `artifacts.json` records the corrected post-retry implementations.
   This is a pre/post-remediation identity split, not a claim that the bytes were
   identical.
4. A v0001 retry reused the same remote Base-eval log path. The ledger preserves
   the different observed hashes, but both historical byte streams are no
   longer recoverable from that single URI.
5. Three v0001 local produced-artifact paths are written relative to the
   `rsi-control` directory even though the file declares
   `path_base=bootcamp_root`. Their committed bytes and hashes match when
   resolved from `rsi-control`; this path semantic is documented rather than
   silently rewriting the completed record.
6. Primary `gate_margins` were not stored alongside the values, although they
   are deterministically recomputable as total 16, General 16, Math 4,
   Finance 9, Code 6, Format 22, eligibility 2, and infrastructure 0.

## Recovery rule

If the AutoDL run root becomes reachable again, recovery must be append-only:
verify the remote checkpoint and full collector inventory against the identities
above, copy them to a new immutable archive location, and add a new Git commit.
Do not edit old contracts, ledgers, attempts, or self-hashed evidence in place.

The future two-GPU policy is likewise prospective. The completed v0002 runner
and contracts correctly record single-GPU execution; DDP implementation belongs
to a new, not-yet-bound RSI version.
