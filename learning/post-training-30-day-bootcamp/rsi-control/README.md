# RSI control

This folder is the compact source of truth for recursive improvement. It is not
SFT-specific: a version may change model training, data, Harness, verifier, or
runtime policy. The current goal happens to use SFT as its allowed lever.

Read these files in order:

1. [`goal.json`](goal.json) — what “done” means now;
2. [`state.json`](state.json) — current version and next action;
3. [`TAXONOMY.md`](TAXONOMY.md) — symptom, first-broken-invariant failure
   diagnosis, and intervention boundaries;
4. [`CONTEXT-GRAPH.md`](CONTEXT-GRAPH.md) — deterministic graph addressing and
   compact SubAgent context packets;
5. [`failure-taxonomy.json`](failure-taxonomy.json) — machine-readable failure
   modes, symptoms, legacy aliases, intervention crosswalks, and worked cases;
6. [`taxonomy.json`](taxonomy.json) — bounded canonical intervention lever IDs;
7. [`metrics.json`](metrics.json) — outcome, diagnosis, cost, and RSI-policy metrics;
8. the current directory under `versions/`.

The append-only archive limitations for the completed v0001/v0002 evidence are
recorded in [`ARCHIVE-AUDIT-20260813.md`](ARCHIVE-AUDIT-20260813.md).

Validate before and after every update:

```bash
python3 rsi_control.py validate
python3 rsi_control.py status
```

Locate an atomic concept and build a bounded context packet before delegating a
diagnosis subtask:

```bash
python3 rsi_context.py search "template rendering"
python3 rsi_context.py pack \
  --seed failure.data.supervision.template_rendering \
  --output /tmp/rsi-context.json
```

The graph and packet are deterministic derived views. Do not edit them or use
them as a replacement for canonical taxonomy, version, or evidence records.

## Minimal lifecycle

```text
goal → symptom → first broken invariant / failure diagnosis
     → intervention lever → version → run → attempt/retry
     → artifacts + metrics → decision → next version or goal achieved
```

Keep those three analysis axes separate: a symptom is not a root cause, and a
failure location does not have to share a name or layer with its intervention.
The prospective v0003+ policy and completed-history boundary are defined in
[`TAXONOMY.md`](TAXONOMY.md).

- **Version** (`rsi-vNNNN`): one hypothesis and improvement policy. Any
  semantic change creates a new version.
- **Run**: one phase and seed under that version.
- **Attempt**: one operational launch of the same run contract.
- **Retry**: same version/run/seed/data/config/runtime identity; otherwise it is
  a new attempt, run, or version.

Every version owns exactly three compact records:

- `version.json`: context, assumption, lever, plan, and status;
- `artifacts.json`: immutable inputs/implementation plus produced artifacts;
- `metrics.json`: prediction, observed metrics, cost, and decision.

Run-level retry logs live under
`runs/<run>/attempts/<attempt>/logs/retries.jsonl`. Large checkpoints and GPU
logs remain in artifact storage; this folder records their URI and hashes.
Each retry event records the failure layer/code, remediation, retry ordinal,
and before/after run-contract hashes; semantic drift is rejected.

## Iteration rules

From `rsi-v0002` onward, change one primary lever and at most one necessary
dependent lever. Record the prediction and falsifier before execution. Never
hide failed attempts, replace a hard gate with an average, modify evaluation to
promote a candidate, or train on eval cases.

From `rsi-v0003` onward, use exact canonical failure and intervention leaves and
embed the structured diagnosis defined in [`TAXONOMY.md`](TAXONOMY.md) before
execution. Completed v0001/v0002 records remain grandfathered and append-only.

`rsi-v0001` is the already implemented Day20-v2 baseline reset. It changes
several SFT mechanisms together, so it establishes a trustworthy baseline but
does not support single-lever causal attribution.

To continue, create the next `versions/rsi-vNNNN`, reset its metrics and runs,
declare one lever and falsifier, then update `state.json`. Validation must pass
before execution.

## Future training topology policy

Starting with the next not-yet-bound training version, every training run must
use both provisioned AutoDL GPUs with distributed data parallelism. Evaluation
may remain single-GPU unless independent candidates can be isolated without
changing their evidence identities.

The two-GPU conversion is a runtime-topology change, not a bitwise-equivalent
retry. Precommit it in the next run contract and keep the scientific recipe
fixed: the same source rows and global order, seeds, target encoding, optimizer
and learning-rate schedule, LoRA configuration, exact supervised-token budgets,
checkpoint token targets, inference template, evaluators, scorers, and gates.
Keep the effective global batch at 8 by using world size 2, per-device batch 2,
and gradient accumulation 2; the prior single-GPU setting was world size 1,
per-device batch 2, and gradient accumulation 4.

Before any GPU work, fail closed unless a materialization audit proves that the
distributed sampler neither pads nor duplicates rows, every optimizer step maps
to the intended ordered global batch of 8 sample IDs (apart from the unchanged
deterministic final partial batch), optimizer-step and token checkpoint schedules
are unchanged, both GPUs are visible, and the training runtime reports world
size 2. Primary and confirmation runs within a version must use the same
topology. Because distributed reduction order can change the numeric trajectory,
compare outcomes by the frozen evaluation gates rather than claiming bitwise
reproducibility.
