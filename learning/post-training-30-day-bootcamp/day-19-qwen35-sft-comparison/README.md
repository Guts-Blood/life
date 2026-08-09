# Day 19 — Qwen3.5 controlled SFT comparison

Status: `done_standalone_diagnostic / no_promoted_s1`
Scope: standalone pull-forward experiment；不是顺序课程的 Day 19 optimizer/failure-injection 完成记录。

This run migrates the audited Day 12 `A`, `B`, and retained best-code `E`
recipes to the Day 18-validated Qwen3.5-4B Megatron entrypoint. The three runs
share exactly 64,608 Qwen3.5 supervised tokens, one model start, one seed, one
optimizer contract, and one dev evaluation protocol.

The runner keeps exactly one model-only Megatron checkpoint for each of
`baseline-a`, `baseline-b`, and `best-e`. HF exports are temporary validation
artifacts and are deleted only after the exported model loads, generates the
frozen dev predictions, and the evaluation summary is durable.

HumanEval completions must never run on the AutoDL host. If the pinned E2B
sandbox is unavailable, finalization emits `DAY19-PENDING.json` instead of a
false `DAY19-PASS.json`; the retained predictions can be sandbox-scored later
without retraining.

Remote gate order:

```bash
bash run_day19_autodl.sh init
bash run_day19_autodl.sh prepare
bash run_day19_autodl.sh train-all
bash run_day19_autodl.sh eval-all
bash run_day19_autodl.sh finalize
```

## Qwen3.5 response-adapter v2 diagnostic

The canonical Day 19 artifacts remain immutable.  The v2 diagnostic creates
new sidecars and does not retrain, replace a checkpoint, edit the frozen Day 10
scorers, or overwrite `DAY19-RESULTS.json`.

The protocol has two separate adapters:

1. `qwen35_response_adapter.py` handles only the model-family response
   boundary.  Without token evidence it removes one exact leading
   `<think>\n\n</think>\n\n` prefix.  With ms-swift `return_details`, the
   generated token IDs and their prefix-free decode are authoritative.
2. `day19_humaneval_adapter.py` handles only the task output.  It reuses the
   frozen Markdown-fence extractor and classifies code as a complete
   `solution` or a prompt `completion`.  A solution must define the expected
   top-level entry point exactly once; a completion must remain wholly inside
   the original function body.  Source imports and helpers before the original
   entry point are retained.  The adapter never changes a function name,
   signature, or implementation.

`rescore_day19_qwen35_v2.py` verifies every legacy row and its scorer result,
then writes a SHA-bound `*.qwen35-v2.*` sidecar.  Non-code answers continue to
use the frozen MMLU, GSM8K, and TAT-QA scoring semantics after response-boundary
adaptation.  `score_day19_code_e2b_v2.py` executes code only in the same pinned,
fresh, network-denied E2B environment, using explicit solution/completion
composition.
Candidates that fail the entry-point/body-containment contract are assigned a
deterministic zero without being submitted to E2B; eligible candidates alone
are executed in fresh sandboxes.  Every sidecar binds the frozen manifest,
scorer, adapter/parser source files, original evaluation summary, checkpoint,
and recomputed model-file snapshot.

After AutoDL is available, run only the diagnostic path against the immutable
canonical run `/root/autodl-tmp/runs/day19-qwen35-20260808T100142Z`:

```bash
bash run_day19_autodl.sh eval-c0
bash run_day19_autodl.sh rescore-v2-all
bash run_day19_autodl.sh sandbox-v2-all
```

If a sandbox call is interrupted after one recipe succeeds, resume by recipe
instead of rerunning the no-overwrite batch:

```bash
bash run_day19_autodl.sh sandbox-v2 baseline-b
```

Do not run `train` or `train-all` for this diagnosis.  `eval-c0` evaluates the
untouched Qwen3.5-4B Base without creating a checkpoint.  Existing A/B/E raw
outputs have no token-level response capture, so their sidecars use the strict
one-prefix fallback.  That fallback is allowlisted only for the retained A/B/E
recipes.  C0 and future evaluations retain prompt/generated token IDs plus
recomputed hashes through ms-swift `RequestConfig(return_details=True)`; an
incomplete capture fails closed.  The evaluator also checks the live Qwen3.5
template prefix before generation and publishes predictions only after the
whole attempt completes.

Expected new artifacts per recipe (`baseline-a`, `baseline-b`, `best-e`, and
`untouched-c0`):

```text
eval/<recipe>.qwen35-v2.predictions.jsonl
eval/<recipe>.qwen35-v2.json
eval/<recipe>-code-e2b-qwen35-v2.jsonl
eval/<recipe>-code-e2b-qwen35-v2-summary.json
```

The causal decision is made only after all four E2B summaries exist:

- compare untouched C0 with baseline B on Math and Finance to separate base
  limitations from SFT regression;
- compare legacy and v2 executable Code scores to quantify parser distortion;
- keep Best-E rejected if its generation-ceiling failure remains;
- do not propose another SFT run until these zero-training checks finish.

The canonical standalone Day 19 directory intentionally retains no remote
prediction JSONL files.  Local tests therefore prove adapter/parser/provenance
behavior with synthetic fixtures; any copied report-source workspace is a
separate generated artifact and is not the canonical run directory.

Local verification:

```bash
python3 -m unittest discover \
  -s learning/post-training-30-day-bootcamp/day-19-qwen35-sft-comparison \
  -p 'test_*.py' -v
bash -n learning/post-training-30-day-bootcamp/day-19-qwen35-sft-comparison/run_day19_autodl.sh
```
