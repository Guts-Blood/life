# Day 09 Deduplication and Decontamination Report

Status: `complete`

This report separates exact duplicate removal, near-duplicate candidate review, and train/eval overlap checks. A candidate is not treated as a confirmed removal unless a terminal decision explicitly says so.

## Cohorts and matching policy

- Training cohort before exact deduplication: 7,864 Step 5 downstream-eligible records.
- Training cohort after exact deduplication: 7,860 records.
- Eval candidate cohort: 160 records, 40 each for general, math, code, and finance.
- Exact normalization: Unicode NFKC, lowercase, collapsed whitespace, stripped boundaries, punctuation retained.
- Near matcher: `rapidfuzz.fuzz.ratio` 3.14.3, threshold 90, minimum 80 normalized characters for training near-duplicate candidates.
- Train/eval checks treat prompt, answer/reference, and complete record as separate boundaries; exact and near candidates are disjoint.

## Exact duplicate decisions

Three normalized complete-record groups produced four confirmed removals:

| Removed sample | Retained survivor |
|---|---|
| `finance:DVN/2015/page_92.pdf-3` | `finance:DVN/2015/page_92.pdf-2` |
| `finance:MKTX/2004/page_24.pdf-3` | `finance:MKTX/2004/page_24.pdf-2` |
| `general:personas_IF_nadl6nukgi377wx44dsnsryf` | `general:personas_IF_88qi516we95weq7b2dc6gr58` |
| `general:personas_IF_sjqdvycp5vs3s3u26jmaekiz` | `general:personas_IF_88qi516we95weq7b2dc6gr58` |

Accounting is `7,864 - 4 = 7,860`. The removals subtract 382 raw tokens, 466 input tokens, and 78 supervised tokens. Prompt-only or answer-only matches remain evidence and do not trigger automatic deletion.

## Near-duplicate candidates and Gate B

Step 7 generated 1,129 candidate-field records representing 812 unique sample pairs and 663 affected samples:

| Matched field | Candidate records | Score 90–<95 | Score 95–<100 | Score 100 |
|---|---:|---:|---:|---:|
| normalized prompt | 378 | 269 | 109 | 0 |
| normalized answer | 424 | 292 | 89 | 43 |
| normalized complete record | 327 | 240 | 87 | 0 |

The user manually reviewed 20 stratified pairs and chose `keep_both` for all 20. The remaining 792 unique pairs were retained under the frozen non-terminal policy that similarity candidates are not removals. Those 792 decisions are policy decisions, not claimed as manual review. Confirmed near-duplicate removals: 0.

## Train/eval overlap

The 7,860-record training pool was compared with all 160 eval candidates across three boundaries:

| Boundary | Exact candidates | Near candidates |
|---|---:|---:|
| train prompt vs eval prompt | 0 | 0 |
| train answer vs eval reference | 0 | 0 |
| train complete record vs eval complete record | 0 | 0 |

No train or eval record was automatically deleted. Confirmed contamination removals: 0. Day 09 freezes eval candidate content and IDs only; decoder settings, scoring, baselines, and the final eval protocol remain Day 10 work.

## Final accounting and evidence

```text
Step 5 accepted                         7,864
- exact complete-record duplicates         4
- confirmed near-duplicate removals         0
- confirmed contamination removals          0
= clean parent pool                     7,860
```

- Exact-dedup summary: `tmp/day09-work/step6-exact-dedup/exact-dedup-summary.json`, SHA-256 `a20c7828c6b1707d0baa91e0db4ee752b88f456766c50f26cac93bec9fc398a0`.
- Near-candidate summary: `tmp/day09-work/step7-near-candidates/near-duplicate-summary.json`, SHA-256 `e7ffb19a1047a372a5dce2a5153d8221513bd5f37a181205391d38b2688a1a75`.
- Near-candidate ledger: `tmp/day09-work/step7-near-candidates/near-duplicate-candidates.jsonl`, SHA-256 `12ad9b0ddcfb1d66105e68da6ee669ca0f7e97336d9934c1d0a58eaf19434b14`.
- Gate B ledger: `artifacts/reports/day09-gate-b-review.jsonl`, SHA-256 `3335a8d85ecdf4fd8d53646c53743ef67f7c48daa472888be1c54f1dcab8e0f0`.
- Train/eval candidate ledger: `tmp/day09-work/step9-decontamination/train-eval-overlap-candidates.jsonl`, empty-file SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

The last empty-file hash above is verified by the Step 9 summary and should be interpreted together with its row count of zero.
