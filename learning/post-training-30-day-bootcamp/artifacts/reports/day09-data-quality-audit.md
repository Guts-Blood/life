# Day 09 Pre-filter Data Quality Audit

## Technical summary

- Step 3 produced 7,864 contract-accepted records from 8,000 candidates (98.30%); 136 records remain outside this pre-filter cohort.
- math contributes 66.51% of supervised tokens but 23.88% of examples. Example share therefore does not represent training-signal share.
- Sources that explicitly declare synthetic construction contribute 74.57% of examples and 97.77% of supervised tokens. Synthetic provenance is the largest pre-filter composition risk.
- The deterministic refusal heuristic flags 14 records (0.18%), while the prompt-prefix heuristic places 389 records (4.95%) in repeated clusters. Both are review hints, not automatic quality failures.

## The accepted pool is token-heavy in a different mix than its example counts

All percentages below use the 7,864 Step 3 accepted records as the denominator. Raw-token and supervised-token percentages are calculated independently. Every Evidence value resolves to the complete sample-ID list in the evidence index.

### Source dataset

| Value | Examples | Example % | Raw tokens | Raw % | Supervised tokens | Supervised % | Evidence |
|---|---:|---:|---:|---:|---:|---:|---|
| allenai/tulu-3-sft-personas-code | 2,000 | 25.43% | 739,180 | 17.65% | 264,296 | 9.53% | `source=allenai/tulu-3-sft-personas-code` |
| allenai/tulu-3-sft-personas-instruction-following | 1,986 | 25.25% | 717,795 | 17.14% | 602,927 | 21.73% | `source=allenai/tulu-3-sft-personas-instruction-following` |
| allenai/tulu-3-sft-personas-math | 1,878 | 23.88% | 2,383,964 | 56.93% | 1,845,315 | 66.51% | `source=allenai/tulu-3-sft-personas-math` |
| bevaya/FinQA | 2,000 | 25.43% | 346,266 | 8.27% | 61,786 | 2.23% | `source=bevaya/FinQA` |

### Skill

| Value | Examples | Example % | Raw tokens | Raw % | Supervised tokens | Supervised % | Evidence |
|---|---:|---:|---:|---:|---:|---:|---|
| code | 2,000 | 25.43% | 739,180 | 17.65% | 264,296 | 9.53% | `skill=code` |
| finance | 2,000 | 25.43% | 346,266 | 8.27% | 61,786 | 2.23% | `skill=finance` |
| general | 1,986 | 25.25% | 717,795 | 17.14% | 602,927 | 21.73% | `skill=general` |
| math | 1,878 | 23.88% | 2,383,964 | 56.93% | 1,845,315 | 66.51% | `skill=math` |

### Subskill

| Value | Examples | Example % | Raw tokens | Raw % | Supervised tokens | Supervised % | Evidence |
|---|---:|---:|---:|---:|---:|---:|---|
| code_generation | 2,000 | 25.43% | 739,180 | 17.65% | 264,296 | 9.53% | `subskill=code_generation` |
| constraint_following | 1,986 | 25.25% | 717,795 | 17.14% | 602,927 | 21.73% | `subskill=constraint_following` |
| financial_numerical_reasoning | 2,000 | 25.43% | 346,266 | 8.27% | 61,786 | 2.23% | `subskill=financial_numerical_reasoning` |
| mathematical_reasoning | 1,878 | 23.88% | 2,383,964 | 56.93% | 1,845,315 | 66.51% | `subskill=mathematical_reasoning` |

### Language

| Value | Examples | Example % | Raw tokens | Raw % | Supervised tokens | Supervised % | Evidence |
|---|---:|---:|---:|---:|---:|---:|---|
| en | 7,864 | 100.00% | 4,187,205 | 100.00% | 2,774,324 | 100.00% | `language=en` |

## Most accepted sequences fit below 2048, but length and response workload are source-dependent

Overall input length is p50=346, p90=1,338, and p99=1,873 tokens. Supervised response length is p50=125, p90=1,030, and p99=1,554. Longer responses are not treated as higher quality.

### Input-token length buckets

| Value | Examples | Example % | Raw tokens | Raw % | Supervised tokens | Supervised % | Evidence |
|---|---:|---:|---:|---:|---:|---:|---|
| 1-256 | 2,769 | 35.21% | 413,696 | 9.88% | 131,782 | 4.75% | `length_bucket=1-256` |
| 1025-2048 | 1,546 | 19.66% | 2,114,264 | 50.49% | 1,657,262 | 59.74% | `length_bucket=1025-2048` |
| 257-512 | 2,303 | 29.29% | 773,824 | 18.48% | 341,074 | 12.29% | `length_bucket=257-512` |
| 513-1024 | 1,246 | 15.84% | 885,421 | 21.15% | 644,206 | 23.22% | `length_bucket=513-1024` |

### Supervised-response length proxy

| Value | Examples | Example % | Raw tokens | Raw % | Supervised tokens | Supervised % | Evidence |
|---|---:|---:|---:|---:|---:|---:|---|
| high_257_plus | 3,177 | 40.40% | 3,136,165 | 74.90% | 2,447,902 | 88.23% | `difficulty_proxy=high_257_plus` |
| low_1_64 | 3,020 | 38.40% | 553,792 | 13.23% | 104,271 | 3.76% | `difficulty_proxy=low_1_64` |
| medium_65_256 | 1,667 | 21.20% | 497,248 | 11.88% | 222,151 | 8.01% | `difficulty_proxy=medium_65_256` |

### Length percentiles by slice

| Slice | Input min | p50 | p90 | p99 | max | Supervised min | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| code | 137 | 344 | 635 | 889 | 1,513 | 14 | 78 | 344 | 554 | 817 |
| finance | 80 | 172 | 300 | 487 | 691 | 15 | 28 | 42 | 68 | 140 |
| general | 44 | 331 | 696 | 1,330 | 1,931 | 3 | 255 | 605 | 1,232 | 1,853 |
| math | 527 | 1,261 | 1,716 | 1,998 | 2,044 | 337 | 951 | 1,365 | 1,676 | 1,816 |

The difficulty label above is only a fixed supervised-response-length proxy: low=1–64, medium=65–256, high=257+. It does not establish correctness, reasoning depth, or semantic difficulty.

## Synthetic provenance and repeated prompt prefixes need manual review

Synthetic provenance comes from the pinned source README declarations. Template clusters use the first 16 normalized lexical prompt tokens and require at least five records; they are deliberately candidate signals rather than dedup decisions.

### Source provenance

| Value | Examples | Example % | Raw tokens | Raw % | Supervised tokens | Supervised % | Evidence |
|---|---:|---:|---:|---:|---:|---:|---|
| annotated_benchmark_not_declared_synthetic | 2,000 | 25.43% | 346,266 | 8.27% | 61,786 | 2.23% | `source_provenance=annotated_benchmark_not_declared_synthetic` |
| declared_synthetic | 5,864 | 74.57% | 3,840,939 | 91.73% | 2,712,538 | 97.77% | `source_provenance=declared_synthetic` |

### Template-cluster status

| Value | Examples | Example % | Raw tokens | Raw % | Supervised tokens | Supervised % | Evidence |
|---|---:|---:|---:|---:|---:|---:|---|
| not_repeated | 7,475 | 95.05% | 4,115,147 | 98.28% | 2,760,521 | 99.50% | `template_cluster_status=not_repeated` |
| repeated_prefix_cluster | 389 | 4.95% | 72,058 | 1.72% | 13,803 | 0.50% | `template_cluster_status=repeated_prefix_cluster` |

### Refusal heuristic

| Value | Examples | Example % | Raw tokens | Raw % | Supervised tokens | Supervised % | Evidence |
|---|---:|---:|---:|---:|---:|---:|---|
| no_refusal_phrase_match | 7,850 | 99.82% | 4,184,211 | 99.93% | 2,771,865 | 99.91% | `refusal_status=no_refusal_phrase_match` |
| refusal_phrase_match | 14 | 0.18% | 2,994 | 0.07% | 2,459 | 0.09% | `refusal_status=refusal_phrase_match` |

### Largest repeated prompt-prefix clusters

| Cluster | Examples | Example % | Source counts | Signature preview | Evidence |
|---|---:|---:|---|---|---|
| 00b4be024539 | 37 | 0.47% | finance:37 | use the evidence below to answer the financial question evidence <num> the <num> of <num> is | `template_cluster=00b4be024539` |
| bc7ad9822e44 | 32 | 0.41% | finance:32 | use the evidence below to answer the financial question evidence the <num> net revenue of amount | `template_cluster=bc7ad9822e44` |
| 6c250b0930ed | 26 | 0.33% | finance:26 | use the evidence below to answer the financial question evidence in millions the sales of <num> | `template_cluster=6c250b0930ed` |
| e3d702e6ae5a | 20 | 0.25% | finance:20 | use the evidence below to answer the financial question evidence year ended december <num> in millions | `template_cluster=e3d702e6ae5a` |
| 426cf1c35bee | 12 | 0.15% | finance:12 | use the evidence below to answer the financial question evidence the net sales of <num> is | `template_cluster=426cf1c35bee` |
| 2083277993f2 | 11 | 0.14% | finance:11 | use the evidence below to answer the financial question evidence plan category the equity compensation plans | `template_cluster=2083277993f2` |
| 44dca6f01ee1 | 10 | 0.13% | finance:10 | use the evidence below to answer the financial question evidence the <num> net revenue of in | `template_cluster=44dca6f01ee1` |
| 1afd1628afa7 | 9 | 0.11% | finance:9 | use the evidence below to answer the financial question evidence in millions the net sales of | `template_cluster=1afd1628afa7` |
| aa569430876f | 9 | 0.11% | finance:9 | use the evidence below to answer the financial question evidence in millions except per share amounts | `template_cluster=aa569430876f` |
| 25d17328f7b3 | 8 | 0.10% | finance:8 | use the evidence below to answer the financial question evidence rating equivalent december <num> in millions | `template_cluster=25d17328f7b3` |

## Contract rejections are isolated from the pre-filter baseline

Step 3 rejected 136 records. They do not contribute to any accepted-pool example or token percentage above.

| Rejection reason | Examples | Rejected % | Evidence |
|---|---:|---:|---|
| truncated_assistant | 136 | 100.00% | `contract_rejection_reason=truncated_assistant` |

## Scope, definitions, and method

- Grain: one canonical sample_id per accepted record.
- Cohort: Step 3 accepted records before manual quality decisions, quality filters, exact/near dedup, and decontamination.
- Token units: raw tokens exclude chat-template overhead; input tokens include the frozen template after truncation; supervised tokens are post-truncation labels not equal to -100.
- Percentiles use the nearest-rank method.
- Refusal, response-length difficulty, and prompt-prefix clustering are deterministic risk hints only.
- Exact audit tables are used instead of charts because this is a four-slice static snapshot and sample-ID lookup is the primary need.

## Limitations and robustness checks

- Source-level skill, subskill, language, and provenance labels are pipeline metadata, not independently verified per-example labels.
- The refusal phrase list can miss paraphrases and can produce false positives.
- Response length is confounded with source and answer style; it is not a semantic difficulty measurement.
- Prefix clustering can group structurally similar but substantively different prompts. Step 7 near-duplicate review must not treat it as confirmed duplication.
- Input hashes, aggregate sums, percentage closures, and sample-ID evidence hashes are validated by the audit script.

## Recommended next step

Proceed to Step 8 only after fixing independent eval sources and revisions for general, math, code, and finance. Do not derive evaluation candidates from the training sources or treat the Step 7 similarity threshold as a removal decision.

## Further questions

- Are the source-level language labels correct for every accepted sample?
- Do the largest synthetic prefix clusters contain real task diversity or mostly surface-level persona variation?
- Are long math responses correct and non-redundant, especially in the high response-length proxy bucket?

Audit support hashes:

- summary: `cddabdc420f4da2cb744814411b88bd1c5c1870d84435a944392270b5c27cf52`
- evidence index: `f5536ac58093100db024a02193839903bf13d5a75505b2df12d2b41701d268bf`

<!-- STEP5_FILTER_AUDIT_START -->

## Step 5 preserves all 7,864 accepted records after Gate A approval

The machine-safe pass starts from 7,864 Step 3 accepted records. It applies only confirmed manual rejects and proven deterministic invariant failures. Step 4 risk hints are not deletion rules.

| Stage | Status | Before examples | Removed unique | After examples | Before supervised | After supervised |
|---|---|---:|---:|---:|---:|---:|
| Confirmed manual quality | applied_complete | 7,864 | 0 | 7,864 | 2,774,324 | 2,774,324 |
| Deterministic guardrails | applied | 7,864 | 0 | 7,864 | 2,774,324 | 2,774,324 |

Overall quality-filter accounting is 7,864 - 0 = 7,864. The identity check passed. Downstream eligibility is yes.

### Gate A review is complete

The deterministic review package contains 40 records: 40 reviewed and 0 pending. Review results are 40 accept, 0 reject, and 0 uncertain. No stop condition was triggered, so the filtered pool is eligible for the next pipeline stage.

- Review package: `artifacts/reports/day09-gate-a-review.jsonl`
- Decision ledger: `tmp/day09-work/step5-filter/filter-decisions.jsonl`
- Filter summary: `tmp/day09-work/step5-filter/filter-summary.json`

### Risk hints deliberately remain non-terminal

Refusal phrase matches, repeated prompt prefixes, declared synthetic provenance, and response-length buckets remain review hints. Automatically deleting them would conflate source style or workload with confirmed low quality.

<!-- STEP5_FILTER_AUDIT_END -->

<!-- STEP6_EXACT_DEDUP_START -->

## Step 6 exact dedup removes 4 normalized complete-record duplicates

The downstream-eligible Step 5 cohort contains 7,864 records. Exact dedup retains 7,860 records after removing 4 non-survivors. Prompt-only and answer-only matches remain evidence and do not trigger deletion.

| Match field | Duplicate groups | Affected examples | Duplicate excess | Within source | Cross source | Max group |
|---|---:|---:|---:|---:|---:|---:|
| normalized_prompt | 3 | 7 | 4 | 3 | 0 | 3 |
| normalized_answer | 36 | 83 | 47 | 36 | 0 | 4 |
| normalized_complete_record | 3 | 7 | 4 | 3 | 0 | 3 |

### The survivor rule is deterministic and order-independent

Complete-record groups select one survivor using: manual Gate A accept first, then slice priority finance/general/math/code, then the lowest canonical sample_id. Every removed ID points to that survivor in the decision ledger.

Normalization is `exact_normalization_v1` with config hash `d678e1747eff8ceb60d12ef0e4a1538831ff9795ac1fa66b1928e45639d9c800`: Unicode NFKC, lowercase, collapse whitespace, trim boundaries, and retain punctuation. Complete records are canonical JSON arrays of role and normalized-content pairs.

| Slice | Before | Removed | After | Before supervised | After supervised |
|---|---:|---:|---:|---:|---:|
| code | 2,000 | 0 | 2,000 | 264,296 | 264,296 |
| finance | 2,000 | 2 | 1,998 | 61,786 | 61,734 |
| general | 1,986 | 2 | 1,984 | 602,927 | 602,901 |
| math | 1,878 | 0 | 1,878 | 1,845,315 | 1,845,315 |

Overall accounting is 7,864 - 4 = 7,860; example, raw-token, input-token, and supervised-token identities all passed. Survivor pointers resolve to kept records, and the output complete-record hashes are unique.

### Scope and limitation

This stage proves equality only under the frozen exact normalization. It does not claim that prompt-only, answer-only, paraphrased, or template-similar records are duplicates. Those remain candidates for Step 7 near-duplicate review.

- Match evidence: `tmp/day09-work/step6-exact-dedup/exact-match-groups.jsonl`
- Decision ledger: `tmp/day09-work/step6-exact-dedup/exact-dedup-decisions.jsonl`
- Summary: `tmp/day09-work/step6-exact-dedup/exact-dedup-summary.json`

<!-- STEP6_EXACT_DEDUP_END -->

<!-- STEP7_NEAR_CANDIDATES_START -->

## Step 7 finds 1,129 near-match records without changing the training pool

An exhaustive unordered-pair scan of the 7,860 Step 6 records produced 1,129 field-specific candidate records covering 812 unique sample pairs. Confirmed removals remain zero: the similarity threshold creates review candidates, not deletion labels.

| Field | Candidates | Affected samples | Within source | Cross source | 90-<95 | 95-<100 | 100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| normalized_prompt | 378 | 437 | 378 | 0 | 269 | 109 | 0 |
| normalized_answer | 424 | 224 | 424 | 0 | 292 | 89 | 43 |
| normalized_complete_record | 327 | 412 | 327 | 0 | 240 | 87 | 0 |

### Matcher scope and evidence contract

The matcher is `rapidfuzz.fuzz.ratio` version `3.14.3`, with score >= 90.0 and both normalized values at least 80 characters. Prompt, answer, and complete-record scores are emitted separately. Each candidate stores both sample IDs, sources, slices, score, threshold, matcher config hash, scope, and pending review fields.

### Similarity is not a removal decision

Character-level ratio can flag legitimate shared instructions, finance evidence templates, or mathematical scaffolding. It does not establish semantic duplication. The Step 6 pool therefore remains unchanged, and any future removal requires a Gate B review decision or another frozen deterministic rule.

- Candidates: `tmp/day09-work/step7-near-candidates/near-duplicate-candidates.jsonl`
- Summary: `tmp/day09-work/step7-near-candidates/near-duplicate-summary.json`

<!-- STEP7_NEAR_CANDIDATES_END -->
<!-- STEP7_NEAR_REVIEW_START -->

## Step 7 calibration review samples 20 unique near-match pairs

A deterministic stratified package samples 10 code/code pairs and 10 finance/finance pairs across prompt, answer, and complete-record matches and across the configured score bands. It is a calibration set for human/model review, not an automatic filter.

- Code/code pairs: 10
- Finance/finance pairs: 10
- Pending decisions: 20
- Training pool changed: false
- Review package: `artifacts/reports/day09-step7-near-review.jsonl`

<!-- STEP7_NEAR_REVIEW_END -->
<!-- STEP8_EVAL_CANDIDATES_START -->

## Step 8 fixes 160 independent eval candidates, not the Day 10 protocol

Four independently sourced candidate pools now contain 40 records each. Their IDs, prompts, references, revisions, and content hashes are stable; decoder settings, scoring, baselines, and the final evaluation protocol remain intentionally unfrozen until Day 10.

| Slice | Source | Revision | Split | Candidates | License |
|---|---|---|---|---:|---|
| general | `cais/mmlu` | `c30699e8356da336a370243923dbaf21066bb9fe` | test | 40 | MIT |
| math | `openai/grade-school-math` | `3101c7d5072418e28b9008a6636bde82a006892c` | test | 40 | MIT |
| code | `openai/human-eval` | `6d43fb980f9fee3c892a914eda09951f772ad10d` | test | 40 | MIT |
| finance | `NExTplusplus/TAT-QA` | `870accc41953dcde885aabeb963d94aabdc0fbc3` | dev | 40 | CC-BY-4.0 dataset; MIT code |

### Stability and scope

Every candidate includes `eval_sample_id`, source/revision/split/parent ID, prompt, reference, skill, content hash, selection hash, adapter, license, and source-file hash. MMLU is stratified across four general subjects at 10 candidates each; the other slices use stable-hash sampling from their pinned split.

These pools are inputs to Step 9 train/eval overlap detection. They must not yet be described as decontaminated or as the frozen Day 10 evaluation protocol.

- Candidates: `tmp/day09-work/step8-eval-candidates/*-eval-candidates.jsonl`
- Summary: `tmp/day09-work/step8-eval-candidates/eval-candidates-summary.json`

<!-- STEP8_EVAL_CANDIDATES_END -->

<!-- STEP9_DECONTAMINATION_START -->

## Step 9 scans all train/eval boundaries without automatic deletion

The scan compares 7,860 Step 6 train records with 160 Step 8 eval candidates. It emitted 0 overlap candidates: 0 exact and 0 near. Candidate status is not a contamination decision.

| Boundary | Exact | Near | Total | Near pairs scored | Max eligible score |
|---|---:|---:|---:|---:|---:|
| train_prompt_vs_eval_prompt | 0 | 0 | 0 | 1,257,600 | 52.386238 |
| train_answer_vs_eval_reference | 0 | 0 | 0 | 1,052,730 | 69.148933 |
| train_complete_record_vs_eval_record | 0 | 0 | 0 | 1,257,600 | 55.135136 |

### Evidence contract

Exact equality uses the frozen NFKC/lowercase/whitespace normalization without a length threshold. Near matching uses `rapidfuzz.fuzz.ratio` version `3.14.3`; exact pairs are excluded from near counts. Prompt, answer/reference, and complete-record boundaries remain separate in both the ledger and summary.

No train or eval record was removed. Gate B must review every emitted candidate before a contamination decision can change either pool.

- Candidates: `tmp/day09-work/step9-decontamination/train-eval-overlap-candidates.jsonl`
- Summary: `tmp/day09-work/step9-decontamination/decontamination-summary.json`

<!-- STEP9_DECONTAMINATION_END -->

<!-- STEP10_CLEAN_POOL_START -->

## Step 10 freezes one clean parent pool with 7,860 canonical records

Confirmed quality, exact-dedup, near-review, and contamination decisions have been reconciled into one parent pool shared by future Mix A and Mix B. No new removal occurs after Step 6, so accounting remains 7,864 accepted records minus 4 exact complete-record duplicates equals 7,860 clean records.

| Slice | Examples | Raw tokens | Input tokens | Supervised tokens |
|---|---:|---:|---:|---:|
| code | 2,000 | 739,180 | 781,180 | 264,296 |
| finance | 1,998 | 345,932 | 387,890 | 61,734 |
| general | 1,984 | 717,747 | 759,411 | 602,901 |
| math | 1,878 | 2,383,964 | 2,423,402 | 1,845,315 |
| **Total** | **7,860** | **4,186,823** | **4,351,883** | **2,774,246** |

### Freeze and rebuild evidence

The order-independent clean-pool hash is `ab7dee175c140ef5fa5bf8e0f9197046d166f4e0eef98af5ec14ba543e2eb073`. The manifest hash is `95ce7aeb02efc4dbce50786c51c0e87db27dde4e7f9778e2a6927c5dd991c27f`. Five deterministic rebuild samples passed source/revision/parent, content-hash, transform-chain, positive-supervision, and Step 6 artifact checks.

- Manifest: `artifacts/data/day09-dataset-manifest.json`
- Gate B ledger: `artifacts/reports/day09-gate-b-review.jsonl`
- Rebuild evidence: `artifacts/reports/day09-manifest-rebuild-evidence.jsonl`
- Summary: `tmp/day09-work/step10-clean-pool/clean-pool-summary.json`

<!-- STEP10_CLEAN_POOL_END -->

<!-- STEP11_TOKEN_BUDGET_START -->

## Step 11 fixes an exact common budget of 246,936 supervised tokens

The maximum whole-example, without-replacement budget jointly feasible for Mix A and Mix B is 246,936 supervised tokens. Finance is the binding slice: its 61,734 available tokens must represent 25% in both mixtures. Deterministic subset-sum reconstruction reaches every slice target exactly, so no replacement or answer truncation is required.

| Mix | General | Math | Code | Finance | Total |
|---|---:|---:|---:|---:|---:|
| mix_A_balanced | 61,734 | 61,734 | 61,734 | 61,734 | 246,936 |
| mix_B_targeted | 30,867 | 30,867 | 123,468 | 61,734 | 246,936 |

The common budget is 8.90% of the clean pool's supervised tokens. Ratio tolerance and total-token tolerance are both zero. Occurrence counts are recorded and every selected sample has occurrence_count=1.

Step 11 freezes feasibility and budget only. Step 12 will materialize the final Mix A/B manifests from these proofs.

- Plan: `tmp/day09-work/step11-token-budget/token-budget-plan.json`
- Plan hash: `45c0366af6e18f317da46b0af35ca50aa56b36d138fb25fdd1ce4f200c9ebbed`

<!-- STEP11_TOKEN_BUDGET_END -->

<!-- STEP12_MIXTURES_START -->

## Step 12 materializes two equal-token mixture manifests

Mix A and Mix B each contain 246,936 supervised tokens. Both are direct child manifests of the same clean parent pool and Step 11 plan. All frozen preprocessing, filtering, deduplication, decontamination, source-lineage, ordering, and total-token invariants match; only the target slice ratios and resulting occurrences differ.

| Manifest | Unique examples | Occurrences | Supervised tokens | Manifest hash |
|---|---:|---:|---:|---|
| Mix A balanced | 2,730 | 2,730 | 246,936 | `19ea1a93e860fc5e93f463f7a136ae834ebe041c00f9fe3fa874f66ab1428c7a` |
| Mix B targeted | 3,078 | 3,078 | 246,936 | `8409de2d47c51cf0a29a22f5af0fd7bd136b98cfb1ed7de72682ffebb3bd5aaf` |

No replacement was needed: maximum sampling_count is 1 in both manifests. Canonical sample_id values are preserved and occurrence_id is stored separately. Gate C was explicitly waived by the user rather than passed: 0 of 60 planned occurrences were manually reviewed. The manifests are downstream-eligible only under that documented risk acceptance.

- Mix A file SHA-256: `cb0d4819d383e4a22fd2c389c27c9cf058a20287c08e0fa5d924703f0af77492`
- Mix B file SHA-256: `d87e32b7925080c4e1abc6676354cb35e60af3f2311b969d6441a8aacc319837`
- Step 12 summary hash: `108293e3555036d4c778b218daacad26c3d2c8ea2ae273468853797961a631cb`

<!-- STEP12_MIXTURES_END -->

<!-- STEP13_FINAL_ACCEPTANCE_START -->

## Day 09 completes with reproducibility evidence and one explicit QA waiver

The dataset pipeline is complete with status `complete_with_gate_c_waiver`. Gate C was waived by the user rather than passed: 0 of 60 planned final occurrences received human review. All automated claims below remain valid, while final occurrence-level content quality is an accepted residual risk.

| Final dataset state | Result |
|---|---:|
| Step 3 accepted after the 2,048-token contract | 7,864 |
| Exact complete-record removals | 4 |
| Confirmed near-duplicate removals | 0 |
| Confirmed contamination removals | 0 |
| Clean parent records | 7,860 |
| Clean supervised tokens | 2,774,246 |
| Common supervised-token budget per mix | 246,936 |
| Final automated acceptance checks | 11 / 11 |
| Unit/integration tests | 44 passed |

The fresh-directory replay reproduced all 30 Steps 2–9 data/evidence files byte-for-byte and reproduced the clean records, clean-pool hash, budget, ratios, and selection proofs. A second frozen-input replay reproduced the clean parent manifest, token-budget plan, Mix A, and Mix B files byte-for-byte.

Two path-dependence defects were found and fixed during acceptance: Step 10 lineage evidence no longer assumes the default work directory, and Step 8 eval candidates now serialize repository-relative source paths instead of machine-specific absolute paths.

### Remaining analytical risks

- Gate C is waived, so the final mixtures have no occurrence-level human QA evidence beyond earlier Gate A and Gate B review.
- Mix A and B control supervised tokens exactly, but input tokens differ (728,317 vs 833,607), which can change forward-pass compute.
- Near-duplicate and decontamination conclusions are bounded by the frozen normalization, fields, RapidFuzz ratio matcher, and thresholds; they do not prove absence of arbitrary semantic paraphrases.
- Finance is the binding slice, limiting each mixture to 246,936 supervised tokens, or 8.90% of the clean pool's supervised-token volume.

- Rebuild acceptance: `artifacts/reports/day09-rebuild-acceptance.json`
- Separate dedup/decontamination report: `artifacts/reports/day09-decontamination-report.md`

<!-- STEP13_FINAL_ACCEPTANCE_END -->
