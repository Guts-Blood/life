# Day 22 Qwen3.5 Coding Preference Data Audit

执行日期：`2026-08-13`
结论：`formal promoted-S1 machine gates PASS / human blind review pending`

## 结论先行

Day 22 已从计划文档落成一条可重建、可执行、可重分词、可 fail-closed 的 preference-data pipeline。历史 synthetic smoke 的 63 对仍作为机制证据保留；正式 promoted-S1 multi-round pipeline 已另外冻结 200 个 on-policy、non-synthetic pair，并通过全部 machine gates。

正式 manifest 当前为 `machine_ready=true`、`formal_dpo_ready=false`。唯一 blocker 是 `human_blind_review_pending`：50 个 unique pair 的 primary/swapped worksheet 已生成，但人工 verdict 仍为 0/50。Day 23 在该 gate 完成前不得训练。

## Formal promoted-S1 multi-round 结果

### Rollout、去重与执行分类

- Parent：Day 21 promoted S1 `main-s20260809-lr1e-4-final`，每个候选绑定 downstream key、promotion manifest、merged export manifest、model snapshot、runtime 与 generation contract。
- Base round：330 families × K=6 = 1,980 raw candidates；338 exact duplicates、4 format-invalid，留下 1,638 个 execution-unique candidates。
- Base E2B：746 stable pass、719 stable wrong-answer、173 quarantine；得到 141 个 pass-vs-wrong family，另有 101 both-pass、85 both-fail、3 runtime-only。
- Adaptive R1：对 189 个 quarantine family 用同一 S1、`temperature=0.8`、`top_p=0.95` 补采样 K=12，共 2,268 raw candidates；两张 GPU 以 family 为单位隔离为 1,140/1,128。
- Adaptive 过滤：6 format-invalid、614 round-local exact duplicates、243 cross-round exact duplicates；只对 1,405 个真正新增回答执行 2,810 次 fresh E2B。
- Adaptive E2B：484 stable pass、711 stable wrong-answer、210 quarantine。
- 最终合并：3,043 个唯一候选，合计 1,230 stable pass、1,430 stable wrong-answer、383 quarantine；形成 200 个正式 pair，70 family only-pass、60 family only-fail。

所有 candidate code 只在 pinned E2B Firecracker 中执行，每条 evidence 恰好两次 fresh、断网运行；所有 run ID、candidate/evidence self-hash、test hash、sandbox digest 与 promoted-S1 provenance 都由 merge、labeler、assembler 和独立 validator 重新校验。

### Pair 组成与 processor

| Pair 来源 | 数量 |
|---|---:|
| base chosen / base rejected | 141 |
| base chosen / adaptive rejected | 11 |
| adaptive chosen / base rejected | 6 |
| adaptive chosen / adaptive rejected | 42 |

全部 200 对的 chosen 都是两次 pass，rejected 都是两次 verifier-classified wrong-answer；runtime、timeout、infra error、both-pass、both-fail 和 AST-equivalent response 均不进入 pair。Synthetic pairs = 0。

Qwen3.5-4B-Base processor 使用 revision `1001bb4d826a52d1f399e183466143f4da7b741b` 重新编码 200 对：200/200 pass、prompt token prefix 相同、response-only labels 连续、causal shift 正确、四空格边界受监督、无 truncation。Processor contract SHA-256 为 `12af5b3df5597fc2de4e6e9eebc9abba2e0f14d27971f39dc7e4af83c66e1083`。

### Formal machine gates

| Gate | 观测 | 结果 |
|---|---:|---|
| Accepted pairs 200–500 | 200 | PASS |
| One pair per native family | 200/200 | PASS |
| Promoted-S1 on-policy | 200/200 | PASS |
| Synthetic fraction | 0% | PASS |
| Secure E2B, 2 fresh runs/side | 200/200 | PASS |
| Qwen3.5 processor audit | 200/200 | PASS |
| Cross-split problem/prompt/test/source overlap | 0/0/0/0 | PASS |
| Blind-review packet | 50 unique / 100 presentations | PASS（packet generation） |
| Completed human blind review | 0/50 | PENDING |

正式 split 为 train/dev/heldout = 154/17/29。Machine-gate audit SHA-256 为 `a01bfdb8de74f655fc279e8da7c1d0cea41ae9175ac013abb191d3c7549ca032`；formal manifest SHA-256 为 `5b918b2a723fc3d44449193a5a72d5eb6a669a420243bbf08681d4b62fe0b26c`。

## 数据来源与选择

- Source：`google-research-datasets/mbpp`
- Revision：`4bb6404fdc6cacfda99d4ac4205087b89d32030c`
- License：`CC-BY-4.0`；本报告与 manifest 保留 attribution、revision、原始文件 hash 与派生方法。
- 只使用官方 train Parquet：
  - full：SHA-256 `09d125ca31edacb7800be8c67c45abff618faf0214ff551291817d06bdb914ae`
  - sanitized：SHA-256 `d95f8ad6d2fff08fe4826122d6e3e31f75716825d0c5c340d297aca5e9e0de0e`
- Day 20 main 有 422 个 MBPP-train rows / 351 个 native task families；排除 probe 中 21 个 family 后，clean pool 为 330。
- full/sanitized 以 native task ID 为 family，绝不跨 variant 混用 prompt/code/tests；最终 selection 为 107 sanitized + 223 full。
- 80-task smoke 在生成前按 family 冻结为 60 train / 10 dev / 10 heldout。一个 task 的 prompt、test 和所有变体只能进入一个 split。

原始 Parquet 仅 staging 到 gitignored 的 `tmp/day22-raw/mbpp/`；提交的 seed manifest 保存 lineage 和 hashes，不依赖临时绝对路径。

## Pair 构建与执行证据

Chosen 使用 MBPP canonical solution。Rejected 来自版本 `day22.mbpp_single_ast_bug_v1` 的单点确定性 AST mutation；80 个 task 共生成 500 个 mutation candidate，首轮每题固定选择 index 0 做机制 smoke。

所有 chosen/rejected 使用同一 test bytes，在 E2B template `rki5dems9wqfm4r03t7g` 中各 fresh replay 两次：Firecracker isolation、断网、无 mount/env/MCP、Python 3.11.6、2 vCPU、512 MiB、4 秒 wall timeout。共执行 320 个 fresh runs。

| Pair outcome | 数量 | 处理 |
|---|---:|---|
| chosen `pass,pass`; rejected `wrong_answer,wrong_answer` | 63 | audited smoke pair |
| both pass | 12 | quarantine；公开测试无法区分 |
| rejected runtime error 两次 | 5 | quarantine；不冒充 wrong answer |

63 个 accepted pair 的 split 为 50/7/6；执行证据中 chosen pass runs = 126，rejected wrong-answer runs = 126。没有 timeout、infra error 或不稳定结果进入 pairs。

## Qwen3.5 processor 与 mask

原计划依赖的 Day 15 processor artifact 从未生成，因此本次从真实 Qwen3.5-4B-Base snapshot 和 Day 20 v3 target encoding 重新解析并冻结 Day 22 contract：

- Model revision：`1001bb4d826a52d1f399e183466143f4da7b741b`
- Runtime：Python 3.11.15、Transformers 5.12.1、ms-swift 4.5.0.dev0、huggingface-hub 1.27.0
- Template contract SHA-256：`7f835342861cc9aabfd6299228033df80437b183227e04db311450437cd07ded`
- Processor contract SHA-256：`12af5b3df5597fc2de4e6e9eebc9abba2e0f14d27971f39dc7e4af83c66e1083`

80/80 pairs 均通过逐 token audit：chosen/rejected prompt prefix 相同，labels 只覆盖连续 response span，四空格首 token 受监督，causal shift 正确，无 truncation。Day 20 v3 会把末尾 newline 保持 masked，因此合法 span 是 `response_span.end <= input_token_count`，实际为末尾保留 1 个 masked token。

## Bias、split 与 review

- Accepted relative response-token length delta：matched (`≤0.10`) 54、mid (`>0.10, ≤0.50`) 6、large (`>0.50`) 3。
- Source variant：full 41、sanitized 22。
- Test count：3 tests 59、4 tests 3、6 tests 1；MBPP 的公开测试较弱，因此 both-pass 不转成 preference。
- problem、prompt、test、source family 的 train/dev/heldout 两两 overlap 都为 0。
- 已生成 50 个 unique pair 的 blind-review worksheet；每对各有原始与交换顺序两次展示，共 100 rows。Concealed key 独立保存，当前 completed human reviews = 0。

MBPP 是单一 source，不能把 split 伪称为 source-held-out。并且这些 family 来自 Day 20 main；若当前/后续 S1 已消费该 main，本 heldout 只代表 preference-stage held-out 和 MBPP in-domain diagnostic，不能宣称 model-level unseen 或泛化。

## Historical smoke formal readiness gates

| Gate | 观测 | 结果 |
|---|---:|---|
| Accepted pairs ≥200 | 63 | FAIL |
| Synthetic fraction ≤20% | 100% | FAIL |
| Human blind review ≥50 | 0 completed；worksheet 50 | FAIL |
| Secure sandbox for all accepted pairs | 63/63 | PASS |
| Promoted-S1 on-policy non-synthetic pairs | 0 | FAIL |

这张表只记录历史 smoke 当时为何不能作为正式数据；它没有被覆盖或伪装成 formal bundle。后续 promoted-S1 multi-round formal bundle 已解决数量、synthetic 与 on-policy 三项问题，现在只剩人工盲审。

## 产物与验证

主要入口：

- Manifest：[`../data/day22-qwen35-coding-preference-manifest.json`](../data/day22-qwen35-coding-preference-manifest.json)
- Pairs：[`../data/day22-qwen35-coding-preference-pairs.jsonl`](../data/day22-qwen35-coding-preference-pairs.jsonl)
- Split IDs：[`../data/day22-qwen35-coding-preference-split-ids.json`](../data/day22-qwen35-coding-preference-split-ids.json)
- E2B evidence：[`../eval/day22-qwen35-mbpp-smoke-e2b-sandbox-evidence.jsonl`](../eval/day22-qwen35-mbpp-smoke-e2b-sandbox-evidence.jsonl)
- Processor evidence：[`../eval/day22-qwen35-mbpp-smoke-processor-audit.jsonl`](../eval/day22-qwen35-mbpp-smoke-processor-audit.jsonl)
- Processor contract：[`../configs/day22-qwen35-processor-template-contract.json`](../configs/day22-qwen35-processor-template-contract.json)
- Validator：[`../scripts/validate_day22_coding_preference.py`](../scripts/validate_day22_coding_preference.py)

从 bootcamp 仓库根目录验证：

```bash
python3 learning/post-training-30-day-bootcamp/artifacts/scripts/validate_day22_coding_preference.py
python3 -m unittest discover \
  -s learning/post-training-30-day-bootcamp/day-22-preference-data \
  -p 'test_*.py' -v
```

第一条应返回 `status=valid_audited_smoke`；当前 Day 22 全套 98 个测试应全部通过。下游必须使用严格 gate：

```bash
python3 learning/post-training-30-day-bootcamp/artifacts/scripts/validate_day22_coding_preference.py \
  --require-formal-ready
```

当前该命令预期返回退出码 3，防止将 synthetic smoke 误当成正式 DPO 数据。Manifest SHA-256 为 `794b13c7ba852a03368f370a6c9760be484e1fc2f93f0edea7df991698bd21c5`。

正式 bundle 的主要入口：

- Manifest：[`../data/day22-qwen35-formal-s1-manifest.json`](../data/day22-qwen35-formal-s1-manifest.json)
- Pairs：[`../data/day22-qwen35-formal-s1-preference-pairs.jsonl`](../data/day22-qwen35-formal-s1-preference-pairs.jsonl)
- Split IDs：[`../data/day22-qwen35-formal-s1-split-ids.json`](../data/day22-qwen35-formal-s1-split-ids.json)
- Candidate E2B evidence：[`../eval/day22-qwen35-formal-s1-multiround-candidate-e2b-evidence.jsonl`](../eval/day22-qwen35-formal-s1-multiround-candidate-e2b-evidence.jsonl)
- Processor audit：[`../eval/day22-qwen35-formal-s1-processor-audit.jsonl`](../eval/day22-qwen35-formal-s1-processor-audit.jsonl)
- Assembly audit：[`day22-qwen35-formal-s1-assembly-audit.json`](day22-qwen35-formal-s1-assembly-audit.json)
- Blind worksheet：[`../data/day22-qwen35-formal-s1-blind-review.jsonl`](../data/day22-qwen35-formal-s1-blind-review.jsonl)
- Concealed key：[`../data/day22-qwen35-formal-s1-blind-review-key.json`](../data/day22-qwen35-formal-s1-blind-review-key.json)
- Independent validator：[`../scripts/validate_day22_formal_s1_bundle.py`](../scripts/validate_day22_formal_s1_bundle.py)
- Review protocol：[`../../day-22-preference-data/FORMAL-S1-BLIND-REVIEW-PROTOCOL.md`](../../day-22-preference-data/FORMAL-S1-BLIND-REVIEW-PROTOCOL.md)
- Review finalizer：[`../../day-22-preference-data/finalize_day22_formal_s1_review.py`](../../day-22-preference-data/finalize_day22_formal_s1_review.py)

从 bootcamp 仓库根目录重验正式 machine bundle：

```bash
python3 learning/post-training-30-day-bootcamp/artifacts/scripts/validate_day22_formal_s1_bundle.py \
  --manifest learning/post-training-30-day-bootcamp/artifacts/data/day22-qwen35-formal-s1-manifest.json
```

预期状态为 `valid_formal_s1_bundle_human_review_pending`、machine gate `PASS`。加 `--require-formal-ready` 必须返回退出码 3，且唯一 blocker 必须是 `human_blind_review_pending`。

盲审完成后的命令、判定口径与 ready-manifest 路径见 review protocol。当前 200 对恰好等于最低门槛，因此任何人工 exclusion 都会保持 blocked；finalizer 会落审计理由但绝不生成伪 ready artifact。若 50 对全部一致支持 verifier chosen，finalizer 生成独立 ready manifest，validator 的 `--require-formal-ready` 才能返回 0。
