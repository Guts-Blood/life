# Day 09 Assignment — 数据质量、Mixture、Lineage 与去污染

日期：`2026-08-04`

状态：`complete_with_gate_c_waiver`

预计用时：`4–6 小时`

硬件：`CPU only`

本文档的用途：按顺序执行 Day 09。每一步都包含目的、动作、产物、正确证据、错误信号和停止条件。不要因为脚本运行成功就勾选任务；只有对应证据通过时才能勾选。

---

## 0. 如何使用这份 Assignment

按 `Step 0 -> Step 13` 执行，不跳步。

每次只执行一个 Step，可以直接对 Codex 说：

```text
执行 Day 9 assignment 的 Step N。
完成后停在该 Step 的验收点，说明产物、测试、实际数据和残余风险，不自动进入下一步。
```

四个需要人工判断的节点：

1. `Gate 0`：确认 source、license 用途、target slice 和 mixture 比例。
2. `Gate A`：人工审核每个 source 10 条，共 40 条。
3. `Gate B`：人工确认 near duplicate 和 train/eval overlap candidates。
4. `Gate C`：最终抽查 Mix A/B 各 30 个 training occurrences。

其他步骤都应是确定性、可重跑的自动处理。

---

## 1. Assignment 背景

Day 08 已经固定了一条 SFT sample 的变换契约：

```text
raw messages
-> chat template
-> input IDs
-> assistant-only labels
-> truncation
-> supervised tokens
```

Day 09 要解决的是 dataset-level 问题：

```text
样本从哪里来？
使用的是哪一个 source snapshot？
经过了哪些处理？
为什么被保留或删除？
是否有重复或 eval 污染？
模型实际看到的 supervised-token 分布是什么？
A/B 能否只改变 mixture 比例？
```

本 assignment 不训练模型。它为 Day 10 的 frozen eval 和 Day 12 的受控 SFT 提供数据资产。

---

## 2. Definition of Done

必须同时满足：

- [x] 任意 training occurrence 都能追溯到 canonical sample、source、revision 和 parent record。
- [x] 每个 source 都同时报告 example、raw-token 和 supervised-token 分布。
- [x] 每个 source 至少完成 10 条人工质量审核。
- [x] exact duplicate、near-duplicate candidate 和 train/eval overlap 分开报告。
- [x] 所有删除都有 before/after count、sample ID 和 reason。
- [x] Mix A/B 使用相同 clean parent pool 和 preprocessing。
- [x] Mix A/B 的实际总 supervised tokens 完全相同。
- [x] Mix A/B 唯一有意改变的变量是 slice token 比例。
- [ ] Mix A/B 各 30 个 occurrences 的最终人工 QA 通过。**用户明确豁免；0/60 reviewed，不声称通过。**
- [x] 从空临时输出目录重跑能生成相同的 sample selection 和 manifest hashes。

Day 09 以一个明确例外完成：Gate C 被用户豁免而非通过。自动验收和复现性均通过；最终 manifests 的 downstream eligibility 仅建立在该风险接受之上。

---

## 3. 固定输入

### 3.1 Day 08 preprocessing contract

本 assignment 必须复用 [`day08-sft-data-contract.md`](../artifacts/data/day08-sft-data-contract.md)，不得另起一套口径。

| 项目 | 固定值 |
|---|---|
| Model/tokenizer | `Qwen/Qwen3-0.6B-Base` |
| Revision | `ddc928429ed09d9ad603fd762053d0434c15e865` |
| Transformers | `4.57.3` |
| Template SHA-256 | `44d5f08f3f72b837eaad09f13a54c1f9f4eb58d75240334548b7fd52a5437fa5` |
| Tokenizer config SHA-256 | `7d0fc4691628c9f6b0c77f8138a80517896a307e074dd5ac87e279ea85515774` |
| Audit max length | `2048`（Day 09 length amendment v3；Day 08 default 为 `256`） |
| Truncation | right |
| Padding | right，`attention_mask=0`，`labels=-100` |
| Loss mask | 所有 assistant body 及其 `<|im_end|>` |
| Packing | disabled |
| 不完整 final assistant | reject |
| 截断后零监督 | reject |

Length amendments 只修改整条 rendered sequence 的上限：v2 从 `256` 调整为 `1024`，v3 再从 `1024` 调整为 `2048`。Tokenizer revision、template hash、right truncation、assistant-only labels、padding、packing 和所有 post-truncation validators 保持 Day 08 口径不变。原 `256` 与 `1024` 运行结果均保留为诊断和比较证据。

### 3.2 Training source candidates

| Slice | Dataset | Pinned revision | 候选量 |
|---|---|---|---:|
| general / instruction-following | [`allenai/tulu-3-sft-personas-instruction-following`](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following) | `fe0c7d350c9b4542b8d829a6f1daa1c259f0ba0e` | 2,000 |
| math | [`allenai/tulu-3-sft-personas-math`](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-math) | `46add032f5203fd96efbd54d074198981ed57d25` | 2,000 |
| code | [`allenai/tulu-3-sft-personas-code`](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-code) | `1412abe88dd2976af977260788e033013449f7b2` | 2,000 |
| finance / numerical reasoning | [`bevaya/FinQA`](https://huggingface.co/datasets/bevaya/FinQA) | `3d6a736bc67e06bc15fbf3618d88204a57c5b25e` | 2,000 |

前三个 Tulu source 的 dataset card 标记为 ODC-BY，包含 synthetic model outputs。FinQA 使用完整字段的 Hugging Face repackaging，并额外记录 canonical GitHub revision `0f16e2867befa6840783e58be38c9efb9229d742`；其 annotation、底层 FinTabNet 和 IBM mirror 的 license metadata 不完全一致。四个 source 均仅用于本次个人研究/教育 assignment，不将其视为已完成商业数据合规审查。

FinQA 只从官方 `train` split 进入训练候选。转换固定为：`gold_inds + question -> user`，`program + answer -> assistant`；不调用模型生成新答案。`validation/test` 只下载并隔离，不在 Gate 0 自动指定为 Day 10 eval。

### 3.3 拟议 mixture

```text
mixture unit = post-template/post-mask/post-truncation supervised tokens
target slice = code

Mix A balanced:
  general = 25%
  math    = 25%
  code    = 25%
  finance = 25%

Mix B targeted:
  general = 12.5%
  math    = 12.5%
  code    = 50%
  finance = 25%
```

总 supervised-token budget 不在本阶段伪造；它在 clean pool 生成后根据实际 token counts 决定。

---

## 4. 目录和边界

工作 repo：`/Users/jiaweiqian/Desktop/python_projects/life`

代码与实验说明：

```text
learning/post-training-30-day-bootcamp/
└── day-09-data-quality-mixture-lineage/
    ├── assignment_day9.md
    ├── day09_assignment.json
    ├── prepare_day09_data.py
    ├── audit_day09_quality.py
    ├── build_day09_mixtures.py
    ├── test_day09_pipeline.py
    └── manual-review.jsonl
```

必交产物：

```text
artifacts/data/day09-dataset-manifest.json
artifacts/data/day09-mix-A-balanced.json
artifacts/data/day09-mix-B-targeted.json
artifacts/reports/day09-data-quality-audit.md
artifacts/reports/day09-decontamination-report.md
```

机器可读审计日志：

```text
artifacts/logs/day09-removed-samples.jsonl
artifacts/logs/day09-near-duplicate-candidates.jsonl
artifacts/logs/day09-eval-overlap-candidates.jsonl
```

原始下载缓存：

```text
/Users/jiaweiqian/Desktop/python_projects/life/tmp/day09-hf-cache/
```

`tmp/` 已被 Git 忽略。不把原始 Parquet 提交到 repo。不修改 `learning/training-learning/ms-swift/`；Day 09 是实验数据层，`ms-swift` 仅在 Day 12 消费最终 manifest。

---

## 5. 全局正确性规则

### 5.1 原始 source 不可变

```text
raw snapshot -> derived candidate -> clean pool -> mixture manifest
```

只生成 derived artifacts，不覆盖 raw snapshot。

### 5.2 三类身份不混淆

```text
canonical sample_id:
  一条内容资产的稳定身份

content_hash:
  最终 messages 内容的指纹

occurrence_id / sampling_count:
  该 canonical sample 在某个 mixture 中被看到的次数或位置
```

过采样不伪造新 canonical IDs。

### 5.3 三种 token 不混淆

```text
examples != raw tokens != supervised tokens
```

Day 09 的 mixture 单位是 supervised tokens。

### 5.4 Dedup 不等于 Decontamination

```text
dedup:
  train 内部或跨 train sources 的重复

decontamination:
  train 和 eval 之间的 prompt/answer/record overlap
```

### 5.5 候选不等于删除

Near duplicate 和 eval overlap 只能先生成 candidates。未经规则或人工决定，不静默删除。

---

## Step 0 — 预注册 Assignment

**目的**

在看到数据结果前，固定 source、口径、目标和阈值，防止事后调参。

**动作**

创建 `day09_assignment.json`，至少写入：

```text
source identifiers and revisions
license notes
candidate_samples_per_slice = 2000
stable selection method
selection seed / shuffle seed
Day 08 preprocessing hashes
normalization policy
near matcher name/version/threshold
target slice = code
Mix A / Mix B target ratios
manual review sample size
```

建议 exact normalization v1：

```text
Unicode NFKC
lowercase
collapse consecutive whitespace
strip leading/trailing whitespace
keep punctuation
```

Near threshold 必须与具体 matcher 一起记录；单独写 `0.85` 没有可复现意义。

**产物**

```text
day09_assignment.json
```

**正确证据**

- 每个 source 使用 40 位 immutable revision，不是 `main`。
- preprocessing hashes 与 Day 08 一致。
- Mix A/B 比例和 target slice 已在读取质量结果前固定。
- 选样、normalization 和 matcher 可由配置完整重建。

**错误信号**

- revision 使用 `main` 或 `latest`。
- 只写 threshold，不写 matcher 和比较字段。
- 看到某个 slice 质量好后再把它改成 target slice。

**Gate 0：人工确认**

- [x] 同意仅用于个人研究/教育。
- [x] 同意四个 pinned sources，包括 FinQA。
- [x] 同意每个 source 候选 2,000 条。
- [x] 同意 target slice 为 code。
- [x] 同意 Mix A = `25%,25%,25%,25%`，Mix B = `12.5%,12.5%,50%,25%`（general/math/code/finance）。

---

## Step 1 — 建立最小可测试代码骨架

**目的**

将 source loading、preprocessing、audit 和 mixture selection 分开，避免一个脚本同时改变所有状态。

**动作**

建立：

```text
prepare_day09_data.py
audit_day09_quality.py
build_day09_mixtures.py
test_day09_pipeline.py
```

建议边界：

```text
prepare:
  load -> select -> canonicalize -> Day 08 preprocess -> manifest

audit:
  distributions -> manual sample -> exact -> near -> eval overlap

build:
  clean pool -> equal-token A/B manifests

tests:
  identity -> hashes -> counts -> invariants -> reproducibility
```

**正确证据**

- `--help` 不需要下载数据即可运行。
- unit tests 使用 3–5 条 local fixtures，不依赖网络。
- 输出目录可配置，测试不污染正式 artifacts。
- Day 08 处理逻辑被复用，没有复制一套不同 mask 逻辑。

**错误信号**

- import 脚本时立即下载数据。
- 测试需要 Hugging Face 网络才能通过。
- 代码覆盖 raw files 或将大文件写入 Git 路径。

- [x] Step 1 测试通过。

---

## Step 2 — 获取 Source Snapshot 并稳定选样

**目的**

得到可重建的 8,000 条 candidate pool，不依赖 dataset 当前排序。

**动作**

对每个 source：

```text
1. 按 pinned revision 读取 train split。
2. 验证 original id 非空且在 source 内唯一。
3. 计算 sha256(source + revision + original_id)。
4. 按 selection hash 排序。
5. 取前 2,000 条。
```

不直接取 source 的前 2,000 条，避免原始排序偏差。

**产物**

Candidate manifest，每条至少包含：

```text
source
revision
original_id / parent_id
selection_hash
raw messages
skill / subskill
language
retrieved_at
```

**正确证据**

```text
general candidates = 2,000
math candidates    = 2,000
code candidates    = 2,000
finance candidates = 2,000
duplicate parent IDs per source = 0
```

在不改配置的情况下重跑，8,000 个 `(source, parent_id)` 完全相同。

**错误信号 / 停止条件**

- 实际获取的 source SHA 与 pinned revision 不一致。
- original IDs 大量缺失或重复。
- 重跑选出了不同 IDs。
- 未经记录地 fallback 到 `main`。

- [x] Step 2 选样证据通过。

**2026-08-04 实际证据**

| Slice | Train rows | Candidates | Candidate JSONL SHA-256 |
|---|---:|---:|---|
| general | 29,980 | 2,000 | `7c04077f7fa6ff217d4ee898d4b41fbc0b15788e59ed3715625046f048342182` |
| math | 149,960 | 2,000 | `d6e5ae115bcd82e71c5ac45ce5979827c979d1c5ebeffd77b1a297e3ac8dc194` |
| code | 34,999 | 2,000 | `edfcdf7d581ceae4c4f475dd68aca508cbe8b704d4bcf3e9d70a35262c4b5fb5` |
| finance | 6,251 | 2,000 | `7c09b8b07f8c7b848b6b11047f53b8a0eb342b091870988aa3ba23c583a6028a` |

- 四个 source 的实际 resolved revision 均与 Gate 0 一致。
- 扫描完整 train split 时未发现缺失或重复 parent ID。
- 在独立临时输出目录重跑，四个 candidate JSONL hash 全部一致。
- FinQA validation 883 行、test 1,147 行已下载但未进入训练 candidate pool。
- 原始 snapshot 约 457 MiB，candidate work files 约 23 MiB，均位于被 Git 忽略的 `tmp/`。

---

## Step 3 — 应用 Day 08 Data Contract

**目的**

知道每条候选样本实际参与 loss 的 token 数，并拒绝无法形成完整监督目标的样本。

**动作**

```text
schema validate
-> template render
-> tokenize
-> assistant-only labels
-> right truncation at 2048
-> post-truncation validation
-> token counts
```

应该拒绝：

```text
illegal role order
empty assistant
template control-token injection
incomplete final assistant after truncation
zero supervised tokens after truncation
missing supervised assistant <|im_end|>
```

**产物**

每个 accepted canonical sample：

```text
sample_id
source / license / revision / parent_id
split / skill / subskill / language
content_hash
transform_chain
raw_token_count
input_token_count
supervised_token_count
```

每个 rejected sample：

```text
sample_id
rejected_stage
reason_code
source / parent_id
before state
```

**正确证据**

- accepted 样本 `supervised_token_count > 0`。
- accepted 样本的 final assistant `<|im_end|>` 受监督。
- user/system/header/padding labels 都是 `-100`。
- candidate count = accepted count + rejected count。
- 抽一条样本能复现 Day 08 的 token/label boundary trace。

**错误信号**

- 只因 `attention_mask=0` 就假设 padding 不参与 loss。
- 在 assistant 被截断后仍将答案前缀当作完整 target。
- 用 raw tokens 或 input tokens 代替 supervised tokens。
- accepted + rejected 不等于输入总数。

- [x] Step 3 data-contract 验收通过。

**Run v1 — max_length=256（诊断证据，已被 amendment v2 取代）**

| Slice | Candidates | Accepted | Rejected | Accept rate | Zero supervision | Truncated assistant |
|---|---:|---:|---:|---:|---:|---:|
| general | 2,000 | 725 | 1,275 | 36.25% | 3 | 1,272 |
| math | 2,000 | 0 | 2,000 | 0.00% | 1,488 | 512 |
| code | 2,000 | 359 | 1,641 | 17.95% | 905 | 736 |
| finance | 2,000 | 1,696 | 304 | 84.80% | 220 | 84 |
| total | 8,000 | 2,780 | 5,220 | 34.75% | 2,616 | 2,604 |

Accepted token totals：

| Slice | Raw tokens | Input tokens | Supervised tokens |
|---|---:|---:|---:|
| general | 96,499 | 111,720 | 63,805 |
| math | 0 | 0 | 0 |
| code | 71,764 | 79,297 | 18,044 |
| finance | 248,029 | 283,644 | 50,963 |
| total | 416,292 | 474,661 | 132,812 |

- `candidate_count == accepted_count + rejected_count` 对全部 source 成立。
- 所有 rejection 均发生在 encoded stage；没有 schema-stage rejection。
- 11 条 accepted 只截掉 final `<|im_end|>` 后的非监督尾部 token；final assistant boundary 仍完整且受监督。
- boundary trace SHA-256：`1035b94c92128567e1142d5791c1cd5d1f8614cbd413f41a4a190abeb29d9b94`。
- tokenizer、template、tokenizer-config 和 Transformers version 均与 Day 08 frozen contract 一致。
- 在独立临时输出目录以 `--local-files-only` 重跑，summary、boundary trace 和全部 accepted/rejected JSONL hash 一致。

**Run v1 触发的修订**

`math` 在 `max_length=256` 下没有任何 accepted sample，因此无法构建预注册的四 slice mixture。用户已在查看该结果后明确批准把整序列上限修订为 `1024`；该修改记录为 length amendment v2，重新执行 Step 3，不覆盖本段 v1 证据。

**Run v2 — max_length=1024（比较证据，已被 amendment v3 取代）**

| Slice | Candidates | Accepted | Rejected | Accept rate | Zero supervision | Truncated assistant |
|---|---:|---:|---:|---:|---:|---:|
| general | 2,000 | 1,943 | 57 | 97.15% | 0 | 57 |
| math | 2,000 | 381 | 1,619 | 19.05% | 0 | 1,619 |
| code | 2,000 | 1,996 | 4 | 99.80% | 0 | 4 |
| finance | 2,000 | 2,000 | 0 | 100.00% | 0 | 0 |
| total | 8,000 | 6,320 | 1,680 | 79.00% | 0 | 1,680 |

Accepted token totals：

| Slice | Raw tokens | Input tokens | Supervised tokens |
|---|---:|---:|---:|
| general | 661,710 | 702,513 | 550,196 |
| math | 332,410 | 340,409 | 244,467 |
| code | 734,563 | 776,479 | 262,020 |
| finance | 346,266 | 388,266 | 61,786 |
| total | 2,074,949 | 2,207,667 | 1,118,469 |

- 相比 v1，accepted 从 2,780 增加到 6,320，增加 3,540 条；所有 `zero_supervision` 均消失。
- 所有 1,680 条 rejection 都是 `encoded/truncated_assistant`，没有 schema-stage rejection。
- 两条 accepted math sample 只截掉 final `<|im_end|>` 后的非监督尾部 token；final assistant boundary 仍完整且受监督。
- boundary trace SHA-256：`98b6cd1adb5217e6e64629eac8205096b0b9bc0a139ab12a5772cad5ffd0f5ec`。
- preprocess summary SHA-256：`dbbd00e4e987fb5d97906706e15012874cb050e8005a3026c4704099435a86e3`。
- 独立临时输出目录以 `--local-files-only` 重跑，summary、boundary trace 和全部 accepted/rejected JSONL hash 一致。
- Math 已从空池恢复为 381 条，但 80.95% 候选仍因答案不完整而拒绝；后续构建 mixture 时需要监控 math occurrence 复用率和有效多样性。

**Run v2 触发的修订**

`max_length=1024` 已解决零监督问题，但 `math` 仍有 1,619/2,000（80.95%）候选因为 final assistant 不完整而被拒绝，而另外三个 slice 的接受率均不低于 97.15%。用户明确批准试验 `2048`；该修改记录为 length amendment v3，并保留 v1/v2 结果作为比较证据。

**Run v3 — max_length=2048（当前有效结果）**

| Slice | Candidates | Accepted | Rejected | Accept rate | Zero supervision | Truncated assistant |
|---|---:|---:|---:|---:|---:|---:|
| general | 2,000 | 1,986 | 14 | 99.30% | 0 | 14 |
| math | 2,000 | 1,878 | 122 | 93.90% | 0 | 122 |
| code | 2,000 | 2,000 | 0 | 100.00% | 0 | 0 |
| finance | 2,000 | 2,000 | 0 | 100.00% | 0 | 0 |
| total | 8,000 | 7,864 | 136 | 98.30% | 0 | 136 |

Accepted token totals：

| Slice | Raw tokens | Input tokens | Supervised tokens |
|---|---:|---:|---:|
| general | 717,795 | 759,501 | 602,927 |
| math | 2,383,964 | 2,423,402 | 1,845,315 |
| code | 739,180 | 781,180 | 264,296 |
| finance | 346,266 | 388,266 | 61,786 |
| total | 4,187,205 | 4,352,349 | 2,774,324 |

- 相比 v2，accepted 从 6,320 增加到 7,864，增加 1,544 条：general `+43`、math `+1,497`、code `+4`、finance `+0`。
- 所有 136 条 rejection 都是 `encoded/truncated_assistant`；没有 schema-stage rejection，也没有 `zero_supervision`。
- 逐条验证了 8,000 个唯一 `sample_id`、required fields、content hash、transform chain、token count bounds 与 summary totals；accepted 样本均有监督 token，且 boundary trace 的最后一个监督 token 是 `<|im_end|>`。
- 本轮 `accepted_truncated_count=0`。boundary trace SHA-256 仍为 `98b6cd1adb5217e6e64629eac8205096b0b9bc0a139ab12a5772cad5ffd0f5ec`，因为固定抽取的 trace 样本只有 339 tokens，在 1024 和 2048 两档下编码完全相同。
- preprocess summary SHA-256：`9ee3e5a6bd8851fc3357ddd812c71393505f47980a7cfba582cf0371bd3e44a6`。
- 在独立临时输出目录以 `--local-files-only` 重跑，目录内全部 10 个文件逐字节一致。
- 剩余 122 条 math 与 14 条 general 候选在 2048 上限内仍无法保留完整 final assistant，因此继续按 frozen contract 拒绝；不把答案前缀错误地当成完整 target。

---

## Step 4 — 生成质量与分布画像

**目的**

在删除数据前看到真实风险，并为过滤后的 before/after 对比建立 baseline。

**动作**

同时按以下维度统计 examples、raw tokens 和 supervised tokens：

```text
source
skill / subskill
language
length buckets
refusal / non-refusal
difficulty proxy
synthetic/template cluster
```

长度至少报告：

```text
min / p50 / p90 / p99 / max
truncation reject count
```

**产物**

`day09-data-quality-audit.md` 的 pre-filter 部分。

**正确证据**

- 每个表都清楚写明统计单位。
- example percentages 和 supervised-token percentages 分开。
- 各 source percentages 之和在浮点误差内等于 100%。
- 聚合数能回到构成它的 sample IDs。

**错误信号**

- 报告只有 example count。
- 把“长”或“被截断”直接当成低质量，没有记录发生在 prompt 还是 answer。
- 在看到分布后不更新配置版本就修改统计口径。

- [x] Step 4 pre-filter audit 通过。

**实际结果 — pre-filter audit v1**

审计 cohort 是 Step 3 的 7,864 条 accepted records；136 条 contract rejection 单独报告且不进入下列比例。Step 4 没有删除或修改任何样本。

| Metric | Result |
|---|---:|
| Accepted examples | 7,864 |
| Raw tokens | 4,187,205 |
| Input tokens | 4,352,349 |
| Supervised tokens | 2,774,324 |
| Contract rejected examples | 136 |
| Accepted truncated examples | 0 |

主要画像：

- Math 占 accepted examples 的 23.88%，但占 supervised tokens 的 66.51%；example share 不能代表训练信号权重。
- 三个 source README 明确声明为 synthetic 的 Tulu slices 合计占 74.57% examples、97.77% supervised tokens，是 Gate A 最需要核查的 composition risk。
- Input length 的 p50/p90/p99 为 346/1,338/1,873；supervised length 的 p50/p90/p99 为 125/1,030/1,554。
- 固定拒答短语规则命中 14 条（0.18%）；normalized prompt-prefix 规则将 389 条（4.95%）标记为 repeated-cluster candidates。两者都只是 review hints，不是自动删除理由。
- response-length difficulty proxy 中 high bucket 占 40.40% examples、88.23% supervised tokens；该 proxy 不代表语义难度或正确性。

产物与证据：

- artifacts/reports/day09-data-quality-audit.md
- tmp/day09-work/step4-audit/audit-summary.json，SHA-256 cddabdc420f4da2cb744814411b88bd1c5c1870d84435a944392270b5c27cf52
- tmp/day09-work/step4-audit/evidence-index.jsonl，SHA-256 f5536ac58093100db024a02193839903bf13d5a75505b2df12d2b41701d268bf
- 9 个 accepted-pool dimensions 的 example/raw/input/supervised 汇总全部闭合到总量；每个 dimension 的 percentages 在浮点误差内闭合到 100%。
- 所有 aggregate evidence keys 均能解析到完整、互斥且覆盖 cohort 的 sample IDs；sample-ID list hashes 复算一致。
- 独立临时目录重跑后 summary、evidence index 和 Markdown report 逐字节一致。

---

## Gate A — 每个 Source 人工审核 10 条

**目的**

发现 validator 和聚合数看不到的事实性、可用性、难度和合成模板风险。

**抽样**

```text
general = 10
math    = 10
code    = 10
finance = 10
total   = 40
```

审核包必须包含：

```text
sample_id
source / parent_id
prompt
answer
raw/input/supervised tokens
automatic risk hints
review_result
reason_code
review_note
```

`review_result` 三选一：

```text
accept
reject
uncertain
```

建议 reason codes：

```text
factual_error
incorrect_reasoning
incorrect_code
irrelevant_answer
empty_or_unusable
excessive_refusal
unsafe_content
too_easy
template_repetition
language_mismatch
truncated_supervision
suspected_synthetic_bias
```

**正确证据**

- 40 条全部有 result 和 note。
- 确认错误的具体 sample 可立即 reject。
- `3/10` 只记为 sample risk signal，不直接声称 source 全量错误率为 30%。
- source-level 决定要么有更大/分层抽样，要么明确标记 uncertainty。

**停止条件**

- 某 source 出现多个严重事实/代码错误。
- 大量 answer 高度模板化。
- 审核者无法判断 math/code 正确性，但又没有可执行 verifier。

出现停止条件时，暂停该 source 进入 mixture，先扩大或分层抽样。

- [x] Gate A 审核通过或有已记录的扩大抽样结论。

---

## Step 5 — 应用确定性质量过滤

**目的**

将明确错误和契约违规转换成可重放的 filter decisions。

**建议顺序**

```text
schema reject
-> zero/incomplete supervision reject
-> confirmed manual-quality reject
-> deterministic rule-based reject
```

每个 filter 记录：

```text
filter name/version/hash
before examples/tokens
after examples/tokens
removed sample IDs
reason codes
```

**正确证据**

```text
before_count - removed_unique_count = after_count
```

若同一 sample 可能命中多个 filter，需要记录首个终止 reason 和全部 matched reasons，不能把各 filter 命中数直接相加当作 unique removals。

**错误信号**

- 删除了 sample 但没有 sample ID。
- 只有聚合 reason count，无法复核具体记录。
- 手动改正 answer 却不在 transform chain 记录新 parent/transform。

- [x] Step 5 filter accounting 通过。

**实际结果 — machine-safe filter pass v1**

本轮已执行可证明的硬规则过滤并生成完整 decision ledger。用户于 2026-08-05 完成人工审核：40 条全部 accept，0 reject，0 uncertain；未触发扩大抽样停止条件。Gate A 与 Step 5 均为 complete，下游资格为 true。

| Stage | Before | Removed unique | After | Before supervised tokens | After supervised tokens |
|---|---:|---:|---:|---:|---:|
| Step 3 contract gate | 8,000 | 136 | 7,864 | n/a | 2,774,324 |
| Confirmed manual quality | 7,864 | 0 | 7,864 | 2,774,324 | 2,774,324 |
| Deterministic guardrails | 7,864 | 0 | 7,864 | 2,774,324 | 2,774,324 |

解释：

- 136 条 contract removal 全部是 Step 3 的 truncated_assistant，并以 sample ID 和 terminal reason 写入统一 decision ledger。
- 7,864 条 accepted records 没有命中 missing lineage、sample/parent mismatch、content-hash mismatch、空 prompt/answer、非正 supervision 或 input-length 越界，因此 machine-safe quality removal 为 0。
- Step 4 的 refusal、synthetic provenance、response length 和 repeated-prefix cluster 仍是 risk hints，不是删除条件。
- 0 quality removals 表示 40 条 Gate A 样本均被人工接受，且 7,864 条 accepted records 没有可证明的硬违规；这个结论只覆盖预注册的 Gate A 抽样和确定性 guardrails，不外推为全量逐条人工验证。

Gate A review package：

| Slice | Review rows |
|---|---:|
| general | 10 |
| math | 10 |
| code | 10 |
| finance | 10 |
| total | 40 |

风险分层抽样覆盖 1 条 refusal hint、5 条 repeated-prefix hints 和 16 条 high-response-length proxy samples。40 条 `review_result=accept`，均记录统一 review note、`reviewer=user` 和 `reviewed_at=2026-08-05`；accept 无拒绝原因，因此 `reason_code=null`。

产物与证据：

- artifacts/reports/day09-gate-a-review.jsonl，SHA-256 f12d1f1f1198c8491ed8d00e6e449e359803eb284ff16dd8b7d641ce0bb49ced
- tmp/day09-work/step5-filter/filter-decisions.jsonl，包含 8,000 个唯一 sample decisions，SHA-256 e4078038909a51baafa4f8e0c64daf0ca82de800326cc68146340530f72f5dd0
- tmp/day09-work/step5-filter/filter-summary.json，SHA-256 478aa893e3276bff7da02f33fd7162959a180b369f3f7c08ac7fc76f8a48378f
- 四个 filtered JSONL 的并集与 Step 3 accepted pool 的 7,864 条记录完全一致。
- before_count - removed_unique_count = after_count 成立，raw/input/supervised token totals 全部闭合。
- 独立临时目录重跑后 review package、summary、decision ledger、四个 filtered pools 和更新后的质量报告逐字节一致。

---

## Step 6 — Exact Dedup

**目的**

确定性删除 train 内部和跨 source 的完全重复内容。

**作用域**

```text
within source
cross training sources
```

**比较对象**

分开计算：

```text
normalized prompt hash
normalized answer hash
normalized complete-record hash
```

实际 exact dedup survivor 默认以 complete-record hash 分组；prompt-only/answer-only exact 匹配可先作为风险信号，不必然自动删除。

**Survivor rule**

必须在全量运行前固定，例如：

```text
manual accept first
-> preferred source priority
-> lowest canonical sample_id
```

**正确证据**

- 同一 complete-record duplicate group 最终只有一个 survivor。
- 每个 removed ID 都指向 survivor ID。
- normalization version/hash 已写入报告。
- exact dedup 后重新计算 example 和 token distributions。

**错误信号**

- 不记录 normalization 就声称 exact。
- 按 raw bytes 匹配，但报告声称已处理大小写/空白差异。
- survivor 由 dataloader 当前顺序决定。

- [x] Step 6 exact dedup 验收通过。

**实际结果 — exact normalization v1**

输入为 Step 5 的 7,864 条 downstream-eligible records。规范化固定为 Unicode NFKC、lowercase、连续空白折叠、首尾 trim、保留标点；complete record 使用 `[role, normalized content]` pair 的 canonical JSON 序列化。Survivor 顺序固定为：Gate A manual accept → finance/general/math/code slice priority → canonical sample_id 字典序。

| Match field | Duplicate groups | Affected examples | Duplicate excess | 删除策略 |
|---|---:|---:|---:|---|
| normalized prompt | 3 | 7 | 4 | evidence only |
| normalized answer | 36 | 83 | 47 | evidence only |
| normalized complete record | 3 | 7 | 4 | remove non-survivors |

三个 complete-record groups 均为 within-source：一个 general group 有 3 条相同记录，两个 FinQA groups 各有 2 条相同记录。最终删除 4 个 non-survivors：general 2 条、finance 2 条；code 和 math 无删除。未发现 cross-source exact complete-record group。

```text
examples:          7,864 - 4   = 7,860
raw tokens:      4,187,205 - 382 = 4,186,823
input tokens:    4,352,349 - 466 = 4,351,883
supervised tokens: 2,774,324 - 78 = 2,774,246
```

验证结果：每个 complete-record group 只剩一个 survivor；4 个 removed IDs 均回指仍在输出 pool 的 survivor；输出 pool 的 normalized complete-record hash 全部唯一；四种 accounting identities 全部通过。

产物与证据：

- tmp/day09-work/step6-exact-dedup/exact-match-groups.jsonl，包含 42 个分字段 exact-match groups，SHA-256 d27244c7c2657f63a4a060e8df2f278fd9f42b40f142c6460399dda71e7b8205
- tmp/day09-work/step6-exact-dedup/exact-dedup-decisions.jsonl，包含 7,864 个唯一 sample decisions，SHA-256 586356fc89426b81909e8d1449f8f5312a60b4681522718bd0b099fe23842bcb
- tmp/day09-work/step6-exact-dedup/exact-dedup-summary.json，SHA-256 a20c7828c6b1707d0baa91e0db4ee752b88f456766c50f26cac93bec9fc398a0
- normalization config SHA-256 d678e1747eff8ceb60d12ef0e4a1538831ff9795ac1fa66b1928e45639d9c800
- 独立临时目录重跑后 summary、match groups、decision ledger、四个 deduped pools 和 Markdown report 逐字节一致。

---

## Step 7 — Near-Duplicate Candidates

**目的**

发现改写、模板化和部分重叠，但不把 similarity threshold 误当成真值标签。

**动作**

对 prompt、answer 和 complete record 分开生成 candidates，记录：

```text
sample_id_A / sample_id_B
source_A / source_B
matched_field
matcher name/version/config
similarity score
threshold
candidate scope
```

候选数过多时，可以按下列顺序生成人工审核批次：

```text
cross-source first
-> higher similarity
-> higher sampling weight
-> longer supervised content
```

不能因候选数过多就不留配置记录地修改 threshold。

**正确证据**

- candidates 保留原文或可定位到原文的 IDs。
- 报告区分 candidate count 和 confirmed removal count。
- 任何删除都来自固定规则或 Gate B 的人工决定。

**错误信号**

- `score >= threshold` 后直接删除。
- 把共享通用短语当成整条样本重复。
- 不区分 prompt near-match 和 complete-record near-match。

- [x] Step 7 candidate generation 通过。

**实际结果 — RapidFuzz ratio candidates v1**

输入为 Step 6 的 7,860 条 exact-deduped records；训练 pool 在本步骤保持不变。为与本 Step 的验收正文一致，matcher config 明确覆盖 `normalized_prompt`、`normalized_answer` 和 `normalized_complete_record` 三个独立字段。固定 matcher 为 `rapidfuzz.fuzz.ratio==3.14.3`，threshold 为 90.0，且 pair 两侧的实际 normalized content 均至少 80 characters；complete-record 的 JSON/role 包装字符不计入长度门槛。

| Field | Eligible examples | Unordered pairs scored | Candidates | Affected samples | Within source | Cross source |
|---|---:|---:|---:|---:|---:|---:|
| normalized prompt | 7,858 | 30,870,153 | 378 | 437 | 378 | 0 |
| normalized answer | 5,952 | 17,710,176 | 424 | 224 | 424 | 0 |
| normalized complete record | 7,860 | 30,885,870 | 327 | 412 | 327 | 0 |

合计产生 1,129 条 field-specific candidate records，对应 812 个唯一 sample pairs、663 个唯一 samples。候选全部来自 source 内部：finance/finance 902 条，code/code 227 条；general 和 math 在当前门槛下没有候选。Similarity score 只产生 pending review candidate，confirmed removal count 为 0，Step 6 的 7,860 条训练 pool 未改变。

高分候选也不能直接视为重复。例如 `finance:IP/2014/page_66.pdf-1` 与 `finance:IP/2014/page_66.pdf-2` 的 complete-record score 为 99.431816，但前者询问 2014 并回答 59%，后者询问 2013 并回答 58%；这是共享 evidence/template 的合法相邻问题，而不是可自动删除的重复。

产物与证据：

- tmp/day09-work/step7-near-candidates/near-duplicate-candidates.jsonl，SHA-256 12ad9b0ddcfb1d66105e68da6ee669ca0f7e97336d9934c1d0a58eaf19434b14
- tmp/day09-work/step7-near-candidates/near-duplicate-summary.json，SHA-256 e7ffb19a1047a372a5dce2a5153d8221513bd5f37a181205391d38b2688a1a75
- artifacts/reports/day09-step7-near-review.jsonl：确定性分层抽取 20 个唯一 pairs（code 10、finance 10），覆盖 prompt/answer/complete-record 与不同 score bands；SHA-256 1ce12035e4c862b3655e2b35ba87ef9cabad2482ed9f8b4c9f6ab2c4649cd1f6。所有 review decisions 当前保持 pending，不改变训练 pool。
- artifacts/reports/day09-step7-near-review.summary.json，SHA-256 003ebaedf93afe449e03c46e0c5a84250ee72b97f89b51275fa45a85d49f8f06。
- 每条 candidate 均记录 pair IDs、sources/slices、matched field、matcher/version/config hash、score、threshold、scope 和 pending review fields。
- Candidate IDs 唯一、pair order 稳定、score/length 门槛全部复核通过；28 项 pipeline tests 通过。
- 独立临时目录重跑后 candidates、summary 和 Markdown report 逐字节一致。

---

## Step 8 — 准备 Day 10 Eval Candidate Pool

**目的**

在训练 manifest 冻结前得到可用于 contamination check 的 eval candidates。Day 09 只固定候选内容和 IDs；Day 10 再固定完整 decoder/scorer/baseline。

**动作**

为下列 slice 各准备 30–50 条 candidates：

```text
general
math
code
finance
```

每条至少包含：

```text
eval_sample_id
source / revision / split / parent_id
prompt
reference
skill
content_hash
```

评测 source 和 revision 需要在该 Step 开始前单独确认；不从 training sources 中临时挑一批有利样本当 eval。

**正确证据**

- eval IDs 和 content 已稳定。
- eval candidates 不被用于调整 training filters 以外的模型选择。
- Day 09 报告清楚标记它们是 candidates，不是 Day 10 frozen protocol。

**停止条件**

若 eval candidate sources/revisions 未确定，Day 09 可完成 Step 7，但不能越过 Step 8 声称 decontamination 完成。

- [x] Step 8 eval candidate pool 已固定。

**实际结果 — 四个独立 eval candidate pools v1**

用户在进入本 Step 时确认采用 MMLU、GSM8K、HumanEval 与 TAT-QA。每个 source 固定具体 revision，并为 general、math、code、finance 各稳定抽取 40 条，共 160 条。它们与 Day 09 training sources 独立。

| Slice | Source | Revision | Split | Candidates | 选择方式 |
|---|---|---|---|---:|---|
| general | `cais/mmlu` | `c30699e8356da336a370243923dbaf21066bb9fe` | test | 40 | global_facts、high_school_us_history、philosophy、professional_psychology 各 10 条 |
| math | `openai/grade-school-math` | `3101c7d5072418e28b9008a6636bde82a006892c` | test | 40 | 对固定 split 按 stable hash 排序 |
| code | `openai/human-eval` | `6d43fb980f9fee3c892a914eda09951f772ad10d` | test | 40 | 对固定 split 按 stable hash 排序 |
| finance | `NExTplusplus/TAT-QA` | `870accc41953dcde885aabeb963d94aabdc0fbc3` | dev | 40 | 对固定 split 按 stable hash 排序 |

MMLU 官方 `hendrycks/test` repository 固定为 `4450500f923c49f1fb1dd3d99108a0bd9717b660`；由于该仓库把数据放在一个外部 tar，实际 bytes 从固定 revision 的 `cais/mmlu` mirror 获取，并逐文件记录 SHA-256。GSM8K 缺少原生 row ID，因此 `parent_id` 由固定 test JSONL 的 zero-based row index 派生；MMLU 同样使用 subject/split/row index。HumanEval 与 TAT-QA 使用原生 `task_id` / question `uid`。

每条记录均包含要求的 `eval_sample_id/source/revision/split/parent_id/prompt/reference/skill/content_hash`，并额外记录 selection hash、adapter、license、source file 与 source-file hash。四个 output 均为 40 条，eval IDs、content hashes 唯一且可重算；training pool 未改变。

产物：

- `tmp/day09-work/step8-eval-candidates/general-eval-candidates.jsonl`，SHA-256 `fee5ff431b2d20528b6a64c15bb23fd57a82a60e55c493046ff3bff04865ff6c`
- `tmp/day09-work/step8-eval-candidates/math-eval-candidates.jsonl`，SHA-256 `6023cb474a1f29eade19bdb03d256cd327a4c9a0e6fc9d24953fc3cee558f1ff`
- `tmp/day09-work/step8-eval-candidates/code-eval-candidates.jsonl`，SHA-256 `51d7083a0789e52b75176df194884aa92b744361afbd8ae9d54f4ef77a06ae54`
- `tmp/day09-work/step8-eval-candidates/finance-eval-candidates.jsonl`，SHA-256 `572ebdc04c4dcacc6e54496f4707a07f8b0010cd53b70ace495d7356b9f6ddad`
- `tmp/day09-work/step8-eval-candidates/eval-candidates-summary.json`，SHA-256 `ec04d421d5a684f3001b56ade7e88419b27a0c8dd238c680209a07981f3bfc65`

Step 13 发现并修复了 `source_file` 绝对本机路径问题；当前记录使用仓库相对路径，因此从相对或绝对 `--source-root` 调用都会产生相同 candidate bytes。

相同 config/source bytes 下完整重跑后，20-pair package、四个 candidate JSONL、两个 summary 和 audit report 的 SHA-256 全部不变；28 项 Day 09 tests 通过。

边界：这里只冻结 candidate content 和 IDs，尚未完成 Step 9 decontamination，也未冻结 Day 10 decoder/scorer/baseline/protocol。

---

## Step 9 — Train/Eval Decontamination

**目的**

防止模型在训练中看到 eval 问题、答案或高度近似改写。

**必查边界**

```text
train prompt vs eval prompt: exact + near
train answer vs eval reference: exact + near
train complete record vs eval record: exact + near
```

不能只比较 complete-record hash。Train 可能包含长推理，eval 只包含短 reference，但它们的 prompt 仍可能 exact match。

**产物**

```text
train_sample_id
eval_sample_id
matched_field
match_type
similarity score
matcher config
decision
decision reason
```

**正确证据**

- exact 和 near 分开报告。
- prompt、answer 和 complete record 分开报告。
- candidate 不等于 confirmed contamination。
- 确认删除后重新计算 train manifest hash。

**错误信号**

- full-record overlap 为 0 就声称无污染。
- 把 public benchmark 原题增强后加入 train，仍用该 benchmark 声称泛化提升。
- 从 eval 中删除模型做错的题，却保留训练里的污染样本。

- [x] Step 9 overlap candidates 已生成。

**实际结果 — train/eval overlap candidates v1**

输入为 Step 6 的 7,860 条 train records 与 Step 8 的 160 条 eval candidates。三条边界分别执行 normalized exact equality 与 RapidFuzz ratio near matching；exact 对短文本不设长度门槛，near 使用预注册的 90.0 threshold，prompt/answer-reference 最短 40 characters，complete record 最短 80 characters。Exact pairs 不重复计入 near。

| Boundary | Exact pairs checked | Near pairs scored | Exact candidates | Near candidates | 最高 near-eligible score |
|---|---:|---:|---:|---:|---:|
| train prompt vs eval prompt | 1,257,600 | 1,257,600 | 0 | 0 | 52.386238 |
| train answer vs eval reference | 1,257,600 | 1,052,730 | 0 | 0 | 69.148933 |
| train complete record vs eval record | 1,257,600 | 1,257,600 | 0 | 0 | 55.135136 |

Answer/reference near scan 中 25 条短 eval references 与 62 条短 train answers 未进入 near ratio；它们仍全部进入 exact equality，因此短答案的精确泄露没有被长度门槛跳过。当前配置下总 candidate count 为 0，confirmed contamination 为 0，train/eval pools 均未改变。

该负结果只证明在固定 normalization、三条边界、ratio matcher 与 threshold 下没有命中；不能扩展为对任意语义改写都绝对无污染。为避免只报告“0”，summary 同时保留每条边界观察到的最高分及对应 train/eval IDs。

产物：

- `tmp/day09-work/step9-decontamination/train-eval-overlap-candidates.jsonl`：0 rows；空文件 SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `tmp/day09-work/step9-decontamination/decontamination-summary.json`，SHA-256 `31f8cfc55454380700457132320ace6c98195d0246f84469f7d0ec944bbcd80a`
- `find_day09_eval_overlap.py` 记录 matcher/config/input/output hashes、coverage、最高分、validation checks 与 pending decision schema。
- 31 项 Day 09 tests 通过；相同输入/config 完整重跑后 candidate ledger、summary 与 audit report hashes 全部不变；本步骤没有自动删除动作。

边界：Step 9 candidate generation 已完成；Gate B 仍单独保留，因为 candidate count 为 0 不应由生成脚本擅自写入人工审核结论。

---

## Gate B — 人工确认 Duplicate / Contamination Candidates

**人工决定选项**

```text
keep_both
remove_train_A
remove_train_B
remove_from_eval_candidate
not_duplicate
confirmed_contamination
uncertain
```

每个决定必须有 reason。

**判断问题**

```text
是实质内容重复，还是共享通用短语？
是整个解题逻辑重复，还是只有模板相似？
train 是否包含 eval 原题？
answer 是否泄露 eval reference？
这个 overlap 是否足以破坏 held-out 条件？
```

**正确证据**

- 所有高风险 train/eval candidates 已决定。
- `uncertain` 高风险项不进入最终 train manifest，或有明确保守策略。
- 决定人、时间、理由和 candidate score 全部保留。

- [x] Gate B 已完成。

**实际结果 — scoped Gate B ledger v1**

Step 7 的 812 个唯一 near pairs 全部写入独立 decision ledger。用户实际人工审阅的 20 个分层 pairs 记为 `keep_both`；其余 792 个 pairs 明确标记为 `candidate_is_non_terminal_no_confirmed_removal` policy basis，而不是伪装成人工逐条审核。Step 9 train/eval candidate count 为 0，因此没有 contamination removal item。

| Decision basis | Pairs | Decision | Confirmed removals |
|---|---:|---|---:|
| user manual stratified review | 20 | keep_both | 0 |
| candidate-is-not-removal policy | 792 | keep_both | 0 |
| train/eval contamination review | 0 | no candidates | 0 |

- `artifacts/reports/day09-gate-b-review.jsonl`：812 rows，SHA-256 `3335a8d85ecdf4fd8d53646c53743ef67f7c48daa472888be1c54f1dcab8e0f0`
- `artifacts/reports/day09-gate-b-review.summary.json`，SHA-256 `23b855e2e3ebe6a78b6d35b1ea569e958f3f880430399699447a09b17c4c63b9`
- 决策结果：812 keep_both、0 removal、0 uncertain；confirmed removed sample IDs 为空。

---

## Step 10 — 冻结 Clean Parent Pool

**目的**

将所有已确认决定应用到 canonical samples，得到 A/B 共用的唯一 parent pool。

**动作**

```text
apply quality decisions
-> apply exact dedup
-> apply confirmed near-duplicate decisions
-> apply confirmed contamination decisions
-> recompute distributions
-> freeze clean-pool hash
```

**产物**

```text
artifacts/data/day09-dataset-manifest.json
```

Manifest header 至少包含：

```text
manifest name/version/created_at
source revisions and licenses
selection config hash
preprocessing config hash
filter config hash
dedup config hash
decontamination report hash
example/raw/supervised-token totals
manifest hash
```

每条 canonical record 至少包含：

```text
sample_id
source / license / revision / parent_id
split / skill / subskill / language
content_hash
transform_chain
raw/input/supervised-token counts
```

**正确证据**

- 随机抽取 5 条，可沿 `sample_id -> source/revision/parent_id -> raw -> transform -> final hash` 重建。
- manifest 中无 zero-supervision records。
- confirmed removals 不在 clean pool。
- clean total = pre-filter accepted - unique confirmed removals。
- clean-pool hash 不依赖当前文件遍历顺序。

**错误信号**

- 只保存 `content_hash`，无法回到 parent record。
- 报告删除了样本，但 clean pool 仍包含该 ID。
- 应用删除后没有重算分布和 hash。

- [x] Step 10 clean parent pool 已冻结。

**实际结果 — clean parent pool v1**

Step 10 从 Step 6 canonical rows 开始，只应用 confirmed removals。Gate B 没有新增 near-duplicate 或 contamination removal，因此 clean pool 仍为 7,860 条。记录按 `sample_id` 升序冻结，pool hash 对原始文件遍历顺序不敏感。

| Slice | Examples | Raw tokens | Input tokens | Supervised tokens |
|---|---:|---:|---:|---:|
| general | 1,984 | 717,747 | 759,411 | 602,901 |
| math | 1,878 | 2,383,964 | 2,423,402 | 1,845,315 |
| code | 2,000 | 739,180 | 781,180 | 264,296 |
| finance | 1,998 | 345,932 | 387,890 | 61,734 |
| **Total** | **7,860** | **4,186,823** | **4,351,883** | **2,774,246** |

删除恒等式为 `7,864 Step 3/5 accepted - 4 exact complete-record duplicates - 0 confirmed near removals - 0 contamination removals = 7,860 clean records`。全部 records 具有正 supervised-token count，required lineage fields 齐全，content hashes 可重算，四个 Step 6 confirmed removed IDs 均不在 manifest。

冻结 hashes 与证据：

- clean-pool hash：`ab7dee175c140ef5fa5bf8e0f9197046d166f4e0eef98af5ec14ba543e2eb073`
- manifest hash：`95ce7aeb02efc4dbce50786c51c0e87db27dde4e7f9778e2a6927c5dd991c27f`
- `artifacts/data/day09-dataset-manifest.json`，file SHA-256 `97661d5482256dbaa3f33a96b32703fcdbc1f1aaa5084b64e4baba80b8f984bb`
- `artifacts/reports/day09-manifest-rebuild-evidence.jsonl`：5 条确定性 lineage rebuild records，SHA-256 `0b317b874111f387d010544f2d6f9685847f4027aabe9abdeea1cb9b6f7b922a`
- `tmp/day09-work/step10-clean-pool/clean-pool-summary.json`，SHA-256 `a152bbe9ddc97410faba75397c603cb014e4604c9390c87796373ce95efe04b3`
- Step 10 仅绑定 Step 0–10 dependency config；新增 Step 11/12 配置不会改变已冻结的 clean-pool manifest。

5 条 rebuild samples 均通过 `sample_id -> source/revision/parent_id -> candidate artifact -> transform_chain -> final content hash -> Step 6 artifact` 检查。Step 10 manifest 已标记 `downstream_eligible=true`，可供 Step 11 读取真实 supervised-token counts。

---

## Step 11 — 决定共同 Supervised-Token Budget

**目的**

找到 Mix A/B 都能实现的共同总训练信号量，不让总训练量成为混杂变量。

**动作**

1. 读取 clean pool 每条真实 supervised-token count。
2. 根据 A/B ratio 搜索共同可实现总量。
3. 允许有放回采样，但记录 `sampling_count` 或 occurrences。
4. 预先固定 slice ratio tolerance。
5. 不为凑 token 人为截断 answer 或伪造 supervised tokens。

**正确证据**

```text
actual_total_tokens_A == actual_total_tokens_B
```

每个 slice 的 actual ratio 在预注册 tolerance 内。

**错误信号**

- 只使用平均 tokens/example 估算最终比例。
- A/B examples 一样多就声称 token budget 一样。
- Mix B 使用了更多总 tokens。

- [x] Step 11 共同 budget 已固定。

**实际结果 — exact common supervised-token budget v1**

基于 Step 10 manifest 中每条样本的真实 `supervised_token_count`，在整样本、无放回条件下做整数 subset-sum 可达性搜索。共同预算取两个 mixture 都能实现的最大值：`246,936` supervised tokens，占 clean pool 训练信号的 `8.90%`。finance 只有 `61,734` supervised tokens，而两种 mixture 都要求 finance 占 25%，因此它构成严格上界：`61,734 / 25% = 246,936`。

| Mixture | General | Math | Code | Finance | Total |
|---|---:|---:|---:|---:|---:|
| Mix A balanced tokens | 61,734 | 61,734 | 61,734 | 61,734 | **246,936** |
| Mix A selected examples | 207 | 65 | 460 | 1,998 | **2,730** |
| Mix B targeted tokens | 30,867 | 30,867 | 123,468 | 61,734 | **246,936** |
| Mix B selected examples | 105 | 32 | 943 | 1,998 | **3,078** |

所有 slice 都精确命中目标：total-token difference 为 0、absolute ratio difference 为 0。没有截断 answer，没有伪造 token，也没有使用 replacement；每条被选样本的 `occurrence_count=1`。稳定顺序由 `sha256(version + NUL + mix + NUL + slice + NUL + sample_id)` 决定，随后确定性反向重建 subset，Step 12 可直接据此生成两个最终 manifests。

冻结证据：

- `tmp/day09-work/step11-token-budget/token-budget-plan.json`：包含预算上界、slice targets、完整 sample selections、occurrence counts 和验证断言。
- ratio tolerance：`0.0`；total-token tolerance：`0`。
- sampling mode：`without_replacement`；fallback replacement 未启用。
- plan hash：`45c0366af6e18f317da46b0af35ca50aa56b36d138fb25fdd1ce4f200c9ebbed`。
- plan file SHA-256：`d1e536d5f1f1703487b80d9ac281d94ce5dda98489ca012984fe1d376d84e873`。
- 38 项 Day 09 tests 通过；相同输入/config 重跑后 plan 与 audit report 均须 byte-for-byte 不变。

---

## Step 12 — 构建 Mix A / Mix B

**目的**

生成两个可用于 Day 12 单变量训练的 manifest。

**Mix A**

```text
general/math/code/finance = 25% / 25% / 25% / 25% supervised tokens
```

**Mix B**

```text
general/math/code/finance = 12.5% / 12.5% / 50% / 25% supervised tokens
```

**Occurrence 表示**

允许：

```json
{
  "canonical_sample_id": "code-17",
  "sampling_count": 3,
  "supervised_tokens_per_occurrence": 400
}
```

或使用三个不同 `occurrence_id` 指向同一 `canonical_sample_id`。

不允许：

```text
code-17-copy-1
code-17-copy-2
code-17-copy-3
```

这会伪造三个 canonical sources。

**A/B 共同不变量**

```text
parent-pool hash
source revisions
quality/filter hashes
dedup/decontamination hashes
tokenizer/template/max length/mask
selection/shuffle algorithms and seeds
actual total supervised tokens
future Base/model/train/eval protocol
```

**自动断言**

```text
A.total_supervised_tokens == B.total_supervised_tokens
A.parent_pool_hash == B.parent_pool_hash
A.preprocessing_hash == B.preprocessing_hash
A.filter_hash == B.filter_hash
A.decontamination_hash == B.decontamination_hash
A.target_ratios != B.target_ratios
A.manifest_hash != B.manifest_hash
```

**产物**

```text
artifacts/data/day09-mix-A-balanced.json
artifacts/data/day09-mix-B-targeted.json
```

**错误信号**

- A/B filter version、max length 或 seed 不同。
- 把 manifest hashes 不同当成错误。Mixture 不同时 child manifest hashes 应当不同。
- 不记录 sampling count，隐藏过采样。

- [x] Step 12 A/B manifests 已生成且自动断言通过。

**实际结果 — materialized mixture manifests v1**

Step 12 没有重新采样，而是严格消费 Step 11 plan 中已经冻结的 sample selections。每个 occurrence 都复制 clean parent record 的完整训练字段，并额外记录 `canonical_sample_id`、独立的 `occurrence_id`、`occurrence_index`、`sampling_count` 和 `supervised_tokens_per_occurrence`。原始 `sample_id` 保持不变，没有制造伪 canonical identities。

| Manifest | Unique canonical examples | Occurrences | Raw tokens | Input tokens | Supervised tokens | Max sampling count |
|---|---:|---:|---:|---:|---:|---:|
| Mix A balanced | 2,730 | 2,730 | 670,987 | 728,317 | **246,936** | 1 |
| Mix B targeted | 3,078 | 3,078 | 768,969 | 833,607 | **246,936** | 1 |

| Mixture | General | Math | Code | Finance | Total |
|---|---:|---:|---:|---:|---:|
| Mix A | 61,734 (25%) | 61,734 (25%) | 61,734 (25%) | 61,734 (25%) | **246,936** |
| Mix B | 30,867 (12.5%) | 30,867 (12.5%) | 123,468 (50%) | 61,734 (25%) | **246,936** |

10/10 自动断言通过：两者的 parent-pool、source revisions、preprocessing、quality filter、exact/near dedup、decontamination、token-budget plan、occurrence-order algorithm/seed 和实际 supervised-token 总量完全一致；target ratios 与 child manifest hashes 按预期不同。

冻结产物与 hashes：

- `artifacts/data/day09-mix-A-balanced.json`：manifest hash `19ea1a93e860fc5e93f463f7a136ae834ebe041c00f9fe3fa874f66ab1428c7a`；file SHA-256 `cb0d4819d383e4a22fd2c389c27c9cf058a20287c08e0fa5d924703f0af77492`。
- `artifacts/data/day09-mix-B-targeted.json`：manifest hash `8409de2d47c51cf0a29a22f5af0fd7bd136b98cfb1ed7de72682ffebb3bd5aaf`；file SHA-256 `d87e32b7925080c4e1abc6676354cb35e60af3f2311b969d6441a8aacc319837`。
- `tmp/day09-work/step12-mixtures/mixture-summary.json`：summary hash `108293e3555036d4c778b218daacad26c3d2c8ea2ae273468853797961a631cb`；file SHA-256 `ffc257264ef1440c24a9c55564c297a5564984eaab9e2b74f533d4ae7117bd10`。
- occurrence order 使用 seed `20260804` 和冻结的 SHA-256 sort；相同输入/config 必须 byte-for-byte 重建。
- 42 项 Day 09 tests 全部通过；二次完整 materialization 后 Mix A、Mix B、summary 和 audit report 的 file hashes 全部不变。

Gate C 被用户明确豁免而不是通过。两个 manifest 记录 `gate_c_status=waived_by_user`、0/60 reviewed 和 claim boundary；`downstream_training_eligible=true` 仅表示用户接受该残余风险，不构成人工 QA 证据。

---

## Gate C — A/B 各抽查 30 个 Occurrences

**目的**

确认机器统计与真实训练暴露一致，并发现被过采样放大的低质量样本。

抽查：

```text
Mix A = 30 occurrences
Mix B = 30 occurrences
```

不只抽 canonical samples；按实际 occurrence distribution 抽样。

审查：

```text
content quality
lineage
skill/language labels
supervised-token count
sampling count / occurrence reference
eval-overlap status
actual slice ratio
```

**正确证据**

- 每个 occurrence 都能追到 canonical sample 和 raw parent。
- 高 sampling-count 样本没有未被发现的严重质量问题。
- 实际抽样频率与 manifest 中的 occurrence weights 一致。

- [x] Gate C disposition 已记录：`waived_by_user`。计划 60 条、实际 0 条；未声称人工 QA 通过。

---

## Step 13 — 重建、验收与交付

**目的**

证明产物不是当前进程、缓存、文件顺序或人工操作的偶然结果。

**动作**

1. 保留已冻结正式 artifacts。
2. 在新的临时输出目录中，使用相同 config 重跑全部确定性步骤。
3. 比较：

```text
selected parent IDs
accepted/rejected IDs
exact duplicate groups
confirmed removal IDs
clean-pool hash
Mix A hash
Mix B hash
aggregate distributions
```

4. 运行全部 unit/integration tests。
5. 完成五个必交产物。

**正确证据**

- 两次运行的确定性 IDs 和 hashes 一致。
- 审核结论作为固定输入重放，不在重跑时丢失。
- 报告中的数量与 JSON/JSONL 产物一致。
- 没有 GPU 训练作为 Day 09 验收证据。

**一票否决**

- 重跑使用了不同 source revision。
- 重跑无法复现 sample selection。
- A/B 总 supervised tokens 不同。
- near/eval candidates 被静默删除。
- 报告数和实际 manifest 不一致。

- [x] Step 13 可复现验收通过。

**实际结果 — two-layer rebuild acceptance**

- 在全新临时目录从 pinned cached sources 重跑 Steps 2–11；Steps 2–9 的 30 个数据 JSONL/evidence files byte-for-byte 一致，clean records、clean-pool hash、共同预算、ratios 和完整 selection proofs 一致。
- 从冻结审核输入独立重放 Steps 10–12；clean parent manifest、token-budget plan、Mix A 和 Mix B 四个正式 artifacts 的 file SHA-256 全部精确一致。
- 44 项 unit/integration tests 通过；11/11 final acceptance checks 通过。
- 重建过程发现并修复两项路径依赖：Step 10 lineage evidence 的 hard-coded work-dir lookup，以及 Step 8 eval `source_file` 的绝对路径序列化。
- 验收证据：`artifacts/reports/day09-rebuild-acceptance.json`；acceptance hash `8d2dbd871e7902086a4fd1cd31084ab10b3506c6c06717652205cfb0fa9f0d0c`；file SHA-256 `fedf089c974672c767b4dde6af0ad635f7d53f5a6c78c7bf705c815ddcae6bcf`。

---

## 6. 必交产物验收表

| 产物 | 必须回答的问题 | 关键验收 |
|---|---|---|
| `day09-dataset-manifest.json` | clean pool 里是谁，从哪来？ | 任意 sample 能追到 raw parent |
| `day09-data-quality-audit.md` | 数据质量和实际 token 分布如何？ | examples/raw/supervised tokens 均报告 |
| `day09-decontamination-report.md` | 哪些是 exact、near 或 eval overlap？ | candidates、decisions、removals 分开 |
| `day09-mix-A-balanced.json` | balanced 时模型实际看到什么？ | 四个 slice 按 supervised tokens 平衡 |
| `day09-mix-B-targeted.json` | targeted 时哪个 slice 被提高？ | 与 A 同 token budget，只改变比例 |

---

## 7. 最短实际排查路径

不需要背全部 schema。出问题时沿下面的 key 追踪：

```text
manifest_hash
-> canonical sample_id
-> source + revision + parent_id
-> raw record
-> transform_chain
-> final content_hash
-> supervised_tokens x sampling_count
```

如果是 eval 异常：

```text
eval_sample_id
-> matched train sample_id
-> matched field
-> exact/near score
-> human decision
-> removal and rebuilt manifest hash
```

如果是 mixture 异常：

```text
manifest hash
-> slice actual supervised-token totals
-> high sampling-count samples
-> source/token distribution
-> preprocessing/filter hashes
```

---

## 8. Progress Log

| 节点 | 状态 | 证据/产物 | 阻塞 |
|---|---|---|---|
| Step 0 预注册 | `complete` | day09_assignment.json | |
| Step 1 代码骨架 | `complete` | prepare/audit/mixture scripts + tests | |
| Step 2 稳定选样 | `complete` | tmp/day09-work/*-candidates.jsonl | |
| Step 3 Day 08 contract | `complete` | tmp/day09-work/step3-max2048 | |
| Step 4 pre-filter audit | `complete` | artifacts/reports/day09-data-quality-audit.md + evidence index | |
| Gate A 质量审核 | `complete` | artifacts/reports/day09-gate-a-review.jsonl | 40 accept / 0 reject / 0 uncertain |
| Step 5 质量过滤 | `complete` | tmp/day09-work/step5-filter | downstream eligibility=true |
| Step 6 exact dedup | `complete` | tmp/day09-work/step6-exact-dedup | 7,864 - 4 = 7,860; downstream eligibility=true |
| Step 7 near candidates | `complete_candidates_only` | tmp/day09-work/step7-near-candidates + artifacts/reports/day09-step7-near-review.jsonl | 1,129 field candidates / 812 pairs; 20 manually keep_both; 0 confirmed removals |
| Step 8 eval candidates | `complete_candidates_only` | tmp/day09-work/step8-eval-candidates | 4 x 40 = 160 candidates; IDs/content fixed; Day 10 protocol not frozen |
| Step 9 decontamination | `complete_candidates_only` | tmp/day09-work/step9-decontamination | 0 exact / 0 near candidates; max scores 52.39 / 69.15 / 55.14; pools unchanged |
| Gate B 候选确认 | `complete_for_clean_pool` | artifacts/reports/day09-gate-b-review.jsonl | 20 manual keep_both + 792 non-terminal policy keep; 0 train/eval candidates; 0 removals |
| Step 10 clean pool | `complete_frozen` | artifacts/data/day09-dataset-manifest.json | 7,860 records; 2,774,246 supervised tokens; downstream eligible |
| Step 11 token budget | `complete_frozen` | tmp/day09-work/step11-token-budget/token-budget-plan.json | 246,936 supervised tokens each; exact ratios; no replacement |
| Step 12 A/B manifests | `complete_with_gate_c_waiver` | artifacts/data/day09-mix-A-balanced.json + day09-mix-B-targeted.json | 246,936 tokens each; 10/10 assertions; eligible under documented waiver |
| Gate C 最终 QA | `waived_by_user` | day09_assignment.json + manifests | 0/60 reviewed; not claimed as passed |
| Step 13 重建验收 | `complete` | artifacts/reports/day09-rebuild-acceptance.json | 11/11 checks; 44 tests; exact frozen artifact replay |

---

## 9. 完成时的 Daily Log

### 最大的数据质量风险

```text
1. Gate C 被豁免：Mix A/B 最终 occurrence-level 人工 QA 为 0/60，这是当前最大的残余质量风险。
2. Finance 只有 61,734 supervised tokens，限制共同预算为 246,936（clean pool 的 8.90%）。
3. A/B supervised-token 总量相同，但 input tokens 分别为 728,317 与 833,607；Day 12 应同时报告实际 compute/input-token 差异。
4. Near matching 只覆盖 frozen normalization + RapidFuzz ratio threshold，不证明不存在任意语义改写污染。
```

### A/B 实际 supervised-token 比例

```text
Mix A: general=61,734 (25%), math=61,734 (25%), code=61,734 (25%), finance=61,734 (25%), total=246,936
Mix B: general=30,867 (12.5%), math=30,867 (12.5%), code=123,468 (50%), finance=61,734 (25%), total=246,936
```

### 去重/去污染结果

```text
exact complete-record groups=3; exact removed records=4
near candidate records=1,129; unique near pairs=812
manually reviewed near pairs=20; confirmed near removals=0
train/eval exact candidates=0
train/eval near candidates=0
confirmed contamination removals=0
```

### Day 10 第一动作

```text
将通过 Day 09 overlap audit 的 eval candidates 冻结为完整 eval protocol，
并在训练前运行 Base checkpoint 的 per-sample baseline。
```
