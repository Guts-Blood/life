# Day 22 — Coding Preference Data：Provenance、Processor 与 Sandbox Evidence

日期：`2026-08-17`
执行日期：`2026-08-13`（提前完成 machine pipeline）
关闭日期：`2026-08-14`
状态：`closed_experimental — completed_experimental_ai_assisted；Day 23 实验路径已解锁，formal-human 路径仍未声明通过`
强度：4–5 小时

## 主要目标

为 Qwen3.5 v2 构建一份可审计的 coding preference dataset。重点不是先训练 DPO，而是确保 prompt/chosen/rejected、processor/template、代码测试、偏好理由、长度差与 train/dev/held-out 边界可信。

Day 01–12 的 raw bad cases 可以作为选题线索，但旧 rendered/tokenized records、0.6B generations 和 model-specific hashes 不进入 v2 pair manifest。

## 理论（60 分钟）

精读清单：[Day 22 — Preference data reliability](../SCALING-BOOK-READING-GUIDE.md#day-22)。

- Pairwise preference 表达相对偏好，不天然等于代码正确、安全或可执行。
- Unit-test/verifier、human review、LLM judge、best-of-N 和 synthetic repair pairs 的偏差不同。
- Length/style/position bias、test overfitting、invalid sandbox result、tie/ambiguous pair 与 prompt contamination。
- Random row split 为什么会让同一 problem/test/template/source family 跨 train/held-out。

## Schema 与 Coding（120 分钟）

Manifest 至少包含：

- `pair_id`、problem/prompt hash、chosen/rejected raw text；
- Qwen3.5 model key、processor/tokenizer/template revision/hash、rendered prompt hash；
- chosen/rejected response token spans、length、truncation/status；
- programming language、task/source/license、creation method；
- test/sandbox image or environment digest、test manifest/hash、timeout、stdout/stderr/exit code；
- annotator/judge/verifier name/version、rubric、timestamp、quality flags；
- prompt/problem/test/source-family split key。

实现 validator：空值、chosen=rejected、语义/近重复、template mismatch、response span 错位、异常长度差、不可复现测试、timeout/error 被当成 wrong answer、prompt/test/source-family 泄漏。

原计划引用的 Day 15 processor artifact 从未生成；本次改为冻结真实的 Day 22 Qwen3.5-4B-Base processor/template contract，并重新 tokenize chosen/rejected。断言 prompt 部分完全一致且只计算 response log-prob，禁止复用 v1 token IDs 或 assistant spans。

## 数据实验（105–120 分钟）

- 从来源清楚的 coding tasks、Day 15/16 v2 bad cases 或可授权数据中整理 200–500 pairs；规模不足时宁可少而可审计。
- 在固定 CPU sandbox 中重放 chosen/rejected tests，保存 raw evidence；测试不足以区分时标记 `ambiguous`，不伪造偏好。
- 人工盲审至少 50 pairs，允许 `tie/ambiguous/reject`，记录 rubric disagreement。
- 构造 length-matched、large-Δlength、source-held-out、test-family-held-out 和 timeout/error slices。
- 交换 chosen/rejected 展示顺序做 position-bias 检查；不修改语义来“平衡”数据。
- 按 problem/test/source family 切 train/dev/held-out，并在 Day 23 前冻结 IDs 与 checksums。

## 2026-08-13 已执行的 audited smoke

- 数据源仅使用 `google-research-datasets/mbpp` pinned revision `4bb6404fdc6cacfda99d4ac4205087b89d32030c` 的 `full/train` 与 `sanitized/train`，license 为 `CC-BY-4.0`；validation/test/prompt 文件未进入构建。
- Day 20 main 的 351 个 MBPP train task family 中排除 probe 使用过的 21 个，冻结 330 个 family；full/sanitized 按原生 task ID 合并，得到 107 sanitized + 223 full。
- 从 330 个 family 中固定选择 80 个 smoke task（60/10/10），以 canonical solution 为 chosen、单点确定性 AST bug 为 rejected；共保留 500 个可追溯 mutation candidate。
- E2B Firecracker 对 80 对数据执行 320 次 fresh、断网重放：63 对为 chosen `pass,pass` / rejected `wrong_answer,wrong_answer`；12 对 both-pass、5 对 runtime-error 均 quarantine，未强贴 preference 标签。
- Qwen3.5 pair processor audit 为 80/80 pass：prompt token prefix 相同、response-only labels 连续、causal shift 与四空格边界通过、无 truncation；processor contract hash 为 `12af5b3df5597fc2de4e6e9eebc9abba2e0f14d27971f39dc7e4af83c66e1083`。
- 63 个 audited pairs 按 family 落为 train/dev/heldout = 50/7/6；problem、prompt、test、source 四类跨 split overlap 都为 0。长度 token 差切片为 matched/mid/large = 54/6/3。
- 已生成 50 对、每对 A/B 与 B/A 两种展示的盲审 worksheet（100 行）及分离的 concealed key；当前人工完成数仍为 0。

这 63 对的角色是 `audited_smoke_not_formal_dpo_data`。Formal readiness 仍为 `BLOCKED`：数量 63 < 200、synthetic 占比 100% > 20%、尚无人工盲审结果、尚无 promoted-S1 on-policy pair。安全 sandbox gate 已通过 63/63。完整证据与命令见 [Day 22 audit report](../artifacts/reports/day22-qwen35-coding-preference-audit.md)。

## 2026-08-13 formal promoted-S1 multi-round bundle

- Day 21 downstream-ready S1 已完成 archive、merged export 与 `4/4` fresh-process exact token-ID parity；Day 22 每条候选均绑定同一个 promoted checkpoint、promotion manifest 与 downstream key。
- Base rollout 使用冻结的 `temperature=0.8`、`top_p=0.95`、每题 `K=6`，对 330 个 family 生成 1,980 条回答。格式过滤与 exact-response 去重后，1,638 条进入 E2B 双跑：746 个 stable pass、719 个 stable wrong-answer、173 个 quarantine，形成 141 对正式候选 pair。
- 其余 189 个 one-sided/runtime-only family 使用同一 S1 与相同行为采样参数补采样 `K=12`，两张 GPU 按 family 隔离生成 2,268 条回答。跨轮过滤 6 条格式无效、614 条轮内重复和 243 条首轮重复后，仅对 1,405 个真正新增回答执行 E2B 双跑。
- 合并后共有 3,043 个唯一候选：1,230 stable pass、1,430 stable wrong-answer、383 quarantine。最终 200 个 family 同时具备 pass 与 wrong-answer；70 个 only-pass、60 个 only-fail 继续 quarantine。没有用 runtime/timeout/infra error、跨题配对或 synthetic mutation 凑数。
- 最终 200 对的轮次组成：141 对全部来自 base round，11 对为 base chosen/adaptive rejected，6 对为 adaptive chosen/base rejected，42 对两边均来自 adaptive round。全部 pair 都是 promoted-S1 on-policy，synthetic 比例为 0。
- 最终 Qwen3.5 processor audit 为 200/200 pass，无 truncation；secure E2B 对每个入选 chosen/rejected 均保留两次 fresh、断网运行证据。
- 正式 split 为 train/dev/heldout = 154/17/29；problem、prompt、test、source 四类跨 split overlap 均为 0。
- 已生成 50 个 unique pair、每对 primary/swapped 两次展示的 100 行 blind-review worksheet，concealed key 单独保存。当前 `machine_ready=true`，唯一 formal blocker 为 `human_blind_review_pending`，所以 Day 23 仍不得启动 formal DPO。

正式 manifest SHA-256 为 `5b918b2a723fc3d44449193a5a72d5eb6a669a420243bbf08681d4b62fe0b26c`；完整机器 gate 与可重放命令见 [Day 22 audit report](../artifacts/reports/day22-qwen35-coding-preference-audit.md)。

## 2026-08-13 混合盲审校准（10 human + 90 sub-agent）

- 用户独立完成 worksheet 前 10 个展示；冻结快照已保存，非 answer 字段与空白 worksheet 完全相同。
- 剩余 90 个展示拆成三个 30-case packet，由三个 sub-agent 独立静态盲审；reviewer 只拿到 prompt 与 A/B，没有 concealed key、测试、执行 evidence 或 verifier 方向，也没有执行候选代码。
- 100 个展示中 89 个给出 A/B，11 个为 `ambiguous/tie`。A/B 与执行 verifier 方向一致 84/89（94.4%）：human 为 6/10，sub-agent 为 78/79。
- 50 个 pair 中，43 个两次展示都给出方向，其中 39 个语义一致；另有 7 个 pair 含非定向判断。40 个纯 sub-agent pair 中，35 个双向明确且 35/35 换位后一致，5 个含非定向判断。
- 该结果揭示了真实 prompt/test 口径歧义与“简洁代码掩盖逻辑错误”的风险，但它是 `audit_only_not_formal_human_review`：90 个 AI judgment 不能伪装成人工结论，formal human gate 仍保持 `BLOCKED`。

完整结果见 [混合盲审报告](../artifacts/reports/day22-qwen35-formal-s1-ai-assisted-review.md)。

## 2026-08-14 Codex 争议裁决与实验关闭

- 将混合盲审中 7 个含 `ambiguous/tie` 的 pair 与 4 个换位方向冲突的 pair 合并为 11 个 unique 争议 case。
- 一个全新 Codex sub-agent 只读取无标签 prompt、A/B 与公开测试，未接触 concealed key、verifier direction、执行 evidence 或正式 pair 文件；分别按语义正确性、公开测试对齐、格式遵循与鲁棒性打 10 分制分数。
- 11/11 个最终评分方向映射回 concealed key 后均支持 verifier-chosen；`keep=11`、`exclude=0`，没有 label flip。
- 实验性 DPO 输入仍为 200 对，split 保持 train/dev/heldout = 154/17/29；保留 pair 的文本、E2B evidence、processor audit 与 `pair_sha256` 均和 machine-verified source 完全相同。
- 因为本次没有剔除 pair，experimental manifest 直接复用 formal 200-pair 文件；不再物化一份字节完全相同的副本。若未来裁决产生 exclusion，finalizer 会强制写入独立输出并禁止覆盖 formal pairs。
- 独立 experimental validator 已通过，状态为 `valid_completed_experimental_ai_assisted`。原 formal manifest 保持 byte-identical pending 状态，`formal_human_review_ready=false`，未把 AI 判断伪装成人工。

实验关闭 manifest SHA-256：`3e812fa76a61cbeccf4f6371de56c75baa2913710ff9da05ec068d39e907f6ce`。完整裁决见 [experimental close report](../artifacts/reports/day22-qwen35-experimental-ai-assisted-close.md)。

## 资源与租卡

- S1 rollout 在 AutoDL 两张 RTX PRO 6000 Blackwell 上按 family 隔离并行；base 与 supplemental 均保留 shard receipt、日志与合并 manifest。补采样完成并校验下载后 GPU 已释放，后续步骤不再依赖租卡。
- Processor/tokenization、validator、split 与 artifact assembly 在本地 CPU 完成；候选代码只在 pinned E2B Firecracker sandbox 中执行。
- 今天不启动 DPO，不修改 Day 15/16 scorer 或 frozen eval。

## Evidence-first 产物

- `../artifacts/data/day22-qwen35-coding-preference-manifest.json`
- `../artifacts/data/day22-qwen35-coding-preference-pairs.jsonl`
- `../artifacts/data/day22-qwen35-coding-preference-split-ids.json`
- `../artifacts/data/day22-qwen35-coding-preference-blind-review.jsonl`
- `../artifacts/configs/day22-qwen35-processor-template-contract.json`
- `../artifacts/eval/day22-qwen35-mbpp-smoke-e2b-sandbox-evidence.jsonl`
- `../artifacts/eval/day22-qwen35-mbpp-smoke-processor-audit.jsonl`
- `../artifacts/scripts/validate_day22_coding_preference.py`
- `../artifacts/reports/day22-qwen35-coding-preference-audit.md`
- `../artifacts/data/day22-qwen35-formal-s1-manifest.json`
- `../artifacts/data/day22-qwen35-formal-s1-preference-pairs.jsonl`
- `../artifacts/data/day22-qwen35-formal-s1-split-ids.json`
- `../artifacts/data/day22-qwen35-formal-s1-blind-review.jsonl`
- `../artifacts/data/day22-qwen35-formal-s1-blind-review-key.json`
- `../artifacts/eval/day22-qwen35-formal-s1-multiround-candidate-e2b-evidence.jsonl`
- `../artifacts/eval/day22-qwen35-formal-s1-processor-audit.jsonl`
- `../artifacts/configs/day22-qwen35-formal-s1-processor-contract.json`
- `../artifacts/reports/day22-qwen35-formal-s1-assembly-audit.json`
- `../artifacts/data/day22-qwen35-formal-s1-human-10-review-snapshot.jsonl`
- `../artifacts/data/day22-qwen35-formal-s1-ai-assisted-review-annotations.jsonl`
- `../artifacts/reports/day22-qwen35-formal-s1-ai-assisted-review-audit.json`
- `../artifacts/reports/day22-qwen35-formal-s1-ai-assisted-review.md`
- `../artifacts/data/day22-qwen35-experimental-codex-adjudication-packet.jsonl`
- `../artifacts/data/day22-qwen35-experimental-codex-adjudication-scores.jsonl`
- `../artifacts/data/day22-qwen35-experimental-ai-assisted-split-ids.json`
- `../artifacts/data/day22-qwen35-experimental-ai-assisted-manifest.json`
- `../artifacts/reports/day22-qwen35-experimental-ai-assisted-close.md`
- `../artifacts/scripts/validate_day22_formal_s1_bundle.py`
- `../artifacts/scripts/validate_day22_experimental_close.py`
- `FORMAL-S1-BLIND-REVIEW-PROTOCOL.md`
- `finalize_day22_formal_s1_review.py`

## 验收

- [x] 每个 smoke pair 可追溯到 source、creation method、test evidence 和偏好依据。
- [x] Qwen3.5 processor/template 与 response masks 有逐 token audit，v1 tokens 未复用。
- [x] Ambiguous/error 没被强行变成 preference；50 对盲审 worksheet 已生成。
- [ ] 50 对人工盲审完成并裁决 position-swap disagreement（worksheet/key 已冻结；当前人工 10/100 个展示、0/50 个完整 pair；其余 90 个仅为 AI-assisted 校准）。
- [x] Length/source/test-family/timeout-error 有独立 slice；本数据只有单一 MBPP source，不能伪造 source-held-out。
- [x] Smoke held-out 按 problem/test/source family 隔离并冻结；它只是 preference-stage held-out / MBPP in-domain diagnostic，不是 model-level unseen。
- [x] 200 个正式 pair、synthetic = 0%、promoted-S1 on-policy provenance 与全部 machine gates 达标。
- [x] 11 个混合盲审争议 pair 已由独立 Codex 评分裁决；实验性 200-pair 输入与 close manifest 已冻结并通过独立 validator。

## Daily Log

### Provenance / processor coverage

330 个 clean MBPP train family 已冻结；历史 smoke 80/80、正式 bundle 200/200 processor pair audits pass。

### Sandbox replay / ambiguous pairs

历史 smoke 为 63 eligible。正式 multi-round 为 3,043 个唯一候选、200 eligible pairs；70 only-pass 与 60 only-fail family quarantine；无 runtime/timeout/infra error 入选。

### Length / source / test bias

历史 smoke length matched/mid/large = 54/6/3。正式 bundle 的 pair-level长度、source/test 与 execution slices 已写入 machine-gate audit 和 pair rows；单一 MBPP source 不伪造 source-held-out。

### Held-out leakage check

历史 smoke split 50/7/6；正式 split 154/17/29。两者均按 family 冻结，正式 problem/prompt/test/source 跨 split overlap 均为 0。

### Day 23 第一动作

Day 23 实验路径直接消费 `../artifacts/data/day22-qwen35-experimental-ai-assisted-manifest.json` 与其绑定的 200-pair 文件；先重跑 `validate_day22_experimental_close.py`，再做 DPO loss/mask/memory preflight。该路径已由用户明确授权，不再等待剩余人工盲审。

若未来要发布 formal human-reviewed 结论，仍需由人完成并裁决 50 对盲审；formal validator 的严格模式仍应返回退出码 3，唯一 blocker 为 `human_blind_review_pending`。这不再阻塞当前实验性 Day 23 主线。

人工阶段按 [Formal S1 blind-review protocol](FORMAL-S1-BLIND-REVIEW-PROTOCOL.md) 操作：复制空白 worksheet，独立填写 100 行展示，再运行 `finalize_day22_formal_s1_review.py`。Finalizer 不覆盖任何冻结文件，会通过 concealed key 映射 A/B、重算 position consistency，并只在全部 gates 通过时另发 ready manifest；独立 validator 会再次从原始 worksheet/key 重算，不信任可重封的汇总字段。

Reviewer 可直接使用[中文离线盲审页](../artifacts/data/day22-qwen35-formal-s1-blind-review-zh.html)；操作说明见[中文盲审指南](FORMAL-S1-BLIND-REVIEW-GUIDE-ZH.md)。页面只中文化说明、标签和选项，case 的题目与 A/B 原文保持冻结内容，导出的 completed JSONL 可直接交给 finalizer。
