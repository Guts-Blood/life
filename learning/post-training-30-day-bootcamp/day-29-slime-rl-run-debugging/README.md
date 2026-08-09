# Day 29 — Verified slime × Qwen3.5 Coding RL 最小闭环

日期：`2026-08-24`  
状态：`not_started`  
强度：4–5 小时人工工作；GPU wall time 由 Day 25/26 的实测结果决定

## 主要目标

仅当 Day 26 输出 `go_day29` 时，使用其 runtime-verified slime release/tag/SHA、container、Qwen3.5 codepath 和 topology，对 Day 21 promoted SFT anchor 跑通最小 coding RL 闭环，并用 debug dump 做 train-only replay；随后只修改一个可单测的 reward component 重跑。

若 Day 26 输出 `slime_qwen35_compatibility_blocked`，今天的正确结果也是 `blocked`：整理最小 reproducer、失败 gate、官方 support boundary 和下一验证动作，不启动替代 recipe。禁止换 0.6B、Qwen3、其他模型或未验证 slime 版本完成一个无关闭环。

官方入口：[slime repository](https://github.com/THUDM/slime)、[slime releases](https://github.com/THUDM/slime/releases)。实际命令和参数只引用 Day 26 已冻结并实测的 checkout，不从 `main` 或历史 `v0.3.0` 阅读材料猜测。

## Hard Prerequisites

- Day 26 唯一结论为 `go_day29`，且 S0–S5 均有 runtime evidence。
- Policy parent 的 model/adapter/hash 与 Day 21 promoted Qwen3.5 SFT anchor 完全一致；Day 01–12 v1 checkpoints 只保留为历史索引。
- Day 24 coding trajectory/reward/sandbox contract 已在该 runtime 下 replay 通过。
- Day 25 实测的 max-length、显存余量和 learner/rollout placement 已冻结；不得静默扩大长度、group、batch 或并发。

任一 prerequisite 不成立就停止在 preflight，不把 `runtime_unverified` 写成成功。

## 理论与 Gate 预演（30–45 分钟）

精读清单：[Day 29 — slime runtime gates](../SCALING-BOOK-READING-GUIDE.md#day-29)。

- 从 Day 28 gate 表复述 Sample/DataSource/rollout/reward/train/weight version。
- 预写 zero variance、reward hacking、truncation、KL/entropy anomaly、stale weight 和 rollout bottleneck 的 signature。
- 冻结 baseline/modified reward、frozen coding prompts/tests 和每个 gate 的停止条件。
- 核对 resolved runtime report；任何与 Day 26 不同的依赖、model revision、processor/template 或 placement 都视为新兼容性验证，不能在今天临时放行。

## Runtime Contract（30–45 分钟）

- 使用 Day 26 保存的 exact release/tag/SHA、image digest、dependency lock、resolved config 与 verified launcher。
- 使用 Day 21 SFT anchor、Day 15 processor/template revisions、Day 24 sandbox/reward versions。
- 沿用 Day 25 实测的 prompt/completion cap 与 vLLM/SGLang context allocation；不得回到模型原生长上下文上限来配置 KV cache。
- 必须记录 rollout/group IDs、raw token IDs/text、status、reward components、loss mask、rollout/old/current/ref log-prob、policy/weight version、train step 和 sync duration。
- 使用框架已有 trace/debug 能力；若证据不足，只增加轻量、可移除且有单测的 observability，不修改算法语义。

## 训练 / 实验（180 分钟启动与分析）

严格按 gate 顺序：

1. **G0 config/import**：两条短 coding prompt 能加载 exact SFT policy、processor、reward/sandbox 和已验证 placement；保存 resolved config 与 hashes。
2. **G1 rollout-only**：生成最小 batch，保存 raw trajectory/debug dump；核对 token/template/status/truncation 和 rollout policy version。
3. **G2 reward replay**：从同一 raw dump 在隔离 sandbox 中离线重跑 tests/reward，逐 component 对齐并保留 failure semantics。
4. **G3 train-only replay**：从已保存 dump 构造 learner batch，完成一个 optimizer step；再次从同一初始 state replay，对比首步 loss/metrics 并说明 exactness 边界。
5. **G4 full loop**：完成 `rollout → sandbox reward → train update → weight sync → next rollout`，用 hash/version 证明下一批使用更新后的 policy。
6. **G5 controlled change**：只修改一个 reward component 或权重；unit tests 通过后最多完成 1–3 updates，比较逐样本行为、frozen coding correctness、length、KL/entropy 和 group variance。

每个 gate 失败就停在该层，保留 dump/log/minimal reproducer；不跳过 replay 直接扩规模，也不通过修改 model/runtime 来绕过失败。

## 资源与租卡

- 只使用 Day 26 runtime-verified、Day 25 memory-preflight 通过的最小 topology；若记录的是 1×H100 colocate 就使用该配置，若记录的是同机 2×H100 learner/rollout 分离就使用后者。
- 开卡前冻结 GPU-hour 上限、prompt/completion cap、group/batch、reference/KL 选择和停止条件。
- 不运行未验证的 8×H100 Stretch，不当天猜测缩容、offload、tensor parallel 或 adapter sync 参数。
- Reward 无有效 variance、持续 OOM/NaN、trajectory 无法落盘、sandbox evidence 不可重放或 weight version 无法验证时立即停止。

## Evidence-first 产物

- `../artifacts/logs/day29-slime-qwen35-trajectories/`
- `../artifacts/reports/day29-slime-qwen35-minimum-loop.md`
- exact runtime/model/processor/topology manifest
- baseline/modified reward diff 与 tests
- train-only replay comparison、weight-version timeline 与 next-version evidence
- blocked 时的 minimal reproducer、support boundary 与下一验证动作

## 验收

- [ ] Day 26 为 `go_day29`；否则本日明确停在 blocked evidence，没有 fallback run。
- [ ] G1 raw trajectory/reward 可离线重算，G3 从保存 dump 真正完成 optimizer step。
- [ ] G4 完成闭环且下一轮 rollout 有新 weight-version 证据。
- [ ] G5 只改变一个 reward component，并同时报告 frozen correctness、length、KL/entropy 和 group variance。
- [ ] Active lineage 始终为 Day 15 Qwen3.5 Base → Day 21 SFT anchor → Day 29 policy versions；v1 历史未混入。

## Daily Log

### Day 26 go/no-go / exact runtime

### Parent / processor / topology / length caps

### G0–G5 results

### Train-only replay exactness

### Reward 修改前后

### Weight-version timeline

### Day 30 第一动作
