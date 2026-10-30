# Day 33 — Deferred Teacher TP SFT One-step/Resume Gate

状态：`deferred_unselected`
日期：`unscheduled_after_day30`
强度：当前 0 GPU；仅独立 teacher-extension charter v2 激活后重新估算

## 当前执行状态

本页保留原 teacher TP SFT gate 作为未来模板，当前不得执行。活动 trainable policy 只有 `Qwen/Qwen3.5-4B-Base` 的 S0→S1→S2 lineage；不得从目录名中的 `8b` 推断或选择 8B、9B teacher。

激活本页必须同时具备：用户明确决定 exact teacher model、immutable revision、processor/loader 与 license 已冻结；charter v2 已创建；teacher/student compatibility、容量 smoke、GPU-hour cap 和停止条件通过。缺任一项即保持 deferred。

## 主要目标

若未来激活，在长 SFT 前证明已明确选择的 teacher `T0` 训练链路正确：真实数据、loss mask、optimizer update、distributed checkpoint 和新进程 resume 全部通过。Full/LoRA/QLoRA 必须在 charter v2 中另行决定。

## 理论 / 定向阅读（45–60 分钟）

精读清单：[Day 33 — 8B TP SFT Gate](../SCALING-BOOK-READING-GUIDE.md#day-33)。

## Gate 顺序

1. `G0 config/placement`：resolved config、rank groups、model/data hashes 与显存估算一致。
2. `G1 forward`：一条 Day 08 风格 token trace 与各 rank batch/label shape 正确。
3. `G2 one update`：loss、nonzero finite grad、clip、optimizer/scheduler step 正确。
4. `G3 tiny overfit`：固定 8–32 条样本有可解释的 loss/生成变化。
5. `G4 distributed save/reload`：退出全部进程，恢复 optimizer/RNG/data cursor 后继续至少 3 steps。
6. `G5 eval export`：导出 inference checkpoint，运行五条 capstone dev dry run。

## 记录

- raw/non-padding/label tokens/s；
- step time、MFU proxy、peak memory 与 allocator headroom；
- 每个 TP/DP rank 的 state ownership；
- checkpoint shards、optimizer/scheduler/RNG/sampler 和 conversion lineage；
- exactness level 与首个不可避免的 nondeterministic source。

## Evidence-first 产物

- `../artifacts/reports/capstone/day33-teacher-tp-sft-gates.md`
- `../artifacts/configs/capstone/day33-teacher-tp-sft/`
- `../artifacts/checkpoints/capstone/T0-gate-checkpoint-manifest.json`
- token trace、per-rank logs、resume comparison、five-sample predictions

## 验收

- [ ] 独立 teacher-extension charter v2 与用户 model decision 已存在；否则本页保持 deferred 且不产生 checkpoint。
- [ ] 激活后 G0–G5 全通过；任何 skipped gate 有明确阻塞证据。
- [ ] 训练修改的是预期 full parameters；若改用 PEFT，charter 已新版本化。
- [ ] distributed resume 恢复 model/optimizer/scheduler/RNG/data position。
- [ ] eval export 的 tokenizer/template/hash 与 Day 31 protocol 一致。
- [ ] 长 run 的 GPU-hour cap、checkpoint cadence 和停止条件已冻结。
