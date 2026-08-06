# Day 33 — 8B TP SFT One-step/Resume Gate

状态：`not_started`
日期：`unscheduled_after_day30`
强度：4–5 小时；2–4×80GB GPU 短 smoke

## 主要目标

在长 SFT 前证明 8B `T0` 的 TP full-parameter training 链路正确：真实数据、loss mask、optimizer update、distributed checkpoint 和新进程 resume 全部通过。

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

- `../artifacts/reports/capstone/day33-8b-tp-sft-gates.md`
- `../artifacts/configs/capstone/day33-8b-tp-sft/`
- `../artifacts/checkpoints/capstone/T0-gate-checkpoint-manifest.json`
- token trace、per-rank logs、resume comparison、five-sample predictions

## 验收

- [ ] G0–G5 全通过；任何 skipped gate 有明确阻塞证据。
- [ ] 训练修改的是预期 full parameters；若改用 PEFT，charter 已新版本化。
- [ ] distributed resume 恢复 model/optimizer/scheduler/RNG/data position。
- [ ] eval export 的 tokenizer/template/hash 与 Day 31 protocol 一致。
- [ ] 长 run 的 GPU-hour cap、checkpoint cadence 和停止条件已冻结。
