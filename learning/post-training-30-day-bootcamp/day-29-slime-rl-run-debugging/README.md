# Day 29 — slime 多卡 RL 运行、插桩与 Reward 修改

日期：`2026-08-24`  
状态：`not_started`  
强度：4–5 小时人工工作；训练可继续 6–10 小时

## 主要目标

用官方 slime/SGLang/Megatron 路径在真实多卡上跑通 RL 闭环，把 Day 26/28 的静态调用链与 runtime evidence 对齐；修改 reward 扩展点后重跑，证明不只是复制命令。

## 理论（45 分钟）

精读清单：[Day 29 — slime runtime 与 failure signatures](../SCALING-BOOK-READING-GUIDE.md#day-29)。开卡前重读剩余 `STATIC_ONLY` 的 caller 和预期日志。

- On-policy、reference/KL、group-normalized advantage。
- `num_generations`、completion-level batch、zero-variance groups。
- Reward hacking、entropy/KL、length drift、rollout bottleneck。

## Coding / Runtime 插桩（60 分钟）

- 验证 Day 26 的 accuracy/format reward unit tests。
- 打开或增加轻量日志：Ray actor/GPU mapping、rollout ID、Sample count/schema、reward components、train step、weight version/sync duration。
- 所有本地 slime 变更保存为最小 diff；先用原始 reward 跑，再替换为修改版。

## 训练 / 实验（180 分钟启动/分析）

- Model：官方 recipe 对应的 `Qwen/Qwen3-4B`；Data：小型可验证数学任务。
- Run A：原始 accuracy+format reward，先 1 rollout/update gate，再完成 5–10 updates。
- 沿日志逐边验证：SGLang generation → reward → Sample/buffer → Megatron loss/update → weight sync → 新 weight version rollout。
- Run B：修改 reward 权重或增加一个可单测的 reward component，完成 2–5 updates。
- 比较 reward components、group variance、KL、length、rollout/train/sync time 和逐样本行为。
- 若时间允许，恒定 reward/zero-variance 只做 1-step failure injection；不追求收敛。

参数以固定 slime commit 的 [官方 Quick Start](https://thudm.github.io/slime/get_started/quick_start.html) 为准。

## 资源与租卡

- Core：同机 8×H100 80GB，预计 4–8 wall-clock 小时，优先贴近官方 Qwen3-4B recipe 以减少适配时间。
- 预算受限可缩为 4×H100，但必须在 Day 26 预先完成显存/placement 改造；不要当天临时缩容。
- Stop：reward 无 variance、持续 OOM/NaN、rollout 输出不可解析、metrics 未落盘。
- Run A/Run B、logs、diff、checkpoint/weight version、逐样本结果同步后关机。

## 验收

- [ ] Online loop 真正完成 rollout→reward→Megatron update→weight sync→next rollout。
- [ ] Day 26/28 静态图至少六条边获得 runtime log/trace 证据。
- [ ] 原始与修改版 reward 都完成 update，diff、unit tests 和逐样本行为可审计。
- [ ] Reward 与 frozen accuracy/length/KL/group variance 一起报告，不单独宣布成功。
- [ ] `slime-runtime-report.md` 包含 resource map、call chain、weight-version timeline 和至少五种 failure signature。

## Daily Log

### slime commits / Docker / GPU topology / hours

### Static call chain vs runtime evidence

### Reward 修改前后

### Day 30 第一动作
