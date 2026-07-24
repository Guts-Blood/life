# Day 12 — Qwen3-1.7B 主 SFT

日期：`2026-08-07`  
状态：`not_started`  
强度：4–5 小时人工工作；训练可夜间继续

## 主要目标

完成足以产生可评测行为变化的全参 SFT，并留下 early/middle/final checkpoint 与完整 run manifest。

## 理论（45 分钟）

精读清单：[Day 12 — Token budget、训练时间与 checkpoint](../SCALING-BOOK-READING-GUIDE.md#day-12)。完成 `steps -> global tokens -> FLOPs -> H100 hours` 换算并预注册 checkpoint 选择规则。

- Steps/epochs/tokens/effective batch 换算。
- LR schedule、warmup、overfitting、held-out loss。
- Checkpoint selection 不能只看 final train loss。

## Coding（75 分钟）

- 增加 run manifest 自动保存 config、commit、dataset hash 和环境。
- 输出曲线数据为 CSV/JSON。
- 准备 checkpoint sweep：小 frozen eval 对 early/middle/final 统一生成。

## 训练 / 实验（150 分钟启动与监控）

- 使用 Day 11 通过的配置，只扩大 steps/token budget。
- 建议 200–500 steps；保存 early/middle/final。
- 观察 train/eval loss、grad norm、LR、tokens/s、memory 和 ETA。
- 夜间运行前必须验证日志落盘和 checkpoint resume。

## 资源与租卡

- 推荐：1×H100 80GB，预计 6–12 wall-clock 小时。
- 先按量做 20-step recheck，再根据 ETA 决定继续按量或按日。
- Stop：eval loss 持续恶化、NaN、数据吞吐中断、磁盘进入危险余量。
- 训练完成同步 manifest/curves/selected-checkpoint metadata，再关机。

## 验收

- [ ] 至少一个 checkpoint 在预定义小 eval 上优于 Base。
- [ ] 数据 tokens、训练时间、GPU hours、最佳 checkpoint 可查。
- [ ] `day12-main-sft.md` 写明主要 bad cases 和下一步 ablation。

## Daily Log

### Token budget / GPU hours

### Selected checkpoint

### Day 13 Reading 问题
