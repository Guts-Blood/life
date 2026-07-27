# Day 17 — Exact Checkpoint Resume 与可复现性

日期：`2026-08-12`
状态：`not_started`
强度：4–5 小时

## 主要目标

证明 checkpoint 不只是“能加载权重”，而是能恢复 model、optimizer、scheduler、RNG 和 dataloader cursor，使中断续训与不中断训练走过相同的数据和更新。

## 理论（60 分钟）

精读清单：[Day 17 — Checkpoint/resume state](../SCALING-BOOK-READING-GUIDE.md#day-17)。

- Model/gradient、Adam moments、optimizer step、scheduler step 分别决定什么。
- Python、NumPy、Torch CPU/CUDA RNG 与 dropout/data shuffle 的关系。
- Sampler epoch、dataset cursor、worker seed、gradient accumulation 边界。
- “加载成功”“数值接近”“逐 step 一致”“bitwise exact”是四个不同等级。

先列 state inventory，并写明缺少每一项会从哪个 step 开始分叉。

## Coding（90 分钟）

- 实现 checkpoint audit：列出并 hash model、optimizer、scheduler、scaler、RNG、sampler/dataloader state。
- 为每个 step 保存 sample IDs、LR、loss、grad norm 和累计 label-token 数。
- 实现两条 run 的逐 step 对齐比较，以及若干关键 tensor 的 checksum/`max_abs_diff`。
- 写一个真正退出进程再恢复的脚本；禁止在同一 Python 进程中伪装 resume。

## 训练 / 实验（120–150 分钟）

使用 Day 16 稳定配置和固定数据：

- Run A：从 step 0 连续训练到 step 40。
- Run B1：从相同初始权重训练到 step 20，保存 checkpoint，正常退出。
- Run B2：新进程从该 checkpoint 恢复到 step 40。
- 比较 step 21–40 的 sample IDs、LR、loss、grad norm、最终权重与 optimizer state。
- Failure probe：复制 checkpoint 后有意遗漏 dataloader cursor 或 scheduler state，只跑 3–5 steps，证明审计能定位分叉；不要覆盖正确 checkpoint。

若当前 kernel/framework 无法 bitwise deterministic，必须先固定硬件和 deterministic 设置，再报告首个分叉 step、误差量级与具体 non-deterministic source，不能把“曲线差不多”叫 exact resume。

## 资源与租卡

- 1×H100 80GB，预计 3–5 小时；使用小模型/短序列即可。
- 所有 run 固定镜像、commit、GPU 型号和数据 worker 数。
- 正确 checkpoint 与故障注入副本分目录保存；验证完整后关机。

## Evidence-first 产物

- `../artifacts/scripts/checkpoint_audit.py`
- `../artifacts/reports/day17-resume-equivalence.md`
- Run A/B 的 configs、sample-ID timeline、state manifest 与 tensor diff

## 验收

- [ ] 正确恢复包含 model、optimizer、scheduler、RNG 和 dataloader cursor。
- [ ] 确实退出并启动新进程。
- [ ] 给出 exactness 等级和逐 step 证据，不只给最终 loss。
- [ ] 故障注入能在预期位置造成并定位分叉。

## Daily Log

### State inventory

### Run A vs Run B

### 首个分叉与原因

### Day 18 第一动作
