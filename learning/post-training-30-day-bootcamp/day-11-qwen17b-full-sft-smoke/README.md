# Day 11 — Qwen3-1.7B 全参 SFT Smoke

日期：`2026-08-06`  
状态：`not_started`  
强度：4–5 小时

## 主要目标

跑通 Base model 全参 SFT，在扩大训练前确认 loss、显存、日志、checkpoint 和 inference 全部正确。

## 理论（60 分钟）

精读清单：[Day 11 — Full SFT 模型状态与预算](../SCALING-BOOK-READING-GUIDE.md#day-11)。启动前必须完成 weight/gradient/optimizer/activation 四项预测表。

- Full SFT vs LoRA 的参数更新、Adam states、显存和可塑性。
- BF16、AdamW、warmup、gradient clipping、effective batch。

## Coding（75 分钟）

- 固化 full-SFT config，所有重要默认参数显式化。
- 日志解析输出 loss、grad norm、LR、tokens/s、step time、peak allocated/reserved。
- 保存 run manifest：code/model/data/environment/seed。

## 训练 / 实验（150 分钟）

- Model：`Qwen/Qwen3-1.7B-Base`；Data：Day 09 通过校验的 SFT 数据。
- 先 10 steps；批准后跑 50 steps。
- 保存 step 25/50 checkpoint；对固定 prompts 和 5 条 train samples 生成。
- 与 Day 02/03 显存和 step-time 预测对比。

## 资源与租卡

- 推荐：1×H100 80GB，预计 3–5 小时。
- 可替代：1×A100 80GB；40/48GB 需重新估算，不作为默认。
- Stop：NaN、有效 labels 异常、显存超预算或 checkpoint 无法加载。

## 验收

- [ ] 10-step gate 通过才跑 50 steps。
- [ ] Checkpoint 能加载并 inference。
- [ ] 预估/实测差异有解释。
- [ ] 明确批准或否决 Day 12 主训练。

## Daily Log

### 预估 vs 实测

### Go/No-Go

### Day 12 第一动作
