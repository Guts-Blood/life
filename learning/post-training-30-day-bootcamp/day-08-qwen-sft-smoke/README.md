# Day 08 — Qwen3-4B LoRA SFT Smoke

日期：`2026-08-03`  
状态：`not_started`  
强度：4–5 小时

## 主要目标

用官方路径完成 `baseline -> LoRA SFT -> checkpoint -> inference -> compare`，同时验证 AutoDL 环境、缓存、日志和备份流程。

## 理论（60 分钟）

精读清单：[Day 08 — Qwen/ms-swift 最小 SFT](../SCALING-BOOK-READING-GUIDE.md#day-08)。今天暂停 Scaling Book 新章节，精读 ms-swift launch path、Qwen recipe 与 LoRA 概念。

- LoRA 的低秩更新、rank、alpha、target modules。
- Effective batch、warmup、save/eval cadence。
- 阅读 [Qwen3 ms-swift recipe](https://github.com/QwenLM/Qwen3/discussions/1301) 的 Qwen SFT 示例。

必须回答：LoRA 减少了哪些 trainable states？为什么 activation 不会按 trainable parameters 同比例下降？

## Coding（75 分钟）

- 将官方命令固化为 `artifacts/configs/day08-qwen4b-lora-sft.yaml` 或 launch script。
- 准备固定 20 prompts，写/完善 `run_generation.py`，输出逐样本 JSONL。
- 记录 code/model/dataset revision 和完整 generation config。

## 训练 / 实验（120–150 分钟）

- Model：`Qwen/Qwen3-4B-Instruct-2507`。
- Data：官方示例中的中英 Alpaca，各 500 条。
- 先生成 20 条 Base；再跑 50+ steps LoRA；最后加载 adapter 生成同样 20 条。
- 指标：loss、grad norm、LR、tokens/s、step time、peak memory。
- Stop：10 steps 内 NaN/labels 异常/OOM 时停止，先修 pipeline。

参考命令见 [ms-swift Quick Start](https://github.com/modelscope/ms-swift/blob/main/README.md)。参数以 Day 01 固定 commit 的 `--help` 为准。

## 资源与租卡

- 推荐：1×H100 80GB，预计开机 3–5 小时。
- 可替代：A100 40/80GB、L40S 48GB；本月性能对比尽量固定 H100。
- 完成后同步 config、logs、20×2 predictions 和 adapter metadata，再关机。

## 验收

- [ ] Checkpoint/adapter 可重新加载。
- [ ] Base/SFT 完全相同 prompt/template/decoding。
- [ ] 20 条输出标记改善/无变化/退化/格式变化。
- [ ] 写一页 `day08-smoke-report.md`。

## Daily Log

### Run ID / GPU hours

### 关键指标

### 20 条对比结论

### Day 09 第一动作
