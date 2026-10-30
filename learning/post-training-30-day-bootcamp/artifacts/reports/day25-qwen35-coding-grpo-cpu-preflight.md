# Day 25 — Qwen3.5-4B Coding GRPO 开卡前审计

日期：`2026-08-16`  
状态：`pass_cpu_ready_gpu_pending`  
GPU 消耗：`0 GPU-hour`

## 结论

Day 25 的 CPU、数据、reward 和配置准备已经闭环，可以进入单张 H100 80GB 的目标 runtime binding 与 G1 rollout-only。当前没有需要在本地继续修补的 blocker；剩余项目都必须由付费 GPU/runtime 实测，不能用静态检查代替。

本次没有运行模型训练、vLLM engine、目标机 E2B、optimizer update 或 weight sync，也没有声明 GRPO 能力提升。

## 已冻结的实验问题

从 Day 21 promoted S1 启动 fresh LoRA，用最小 coding GRPO 回答三件事：

1. 一个真实 `G=4` rollout batch 能否经过隔离 sandbox，产生可回放、可离线重算的 scalar reward 与 sample-std advantage？
2. 能否在固定数据/长度/topology 下完成恰好一个有限 optimizer update，并保存完整状态？
3. 更新后的 LoRA 能否同步到 colocate vLLM，使下一批 rollout 明确属于新 trainer step？

学习重点不是“跑出 reward 曲线”，而是建立 on-policy 训练的对象边界：rollout 属于哪个 policy version、模型错误与 infra 错误如何分开、reward 何时才有资格进入 trainer、optimizer 后 rollout weights 如何更新，以及训练 reward 为何不能替代 frozen eval。

## CPU gates

| Gate | 结果 | 证据 |
|---|---|---|
| Parent lineage | PASS | 仅 Day 21 `main-s20260809-lr1e-4-final` promoted merged S1；Day 23 checkpoint 未使用 |
| Train/eval | PASS | train32、eval40，family 完全不相交 |
| Processor/template | PASS | 72/72 rows；最大 prompt 88 tokens；零截断；non-thinking prefix mask 与首个四空格 token supervision 通过 |
| Reward boundary | PASS | tests-only；G=4；payload hash 先验；fresh E2B contract；infra retry once + atomic abort |
| G1 pre-optimizer stop | PASS（单测） | sealed ledger 后、reward scalar 返回前主动抛出 sentinel |
| Trainer semantics | PASS | 四份 JSON 真实解析；32-row loader/plugin 初始化；fresh LoRA、`beta=0`、无 ref/resume |
| Advantage parity | PASS | pinned ms-swift sample std (`ddof=1`) + `1e-4` 与 offline reducer 最大误差 `<=1e-12` |
| Regression | PASS | Day 22 sandbox 13、Day 24 reward 22、Day 25 adapter 7；live E2B test 1 条在常规 CPU run 中跳过 |

一键预检共 8 个 gates，content SHA-256 为 `8b38235160b9f9561ee2a6cd1e49e0e31acff124dea6c2d5afb7817a2e35ed5b`。

## 冻结配置

| 维度 | 值 |
|---|---|
| Parent | Day 21 promoted merged S1 |
| Adapter | fresh LoRA，rank 8，alpha 16，dropout 0；vision/aligner frozen |
| Objective | GRPO，G=4，tests-only reward，`beta=0`，无 reference |
| Batch | generation batch 8，per-device train batch 1，gradient accumulation 8，resolved steps/generation 8 |
| Length | prompt audit cap 512（observed max 88），completion 512，trainer max 1024，vLLM max model len 1152 |
| Rollout | vLLM colocate，TP=1，LoRA sync，memory utilization 0.30，sleep level 1 |
| Primary hardware | 1×H100 80GB；全程至少 15% free-memory margin |
| G1 | rollout + sandbox + sealed reward；有意在 scalar 返回 trainer 前停止 |
| G2 | fresh run，`max_steps=1` |
| G3 | fresh run，`max_steps=2`；核对 `trainer-step-0 → trainer-step-1` |
| G4 | 仅 G0–G3 全通过后，fresh run，最多 10 updates |

CPU contract content SHA-256：`59e4a07abbed6581650784b264fb43da81ce913c74c81f1be2b7880504f3ed2c`。GPU binder 已硬绑定该 trust root；contract 或被纳入 closure 的源码漂移都会 fail closed。

## 数据边界

- Train32 来自 Day 22 train split，按稳定 hash 选择，每个 family 只保留 prompt、code prefix、reward tests 与 sealed lineage。
- Eval40 为 20 个 Day 22 dev families + 20 个 heldout families，与 train32 disjoint。
- Eval40 对 Day 25 optimizer 独立，但不宣称在整个项目历史中从未被任何 prior audit 接触；这一限制会保留到最终报告。
- Reward 只读取 reward tests；eval tests 不进入训练 reward。

## 修复的关键接缝

Day 24 原 reducer 已改为 pinned ms-swift 的真实 group normalization：PyTorch sample std (`correction=1`) 与 denominator `std + 1e-4`。既有 16 条 E2B raw evidence 未重跑，schema 与 group rewards 从持久化 evidence 重建；因此修复没有制造新的 sandbox 观测。

实现保持为薄 adapter：Day 25 只负责 ms-swift batch/metadata 转换、并发 verifier 调用、证据持久化和 scalar 返回，不复制 Day 24 的测试执行、reward weights 或 reducer 逻辑。这样减少了跨层重复和两个 reward 实现静默分叉的风险。

## 仍需目标 GPU 消解的 blockers

这些不是 CPU 准备缺口，而是 Day 25 的真实 runtime gates：

1. 目标机必须重新核对 promoted S1 的全部文件大小/SHA-256、clean ms-swift commit 和 exact package set。
2. 本机没有创建 vLLM engine；Qwen3.5-4B + fresh LoRA 在目标 vLLM `>=0.17` 上的实际 load/sync 尚未证明。
3. 单张 H100 80GB 的峰值显存尚未测量；任一采样低于 15% free margin 即停止。
4. 目标训练进程到 E2B 的 live connectivity/latency 尚未验证。
5. G1、G2、G3 尚未执行；optimizer、checkpoint、logprob/ratio/entropy/grad norm 与 next-version rollout 都没有实证。
6. G4 与 frozen eval40 在 G0–G3 通过前保持 blocked；训练 reward 不足以晋级 checkpoint。

## Trust roots / artifacts

| Artifact | Content SHA-256 |
|---|---|
| Data manifest | `b52eaba89df057477c9582f28827b502446910cb990e756eb936f608e6952f99` |
| Processor audit | `ad751b7580b8f67c91dd64c081255ae4c3aa5ffedb254d7e99e6c164dacb2e3e` |
| CPU run contract | `59e4a07abbed6581650784b264fb43da81ce913c74c81f1be2b7880504f3ed2c` |
| Argument audit | `fc597ba71cf59669ad5eb7db542f6ebd1a6895d2f5e26d34fcce5d79838c5c5b` |
| CPU preflight | `8b38235160b9f9561ee2a6cd1e49e0e31acff124dea6c2d5afb7817a2e35ed5b` |

执行入口：

- `day-25-grpo-small-model-lab/run_day25_cpu_preflight.py`
- `day-25-grpo-small-model-lab/bind_day25_qwen35_gpu.py`
- `day-25-grpo-small-model-lab/DAY25-GPU-RUNBOOK.md`

下一步不是直接跑 10 steps，而是按 runbook 在目标卡依次完成 binding → G1 → offline reward audit → G2 → G3；每个 gate 单独授权下一步。
