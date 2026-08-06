# Day 25 — ms-swift 小模型 GRPO Lab

日期：`2026-08-20`
状态：`not_started`
强度：4–5 小时

## 主要目标

使用 ms-swift 完成一个可审计的小模型 GRPO 实验，把 Day 24 的 object/reward contract 映射到真实日志。TRL 只作为算法接口参照，不再运行第二套重复实验。

## 理论（60 分钟）

精读清单：[Day 25 — GRPO small-model lab](../SCALING-BOOK-READING-GUIDE.md#day-25)。

- 一个 prompt 的 `G` 个 completions 如何组成 group。
- Group-relative advantage、zero-variance group、policy/reference KL。
- rollout log-prob 与训练时 current log-prob 的角色。
- 采样温度、max completion length 和 reward variance 如何影响训练信号。

## Coding（75 分钟）

- 固定 ms-swift commit、model revision、dataset/reward version、seed 和完整 resolved config。
- 将 Day 24 verifier 接入 ms-swift 支持的 reward 扩展点；先跑 unit tests。
- 添加逐样本/逐 group 日志：prompt ID、completion、reward components、group mean/std、length、truncation、policy step。
- 写一个离线审计脚本，重算 reward 并和训练日志比较。

## 训练 / 实验（150 分钟）

- Model：优先使用 Day 12/23 已验证的 Qwen3-0.6B 阶梯 post-SFT checkpoint；如 pinned ms-swift 不支持，则使用该 commit 官方确认的最小 post-trained Qwen 起点并记录原因。
- Data：32–128 条可验证的短数学/格式任务；独立冻结 30–50 条 eval prompts。
- Run A：训练前生成 baseline，保存每条 completion 和 verifier evidence。
- Run B：每 prompt 产生多条 completions，先过 1 rollout/1 update gate，再运行 10–30 updates。
- 比较 reward components、zero-variance group ratio、KL、entropy、response length、truncation、grad norm、rollout/train time 和 frozen eval pass rate。
- 可选 Run C：只改变一个 reward component，最多 5–10 updates；不以训练 reward 单独宣称成功。

参数必须以 pinned ms-swift 的官方 GRPO 文档和 `--help` 为准，不复制未经版本核对的命令。TRL 仅用来核对 GRPO 术语/公式，不混用 Trainer 或 config。

## 资源与租卡

- 1×H100 80GB，预计 3–6 小时；小模型可根据 pinned recipe 使用等价显存卡。
- 1-update gate 不通过不扩大；reward 无方差、持续 NaN/OOM 或样本未落盘立即停止。
- 所有 completions、configs、metrics 和 checkpoint 同步后关机。

## Evidence-first 产物

- `../artifacts/configs/day25-ms-swift-grpo/`
- `../artifacts/logs/day25-trajectories.jsonl`
- `../artifacts/reports/day25-grpo-lab.md`

## 验收

- [ ] 真正完成 rollout→reward→optimizer update，不只是生成。
- [ ] 任一 reward 可由离线脚本从 raw trajectory 重算。
- [ ] 报告包含逐 group 方差、KL/entropy/length/truncation 与 frozen eval。
- [ ] 能解释 observed metrics 与 GRPO 公式中每个量的映射。
- [ ] 没有把 TRL 和 ms-swift 的参数名或实现细节混为一谈。

## Optional Capstone Handoff

复用 direct-RL reward adapter、逐 trajectory/group 日志、offline reward audit、one-update gate 和 dev selection 方法。Day 25 的 0.6B math/format checkpoint 不是正式 S2；Day 37 必须从 capstone 的 exact 4B S1、shared policy prompts 和冻结 student budget重跑 direct-RL control。

## Daily Log

### Pinned versions / resolved config

### 1-update gate

### Training reward vs frozen eval

### Day 26 第一动作
