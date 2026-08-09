# AutoDL 与 GPU 资源计划

更新日期：2026-08-07。卡价、库存和实例拓扑以开机时 AutoDL 页面为准；本计划给出资源上限和 readiness gate，不写死单价。

## 原则

1. 数据 schema、清洗、mixture、manifest、静态 codepath、quiz 和报告均不开 GPU。
2. 开机前必须有 pinned code/model/data、可执行命令、`max_steps`、成功条件、停止条件和输出路径。
3. 先用 tiny overfit/5–20 step smoke 验证数据与 loss，再进行受控 SFT、DPO 或 GRPO。
4. 每个实验只改变一个预注册变量；不为“卡还开着”临时扩大训练。
5. checkpoint 必须先通过 save/load；长任务再验证中断恢复。
6. 8×H100 slime 是 Stretch，不是毕业门槛。Core 先完成 ms-swift 主线；slime 只有在 Day 26 证明固定 release 完整支持 Qwen3.5 后才分配 GPU。
7. Day 31–42 Capstone 与 30-Day Core 分账；每个 block 在前一级 smoke 后单独冻结 GPU-hour cap。Teacher/OPD 当前 deferred，GPU 预算为 0。
8. Day 1–12 的 `Qwen3-0.6B-Base`、Transformers 4.57.3 与已有 checkpoint 是历史复现链；Qwen3.5-4B v2 使用独立环境、模型合同、baseline 和预算，不原地覆盖。

AutoDL 按实例开关机时间计费，不按 GPU kernel 活跃时间计费。关机后 GPU 不再预留，实例数据也有保留边界；使用前复核官方[计费规则](https://www.autodl.com/docs/price/)、[省钱说明](https://www.autodl.com/docs/save_money/)与[实例数据规则](https://www.autodl.com/docs/instance_data/)。

## 推荐资源

| 场景 | 默认资源 | 说明 |
|---|---|---|
| 数据审计、manifest、quiz、配置 review | CPU/无卡模式 | 不能用 GPU 掩盖数据准备不足 |
| 0.6B tiny overfit、SFT、eval（Day 01–12 历史） | 1×RTX 4090 24GB | 只用于复现 v1，不外推到 Qwen3.5 |
| Qwen3.5 onboarding / coding LoRA SFT / resume | 默认 1×H100 80GB；48GB 仅在 peak smoke 后可用 | text-only、vision/aligner frozen；保留 10%–15% 余量 |
| Qwen3.5 Megatron minimum codepath | 2×H100 同机 | 只验证 GDN/model mapping、process group、state ownership、checkpoint/export |
| Qwen3.5 DPO | 1×H100 80GB 起步 | 必须从 promoted S1；冻结 policy/reference placement |
| Qwen3.5 coding GRPO | 1×H100 80GB colocated smoke；稳定运行优先 2×48/80GB 分离 | 先验证 reward、mask、G、8K cap、KV peak 与 policy version |
| slime 在线 RL Core | Day 26 验证的 Qwen3.5 最小支持拓扑，预算上限 4×H100 同机 | 未验证则不租卡，继续使用 ms-swift 主线 |
| slime 官方 8×H100 recipe | 8×H100 同机 | Stretch；Core 全通过且单独批准预算后才开 |
| 30B+ full post-training | 本月不租 | 只保留配置判断题，不作为最终项目 |
| Qwen3.5 single/TP2 parity | 同机 2×H100 80GB | 单卡 reference 与 TP2 使用相同 processor/batch/global label tokens；只做 correctness run |
| Qwen3.5 direct coding RL Capstone | 1×80GB colocated smoke；稳定运行优先 2×48/80GB | S1→S2；根据 Day 32/37 实测冻结 full/LoRA/QLoRA 与 topology |
| Deferred teacher / OPD | CPU 模板，0 GPU | `teacher_model_id=null`；只有用户另批 charter v2 后重新做资源计划 |

选择多卡实例时记录 `nvidia-smi topo -m`、GPU 型号/显存、CPU 内存、数据盘、driver/CUDA/NCCL、同机与否。卡名相同不代表互联相同。

### Qwen3.5-4B v2 planning envelope

Qwen3.5-4B-Base 是带 vision encoder 的统一多模态 checkpoint，即使首轮课程只训练文本 coding 数据，也不能把它当作旧 `AutoModelForCausalLM` 的直接替代。v2 默认加载完整 checkpoint，使用 `AutoProcessor` 与 `AutoModelForMultimodalLM`/`Qwen3_5ForConditionalGeneration`，并在任何 optimizer 创建前断言 vision tower 与 aligner 已冻结、其梯度和 LoRA target 均为空。若以后提取纯文本 backbone，必须建立单独的转换、权重 lineage 和输出 parity 合同。

下表全部是 **planning envelope，不是容量承诺或官方 NVIDIA 峰值**。每个新 topology 都要先跑完整 `load -> forward -> backward -> optimizer step -> save/reload` peak smoke，记录 allocated/reserved peak，并在最长已批准序列下保留至少 10%–15% 显存余量；否则降序列、改分片或停止。

| 资源层级 | v2 允许范围 | 进入条件与边界 |
|---|---|---|
| 1×24GB | QLoRA tiny smoke | 只验证少量样本、短序列、mask、梯度、save/reload；不批准 BF16 LoRA 长跑或 colocated rollout |
| 1×48GB | BF16 LoRA learner | text-only、vision/aligner frozen；先从短序列与 batch 1 实测，不能因模型名含 “4B” 直接批准 8K+ |
| 1×80GB | BF16 LoRA 或短 colocated smoke | 可试 learner 与 rollout colocate，但必须分别记录训练峰值、KV cache 和引擎保留显存；OOM 降级后不能静默改变实验协议 |
| 2×48GB 或 2×80GB 同机 | 稳定 learner + rollout 分离 | 默认每个角色独占 GPU；记录 topology、policy weight sync、rollout engine 与每角色 peak |
| 2×80GB 同机 | full-parameter GRPO experimental | 只作 one-update 可行性验证；任何 reference/KV/optimizer 挤压或余量不足都升级拓扑，不进入长跑 |
| 3×80GB 同机 | full-parameter GRPO 较稳妥起点 | learner 分片与 rollout 分离；仍须以本机 measured peak 和吞吐批准，不是保证可跑 |
| 4×80GB+ 同机 | PPO 或更多常驻角色 | policy、rollout、reference、value/reward 的 placement 逐项冻结；未证明 PPO 必要性时不租 |

官方模型卡给出的原生 context 是 262,144 tokens，但 v2 RL 首轮把 `prompt + completion` hard cap 固定在 **8K–12K**，具体值只在 peak/KV smoke 后冻结。原生上限描述模型能力，不是课程预算。reward、格式检查、代码单测和 sandbox verifier 优先在 CPU 隔离环境执行；只有模型 judge 明确属于实验对象时才为 reward 分配 GPU。

当前官方资料没有覆盖“Qwen3.5-4B-Base、text-only、冻结 vision/aligner、我们的 LoRA/GRPO 参数、NVIDIA 目标卡”这一完整组合的权威 peak memory。[ms-swift PR #9800](https://github.com/modelscope/ms-swift/pull/9800) 的可见 Qwen3.5-4B GRPO 验证使用 `2×Ascend 910B3 64 GiB`、LoRA、`256+128` tokens、`num_generations=2` 且只跑 1 step；它是 NPU codepath smoke，不可外推为 NVIDIA 容量或稳定长跑证据。

磁盘 gate 也不再使用固定 `50 GiB`：开机前按 `base snapshot + optimizer/shards + retained checkpoints + merged export + rollout/logs + staging copy` 逐项估算，再保留至少 15% 文件系统余量；save/reload smoke 产生的实际字节数必须回填后续 block。checkpoint retention 或导出格式变化时重新计算。

v2 依据：[Qwen3.5-4B-Base model card](https://huggingface.co/Qwen/Qwen3.5-4B-Base)、[Transformers Qwen3.5 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_5)、[ms-swift Qwen3.5 Best Practice](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Qwen3_5-Best-Practice.md)与[ms-swift FAQ](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Frequently-asked-questions.md)。

## 分阶段租卡上限

下表是预算窗口，不是必须耗尽的配额。wall time 到上限仍未得到关键证据时应关机复盘。

| Block | Days | 配置 | 建议 wall time | 目标 | 关机 Gate |
|---|---|---|---:|---|---|
| A | 08–10 | CPU；eval 时 1×H100 | 2–4h GPU | 数据契约、manifest、Base 逐样本 baseline | sample audit 与 frozen eval 已落盘 |
| B | 11–12 | 1×RTX 4090 24GB | 8–14h | tiny overfit、受控 SFT、checkpoint 对比 | loss/mask/生成闭环与 early/mid/final 证据完整 |
| C | 15–17 | 默认 1×H100 80GB | smoke 后冻结，参考 8–12h | Qwen3.5 onboarding、新 baseline、coding SFT/S1 与 exact resume | M1–M6、promotion 与 continuity 全部有 evidence |
| D | 18 | 2×H100 同机 | 2–4h | Qwen3.5 Megatron/GDN 最小 codepath | rank/group/model mapping/checkpoint/export 有 runtime evidence |
| E | 19 | 1×H100；必要时复用 2 卡 | 3–5h | optimizer/LR 对照、failure injection 与 profiler | LR、mask/processor、resume、throughput failure 可识别、可恢复 |
| F | 22–23 | CPU 准备；1×H100 80GB | 3–6h GPU | coding preference audit 与 Qwen3.5 DPO smoke | parent=S1；pair/logprob/mask 与 v2 eval 完整 |
| G | 24–25 | 1×80GB smoke；优先 2×48/80GB 分离 | smoke 后冻结，参考 4–8h | Qwen3.5 coding GRPO | parent=S1；reward/group/8K cap/peak/update 守恒 |
| H | 26–28 | CPU/无卡模式 | 0h | 固定 release 的 Qwen3.5 compatibility、replay 与 runbook | 明确 pass 或 blocked；不猜参数、不换模型 |
| I | 29 | Day 26 验证的最小拓扑，≤4×H100 | smoke 后冻结，参考 4–8h | verified-runtime rollout/replay/full loop 与 reward 修改 | 不支持 slime 时保留 ms-swift 主线证据，不伪造 full loop |
| J | 30 | 1×80GB 或已验证的最小 topology | 2–3h | Qwen3.5 clean reproduction | Base→S1→DPO/GRPO 分支与 v1/v2 boundary 可重建 |

### Optional Capstone Blocks（与 A–J 分账）

| Block | Days | 参考配置 | 初始 planning window | 目标 | 关机 Gate |
|---|---|---|---:|---|---|
| K | 31 | CPU/无卡 | 0h GPU | S0/数据/eval/runtime charter、teacher=null | manifests/scorer/selection/frozen seal 完整 |
| L | 32 | 同机 2×H100 80GB | 2–4h wall | Qwen3.5 single/TP2 parity、full/LoRA/QLoRA measured capacity | loss/grad/peak/checkpoint conversion 分叉可解释 |
| M | 33–36 | CPU/无卡 | 0h GPU | deferred teacher templates | 无 T0/T1/T2、无租卡；等待 charter v2 |
| N | 37 | 1×80GB smoke；优先 2×48/80GB 分离 | smoke 后冻结 | S1→direct coding RL→S2 | reward/guardrail/promotion 与 student budget 锁定 |
| O | 38–40 | CPU/无卡 | 0h GPU | deferred teacher-trace/OPD templates | 无 S1d/S3/S3d、无伪 teacher payload |
| P | 41–42 | 1–2× inference；最小 replay topology | 3–6h wall | S0/S1/S2 confirmation、成本与 clean replay | consumption record、paired evidence、clean reproduction 完整 |

活动 Capstone 不预付一个总 GPU-hours 配额；Day 32 的 measured capacity 与 Day 37 one-update peak 分别冻结后续上限。Deferred teacher/OPD 不占 GPU 预算，只有 charter v2 获批后才另做容量与成本核算。

8×H100 Stretch 另设 2–4 小时硬上限，只允许在 Block I 已成功、官方 recipe 与当前 pinned commit 完全对齐、数据/模型缓存命中且用户再次确认预算后执行。

## 开机前 Checklist

- [ ] 今日问题只能通过 GPU 回答，CPU 准备已结束。
- [ ] model/tokenizer/dataset/code/container 都有固定 revision 或 digest。
- [ ] 抽样检查了 raw、rendered、tokens、labels/mask；RL 还检查 reward 与 metadata。
- [ ] 命令已做 help/config/path dry check，输出目录不覆盖既有 run。
- [ ] 写明唯一自变量、baseline、`max_steps`、timeout、成功和停止条件。
- [ ] 模型、数据与镜像已缓存或估算下载时间；多卡前不临时编译大依赖。
- [ ] `nvidia-smi topo -m`、磁盘、CPU 内存、driver/CUDA/NCCL 已记录。
- [ ] 日志、逐样本输出、checkpoint 和 profiler 路径均可持久化。
- [ ] SSH 断开不影响任务；训练异常不会无限重启。
- [ ] 备份和关机步骤已准备。

## 各阶段 Readiness Gate

### 受控 SFT

- [ ] 一条样本能追到 label token，tiny dataset 能 overfit。
- [ ] Base frozen eval 已完成。
- [ ] accounting 显示目标配置有显存余量。
- [ ] save/load smoke 已通过。

### DPO

- [ ] SFT 起点 checkpoint 已冻结。
- [ ] prompt/chosen/rejected 使用同一 template，顺序与 mask 已抽查。
- [ ] preference margin、长度/source 分布和重复率已审计。
- [ ] 短 smoke 的 chosen/rejected logprob 对齐已验证。

### ms-swift GRPO

- [ ] reward 在手工构造的正/负样本上结果正确。
- [ ] prompt/group/completion/reward/advantage 的数量与 grain 明确。
- [ ] rollout response mask、EOS/truncation 和 policy version 可追踪。
- [ ] zero-variance 与过长生成有停止条件。

### Qwen3.5-4B v2

- [ ] 独立 Python 3.12 环境的 exact lock、容器 digest、模型 revision、processor/template hashes 已冻结；没有修改 Day 1–12 历史环境。
- [ ] 完整模型通过 `AutoProcessor` 与多模态 conditional-generation loader 加载；没有把 full checkpoint 当成 CausalLM drop-in。
- [ ] 数据为 text-only coding scope；vision/aligner 的 `requires_grad`、梯度、optimizer state 与 adapter targets 均经断言为零。
- [ ] Base baseline、token/mask golden、tiny overfit、save/reload 与 inference parity 均使用同一 v2 模型协议。
- [ ] 当前 topology 已完成 peak smoke，训练和 rollout 各有 10%–15% headroom；RL 总长度不超过冻结的 8K–12K cap。
- [ ] reward/verifier 在 CPU sandbox 通过正负例与超时测试；GPU 角色只保留模型计算。
- [ ] 磁盘按实际 checkpoint/export 字节动态核算，保留策略不会在运行中耗尽数据盘。

### slime 多卡（仅 Qwen3.5 兼容 gate 通过后）

- [ ] ms-swift GRPO 已跑通，算法与数据 bug 不再借多卡定位。
- [ ] slime commit、官方容器、SGLang/Megatron 版本与启动脚本固定，且 release 明确通过 Qwen3.5 load→rollout→train→weight-sync。
- [ ] model conversion/load、prompt data 和 custom reward 在无卡/单卡层面已验证。
- [ ] rollout 与 train 的 batch invariant 已手算并由配置检查通过。
- [ ] 2–4 卡拓扑、角色放置、显存预算和 weight-sync 路径已写入 runbook。
- [ ] 8 卡参数未混入 Core 配置；不支持时状态写 `blocked` 并继续 ms-swift 主线，不换模型。

### Qwen3.5-4B Policy Capstone

- [ ] S0 exact revision、processor/template、vision/aligner freeze、data/code/container 与 S1/S2 selection policy 已冻结。
- [ ] single/TP2 parity、global-batch assertion、distributed save/reload 与 inference export 已通过。
- [ ] full/LoRA/QLoRA accounting 包含 weights/gradients/optimizer/activations/logits/KV/temporary peak。
- [ ] Attention/GDN、vocab、sequence-parallel 与 TP degree 的 layout 已验证。
- [ ] S1→S2 长 run 前已通过 one-update、resume、rollout/reward replay 和 short eval export。

### Deferred Teacher / OPD Extension

- [ ] 当前 `teacher_model_id/revision=null`，T0/T1/T2/S1d/S3/S3d 不存在，GPU 预算为 0。
- [ ] 只有用户另行选择 exact teacher 并批准 charter v2 后，才重新执行 tokenizer/processor alignment、teacher advantage、placement、replay 与成本 gate。
- [ ] 未激活时不得用 Qwen3.5-9B、Qwen3-8B 或任意方便加载的 checkpoint 顶替。

## 数据、镜像与关机

- 环境稳定后保存一次基础镜像；参考[镜像文档](https://www.autodl.com/docs/image/)。
- 训练数据先复制到本地数据盘。共享[文件存储](https://www.autodl.com/docs/fs/)用于备份，不默认作为训练热路径。
- Git 保存代码、配置、manifest、精简日志与报告；大 checkpoint 保存到可靠对象/文件存储并记录 checksum。
- 关机前依次确认：进程退出、日志 flush、checkpoint 可读、逐样本结果存在、关键产物已同步、run status 已更新。
- 未验证退出码与备份前，不把自动关机直接拼到长训练命令后。
