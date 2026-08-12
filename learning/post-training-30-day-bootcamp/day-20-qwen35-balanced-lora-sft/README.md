# Day 20 — Qwen3.5-4B balanced LoRA SFT

状态：`standalone_probe_closed / no_passing_probe / main_not_started / Base_active`
范围：standalone pull-forward experiment；不是顺序课程的 Day 20 weekend failure-signature review 完成记录。

> 以上状态与本文其余命令描述的是已封存的 v1 实验。新的候选工厂、
> 训练前 E2B gate、24k/320k 数据合同和 v3 评估链路见
> [V2-GPU-RUNBOOK.md](./V2-GPU-RUNBOOK.md)。v2 不复用或改写 v1 产物。

Day 20 的目标不是继续扩大 Full SFT，而是用一次可回退、可审计的 LoRA 实验回答一个更具体的问题：在 Qwen3.5 原生模板和解析链路正确的前提下，均衡复用四类能力数据，是否能在不牺牲 Base 广泛能力的情况下得到一个可靠的新锚点。

最终结果只有两种：某个 `early`、`mid` 或 `final` LoRA 同时通过全部门槛，成为唯一允许合并的 winner；否则不生成 `DAY20-PASS.json`，继续使用 Base。Day 20 不会为了“必须有一个 SFT 结果”而降低门槛。

## 为什么 Day 19 Full SFT 可能不如 Base，甚至不如小模型

当前证据支持两个同时存在的因素，不能只归因于参数量。

第一类是测量偏差。旧链路没有完整遵循 Qwen3.5 的响应边界：模板、非思考前缀、生成 token 与最终文本的切分，以及 HumanEval 的 solution/completion 判定会把不少有效代码判成格式错误。重打分证明这主要改写 Code 的可执行候选面；Math/Finance 的旧输出并没有被新 parser 救回，因此它们的退化不能继续归因于格式问题。

第二类是真实训练退化。Day 19 用约 64,608 个窄分布监督 token 更新 4B 模型的全部可训练参数；任务比例、固定答案外壳和单轮 SFT 风格都可能使模型过拟合局部格式，并造成灾难性遗忘。相比之下，旧 0.6B/1.6B 小模型使用了更成熟的配方、更多轮 rollout 和经过筛选的 E 数据。小模型在匹配的任务分布上胜过一个被窄数据 Full SFT 扰动的 4B 模型并不矛盾。

Day 20 同时控制这两类因素：评测固定使用 Qwen3.5 response-adapter v2 和生成 token 证据；训练只在语言层的有限投影上加 LoRA，并用四技能等 token 预算、短 LR probe、三个 token checkpoint 和 Base fallback 限制真实能力漂移。

## 冻结实验合同

- Base 是 `/root/autodl-tmp/Qwen--Qwen3.5-4B-Base/1001bb4d826a52d1f399e183466143f4da7b741b`。
- 单卡 GPU 0 必须是 H800；BF16，batch size 2，gradient accumulation 4，`max_length=2304`，一轮训练，不 packing，不 padding-free，不打乱数据。
- 模板固定为 `qwen3_5`，`enable_thinking=false`，`add_non_thinking_prefix=true`，`loss_scale=default+ignore_empty_think`。每条训练记录先由真实 ms-swift 模板审计，训练前再现场重新编码；监督 token 只按 `labels[1:] != -100` 计数。
- LoRA 固定为 rank 8、alpha 16、dropout 0.05、bias none。目标只允许 `language_model.layers` 中的 self-attention、Qwen3.5 原生 linear-attention 和 MLP 投影；视觉塔、aligner、embedding 与 `lm_head` 不得进入 LoRA。
- ms-swift 固定在 commit `565a1ad586a21d24b23931c52d2c62b49c39bee8`，且 worktree 必须干净。runner 不会安装或更新任何依赖。
- Probe 共 16,000 supervised tokens，每技能 4,000；主训练共 256,000，每技能 64,000。主训练只保留约 64,000、153,600、256,000 token 的 `early`、`mid`、`final` 三个完整 checkpoint；probe 只保留一个 model-only checkpoint。
- 固定 dev 是 112 条、四技能各 28 条；LR probe 使用其中稳定选择的 32 条诊断集、每技能 8 条。frozen test 不在 Day 20 消耗。

四类目标保持任务原生格式：

- General：MMLU 选择题只输出 `Final answer: A` 到 `D`；Tulu 高质量通用指令用于 replay。
- Math：GSM8K 保留推理，末行必须是显式数值 `Final answer: ...`。
- Finance：TAT-QA 只使用 canonical value/scale，例如数值、百分比、thousand/million/billion；不训练 FinQA DSL。
- Code：只训练函数体 continuation，不允许 Markdown fence、解释性 prose 或完整 `def/class` 重写。MBPP 优先，Tulu Code 只补足精确 token 容量。

## 固定六源

数据 adapter 只读取允许的 JSON/Parquet 字节，不下载或执行数据仓库中的 Python 文件。每个本地 staging 项都必须声明下表中的精确 revision；文件 SHA-256、原始行 SHA-256、转换器和 gold evidence 会进入 manifest。

| key | 固定来源 | revision | 用途 |
| --- | --- | --- | --- |
| `mmlu` | `cais/mmlu` | `c30699e8356da336a370243923dbaf21066bb9fe` | General MCQ，`all/auxiliary_train-*.parquet` |
| `tulu` | `allenai/tulu-3-sft-personas-instruction-following` | `fe0c7d350c9b4542b8d829a6f1daa1c259f0ba0e` | General instruction replay，`data/train-*.parquet` |
| `gsm8k` | `openai/gsm8k` | `740312add88f781978c0658806c59bc2815b9866` | Math，`main/train-*.parquet` |
| `tatqa` | `next-tat/TAT-QA` | `c96247f5077eac447f63527fd3dcfdc58bb56d6a` | Finance，`tatqa_dataset_train.json` |
| `mbpp` | `google-research-datasets/mbpp` | `4bb6404fdc6cacfda99d4ac4205087b89d32030c` | Code 主源，sanitized/full 的 train/validation/test Parquet |
| `tulu_code` | `allenai/tulu-3-sft-personas-code` | `1412abe88dd2976af977260788e033013449f7b2` | Code 容量补充，`data/train-*.parquet` |

主训练格式预算是 MMLU 31,998、Tulu General 32,002、GSM8K 64k、TAT-QA 64k、Code 64k；probe 对应为 1,998、2,002、4k、4k、4k。这个两 token 的偏移是远端真实模板审计后的必要合同修订：canonical MMLU 目标恒为 6 个 supervised tokens，32,000/2,000 在不复制样本时不可达。General 总预算仍严格是 64k/4k。Probe 必须至少产生 5 个 optimizer steps，并满足每种格式的最小记录数。

## 本地检查与无覆盖部署

先在本地验证代码；这些命令不会连接 AutoDL：

```bash
DAY20_LOCAL_DIR=learning/post-training-30-day-bootcamp/day-20-qwen35-balanced-lora-sft
python3 -m py_compile "${DAY20_LOCAL_DIR}"/*.py
python3 -m unittest discover -s "${DAY20_LOCAL_DIR}" -p 'test_*.py' -v
bash -n "${DAY20_LOCAL_DIR}/run_day20_autodl.sh"
python3 -m json.tool "${DAY20_LOCAL_DIR}/day20-source-config.example.json" >/dev/null
```

连接前只做用户指定的只读检查：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 autodl-31007 \
  'hostname && whoami && pwd'
```

预期用户是 `root`。若原样返回 `Connection refused`，立即停止；不要修改 `~/.ssh/config` 或 SSH key。此时通常是实例未开机，或控制台分配的 SSH 端口已改变。

首次部署必须先上传到一个不存在的 staging 目录，再原子改名；目标已存在时应比较文件而不是覆盖、删除或使用 `rsync --delete`：

```bash
DAY20_LOCAL_DIR=learning/post-training-30-day-bootcamp/day-20-qwen35-balanced-lora-sft
DAY20_REMOTE_DIR=/root/autodl-tmp/post-training-30-day-bootcamp/day-20-qwen35-balanced-lora-sft
DAY20_REMOTE_STAGE=/root/autodl-tmp/post-training-30-day-bootcamp/.day20-qwen35-balanced-lora-sft-upload-20260809

ssh autodl-31007 "test ! -e '${DAY20_REMOTE_DIR}' && test ! -e '${DAY20_REMOTE_STAGE}' && mkdir '${DAY20_REMOTE_STAGE}'"
rsync -a --checksum "${DAY20_LOCAL_DIR}/" "autodl-31007:${DAY20_REMOTE_STAGE}/"
ssh autodl-31007 "test ! -e '${DAY20_REMOTE_DIR}' && mv -T '${DAY20_REMOTE_STAGE}' '${DAY20_REMOTE_DIR}'"
```

如果远端 Day 20 目录已经存在，上述第一条命令会失败，这是预期的 no-overwrite 行为。不要用覆盖上传绕过它。

## 离线 source staging 与上传映射

本机已经 staging 的源根目录示例是 `/tmp/day20-offline-sources.eM97Kl`。只上传 adapter allowlist 中的数据字节，不上传 `.git`、README、dataset script、测试集或 MMLU 其余科目目录：

```bash
DAY20_LOCAL_SOURCES=/tmp/day20-offline-sources.eM97Kl
DAY20_REMOTE_RAW=/root/autodl-tmp/qwen35-v2/day20-raw
DAY20_REMOTE_RAW_STAGE=/root/autodl-tmp/qwen35-v2/.day20-raw-upload-20260809

ssh autodl-31007 "test ! -e '${DAY20_REMOTE_RAW}' && test ! -e '${DAY20_REMOTE_RAW_STAGE}' && mkdir '${DAY20_REMOTE_RAW_STAGE}'"
(
  cd "${DAY20_LOCAL_SOURCES}"
  rsync -aR --checksum \
    ./mmlu/all/auxiliary_train-00000-of-00001.parquet \
    ./tulu_if/data/train-00000-of-00001.parquet \
    ./gsm8k/main/train-00000-of-00001.parquet \
    ./tatqa/tatqa_dataset_train.json \
    ./mbpp/sanitized/train-00000-of-00001.parquet \
    ./mbpp/sanitized/validation-00000-of-00001.parquet \
    ./mbpp/sanitized/test-00000-of-00001.parquet \
    ./mbpp/full/train-00000-of-00001.parquet \
    ./mbpp/full/validation-00000-of-00001.parquet \
    ./mbpp/full/test-00000-of-00001.parquet \
    ./tulu_code/data/train-00000-of-00001.parquet \
    "autodl-31007:${DAY20_REMOTE_RAW_STAGE}/"
)
ssh autodl-31007 "test ! -e '${DAY20_REMOTE_RAW}' && mv -T '${DAY20_REMOTE_RAW_STAGE}' '${DAY20_REMOTE_RAW}'"
```

上传后的映射是：

| config key | 远端目录/文件 |
| --- | --- |
| `mmlu` | `/root/autodl-tmp/qwen35-v2/day20-raw/mmlu` |
| `tulu` | `/root/autodl-tmp/qwen35-v2/day20-raw/tulu_if` |
| `gsm8k` | `/root/autodl-tmp/qwen35-v2/day20-raw/gsm8k` |
| `tatqa` | `/root/autodl-tmp/qwen35-v2/day20-raw/tatqa` |
| `mbpp` | `/root/autodl-tmp/qwen35-v2/day20-raw/mbpp` |
| `tulu_code` | `/root/autodl-tmp/qwen35-v2/day20-raw/tulu_code` |

把仓库中的 `day20-source-config.example.json` 作为内容基线，发布到 runner 的固定默认位置 `/root/autodl-tmp/qwen35-v2/day20-source-config.json`。同样必须先写 staging 文件，再在目标不存在时原子改名：

```bash
DAY20_LOCAL_CONFIG=learning/post-training-30-day-bootcamp/day-20-qwen35-balanced-lora-sft/day20-source-config.example.json
DAY20_REMOTE_CONFIG=/root/autodl-tmp/qwen35-v2/day20-source-config.json
DAY20_REMOTE_CONFIG_STAGE=/root/autodl-tmp/qwen35-v2/.day20-source-config-upload-20260809.json

ssh autodl-31007 "test ! -e '${DAY20_REMOTE_CONFIG}' && test ! -e '${DAY20_REMOTE_CONFIG_STAGE}'"
scp "${DAY20_LOCAL_CONFIG}" "autodl-31007:${DAY20_REMOTE_CONFIG_STAGE}"
ssh autodl-31007 "test ! -e '${DAY20_REMOTE_CONFIG}' && mv -T '${DAY20_REMOTE_CONFIG_STAGE}' '${DAY20_REMOTE_CONFIG}'"
```

`prepare` 会重新计算每个文件和每条源记录的哈希，并用远端真实 Qwen3.5 tokenizer/template 审计容量；revision 声明不能替代这个字节级检查。

## E2B rotated credential：当前硬阻断

任何 `sandbox`、`probe-all` 或最终 promotion 在执行 Code 前都必须具备新的、已轮换 E2B credential。旧凭据、历史 shell 环境变量和此前暴露过的 key 都不得复用。

runner 只接受以下两个由凭据管理员安全放置的文件：

- `/root/autodl-tmp/secrets/day20-e2b.env`：root 所有、mode `0600`，恰好一个非空行 `E2B_API_KEY=<rotated-secret>`。
- `/root/autodl-tmp/secrets/day20-e2b-attestation.json`：root 所有、mode `0600`；`schema_version=1`、`domain=day20.e2b_credential_attestation`、`status=rotated`，包含 `created_at_utc`、credential 文件 SHA-256，以及对排除 `attestation_sha256` 后 canonical JSON 的 SHA-256 自签名。

E2B 运行时必须已经存在于 `/root/autodl-tmp/envs/day12-e2b/bin/python`，版本恰为 `e2b==2.37.0`。缺少 credential、attestation 或既有运行时时应停止并报告 blocker；不要 `pip install`，不要把 secret 写入仓库、命令行、日志或 README。

HumanEval 生成代码永远不在 AutoDL 主机运行。主机只做解析、语法/containment 静态检查；通过 eligibility 的代码在 fresh、network-disabled E2B sandbox 中执行。`--live-smoke` 只在当前 Day 20 run 第一次 sandbox 调用时发生。

## AutoDL 正式运行顺序

runner 的固定路径是：

```bash
cd /root/autodl-tmp/post-training-30-day-bootcamp/day-20-qwen35-balanced-lora-sft
```

可使用远端已有的 tmux 保持会话，但不要为了本实验安装 tmux 或任何依赖：

```bash
tmux new-session -s day20
```

### 1. 初始化与准备

```bash
bash run_day20_autodl.sh init
bash run_day20_autodl.sh prepare
```

`init` 会拒绝非 root、非 H800 GPU 0、低于 80 GiB 可用空间、Base 缺失、ms-swift commit 漂移或 dirty worktree。run root 写入 `/root/autodl-tmp/qwen35-v2/state/current-day20-run-root`；若已有未完成 run，不能另建新 run。

`prepare` 默认优先使用 `/root/autodl-tmp/qwen35-v2/day20-sources/{general,math,finance,code}.normalized.qwen35.jsonl` 和同目录的 `SOURCE-ADAPTER-MANIFEST.json`；若这四个已审计文件不齐，才使用上述 `/root/autodl-tmp/qwen35-v2/day20-source-config.json` 和固定 source adapter。成功标志是当前 run 下新建 `DAY20-MANIFEST.json`。

### 2. LR probes

推荐一次跑完 Base diagnostic 与三个 LR：

```bash
bash run_day20_autodl.sh probe-all
```

等价的逐个训练/评测入口是：

```bash
bash run_day20_autodl.sh probe 1e-5
bash run_day20_autodl.sh probe 3e-5
bash run_day20_autodl.sh probe 1e-4
```

但是单独 `probe LR` 不会创建最终的 `PROBE-SELECTION.json`；正式流程仍需 `probe-all` 完成 Base、三组 probe、Qwen3.5 v2 重打分、E2B 和选择。每个 probe 至少要比 Base diagnostic 总分高 2；Math、Finance、Code 各自最多退化 1；Code sandbox eligibility 至少 7/8。通过者按总分、General、较低 LR 排序。无通过 probe 时状态为 `no_passing_probe`，`train-main` 必须失败关闭。

### 3. 主训练

```bash
bash run_day20_autodl.sh train-main
```

该命令只接受经 immutable Base/probe/E2B artifacts 重新计算并验证的 LR selection。主训练 checkpoint 含 optimizer state，允许从当前 run 内最后一个完整且哈希通过的 checkpoint 恢复；不完整 attempt 不会冒充成功 checkpoint。

### 4. Base 与三个 checkpoint 的完整 dev 回归

```bash
for DAY20_CANDIDATE in base early mid final; do
  bash run_day20_autodl.sh eval "${DAY20_CANDIDATE}"
  bash run_day20_autodl.sh sandbox "${DAY20_CANDIDATE}"
done
```

`eval` 先生成 raw summary/predictions，再用 Qwen3.5 response-adapter v2 生成 SHA-bound sidecar；`sandbox` 只消费对应 sidecar 中通过 containment 的 Code。两者均为 no-overwrite 发布，不能通过删除部分 evidence 来伪造重跑。

需要诊断时可单独调用：

```bash
bash run_day20_autodl.sh eval base
bash run_day20_autodl.sh sandbox base
```

候选名只允许 `base`、`base-probe`、`early`、`mid`、`final` 和三个 `probe-<LR>`。

### 5. Finalize 与 winner-only merge

```bash
bash run_day20_autodl.sh finalize
```

runner 没有允许人工指定 checkpoint 的公共 `merge` 子命令。合并是 `finalize` 的受控后半段：它先验证四个完整候选及所有 lineage，发布 `DAY20-RESULTS.json`；仅当 `DAY20-PASS.json` 存在时才把 marker 指定的唯一 winner 与冻结 Base 合并。不要直接运行 `swift export` 绕过 promotion。

每个 LoRA 候选必须同时满足：

| 门槛 | 最低值 |
| --- | ---: |
| 总正确数 | 65 / 112 |
| General | 5 / 28 |
| Math | 17 / 28 |
| Finance | 10 / 28 |
| Code（E2B） | 14 / 28 |
| 格式合规 | 90 / 112 |
| Code sandbox execution eligible | 26 / 28 |
| infrastructure failures | 必须为 0 |

合格候选按 `总分降序 → General 降序 → Code 降序 → checkpoint token 较早` 排序。Base 只作为 fallback，不参加 LoRA promotion；若无合格 LoRA，`finalize` 只保留结果证据并打印 `Base remains active`，不得创建或手工补写 PASS，也不得 merge。

## 不可变安全边界

- 不重新训练 Day 18/19，不删除任何 Day 18 checkpoint，也不删除 Day 19 `best-e` 的 `checkpoint-423`。
- Day 19 canonical run 是 `/root/autodl-tmp/runs/day19-qwen35-20260808T100142Z`。表现差的 baseline A `checkpoint-667` 与 baseline B `checkpoint-741` 已按要求删除；不要重建它们。其余 Day 19 eval、日志、PASS 与 Best-E 继续保留。
- 不安装 Codex、Python 包、CUDA、ms-swift、E2B 或 tmux；不改 `~/.ssh/config`，不改 SSH key。
- 不在 AutoDL 主机执行模型生成代码或数据仓库脚本；Code 只在固定 E2B sandbox 执行。
- 不用 `rm`、`--delete`、覆盖上传或人工改 JSON 绕过 no-overwrite。失败 attempt、manifest、hash、checkpoint package 和评测 sidecar 都是诊断证据。
- 只合并 `DAY20-PASS.json` 指定的 winner；不合并 probe、Base 或未过门槛 checkpoint。
