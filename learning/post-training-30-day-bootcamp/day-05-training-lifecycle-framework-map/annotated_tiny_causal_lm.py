"""
一个可以在 CPU 上直接运行的、带完整尺寸注释的 tiny causal LM 训练示例。

运行：

    /Users/jiaweiqian/miniforge3/envs/agent_backend/bin/python \
      learning/post-training-30-day-bootcamp/day-05-training-lifecycle-framework-map/annotated_tiny_causal_lm.py

这个脚本刻意不使用 Hugging Face Trainer，目的是把以下边界全部显式展开：

    raw token sample
      -> collator / padding
      -> token embedding
      -> Q/K/V
      -> causal self-attention
      -> MLP
      -> final hidden states
      -> vocabulary logits
      -> shifted token loss
      -> backward
      -> gradient accumulation
      -> gradient clipping
      -> optimizer / scheduler
      -> checkpoint / resume

重要术语：

1. Parameter（参数）
   跨 batch 长期保存并由 optimizer 更新，例如 embedding.weight、W_Q、W_K、W_V。

2. Activation（激活）
   某次 forward 临时产生的 tensor，例如 embeddings、Q/K/V、attention probabilities、
   hidden states 和 logits。训练时 autograd 会保留 backward 所需的 activation。

3. Gradient（梯度）
   loss.backward() 产生并累加在 parameter.grad 中；它的 shape 与对应 parameter 相同。

4. Optimizer state（优化器状态）
   AdamW 为每个可训练参数维护一阶矩、二阶矩和 step。它们不是 model parameter，
   但 exact resume 时必须保存。

这不是生产训练实现：

- 使用 learned position embedding，便于看清 shape；很多现代 LLM 使用 RoPE。
- 不实现 KV cache、FlashAttention、mixed precision、distributed training。
- 数据是小型合成 token 序列，只用于验证训练数据流。
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# -----------------------------------------------------------------------------
# 0. 实验配置
# -----------------------------------------------------------------------------

# 特殊 token ID。真实模型由 tokenizer 定义这些 ID。
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
USER_ID = 3
ASSISTANT_ID = 4

# V：vocabulary size。每个 logit vector 都有 VOCAB_SIZE 个分量。
VOCAB_SIZE = 32

# D：d_model / hidden size / residual-stream width。
# 每个 token 在 Transformer 主干中由一个 D_MODEL 维向量表示。
D_MODEL = 16

# H：attention head 数；Dh：每个 head 中 Q/K/V vector 的宽度。
NUM_HEADS = 4
HEAD_DIM = D_MODEL // NUM_HEADS  # 16 / 4 = 4

# MLP 中间层宽度，真实 LLM 中常比 d_model 大数倍。
D_FF = 4 * D_MODEL  # 64

NUM_LAYERS = 2
MAX_SEQUENCE_LENGTH = 16

MICRO_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 3e-3
MAX_GRAD_NORM = 1.0


# -----------------------------------------------------------------------------
# 1. 数据：构造 assistant-only SFT labels
# -----------------------------------------------------------------------------


class ToySFTDataset(Dataset):
    """每条样本都包含 prompt 和一个单-token answer。

    序列格式：

        [BOS, USER, prompt..., ASSISTANT, answer, EOS]

    labels 格式：

        [-100, -100, -100..., -100, answer, EOS]

    因而每条样本恰好有两个有效 target：answer 和 EOS。
    prompt 虽不作为直接预测目标，仍是预测 answer 时使用的上下文。
    """

    def __init__(self) -> None:
        prompt_answer_pairs = [
            ([5, 6], 7),
            ([8, 9], 10),
            ([11], 12),
            ([13, 14, 15], 16),
            ([17, 18], 19),
            ([20], 21),
            ([22, 23, 24], 25),
            ([26, 27], 28),
        ]

        self.samples: list[dict[str, list[int]]] = []
        for prompt_tokens, answer_token in prompt_answer_pairs:
            input_ids = [
                BOS_ID,
                USER_ID,
                *prompt_tokens,
                ASSISTANT_ID,
                answer_token,
                EOS_ID,
            ]

            # BOS、USER、prompt 和 ASSISTANT marker 都只作为上下文，不直接计入 loss。
            number_of_context_tokens = 3 + len(prompt_tokens)
            labels = [
                *([-100] * number_of_context_tokens),
                answer_token,
                EOS_ID,
            ]

            assert len(input_ids) == len(labels)
            self.samples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.samples[index]


def collate_sft_batch(samples: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
    """将不同长度的 Python lists padding 成一个 tensor batch。

    若本 batch 最长样本长度为 T，则输出：

        input_ids:      [B, T], torch.long
        labels:         [B, T], torch.long
        attention_mask: [B, T], torch.bool

    padding 位置：

        input_ids      = PAD_ID
        labels         = -100  （不产生 token loss）
        attention_mask = False （不能作为 attention key/value）
    """

    batch_size = len(samples)
    sequence_length = max(len(sample["input_ids"]) for sample in samples)

    input_ids = torch.full(
        (batch_size, sequence_length),
        fill_value=PAD_ID,
        dtype=torch.long,
    )
    labels = torch.full(
        (batch_size, sequence_length),
        fill_value=-100,
        dtype=torch.long,
    )
    attention_mask = torch.zeros(
        (batch_size, sequence_length),
        dtype=torch.bool,
    )

    for row, sample in enumerate(samples):
        sample_length = len(sample["input_ids"])
        input_ids[row, :sample_length] = torch.tensor(sample["input_ids"])
        labels[row, :sample_length] = torch.tensor(sample["labels"])
        attention_mask[row, :sample_length] = True

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


# -----------------------------------------------------------------------------
# 2. 模型：显式写出 attention 中所有主要 activation
# -----------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        assert D_MODEL % NUM_HEADS == 0

        # 这些都是可训练 parameter。
        #
        # nn.Linear(in_features=D, out_features=D, bias=False) 的 weight shape
        # 在 PyTorch 中是 [out_features, in_features] = [D, D]。
        self.q_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.v_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.out_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        capture_shapes: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """计算一层 causal multi-head self-attention。

        输入：
            x:              [B, T, D]
            attention_mask: [B, T]

        输出：
            attention_output: [B, T, D]

        其中：
            B  = batch size
            T  = sequence length
            D  = d_model
            H  = num heads
            Dh = head dim = D / H
        """

        batch_size, sequence_length, hidden_size = x.shape
        assert hidden_size == D_MODEL

        trace: dict[str, Any] = {}

        # 一次矩阵乘法同时处理 B*T 个 token position，不需要 Python for loop。
        #
        # [B, T, D] @ [D, D] -> [B, T, D]
        q_before_split = self.q_proj(x)
        k_before_split = self.k_proj(x)
        v_before_split = self.v_proj(x)

        # 把总宽度 D 拆成 H 个 head，每个 head 宽度 Dh：
        #
        # [B, T, D]
        #   -> view [B, T, H, Dh]
        #   -> transpose [B, H, T, Dh]
        q = q_before_split.view(
            batch_size, sequence_length, NUM_HEADS, HEAD_DIM
        ).transpose(1, 2)
        k = k_before_split.view(
            batch_size, sequence_length, NUM_HEADS, HEAD_DIM
        ).transpose(1, 2)
        v = v_before_split.view(
            batch_size, sequence_length, NUM_HEADS, HEAD_DIM
        ).transpose(1, 2)

        # 每个 head 内，让每个 query position 与每个 key position 做点积：
        #
        # Q:                [B, H, T, Dh]
        # K.transpose:      [B, H, Dh, T]
        # attention_scores: [B, H, T, T]
        #
        # scores[b, h, i, j] 表示：
        # batch b、head h 中，位置 i 的 query 对位置 j 的 key 的原始注意力分数。
        attention_scores = q @ k.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(HEAD_DIM)

        # causal_mask[i, j] == True 表示 j 在 i 的未来，必须屏蔽。
        #
        # shape [T, T]，例如 T=4：
        #
        # False True  True  True
        # False False True  True
        # False False False True
        # False False False False
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=x.device,
            ),
            diagonal=1,
        )
        attention_scores = attention_scores.masked_fill(
            causal_mask[None, None, :, :],
            float("-inf"),
        )

        # attention_mask=False 的 padding token 不能作为 key/value。
        #
        # [B, T] -> [B, 1, 1, T]，随后广播到所有 head 和 query position。
        key_padding_mask = ~attention_mask[:, None, None, :]
        attention_scores = attention_scores.masked_fill(
            key_padding_mask,
            float("-inf"),
        )

        # 对最后一维 key positions 做 softmax：
        #
        # attention_probs.shape = [B, H, T, T]
        # attention_probs[b,h,i,:].sum() == 1
        attention_probs = F.softmax(attention_scores, dim=-1)

        # 用 attention probability 对 value vectors 加权求和：
        #
        # [B, H, T, T] @ [B, H, T, Dh] -> [B, H, T, Dh]
        context_per_head = attention_probs @ v

        # 把 H 个 head 拼回总宽度 D：
        #
        # [B, H, T, Dh]
        #   -> transpose [B, T, H, Dh]
        #   -> reshape [B, T, D]
        context = context_per_head.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            D_MODEL,
        )

        # 再经过一个可训练输出投影，shape 保持 [B, T, D]。
        attention_output = self.out_proj(context)

        if capture_shapes:
            trace = {
                "attention.input": tuple(x.shape),
                "attention.q_before_split": tuple(q_before_split.shape),
                "attention.k_before_split": tuple(k_before_split.shape),
                "attention.v_before_split": tuple(v_before_split.shape),
                "attention.q_per_head": tuple(q.shape),
                "attention.k_per_head": tuple(k.shape),
                "attention.v_per_head": tuple(v.shape),
                "attention.scores": tuple(attention_scores.shape),
                "attention.probabilities": tuple(attention_probs.shape),
                "attention.context_per_head": tuple(context_per_head.shape),
                "attention.concatenated_context": tuple(context.shape),
                "attention.output": tuple(attention_output.shape),
            }

        return attention_output, trace


class FeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # [D] -> [D_FF] -> [D]
        self.up_proj = nn.Linear(D_MODEL, D_FF)
        self.down_proj = nn.Linear(D_FF, D_MODEL)

    def forward(
        self,
        x: torch.Tensor,
        capture_shapes: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # x:          [B, T, D]
        # mlp_hidden: [B, T, D_FF]
        # output:     [B, T, D]
        mlp_hidden = F.gelu(self.up_proj(x))
        output = self.down_proj(mlp_hidden)

        trace: dict[str, Any] = {}
        if capture_shapes:
            trace = {
                "mlp.input": tuple(x.shape),
                "mlp.expanded_activation": tuple(mlp_hidden.shape),
                "mlp.output": tuple(output.shape),
            }
        return output, trace


class TransformerBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(D_MODEL)
        self.attention = CausalSelfAttention()
        self.mlp_norm = nn.LayerNorm(D_MODEL)
        self.mlp = FeedForward()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        capture_shapes: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        trace: dict[str, Any] = {}

        # Pre-norm attention + residual：
        #
        # normalized、attention_output、x 的 shape 都是 [B, T, D]。
        normalized = self.attention_norm(x)
        attention_output, attention_trace = self.attention(
            normalized,
            attention_mask,
            capture_shapes,
        )
        x = x + attention_output

        # Pre-norm MLP + residual，shape 仍是 [B, T, D]。
        normalized = self.mlp_norm(x)
        mlp_output, mlp_trace = self.mlp(normalized, capture_shapes)
        x = x + mlp_output

        if capture_shapes:
            trace.update(attention_trace)
            trace["block.after_attention_residual"] = tuple(x.shape)
            trace.update(mlp_trace)
            trace["block.output"] = tuple(x.shape)

        return x, trace


class TinyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # 可训练 token embedding parameter：
        # token_embedding.weight.shape = [V, D]
        self.token_embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)

        # 教学用 learned position embedding：
        # position_embedding.weight.shape = [MAX_T, D]
        self.position_embedding = nn.Embedding(MAX_SEQUENCE_LENGTH, D_MODEL)

        self.blocks = nn.ModuleList(
            [TransformerBlock() for _ in range(NUM_LAYERS)]
        )
        self.final_norm = nn.LayerNorm(D_MODEL)

        # 最终 vocabulary projection：
        #
        # hidden [B,T,D] -> logits [B,T,V]
        # lm_head.weight.shape = [V,D]
        #
        # 很多真实 LLM 会令 lm_head.weight 与 token_embedding.weight 共享；
        # 本例保持独立，便于分别观察输入 embedding 和输出 projection。
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        capture_shapes: bool = False,
    ) -> dict[str, Any]:
        """完成一次 causal LM forward，并在给定 labels 时计算全部有效 token loss."""

        batch_size, sequence_length = input_ids.shape
        assert sequence_length <= MAX_SEQUENCE_LENGTH
        assert attention_mask.shape == (batch_size, sequence_length)

        trace: dict[str, Any] = {
            "batch.input_ids": tuple(input_ids.shape),
            "batch.attention_mask": tuple(attention_mask.shape),
        }

        # token_embeddings：
        # [B,T] 的整数索引 -> [B,T,D] 的浮点 activation。
        token_embeddings = self.token_embedding(input_ids)

        # positions：[T] = [0,1,...,T-1]
        # position_embeddings：[T,D]，加到每个 batch row 时自动广播成 [B,T,D]。
        positions = torch.arange(sequence_length, device=input_ids.device)
        position_embeddings = self.position_embedding(positions)

        # x 是进入第一层 Transformer 的 residual-stream activation。
        # x.shape = [B,T,D]
        x = token_embeddings + position_embeddings[None, :, :]

        if capture_shapes:
            trace.update({
                "embedding.parameter": tuple(self.token_embedding.weight.shape),
                "embedding.token_activation": tuple(token_embeddings.shape),
                "embedding.position_activation": tuple(position_embeddings.shape),
                "embedding.combined_activation": tuple(x.shape),
            })

        # 每一层输入/输出都保持 [B,T,D]；内部会暂时产生 Q/K/V、scores 和 MLP activation。
        for layer_index, block in enumerate(self.blocks):
            x, block_trace = block(
                x,
                attention_mask,
                capture_shapes=capture_shapes and layer_index == 0,
            )
            if capture_shapes and layer_index == 0:
                trace.update({
                    f"layer_0.{name}": shape
                    for name, shape in block_trace.items()
                })

        # final_hidden_states 是送入 LM head 的 contextual representation。
        # shape = [B,T,D]
        final_hidden_states = self.final_norm(x)

        # 对所有 B*T 个 position 一次性做 vocabulary projection：
        #
        # [B,T,D] @ [D,V] -> [B,T,V]
        #
        # logits[b,t,:] 是一个 V 维 vector，表示位置 t 对“下一个 token”的未归一化分数。
        logits = self.lm_head(final_hidden_states)

        result: dict[str, Any] = {
            "logits": logits,
            "loss": None,
            "per_token_loss": None,
            "valid_label_mask": None,
            "trace": trace,
        }

        if capture_shapes:
            trace.update({
                "final_hidden_states": tuple(final_hidden_states.shape),
                "lm_head.parameter": tuple(self.lm_head.weight.shape),
                "logits": tuple(logits.shape),
            })

        if labels is None:
            return result

        assert labels.shape == input_ids.shape

        # Causal next-token 对齐：
        #
        # 原 logits positions: 0, 1, ..., T-2  （丢掉最后一个 logits）
        # 原 label positions:  1, 2, ..., T-1  （丢掉第一个 label）
        #
        # 注意：这里不是只取一个位置，而是一次取出全部 T-1 对预测。
        shift_logits = logits[:, :-1, :].contiguous()  # [B,T-1,V]
        shift_labels = labels[:, 1:].contiguous()      # [B,T-1]

        # flatten 后，CrossEntropy 把 B*(T-1) 个 position 当作独立分类样本，
        # 每个样本都是在 V 个 token 中预测一个 target。
        flat_logits = shift_logits.view(-1, VOCAB_SIZE)  # [B*(T-1),V]
        flat_labels = shift_labels.view(-1)              # [B*(T-1)]

        # reduction="none" 让每个 position 的 loss 都显式保留下来。
        # target == -100 的位置由 ignore_index 忽略，返回的 position loss 为 0。
        flat_token_loss = F.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=-100,
            reduction="none",
        )
        per_token_loss = flat_token_loss.view(
            batch_size,
            sequence_length - 1,
        )

        valid_label_mask = shift_labels.ne(-100)  # [B,T-1], bool
        valid_label_count = valid_label_mask.sum()
        assert valid_label_count.item() > 0

        # 最终 scalar loss 是所有有效 target-token loss 的平均值，不是只取最后一位。
        loss = per_token_loss.sum() / valid_label_count

        result.update({
            "loss": loss,
            "per_token_loss": per_token_loss,
            "valid_label_mask": valid_label_mask,
        })

        if capture_shapes:
            trace.update({
                "labels": tuple(labels.shape),
                "shift_logits": tuple(shift_logits.shape),
                "shift_labels": tuple(shift_labels.shape),
                "flat_logits": tuple(flat_logits.shape),
                "flat_labels": tuple(flat_labels.shape),
                "per_token_loss": tuple(per_token_loss.shape),
                "valid_label_count": int(valid_label_count.item()),
                "scalar_loss": tuple(loss.shape),  # scalar tensor 的 shape 是 ()
            })

        return result


# -----------------------------------------------------------------------------
# 3. 训练辅助函数
# -----------------------------------------------------------------------------


def gradient_l2_norm(model: nn.Module) -> float:
    """计算当前所有 parameter.grad 的全局 L2 norm，但不修改 gradient."""

    squared_norm = torch.zeros((), dtype=torch.float32)
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared_norm += parameter.grad.detach().float().pow(2).sum()
    return squared_norm.sqrt().item()


def print_shape_trace(trace: dict[str, Any]) -> None:
    print("\n=== First forward: activation / parameter shape trace ===")
    for name, shape in trace.items():
        print(f"{name:48s} {shape}")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    global_step: int,
) -> None:
    """保存 continuous resume 所需的主要状态。

    生产训练还应保存 dataloader/sampler progress、Python/NumPy RNG、
    distributed sharding metadata、AMP GradScaler 等。
    """

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": global_step,
            "torch_rng_state": torch.get_rng_state(),
        },
        path,
    )


def make_optimizer_and_scheduler(
    model: nn.Module,
    total_optimizer_steps: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    # 每执行一次 scheduler.step()，下一次 update 使用的 LR 略微降低。
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: max(
            0.1,
            1.0 - 0.5 * step / max(total_optimizer_steps, 1),
        ),
    )
    return optimizer, scheduler


# -----------------------------------------------------------------------------
# 4. 完整训练 loop
# -----------------------------------------------------------------------------


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cpu")

    dataset = ToySFTDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_sft_batch,
    )

    # 本例有 8 samples / micro-batch 2 = 4 micro steps；
    # accumulation=2，所以产生 2 次 optimizer updates。
    assert len(dataloader) % GRADIENT_ACCUMULATION_STEPS == 0
    total_optimizer_steps = len(dataloader) // GRADIENT_ACCUMULATION_STEPS

    model = TinyCausalLM().to(device)
    optimizer, scheduler = make_optimizer_and_scheduler(
        model,
        total_optimizer_steps,
    )

    print("=== Model constants ===")
    print(
        f"V={VOCAB_SIZE}, D={D_MODEL}, H={NUM_HEADS}, "
        f"Dh={HEAD_DIM}, D_ff={D_FF}"
    )
    print(
        f"micro_batch={MICRO_BATCH_SIZE}, "
        f"accumulation={GRADIENT_ACCUMULATION_STEPS}, "
        f"optimizer_updates={total_optimizer_steps}"
    )

    model.train()

    # set_to_none=True 不分配全零 grad tensor；第一次 backward 时再创建。
    optimizer.zero_grad(set_to_none=True)

    global_step = 0       # optimizer.step() 次数
    first_forward = True

    for micro_step, batch in enumerate(dataloader, start=1):
        batch = {
            name: tensor.to(device)
            for name, tensor in batch.items()
        }

        # -------------------------- FORWARD --------------------------
        #
        # 产生 activation、logits 和 scalar loss。
        # 这一阶段读取 parameter，但不修改 parameter 或 parameter.grad。
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            capture_shapes=first_forward,
        )
        raw_loss = outputs["loss"]
        assert isinstance(raw_loss, torch.Tensor)

        if first_forward:
            print_shape_trace(outputs["trace"])
            print("\nFirst batch input_ids:")
            print(batch["input_ids"])
            print("First batch labels:")
            print(batch["labels"])
            print("Valid shifted-label mask:")
            print(outputs["valid_label_mask"])
            print("Per-position loss (masked positions are 0):")
            print(outputs["per_token_loss"].detach())
            first_forward = False

        # --------------------- GRADIENT SCALING ----------------------
        #
        # PyTorch 默认让多次 backward 的 parameter.grad 相加。
        # 除以 accumulation steps 后，累积结果是 micro-batch gradients 的平均值。
        loss_for_backward = raw_loss / GRADIENT_ACCUMULATION_STEPS

        # -------------------------- BACKWARD -------------------------
        #
        # 反向传播经过：
        #
        # scalar loss
        #   -> logits [B,T,V]
        #   -> lm_head.weight.grad [V,D]
        #   -> hidden-state gradients [B,T,D]
        #   -> MLP / QKV / embedding parameter gradients
        #
        # backward 只写入/累加 .grad；它不修改 model parameter。
        loss_for_backward.backward()

        valid_tokens = int(outputs["valid_label_mask"].sum().item())
        accumulated_grad_norm = gradient_l2_norm(model)
        print(
            f"\nmicro_step={micro_step} "
            f"raw_loss={raw_loss.item():.6f} "
            f"loss_used_for_backward={loss_for_backward.item():.6f} "
            f"valid_target_tokens={valid_tokens} "
            f"accumulated_grad_norm={accumulated_grad_norm:.6f}"
        )

        accumulation_boundary = (
            micro_step % GRADIENT_ACCUMULATION_STEPS == 0
        )
        if not accumulation_boundary:
            # 此时 parameter 未更新；下一个 backward 会继续累加到现有 .grad。
            continue

        # 保存一个 parameter 的 update 前副本，用于证明 optimizer.step() 才会改参数。
        q_weight = model.blocks[0].attention.q_proj.weight
        q_weight_before_step = q_weight.detach().clone()

        # ---------------------- GRADIENT CLIPPING --------------------
        #
        # clip_grad_norm_ 返回裁剪前的 global grad norm。
        # 若 norm > MAX_GRAD_NORM，则所有 gradients 按同一比例缩小。
        pre_clip_grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=MAX_GRAD_NORM,
        )
        post_clip_grad_norm = gradient_l2_norm(model)

        # ----------------------- OPTIMIZER STEP ----------------------
        #
        # 这里才真正更新：
        #   1. model parameters
        #   2. AdamW first/second moments
        optimizer.step()

        # scheduler 更新下一次 optimizer update 使用的 learning rate。
        scheduler.step()

        max_q_weight_change = (
            q_weight.detach() - q_weight_before_step
        ).abs().max().item()

        global_step += 1
        print(
            f"optimizer_step={global_step} "
            f"pre_clip_norm={float(pre_clip_grad_norm):.6f} "
            f"post_clip_norm={post_clip_grad_norm:.6f} "
            f"max_abs_q_weight_change={max_q_weight_change:.8f} "
            f"next_lr={scheduler.get_last_lr()[0]:.8f}"
        )

        # 清除刚才消费掉的 gradients。optimizer state 和 model parameters 保留。
        optimizer.zero_grad(set_to_none=True)

    assert global_step == total_optimizer_steps

    # ---------------------- CHECKPOINT / RESUME ---------------------
    #
    # 使用临时文件演示，不在仓库里留下二进制 checkpoint。
    checkpoint_path = Path(tempfile.gettempdir()) / "day05_tiny_causal_lm.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        global_step,
    )

    # 创建全新的对象，再把 checkpoint 状态加载进去。
    resumed_model = TinyCausalLM().to(device)
    resumed_optimizer, resumed_scheduler = make_optimizer_and_scheduler(
        resumed_model,
        total_optimizer_steps,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    resumed_model.load_state_dict(checkpoint["model_state_dict"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    resumed_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    resumed_global_step = int(checkpoint["global_step"])

    # 验证恢复后的 model parameter 与保存前完全一致。
    for original, resumed in zip(
        model.parameters(),
        resumed_model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(original, resumed)

    print("\n=== Checkpoint verification ===")
    print(f"checkpoint_path={checkpoint_path}")
    print(f"resumed_global_step={resumed_global_step}")
    print("model_parameters_match=True")
    print("Training lifecycle completed successfully.")


if __name__ == "__main__":
    main()
