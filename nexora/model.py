"""
nexora/model.py
---------------
Nexora Native v1 模型架構。
Decoder-only Transformer，與根目錄 model.py（GPT-2 路線）完全分離。

不引用任何外部 pretrained weights，不依賴 HuggingFace / GPT-2 / CKIP / UER。
所有權重於 NexoraModel.__init__() 中隨機初始化。
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from nexora.config import NexoraConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: NexoraConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_size = cfg.n_embd // cfg.n_head
        self.n_embd = cfg.n_embd

        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.out = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.attn_drop = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out(y))


class FeedForward(nn.Module):
    def __init__(self, cfg: NexoraConfig):
        super().__init__()
        ffn = 4 * cfg.n_embd
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, ffn),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: NexoraConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ff = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class NexoraModel(nn.Module):
    """
    Nexora Native v1 語言模型。
    Decoder-only Transformer，所有參數隨機初始化，不載入任何外部 checkpoint。
    """

    def __init__(self, cfg: NexoraConfig):
        super().__init__()
        self.cfg = cfg
        self.gradient_checkpointing = False

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.emb_drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self._init_weights()
        n = sum(p.numel() for p in self.parameters())
        print(f"[NexoraModel] 已建立 Nexora-Nano v1"
              f" (n_embd={cfg.n_embd}, n_head={cfg.n_head},"
              f" n_layer={cfg.n_layer}, vocab={cfg.vocab_size})")
        print(f"[NexoraModel] 總參數量: {n:,} ({n/1e6:.3f}M)")
        print(f"[NexoraModel] 所有參數均為隨機初始化，不依賴任何外部 pretrained weights。")

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # 縮放殘差投影（GPT-2 paper 的做法：std /= sqrt(2 * n_layer)）
        scale = (2 * self.cfg.n_layer) ** -0.5
        for name, p in self.named_parameters():
            if name.endswith("out.weight") or name.endswith("ff.net.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 * scale)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        idx:     (B, T) token IDs
        targets: (B, T) 下一步預測標籤，-100 的位置不計入 loss
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, (
            f"輸入長度 {T} 超過 block_size {self.cfg.block_size}"
        )
        pos = torch.arange(T, device=idx.device)
        x = self.emb_drop(self.token_emb(idx) + self.pos_emb(pos))

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = grad_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                for b in range(logits.size(0)):
                    logits[b, sorted_idx[b][remove[b]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

            if eos_id is not None and idx.size(0) == 1 and next_id.item() == eos_id:
                break

        self.train()
        return idx

    def param_summary(self) -> dict:
        """回傳各部分參數量統計。"""
        token_emb = self.token_emb.weight.numel()
        pos_emb = self.pos_emb.weight.numel()
        blocks = sum(p.numel() for block in self.blocks for p in block.parameters())
        ln_f = sum(p.numel() for p in self.ln_f.parameters())
        head = self.head.weight.numel()
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "non_trainable": total - trainable,
            "token_embedding": token_emb,
            "position_embedding": pos_emb,
            "transformer_blocks": blocks,
            "final_layernorm": ln_f,
            "output_head": head,
        }
