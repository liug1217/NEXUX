"""
model.py
--------
一個簡化版的 GPT (decoder-only Transformer) 模型。
架構:token embedding + position embedding -> N 層 Transformer block -> 輸出層

每個 Transformer block 包含:
  1. Multi-head self-attention(帶因果遮罩,只能看到自己與之前的 token)
  2. Feed-forward 網路
  3. 兩個 LayerNorm + 殘差連接 (residual connection)
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from config import Config


class MultiHeadAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd 必須能被 n_head 整除"

        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head

        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 因果遮罩(causal mask):下三角矩陣,確保每個位置只能看到自己與之前的 token
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape  # batch, time(序列長度), channel(n_embd)

        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=2)

        # 拆成多個 head: (B, T, C) -> (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v  # (B, n_head, T, head_size)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # 合併多頭

        out = self.resid_dropout(self.out_proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """一個完整的 Transformer decoder block。"""

    def __init__(self, config: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # 殘差連接 + attention
        x = x + self.ff(self.ln2(x))     # 殘差連接 + feed-forward
        return x


class GPTModel(nn.Module):
    def __init__(self, config: Config, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        idx: (B, T) 輸入的 token id
        targets: (B, T) 標籤(下一個字元),訓練時才會傳入。
                 targets 裡標成 -100 的位置,代表「不計入 loss」,
                 這是 SFT(問答微調)訓練時,用來蓋住「問題」部分、
                 只強迫模型學會生成「答案」部分的機制。
        回傳: logits, loss(若無 targets 則 loss 為 None)
        """
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"輸入長度 {T} 超過模型支援的 block_size {self.config.block_size}"
        )

        pos = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)  # (B, T, n_embd)
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # ignore_index=-100:targets 裡標成 -100 的位置不計入 loss。
            # 這其實是 PyTorch 的預設值,這裡明確寫出來,是為了讓
            # SFT 訓練時的用途更清楚(dataset.py 的 SFTDataset 會
            # 把「問題」部分的標籤設成 -100)。
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        自迴歸生成文字。
        idx: (B, T) 目前已有的 token 序列(prompt)
        回傳: (B, T + max_new_tokens) 的完整序列
        """
        self.eval()
        for _ in range(max_new_tokens):
            # 若序列長度超過 block_size,只取最後 block_size 個 token 當作 context
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)  # 只取最後一個位置的 logits

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_id], dim=1)

        self.train()
        return idx
