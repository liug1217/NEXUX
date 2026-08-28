"""
app.py — NEXUX AI 聊天機器人（Gradio 雲端版 · HuggingFace Spaces）
====================================================================
啟動時自動完成:
1. 載入 1.29 GB 無污染母體大腦 model.pt（Loss 0.0294）
2. 載入 TF-IDF RAG 索引，秒級注入最新知識（零重訓）
3. 對外開放 Gradio 原生免費 CORS-free JSON/WebSocket API

獨立主站 ai.nexuxai.net 可直接 fetch 這個端點取得回覆。
"""

import gc
import os
import re
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertTokenizer

from rag_engine import RAGEngine


# ==================== 推理設定 ====================

@dataclass
class InferenceConfig:
    block_size: int = 1024
    n_embd: int = 1024
    n_head: int = 16
    n_layer: int = 24
    dropout: float = 0.0
    max_new_tokens: int = 100
    temperature: float = 0.5
    top_k: int = 20
    top_p: float = 0.8
    repetition_penalty: float = 2.0


# ==================== GPTModel 完整架構 ====================

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout_p = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size
        pos = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0,
                 top_k=None, top_p=None, repetition_penalty=1.0, eos_id=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if repetition_penalty != 1.0:
                for b in range(idx.size(0)):
                    seen = set(idx[b].tolist())
                    for tid in seen:
                        if logits[b, tid] > 0:
                            logits[b, tid] /= repetition_penalty
                        else:
                            logits[b, tid] *= repetition_penalty

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                remove = cum_probs > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                for b in range(logits.size(0)):
                    logits[b, sorted_indices[b][remove[b]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

            if eos_id is not None and idx.size(0) == 1 and next_id.item() == eos_id:
                break
        return idx


# ==================== LoRA 熔接（Merge） ====================


# (LoRA merge 已移除 — 改用 RAG 注入知識，不再需要重訓)


# ==================== 文字後處理 ====================

_TURN_PATTERN = re.compile(r"\nA[:：]|\nB[:：]|\n問[:：]|\n答[:：]")


def truncate_reply(text):
    match = _TURN_PATTERN.search(text)
    return text[:match.start()].rstrip() if match else text.rstrip()


# ==================== 模型載入 ====================

def load_everything():
    """一次載入: 母體大腦 + TF-IDF RAG 索引"""
    device = "cpu"  # HF Spaces 免費版用 CPU

    model_path = "model.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到 {model_path}")

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    arch = ckpt["architecture"]
    config = InferenceConfig(
        block_size=arch["block_size"],
        n_embd=arch["n_embd"],
        n_head=arch["n_head"],
        n_layer=arch["n_layer"],
    )

    model = GPTModel(config, ckpt["vocab_size"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[app] 母體載入完成: {n_params / 1e6:.0f}M 參數 (Loss 0.0294)")

    vocab_path = "vocab.txt"
    tokenizer = BertTokenizer(
        vocab_file=vocab_path,
        do_lower_case=True,
        clean_up_tokenization_spaces=True,
    )
    print(f"[app] 詞表: {tokenizer.vocab_size} tokens")

    rag = RAGEngine.load("rag_index.json")
    if rag:
        print(f"[app] TF-IDF RAG 索引: {len(rag.docs)} 筆 Q&A")
    else:
        print("[app] 未找到 rag_index.json，純模型推理")

    gc.collect()
    return model, tokenizer, config, rag


# ==================== 推理邏輯 ====================

def build_prompt(message, history, tokenizer, config, rag=None):
    """組裝多輪對話 Prompt，RAG 檢索相關 Q&A 當作 few-shot 範例注入"""
    budget = max(config.block_size - config.max_new_tokens, 8)

    rag_context = ""
    if rag:
        rag_context = rag.retrieve_context(message, top_k=2)

    tail = f"問:{message}\n答:"
    used = len(tokenizer.encode(tail, add_special_tokens=False))

    if rag_context:
        rag_tokens = len(tokenizer.encode(rag_context, add_special_tokens=False))
        if used + rag_tokens < budget:
            used += rag_tokens
        else:
            rag_context = ""

    kept = []
    for user_msg, bot_msg in reversed(history or []):
        for text, tag in [(bot_msg, "答:"), (user_msg, "問:")]:
            if not text:
                continue
            piece = f"{tag}{text}\n"
            piece_len = len(tokenizer.encode(piece, add_special_tokens=False))
            if used + piece_len > budget:
                break
            kept.append(piece)
            used += piece_len
        else:
            continue
        break

    kept.reverse()
    return "".join(kept) + rag_context + tail


def generate_reply(message, history, model, tokenizer, config, rag):
    if rag:
        direct = rag.direct_answer(message, threshold=0.25)
        if direct:
            return direct

    prompt = build_prompt(message, history, tokenizer, config, rag)
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not input_ids:
        return "（無法編碼輸入）"

    idx = torch.tensor([input_ids], dtype=torch.long)
    out = model.generate(
        idx,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
        eos_id=tokenizer.sep_token_id,
    )

    full = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
    reply = full[len(prompt):] if full.startswith(prompt) else full
    return truncate_reply(reply) or "（模型未生成回覆）"


# ==================== Gradio 介面 ====================

def main():
    import gradio as gr

    print("[app] 正在載入模型...")
    model, tokenizer, config, rag = load_everything()
    print("[app] 模型就緒,啟動 Gradio API")

    def chat_fn(message, history):
        return generate_reply(message, history, model, tokenizer, config, rag)

    demo = gr.ChatInterface(
        fn=chat_fn,
        title="NEXUX AI · 345M 繁體中文大語言模型",
        description="純手寫自研 345M Transformer · TF-IDF RAG 即時知識檢索",
        examples=[
            "你好",
            "為什麼天空是藍色的？",
            "地震時怎麼辦？",
            "什麼是 Python 的 list？",
            "幫我想一個故事開頭",
        ],
        theme=gr.themes.Soft(),
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
