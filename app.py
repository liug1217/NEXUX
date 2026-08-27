"""
app.py — NEXUX AI 聊天機器人（Streamlit Cloud 部署版）
純推理模式,從 int8 量化 npz 權重載入,完整保留原創 GPTModel 架構。
不依賴本機訓練環境的任何模組,完全獨立運行。
"""

import gc
import json
import os
import re
from dataclasses import dataclass

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertTokenizer


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


# ==================== GPTModel 完整架構（原創自研 decoder-only Transformer） ====================

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
        assert T <= self.config.block_size, (
            f"輸入長度 {T} 超過 block_size {self.config.block_size}"
        )
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
    def generate_stream(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        top_p=None,
        repetition_penalty=1.0,
        eos_id=None,
    ):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if repetition_penalty != 1.0:
                for b in range(idx.size(0)):
                    seen_tokens = set(idx[b].tolist())
                    for token_id in seen_tokens:
                        if logits[b, token_id] > 0:
                            logits[b, token_id] /= repetition_penalty
                        else:
                            logits[b, token_id] *= repetition_penalty

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(
                    logits, descending=True, dim=-1
                )
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[
                    :, :-1
                ].clone()
                sorted_indices_to_remove[:, 0] = False
                for b in range(logits.size(0)):
                    indices_to_remove = sorted_indices[b][
                        sorted_indices_to_remove[b]
                    ]
                    logits[b, indices_to_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            yield next_id[0].tolist()

            if eos_id is not None and idx.size(0) == 1 and next_id.item() == eos_id:
                break


# ==================== int8 npz 權重反量化載入 ====================

def _dequant_one(npz_files, key, quant_bits):
    for npz in npz_files:
        if key not in npz.files:
            continue
        qmin_key = f"{key}|qmin"
        qscale_key = f"{key}|qscale"

        if qmin_key in npz.files and qscale_key in npz.files:
            qmin = float(npz[qmin_key])
            qscale = float(npz[qscale_key])
            packed = np.array(npz[key])

            if quant_bits == 4:
                numel_key = f"{key}|numel"
                shape_key = f"{key}|shape"
                numel = int(npz[numel_key])
                hi = packed >> 4
                lo = packed & 0x0F
                flat = np.empty(len(packed) * 2, dtype=np.uint8)
                flat[0::2] = hi
                flat[1::2] = lo
                del packed, hi, lo
                arr = flat[:numel].astype(np.float32) * qscale + qmin
                del flat
                if shape_key in npz.files:
                    shape = tuple(int(s) for s in npz[shape_key])
                    arr = arr.reshape(shape)
            else:
                arr = packed.astype(np.float32) * qscale + qmin
                del packed
        else:
            arr = npz[key].astype(np.float32)

        t = torch.from_numpy(arr).half()
        del arr
        return t
    return None


def _load_model_from_npz(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    cfg = meta["config"]
    base_dir = os.path.dirname(meta_path) or "."
    num_parts = meta.get("num_parts", 0)
    npz_prefix = meta.get("npz_prefix", "")
    quant_bits = meta.get("quant_bits", 8)

    if num_parts > 0 and npz_prefix:
        npz_files = [
            np.load(os.path.join(base_dir, f"{npz_prefix}_{i}.npz"))
            for i in range(num_parts)
        ]
    else:
        npz_files = [np.load(os.path.join(base_dir, "weights.npz"))]

    all_data_keys = set()
    for npz in npz_files:
        for k in npz.files:
            if "|" not in k:
                all_data_keys.add(k)

    config = InferenceConfig(
        block_size=cfg["block_size"],
        n_embd=cfg["n_embd"],
        n_head=cfg["n_head"],
        n_layer=cfg["n_layer"],
    )

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    model = GPTModel(config, cfg["vocab_size"])
    torch.set_default_dtype(prev_dtype)
    model.eval()
    gc.collect()

    has_head_weight = False
    with torch.no_grad():
        for npz_key in all_data_keys:
            real_key = npz_key.replace("__", ".")
            if real_key == "head.weight":
                has_head_weight = True
            tensor = _dequant_one(npz_files, npz_key, quant_bits)
            if tensor is None:
                continue

            parts = real_key.split(".")
            obj = model
            for part in parts[:-1]:
                if part.isdigit():
                    obj = obj[int(part)]
                else:
                    obj = getattr(obj, part)

            attr_name = parts[-1]
            param = getattr(obj, attr_name, None)
            if isinstance(param, nn.Parameter):
                param.data.copy_(tensor)
            elif isinstance(param, torch.Tensor):
                param.copy_(tensor)
            else:
                setattr(obj, attr_name, tensor)
            del tensor

        gc.collect()

    if not has_head_weight:
        model.head.weight.data.copy_(model.token_emb.weight.data)

    for npz in npz_files:
        npz.close()
    gc.collect()

    n_params = sum(p.numel() for p in model.parameters())
    return model, config, n_params


# ==================== 文字後處理（截斷模型幻想的下一輪對話） ====================

_TURN_PATTERN = re.compile(r"\nA[:：]|\nB[:：]|\n問[:：]|\n答[:：]")


def find_next_turn_marker(text):
    return _TURN_PATTERN.search(text)


# ==================== 多輪對話上下文組裝 ====================


def build_context_prompt(history, prompt, tokenizer, block_size, max_new_tokens):
    budget = max(block_size - max_new_tokens, 8)
    tail = f"問:{prompt}\n答:"
    used = len(tokenizer.encode(tail, add_special_tokens=False))
    kept = []
    for turn in reversed(history or []):
        text = (turn.get("content") or "").strip()
        if not text:
            continue
        tag = "問:" if turn.get("role") == "user" else "答:"
        piece = f"{tag}{text}\n"
        piece_len = len(tokenizer.encode(piece, add_special_tokens=False))
        if used + piece_len > budget:
            break
        kept.append(piece)
        used += piece_len
    kept.reverse()
    return "".join(kept) + tail


# ==================== 模型載入（Streamlit 快取,整個生命週期只載入一次） ====================


@st.cache_resource
def load_model():
    model, config, n_params = _load_model_from_npz("weights_meta_pretrained.json")
    tokenizer = BertTokenizer(
        vocab_file="vocab_pretrained.txt",
        do_lower_case=True,
        clean_up_tokenization_spaces=True,
    )
    return model, tokenizer, config, n_params


# ==================== 串流文字生成器（對接 st.write_stream） ====================


def generate_response(
    prompt, history, model, tokenizer, config,
    temperature, max_new_tokens, top_k, top_p, repetition_penalty,
):
    wrapped = build_context_prompt(
        history, prompt, tokenizer, config.block_size, max_new_tokens
    )
    input_ids = tokenizer.encode(wrapped, add_special_tokens=False)
    if not input_ids:
        yield "（無法編碼輸入,請換一句話試試）"
        return

    idx = torch.tensor([input_ids], dtype=torch.long)
    accumulated_ids = []
    sent_len = 0
    HOLD = 3

    for token_ids in model.generate_stream(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_id=tokenizer.sep_token_id,
    ):
        accumulated_ids.extend(token_ids)
        text = tokenizer.decode(accumulated_ids, skip_special_tokens=True)

        marker = find_next_turn_marker(text)
        if marker:
            final = text[: marker.start()].rstrip()
            if len(final) > sent_len:
                yield final[sent_len:]
            return

        safe_len = max(0, len(text) - HOLD)
        if safe_len > sent_len:
            yield text[sent_len:safe_len]
            sent_len = safe_len

    text = tokenizer.decode(accumulated_ids, skip_special_tokens=True).rstrip()
    if len(text) > sent_len:
        yield text[sent_len:]


# ==================== Streamlit 介面 ====================

st.set_page_config(page_title="NEXUX AI", page_icon="🤖", layout="centered")

st.title("🤖 NEXUX AI")
st.caption("純手寫自研架構 · 345M 繁體中文大語言模型")

# ---- 側邊欄：生成參數 ----
with st.sidebar:
    st.header("⚙️ 生成參數")
    temperature = st.slider(
        "Temperature（溫度）", 0.1, 2.0, 0.5, 0.1, help="越高越隨機,越低越確定"
    )
    max_tokens = st.slider("最大生成長度", 10, 300, 100, 10)
    top_p = st.slider("Top-P（核採樣）", 0.1, 1.0, 0.8, 0.05)
    top_k = st.slider("Top-K", 1, 100, 20, 1)
    rep_penalty = st.slider("重複懲罰", 1.0, 3.0, 2.0, 0.1)
    st.divider()
    if st.button("🗑️ 清除對話紀錄"):
        st.session_state.messages = []
        st.rerun()

# ---- 檢查權重檔案 ----
if not os.path.exists("weights_meta_pretrained.json"):
    st.error("找不到 `weights_meta_pretrained.json`,請確認權重檔案已在專案根目錄。")
    st.stop()
if not os.path.exists("vocab_pretrained.txt"):
    st.error("找不到 `vocab_pretrained.txt`,請確認詞表檔案已在專案根目錄。")
    st.stop()

# ---- 載入模型 ----
with st.spinner("首次載入模型中,請稍候..."):
    model, tokenizer, config, n_params = load_model()

with st.sidebar:
    st.divider()
    st.caption(f"模型參數量：{n_params / 1e6:.0f}M")
    st.caption(f"架構：{config.n_layer}L / {config.n_head}H / {config.n_embd}D")
    st.caption(f"上下文長度：{config.block_size} tokens")
    st.caption("推理精度：float16（記憶體優化）")

# ---- 初始化對話歷史 ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- 顯示歷史訊息 ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---- 接收輸入 & 串流生成回覆 ----
if prompt := st.chat_input("輸入你的問題..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(
            generate_response(
                prompt=prompt,
                history=st.session_state.messages[:-1],
                model=model,
                tokenizer=tokenizer,
                config=config,
                temperature=temperature,
                max_new_tokens=max_tokens,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=rep_penalty,
            )
        )

    st.session_state.messages.append({"role": "assistant", "content": response})
