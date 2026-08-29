"""
app.py — NEXUX AI 聊天機器人（Streamlit Cloud 部署版）
從 Hugging Face Hub 自動下載權重,純 CPU 推理,完整保留原創 GPTModel 架構。
"""

import gc
import re
from dataclasses import dataclass

import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F
from huggingface_hub import hf_hub_download
from transformers import BertTokenizer

HF_REPO_ID = "liug1217/NEXUX"


# ==================== 推理設定（從 checkpoint 自動讀取架構） ====================

@dataclass
class InferenceConfig:
    block_size: int = 1024
    n_embd: int = 1024
    n_head: int = 16
    n_layer: int = 24
    dropout: float = 0.0
    max_new_tokens: int = 150
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.1


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
            idx_cond = idx[:, -self.config.block_size :]
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


# ==================== 從 Hugging Face Hub 下載並載入模型 ====================


@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename="model.pt")
        vocab_path = hf_hub_download(repo_id=HF_REPO_ID, filename="vocab.txt")
    except Exception:
        return None, None, None, 0, (
            "⏳ 權重孵化中！請等本地 33135 步完成後，"
            "將模型上傳至 HF 倉庫 liug1217/NEXUX。"
        )

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    arch = checkpoint.get("architecture", {})
    config = InferenceConfig(
        block_size=arch.get("block_size", 1024),
        n_embd=arch.get("n_embd", 1024),
        n_head=arch.get("n_head", 16),
        n_layer=arch.get("n_layer", 24),
    )
    vocab_size = checkpoint["vocab_size"]

    state_dict = checkpoint.pop("model_state_dict")
    del checkpoint
    gc.collect()

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    model = GPTModel(config, vocab_size=vocab_size)
    torch.set_default_dtype(prev_dtype)

    for key in list(state_dict.keys()):
        if state_dict[key].is_floating_point():
            state_dict[key] = state_dict[key].half()

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    del state_dict
    gc.collect()

    real_missing = [k for k in missing if "attn.mask" not in k]
    real_unexpected = [k for k in unexpected if "attn.mask" not in k]
    warn_msg = None
    if real_missing:
        warn_msg = f"權重缺失: {real_missing}"
    if real_unexpected:
        warn_msg = f"多餘權重: {real_unexpected}"

    model.eval()

    tokenizer = BertTokenizer(
        vocab_file=vocab_path,
        do_lower_case=True,
        clean_up_tokenization_spaces=True,
    )

    n_params = sum(p.numel() for p in model.parameters())
    return model, tokenizer, config, n_params, warn_msg


# ==================== 串流文字生成器（對接 st.write_stream） ====================


def generate_response(
    prompt, history, model, tokenizer, config,
    temperature, max_new_tokens, top_p, repetition_penalty,
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
        top_k=config.top_k,
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

# ---- 側邊欄：生成參數控制面板 ----
with st.sidebar:
    st.header("⚙️ 生成參數")
    temperature = st.slider(
        "🌡️ Temperature（創意度）",
        min_value=0.1, max_value=2.0, value=0.8, step=0.1,
        help="越高回答越有創意但可能離題,越低越保守精準",
    )
    top_p = st.slider(
        "🎯 Top-p（核採樣）",
        min_value=0.1, max_value=1.0, value=0.9, step=0.05,
        help="只從累積機率前 p% 的候選字中取樣",
    )
    rep_penalty = st.slider(
        "🔁 Repetition Penalty（重複懲罰）",
        min_value=1.0, max_value=3.0, value=1.1, step=0.1,
        help="大於 1.0 時降低重複字詞出現的機率",
    )
    max_tokens = st.slider(
        "📏 最大生成長度",
        min_value=10, max_value=300, value=150, step=10,
    )
    st.divider()
    if st.button("🗑️ 清除對話紀錄"):
        st.session_state.messages = []
        st.rerun()

# ---- 載入模型（首次自動從 HuggingFace 下載） ----
with st.spinner("🚀 首次啟動：正在從 Hugging Face 下載模型權重..."):
    model, tokenizer, config, n_params, status_msg = load_model()

if status_msg and model is None:
    st.warning(status_msg)
    st.info(
        "📦 上傳指令：\n\n"
        "```bash\n"
        "# 先剝離 optimizer state（2 GB → ~1.3 GB）\n"
        "python -c \"\n"
        "import torch\n"
        "ckpt = torch.load('checkpoint_pretrained.pt', map_location='cpu')\n"
        "torch.save({\n"
        "    'model_state_dict': ckpt['model_state_dict'],\n"
        "    'vocab_size': ckpt['vocab_size'],\n"
        "    'architecture': ckpt['architecture'],\n"
        "    'sft_applied': ckpt.get('sft_applied', False),\n"
        "}, 'model.pt')\n"
        "\"\n"
        "# 上傳到 HuggingFace\n"
        "huggingface-cli upload liug1217/NEXUX model.pt\n"
        "huggingface-cli upload liug1217/NEXUX vocab_pretrained.txt vocab.txt\n"
        "```"
    )
    st.stop()

if status_msg:
    st.warning(status_msg)

# ---- API 模式：網址列 ?prompt=... 直接推理回傳 ----
import urllib.parse

query_params = st.query_params
if "prompt" in query_params:
    target_prompt = urllib.parse.unquote(query_params["prompt"])
    reply_chunks = list(generate_response(
        prompt=target_prompt,
        history=[],
        model=model,
        tokenizer=tokenizer,
        config=config,
        temperature=0.8,
        max_new_tokens=150,
        top_p=0.9,
        repetition_penalty=1.1,
    ))
    full_reply = "".join(reply_chunks)
    st.json({"status": "success", "prompt": target_prompt, "reply": full_reply})
    st.stop()

# ---- 側邊欄：模型資訊 ----
with st.sidebar:
    st.divider()
    st.caption(f"📊 模型參數量：{n_params / 1e6:.0f}M")
    st.caption(f"🏗️ 架構：{config.n_layer}L / {config.n_head}H / {config.n_embd}D")
    st.caption(f"📐 上下文長度：{config.block_size} tokens")
    st.caption("⚡ 推理精度：float16（記憶體優化）")
    st.caption(f"🤗 權重來源：[{HF_REPO_ID}](https://huggingface.co/{HF_REPO_ID})")

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
                top_p=top_p,
                repetition_penalty=rep_penalty,
            )
        )

    st.session_state.messages.append({"role": "assistant", "content": response})
