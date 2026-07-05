"""
numpy_gpt.py
------------
這是 model.py 的「輕量版」,用純 numpy 重新實作跟 model.py 完全相同的
數學運算(embedding、多頭注意力、feedforward、layer norm),但不依賴 torch。

這個檔案只負責「推理」(根據已經訓練好的權重生成文字),不能拿來訓練,
因為沒有實作反向傳播。訓練還是要用 model.py + train.py(需要 torch)。

之所以要獨立寫一份,是因為 torch 這個套件太大,無法塞進 Vercel 的
Serverless Function 大小限制裡,而 numpy 小很多,適合拿來部署。
"""

import json
import numpy as np


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """對最後一個維度做 layer normalization,對應 torch 的 nn.LayerNorm。"""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    normed = (x - mean) / np.sqrt(var + eps)
    return normed * weight + bias


def gelu(x: np.ndarray) -> np.ndarray:
    """
    GELU 激活函數,使用 tanh 近似公式(跟 GPT-2 原始實作相同)。
    跟 torch 預設的精確版 GELU 會有極小誤差,但不影響生成效果,
    換來的好處是完全不需要 scipy,只靠 numpy 就能算。
    """
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)  # 數值穩定化
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class NumpyGPT:
    """
    純 numpy 版的 GPT 推理引擎。
    只支援 batch_size = 1 的生成(對聊天網頁來說已經足夠)。
    """

    def __init__(self, weights_path: str):
        with open(weights_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cfg = data["config"]
        self.vocab_size = cfg["vocab_size"]
        self.n_embd = cfg["n_embd"]
        self.n_head = cfg["n_head"]
        self.n_layer = cfg["n_layer"]
        self.block_size = cfg["block_size"]
        self.head_size = self.n_embd // self.n_head
        self.is_sft = data.get("sft_applied", False)

        # 把所有權重轉成 numpy array,方便後續矩陣運算
        self.w = {name: np.array(value, dtype=np.float64) for name, value in data["weights"].items()}

    def _linear(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
        """對應 torch 的 nn.Linear:y = x @ weight.T + bias"""
        out = x @ weight.T
        if bias is not None:
            out = out + bias
        return out

    def _attention(self, x: np.ndarray, layer: int) -> np.ndarray:
        T, C = x.shape
        prefix = f"blocks.{layer}.attn"

        qkv = self._linear(x, self.w[f"{prefix}.qkv_proj.weight"], self.w[f"{prefix}.qkv_proj.bias"])
        q, k, v = np.split(qkv, 3, axis=-1)  # 各自 (T, C)

        # 拆成多頭: (T, C) -> (n_head, T, head_size)
        def split_heads(t):
            return t.reshape(T, self.n_head, self.head_size).transpose(1, 0, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        att = (q @ k.transpose(0, 2, 1)) * (self.head_size ** -0.5)  # (n_head, T, T)

        # 因果遮罩:只能看到自己與之前的位置
        mask = np.tril(np.ones((T, T)))
        att = np.where(mask == 0, -np.inf, att)
        att = softmax(att, axis=-1)

        out = att @ v  # (n_head, T, head_size)
        out = out.transpose(1, 0, 2).reshape(T, C)  # 合併多頭

        return self._linear(out, self.w[f"{prefix}.out_proj.weight"], self.w[f"{prefix}.out_proj.bias"])

    def _feedforward(self, x: np.ndarray, layer: int) -> np.ndarray:
        prefix = f"blocks.{layer}.ff.net"
        h = self._linear(x, self.w[f"{prefix}.0.weight"], self.w[f"{prefix}.0.bias"])
        h = gelu(h)
        return self._linear(h, self.w[f"{prefix}.2.weight"], self.w[f"{prefix}.2.bias"])

    def forward(self, idx: list[int]) -> np.ndarray:
        """
        idx: 長度為 T 的 token id 列表(單一序列,不是 batch)。
        回傳: (T, vocab_size) 的 logits。
        """
        T = len(idx)
        assert T <= self.block_size, f"輸入長度 {T} 超過 block_size {self.block_size}"

        tok_emb = self.w["token_emb.weight"][idx]          # (T, C)
        pos_emb = self.w["pos_emb.weight"][:T]              # (T, C)
        x = tok_emb + pos_emb

        for layer in range(self.n_layer):
            ln1_out = layer_norm(x, self.w[f"blocks.{layer}.ln1.weight"], self.w[f"blocks.{layer}.ln1.bias"])
            x = x + self._attention(ln1_out, layer)

            ln2_out = layer_norm(x, self.w[f"blocks.{layer}.ln2.weight"], self.w[f"blocks.{layer}.ln2.bias"])
            x = x + self._feedforward(ln2_out, layer)

        x = layer_norm(x, self.w["ln_f.weight"], self.w["ln_f.bias"])
        logits = x @ self.w["head.weight"].T  # (T, vocab_size),head 沒有 bias
        return logits

    def generate(
        self,
        idx: list[int],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        seed: int | None = None,
    ) -> list[int]:
        """自迴歸生成,回傳完整序列(prompt + 新生成的 token)。"""
        rng = np.random.default_rng(seed)
        idx = list(idx)

        for _ in range(max_new_tokens):
            idx_cond = idx[-self.block_size:]
            logits = self.forward(idx_cond)
            last_logits = logits[-1] / max(temperature, 1e-5)

            if top_k is not None:
                top_k = min(top_k, last_logits.shape[-1])
                threshold = np.sort(last_logits)[-top_k]
                last_logits = np.where(last_logits < threshold, -np.inf, last_logits)

            probs = softmax(last_logits)
            next_id = rng.choice(len(probs), p=probs)
            idx.append(int(next_id))

        return idx
