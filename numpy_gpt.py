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
import os
import numpy as np

# 對應 export_weights.py:npz 的 key 用 "__" 取代原本 state_dict 名稱裡的 "."。
_KEY_SEP_ORIGINAL = "."
_KEY_SEP_NPZ = "__"


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

    def __init__(self, meta_path: str, npz_filename: str = "weights.npz"):
        """
        meta_path: weights_meta.json 的路徑,實際權重從同目錄的 npz 讀取。
        支援單檔(weights.npz)和多檔(weights_pretrained_0.npz, _1.npz, ...)。
        """
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cfg = data["config"]
        self.vocab_size = cfg["vocab_size"]
        self.n_embd = cfg["n_embd"]
        self.n_head = cfg["n_head"]
        self.n_layer = cfg["n_layer"]
        self.block_size = cfg["block_size"]
        self.head_size = self.n_embd // self.n_head
        self.is_sft = data.get("sft_applied", False)

        base_dir = os.path.dirname(meta_path)
        num_parts = data.get("num_parts", 0)
        npz_prefix = data.get("npz_prefix", "")

        if num_parts > 0 and npz_prefix:
            npz_files_list = [
                np.load(os.path.join(base_dir, f"{npz_prefix}_{i}.npz"))
                for i in range(num_parts)
            ]
        else:
            npz_files_list = [np.load(os.path.join(base_dir, npz_filename))]

        quant_bits = data.get("quant_bits", 8)

        self.w = {}
        for npz in npz_files_list:
            meta_keys = {k for k in npz.files if "|" in k}
            data_keys = [k for k in npz.files if k not in meta_keys]

            for key in data_keys:
                qmin_key, qscale_key = f"{key}|qmin", f"{key}|qscale"
                numel_key = f"{key}|numel"
                if qmin_key in npz.files and qscale_key in npz.files:
                    qmin = float(npz[qmin_key])
                    qscale = float(npz[qscale_key])
                    if quant_bits == 4 and numel_key in npz.files:
                        numel = int(npz[numel_key])
                        packed = npz[key]
                        hi = (packed >> 4).astype(np.float64)
                        lo = (packed & 0x0F).astype(np.float64)
                        flat = np.empty(len(packed) * 2, dtype=np.float64)
                        flat[0::2] = hi
                        flat[1::2] = lo
                        value = flat[:numel] * qscale + qmin
                        shape_key = f"{key}|shape"
                        if shape_key in npz.files:
                            value = value.reshape(npz[shape_key])
                    else:
                        value = npz[key].astype(np.float64) * qscale + qmin
                else:
                    value = npz[key].astype(np.float64)
                self.w[key.replace(_KEY_SEP_NPZ, _KEY_SEP_ORIGINAL)] = value

    def _linear(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
        """對應 torch 的 nn.Linear:y = x @ weight.T + bias"""
        out = x @ weight.T
        if bias is not None:
            out = out + bias
        return out

    def _attention(self, x: np.ndarray, layer: int, cache_entry: dict | None = None) -> np.ndarray:
        """
        cache_entry: 選填。如果有傳,算完這批位置的 k/v 後會存進
        cache_entry["k"]/["v"](形狀 (n_head, T, head_size)),之後
        generate() 產生新 token 時,_attention_step() 才能接續使用這份
        快取,不用每次都從頭重算整段 prompt。只有第一次處理完整 prompt
        時才需要傳這個參數。
        """
        T, C = x.shape
        prefix = f"blocks.{layer}.attn"

        qkv = self._linear(x, self.w[f"{prefix}.qkv_proj.weight"], self.w[f"{prefix}.qkv_proj.bias"])
        q, k, v = np.split(qkv, 3, axis=-1)  # 各自 (T, C)

        # 拆成多頭: (T, C) -> (n_head, T, head_size)
        def split_heads(t):
            return t.reshape(T, self.n_head, self.head_size).transpose(1, 0, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        if cache_entry is not None:
            cache_entry["k"], cache_entry["v"] = k, v

        att = (q @ k.transpose(0, 2, 1)) * (self.head_size ** -0.5)  # (n_head, T, T)

        # 因果遮罩:只能看到自己與之前的位置
        mask = np.tril(np.ones((T, T)))
        att = np.where(mask == 0, -np.inf, att)
        att = softmax(att, axis=-1)

        out = att @ v  # (n_head, T, head_size)
        out = out.transpose(1, 0, 2).reshape(T, C)  # 合併多頭

        return self._linear(out, self.w[f"{prefix}.out_proj.weight"], self.w[f"{prefix}.out_proj.bias"])

    def _attention_step(self, x_new: np.ndarray, layer: int, cache_entry: dict) -> np.ndarray:
        """
        KV-cache 版的單一新 token 注意力計算。x_new 是 (1, C),只有這一個
        新位置。跟 _attention() 算出來的結果在數學上完全等價,差別只在於
        不用重算前面已經算過的位置——cache_entry 存著前面每一層累積下來的
        k、v(形狀 (n_head, T_so_far, head_size)),這裡只需要算「這個新
        token 的 q/k/v」,把新的 k/v 接到快取後面,再讓新的 q 對「快取裡
        全部的 k/v(含這個新 token 自己)」做注意力,不需要因果遮罩
        (反正這裡只有一個新位置,天生就只看得到自己跟之前的位置)。
        """
        prefix = f"blocks.{layer}.attn"
        qkv = self._linear(x_new, self.w[f"{prefix}.qkv_proj.weight"], self.w[f"{prefix}.qkv_proj.bias"])
        q, k, v = np.split(qkv, 3, axis=-1)  # 各自 (1, C)

        def split_heads_new(t):
            return t.reshape(1, self.n_head, self.head_size).transpose(1, 0, 2)  # (n_head, 1, head_size)

        q, k_new, v_new = split_heads_new(q), split_heads_new(k), split_heads_new(v)

        if cache_entry["k"] is None:
            k_all, v_all = k_new, v_new
        else:
            k_all = np.concatenate([cache_entry["k"], k_new], axis=1)  # (n_head, T_so_far+1, head_size)
            v_all = np.concatenate([cache_entry["v"], v_new], axis=1)
        cache_entry["k"], cache_entry["v"] = k_all, v_all

        att = (q @ k_all.transpose(0, 2, 1)) * (self.head_size ** -0.5)  # (n_head, 1, T_so_far+1)
        att = softmax(att, axis=-1)
        out = att @ v_all  # (n_head, 1, head_size)
        out = out.transpose(1, 0, 2).reshape(1, self.n_embd)  # 合併多頭 -> (1, C)

        return self._linear(out, self.w[f"{prefix}.out_proj.weight"], self.w[f"{prefix}.out_proj.bias"])

    def _feedforward(self, x: np.ndarray, layer: int) -> np.ndarray:
        prefix = f"blocks.{layer}.ff.net"
        h = self._linear(x, self.w[f"{prefix}.0.weight"], self.w[f"{prefix}.0.bias"])
        h = gelu(h)
        return self._linear(h, self.w[f"{prefix}.2.weight"], self.w[f"{prefix}.2.bias"])

    def forward(self, idx: list[int], cache: list[dict] | None = None) -> np.ndarray:
        """
        idx: 長度為 T 的 token id 列表(單一序列,不是 batch)。
        cache: 選填,傳入一個長度 n_layer、每個元素是 {"k": None, "v": None}
        的 list,呼叫完之後會被填入這批位置算出來的 k/v,供後續
        forward_step() 累加使用(見 generate() 怎麼用這個機制)。
        回傳: (T, vocab_size) 的 logits。
        """
        T = len(idx)
        assert T <= self.block_size, f"輸入長度 {T} 超過 block_size {self.block_size}"

        tok_emb = self.w["token_emb.weight"][idx]          # (T, C)
        pos_emb = self.w["pos_emb.weight"][:T]              # (T, C)
        x = tok_emb + pos_emb

        for layer in range(self.n_layer):
            ln1_out = layer_norm(x, self.w[f"blocks.{layer}.ln1.weight"], self.w[f"blocks.{layer}.ln1.bias"])
            cache_entry = cache[layer] if cache is not None else None
            x = x + self._attention(ln1_out, layer, cache_entry=cache_entry)

            ln2_out = layer_norm(x, self.w[f"blocks.{layer}.ln2.weight"], self.w[f"blocks.{layer}.ln2.bias"])
            x = x + self._feedforward(ln2_out, layer)

        x = layer_norm(x, self.w["ln_f.weight"], self.w["ln_f.bias"])
        # 預訓練模型(如 ckiplab/gpt2-base-chinese)輸出層跟輸入 embedding 是
        # tied(共用同一份權重),匯出時故意不重複存 head.weight 省空間,
        # 這裡找不到就退回用 token_emb.weight。
        head_weight = self.w.get("head.weight", self.w["token_emb.weight"])
        logits = x @ head_weight.T  # (T, vocab_size),head 沒有 bias
        return logits

    def forward_step(self, token_id: int, pos: int, cache: list[dict]) -> np.ndarray:
        """
        KV-cache 版的單步前向運算:只處理「一個新 token」,搭配已經累積好
        k/v 的 cache(第一次呼叫前,cache 必須先用 forward(idx, cache=...)
        處理過 prompt),讓 generate() 產生第二個以後的新 token 時,不用
        每次都把整段已生成的文字重新算一遍,大幅減少重複運算量。

        pos: 這個新 token 在整個序列裡的位置索引(從 0 算起),決定要用
             哪一個 position embedding。
        回傳: (vocab_size,) 這一個新位置的 logits。
        """
        assert pos < self.block_size, f"位置 {pos} 超過 block_size {self.block_size}"

        tok_emb = self.w["token_emb.weight"][[token_id]]   # (1, C)
        pos_emb = self.w["pos_emb.weight"][pos:pos + 1]     # (1, C)
        x = tok_emb + pos_emb

        for layer in range(self.n_layer):
            ln1_out = layer_norm(x, self.w[f"blocks.{layer}.ln1.weight"], self.w[f"blocks.{layer}.ln1.bias"])
            x = x + self._attention_step(ln1_out, layer, cache[layer])

            ln2_out = layer_norm(x, self.w[f"blocks.{layer}.ln2.weight"], self.w[f"blocks.{layer}.ln2.bias"])
            x = x + self._feedforward(ln2_out, layer)

        x = layer_norm(x, self.w["ln_f.weight"], self.w["ln_f.bias"])
        head_weight = self.w.get("head.weight", self.w["token_emb.weight"])
        logits = x @ head_weight.T  # (1, vocab_size)
        return logits[0]

    def generate(
        self,
        idx: list[int],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        seed: int | None = None,
        eos_id: int | None = None,
    ) -> list[int]:
        """
        自迴歸生成,回傳完整序列(prompt + 新生成的 token)。

        top_p: 核採樣(nucleus sampling),只保留累積機率達到 top_p 的最小候選集合。
        repetition_penalty: 大於 1.0 時,會降低「已經出現過的 token」被再次選中的
                             機率,減少連續重複同一個字或符號的情況。
        eos_id: 結束符號的 token id,生成到這個 id 就提早停止,不用生成滿
                max_new_tokens。預設 None(不啟用),char-level 模型沒有
                這個概念時維持原本行為。
        """
        rng = np.random.default_rng(seed)
        idx = list(idx)

        # KV-cache:prompt 這段用完整的 forward() 算一次、順便把每一層的
        # k/v 存進 cache,之後每多生成一個新 token,只需要呼叫
        # forward_step() 算「這一個新位置」,不用把已經生成的內容重新算
        # 一遍。這個模型有 12 層、768 維,沒有這個機制的話,生成速度會隨著
        # 已生成長度增加而越來越慢(每個字都要重算整段歷史),實測一個字
        # 要 2 秒以上,加了快取後單步只需算「新增的這一小段」,能大幅縮短
        # 生成時間,是能不能部署上線的關鍵。
        idx_cond = idx[-self.block_size:]
        cache = [{"k": None, "v": None} for _ in range(self.n_layer)]
        logits = self.forward(idx_cond, cache=cache)
        last_logits = logits[-1]
        next_pos = len(idx_cond)  # 下一個要生成的 token,在整個序列裡的位置索引

        for step in range(max_new_tokens):
            if step > 0:
                # 第一輪的 logits 已經在迴圈外、處理 prompt 時順便算好了;
                # 第二輪開始,每輪都要先用「上一輪選出的 token」算出這一輪
                # 的 logits(只算這一個新位置,靠 cache 避免重算歷史)。
                last_logits = self.forward_step(idx[-1], next_pos - 1, cache)

            scaled_logits = last_logits / max(temperature, 1e-5)

            # ---- 重複懲罰 ----
            if repetition_penalty != 1.0:
                for token_id in set(idx):
                    if scaled_logits[token_id] > 0:
                        scaled_logits[token_id] /= repetition_penalty
                    else:
                        scaled_logits[token_id] *= repetition_penalty

            # ---- top_k ----
            if top_k is not None:
                k = min(top_k, scaled_logits.shape[-1])
                threshold = np.sort(scaled_logits)[-k]
                scaled_logits = np.where(scaled_logits < threshold, -np.inf, scaled_logits)

            # ---- top_p(核採樣) ----
            if top_p is not None:
                sorted_idx = np.argsort(scaled_logits)[::-1]
                sorted_logits = scaled_logits[sorted_idx]
                sorted_probs = softmax(sorted_logits)
                cumulative_probs = np.cumsum(sorted_probs)

                # 找出累積機率超過 top_p 的位置,把這些位置之後的候選都排除
                remove_mask = cumulative_probs > top_p
                # 保留第一個超過門檻的候選,避免完全沒有候選可選
                remove_mask[1:] = remove_mask[:-1].copy()
                remove_mask[0] = False

                indices_to_remove = sorted_idx[remove_mask]
                scaled_logits[indices_to_remove] = -np.inf

            probs = softmax(scaled_logits)
            next_id = rng.choice(len(probs), p=probs)
            idx.append(int(next_id))
            next_pos += 1

            if eos_id is not None and int(next_id) == eos_id:
                break

        return idx

    def generate_stream(
        self,
        idx: list[int],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        seed: int | None = None,
        eos_id: int | None = None,
    ):
        """
        跟 generate() 邏輯完全一樣(同樣靠 KV-cache 加速,不影響生成速度),
        差別只在於用 yield 把「每一個新產生的 token id」逐一吐出來,而不是
        等全部生成完才一次回傳完整序列。給 api/generate.py 做串流回應、
        即時回報目前已生成的 token 數量用。
        """
        rng = np.random.default_rng(seed)
        idx = list(idx)

        idx_cond = idx[-self.block_size:]
        cache = [{"k": None, "v": None} for _ in range(self.n_layer)]
        logits = self.forward(idx_cond, cache=cache)
        last_logits = logits[-1]
        next_pos = len(idx_cond)

        for step in range(max_new_tokens):
            if step > 0:
                last_logits = self.forward_step(idx[-1], next_pos - 1, cache)

            scaled_logits = last_logits / max(temperature, 1e-5)

            if repetition_penalty != 1.0:
                for token_id in set(idx):
                    if scaled_logits[token_id] > 0:
                        scaled_logits[token_id] /= repetition_penalty
                    else:
                        scaled_logits[token_id] *= repetition_penalty

            if top_k is not None:
                k = min(top_k, scaled_logits.shape[-1])
                threshold = np.sort(scaled_logits)[-k]
                scaled_logits = np.where(scaled_logits < threshold, -np.inf, scaled_logits)

            if top_p is not None:
                sorted_idx = np.argsort(scaled_logits)[::-1]
                sorted_logits = scaled_logits[sorted_idx]
                sorted_probs = softmax(sorted_logits)
                cumulative_probs = np.cumsum(sorted_probs)

                remove_mask = cumulative_probs > top_p
                remove_mask[1:] = remove_mask[:-1].copy()
                remove_mask[0] = False

                indices_to_remove = sorted_idx[remove_mask]
                scaled_logits[indices_to_remove] = -np.inf

            probs = softmax(scaled_logits)
            next_id = int(rng.choice(len(probs), p=probs))
            idx.append(next_id)
            next_pos += 1

            yield next_id

            if eos_id is not None and next_id == eos_id:
                break
    