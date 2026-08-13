"""
torch_inference.py
------------------
Railway 專用的 PyTorch 推理引擎。從 int4/int8 npz 權重反量化後載入
model.py 的 GPTModel，用 PyTorch 做推理——即使在純 CPU 上也比 numpy
快很多倍（PyTorch 內建 optimized BLAS + 多線程矩陣運算）。

Vercel 因為套件大小限制不能裝 torch，所以 Vercel 繼續用 numpy_gpt.py；
Railway 是 Docker 容器沒有限制，可以裝完整的 torch。
"""

import json
import os

import numpy as np
import torch

from model import GPTModel
from config import Config

_KEY_SEP_NPZ = "__"
_KEY_SEP_ORIGINAL = "."


def _load_npz_to_state_dict(meta_path: str):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    cfg = meta["config"]
    base_dir = os.path.dirname(meta_path)
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

    state_dict = {}
    for npz in npz_files:
        meta_keys = {k for k in npz.files if "|" in k}
        data_keys = [k for k in npz.files if k not in meta_keys]

        for key in data_keys:
            real_key = key.replace(_KEY_SEP_NPZ, _KEY_SEP_ORIGINAL)
            qmin_key = f"{key}|qmin"
            qscale_key = f"{key}|qscale"
            numel_key = f"{key}|numel"
            shape_key = f"{key}|shape"

            if qmin_key in npz.files and qscale_key in npz.files:
                qmin = float(npz[qmin_key])
                qscale = float(npz[qscale_key])
                packed = np.array(npz[key])

                if quant_bits == 4 and numel_key in npz.files:
                    numel = int(npz[numel_key])
                    hi = packed >> 4
                    lo = packed & 0x0F
                    flat = np.empty(len(packed) * 2, dtype=np.uint8)
                    flat[0::2] = hi
                    flat[1::2] = lo
                    arr = flat[:numel].astype(np.float32) * qscale + qmin
                    if shape_key in npz.files:
                        shape = tuple(int(s) for s in npz[shape_key])
                        arr = arr.reshape(shape)
                else:
                    arr = packed.astype(np.float32) * qscale + qmin
            else:
                arr = npz[key].astype(np.float32)

            state_dict[real_key] = torch.from_numpy(arr)

    return state_dict, cfg, meta


def load_model(meta_path: str) -> tuple[GPTModel, dict]:
    state_dict, cfg, meta = _load_npz_to_state_dict(meta_path)

    config = Config()
    config.vocab_size = cfg["vocab_size"]
    config.n_embd = cfg["n_embd"]
    config.n_head = cfg["n_head"]
    config.n_layer = cfg["n_layer"]
    config.block_size = cfg["block_size"]
    config.dropout = 0.0

    model = GPTModel(config, cfg["vocab_size"])

    if "head.weight" not in state_dict:
        state_dict["head.weight"] = state_dict["token_emb.weight"].clone()

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    info = {
        "n_layer": cfg["n_layer"],
        "n_embd": cfg["n_embd"],
        "block_size": cfg["block_size"],
        "is_sft": meta.get("sft_applied", False),
    }
    return model, info
