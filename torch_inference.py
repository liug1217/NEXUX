"""
torch_inference.py
------------------
Railway 專用的 PyTorch 推理引擎。從 int4/int8 npz 權重反量化後載入
model.py 的 GPTModel（float16），用 PyTorch 做推理。

記憶體優化：
- 直接以 float16 初始化模型（避免 float32→float16 轉換的雙倍峰值）
- 逐個權重載入，立刻釋放暫存
- 總峰值 ~850MB
"""

import json
import os
import gc

import numpy as np
import torch

_KEY_SEP_NPZ = "__"
_KEY_SEP_ORIGINAL = "."


def _dequant_one(npz_files, key, quant_bits):
    for npz in npz_files:
        if key not in npz.files:
            continue
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


def load_model(meta_path: str):
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

    from config import Config
    from model import GPTModel

    config = Config()
    config.n_embd = cfg["n_embd"]
    config.n_head = cfg["n_head"]
    config.n_layer = cfg["n_layer"]
    config.block_size = cfg["block_size"]
    config.dropout = 0.0

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    model = GPTModel(config, cfg["vocab_size"])
    torch.set_default_dtype(prev_dtype)
    model.eval()

    gc.collect()

    has_head_weight = False
    with torch.no_grad():
        for npz_key in all_data_keys:
            real_key = npz_key.replace(_KEY_SEP_NPZ, _KEY_SEP_ORIGINAL)
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
            if isinstance(param, torch.nn.Parameter):
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

    info = {
        "n_layer": cfg["n_layer"],
        "n_embd": cfg["n_embd"],
        "block_size": cfg["block_size"],
        "is_sft": meta.get("sft_applied", False),
    }
    return model, info
