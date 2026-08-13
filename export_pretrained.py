"""
export_pretrained.py
----------------------
把微調好的預訓練模型(checkpoint_pretrained.pt)匯出成部署用格式。
使用 int8 仿射量化壓縮權重,大模型會自動拆成多個 npz 檔案
(每個 < 95MB,避免超過 GitHub 100MB 限制)。

輸出檔案:
    weights_meta_pretrained.json  架構等中繼資料(含 num_parts)
    weights_pretrained_0.npz      int8 量化權重(第 0 部分)
    weights_pretrained_1.npz      ...（大模型才會有多個 part）
"""

import json
import os

import numpy as np
import torch

_KEY_SEP_ORIGINAL = "."
_KEY_SEP_NPZ = "__"

MAX_PART_BYTES = 90 * 1024 * 1024  # 90MB per part


def _quantize_tensor(tensor: torch.Tensor) -> tuple[np.ndarray, float, float]:
    arr = tensor.numpy().astype(np.float32)
    qmin_val = float(arr.min())
    qmax_val = float(arr.max())
    scale = (qmax_val - qmin_val) / 255.0 if qmax_val > qmin_val else 1.0
    quantized = np.round((arr - qmin_val) / scale).astype(np.uint8)
    return quantized, qmin_val, scale


def _quantize_tensor_int4(tensor: torch.Tensor) -> tuple[np.ndarray, float, float]:
    arr = tensor.numpy().astype(np.float32).ravel()
    qmin_val = float(arr.min())
    qmax_val = float(arr.max())
    scale = (qmax_val - qmin_val) / 15.0 if qmax_val > qmin_val else 1.0
    q = np.clip(np.round((arr - qmin_val) / scale), 0, 15).astype(np.uint8)
    if len(q) % 2 != 0:
        q = np.append(q, np.uint8(0))
    packed = (q[0::2] << 4) | q[1::2]
    return packed, qmin_val, scale


def export_pretrained(
    checkpoint_path: str = "checkpoint_pretrained.pt",
    meta_path: str = "weights_meta_pretrained.json",
    npz_prefix: str = "weights_pretrained",
    use_int4: bool = False,
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"找不到 {checkpoint_path},請先執行「python run_pretrained_sft.py」完成微調。"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    arch = checkpoint["architecture"]

    all_arrays = {}
    qfunc = _quantize_tensor_int4 if use_int4 else _quantize_tensor
    if use_int4:
        print("[export_pretrained] 使用 int4 量化(適合大模型部署)")
    for name, tensor in state_dict.items():
        if "attn.mask" in name:
            continue
        key = name.replace(_KEY_SEP_ORIGINAL, _KEY_SEP_NPZ)
        quantized, qmin_val, scale = qfunc(tensor)
        all_arrays[key] = quantized
        all_arrays[f"{key}|qmin"] = np.float32(qmin_val)
        all_arrays[f"{key}|qscale"] = np.float32(scale)
        if use_int4:
            all_arrays[f"{key}|numel"] = np.int64(tensor.numel())

    parts = []
    current_part = {}
    current_size = 0

    for key, arr in all_arrays.items():
        arr_size = arr.nbytes
        if current_size + arr_size > MAX_PART_BYTES and current_part:
            parts.append(current_part)
            current_part = {}
            current_size = 0
        current_part[key] = arr
        current_size += arr_size

    if current_part:
        parts.append(current_part)

    old_files = [f for f in os.listdir(".") if f.startswith(npz_prefix) and f.endswith(".npz")]
    for f in old_files:
        os.remove(f)

    total_size = 0
    for i, part in enumerate(parts):
        part_path = f"{npz_prefix}_{i}.npz"
        np.savez_compressed(part_path, **part)
        part_size = os.path.getsize(part_path)
        total_size += part_size
        print(f"[export_pretrained] Part {i}: {part_path} ({part_size / 1024 / 1024:.1f} MB)")

    meta = {
        "config": {
            "vocab_size": checkpoint["vocab_size"],
            "n_embd": arch["n_embd"],
            "n_head": arch["n_head"],
            "n_layer": arch["n_layer"],
            "block_size": arch["block_size"],
        },
        "sft_applied": checkpoint.get("sft_applied", False),
        "quantized": True,
        "quant_bits": 4 if use_int4 else 8,
        "num_parts": len(parts),
        "npz_prefix": npz_prefix,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    print(f"[export_pretrained] 共 {len(parts)} 個 part,總計 {total_size / 1024 / 1024:.1f} MB")
    for i in range(len(parts)):
        part_path = f"{npz_prefix}_{i}.npz"
        sz = os.path.getsize(part_path) / (1024 * 1024)
        if sz >= 100:
            print(f"[export_pretrained] 警告: {part_path} ({sz:.1f} MB) 超過 GitHub 100MB 限制！")


if __name__ == "__main__":
    import sys
    export_pretrained(use_int4="--int4" in sys.argv)
