"""
export_pretrained.py
----------------------
把微調好的預訓練模型(checkpoint_pretrained.pt)匯出成部署用格式,
邏輯跟 export_weights.py(char-level 模型用)類似,但這個模型有 102M
參數,float16 匯出後約 195~208MB,超過 GitHub 單檔 100MB 上限,所以改用
int8 仿射量化(每個 tensor 各自算 min/scale,量化成 0~255 的整數,推論時
再還原成浮點數),實測壓縮後約 86MB,在限制內。

量化品質已經驗證過(見 docs/MODEL_MIGRATION.md):對同一段輸入,量化前後
生成內容維持相近的語意品質,沒有明顯劣化。

輸出檔案:
    weights_meta_pretrained.json  架構等中繼資料
    weights_pretrained.npz        int8 量化權重 + 各 tensor 的還原參數
    vocab_pretrained.txt          BertWordpieceTokenizer 用的詞表(已存在,
                                   這裡不會重新產生,convert_pretrained.py
                                   已經輸出過)

用法:
    python export_pretrained.py
"""

import json
import os

import numpy as np
import torch

_KEY_SEP_ORIGINAL = "."
_KEY_SEP_NPZ = "__"


def _quantize_tensor(tensor: torch.Tensor) -> tuple[np.ndarray, float, float]:
    """
    仿射量化(affine quantization):把浮點數線性映射到 0~255 的整數。
    還原公式:原始值 ≈ 量化整數 * scale + qmin
    """
    arr = tensor.numpy().astype(np.float32)
    qmin_val = float(arr.min())
    qmax_val = float(arr.max())
    scale = (qmax_val - qmin_val) / 255.0 if qmax_val > qmin_val else 1.0
    quantized = np.round((arr - qmin_val) / scale).astype(np.uint8)
    return quantized, qmin_val, scale


def export_pretrained(
    checkpoint_path: str = "checkpoint_pretrained.pt",
    meta_path: str = "weights_meta_pretrained.json",
    npz_path: str = "weights_pretrained.npz",
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"找不到 {checkpoint_path},請先執行「python run_pretrained_sft.py」完成微調。"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    arch = checkpoint["architecture"]

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
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    arrays = {}
    for name, tensor in state_dict.items():
        if "attn.mask" in name:
            continue  # buffer,不是訓練出來的權重,不需要匯出,推論端會自動生成
        key = name.replace(_KEY_SEP_ORIGINAL, _KEY_SEP_NPZ)
        quantized, qmin_val, scale = _quantize_tensor(tensor)
        arrays[key] = quantized
        arrays[f"{key}|qmin"] = np.float32(qmin_val)
        arrays[f"{key}|qscale"] = np.float32(scale)

    np.savez_compressed(npz_path, **arrays)

    meta_size_kb = os.path.getsize(meta_path) / 1024
    npz_size_mb = os.path.getsize(npz_path) / (1024 * 1024)
    print(f"[export_pretrained] 已匯出 {meta_path}({meta_size_kb:.1f} KB)"
          f" 與 {npz_path}({npz_size_mb:.2f} MB,int8 量化)")
    if npz_size_mb >= 100:
        print("[export_pretrained] 警告:檔案仍超過 GitHub 單檔 100MB 限制,需要進一步分片或量化。")


if __name__ == "__main__":
    export_pretrained()
