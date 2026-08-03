"""
export_weights.py
------------------
這支程式只在「你自己的電腦」上執行,目的是把 train.py 訓練出來的
checkpoint.pt(torch 格式)轉換成 Vercel 上的 numpy 推理引擎(numpy_gpt.py)
看得懂、體積又夠小的格式:weights_meta.json(架構等中繼資料,檔案很小)
+ weights.npz(實際權重數字,壓縮二進位格式)。

為什麼要多這一步?
因為 PyTorch(torch)這個套件本身非常大,直接把它整個裝進 Vercel 的
Serverless Function 會超過大小限制(目前實測整包超過 700MB,上限是 500MB)。

解法是:訓練的時候繼續用 torch(在自己電腦上,沒有大小限制),
但「部署到網路上」的推理部分改用 numpy 重新實作一次前向運算,
numpy 比 torch 小很多,才塞得進 Vercel 的限制裡。

原本是輸出成單一的 weights.json(數字用文字格式,5位有效數字),
但架構放大後(19.76M參數)文字格式膨脹到 184MB,超過 GitHub 單檔
100MB 的硬性推送上限(不是警告,是直接拒絕)。改成 numpy 原生的
.npz(壓縮二進位)格式儲存 float16 數字,同樣參數量下檔案大小只有
文字格式的四分之一到五分之一左右,才能繼續留在 100MB 以內推送。

使用方式:
    python train.py            # 先訓練出 checkpoint.pt
    python export_weights.py   # 再執行這支,會產生 weights_meta.json + weights.npz
    把這兩個檔案一起跟 tokenizer.json commit 上傳到 GitHub
"""

import json
import os
import numpy as np
import torch

from config import Config
from tokenizer import CharTokenizer

# npz 的 key 不能用 numpy.savez 的關鍵字引數語法帶點號(容易誤解成語法限制,
# 這裡直接用明確的字典介面繞開疑慮),統一用 "__" 取代原本 state_dict 名稱裡
# 的 ".",讀取時再換回來,避免任何因為特殊字元造成的相容性疑慮。
_KEY_SEP_ORIGINAL = "."
_KEY_SEP_NPZ = "__"


def export_weights(
    config: Config | None = None,
    meta_path: str = "weights_meta.json",
    npz_path: str = "weights.npz",
):
    config = config or Config()

    if not os.path.exists(config.checkpoint_path):
        raise FileNotFoundError(
            f"找不到 {config.checkpoint_path},請先執行「python train.py」訓練模型。"
        )
    if not os.path.exists(config.tokenizer_path):
        raise FileNotFoundError(
            f"找不到 {config.tokenizer_path},請先執行「python train.py」訓練模型。"
        )

    checkpoint = torch.load(config.checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    vocab_size = checkpoint["vocab_size"]

    # 優先使用 checkpoint 裡實際記錄的架構參數(新版 train.py 才會有這個欄位)。
    # 如果是舊的 checkpoint 沒有這個欄位,才退回去讀 config.py 當下的設定
    # (但這樣有風險:如果訓練時用的參數跟 config.py 現在的設定不一樣,
    # 匯出的架構就會跟實際權重尺寸不一致,導致載入時 reshape 出錯)。
    if "architecture" in checkpoint:
        arch = checkpoint["architecture"]
        print("[export_weights] 使用 checkpoint 裡記錄的架構參數(較安全)")
    else:
        arch = {
            "n_embd": config.n_embd,
            "n_head": config.n_head,
            "n_layer": config.n_layer,
            "block_size": config.block_size,
        }
        print(
            "[export_weights] 警告:這是舊版 checkpoint,沒有記錄架構參數,"
            "改用 config.py 目前的設定。如果訓練時的參數跟現在 config.py 不一致,"
            "匯出結果可能會有問題,建議重新執行一次 train.py 產生新版 checkpoint。"
        )

    is_sft = checkpoint.get("sft_applied", False)
    if is_sft:
        print("[export_weights] 偵測到這是經過 SFT 微調的模型,會標記為問答模式")

    meta = {
        "config": {
            "vocab_size": vocab_size,
            "n_embd": arch["n_embd"],
            "n_head": arch["n_head"],
            "n_layer": arch["n_layer"],
            "block_size": arch["block_size"],
        },
        "sft_applied": is_sft,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    # float16 儲存:原本文字格式保留5位有效數字,float16 大約只有3~4位
    # 有效數字,精度略降,但對這種小型字元級模型的生成效果影響不到能察覺
    # 的程度(推論時計算過程還是會轉回 float64,只有「存檔的精度」變粗)。
    arrays = {
        name.replace(_KEY_SEP_ORIGINAL, _KEY_SEP_NPZ): tensor.numpy().astype(np.float16)
        for name, tensor in state_dict.items()
    }
    np.savez_compressed(npz_path, **arrays)

    meta_size_kb = os.path.getsize(meta_path) / 1024
    npz_size_mb = os.path.getsize(npz_path) / (1024 * 1024)
    print(f"[export_weights] 已匯出 {meta_path}({meta_size_kb:.1f} KB)"
          f" 與 {npz_path}({npz_size_mb:.2f} MB)")
    print("[export_weights] 接下來把這兩個檔案跟 tokenizer.json 一起 commit 上傳即可。")


if __name__ == "__main__":
    export_weights()
