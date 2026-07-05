"""
export_weights.py
------------------
這支程式只在「你自己的電腦」上執行,目的是把 train.py 訓練出來的
checkpoint.pt(torch 格式)轉換成一份單純的 JSON 檔案(weights.json)。

為什麼要多這一步?
因為 PyTorch(torch)這個套件本身非常大,直接把它整個裝進 Vercel 的
Serverless Function 會超過大小限制(目前實測整包超過 700MB,上限是 500MB)。

解法是:訓練的時候繼續用 torch(在自己電腦上,沒有大小限制),
但「部署到網路上」的推理部分改用 numpy 重新實作一次前向運算,
numpy 比 torch 小很多,才塞得進 Vercel 的限制裡。

使用方式:
    python train.py            # 先訓練出 checkpoint.pt
    python export_weights.py   # 再執行這支,會產生 weights.json
    把 weights.json 一起 commit 上傳到 GitHub(取代原本的 checkpoint.pt)
"""

import json
import os
import torch

from config import Config
from tokenizer import CharTokenizer


def export_weights(config: Config | None = None, output_path: str = "weights.json"):
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

    # 把每一個 tensor 轉成單純的巢狀 list,這樣才能存進 JSON
    weights = {name: tensor.tolist() for name, tensor in state_dict.items()}

    export_data = {
        "config": {
            "vocab_size": vocab_size,
            "n_embd": arch["n_embd"],
            "n_head": arch["n_head"],
            "n_layer": arch["n_layer"],
            "block_size": arch["block_size"],
        },
        "weights": weights,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[export_weights] 已匯出至 {output_path}({size_mb:.2f} MB)")
    print("[export_weights] 接下來把這個檔案跟 tokenizer.json 一起 commit 上傳即可。")


if __name__ == "__main__":
    export_weights()
