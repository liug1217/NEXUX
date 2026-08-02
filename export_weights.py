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
import numpy as np
import torch

from config import Config
from tokenizer import CharTokenizer


def _array_to_json(arr: np.ndarray) -> str:
    """
    把 numpy 陣列轉成 JSON 陣列文字,數字只保留5位有效數字。

    json.dump() 預設會用 Python float 的完整雙精度表示法印出每個數字
    (動輒十幾位小數),但權重本來就是 float32 訓練出來的,只有大約7位
    有效數字才有意義,而且單純对 tensor 做四捨五入沒有用——四捨五入後的
    值一樣是個二進位浮點數,repr() 出來還是一樣長。真正能縮小檔案的方式
    是直接用字串格式化控制輸出的位數,而不是仰賴 Python 自動選出的
    「最短能還原原值」表示法。這樣可以讓 weights.json 縮小到原本的
    三分之一以下,留在 GitHub 單檔 100MB 的推送上限之內。
    """
    if arr.ndim == 1:
        return "[" + ",".join(np.char.mod("%.5g", arr)) + "]"
    return "[" + ",".join(_array_to_json(sub) for sub in arr) + "]"


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

    is_sft = checkpoint.get("sft_applied", False)
    if is_sft:
        print("[export_weights] 偵測到這是經過 SFT 微調的模型,會標記為問答模式")

    header = {
        "config": {
            "vocab_size": vocab_size,
            "n_embd": arch["n_embd"],
            "n_head": arch["n_head"],
            "n_layer": arch["n_layer"],
            "block_size": arch["block_size"],
        },
        "sft_applied": is_sft,
    }

    # weights 這部分刻意不透過 json.dump(),改用 _array_to_json() 手動控制
    # 每個數字的輸出精度(見該函式的說明),其餘結構(config、sft_applied)
    # 資料量很小,直接用標準 json.dumps() 沒有影響。
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(header)[:-1])  # 去掉結尾的 "}",接著手動補上 weights
        f.write(',"weights":{')
        f.write(",".join(
            f'{json.dumps(name)}:{_array_to_json(tensor.numpy())}'
            for name, tensor in state_dict.items()
        ))
        f.write("}}")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[export_weights] 已匯出至 {output_path}({size_mb:.2f} MB)")
    print("[export_weights] 接下來把這個檔案跟 tokenizer.json 一起 commit 上傳即可。")


if __name__ == "__main__":
    export_weights()
