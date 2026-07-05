"""
config.py
---------
集中管理所有超參數與路徑設定。
之後想調整模型大小、訓練時間、學習率等,都只需要改這個檔案。
"""

from dataclasses import dataclass
import torch


@dataclass
class Config:
    # ------- 資料路徑 -------
    data_dir: str = "data"               # 語料資料夾,底下所有 .txt 檔案都會被讀取並合併
    checkpoint_path: str = "checkpoint.pt"  # 模型權重存放位置
    tokenizer_path: str = "tokenizer.json"  # 詞表存放位置

    # ------- 資料切分 -------
    train_split: float = 0.9   # 90% 訓練 / 10% 驗證
    block_size: int = 64      # 模型一次看多長的文字(context length)

    # ------- 模型架構 -------
    n_embd: int = 64       # embedding 維度
    n_head: int = 4        # attention head 數量
    n_layer: int = 4        # transformer block 層數
    dropout: float = 0.1

    # ------- 訓練超參數 -------
    batch_size: int = 32
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5   # cosine 衰減後的最低學習率
    warmup_iters: int = 100           # 前 N 步線性 warmup,避免一開始梯度過大
    max_iters: int = 2000
    weight_decay: float = 0.01        # AdamW 的權重衰減,抑制過擬合
    grad_clip: float = 1.0            # 梯度裁剪上限,避免梯度爆炸(0 表示不裁剪)
    eval_interval: int = 100   # 每多少步驟評估一次
    eval_iters: int = 10       # 評估時取多少個 batch 平均

    # ------- Checkpoint / 續訓練 -------
    resume: bool = False   # True 時會從 checkpoint_path 載入既有權重,接續訓練
    save_interval: int = 500   # 每多少步驟額外存一次 checkpoint(避免中斷後全部重來)

    # ------- 推理 (inference) 參數 -------
    max_new_tokens: int = 300
    temperature: float = 0.8
    top_k: int = 50

    # ------- 裝置 -------
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1337


config = Config()
