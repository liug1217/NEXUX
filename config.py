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
    block_size: int = 96      # 模型一次看多長的文字(context length)

    # ------- 模型架構 -------
    n_embd: int = 128       # embedding 維度
    n_head: int = 8        # attention head 數量
    n_layer: int = 6        # transformer block 層數
    dropout: float = 0.1

    # ------- 訓練超參數 -------
    batch_size: int = 32
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5   # cosine 衰減後的最低學習率
    warmup_iters: int = 200           # 前 N 步線性 warmup,避免一開始梯度過大
    max_iters: int = 5000      # 語料量變大,從頭訓練,步數比原本4000略增
    weight_decay: float = 0.01        # AdamW 的權重衰減,抑制過擬合
    grad_clip: float = 1.0            # 梯度裁剪上限,避免梯度爆炸(0 表示不裁剪)
    eval_interval: int = 200   # 每多少步驟評估一次
    eval_iters: int = 10       # 評估時取多少個 batch 平均

    # ------- Checkpoint / 續訓練 -------
    resume: bool = False   # 這次新增語料引入了253個原本詞表沒有的新字,詞表必須重建,
                            # 導致 embedding 維度對不上舊 checkpoint,無法沿用續訓,只能從頭重新訓練
    save_interval: int = 500   # 每多少步驟額外存一次 checkpoint(避免中斷後全部重來)

    # ------- SFT(問答微調)設定 -------
    # SFT 是在 train.py 純接龍訓練完成之後,額外進行的第二階段訓練,
    # 目的是讓模型學會「看到問題,就該認真回答」這種行為模式,
    # 而不是像純接龍一樣不分青紅皂白地接續所有文字。
    sft_data_path: str = "sft_data.jsonl"   # prepare_sft_data.py 產生的結構化資料
    sft_learning_rate: float = 5e-5         # 比預訓練的學習率小,避免破壞已經學到的語言能力
    sft_max_iters: int = 1000                # 資料量持續增加,依比例調高步數,同時避免過擬合
    sft_eval_interval: int = 150

    # ------- 推理 (inference) 參數 -------
    max_new_tokens: int = 80   # 縮短生成長度,避免回完問題後模型繼續自由接龍出下一輪假對話
    temperature: float = 0.7       # 調高,避免每次都固定選同一個「最安全」的開頭字,增加回覆變化
    top_k: int = 40
    top_p: float = 0.9             # 核採樣門檻,搭配 top_k 一起使用效果較好
    repetition_penalty: float = 1.5  # 提高懲罰力道,更積極避免重複字詞

    # ------- 裝置 -------
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1337


config = Config()
