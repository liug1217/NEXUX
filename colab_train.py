# ===== Google Colab SFT 訓練腳本 =====
# 在 Colab 上用 T4/V100/A100 GPU 訓練，比本機 GTX 1660 SUPER 快 4-10 倍。
# batch_size 和所有參數都自動根據 GPU VRAM 計算，不需要手動設定。
#
# 使用方式（在 Colab 裡一格一格執行）：
#
# --- Cell 1: 安裝套件 ---
# !pip install -q torch bitsandbytes transformers
#
# --- Cell 2: 拉程式碼 ---
# !git clone https://github.com/liug1217/NEXUX.git /content/nexux
# %cd /content/nexux
#
# --- Cell 3: 下載並轉換預訓練模型（約 1 分鐘） ---
# !python convert_pretrained.py
#
# --- Cell 4: 開始訓練（全自動，T4 大約 30-60 分鐘） ---
# !python -u colab_train.py
#
# --- Cell 5: 匯出權重 ---
# !python export_pretrained.py
#
# --- Cell 6: 評估 ---
# !python eval_open_ended.py --pretrained
#
# --- Cell 7: 下載或推送 ---
# from google.colab import files
# files.download('weights_meta_pretrained.json')
# files.download('weights_pretrained.npz')
# # 或者直接 push:
# # !git add weights_meta_pretrained.json weights_pretrained.npz data/*.jsonl
# # !git commit -m "Colab SFT 訓練完成"
# # !git push

import os
import sys
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")
sys.path.insert(0, ".")

from config import Config
from bert_wordpiece_tokenizer import BertWordpieceTokenizer
from train_sft import train_sft, _ensure_sft_data_up_to_date
from run_pretrained_sft import _auto_block_size

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024) if torch.cuda.is_available() else 0
print(f"[colab_train] GPU: {gpu_name} ({vram_mb}MB VRAM)")

epochs = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 1

tokenizer = BertWordpieceTokenizer.load_from_vocab_txt("vocab_pretrained.txt")
print(f"[colab_train] 詞表大小: {tokenizer.vocab_size}")

base = Config()
_ensure_sft_data_up_to_date(base)

block_size = _auto_block_size(base.sft_data_path, tokenizer)

config = Config(
    **{
        **base.__dict__,
        "checkpoint_path": "checkpoint_pretrained.pt",
        "block_size": block_size,
        "sft_batch_size": 0,
        "dropout": 0.1,
        "sft_max_iters": 0,
        "sft_epochs": epochs,
        "sft_eval_interval": 0,
        "seed": base.seed,
    }
)

print(f"[colab_train] {epochs} epoch(s)，block_size={block_size}")
print(f"[colab_train] batch_size 和步數由 train_sft 實測 VRAM 自動計算")
train_sft(config=config, tokenizer=tokenizer)
