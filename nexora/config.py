"""
nexora/config.py
----------------
Nexora Native v1 (Nexora-Nano) 的設定檔。
與根目錄 config.py（GPT-2 路線）完全無關，不引用它。

架構目標：
  n_embd=256, n_head=8, n_layer=6, block_size=256, vocab_size=4096
  從零隨機初始化，不載入任何外部 pretrained weights。
"""

from dataclasses import dataclass, field
import torch


@dataclass
class NexoraConfig:
    # ── 模型架構 ──────────────────────────────────────────────────────────
    n_embd: int = 256       # embedding / hidden 維度
    n_head: int = 8         # attention head 數（head_size = n_embd // n_head = 32）
    n_layer: int = 6        # Transformer block 層數
    block_size: int = 256   # context length（最長輸入 token 數）
    vocab_size: int = 4096  # 詞表大小（固定，含 special tokens）
    dropout: float = 0.1

    # ── 資料路徑 ──────────────────────────────────────────────────────────
    data_dir: str = "data"
    tokenizer_path: str = "nexora/nexora_vocab.json"
    checkpoint_dir: str = "nexora/checkpoints"
    checkpoint_path: str = "nexora/checkpoints/nexora_latest.pt"
    best_checkpoint_path: str = "nexora/checkpoints/nexora_best.pt"

    # ── 資料切分 ──────────────────────────────────────────────────────────
    train_frac: float = 0.90
    val_frac: float = 0.08
    # test = 1 - train - val = 0.02

    # ── 訓練超參數 ────────────────────────────────────────────────────────
    batch_size: int = 64
    grad_accumulation: int = 4          # 有效 batch = batch_size × grad_accumulation
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_iters: int = 500
    max_iters: int = 20000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    use_amp: bool = True                # 混合精度

    # ── 評估 / checkpoint ─────────────────────────────────────────────────
    eval_interval: int = 500            # 每幾步評估一次 val loss
    eval_iters: int = 20                # 評估時平均幾個 batch
    save_interval: int = 1000           # 每幾步存 checkpoint
    generate_interval: int = 500        # 每幾步做一次文字生成測試
    early_stop_patience: int = 10       # 連續幾次 eval 沒創新低就停

    # ── 生成參數 ──────────────────────────────────────────────────────────
    max_new_tokens: int = 60
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9

    # ── 裝置 ─────────────────────────────────────────────────────────────
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 42

    # ── 固定測試 prompt（pretraining evaluation 用）──────────────────────
    eval_prompts: list = field(default_factory=lambda: [
        "你好",
        "今天天氣",
        "我想學習",
        "Hello",
        "How are you?",
        "Taiwan is",
    ])

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd 必須能被 n_head 整除"
        assert self.vocab_size <= 65536, "vocab_size 超過合理範圍"
