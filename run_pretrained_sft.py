"""
run_pretrained_sft.py
-----------------------
第二階段:接上現有語料(data/*.jsonl -> sft_data.jsonl),對轉換好的
預訓練 checkpoint(checkpoint_pretrained.pt)做微調。

步數自動計算:config.sft_epochs × 資料筆數,不用手動算。
資料增減時步數自動跟著變,不再會出現「3000步只看了46%資料」的問題。

用法:
    python run_pretrained_sft.py                # 自動算步數(sft_epochs × 資料筆數)
    python run_pretrained_sft.py --smoke         # 只跑 20 步,快速確認不會出錯
    python run_pretrained_sft.py --steps 10000   # 手動指定步數(覆蓋自動計算)
    python run_pretrained_sft.py --epochs 3      # 手動指定 epoch 數
    python run_pretrained_sft.py --seed 42       # 自訂隨機種子
"""

import json
import sys

from config import Config
from bert_wordpiece_tokenizer import BertWordpieceTokenizer
from train_sft import train_sft


def _count_sft_data(path: str) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def main():
    smoke = "--smoke" in sys.argv

    custom_steps = None
    if "--steps" in sys.argv:
        idx = sys.argv.index("--steps")
        custom_steps = int(sys.argv[idx + 1])

    custom_epochs = None
    if "--epochs" in sys.argv:
        idx = sys.argv.index("--epochs")
        custom_epochs = int(sys.argv[idx + 1])

    custom_seed = None
    if "--seed" in sys.argv:
        idx = sys.argv.index("--seed")
        custom_seed = int(sys.argv[idx + 1])

    tokenizer = BertWordpieceTokenizer.load_from_vocab_txt("vocab_pretrained.txt")
    print(f"[run_pretrained_sft] 已載入 BertWordpieceTokenizer,詞表大小: {tokenizer.vocab_size}")

    base = Config()

    if smoke:
        sft_max_iters = 20
    elif custom_steps:
        sft_max_iters = custom_steps
    else:
        from train_sft import _ensure_sft_data_up_to_date
        _ensure_sft_data_up_to_date(base)
        n_examples = _count_sft_data(base.sft_data_path)
        epochs = custom_epochs or base.sft_epochs
        sft_max_iters = epochs * n_examples
        print(f"[run_pretrained_sft] {n_examples} 筆資料 × {epochs} epochs = {sft_max_iters} 步(自動計算)")

    config = Config(
        **{
            **base.__dict__,
            "checkpoint_path": "checkpoint_pretrained.pt",
            "block_size": 1024,
            "n_embd": 768,
            "n_head": 12,
            "n_layer": 12,
            "dropout": 0.1,
            "sft_max_iters": sft_max_iters,
            "sft_eval_interval": 5 if smoke else max(300, sft_max_iters // 10),
            "seed": custom_seed if custom_seed is not None else base.seed,
        }
    )
    if custom_seed is not None:
        print(f"[run_pretrained_sft] 使用自訂 seed: {custom_seed}(預設固定是 {base.seed})")

    print(f"[run_pretrained_sft] {'smoke test(20步)' if smoke else f'完整微調({config.sft_max_iters}步)'}")
    train_sft(config=config, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
