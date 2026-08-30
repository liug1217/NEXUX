"""
run_pretrained_sft.py
-----------------------
第二階段：接上現有語料(data/*.jsonl -> sft_data.jsonl)，對轉換好的
預訓練 checkpoint（checkpoint_pretrained.pt）做微調。

所有關鍵參數自動計算：
  - block_size: 掃描資料取 P95 長度
  - batch_size: 實測 GPU VRAM 算最大值（任何 GPU 都能自動最佳化）
  - 步數: epochs × ceil(資料筆數 / batch_size)

用法：
    python run_pretrained_sft.py                # 全自動
    python run_pretrained_sft.py --smoke         # 只跑 20 步，快速確認不會出錯
    python run_pretrained_sft.py --steps 10000   # 手動指定步數
    python run_pretrained_sft.py --epochs 3      # 手動指定 epoch 數
    python run_pretrained_sft.py --seed 42       # 自訂隨機種子
"""

import json
import sys

from config import Config
from bert_wordpiece_tokenizer import BertWordpieceTokenizer
from train_sft import train_sft


def _auto_block_size(jsonl_path: str, tokenizer, percentile: float = 0.95) -> int:
    """掃描 SFT 資料，取 P95 token 長度當 block_size，上限 1024。"""
    lengths = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            input_ids = tokenizer.encode(item["input"])
            output_ids = tokenizer.encode(item["output"])
            eos_id = getattr(tokenizer, "eos_id", None)
            if eos_id is not None:
                output_ids = output_ids + [eos_id]
            lengths.append(len(input_ids) + len(output_ids))

    if not lengths:
        return 384

    lengths.sort()
    idx = int(len(lengths) * percentile)
    idx = min(idx, len(lengths) - 1)
    p_val = lengths[idx]

    bs = ((p_val + 63) // 64) * 64
    bs = max(128, min(bs, 1024))

    avg_len = sum(lengths) / len(lengths)
    max_len = lengths[-1]
    print(f"[auto_block_size] {len(lengths)} 筆資料: 平均 {avg_len:.0f} tokens, "
          f"最長 {max_len}, P95={p_val} → block_size={bs}")
    return bs


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
    print(f"[run_pretrained_sft] 已載入 BertWordpieceTokenizer，詞表大小: {tokenizer.vocab_size}")

    base = Config()

    from train_sft import _ensure_sft_data_up_to_date
    _ensure_sft_data_up_to_date(base)

    block_size = _auto_block_size(base.sft_data_path, tokenizer)

    if smoke:
        sft_max_iters = 20
    elif custom_steps:
        sft_max_iters = custom_steps
    else:
        sft_max_iters = 0

    epochs = custom_epochs or base.sft_epochs

    config = Config(
        **{
            **base.__dict__,
            "checkpoint_path": "checkpoint_pretrained.pt",
            "block_size": block_size,
            "sft_batch_size": 0,
            "dropout": 0.1,
            "sft_max_iters": sft_max_iters,
            "sft_epochs": epochs,
            "sft_eval_interval": 5 if smoke else 0,
            "seed": custom_seed if custom_seed is not None else base.seed,
        }
    )

    if config.sft_eval_interval == 0:
        config.sft_eval_interval = 300

    if custom_seed is not None:
        print(f"[run_pretrained_sft] 使用自訂 seed: {custom_seed}")

    print(f"[run_pretrained_sft] {'smoke test（20步）' if smoke else f'完整微調（{epochs} epochs）'}")
    train_sft(config=config, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
