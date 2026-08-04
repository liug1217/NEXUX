"""
run_pretrained_sft.py
-----------------------
第二階段:接上現有語料(data/*.jsonl -> sft_data.jsonl),對轉換好的
預訓練 checkpoint(checkpoint_pretrained.pt)做微調。

跟 train_sft.py 的差別只在於:
- tokenizer 換成 BertWordpieceTokenizer(讀 vocab_pretrained.txt)
- checkpoint_path 指向 checkpoint_pretrained.pt,不會動到正式站現在還在用的
  checkpoint.pt / tokenizer.json,微調測試不會影響現有 char-level 模型正常運作。

用法:
    python run_pretrained_sft.py                # 完整跑 sft_max_iters 步(預設 1500)
    python run_pretrained_sft.py --smoke         # 只跑 20 步,快速確認不會出錯
    python run_pretrained_sft.py --steps 10000   # 自訂步數(例如拉長訓練量測試是否
                                                   # 是「訓練不夠」導致內容答非所問)
    python run_pretrained_sft.py --seed 42       # 自訂隨機種子(預設固定是1337,
                                                   # 同樣的語料+同樣的seed重跑會得到
                                                   # 完全一樣的結果,要測試「這次結果是
                                                   # 不是batch_size=1造成的隨機噪音」,
                                                   # 就要換一個不同的seed才有意義)
"""

import sys

from config import Config
from bert_wordpiece_tokenizer import BertWordpieceTokenizer
from train_sft import train_sft


def main():
    smoke = "--smoke" in sys.argv

    custom_steps = None
    if "--steps" in sys.argv:
        idx = sys.argv.index("--steps")
        custom_steps = int(sys.argv[idx + 1])

    custom_seed = None
    if "--seed" in sys.argv:
        idx = sys.argv.index("--seed")
        custom_seed = int(sys.argv[idx + 1])

    tokenizer = BertWordpieceTokenizer.load_from_vocab_txt("vocab_pretrained.txt")
    print(f"[run_pretrained_sft] 已載入 BertWordpieceTokenizer,詞表大小: {tokenizer.vocab_size}")

    base = Config()
    sft_max_iters = 20 if smoke else (custom_steps or base.sft_max_iters)
    config = Config(
        **{
            **base.__dict__,
            "checkpoint_path": "checkpoint_pretrained.pt",
            "block_size": 1024,
            "n_embd": 768,
            "n_head": 12,
            "n_layer": 12,
            "dropout": 0.1,  # 沿用 HF 模型原本的 dropout 設定,微調階段不用刻意加大正則化
            "sft_max_iters": sft_max_iters,
            "sft_eval_interval": 5 if smoke else 300,
            "seed": custom_seed if custom_seed is not None else base.seed,
        }
    )
    if custom_seed is not None:
        print(f"[run_pretrained_sft] 使用自訂 seed: {custom_seed}(預設固定是 {base.seed})")

    print(f"[run_pretrained_sft] {'smoke test(20步)' if smoke else f'完整微調({config.sft_max_iters}步)'}")
    train_sft(config=config, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
