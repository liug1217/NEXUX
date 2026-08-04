"""
eval_open_ended.py
--------------------
固定的開放式問題測試集,用來在每次調整模型/語料/訓練設定後,
一致地比較生成品質有沒有進步(而不是每次隨手挑幾句話測,
測試題目不一樣、沒辦法客觀比較前後差異)。

刻意只放「語料庫裡沒有現成完整答案、必須靠模型真正生成」的開放式問題,
排除 smalltalk/qa_lookup 能直接命中的題目,才是在測「模型本身」的能力,
不是在測規則系統。

用法:
    python eval_open_ended.py                          # 測目前 char-level 正式模型(model.py + tokenizer.json)
    python eval_open_ended.py --pretrained              # 測 checkpoint_pretrained.pt(預訓練微調中的版本)

每次執行只負責印出生成結果,好壞由人工判斷記錄(例如更新
docs/MODEL_MIGRATION.md 裡的對照表),這裡不做自動化評分——
語意是否「答對」目前沒有夠可靠的自動化方式判斷,勉強做只會製造
虛假的精確度。
"""

import sys

TEST_PROMPTS = [
    "謝謝你",
    "哈囉",
    "你覺得友情跟愛情最大的差別是什麼?",
    "如果可以回到過去,你會想改變什麼?",
    "為什麼天空是藍色的?",
    "幫我想一個關於太空旅行的故事開頭",
    "你最近過得怎麼樣?",
    "可以給我一些關於學習新語言的建議嗎?",
]


def eval_char_level(seed: int = 1337):
    import torch
    from config import Config
    from model import GPTModel
    from tokenizer import CharTokenizer

    torch.manual_seed(seed)
    config = Config()
    tokenizer = CharTokenizer.load(config.tokenizer_path)
    checkpoint = torch.load(config.checkpoint_path, map_location="cpu")
    arch = checkpoint.get("architecture", {
        "n_embd": config.n_embd, "n_head": config.n_head,
        "n_layer": config.n_layer, "block_size": config.block_size,
    })
    model_config = Config(**{**config.__dict__, **arch})
    model = GPTModel(model_config, vocab_size=checkpoint["vocab_size"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    for prompt in TEST_PROMPTS:
        text = f"問:{prompt}\n答:"
        ids = tokenizer.encode(text)
        idx = torch.tensor([ids])
        with torch.no_grad():
            out = model.generate(
                idx, max_new_tokens=config.max_new_tokens,
                temperature=config.temperature, top_k=config.top_k,
                top_p=config.top_p, repetition_penalty=config.repetition_penalty,
            )
        reply = tokenizer.decode(out[0, len(ids):].tolist())
        print(f"Q: {prompt}")
        print(f"A: {reply}")
        print()


def eval_pretrained(seed: int = 1337):
    import torch
    from config import Config
    from model import GPTModel
    from bert_wordpiece_tokenizer import BertWordpieceTokenizer

    # 固定隨機種子:取樣式生成(temperature/top_k/top_p)每次結果都不一樣,
    # 不固定種子的話,兩次執行的差異可能來自「運氣」而不是模型/語料/訓練
    # 設定真的變好或變壞,固定種子才能公平比較前後版本。
    torch.manual_seed(seed)

    tokenizer = BertWordpieceTokenizer.load_from_vocab_txt("vocab_pretrained.txt")
    checkpoint = torch.load("checkpoint_pretrained.pt", map_location="cpu")
    arch = checkpoint["architecture"]
    base = Config()
    model_config = Config(**{**base.__dict__, **arch, "dropout": 0.1})
    model = GPTModel(model_config, vocab_size=checkpoint["vocab_size"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    for prompt in TEST_PROMPTS:
        text = f"問:{prompt}\n答:"
        ids = tokenizer.encode(text)
        idx = torch.tensor([ids])
        with torch.no_grad():
            out = model.generate(
                idx, max_new_tokens=100, temperature=0.8, top_k=40,
                top_p=0.9, repetition_penalty=1.3, eos_id=tokenizer.eos_id,
            )
        new_ids = out[0, len(ids):].tolist()
        hit_eos = len(new_ids) > 0 and new_ids[-1] == tokenizer.eos_id
        reply = tokenizer.decode(new_ids)
        print(f"Q: {prompt}")
        print(f"A({len(new_ids)} tokens, 自己停下={hit_eos}): {reply}")
        print()


if __name__ == "__main__":
    if "--pretrained" in sys.argv:
        eval_pretrained()
    else:
        eval_char_level()
