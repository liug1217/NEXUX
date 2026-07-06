"""
inference.py
------------
載入訓練好的模型與 tokenizer,根據使用者輸入的 prompt 生成後續文字。
"""

import os
import torch

from config import Config
from tokenizer import CharTokenizer
from model import GPTModel


def load_model(config: Config):
    if not os.path.exists(config.checkpoint_path):
        raise FileNotFoundError(
            f"找不到模型權重: {config.checkpoint_path},請先執行 train.py 訓練模型。"
        )
    if not os.path.exists(config.tokenizer_path):
        raise FileNotFoundError(
            f"找不到 tokenizer 檔案: {config.tokenizer_path},請先執行 train.py 訓練模型。"
        )

    tokenizer = CharTokenizer.load(config.tokenizer_path)

    checkpoint = torch.load(config.checkpoint_path, map_location=config.device)

    # 優先使用 checkpoint 裡記錄的架構參數,確保跟訓練時的模型結構一致
    if "architecture" in checkpoint:
        arch = checkpoint["architecture"]
        model_config = Config(**{**config.__dict__, **arch})
    else:
        model_config = config

    model = GPTModel(model_config, vocab_size=checkpoint["vocab_size"]).to(config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    is_sft = checkpoint.get("sft_applied", False)
    return model, tokenizer, is_sft


def generate_text(
    prompt: str,
    config: Config | None = None,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    qa_format: bool = True,
) -> str:
    """
    給定 prompt,回傳模型生成的回覆。
    未指定的參數會採用 config.py 裡的預設值。

    qa_format: 如果模型已經經過 SFT(問答微調),預設會把 prompt 包成
               訓練時用過的「問:...\\n答:」格式,讓模型進入回答模式。
               如果想看模型「純接龍」的原始行為,可以把這個參數設成 False。
    """
    config = config or Config()
    model, tokenizer, is_sft = load_model(config)

    max_new_tokens = max_new_tokens or config.max_new_tokens
    temperature = temperature if temperature is not None else config.temperature
    top_k = top_k if top_k is not None else config.top_k

    # 只有模型「真的經過 SFT 訓練」時,才包裝成問答格式;
    # 否則模型從沒見過這種格式,硬套上去只會讓生成效果更差。
    wrapped_prompt = f"問:{prompt}\n答:" if (qa_format and is_sft) else prompt

    idx = torch.tensor(
        [tokenizer.encode(wrapped_prompt)], dtype=torch.long, device=config.device
    )
    if idx.shape[1] == 0:
        raise ValueError("prompt 編碼後長度為 0,可能包含詞表以外的字元。")

    out_idx = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
    )
    full_text = tokenizer.decode(out_idx[0].tolist())
    return full_text[len(wrapped_prompt):] if full_text.startswith(wrapped_prompt) else full_text


if __name__ == "__main__":
    cfg = Config()
    prompt = input("請輸入 prompt: ").strip() or "你好"
    result = generate_text(prompt, cfg)
    print("\n----- 生成結果 -----")
    print(result)
    