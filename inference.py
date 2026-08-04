"""
inference.py
------------
載入訓練好的模型與 tokenizer,根據使用者輸入的 prompt 生成後續文字。

NEXUX v1.0(見 docs/MODEL_MIGRATION.md)之後,正式環境只用
load_pretrained_model() 載入的微調預訓練模型。load_model()(char-level
從零訓練模型)保留在這個檔案裡,是刻意不刪除的封存版本,方便未來需要
回頭比較「從零訓練」跟「微調預訓練模型」兩種做法的差異時還能直接載入
使用,但不再是正式站實際服務的路徑。
"""

import os
import torch

from config import Config
from tokenizer import CharTokenizer
from model import GPTModel
from text_cleanup import truncate_at_next_turn


def load_model(config: Config):
    """
    封存版本:載入 char-level 從零訓練模型(NEXUX v1.0 之前的做法)。
    正式站已經不再使用這條路徑,保留純粹是為了歷史比較,不會被刪除。
    """
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


def load_pretrained_model(
    checkpoint_path: str = "checkpoint_pretrained.pt",
    vocab_path: str = "vocab_pretrained.txt",
    device: str | None = None,
):
    """
    NEXUX v1.0 正式版本:載入微調過的預訓練模型(ckiplab/gpt2-base-chinese
    微調結果,見 docs/MODEL_MIGRATION.md)。checkpoint_pretrained.pt 是本機
    訓練用的中繼檔案(不進 git),只有在本機執行過
    convert_pretrained.py + run_pretrained_sft.py 之後才會存在,主要給
    server.py 本機開發測試用;正式站(Vercel)走的是 numpy_gpt.py 讀
    weights_pretrained.npz 那條路徑,不會用到這支函式。
    """
    from bert_wordpiece_tokenizer import BertWordpieceTokenizer

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"找不到 {checkpoint_path},請先在本機執行"
            "「python convert_pretrained.py」跟「python run_pretrained_sft.py」。"
        )
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"找不到 {vocab_path},請先執行「python convert_pretrained.py」。")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertWordpieceTokenizer.load_from_vocab_txt(vocab_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    arch = checkpoint["architecture"]
    base = Config()
    model_config = Config(**{**base.__dict__, **arch, "dropout": 0.1, "device": device})

    model = GPTModel(model_config, vocab_size=checkpoint["vocab_size"]).to(device)
    # strict=False:checkpoint 裡不含 attn.mask 這種根據 config 自動生成的
    # 因果遮罩 buffer(不是訓練出來的權重),本來就不需要載入。
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    assert not unexpected, f"checkpoint 有架構對不上的多餘權重: {unexpected}"
    assert all("attn.mask" in k for k in missing), f"checkpoint 缺少非 buffer 的權重: {missing}"
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
    reply = full_text[len(wrapped_prompt):] if full_text.startswith(wrapped_prompt) else full_text
    return truncate_at_next_turn(reply)


if __name__ == "__main__":
    cfg = Config()
    prompt = input("請輸入 prompt: ").strip() or "你好"
    result = generate_text(prompt, cfg)
    print("\n----- 生成結果 -----")
    print(result)
    