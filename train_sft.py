"""
train_sft.py
-------------
SFT(問答微調)訓練主程式。

跟 train.py 的差別:
- train.py 是「預訓練」,從零開始,用純接龍的方式,讓模型學會語言的基本規律。
- train_sft.py 是「微調」,建立在預訓練成果之上,額外訓練模型學會
  「看到問題,就該認真回答」這種行為模式。

使用順序一定是:
    1. python train.py              (先預訓練,產生 checkpoint.pt)
    2. python prepare_sft_data.py   (把 data/*.jsonl 對話展開成 SFT 訓練用的 jsonl)
    3. python train_sft.py          (在預訓練成果上,進行問答微調)
    4. python export_weights.py     (匯出最終權重)

SFT 訓練會直接載入 train.py 產生的 checkpoint.pt,在原本的權重基礎上
繼續訓練,而不是從頭開始,所以學習率設得比預訓練時小很多,
避免破壞掉預訓練階段已經學到的語言能力。
"""

import os
import torch

from config import Config
from tokenizer import CharTokenizer
from dataset import SFTDataset
from model import GPTModel


def train_sft(config: Config | None = None, tokenizer=None):
    """
    tokenizer: 可選,傳入的話會直接使用這個 tokenizer 實例,不去讀
    config.tokenizer_path、也不限定一定要是 CharTokenizer(只要有
    encode()/vocab_size 介面即可)。這是為了讓「微調預訓練模型」
    (使用 BertWordpieceTokenizer,見 bert_wordpiece_tokenizer.py)
    能重用同一套訓練迴圈,不用另外複製一份程式碼。
    """
    config = config or Config()
    torch.manual_seed(config.seed)

    if not os.path.exists(config.checkpoint_path):
        raise FileNotFoundError(
            f"找不到 {config.checkpoint_path},請先執行「python train.py」完成預訓練。"
        )
    if tokenizer is None and not os.path.exists(config.tokenizer_path):
        raise FileNotFoundError(
            f"找不到 {config.tokenizer_path},請先執行「python train.py」完成預訓練。"
        )
    if not os.path.exists(config.sft_data_path):
        raise FileNotFoundError(
            f"找不到 {config.sft_data_path},請先執行「python prepare_sft_data.py」產生這份檔案。"
        )

    # ---- 1. 載入 tokenizer(沿用預訓練階段的詞表,不能重新建立) ----
    if tokenizer is None:
        tokenizer = CharTokenizer.load(config.tokenizer_path)
    print(f"[train_sft] 已載入 tokenizer,詞表大小: {tokenizer.vocab_size}")

    # ---- 2. 載入 SFT 訓練資料 ----
    dataset = SFTDataset(config, tokenizer, config.sft_data_path)

    # ---- 3. 載入預訓練好的模型權重 ----
    checkpoint = torch.load(config.checkpoint_path, map_location=config.device)

    # 優先使用 checkpoint 裡記錄的架構參數(跟 export_weights.py 用同一套邏輯,
    # 確保 SFT 階段使用的模型架構,跟預訓練時完全一致)。
    if "architecture" in checkpoint:
        arch = checkpoint["architecture"]
        model_config = Config(
            **{**config.__dict__, **arch}
        )
        print("[train_sft] 使用 checkpoint 裡記錄的架構參數(較安全)")
    else:
        model_config = config
        print("[train_sft] 警告:這是舊版 checkpoint,沒有記錄架構參數,改用 config.py 目前的設定。")

    model = GPTModel(model_config, vocab_size=checkpoint["vocab_size"]).to(config.device)
    # strict=False:checkpoint 裡可能不包含 attn.mask 這種因果遮罩 buffer
    # (根據 config 自動生成、不是訓練出來的權重,本來就不需要存/載入),
    # 這裡明確檢查「缺的只能是 attn.mask」,避免真正的權重被漏掉卻沒發現。
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    assert not unexpected, f"checkpoint 有架構對不上的多餘權重: {unexpected}"
    assert all("attn.mask" in k for k in missing), f"checkpoint 缺少非 buffer 的權重: {missing}"
    print(f"[train_sft] 已載入預訓練權重,起始 loss 應該會比從零訓練低很多")

    # ---- 4. Optimizer(用比預訓練小很多的學習率,避免破壞已學到的能力) ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.sft_learning_rate,
        weight_decay=config.weight_decay,
    )

    # 跟 train.py 一樣開混合精度訓練加速 GPU 運算。
    use_amp = config.use_amp and config.device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("[train_sft] 已啟用混合精度訓練(AMP)")

    # ---- 5. SFT 訓練迴圈 ----
    model.train()
    for step in range(config.sft_max_iters):
        x, y = dataset.get_batch("train")

        with torch.amp.autocast("cuda", enabled=use_amp):
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        if config.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        if step % config.sft_eval_interval == 0 or step == config.sft_max_iters - 1:
            print(f"[SFT step {step:4d}] loss {loss.item():.4f}")

    # ---- 6. 儲存 SFT 完成後的模型(覆蓋掉原本的 checkpoint.pt) ----
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab_size": checkpoint["vocab_size"],
            "step": checkpoint.get("step", 0),
            "architecture": {
                "n_embd": model_config.n_embd,
                "n_head": model_config.n_head,
                "n_layer": model_config.n_layer,
                "block_size": model_config.block_size,
            },
            "sft_applied": True,  # 標記這個 checkpoint 已經經過 SFT 微調
        },
        config.checkpoint_path,
    )
    print(f"[train_sft] SFT 訓練完成,模型已更新至: {config.checkpoint_path}")
    print("[train_sft] 接下來執行「python export_weights.py」重新匯出權重即可。")


if __name__ == "__main__":
    train_sft()
    