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

import glob
import math
import os
import time
import torch

from config import Config
from tokenizer import CharTokenizer
from dataset import SFTDataset
from model import GPTModel


def _get_sft_lr(step: int, max_iters: int, peak_lr: float) -> float:
    warmup_iters = min(100, max_iters // 10)
    min_lr = peak_lr / 10
    if step < warmup_iters:
        return peak_lr * (step + 1) / warmup_iters
    if step >= max_iters:
        return min_lr
    decay_ratio = (step - warmup_iters) / max(1, max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (peak_lr - min_lr)


def _ensure_sft_data_up_to_date(config: Config) -> None:
    """
    sft_data.jsonl 是 prepare_sft_data.py 從 data/*.jsonl 展開出來的中繼檔案,
    不是自動同步的——如果語料改了但忘記重新執行 prepare_sft_data.py,
    train_sft() 會靜靜地用「舊的、沒有反映最新語料」的 sft_data.jsonl 去訓練,
    不會有任何錯誤訊息,整次訓練等於白跑(這個坑真實發生過,詳見
    docs/MODEL_MIGRATION.md「語料沒有真的被訓練到」那一節)。

    這裡在每次訓練前自動比對 data/*.jsonl 裡最新的修改時間,跟
    sft_data.jsonl 的修改時間,只要語料比較新(或 sft_data.jsonl 根本
    不存在),就自動重新產生一次,不用仰賴使用者自己記得手動執行。
    """
    data_files = glob.glob(os.path.join(config.data_dir, "*.jsonl"))
    if not data_files:
        return  # 沒有語料檔案,交給後面既有的檔案存在性檢查去報錯

    newest_data_mtime = max(os.path.getmtime(f) for f in data_files)
    needs_regen = (
        not os.path.exists(config.sft_data_path)
        or os.path.getmtime(config.sft_data_path) < newest_data_mtime
    )
    if needs_regen:
        print(
            f"[train_sft] {config.sft_data_path} 不存在或已過期(比 "
            f"{config.data_dir}/ 底下最新的語料檔案還舊),自動重新執行 "
            "prepare_sft_data.py 產生最新版本..."
        )
        import prepare_sft_data
        prepare_sft_data.main()


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
    _ensure_sft_data_up_to_date(config)

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
    real_unexpected = [k for k in unexpected if "attn.mask" not in k]
    assert not real_unexpected, f"checkpoint 有架構對不上的多餘權重: {real_unexpected}"
    real_missing = [k for k in missing if "attn.mask" not in k]
    assert not real_missing, f"checkpoint 缺少非 buffer 的權重: {real_missing}"
    print(f"[train_sft] 已載入預訓練權重,起始 loss 應該會比從零訓練低很多")

    if model_config.n_layer >= 24:
        model.gradient_checkpointing = True
        print("[train_sft] 大模型,已啟用 gradient checkpointing 節省 VRAM")

    import sys
    if hasattr(torch, "compile") and sys.platform != "win32":
        try:
            model = torch.compile(model)
            print("[train_sft] 已啟用 torch.compile 加速訓練")
        except Exception as e:
            print(f"[train_sft] torch.compile 不可用,跳過: {e}")

    # ---- 4. Optimizer(用比預訓練小很多的學習率,避免破壞已學到的能力) ----
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=config.sft_learning_rate,
            weight_decay=config.weight_decay,
        )
        print("[train_sft] 已啟用 8-bit AdamW（bitsandbytes），VRAM 大幅節省")
    except ImportError:
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
    accum_steps = 8
    n_examples = len(dataset.examples)
    epoch_steps = n_examples if n_examples > 0 else config.sft_max_iters

    def _save_checkpoint(path, tag=""):
        raw_sd = model.state_dict()
        cleaned_sd = {k.replace("_orig_mod.", ""): v for k, v in raw_sd.items()}
        torch.save(
            {
                "model_state_dict": cleaned_sd,
                "optimizer_state_dict": optimizer.state_dict(),
                "vocab_size": checkpoint["vocab_size"],
                "step": checkpoint.get("step", 0),
                "architecture": {
                    "n_embd": model_config.n_embd,
                    "n_head": model_config.n_head,
                    "n_layer": model_config.n_layer,
                    "block_size": model_config.block_size,
                },
                "sft_applied": True,
            },
            path,
        )
        if tag:
            print(f"[train_sft] {tag} 已存檔: {path}")

    best_loss = float("inf")
    best_epoch = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
    train_start_time = time.time()
    for step in range(config.sft_max_iters):
        lr = _get_sft_lr(step, config.sft_max_iters, config.sft_learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = dataset.get_batch("train")

        with torch.amp.autocast("cuda", enabled=use_amp):
            _, loss = model(x, y)
        scaled_loss = loss / accum_steps
        scaler.scale(scaled_loss).backward()
        accum_loss += loss.item()

        if (step + 1) % accum_steps == 0 or step == config.sft_max_iters - 1:
            if config.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step > 0 and step % 50 == 0:
            elapsed = time.time() - train_start_time
            speed = elapsed / step
            eta = speed * (config.sft_max_iters - step)
            eta_h, eta_rem = divmod(int(eta), 3600)
            eta_m, eta_s = divmod(eta_rem, 60)
            print(f"[speed] step {step} | {speed:.3f} s/step | ETA {eta_h:02d}:{eta_m:02d}:{eta_s:02d}", flush=True)

        if step % config.sft_eval_interval == 0 or step == config.sft_max_iters - 1:
            avg_loss = accum_loss / min(step % accum_steps + 1, accum_steps) if accum_loss > 0 else loss.item()
            cur_epoch = (step + 1) / epoch_steps
            print(f"[SFT step {step:5d}] loss {avg_loss:.4f} lr {lr:.2e} epoch {cur_epoch:.1f}")
            accum_loss = 0.0
        elif (step + 1) % accum_steps == 0:
            accum_loss = 0.0

        # 每個 epoch 結束時存一份 checkpoint，追蹤最佳 epoch
        if (step + 1) % epoch_steps == 0:
            cur_epoch = (step + 1) // epoch_steps
            # 用最近的平均 loss 當這個 epoch 的代表
            epoch_loss = loss.item()
            tag = f"epoch {cur_epoch}"
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = cur_epoch
                _save_checkpoint(config.checkpoint_path, f"{tag} (目前最佳, loss={epoch_loss:.4f})")
                tag += " ★ best"
            else:
                _save_checkpoint(config.checkpoint_path + f".epoch{cur_epoch}", f"{tag} (loss={epoch_loss:.4f})")
            print(f"[train_sft] --- epoch {cur_epoch} 結束 --- loss {epoch_loss:.4f} | best so far: epoch {best_epoch}")

    # 最後一步不一定剛好是 epoch 邊界，如果 best checkpoint 不是最後一步的，要還原
    final_epoch = config.sft_max_iters // epoch_steps
    if best_epoch > 0 and best_epoch < final_epoch:
        print(f"[train_sft] 最佳 epoch 是 {best_epoch},最終 checkpoint 已是該版本")
    elif best_epoch == 0:
        _save_checkpoint(config.checkpoint_path, "訓練完成(無 epoch 邊界 checkpoint)")

    print(f"[train_sft] SFT 訓練完成,最佳模型: {config.checkpoint_path} (epoch {best_epoch})")
    print("[train_sft] 接下來執行「python export_pretrained.py」重新匯出權重即可。")


if __name__ == "__main__":
    train_sft()
    