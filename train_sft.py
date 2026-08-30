"""
train_sft.py
-------------
SFT（問答微調）訓練主程式。

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
    自動比對 data/*.jsonl 跟 sft_data.jsonl 的修改時間,
    語料比較新就自動重新產生,不用手動記得執行 prepare_sft_data.py。
    """
    data_files = glob.glob(os.path.join(config.data_dir, "*.jsonl"))
    if not data_files:
        return
    newest_data_mtime = max(os.path.getmtime(f) for f in data_files)
    needs_regen = (
        not os.path.exists(config.sft_data_path)
        or os.path.getmtime(config.sft_data_path) < newest_data_mtime
    )
    if needs_regen:
        print(
            f"[train_sft] {config.sft_data_path} 不存在或已過期,自動重新執行 "
            "prepare_sft_data.py 產生最新版本..."
        )
        import prepare_sft_data
        prepare_sft_data.main()


def _probe_max_batch(model, dataset, config, use_amp) -> int:
    """實際跑一次 forward+backward 測量 VRAM,算出這張 GPU 能跑的最大 batch_size。"""
    if config.device != "cuda":
        return 2

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    baseline = torch.cuda.memory_allocated()

    model.train()
    x_test, y_test = dataset.get_batch("train", batch_size=2)
    with torch.amp.autocast("cuda", enabled=use_amp):
        _, loss = model(x_test, y_test)
    loss.backward()

    peak = torch.cuda.max_memory_allocated()
    model.zero_grad(set_to_none=True)
    del x_test, y_test, loss
    torch.cuda.empty_cache()

    per_sample = (peak - baseline) / 2
    if per_sample <= 0:
        return 2

    total_vram = torch.cuda.get_device_properties(0).total_memory
    available = total_vram * 0.90 - baseline
    max_bs = max(1, int(available / per_sample))
    max_bs = 2 ** int(math.log2(max(1, max_bs)))
    max_bs = min(max_bs, 64)

    vram_mb = total_vram / (1024 * 1024)
    print(f"[auto_batch] VRAM {vram_mb:.0f}MB | baseline {baseline / 1024 / 1024:.0f}MB | "
          f"per_sample {per_sample / 1024 / 1024:.0f}MB → max batch_size={max_bs}")
    return max_bs


def train_sft(config: Config | None = None, tokenizer=None):
    """
    tokenizer: 可選,傳入的話會直接使用這個 tokenizer 實例。
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

    if tokenizer is None:
        tokenizer = CharTokenizer.load(config.tokenizer_path)
    print(f"[train_sft] 已載入 tokenizer,詞表大小: {tokenizer.vocab_size}")

    dataset = SFTDataset(config, tokenizer, config.sft_data_path)

    checkpoint = torch.load(config.checkpoint_path, map_location=config.device)

    if "architecture" in checkpoint:
        arch = checkpoint["architecture"]
        model_config = Config(**{**config.__dict__, **arch})
        print("[train_sft] 使用 checkpoint 裡記錄的架構參數（較安全）")
    else:
        model_config = config
        print("[train_sft] 警告：這是舊版 checkpoint,沒有記錄架構參數")

    model = GPTModel(model_config, vocab_size=checkpoint["vocab_size"]).to(config.device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    real_unexpected = [k for k in unexpected if "attn.mask" not in k]
    assert not real_unexpected, f"checkpoint 有架構對不上的多餘權重: {real_unexpected}"
    real_missing = [k for k in missing if "attn.mask" not in k]
    assert not real_missing, f"checkpoint 缺少非 buffer 的權重: {real_missing}"
    print(f"[train_sft] 已載入預訓練權重,起始 loss 應該會比從零訓練低很多")

    import sys
    if hasattr(torch, "compile") and sys.platform != "win32":
        try:
            model = torch.compile(model)
            print("[train_sft] 已啟用 torch.compile 加速訓練")
        except Exception as e:
            print(f"[train_sft] torch.compile 不可用,跳過: {e}")

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

    use_amp = config.use_amp and config.device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("[train_sft] 已啟用混合精度訓練（AMP）")

    # ---- 自動偵測最大 batch_size（實測 VRAM，不用猜） ----
    sft_bs = _probe_max_batch(model, dataset, config, use_amp)
    print(f"[train_sft] VRAM 實測 → batch_size={sft_bs}")

    vram_pressure = config.block_size * sft_bs
    if model_config.n_layer >= 24 and vram_pressure > 4096:
        model.gradient_checkpointing = True
        print("[train_sft] 大模型+長序列,已啟用 gradient checkpointing 節省 VRAM")

    # ---- SFT 訓練迴圈 ----
    accum_steps = max(1, 8 // sft_bs)
    n_examples = len(dataset.examples)
    epoch_steps = max(1, -(-n_examples // sft_bs))

    # 用實測 batch_size 重新算步數
    if config.sft_max_iters > 0:
        sft_max_iters = config.sft_max_iters
    else:
        sft_max_iters = config.sft_epochs * epoch_steps

    eval_interval = config.sft_eval_interval
    if eval_interval <= 0 or eval_interval > sft_max_iters:
        eval_interval = max(50, sft_max_iters // 10)

    print(f"[train_sft] batch_size={sft_bs}, accum={accum_steps}, "
          f"effective_batch={sft_bs * accum_steps}, {epoch_steps} steps/epoch, "
          f"total {sft_max_iters} steps")

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
    for step in range(sft_max_iters):
        lr = _get_sft_lr(step, sft_max_iters, config.sft_learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = dataset.get_batch("train", batch_size=sft_bs)

        with torch.amp.autocast("cuda", enabled=use_amp):
            _, loss = model(x, y)
        scaled_loss = loss / accum_steps
        scaler.scale(scaled_loss).backward()
        accum_loss += loss.item()

        if (step + 1) % accum_steps == 0 or step == sft_max_iters - 1:
            if config.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step > 0 and step % 50 == 0:
            elapsed = time.time() - train_start_time
            speed = elapsed / step
            eta = speed * (sft_max_iters - step)
            eta_h, eta_rem = divmod(int(eta), 3600)
            eta_m, eta_s = divmod(eta_rem, 60)
            print(f"[speed] step {step} | {speed:.3f} s/step | ETA {eta_h:02d}:{eta_m:02d}:{eta_s:02d}", flush=True)

        if step % eval_interval == 0 or step == sft_max_iters - 1:
            avg_loss = accum_loss / min(step % accum_steps + 1, accum_steps) if accum_loss > 0 else loss.item()
            cur_epoch = (step + 1) / epoch_steps
            print(f"[SFT step {step:5d}] loss {avg_loss:.4f} lr {lr:.2e} epoch {cur_epoch:.1f}")
            accum_loss = 0.0
        elif (step + 1) % accum_steps == 0:
            accum_loss = 0.0

        if (step + 1) % epoch_steps == 0:
            cur_epoch = (step + 1) // epoch_steps
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

    final_epoch = sft_max_iters // epoch_steps
    if best_epoch > 0 and best_epoch < final_epoch:
        print(f"[train_sft] 最佳 epoch 是 {best_epoch},最終 checkpoint 已是該版本")
    elif best_epoch == 0:
        _save_checkpoint(config.checkpoint_path, "訓練完成（無 epoch 邊界 checkpoint）")

    elapsed_total = time.time() - train_start_time
    h, rem = divmod(int(elapsed_total), 3600)
    m, s = divmod(rem, 60)
    print(f"[train_sft] SFT 訓練完成！耗時 {h:02d}:{m:02d}:{s:02d},最佳模型: {config.checkpoint_path} (epoch {best_epoch})")
    print("[train_sft] 接下來執行「python export_pretrained.py」重新匯出權重即可。")


if __name__ == "__main__":
    train_sft()
