"""
train.py
--------
訓練主流程:
1. 讀取語料 -> 建立/載入 tokenizer
2. 建立 dataset、model、optimizer
3. 跑訓練迴圈,定期評估 train/val loss
4. 儲存模型權重與 tokenizer 詞表
"""

import os

# 這台機器上 torch 用多執行緒跑 CPU 運算時,會跟 OpenMP/MKL 等數學函式庫搶執行緒
# 資源導致直接當掉(segmentation fault)。在 import torch 之前,先把這些環境變數
# 設定成單執行緒模式,徹底避開這個問題,不用每次手動在終端機另外設定。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import math
import time
import torch

torch.set_num_threads(1)

from config import Config
from tokenizer import CharTokenizer
from dataset import TextDataset, load_corpus_text
from model import GPTModel


def get_lr(step: int, config: Config) -> float:
    """
    Learning rate schedule: 先線性 warmup,再用 cosine 曲線衰減到 min_learning_rate。
    這能讓訓練初期不會因為學習率太高而不穩定,後期又能收斂得更細緻。
    """
    if step < config.warmup_iters:
        return config.learning_rate * (step + 1) / config.warmup_iters

    if step >= config.max_iters:
        return config.min_learning_rate

    decay_ratio = (step - config.warmup_iters) / max(
        1, config.max_iters - config.warmup_iters
    )
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 1 -> 0
    return config.min_learning_rate + coeff * (
        config.learning_rate - config.min_learning_rate
    )


@torch.no_grad()
def estimate_loss(model: GPTModel, dataset: TextDataset, config: Config):
    """分別在 train / val 上取數個 batch 的平均 loss,減少評估時的雜訊。"""
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(config.eval_iters)
        for i in range(config.eval_iters):
            x, y = dataset.get_batch(split)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(config: Config | None = None):
    config = config or Config()
    torch.manual_seed(config.seed)

    # ---- 1. Tokenizer ----
    # load_corpus_text 會讀取 data_dir 底下所有 .jsonl 檔案(messages 格式)並合併
    # (例如 chat.jsonl、story.jsonl、qa.jsonl),找不到資料夾或
    # 資料夾是空的時候,會直接拋出清楚的錯誤訊息。
    text = load_corpus_text(config.data_dir)

    if os.path.exists(config.tokenizer_path):
        tokenizer = CharTokenizer.load(config.tokenizer_path)
        print(f"[train] 已載入既有 tokenizer,詞表大小: {tokenizer.vocab_size}")
    else:
        tokenizer = CharTokenizer.build_from_text(text)
        tokenizer.save(config.tokenizer_path)
        print(f"[train] 已建立新 tokenizer,詞表大小: {tokenizer.vocab_size}")

    # ---- 2. Dataset ----
    dataset = TextDataset(config, tokenizer)

    # ---- 3. Model & Optimizer ----
    model = GPTModel(config, vocab_size=tokenizer.vocab_size).to(config.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] 模型參數量: {n_params / 1e6:.2f}M,裝置: {config.device}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # 混合精度訓練:矩陣乘法用 float16 算,能有效加速 GPU 運算,
    # loss 計算等數值敏感的部分 autocast 會自動保留在 float32。
    # GradScaler 負責放大/縮小梯度避免 float16 動態範圍不足造成梯度變成 0。
    use_amp = config.use_amp and config.device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("[train] 已啟用混合精度訓練(AMP)")

    start_step = 0

    # ---- 續訓練:如果 config.resume=True 且已有 checkpoint,就接續之前的進度 ----
    if config.resume and os.path.exists(config.checkpoint_path):
        checkpoint = torch.load(config.checkpoint_path, map_location=config.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint.get("step", 0) + 1
        print(f"[train] 已從 checkpoint 接續訓練,起始步數: {start_step}")

    def save_checkpoint(step: int, val_loss: float):
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab_size": tokenizer.vocab_size,
            "step": step,
            "val_loss": val_loss,
            # 把訓練時「實際用到」的架構參數也存進 checkpoint,
            # 這樣之後即使 config.py 的預設值被改掉,
            # export_weights.py 依然能匯出正確、對應得上權重的架構設定,
            # 不會再發生「reshape 尺寸不合」這種錯誤。
            "architecture": {
                "n_embd": config.n_embd,
                "n_head": config.n_head,
                "n_layer": config.n_layer,
                "block_size": config.block_size,
            },
        }
        # 這個專案放在 OneDrive 同步資料夾底下,checkpoint 剛寫入時偶爾會被
        # OneDrive 短暫鎖住觸發同步,導致緊接著的下一次寫入失敗
        # (WinError 1224 / RuntimeError file with a user-mapped section)。
        # 這裡加上短暫重試,避免早停機制因為這種暫時性的檔案鎖定而整個訓練中斷。
        for attempt in range(5):
            try:
                torch.save(payload, config.checkpoint_path)
                return
            except RuntimeError:
                if attempt == 4:
                    raise
                time.sleep(1.5 * (attempt + 1))

    # ---- 4. 訓練迴圈 ----
    # 早停(early stopping):模型參數量相對語料量偏大時,train loss 會持續下降,
    # 但 val loss 過了某個點之後反而會回升(代表模型在背答案而不是學規律)。
    # 這裡不再是「無論如何都存最後一步」,而是只在 val loss 創新低的時候才
    # 存檔,checkpoint.pt 最終保留的會是驗證集表現最好的那個版本,能有效避免
    # 訓練跑越久、實際泛化能力反而越差的問題。
    best_val_loss = float("inf")
    best_step = start_step
    evals_without_improvement = 0
    stopped_early = False
    for step in range(start_step, config.max_iters):
        # 動態調整學習率(warmup + cosine decay)
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if step % config.eval_interval == 0 or step == config.max_iters - 1:
            losses = estimate_loss(model, dataset, config)
            is_best = losses["val"] < best_val_loss
            print(f"[step {step:5d}] train loss {losses['train']:.4f} | "
                  f"val loss {losses['val']:.4f} | lr {lr:.2e}"
                  + (" (新低,已存檔)" if is_best else ""))
            if is_best:
                best_val_loss = losses["val"]
                best_step = step
                evals_without_improvement = 0
                save_checkpoint(step, best_val_loss)
            else:
                evals_without_improvement += 1
                # 早停:連續好幾次評估都沒有創新低,代表已經開始過擬合,
                # 再跑下去只是浪費時間,checkpoint.pt 已經保留最佳版本,
                # 直接結束訓練迴圈即可。
                if evals_without_improvement >= config.early_stop_patience:
                    print(
                        f"[train] 連續 {config.early_stop_patience} 次評估都沒有創新低,"
                        f"提早結束訓練(第 {step} 步)"
                    )
                    stopped_early = True
                    break

        x, y = dataset.get_batch("train")

        with torch.amp.autocast("cuda", enabled=use_amp):
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        # 梯度裁剪:避免某一步梯度過大把權重炸壞。用 AMP 時要先 unscale
        # 梯度才能算出正確的梯度範數,不然裁剪的門檻會被 scaler 的縮放係數打亂。
        if config.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        scaler.step(optimizer)
        scaler.update()

    print(f"[train] 訓練完成{'(提早結束)' if stopped_early else ''},"
          f"最佳 checkpoint 在第 {best_step} 步"
          f"(val loss {best_val_loss:.4f}),已儲存至: {config.checkpoint_path}")


if __name__ == "__main__":
    train()
