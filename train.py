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
    # load_corpus_text 會讀取 data_dir 底下所有 .txt 檔案並合併
    # (例如 chat.txt、story.txt、qa.txt、code.txt),找不到資料夾或
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

    start_step = 0

    # ---- 續訓練:如果 config.resume=True 且已有 checkpoint,就接續之前的進度 ----
    if config.resume and os.path.exists(config.checkpoint_path):
        checkpoint = torch.load(config.checkpoint_path, map_location=config.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint.get("step", 0) + 1
        print(f"[train] 已從 checkpoint 接續訓練,起始步數: {start_step}")

    def save_checkpoint(step: int):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "vocab_size": tokenizer.vocab_size,
                "step": step,
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
            },
            config.checkpoint_path,
        )

    # ---- 4. 訓練迴圈 ----
    for step in range(start_step, config.max_iters):
        # 動態調整學習率(warmup + cosine decay)
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if step % config.eval_interval == 0 or step == config.max_iters - 1:
            losses = estimate_loss(model, dataset, config)
            print(f"[step {step:5d}] train loss {losses['train']:.4f} | "
                  f"val loss {losses['val']:.4f} | lr {lr:.2e}")

        x, y = dataset.get_batch("train")
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 梯度裁剪:避免某一步梯度過大把權重炸壞
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        optimizer.step()

        # 定期額外存檔,避免訓練中斷時全部進度遺失
        if config.save_interval > 0 and step > 0 and step % config.save_interval == 0:
            save_checkpoint(step)
            print(f"[train] 已於第 {step} 步儲存中繼 checkpoint")

    # ---- 5. 儲存最終模型 ----
    save_checkpoint(config.max_iters - 1)
    print(f"[train] 訓練完成,模型已儲存至: {config.checkpoint_path}")


if __name__ == "__main__":
    train()
