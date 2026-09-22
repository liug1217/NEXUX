"""
nexora/train.py
---------------
Nexora Native v1 pretraining 主程式。

不使用任何外部 pretrained weights。
所有參數從隨機初始化開始，學習我們自己的語料。

用法：
    python -m nexora.train                      # 從頭訓練
    python -m nexora.train --resume             # 從最新 checkpoint 繼續
    python -m nexora.train --steps 5000         # 指定步數
    python -m nexora.train --smoke              # 快速 20 步冒煙測試
"""

import argparse
import math
import os
import sys
import time

import torch

# Windows cp950 終端機無法顯示部分 Unicode 字元，統一用 UTF-8 輸出
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", encoding="utf-8", buffering=1)

# 加入上層目錄到 path，讓 nexora/ 內部可以 import nexora.*
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexora.config import NexoraConfig
from nexora.tokenizer import NexoraTokenizer
from nexora.model import NexoraModel
from nexora.dataset import NexoraDataset, build_token_corpus, split_corpus, analyze_corpus


# ── 學習率 schedule ──────────────────────────────────────────────────────────

def get_lr(step: int, cfg: NexoraConfig) -> float:
    if step < cfg.warmup_iters:
        return cfg.learning_rate * (step + 1) / cfg.warmup_iters
    if step >= cfg.max_iters:
        return cfg.min_learning_rate
    ratio = (step - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_learning_rate + coeff * (cfg.learning_rate - cfg.min_learning_rate)


# ── 評估 ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(
    model: NexoraModel,
    dataset: NexoraDataset,
    cfg: NexoraConfig,
) -> dict[str, float]:
    model.eval()
    results = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(cfg.eval_iters):
            x, y = dataset.get_batch(split)
            _, loss = model(x, y)
            losses.append(loss.item())
        avg = sum(losses) / len(losses)
        results[split] = avg
        results[f"{split}_ppl"] = math.exp(min(avg, 20))
    model.train()
    return results


# ── 文字生成測試 ────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_samples(
    model: NexoraModel,
    tokenizer: NexoraTokenizer,
    cfg: NexoraConfig,
) -> None:
    model.eval()
    print("\n── 生成測試 " + "─" * 50)
    for prompt in cfg.eval_prompts:
        ids = tokenizer.encode(prompt, add_bos=True)
        idx = torch.tensor([ids], dtype=torch.long, device=cfg.device)
        out = model.generate(
            idx,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            eos_id=tokenizer.eos_id,
        )
        text = tokenizer.decode(out[0].tolist())
        print(f"  [{prompt!r}] → {text!r}")
    print("─" * 62 + "\n")
    model.train()


# ── Checkpoint ──────────────────────────────────────────────────────────────

def save_checkpoint(
    model: NexoraModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val_loss: float,
    cfg: NexoraConfig,
    path: str,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "nexora_version": "v1",
        "step": step,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "n_embd": cfg.n_embd,
            "n_head": cfg.n_head,
            "n_layer": cfg.n_layer,
            "block_size": cfg.block_size,
            "vocab_size": cfg.vocab_size,
            "dropout": cfg.dropout,
        },
    }, path)


def load_checkpoint(
    model: NexoraModel,
    optimizer: torch.optim.Optimizer,
    path: str,
) -> tuple[int, float]:
    ck = torch.load(path, map_location="cpu")
    assert ck.get("nexora_version") == "v1", (
        f"checkpoint 版本不符，預期 'v1'，實際 {ck.get('nexora_version')!r}"
    )
    model.load_state_dict(ck["model_state_dict"])
    optimizer.load_state_dict(ck["optimizer_state_dict"])
    return ck["step"], ck["best_val_loss"]


# ── 主程式 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="只跑 20 步，快速確認流程")
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    cfg = NexoraConfig()
    if args.smoke:
        cfg.max_iters = 20
        cfg.eval_interval = 10
        cfg.save_interval = 20
        cfg.generate_interval = 10
    elif args.steps:
        cfg.max_iters = args.steps

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # ── 1. 語料統計 ─────────────────────────────────────────────────────
    print("\n══ 語料分析 ════════════════════════════════════════════════")
    stats = analyze_corpus(cfg.data_dir)
    print(f"  總字元數: {stats['total_chars']:,}")
    print(f"  CJK:      {stats['cjk'][1]} ({stats['cjk'][0]:,})")
    print(f"  ASCII:    {stats['ascii_alpha'][1]} ({stats['ascii_alpha'][0]:,})")
    print(f"  數字:     {stats['digit'][1]}")
    print(f"  中文標點: {stats['punct_cn'][1]}")
    print(f"  英文標點: {stats['punct_en'][1]}")
    print(f"  其他:     {stats['other'][1]}")

    # ── 2. Tokenizer ────────────────────────────────────────────────────
    print("\n══ Tokenizer ═══════════════════════════════════════════════")
    if os.path.exists(cfg.tokenizer_path):
        print(f"  載入現有 tokenizer: {cfg.tokenizer_path}")
        tokenizer = NexoraTokenizer.load(cfg.tokenizer_path)
    else:
        print(f"  從語料建立新 tokenizer（vocab_size={cfg.vocab_size}）")
        tokenizer = NexoraTokenizer.build_from_data(cfg.data_dir, max_vocab=cfg.vocab_size)
        tokenizer.save(cfg.tokenizer_path)

    # ── 3. 建立 token corpus ─────────────────────────────────────────────
    print("\n══ 建立訓練資料 ════════════════════════════════════════════")
    corpus = build_token_corpus(cfg.data_dir, tokenizer)
    print(f"  Token 總數: {len(corpus):,}")
    train_data, val_data, test_data = split_corpus(corpus, cfg.train_frac, cfg.val_frac)
    dataset = NexoraDataset(train_data, val_data, test_data, cfg)

    unk_ratio = (corpus == tokenizer.UNK_ID).float().mean().item()
    print(f"  UNK 比例: {unk_ratio:.3%}")

    # ── 4. 建立模型（隨機初始化） ────────────────────────────────────────
    print("\n══ 建立 Nexora Native v1 Model ════════════════════════════")
    model = NexoraModel(cfg).to(cfg.device)

    summary = model.param_summary()
    print(f"\n  ── 參數量明細 ──")
    for k, v in summary.items():
        print(f"    {k:<25}: {v:>12,}")

    # ── 5. Optimizer ─────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp and cfg.device == "cuda")

    # ── 6. Resume ────────────────────────────────────────────────────────
    start_step = 0
    best_val_loss = float("inf")
    no_improve = 0

    if args.resume and os.path.exists(cfg.checkpoint_path):
        print(f"\n  從 checkpoint 繼續: {cfg.checkpoint_path}")
        start_step, best_val_loss = load_checkpoint(model, optimizer, cfg.checkpoint_path)
        print(f"  起始步數: {start_step}, 歷史最佳 val loss: {best_val_loss:.4f}")

    # ── 7. 訓練迴圈 ───────────────────────────────────────────────────────
    print(f"\n══ 開始訓練 ════════════════════════════════════════════════")
    print(f"  裝置: {cfg.device}  AMP: {cfg.use_amp and cfg.device=='cuda'}")
    print(f"  batch_size={cfg.batch_size}  grad_accum={cfg.grad_accumulation}"
          f"  有效batch={cfg.batch_size * cfg.grad_accumulation}")
    print(f"  total_steps={cfg.max_iters}  block_size={cfg.block_size}\n")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()

    for step in range(start_step, cfg.max_iters):
        # 學習率 schedule
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation
        accum_loss = 0.0
        for micro_step in range(cfg.grad_accumulation):
            x, y = dataset.get_batch("train")
            if cfg.use_amp and cfg.device == "cuda":
                with torch.amp.autocast("cuda"):
                    _, loss = model(x, y)
                loss_scaled = loss / cfg.grad_accumulation
                scaler.scale(loss_scaled).backward()
            else:
                _, loss = model(x, y)
                (loss / cfg.grad_accumulation).backward()
            accum_loss += loss.item() / cfg.grad_accumulation

        if cfg.use_amp and cfg.device == "cuda":
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        if cfg.use_amp and cfg.device == "cuda":
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        # ── 評估 ──
        if step % cfg.eval_interval == 0 or step == cfg.max_iters - 1:
            metrics = estimate_loss(model, dataset, cfg)
            dt = time.time() - t0
            t0 = time.time()
            print(
                f"step {step:6d}/{cfg.max_iters}"
                f" | train_loss={metrics['train']:.4f}"
                f" | val_loss={metrics['val']:.4f}"
                f" | val_ppl={metrics['val_ppl']:.1f}"
                f" | lr={lr:.2e}"
                f" | {dt:.1f}s"
            )

            if metrics["val"] < best_val_loss:
                best_val_loss = metrics["val"]
                no_improve = 0
                save_checkpoint(model, optimizer, step, best_val_loss, cfg, cfg.best_checkpoint_path)
                print(f"  ★ 新的最佳 val_loss={best_val_loss:.4f}，已存至 {cfg.best_checkpoint_path}")
            else:
                no_improve += 1
                if no_improve >= cfg.early_stop_patience and not args.smoke:
                    print(f"  早停：連續 {no_improve} 次評估 val loss 未改善，停止訓練。")
                    break

        # ── 生成測試 ──
        if step % cfg.generate_interval == 0 and step > 0:
            generate_samples(model, tokenizer, cfg)

        # ── 定期存 checkpoint ──
        if step % cfg.save_interval == 0 and step > 0:
            save_checkpoint(model, optimizer, step, best_val_loss, cfg, cfg.checkpoint_path)
            print(f"  [checkpoint 已存] step={step}")

    # ── 8. 最終儲存 ───────────────────────────────────────────────────────
    save_checkpoint(model, optimizer, cfg.max_iters, best_val_loss, cfg, cfg.checkpoint_path)
    print(f"\n══ 訓練完成 ════════════════════════════════════════════════")
    print(f"  最終 checkpoint: {cfg.checkpoint_path}")
    print(f"  最佳 val_loss:   {best_val_loss:.4f}")
    print(f"  最佳 checkpoint: {cfg.best_checkpoint_path}")

    # ── 最終生成測試 ──
    generate_samples(model, tokenizer, cfg)


if __name__ == "__main__":
    main()
