"""
nexora/eval.py
--------------
Nexora Native v1 評估工具。

用法：
    python -m nexora.eval                              # 用預設 checkpoint
    python -m nexora.eval --checkpoint path/to/ck.pt  # 指定 checkpoint
    python -m nexora.eval --prompt "你好"              # 只做生成
"""

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexora.config import NexoraConfig
from nexora.tokenizer import NexoraTokenizer
from nexora.model import NexoraModel
from nexora.dataset import NexoraDataset, build_token_corpus, split_corpus


@torch.no_grad()
def compute_val_metrics(
    model: NexoraModel,
    dataset: NexoraDataset,
    cfg: NexoraConfig,
    n_batches: int = 50,
) -> dict:
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = dataset.get_batch("val")
        _, loss = model(x, y)
        losses.append(loss.item())
    avg = sum(losses) / len(losses)
    return {"val_loss": avg, "val_ppl": math.exp(min(avg, 20))}


@torch.no_grad()
def generate(
    model: NexoraModel,
    tokenizer: NexoraTokenizer,
    prompt: str,
    cfg: NexoraConfig,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
) -> str:
    model.eval()
    ids = tokenizer.encode(prompt, add_bos=True)
    idx = torch.tensor([ids], dtype=torch.long, device=cfg.device)
    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens or cfg.max_new_tokens,
        temperature=temperature or cfg.temperature,
        top_k=top_k or cfg.top_k,
        top_p=top_p or cfg.top_p,
        eos_id=tokenizer.eos_id,
    )
    return tokenizer.decode(out[0].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    args = parser.parse_args()

    cfg = NexoraConfig()
    ck_path = args.checkpoint or cfg.best_checkpoint_path
    if not os.path.exists(ck_path):
        ck_path = cfg.checkpoint_path
    if not os.path.exists(ck_path):
        print(f"[eval] 找不到 checkpoint（{ck_path}），請先執行 nexora/train.py")
        return

    print(f"[eval] 載入 checkpoint: {ck_path}")
    ck = torch.load(ck_path, map_location="cpu")
    assert ck.get("nexora_version") == "v1"

    arch = ck["config"]
    loaded_cfg = NexoraConfig(
        n_embd=arch["n_embd"],
        n_head=arch["n_head"],
        n_layer=arch["n_layer"],
        block_size=arch["block_size"],
        vocab_size=arch["vocab_size"],
        dropout=arch.get("dropout", 0.0),
    )

    tokenizer = NexoraTokenizer.load(cfg.tokenizer_path)
    print(f"[eval] tokenizer vocab_size={tokenizer.vocab_size}")

    model = NexoraModel(loaded_cfg).to(loaded_cfg.device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print(f"[eval] 模型載入完成 (step={ck['step']}，最佳 val_loss={ck['best_val_loss']:.4f})")

    summary = model.param_summary()
    print(f"[eval] 總參數量: {summary['total']:,} ({summary['total']/1e6:.3f}M)")

    # 若指定了 prompt，只做生成
    if args.prompt:
        result = generate(model, tokenizer, args.prompt, loaded_cfg,
                          max_new_tokens=args.max_new_tokens,
                          temperature=args.temperature,
                          top_k=args.top_k,
                          top_p=args.top_p)
        print(f"\n[生成]\nprompt: {args.prompt!r}\noutput: {result!r}\n")
        return

    # ── val loss + perplexity ──
    print("\n[eval] 計算 validation metrics...")
    corpus = build_token_corpus(cfg.data_dir, tokenizer)
    train_data, val_data, test_data = split_corpus(corpus, cfg.train_frac, cfg.val_frac)
    dataset = NexoraDataset(train_data, val_data, test_data, loaded_cfg)

    metrics = compute_val_metrics(model, dataset, loaded_cfg)
    print(f"  val_loss = {metrics['val_loss']:.4f}")
    print(f"  val_ppl  = {metrics['val_ppl']:.2f}")

    # ── 固定 prompt 生成 ──
    print("\n[eval] 生成測試：")
    for prompt in cfg.eval_prompts:
        result = generate(model, tokenizer, prompt, loaded_cfg)
        print(f"  [{prompt!r}] → {result!r}")


if __name__ == "__main__":
    main()
