"""
nexora/dataset.py
-----------------
Nexora Native pretraining 用的資料集。
不依賴 GPT-2 路線的任何程式碼，自己讀取 data/*.jsonl。
"""

import glob
import json
import os
import random
import torch

from nexora.config import NexoraConfig
from nexora.tokenizer import NexoraTokenizer


def load_all_text(data_dir: str) -> str:
    """
    讀取 data_dir 底下所有 .jsonl，把每段對話展開成純文字。
    格式：「問:\n{user}\n答:\n{assistant}」，多輪對話依序展開。
    對話之間加換行分隔。
    """
    lines: list[str] = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                    messages = obj.get("messages", [])
                    parts: list[str] = []
                    for msg in messages:
                        role = msg.get("role", "")
                        content = msg.get("content", "").strip()
                        if role == "user":
                            parts.append(f"問:\n{content}")
                        elif role == "assistant":
                            parts.append(f"答:\n{content}")
                    if parts:
                        lines.append("\n".join(parts))
                except Exception:
                    pass
    return "\n\n".join(lines)


def build_token_corpus(
    data_dir: str,
    tokenizer: NexoraTokenizer,
) -> torch.Tensor:
    """
    把所有語料轉成 token ID 序列。
    每段對話前加 <BOS>，結尾加 <EOS>。
    """
    all_ids: list[int] = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                    messages = obj.get("messages", [])
                    parts: list[str] = []
                    for msg in messages:
                        role = msg.get("role", "")
                        content = msg.get("content", "").strip()
                        if role == "user":
                            parts.append(f"問:\n{content}")
                        elif role == "assistant":
                            parts.append(f"答:\n{content}")
                    if parts:
                        text = "\n".join(parts)
                        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
                        all_ids.extend(ids)
                except Exception:
                    pass
    return torch.tensor(all_ids, dtype=torch.long)


def split_corpus(
    data: torch.Tensor,
    train_frac: float,
    val_frac: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """切分成 train / val / test。"""
    n = len(data)
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))
    return data[:t1], data[t1:t2], data[t2:]


class NexoraDataset:
    """
    Pretraining 用的資料集。
    從長序列中隨機裁切 block_size 長度的片段。
    """

    def __init__(
        self,
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        test_data: torch.Tensor,
        cfg: NexoraConfig,
    ):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.cfg = cfg

        print(f"[NexoraDataset] train={len(train_data):,}"
              f"  val={len(val_data):,}"
              f"  test={len(test_data):,} tokens")

    def get_batch(
        self,
        split: str = "train",
        batch_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data = {"train": self.train_data, "val": self.val_data, "test": self.test_data}[split]
        bs = batch_size or self.cfg.batch_size
        block = self.cfg.block_size

        if len(data) <= block:
            raise ValueError(
                f"split='{split}' 資料量({len(data)}) ≤ block_size({block})，資料太少。"
            )
        ix = torch.randint(0, len(data) - block, (bs,))
        x = torch.stack([data[i:i + block] for i in ix])
        y = torch.stack([data[i + 1:i + block + 1] for i in ix])
        return x.to(self.cfg.device), y.to(self.cfg.device)


def analyze_corpus(data_dir: str) -> dict:
    """分析語料中中英文比例（回傳統計字典）。"""
    cjk = ascii_cnt = digit = punct_cn = punct_en = other = total = 0
    for path in sorted(glob.glob(os.path.join(data_dir, "*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                    for msg in obj.get("messages", []):
                        for ch in msg.get("content", ""):
                            total += 1
                            if "一" <= ch <= "鿿":
                                cjk += 1
                            elif ch.isdigit():
                                digit += 1
                            elif ord(ch) < 128:
                                if ch.isalpha() or ch == " ":
                                    ascii_cnt += 1
                                else:
                                    punct_en += 1
                            elif ch in "，。！？；：「」『』【】、…—":
                                punct_cn += 1
                            else:
                                other += 1
                except Exception:
                    pass

    def pct(n): return f"{n/total:.1%}" if total else "0%"
    return {
        "total_chars": total,
        "cjk": (cjk, pct(cjk)),
        "ascii_alpha": (ascii_cnt, pct(ascii_cnt)),
        "digit": (digit, pct(digit)),
        "punct_cn": (punct_cn, pct(punct_cn)),
        "punct_en": (punct_en, pct(punct_en)),
        "other": (other, pct(other)),
    }
