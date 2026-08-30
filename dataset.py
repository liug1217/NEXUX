"""
dataset.py
----------
負責:
1. 讀取語料文字檔
2. 切分成 train / validation
3. 提供 get_batch() 隨機取出訓練用的 (輸入, 標籤) 配對

這裡不使用 torch.utils.data.Dataset + DataLoader 的完整寫法,
而是採用「語言模型訓練最常見」的隨機取樣方式,直接從長序列中裁切片段,
效能更好、程式碼也更精簡。
"""

import os
import glob
import torch
from config import Config
from tokenizer import CharTokenizer
from messages_format import load_conversations, render_messages


def load_corpus_text(data_dir: str) -> str:
    """
    讀取 data_dir 底下所有 .jsonl 語料(見 messages_format.py),把每段對話的
    messages 轉成「問:...\n答:...」文字,再合併成一份完整文字給預訓練階段
    (TextDataset)當作純接龍語料使用。

    這樣即使語料本身存成結構化的 messages 格式,預訓練階段依然能看到跟
    SFT 階段、跟推論時(inference.py / conversation.py)一致的「問:/答:」
    標記,學到的語感才會跟後面的階段接得上。
    """
    conversations = load_conversations(data_dir)
    texts = [render_messages(messages) for messages in conversations]
    texts = [t for t in texts if t]
    return "\n".join(texts)


class TextDataset:
    def __init__(self, config: Config, tokenizer: CharTokenizer):
        self.config = config
        self.tokenizer = tokenizer

        text = load_corpus_text(config.data_dir)

        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

        split_idx = int(len(data) * config.train_split)
        self.train_data = data[:split_idx]
        self.val_data = data[split_idx:]

        print(f"[dataset] 全文長度: {len(text)} 字元")
        print(f"[dataset] 訓練集: {len(self.train_data)} tokens, "
              f"驗證集: {len(self.val_data)} tokens")

    def get_batch(self, split: str = "train"):
        data = self.train_data if split == "train" else self.val_data
        block_size = self.config.block_size
        batch_size = self.config.batch_size

        if len(data) <= block_size:
            raise ValueError(
                f"資料長度({len(data)})小於 block_size({block_size}),"
                "請提供更長的語料或調小 block_size。"
            )

        ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[i: i + block_size] for i in ix])
        y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])

        x, y = x.to(self.config.device), y.to(self.config.device)
        return x, y


class SFTDataset:
    """
    監督式微調(Supervised Fine-Tuning, SFT)用的資料集。

    所有 tokenize 在 __init__ 一次完成,訓練時直接查表取 token ids,
    不再每步重新 encode,大幅減少 CPU 瓶頸。
    """

    def __init__(self, config: Config, tokenizer: CharTokenizer, jsonl_path: str):
        self.config = config
        self.tokenizer = tokenizer

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(
                f"找不到 {jsonl_path},請先執行「python prepare_sft_data.py」產生這份檔案。"
            )

        import json
        import time
        t0 = time.time()
        raw_examples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                raw_examples.append((item["input"], item["output"]))

        if not raw_examples:
            raise ValueError(f"{jsonl_path} 裡沒有任何資料,請確認 data/*.jsonl 內容格式正確。")

        eos_id = getattr(tokenizer, "eos_id", None)
        block_size = config.block_size

        self.tokenized: list[tuple[list[int], list[int]]] = []
        for input_text, output_text in raw_examples:
            input_ids = tokenizer.encode(input_text)
            output_ids = tokenizer.encode(output_text)
            if eos_id is not None:
                output_ids = output_ids + [eos_id]

            full_ids = input_ids + output_ids
            if len(full_ids) > block_size + 1:
                overflow = len(full_ids) - (block_size + 1)
                full_ids = full_ids[overflow:]
                input_len = max(0, len(input_ids) - overflow)
            else:
                input_len = len(input_ids)

            x_ids = full_ids[:-1]
            y_ids = [
                (-100 if (i + 1) < input_len else token_id)
                for i, token_id in enumerate(full_ids[1:])
            ]
            self.tokenized.append((x_ids, y_ids))

        self.examples = raw_examples

        import random
        self._rng = random
        self._shuffled_indices: list[int] = []
        self._cursor = 0
        self._epoch = 0
        self._reshuffle()

        elapsed = time.time() - t0
        print(f"[dataset] 已讀入並預先 tokenize {len(self.tokenized)} 筆 SFT 訓練樣本 ({elapsed:.1f}s)")

    def _reshuffle(self):
        self._shuffled_indices = list(range(len(self.tokenized)))
        self._rng.shuffle(self._shuffled_indices)
        self._cursor = 0
        self._epoch += 1

    def get_batch(self, split: str = "train", batch_size: int = 1):
        """
        取出 batch_size 筆資料,右側補 padding 對齊長度。
        """
        samples_x = []
        samples_y = []
        max_len = 0

        for _ in range(batch_size):
            if self._cursor >= len(self._shuffled_indices):
                self._reshuffle()
            idx = self._shuffled_indices[self._cursor]
            self._cursor += 1

            x_ids, y_ids = self.tokenized[idx]
            samples_x.append(x_ids)
            samples_y.append(y_ids)
            if len(x_ids) > max_len:
                max_len = len(x_ids)

        for i in range(len(samples_x)):
            pad_len = max_len - len(samples_x[i])
            if pad_len > 0:
                samples_x[i] = samples_x[i] + [0] * pad_len
                samples_y[i] = samples_y[i] + [-100] * pad_len

        x = torch.tensor(samples_x, dtype=torch.long, device=self.config.device)
        y = torch.tensor(samples_y, dtype=torch.long, device=self.config.device)
        return x, y


if __name__ == "__main__":
    cfg = Config()
    os.makedirs(cfg.data_dir, exist_ok=True)
    sample_path = os.path.join(cfg.data_dir, "_sample.jsonl")
    if not any(glob.glob(os.path.join(cfg.data_dir, "*.jsonl"))):
        import json
        sample = {"messages": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好世界" * 50},
        ]}
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    text = load_corpus_text(cfg.data_dir)
    tok = CharTokenizer.build_from_text(text)
    ds = TextDataset(cfg, tok)
    x, y = ds.get_batch("train")
    print("x shape:", x.shape, "y shape:", y.shape)
