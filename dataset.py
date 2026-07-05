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


def load_corpus_text(data_dir: str) -> str:
    """
    讀取 data_dir 底下所有 .txt 檔案,依檔名排序後合併成一份完整文字。
    這樣可以把語料依用途拆成多個檔案管理(例如 chat.txt、story.txt、
    qa.txt、code.txt),彼此互不影響,之後要增減某一類語料也很清楚。
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"找不到語料資料夾: {data_dir}\n"
            "請建立這個資料夾,並在裡面放入至少一個 .txt 檔案。"
        )

    txt_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    if not txt_files:
        raise FileNotFoundError(
            f"{data_dir} 資料夾底下沒有任何 .txt 檔案,請至少放入一個語料檔。"
        )

    texts = []
    for path in txt_files:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        texts.append(content)
        print(f"[dataset] 已讀取語料檔: {path} ({len(content)} 字元)")

    # 用換行分隔不同檔案的內容,避免前一個檔案的結尾和下一個檔案的開頭黏在一起
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
        """
        隨機取出一個 batch。
        split: "train" 或 "val"
        回傳: x, y,兩者形狀皆為 (batch_size, block_size)
              y 是 x 往右移一位(下一個字元預測任務的標籤)
        """
        data = self.train_data if split == "train" else self.val_data
        block_size = self.config.block_size
        batch_size = self.config.batch_size

        if len(data) <= block_size:
            raise ValueError(
                f"資料長度({len(data)})小於 block_size({block_size}),"
                "請提供更長的語料或調小 block_size。"
            )

        # 隨機選取 batch_size 個起始點
        ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[i: i + block_size] for i in ix])
        y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])

        x, y = x.to(self.config.device), y.to(self.config.device)
        return x, y


class SFTDataset:
    """
    監督式微調(Supervised Fine-Tuning, SFT)用的資料集。

    跟 TextDataset 最大的不同:
    - TextDataset 是從一整條長文字裡隨機裁切片段,不分「問題」和「答案」。
    - SFTDataset 讀取的是 prepare_sft_data.py 產生的 JSONL,每筆資料都
      清楚分成 input(問題/上下文)和 output(答案),訓練時只會針對
      output 的部分計算 loss,input 的部分會被標成 -100(不計入 loss)。

    這裡刻意簡化成 batch_size = 1(一次只處理一筆樣本),因為每筆資料的
    長度都不一樣,要做批次的話還要處理 padding 和額外的遮罩邏輯,
    對教學用途的專案來說,batch_size = 1 更簡單、更不容易寫錯,
    缺點是訓練速度會慢一點,但因為 SFT 階段的資料量通常不大,影響有限。
    """

    def __init__(self, config: Config, tokenizer: CharTokenizer, jsonl_path: str):
        self.config = config
        self.tokenizer = tokenizer
        self.examples: list[tuple[str, str]] = []

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(
                f"找不到 {jsonl_path},請先執行「python prepare_sft_data.py」產生這份檔案。"
            )

        import json
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                self.examples.append((item["input"], item["output"]))

        if not self.examples:
            raise ValueError(f"{jsonl_path} 裡沒有任何資料,請確認 qa.txt / chat.txt 內容格式正確。")

        print(f"[dataset] 已讀取 {len(self.examples)} 筆 SFT 訓練樣本")

    def get_batch(self, split: str = "train"):
        """
        隨機取一筆 (input, output) 配對,編碼成 (x, y),
        y 裡屬於「問題」的位置會被標成 -100,訓練時會被自動忽略。

        split 參數保留是為了跟 TextDataset 介面一致,SFT 階段目前
        不特別切分驗證集,因為資料量通常較小,拆分意義不大。
        """
        import random
        input_text, output_text = random.choice(self.examples)

        input_ids = self.tokenizer.encode(input_text)
        output_ids = self.tokenizer.encode(output_text)
        full_ids = input_ids + output_ids

        block_size = self.config.block_size

        # 如果整段(問題+答案)超過 block_size,從左邊截斷,優先保留答案部分
        if len(full_ids) > block_size + 1:
            overflow = len(full_ids) - (block_size + 1)
            full_ids = full_ids[overflow:]
            input_len = max(0, len(input_ids) - overflow)
        else:
            input_len = len(input_ids)

        x_ids = full_ids[:-1]
        y_ids = full_ids[1:]

        # y_ids 的索引 i,對應原始序列位置 i+1;
        # 只要這個位置還落在「問題」範圍內(< input_len),就標成 -100,不計入 loss。
        y_ids = [
            (-100 if (i + 1) < input_len else token_id)
            for i, token_id in enumerate(y_ids)
        ]

        x = torch.tensor([x_ids], dtype=torch.long, device=self.config.device)
        y = torch.tensor([y_ids], dtype=torch.long, device=self.config.device)
        return x, y


if __name__ == "__main__":
    # 簡單自我測試:需要先有一個 data/ 資料夾,裡面至少一個 .txt 檔案
    cfg = Config()
    os.makedirs(cfg.data_dir, exist_ok=True)
    sample_path = os.path.join(cfg.data_dir, "_sample.txt")
    if not any(glob.glob(os.path.join(cfg.data_dir, "*.txt"))):
        # 若資料夾是空的,先建立一份小範例方便測試
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write("你好世界" * 200)

    text = load_corpus_text(cfg.data_dir)
    tok = CharTokenizer.build_from_text(text)
    ds = TextDataset(cfg, tok)
    x, y = ds.get_batch("train")
    print("x shape:", x.shape, "y shape:", y.shape)
