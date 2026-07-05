"""
prepare_sft_data.py
--------------------
把 qa.txt(問:.../答:...)和 chat.txt(A:.../B:...)這種已經有明確
「一問一答」結構的語料,解析成結構化的 JSONL 格式,存成 sft_data.jsonl。

每一行 JSONL 都是一筆 {"input": "...", "output": "..."} 的資料,
input 是「問題/上下文」,output 是「該學會生成的回答」。

之後 train_sft.py 會讀取這份 JSONL,只針對 output 的部分計算 loss,
讓模型學會「看到問題,就該認真回答」這個行為模式,而不是像
train.py 純接龍訓練那樣,不分青紅皂白地接續所有文字。

使用方式:
    python prepare_sft_data.py
"""

import json
import re
from config import Config


def parse_qa_file(path: str) -> list[dict]:
    """解析『問:...\n答:...』格式的檔案。"""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 用正規表達式抓出每一組「問:...」「答:...」
    matches = re.findall(r"問[:：](.+?)\n答[:：](.+?)(?=\n\n|\n問|\Z)", content, re.DOTALL)
    for question, answer in matches:
        question = question.strip()
        answer = answer.strip()
        if question and answer:
            pairs.append({
                "input": f"問:{question}\n答:",
                "output": answer,
            })
    return pairs


def parse_chat_file(path: str) -> list[dict]:
    """解析『A:...\nB:...』格式的對話檔案,把每一組 A/B 都當作一筆問答配對。"""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    for i in range(len(lines) - 1):
        line_a = lines[i]
        line_b = lines[i + 1]
        if line_a.startswith("A:") and line_b.startswith("B:"):
            question = line_a[2:].strip()
            answer = line_b[2:].strip()
            if question and answer:
                pairs.append({
                    "input": f"A:{question}\nB:",
                    "output": answer,
                })
    return pairs


def main():
    config = Config()
    all_pairs = []

    qa_path = "data/qa.txt"
    chat_path = "data/chat.txt"

    try:
        qa_pairs = parse_qa_file(qa_path)
        all_pairs.extend(qa_pairs)
        print(f"[prepare_sft_data] 從 {qa_path} 解析出 {len(qa_pairs)} 筆問答資料")
    except FileNotFoundError:
        print(f"[prepare_sft_data] 找不到 {qa_path},略過")

    try:
        chat_pairs = parse_chat_file(chat_path)
        all_pairs.extend(chat_pairs)
        print(f"[prepare_sft_data] 從 {chat_path} 解析出 {len(chat_pairs)} 筆對話資料")
    except FileNotFoundError:
        print(f"[prepare_sft_data] 找不到 {chat_path},略過")

    if not all_pairs:
        raise ValueError(
            "沒有解析出任何資料,請確認 data/qa.txt 和 data/chat.txt 的格式"
            "是否符合『問:...\\n答:...』或『A:...\\nB:...』的格式。"
        )

    output_path = config.sft_data_path
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"[prepare_sft_data] 已產生 {len(all_pairs)} 筆訓練資料,存至 {output_path}")


if __name__ == "__main__":
    main()
    