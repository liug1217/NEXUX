"""
prepare_sft_data.py
--------------------
把 data/ 底下所有 .jsonl 語料(標準 messages 格式,見 messages_format.py)
展開成 SFT 訓練用的 JSONL 格式,存成 sft_data.jsonl。

每一行 JSONL 都是一筆 {"input": "...", "output": "..."} 的資料,
input 是「問:.../答:...」組成的對話上文(包含這一輪的問題,結尾是「答:」),
output 是這一輪該學會生成的回答。

之後 train_sft.py 會讀取這份 JSONL,只針對 output 的部分計算 loss,
讓模型學會「看到問題,就該認真回答」這個行為模式,而不是像
train.py 純接龍訓練那樣,不分青紅皂白地接續所有文字。

跟舊版(直接用正規表達式解析 qa.txt / chat.txt / greeting.txt)不同,
現在所有語料都已經是結構化的 messages 格式,不需要再用正規表達式猜格式;
而且對於多輪對話(例如 chat.jsonl),每一輪回答都會把前面的對話歷史一併
包進 input,讓模型能學到「參考上文」的多輪對話能力,而不是每一輪都當成
獨立、沒有上下文的單輪問答。

使用方式:
    python prepare_sft_data.py
"""

import json
from config import Config
from messages_format import load_conversations


def conversation_to_examples(messages: list[dict]) -> list[dict]:
    """
    把一段對話的 messages,展開成多筆 {"input", "output"} 訓練樣本。
    每遇到一則 assistant 訊息,就用「目前為止的對話上文」當作 input,
    這則 assistant 訊息的內容當作 output,然後把這則回答也併入上文,
    繼續往下一輪展開。
    """
    examples = []
    context = ""
    for message in messages:
        role = message.get("role")
        content = (message.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            context += f"問:{content}\n答:"
        elif role == "assistant":
            if context.endswith("答:"):
                examples.append({"input": context, "output": content})
            context += f"{content}\n"
    return examples


def main():
    config = Config()

    conversations = load_conversations(config.data_dir)
    all_pairs = []
    for messages in conversations:
        all_pairs.extend(conversation_to_examples(messages))

    if not all_pairs:
        raise ValueError(
            f"沒有從 {config.data_dir} 底下的 .jsonl 語料展開出任何訓練資料,"
            "請確認每個檔案都是 [{\"messages\": [{\"role\":..., \"content\":...}, ...]}, ...] 格式。"
        )

    output_path = config.sft_data_path
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"[prepare_sft_data] 從 {len(conversations)} 段對話展開出 {len(all_pairs)} 筆訓練資料,存至 {output_path}")


if __name__ == "__main__":
    main()
