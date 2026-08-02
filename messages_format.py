"""
messages_format.py
-------------------
data/ 底下的語料統一存成標準的 messages JSONL 格式(一行一筆 JSON):

    {"messages": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好!"}]}
    {"messages": [{"role": "user", "content": "寫一段程式"}, {"role": "assistant", "content": "..."}]}

一個檔案是一份 JSONL,每一行是一段完整的對話(可以只有一問一答,
也可以是多輪)。用 JSONL 而不是單一大陣列,新增語料時只要在檔案
最後加一行,不用整份重新解析、重新寫入整個陣列。這個模組負責讀取
這些檔案,以及把 messages 轉成模型實際訓練/推論時用的內部格式:
「問:...\n答:...」這種標記,跟 inference.py / conversation.py /
server.py 裡生成時判斷「該不該停止」用的標記(見 text_cleanup.py 的
TURN_MARKERS)保持一致。
"""

import json
import os
import glob


def load_conversations(data_dir: str) -> list[list[dict]]:
    """
    讀取 data_dir 底下所有 .jsonl 檔案,回傳所有對話的 messages 列表
    (每個元素是一段對話的 messages,例如 [{"role":"user","content":...}, ...])。
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"找不到語料資料夾: {data_dir}\n"
            "請建立這個資料夾,並在裡面放入至少一個 .jsonl 語料檔。"
        )

    jsonl_files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    if not jsonl_files:
        raise FileNotFoundError(
            f"{data_dir} 資料夾底下沒有任何 .jsonl 檔案,請至少放入一個語料檔"
            "(格式見 messages_format.py 說明)。"
        )

    conversations: list[list[dict]] = []
    for path in jsonl_files:
        line_count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                line_count += 1
                messages = item.get("messages") or []
                messages = [m for m in messages if (m.get("content") or "").strip()]
                if messages:
                    conversations.append(messages)
        print(f"[messages_format] 已讀取語料檔: {path} ({line_count} 段對話)")

    return conversations


def render_messages(messages: list[dict]) -> str:
    """
    把一段對話的 messages,轉成「問:...\n答:...」這種內部訓練/推論用的文字。
    user -> 問:, assistant -> 答:,依序接起來,每則訊息後面接一個換行。
    """
    lines = []
    for m in messages:
        tag = "問:" if m.get("role") == "user" else "答:"
        content = (m.get("content") or "").strip()
        if content:
            lines.append(f"{tag}{content}")
    return "\n".join(lines)
