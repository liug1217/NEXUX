"""
qa_lookup.py
------------
針對「已經有標準答案」的內容(qa.jsonl / html.jsonl / python.jsonl 這三類
單輪問答語料),在丟給模型生成之前,先比對使用者的輸入跟語料庫裡的問題
像不像。像的話直接回傳訓練資料裡的答案,完全不經過模型生成。

背景:
現在的模型規模很小(char-level、7.49M參數、語料量對這個規模來說還遠遠
不夠),即使是訓練資料裡「一模一樣出現過」的問題,模型生成時也常常答
非所問。與其讓使用者看到亂猜的錯誤答案,不如對這些「本來就有標準答案」
的內容,直接用比對取代生成——保證正確,而且完全不用等模型跑。

比對方式:
用字元二元組(character bigram)算 Jaccard 相似度,不依賴斷詞(中文沒有
天然的詞界線,字元層級的比對對短句子來說簡單又夠用),輸入跟語料庫裡
每一題的問題比對,取相似度最高的一筆,超過門檻才回傳,否則回傳 None
(交給後面的模型生成或第三方 API 處理)。
"""

import json
import os
import glob

_THRESHOLD = 0.5
_LOOKUP_FILES = ("qa.jsonl", "html.jsonl", "python.jsonl")

_entries: list[tuple[str, str]] | None = None  # (question, answer) pairs,延遲載入


def _bigrams(text: str) -> set[str]:
    text = text.strip()
    if len(text) < 2:
        return {text} if text else set()
    return {text[i:i + 2] for i in range(len(text) - 1)}


def _similarity(a: str, b: str) -> float:
    set_a, set_b = _bigrams(a), _bigrams(b)
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def _load_entries(data_dir: str) -> list[tuple[str, str]]:
    entries = []
    for filename in _LOOKUP_FILES:
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                messages = item.get("messages") or []
                if len(messages) != 2:
                    continue
                user_msg, assistant_msg = messages
                question = (user_msg.get("content") or "").strip()
                answer = (assistant_msg.get("content") or "").strip()
                if question and answer:
                    entries.append((question, answer))
    return entries


def match_qa(prompt: str, data_dir: str = "data") -> str | None:
    """
    輸入使用者的原始 prompt,如果跟語料庫裡某一題的相似度夠高,
    回傳那一題訓練資料裡的答案;否則回傳 None。
    """
    global _entries
    if _entries is None:
        _entries = _load_entries(data_dir)

    prompt = prompt.strip()
    if not prompt:
        return None

    best_score = 0.0
    best_answer = None
    for question, answer in _entries:
        score = _similarity(prompt, question)
        if score > best_score:
            best_score = score
            best_answer = answer

    if best_score >= _THRESHOLD:
        return best_answer
    return None
