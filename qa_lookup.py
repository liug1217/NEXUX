"""
qa_lookup.py
------------
針對「已經有標準答案」的內容(見 _LOOKUP_FILES:qa.jsonl / html.jsonl /
python.jsonl / css.jsonl / javascript.jsonl / logic.jsonl /
troubleshooting.jsonl 這幾類單輪問答語料),在丟給模型生成之前,先比對
使用者的輸入跟語料庫裡的問題像不像。像的話直接回傳訓練資料裡的答案,
完全不經過模型生成。

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

回覆變化:
單純查表回傳一模一樣的句子,會讓人覺得很像機械式罐頭回覆(不像
smalltalk.py 那樣每個類別都準備多種講法、隨機挑一種回)。核心事實內容
不能亂改(改了可能就不準確了),所以做法是:只在自然語言問答類的答案
(qa.jsonl、以及css/javascript/logic/troubleshooting這幾類「說明文字
+ 內嵌程式碼片段」的散文格式答案,不包含html/python的純程式碼答案)
前面,隨機加一句不影響原意的開場白,讓同一題不會每次都一字不差,
同時保留答案本身的正確性。

程式碼答案包 ```:
html.jsonl / python.jsonl 的答案本來就是程式碼,回傳前用 ```html /
```python 包起來,讓 NEXUX.html 前端的「AI回覆有```程式碼```區塊時
自動彈出編輯器面板」機制可以偵測到、彈出面板。這是規則式比對回傳的
固定答案(不是模型生成),保證一定會正確包上,不用靠訓練模型學會
輸出```格式(小模型不保證學得會這種格式規則)。
"""

import json
import os
import random

_THRESHOLD = 0.5
_LOOKUP_FILES = (
    "qa.jsonl",
    "html.jsonl",
    "python.jsonl",
    "css.jsonl",
    "javascript.jsonl",
    "logic.jsonl",
    "troubleshooting.jsonl",
)

# 只套用在自然語言問答(qa.jsonl 以及新加入的 css/javascript/logic/
# troubleshooting),這幾類答案本身是「說明文字 + 內嵌程式碼片段」的
# 散文格式,不是像 html/python 那樣整個答案都是純程式碼,所以不放進
# _FENCE_LANG、允許加開場白 —— 如果硬把整段散文包進```程式碼區塊```,
# 反而會讓說明文字跟程式碼片段混在一起變得很奇怪。
_VARIABLE_OPENERS = [
    "",  # 保留原樣不加開場白,佔比較高的權重(見下方 weights)
    "簡單來說,",
    "這個問題很常被問到,",
    "我來說明一下,",
    "以我所知,",
]
_OPENER_WEIGHTS = [5, 1, 1, 1, 1]  # 大部分時候不加,偶爾加一點變化

_entries: list[tuple[str, str, str | None]] | None = None  # (question, answer, fence_lang),延遲載入

# 檔案名稱對應要包的```語言標籤,None代表qa.jsonl這類自然語言問答,
# 不包```(也才會允許加開場白,見 _load_entries 的 allow_opener 邏輯)。
_FENCE_LANG = {"html.jsonl": "html", "python.jsonl": "python"}


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


def _load_entries(data_dir: str) -> list[tuple[str, str, str | None]]:
    entries = []
    for filename in _LOOKUP_FILES:
        fence_lang = _FENCE_LANG.get(filename)  # None(qa.jsonl)或"html"/"python"
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
                    entries.append((question, answer, fence_lang))
    return entries


def match_qa(prompt: str, data_dir: str = "data") -> str | None:
    """
    輸入使用者的原始 prompt,如果跟語料庫裡某一題的相似度夠高,
    回傳那一題訓練資料裡的答案(自然語言問答會隨機加一句開場白,增加
    一點變化;html/python的程式碼答案改包上```html/```python,讓前端
    的程式碼編輯器面板機制能偵測到);否則回傳 None。
    """
    global _entries
    if _entries is None:
        _entries = _load_entries(data_dir)

    prompt = prompt.strip()
    if not prompt:
        return None

    best_score = 0.0
    best_answer = None
    best_fence_lang = None
    for question, answer, fence_lang in _entries:
        score = _similarity(prompt, question)
        if score > best_score:
            best_score = score
            best_answer = answer
            best_fence_lang = fence_lang

    if best_score < _THRESHOLD:
        return None

    if best_fence_lang is not None:
        return f"```{best_fence_lang}\n{best_answer}\n```"

    opener = random.choices(_VARIABLE_OPENERS, weights=_OPENER_WEIGHTS, k=1)[0]
    return opener + best_answer
