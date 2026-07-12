"""
text_cleanup.py
----------------
生成完文字之後的「後處理」工具。

背景問題:
模型在預訓練階段,把整份 chat.txt(包含 A:、B: 這些對話標記符號)
當作一般文字學過,加上生成時沒有「該停止」的機制,只會一直生成到
設定的字數上限,所以常常在回答完之後,繼續「幻想」出下一輪對話,
開頭又冒出一個新的 A:、B:、問:、答: 之類的標記。

解法:
不需要重新訓練,只要在生成完的文字裡,找到「第一個看起來像是
新一輪對話開頭」的標記,把它跟後面的內容整個切掉,只保留前面
真正屬於這次回答的部分。
"""

import re

# 這些標記通常代表「新一輪對話開始」,一旦在生成內容裡看到,
# 就代表模型已經開始幻想下一輪對話,後面的內容不該顯示給使用者。
TURN_MARKERS = [
    r"\nA[:：]",
    r"\nB[:：]",
    r"\n問[:：]",
    r"\n答[:：]",
]

_PATTERN = re.compile("|".join(TURN_MARKERS))


def truncate_at_next_turn(text: str) -> str:
    """
    找到文字裡「第一個新一輪對話標記」出現的位置,只保留它之前的內容。
    如果完全沒有出現這些標記,就原封不動回傳。
    """
    match = _PATTERN.search(text)
    if match:
        return text[:match.start()].rstrip()
    return text.rstrip()


def find_next_turn_marker(text: str):
    """
    回傳文字裡「第一個新一輪對話標記」的 match 物件(找不到則回傳 None)。
    給 server.py 的串流生成邏輯用,判斷要不要提早停止串流。
    """
    return _PATTERN.search(text)
