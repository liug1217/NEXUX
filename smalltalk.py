"""
smalltalk.py
------------
針對「哈囉」「掰掰」「謝謝你」這類極短、高頻出現的問候/寒暄輸入,
用簡單的關鍵字比對直接回覆固定語料庫裡的句子,不經過模型生成。

背景:
目前的模型規模很小(char-level、不到千萬參數),對這種短輸入
沒辦法穩定分辨「哈囉」「掰掰」「謝謝你」這些完全不同語境的差異,
常常每一句都固定回覆同一個開頭字(例如「嗨」),即使輸入其實是道別
或道謝。與其花更多時間在模型規模的天花板上打轉,不如把這種
高頻、低變化的短輸入,直接交給規則比對,確保回覆一定得體;
真正需要理解與生成的問題,還是交給模型處理。

只有在輸入「夠短」且「明確符合」某個類別關鍵字時才會觸發,
避免不小心誤判掉包含這些字但其實是完整問題的長句(例如
「謝謝你可以解釋一下什麼是區塊鏈嗎」不會被誤判成單純道謝)。
"""

import random
from datetime import datetime, timedelta, timezone

# 輸入字數超過這個長度,就不當作單純的寒暄短句處理,交給模型生成。
MAX_SMALLTALK_LEN = 12

# 使用者主要是台灣繁體中文使用者,「現在幾點」這類問題,回覆用台灣時區(UTC+8)。
_TAIWAN_TZ = timezone(timedelta(hours=8))

_CATEGORIES: list[tuple[str, list[str]]] = [
    ("time", ["幾點"]),
    ("farewell", ["掰掰", "再見", "拜拜", "先走了", "先這樣"]),
    ("thanks", ["謝謝", "感謝", "多謝"]),
    ("apology", ["對不起", "抱歉", "不好意思"]),
    ("morning", ["早安", "早上好", "早,"]),
    ("night", ["晚安"]),
    ("noon", ["午安"]),
    ("greeting", ["哈囉", "嗨", "你好", "hi", "hello"]),
]

_REPLIES: dict[str, list[str]] = {
    "farewell": ["掰掰,下次再聊。", "再見,路上小心。", "掰掰,保重。", "好,掰掰,慢走。"],
    "thanks": ["不客氣,能幫上忙就好。", "不會,舉手之勞而已。", "不客氣,有需要再說一聲。"],
    "apology": ["沒關係,不用在意。", "沒事,你別放在心上。", "沒關係,下次注意就好。"],
    "morning": ["早安,今天也要加油喔。", "早,今天天氣看起來不錯。"],
    "night": ["晚安,好好休息。", "晚安,做個好夢。"],
    "noon": ["午安,吃飽了嗎?"],
    "greeting": [
        "嗨,很高興認識你。",
        "哈囉,最近過得好嗎?",
        "嗨,今天過得如何?",
        "很高興見到你,有什麼想聊的都可以說喔。",
    ],
}


def _current_time_reply() -> str:
    """回傳台灣時區當下的時間,例如「現在是下午3點25分。」。"""
    now = datetime.now(_TAIWAN_TZ)
    period = "上午" if now.hour < 12 else "下午"
    hour12 = now.hour % 12 or 12
    return f"現在是{period}{hour12}點{now.minute}分。"


def match_smalltalk_reply(prompt: str) -> str | None:
    """
    輸入使用者的原始 prompt,如果符合某個寒暄類別就回傳一句回覆,
    否則回傳 None(代表應該交給模型正常生成)。
    """
    text = prompt.strip()
    if not text or len(text) > MAX_SMALLTALK_LEN:
        return None

    for category, keywords in _CATEGORIES:
        if any(kw in text for kw in keywords):
            if category == "time":
                return _current_time_reply()
            return random.choice(_REPLIES[category])

    return None
