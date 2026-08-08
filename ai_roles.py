"""
ai_roles.py
-----------
NexoraAI 第一階段(composer 角色選擇 + 多角色團隊回覆)用的角色定義。

這裡定義的「角色」本質上是同一個 own 小模型,用不同的角色扮演提示語
各問一次——不是真正各自獨立思考、有記憶的智慧體,角色之間也不會互相
看到彼此的回覆再討論。這個限制在 NEXUX.html 的團隊回覆區塊會誠實標示
給使用者看,不誇大宣傳成真正的多智慧體協作(詳見
C:\\Users\\liug1\\.claude\\plans\\squishy-coalescing-coral.md)。

角色扮演提示語風格(「你現在扮演一位XX AI,請...」)跟模型 SFT 訓練時
唯一看過的「問:.../答:...」格式不同,已經用 server.get_model_and_tokenizer()
實測過:跟一般直接提問比,不連貫程度沒有明顯變差(兩者對開放式問題本來
就都偏語意破碎,這是目前模型規模的既有限制,不是這組提示語造成的退步),
所以照原本設計的措辭沿用,不需要簡化。
"""

ROLES = {
    "engineer": {
        "label": "工程師 AI",
        "icon": "💻",
        "prompt_template": (
            "你現在扮演一位工程師AI,請針對以下需求,"
            "從技術與架構的角度提出你的專業建議:\n{prompt}"
        ),
    },
    "designer": {
        "label": "設計師 AI",
        "icon": "🎨",
        "prompt_template": (
            "你現在扮演一位設計師AI,請針對以下需求,"
            "從UI/UX與使用者體驗的角度提出你的專業建議:\n{prompt}"
        ),
    },
    "security": {
        "label": "資安 AI",
        "icon": "🛡️",
        "prompt_template": (
            "你現在扮演一位資安AI,請針對以下需求,"
            "從安全性與風險的角度提出你的專業建議:\n{prompt}"
        ),
    },
    "analyst": {
        "label": "分析師 AI",
        "icon": "📊",
        "prompt_template": (
            "你現在扮演一位分析師AI,請針對以下需求,"
            "從資料與市場分析的角度提出你的專業建議:\n{prompt}"
        ),
    },
}

INTEGRATION_PROMPT_TEMPLATE = (
    "以下是團隊裡不同專業角色,針對「{prompt}」這個需求各自提出的意見:\n"
    "{combined_responses}\n"
    "請你身為核心AI,用兩三句話整合出一個簡短的最終建議。"
)


def build_role_prompt(role: str, user_prompt: str) -> str:
    """回傳指定角色的完整提示語(還沒包上「問:.../答:」格式,由呼叫端負責包)。"""
    return ROLES[role]["prompt_template"].format(prompt=user_prompt)


def build_integration_prompt(user_prompt: str, role_replies: list[tuple[str, str]]) -> str:
    """
    role_replies: [(label, reply), ...] 每個角色的顯示名稱跟回覆內容。
    回傳核心AI整合提示語的完整文字(同樣還沒包「問:.../答:」)。
    """
    combined = "\n".join(f"【{label}】{reply}" for label, reply in role_replies)
    return INTEGRATION_PROMPT_TEMPLATE.format(prompt=user_prompt, combined_responses=combined)
