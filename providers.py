"""
providers.py
-------------
統一包裝幾家第三方 AI API,讓 server.py / api/generate.py 可以用同一種方式呼叫:

    text = call_provider("openai", prompt)

金鑰一律從環境變數讀取(OPENAI_API_KEY / ANTHROPIC_API_KEY /
GOOGLE_AI_API_KEY / GROQ_API_KEY),本機開發時由 .env 檔案提供(見 server.py),
Vercel 上則要在專案的 Environment Variables 設定裡另外加同樣名稱的變數。

這幾個 provider 都是暫時性質,用來在自己的模型還沒訓練完成前先頂著用,
不是這個專案的長期核心功能。
"""

import os

SUPPORTED_PROVIDERS = ["own", "openai", "anthropic", "google", "groq"]


class ProviderError(Exception):
    """呼叫第三方 API 時發生的錯誤,訊息會直接顯示給使用者看。"""


def call_provider(provider: str, prompt: str) -> str:
    if provider == "openai":
        return _call_openai(prompt)
    if provider == "anthropic":
        return _call_anthropic(prompt)
    if provider == "google":
        return _call_google(prompt)
    if provider == "groq":
        return _call_groq(prompt)
    raise ProviderError(f"不支援的 provider: {provider}")


def _require_key(env_name: str) -> str:
    key = os.environ.get(env_name)
    if not key:
        raise ProviderError(f"伺服器沒有設定 {env_name},無法呼叫這個 API。")
    return key


def _call_openai(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=_require_key("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content or ""


def _call_anthropic(prompt: str) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=_require_key("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def _call_google(prompt: str) -> str:
    from google import genai

    client = genai.Client(api_key=_require_key("GOOGLE_AI_API_KEY"))
    resp = client.models.generate_content(model="gemini-flash-latest", contents=prompt)
    return resp.text or ""


def _call_groq(prompt: str) -> str:
    # Groq 提供跟 OpenAI 相容的 API 格式,直接沿用 openai 套件,
    # 只是換一個 base_url,不用額外裝 groq 專用套件。
    from openai import OpenAI

    client = OpenAI(
        api_key=_require_key("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content or ""
