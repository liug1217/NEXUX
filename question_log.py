"""
question_log.py
----------------
把使用者在網站上實際輸入的問題,記錄到一個外部的 Redis(Upstash)資料庫,
方便之後回顧「真實使用者到底都問了什麼」,拿來檢查目前的模型/smalltalk
規則有沒有明顯答不好、或漏掉的常見講法,作為之後擴充語料的依據。

Vercel 的 Serverless Function 沒有可以持久寫入的本機硬碟,沒辦法像本機
開發時那樣直接寫進一個檔案,需要另外接一個外部的儲存服務。這裡選用
Upstash Redis 的 REST API,因為它可以直接用 HTTP 呼叫,不需要額外安裝
任何 SDK,符合 Vercel Serverless Function 對套件大小的限制。

設定方式:
1. 到 Vercel 專案的 Storage 頁籤,新增一個 Upstash for Redis 資料庫
   (或直接到 upstash.com 免費申請一個 Redis 資料庫)。
2. 把它提供的 REST URL 和 REST TOKEN,加進 Vercel 專案的
   Environment Variables:UPSTASH_REDIS_REST_URL、UPSTASH_REDIS_REST_TOKEN。
3. 不需要改任何程式碼,這個模組會自動偵測這兩個環境變數存不存在:
   存在就記錄,不存在就靜默略過(不會讓網站壞掉或回覆變慢)。

想查看蒐集到的問題,直接登入 Upstash 的網頁主控台(Data Browser),
搜尋 key 叫 "collected_questions" 的 list 即可,不需要額外做管理介面。

隱私提醒:這裡會把使用者輸入的原始文字存起來,如果朋友測試時可能會
打進一些比較私人的內容,分享測試連結時記得先讓對方知道。
"""

import json
import os
import time
import urllib.error
import urllib.request

_REDIS_KEY = "collected_questions"
_TIMEOUT_SECONDS = 3


def log_question(prompt: str, provider: str) -> None:
    """
    盡量不影響使用者體驗地,把這次的提問記錄下來。
    任何失敗情況(沒設定環境變數、網路逾時、Upstash 回錯誤)都只會靜默
    略過,絕對不能因為記錄失敗就讓使用者收不到聊天回覆。
    """
    url = os.environ.get("UPSTASH_REDIS_REST_URL")
    token = os.environ.get("UPSTASH_REDIS_REST_TOKEN")
    if not url or not token:
        return

    record = {
        "prompt": prompt,
        "provider": provider,
        "time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + " UTC",
    }

    try:
        req = urllib.request.Request(
            f"{url.rstrip('/')}/rpush/{_REDIS_KEY}",
            # 注意:body 要直接是這筆記錄的 JSON 字串本身,不能再包一層陣列
            # (例如 json.dumps([...])) ——Upstash 的這個 REST 端點會把整個
            # request body 原封不動當成要 push 的值存起來,包了陣列就會把
            # 「陣列的文字表示」存成值,讀出來要多解一層 JSON 才能還原,
            # 這裡是實際打 API 測過兩種寫法比對後才確認的行為。
            data=json.dumps(record, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS)
    except (urllib.error.URLError, OSError, ValueError):
        pass
