"""
quota_manager.py
-----------------
限制「own」模型(跑在自己伺服器上,沒有外部 API 費用,但會佔用運算
資源)的使用量,防止單一使用者無限制濫用拖垮伺服器。第三方模型
(GPT/Claude/Gemini/Groq)不受這裡限制。smalltalk/qa_lookup/weather_lookup
這些不經過模型生成的規則式短路回覆,也不計入額度(它們本來就不佔用
運算資源)。

規則(見 DEFAULT_QUOTA_CONFIG):每個使用者在一個時間窗口內最多能用
多少 token(輸入+輸出token數加總)。窗口從這個使用者「第一次用掉
token」開始算,不是網站啟動時或整點,每個使用者的窗口各自獨立。窗口
結束後自動重置,重新開始下一個額度。

限制數值刻意不寫死在函式邏輯裡,而是集中放在 QuotaConfig 這個設定
物件——之後如果要做不同會員方案(例如付費使用者額度更高),只要
建立不同的 QuotaConfig 執行個體傳進來,不用改 get_status/consume/
is_over_limit 這些核心邏輯。

技術實作:
`api/generate.py` 部署在 Vercel Serverless Function 上,每次呼叫都
可能是全新的執行環境,不能用 Python 全域變數存使用量。這裡沿用專案
已經在用的 Upstash Redis REST API(見 question_log.py,同一組環境變數
UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN,不用另外申請,額度
資料用不同的 key 前綴 "quota:" 區隔,不會跟 collected_questions 那個
key 衝突)。

用 Redis 的 INCRBY(累加用量)+ EXPIRE(設定窗口秒數後自動過期,到期後
key 自動消失,等於自動重置)這兩個指令完整實作,不用自己額外維護
「起算時間」欄位——Redis 的 TTL 本身就是「還剩多久自動消失」,拿來
查詢「還剩多久重置」剛好對應。

沒設定 Upstash 環境變數時(例如本機開發預設狀態):比照 question_log.py
的作法,靜默視為「沒有限制」,不會因為缺設定就讓聊天功能壞掉或卡住。

使用者識別:
優先用前端(NEXUX.html)自己產生、存在 localStorage 的 UUID(見
get_client_identifier),而不是單純用來源 IP——同一個家庭/學校/公司
可能共用一個 IP,行動網路的 IP 也會變動,單用 IP 容易誤傷正常使用者。
IP 只在使用者完全沒帶 UUID 過來時當備援。

**誠實的侷限**:瀏覽器自己回報的 UUID 是可以被使用者自己清掉
localStorage、或改傳別的值來繞過的,這裡沒有做到「防止刻意繞過」等級
的防護,只是讓「一般正常使用、剛好跟別人共用IP」的情況不會被誤傷。
真的要防刻意濫用,需要搭配登入系統或更嚴格的裝置指紋,不在這次範圍。
"""

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

_KEY_PREFIX = "quota:"
_TIMEOUT_SECONDS = 3
_UUID_RE = re.compile(r"^[0-9a-fA-F-]{8,64}$")


@dataclass(frozen=True)
class QuotaConfig:
    limit_tokens: int = 100_000
    window_seconds: int = 5 * 60 * 60  # 18000,5小時


DEFAULT_QUOTA_CONFIG = QuotaConfig()


def get_client_identifier(request, payload: dict | None = None) -> str:
    """
    優先讀前端帶來的 clientId(POST 請求從已解析的 JSON payload 讀,
    GET 請求從 query string 讀),格式粗略檢查一下(只允許
    UUID 常見的英數字+橫線,避免拿使用者可控字串直接當 Redis key
    塞進奇怪的內容)。沒有帶、或格式不像 UUID,就退回用來源 IP。
    """
    client_id = None
    if payload:
        client_id = payload.get("clientId")
    if not client_id:
        client_id = request.args.get("clientId")

    if client_id and _UUID_RE.match(client_id):
        return client_id
    return get_client_ip(request)


def get_client_ip(request) -> str:
    """
    Vercel 會把真實來源 IP 放在 X-Forwarded-For header(可能有多層代理,
    取第一個)。本機開發用 Flask 內建的 remote_addr 當備援。
    """
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _redis_request(path: str):
    """
    對 Upstash Redis REST API 發一個請求,回傳解析後的 JSON 的 "result"
    欄位。任何失敗(沒設定環境變數、網路逾時、Upstash 回錯誤)都回傳
    None,呼叫端要把 None 當成「當作沒有限制」處理,不能讓這裡的失敗
    影響使用者收到聊天回覆。
    """
    url = os.environ.get("UPSTASH_REDIS_REST_URL")
    token = os.environ.get("UPSTASH_REDIS_REST_TOKEN")
    if not url or not token:
        return None

    try:
        req = urllib.request.Request(
            f"{url.rstrip('/')}/{path}",
            headers={"Authorization": f"Bearer {token}"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body.get("result")
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        return None


def get_status(identifier: str, config: QuotaConfig = DEFAULT_QUOTA_CONFIG) -> dict:
    """
    回傳 {"used": int, "limit": int, "remaining": int,
    "reset_in_seconds": int|None, "enabled": bool}。

    enabled=False 代表沒設定 Upstash(本機開發預設狀態),前端應該
    隱藏額度顯示,不能當作 used=0/無限額度來顯示假數字。
    """
    if os.environ.get("UPSTASH_REDIS_REST_URL") is None:
        return {
            "enabled": False,
            "used": 0,
            "limit": config.limit_tokens,
            "remaining": config.limit_tokens,
            "reset_in_seconds": None,
        }

    key = f"{_KEY_PREFIX}{identifier}"
    used_raw = _redis_request(f"get/{key}")
    used = int(used_raw) if used_raw is not None else 0

    ttl_raw = _redis_request(f"ttl/{key}")
    # TTL: -2 代表 key 不存在(這個週期還沒用過任何 token),-1 代表
    # key 存在但沒設定過期時間(理論上不該發生,consume() 一定會補上)。
    reset_in_seconds = ttl_raw if isinstance(ttl_raw, int) and ttl_raw >= 0 else None

    return {
        "enabled": True,
        "used": used,
        "limit": config.limit_tokens,
        "remaining": max(0, config.limit_tokens - used),
        "reset_in_seconds": reset_in_seconds,
    }


def consume(identifier: str, tokens: int, config: QuotaConfig = DEFAULT_QUOTA_CONFIG) -> None:
    """
    累加這次生成用掉的 token 數。如果這個 key 目前沒有 TTL(代表是這個
    使用者這個週期第一次用),補上 EXPIRE。任何失敗都靜默放行,不影響
    使用者已經收到的聊天回覆。
    """
    if tokens <= 0:
        return

    key = f"{_KEY_PREFIX}{identifier}"
    _redis_request(f"incrby/{key}/{tokens}")

    ttl_raw = _redis_request(f"ttl/{key}")
    if ttl_raw == -1:
        _redis_request(f"expire/{key}/{config.window_seconds}")


def is_over_limit(identifier: str, config: QuotaConfig = DEFAULT_QUOTA_CONFIG) -> bool:
    """給 /api/generate 生成前檢查用。Upstash 沒設定時永遠回傳 False(不限制)。"""
    status = get_status(identifier, config)
    if not status["enabled"]:
        return False
    return status["used"] >= status["limit"]
