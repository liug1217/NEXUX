"""
weather_lookup.py
------------------
使用者問「現在天氣/氣溫/濕度/有沒有下雨」這類問題時,直接呼叫中央氣象署
開放資料平台的 API,拿真實觀測資料回答,而不是讓模型自己生成(目前模型
規模太小,天氣這種需要精確數字的問題,生成出來的內容不可信)。

注意這裡用的是「局屬地面氣象測站每小時觀測資料」(C-B0024-001),
拿到的是「最近一次的實際觀測結果」(氣溫、濕度、風速、最近的降雨量等),
不是「未來預報」——如果使用者問的是「明天會不會下雨」這種預報性質的
問題,這個資料集回答不了,這裡刻意不硬答,直接回傳 None 讓後面正常的
smalltalk/qa_lookup/模型生成流程接手處理。

設定方式:
    到中央氣象署開放資料平台(https://opendata.cwa.gov.tw/)申請授權碼,
    設定環境變數 CWA_API_KEY(本機用 .env,Vercel 上要在專案的
    Environment Variables 另外加)。沒有設定這個環境變數的話,這個模組
    會直接讓所有天氣問題都回傳 None,不影響其他功能正常運作。
"""

import json
import os
import re
import time
import urllib.error
import urllib.request

_API_URL = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/C-B0024-001"
_TIMEOUT_SECONDS = 5

# 資料集裡實際的測站名稱,只挑「跟一般人講的城市名稱直接對得上」的測站
# 收錄進來,不牽強地拿其他行政區的測站冒充成某個縣市(例如新竹縣市、
# 苗栗、彰化、南投、雲林、屏東、桃園在這份資料集裡沒有剛好同名的測站,
# 寧可讓這些問題正常交給後面的流程處理,也不要給使用者不準確的地名)。
# 鍵是使用者可能打的講法,值是資料裡實際的 StationName。
_STATION_ALIASES = {
    "台北": "臺北", "臺北": "臺北", "台北市": "臺北", "臺北市": "臺北",
    "新北": "新北", "新北市": "新北",
    "基隆": "基隆", "基隆市": "基隆",
    "台中": "臺中", "臺中": "臺中", "台中市": "臺中", "臺中市": "臺中",
    "台南": "臺南", "臺南": "臺南", "台南市": "臺南", "臺南市": "臺南",
    "高雄": "高雄", "高雄市": "高雄",
    "嘉義": "嘉義", "嘉義市": "嘉義",
    "新竹": "新竹", "新竹市": "新竹",
    "宜蘭": "宜蘭", "宜蘭縣": "宜蘭",
    "花蓮": "花蓮", "花蓮縣": "花蓮",
    "台東": "臺東", "臺東": "臺東", "台東縣": "臺東", "臺東縣": "臺東",
    "澎湖": "澎湖", "澎湖縣": "澎湖",
    "金門": "金門", "金門縣": "金門",
    "馬祖": "馬祖", "連江": "馬祖", "連江縣": "馬祖",
    "阿里山": "阿里山",
    "日月潭": "日月潭",
    "墾丁": "恆春", "恆春": "恆春",
}

_WEATHER_KEYWORDS = ["天氣", "氣溫", "溫度", "濕度", "會不會下雨", "有沒有下雨", "下雨", "氣壓", "風速", "多熱", "多冷"]

# 「未來預報」性質的字眼,這份資料集回答不了,出現這些字眼時直接放棄、
# 交給後面的流程處理,避免拿「現在的觀測值」冒充成「對未來的預測」。
_FORECAST_KEYWORDS = ["明天", "明日", "後天", "這週", "這禮拜", "下週", "未來", "會下雨嗎", "會不會下雨"]

_cache = {"data": None, "fetched_at": 0.0}
_CACHE_TTL_SECONDS = 600  # 資料本身是每小時更新一次,10分鐘內重複問同個城市不用重打API


def _fetch_station_data():
    """
    抓取全部測站的最新觀測資料,加上簡單的記憶體快取(10分鐘內重複呼叫
    直接吃快取),避免同一段時間內每個問天氣的請求都重新打一次 API。
    任何失敗情況(沒設定金鑰、逾時、API回錯誤)都回傳 None,不拋出例外。
    """
    now = time.time()
    if _cache["data"] is not None and (now - _cache["fetched_at"]) < _CACHE_TTL_SECONDS:
        return _cache["data"]

    api_key = os.environ.get("CWA_API_KEY")
    if not api_key:
        return None

    try:
        url = f"{_API_URL}?Authorization={api_key}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return None

    if data.get("success") != "true":
        return None

    _cache["data"] = data
    _cache["fetched_at"] = now
    return data


def _find_station_name(prompt: str) -> str | None:
    # 用比較長的別名優先比對(例如「台北市」要比「台北」先比對到),
    # 避免「台北市」被短別名「台北」提早比對到、結果多餘的「市」字沒被
    # 正確處理(雖然這裡兩者對應到同一個測站,不影響結果,但養成習慣
    # 對之後別名對應到不同測站的情況比較保險)。
    for alias in sorted(_STATION_ALIASES, key=len, reverse=True):
        if alias in prompt:
            return _STATION_ALIASES[alias]
    return None


def match_weather(prompt: str) -> str | None:
    """
    判斷這句話是不是在問「現在」的天氣狀況,而且有提到收錄在
    _STATION_ALIASES 裡的城市名稱,是的話直接呼叫中央氣象署 API 拿真實
    觀測資料組成回覆;不符合就回傳 None,讓呼叫端(server.py /
    api/generate.py)繼續往下走原本的 smalltalk/qa_lookup/模型生成流程。
    """
    if not any(keyword in prompt for keyword in _WEATHER_KEYWORDS):
        return None

    if any(keyword in prompt for keyword in _FORECAST_KEYWORDS):
        return None  # 問的是未來預報,這份資料集回答不了,不要硬答

    station_name = _find_station_name(prompt)
    if station_name is None:
        return None

    data = _fetch_station_data()
    if data is None:
        return None

    for location in data.get("records", {}).get("location", []):
        station = location.get("station", {})
        if station.get("StationName") != station_name:
            continue

        obs_times = location.get("stationObsTimes", {}).get("stationObsTime", [])
        if not obs_times:
            return None

        latest = obs_times[-1]
        elements = latest.get("weatherElements", {})
        observed_at = latest.get("DateTime", "")

        temp = elements.get("AirTemperature")
        humidity = elements.get("RelativeHumidity")
        precipitation = elements.get("Precipitation")

        if temp is None:
            return None

        parts = [f"根據中央氣象署最新觀測資料"]
        if observed_at:
            # DateTime 格式例如 2026-08-03T01:00:00+08:00,只取到分鐘,
            # 不需要秒數和時區這種對使用者不重要的細節。
            time_part = observed_at.replace("T", " ")[:16]
            parts.append(f"({time_part})")
        parts.append(f",{station_name}目前氣溫 {temp}°C")
        if humidity is not None:
            parts.append(f"、相對濕度 {humidity}%")

        # 使用者如果特別問「有沒有下雨」,即使目前沒有降雨,也要明確回答
        # 「沒有」,而不是整段回覆完全不提降雨這件事,讓人搞不清楚是
        # 「沒有下雨」還是「這個資料沒有提供降雨資訊」。
        asked_about_rain = any(k in prompt for k in ("下雨", "會不會下雨", "有沒有下雨"))
        try:
            has_rain = precipitation is not None and float(precipitation) > 0
        except ValueError:
            has_rain = None  # "T" 這種微量降雨、量測不到明確數字的情況

        if has_rain:
            parts.append(f"、最近一小時降雨量 {precipitation}mm")
        elif asked_about_rain and has_rain is False:
            parts.append(",目前沒有降雨")
        parts.append("。")

        return "".join(parts)

    return None
