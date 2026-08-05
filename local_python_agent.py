"""
local_python_agent.py
----------------------
給 NEXUX(ai.nexuxai.net)程式碼編輯器面板用的「本機執行代理程式」。

背景:
網頁裡的程式碼編輯器,Python「▶ 執行」預設用 Pyodide(瀏覽器裡的
WebAssembly Python)執行,好處是不用裝任何東西就能用,壞處是純 CPU、
單執行緒,連 numpy/pandas 都沒開放安裝,更不用說 torch/GPU。如果你自己
電腦本來就有裝「真正的」Python(甚至裝好了完整的資料科學/深度學習
環境),執行這個檔案就能讓網頁改用你電腦上真正的 Python 執行,不再
被 Pyodide 的限制卡住。

用法:
    最簡單的方式(Windows,不用裝 Python、不用打開終端機):下載
    local_python_agent.exe(這個檔案用 PyInstaller 打包而成),雙擊
    就會啟動。執行使用者送來的程式碼時,還是會去找電腦上真正安裝的
    Python 執行(見下面 _resolve_python_executable()),所以電腦本來
    要有裝 Python 這件事沒有改變,改變的只是「啟動代理程式本身」不用
    先裝 Python 才能雙擊執行。

    想先看過原始碼再執行(比較放心)的話,可以改用這個 .py 檔案本身:
    跟 start_local_python_agent.bat 下載到同一個資料夾,雙擊
    start_local_python_agent.bat 就會啟動(雙擊會自動開一個顯示執行
    狀態的視窗,不是要你自己打開終端機輸入指令)。

    也可以自己在終端機執行:
        python local_python_agent.py             # 監聽 127.0.0.1:8799
        python local_python_agent.py --port 9000  # 自訂 port

執行後回到 ai.nexuxai.net(或本機開發中的 NEXUX 網站),按程式碼編輯器
面板的「▶ 執行」,網頁會自動偵測到這個程式在跑,之後每次執行都會改用
它,不用再手動做任何設定。想停用就直接把這個視窗關掉(或按 Ctrl+C)。

**安全性,請務必先看完再執行**:
這個程式會把 ai.nexuxai.net 網頁送過來的程式碼,原封不動地用你電腦上
的 Python 執行——換句話說,只要你開著這個程式,任何你在該網站上按下
「執行」的程式碼,都會用你自己的作業系統帳號權限跑起來。只有你自己
主動信任、要使用這個網站的功能時才執行它。

已經做的防護:
1. 只監聽 127.0.0.1(不對外網開放),區域網路上其他裝置連不到,只有
   這台電腦自己能連。
2. 每個請求都會檢查瀏覽器自動帶上的 Origin header,只有
   https://ai.nexuxai.net、http://localhost:5000、http://127.0.0.1:5000
   (後兩個是本機開發 NEXUX 網站本身時用的)這幾個來源才會被接受,其他
   一律回 403、不回傳 CORS header——這能擋住「你瀏覽器另一個分頁開著
   的惡意網站,偷偷發請求打這個本機服務」這種攻擊(Origin header 是
   瀏覽器自己加的,網頁的 JavaScript 改不了)。

**這個防護做不到的事**:如果你的電腦本身已經中毒、已經有惡意程式在
跑,這個程式擋不住它——但那種情況下,攻擊者早就有能力直接在你的
電腦上執行任意程式碼了,原本就不需要繞這個程式這條路。這裡的 Origin
檢查,防的是「透過瀏覽器發出的跨站請求」,不是全方位的資安保證,
請自行評估風險後再決定要不要執行。
"""

import json
import shutil
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

DEFAULT_PORT = 8799
RUN_TIMEOUT_SECONDS = 10

_ALLOWED_ORIGINS = {
    "https://ai.nexuxai.net",
    "http://localhost:5000",
    "http://127.0.0.1:5000",
}


def _resolve_python_executable():
    """
    平常(直接用 `python local_python_agent.py` 執行)`sys.executable`
    就是真正的 python.exe,直接用就好。

    但如果這個檔案被 PyInstaller 打包成 .exe 執行(`sys.frozen` 為
    True),`sys.executable` 會變成指向這個 exe 自己,不是真正的
    Python 直譯器——直接拿去執行 `-c 程式碼` 會失敗,因為這個 exe
    不認得這個參數。這時候改成去 PATH 裡找使用者電腦上真正安裝的
    Python,才能繼續用到使用者自己裝的套件(numpy/torch等)。
    找不到就回傳 None,呼叫端要處理「沒有本機 Python 可以用」這個情況。
    """
    if not getattr(sys, "frozen", False):
        return sys.executable
    for candidate in ("python", "python3", "py"):
        path = shutil.which(candidate)
        if path:
            return path
    return None


class AgentRequestHandler(BaseHTTPRequestHandler):
    def _origin_allowed(self):
        origin = self.headers.get("Origin", "")
        return origin if origin in _ALLOWED_ORIGINS else None

    def _send_cors_headers(self, origin):
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _reject(self):
        self.send_response(403)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps({"error": "不允許的來源"}).encode("utf-8"))

    def do_OPTIONS(self):
        origin = self._origin_allowed()
        if not origin:
            self._reject()
            return
        self.send_response(204)
        self._send_cors_headers(origin)
        self.end_headers()

    def do_GET(self):
        origin = self._origin_allowed()
        if not origin:
            self._reject()
            return
        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps({"status": "ok", "agent": "nexux-local-python-agent"}).encode("utf-8")
        self.send_response(200)
        self._send_cors_headers(origin)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        origin = self._origin_allowed()
        if not origin:
            self._reject()
            return
        if self.path != "/run":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            payload = {}
        code = (payload.get("code") or "").strip()

        python_executable = _resolve_python_executable()

        if not code:
            result = {"error": "沒有程式碼可以執行"}
        elif not python_executable:
            result = {"error": "找不到本機安裝的 Python,請先安裝 Python(https://python.org)後再使用這個功能。"}
        else:
            try:
                proc = subprocess.run(
                    [python_executable, "-c", code],
                    capture_output=True,
                    text=True,
                    timeout=RUN_TIMEOUT_SECONDS,
                )
                result = {
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "returncode": proc.returncode,
                }
            except subprocess.TimeoutExpired:
                result = {"error": f"執行超過{RUN_TIMEOUT_SECONDS}秒逾時,可能是無窮迴圈,已中止。"}

        body = json.dumps(result, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self._send_cors_headers(origin)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # 預設會把每個請求都印出來,雜訊太多,改成安靜模式。
        pass


def main():
    port = DEFAULT_PORT
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        port = int(sys.argv[idx + 1])

    server = HTTPServer(("127.0.0.1", port), AgentRequestHandler)
    print(f"[local_python_agent] 監聽中: http://127.0.0.1:{port}")
    print("[local_python_agent] 只接受來自 ai.nexuxai.net / 本機開發站的請求(已驗證 Origin)。")
    print("[local_python_agent] 這個程式會執行網頁送過來的任意程式碼,只在信任該網站時保持開啟。")
    print("[local_python_agent] 按 Ctrl+C 隨時可以關閉。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[local_python_agent] 已關閉。")


if __name__ == "__main__":
    main()
