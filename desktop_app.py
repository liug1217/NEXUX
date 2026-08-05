"""
desktop_app.py
---------------
NEXUX 聊天網站的桌面版捷徑,單純省去「自己開瀏覽器、自己打網址」這兩步。

這不是把 AI 模型包進 exe 離線執行——AI 模型還是跑在正式站伺服器上,
這個程式只是用系統原生的 WebView 元件(Windows 上是 WebView2,Mac 上
是 WebKit,不像 Electron 需要額外打包一整份 Chromium)開一個視窗,
載入 https://ai.nexuxai.net,聊天/程式碼編輯器/Pyodide 等等全部都是
網站本身既有的功能,這裡完全不重新實作任何邏輯。

**這個 exe 需要網路連線才能用**,沒有網路的話會顯示無法連線的錯誤畫面,
跟開瀏覽器連不上網站是一樣的情況,不是程式壞掉。

**額外附帶本機 Python 執行能力**:開啟視窗的同時,會在背景執行緒一併
啟動 local_python_agent.py 的服務(見該檔案說明),讓桌面版視窗裡的
程式碼編輯器能直接用真正的本機 Python(有完整套件/GPU),不用使用者
另外再下載、雙擊一次 local_python_agent.exe。這個背景服務只在桌面版
這個視窗開著的時候存在,關掉視窗它也會跟著結束(daemon thread)。

**F11 切換全螢幕**,跟一般瀏覽器習慣一致。

**第一次執行會自動在桌面建捷徑**:只有打包成 exe 執行時才會做這件事
(一般用 `python desktop_app.py` 開發測試不會),讓使用者下載後第一次
雙擊(通常在下載資料夾裡)之後,不用自己手動搬到桌面,之後可以直接
從桌面圖示開啟。捷徑已經存在就不會重複建立。

用法:
    python desktop_app.py                          # 開啟正式站
    python desktop_app.py --url http://localhost:5000  # 本機開發測試用

打包成 exe(本機另外 `pip install pywebview pyinstaller`,這兩個套件
只是打包用,不需要加進 requirements.txt——那份清單是給 Vercel
serverless function 用的,這個桌面版完全不會部署到 Vercel):
    pyinstaller --onefile --windowed --icon=NEXUX.ico --name NEXUX_desktop desktop_app.py
"""

import argparse
import os
import subprocess
import sys
import threading

import webview

import local_python_agent

DEFAULT_URL = "https://ai.nexuxai.net"


def _get_desktop_path():
    """
    讀 Windows 登錄檔取得實際的桌面資料夾路徑,能正確處理 OneDrive
    之類把桌面資料夾重新導向到別處的情況。讀不到就退回猜測預設路徑
    (沒有重新導向的一般情況下,這個猜測本來就是對的)。
    """
    try:
        import winreg
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders",
        ) as key:
            path, _ = winreg.QueryValueEx(key, "Desktop")
            return os.path.expandvars(path)
    except Exception:
        return os.path.join(os.path.expanduser("~"), "Desktop")


def _ensure_desktop_shortcut():
    """
    在使用者的 Windows 桌面建一個指向這個 exe 自己的捷徑(.lnk),
    讓下載後第一次執行(通常在下載資料夾裡雙擊)之後,不用自己手動
    搬到桌面,可以直接從桌面圖示開啟。

    只在真正打包成 exe(sys.frozen)時做這件事——一般開發測試用
    `python desktop_app.py` 執行,sys.executable 是 python.exe 本身,
    建捷徑沒有意義。捷徑已經存在就跳過,不會每次開都重建一次。

    用 PowerShell 呼叫 WScript.Shell 這個 Windows 內建 COM 物件建立
    捷徑,不需要額外安裝 pywin32/winshell 之類的套件;失敗也不影響
    主程式(只是沒有捷徑而已,不是關鍵功能,所以整段包在 try/except
    裡靜默失敗)。

    桌面路徑刻意不用 `os.path.expanduser("~") + "Desktop"` 猜——這台
    機器本身就是「桌面資料夾被 OneDrive 重新導向」的實際案例(真正路徑
    是 OneDrive 底下的 Desktop,不是使用者資料夾直接底下的
    Desktop),猜錯的話捷徑會建到一個不存在的路徑,靜默失敗、使用者
    永遠看不到捷徑。改成讀登錄檔
    (`User Shell Folders`)取得 Windows 實際在用的桌面路徑,這是微軟
    官方推薦、能正確處理資料夾重新導向的方式。
    """
    if not getattr(sys, "frozen", False):
        return

    desktop = _get_desktop_path()
    shortcut_path = os.path.join(desktop, "NEXUX.lnk")
    if os.path.exists(shortcut_path):
        return

    exe_path = sys.executable
    ps_command = (
        f'$s = (New-Object -ComObject WScript.Shell).CreateShortcut("{shortcut_path}"); '
        f'$s.TargetPath = "{exe_path}"; '
        f'$s.IconLocation = "{exe_path}"; '
        f'$s.Save()'
    )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_command],
            capture_output=True,
            timeout=10,
        )
    except Exception:
        pass


class _Api:
    """
    暴露給網頁 JS 呼叫的橋接物件(pywebview 的 js_api 機制),目前只有
    切換全螢幕這一個用途。`window` 要等 create_window() 執行完才拿得到,
    所以先建立空殼,建好視窗後再補上參照。
    """

    def __init__(self):
        self.window = None

    def toggle_fullscreen(self):
        if self.window:
            self.window.toggle_fullscreen()


def _bind_fullscreen_shortcut(window):
    """
    F11 切換全螢幕,跟一般瀏覽器的習慣一致。pywebview 本身沒有內建的
    跨平台快捷鍵綁定,用 evaluate_js 注入一段監聽 keydown 的 JS,
    按下 F11 時呼叫上面 _Api.toggle_fullscreen()。
    """
    window.evaluate_js(
        "document.addEventListener('keydown', function(e) {"
        "  if (e.key === 'F11') {"
        "    e.preventDefault();"
        "    window.pywebview.api.toggle_fullscreen();"
        "  }"
        "});"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()

    threading.Thread(target=_ensure_desktop_shortcut, daemon=True).start()
    threading.Thread(target=local_python_agent.run_server, daemon=True).start()

    api = _Api()
    window = webview.create_window(
        "NEXUX", args.url, width=900, height=750, resizable=True, js_api=api
    )
    api.window = window
    window.events.loaded += lambda *args: _bind_fullscreen_shortcut(window)

    webview.start()


if __name__ == "__main__":
    main()
