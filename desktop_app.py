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

用法:
    python desktop_app.py                          # 開啟正式站
    python desktop_app.py --url http://localhost:5000  # 本機開發測試用

打包成 exe(本機另外 `pip install pywebview pyinstaller`,這兩個套件
只是打包用,不需要加進 requirements.txt——那份清單是給 Vercel
serverless function 用的,這個桌面版完全不會部署到 Vercel):
    pyinstaller --onefile --windowed --name NEXUX_desktop desktop_app.py
"""

import argparse
import threading

import webview

import local_python_agent

DEFAULT_URL = "https://ai.nexuxai.net"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()

    threading.Thread(target=local_python_agent.run_server, daemon=True).start()

    webview.create_window("NEXUX", args.url, width=900, height=750, resizable=True)
    webview.start()


if __name__ == "__main__":
    main()
