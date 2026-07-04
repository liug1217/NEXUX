"""
api/generate.py
----------------
這個檔案的存在,純粹是為了配合 Vercel 的規則:
Vercel 只會把 api/ 資料夾底下的 Python 檔案當作 Serverless Function。

實際的 Flask 應用程式邏輯,都寫在專案根目錄的 server.py 裡
(這樣本機開發時,還是可以直接執行「python server.py」)。
這個檔案只是把根目錄的 app 變數重新匯出一次,讓 Vercel 找得到。
"""

import os
import sys

# 把專案根目錄加進 Python 的搜尋路徑,這樣才能 import 到根目錄的 server.py、
# config.py、model.py 等檔案(因為這個檔案本身位於 api/ 子資料夾裡)。
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server import app  # noqa: E402  # Vercel 會尋找這個名為 app 的 Flask 實例
