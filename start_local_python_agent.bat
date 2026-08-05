@echo off
cd /d "%~dp0"
echo 正在啟動 NEXUX 本機 Python 執行器...
echo (這個視窗請留著開,關掉就代表停用這個功能)
echo.
python local_python_agent.py
if errorlevel 1 (
    echo.
    echo 啟動失敗,請確認這台電腦已安裝 Python(https://python.org),
    echo 而且 local_python_agent.py 這個檔案跟這個 .bat 放在同一個資料夾。
    pause
)
