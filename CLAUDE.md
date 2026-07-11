# CHATBOT 專案筆記

## 模型再訓練與更新流程(重要,務必遵守)

每當 `data/` 資料夾底下任何 `.txt` 語料檔案有新增或修改,或是進行了重新訓練,
都必須執行完整的「訓練 → 匯出 → 上傳」流程,讓部署到 Vercel 上的模型與最新語料保持一致:

1. `python train.py`
   讀取 `data/` 下所有 `.txt`,重新訓練 checkpoint(輸出 `checkpoint.pt`)。
2. `python export_weights.py`
   把 `checkpoint.pt` 轉成 `weights.json`(Vercel 上的 numpy 推理引擎讀這個檔案)。
3. `git add -A`、`git commit`(說明是資料更新/重新訓練)、`git push`
   commit 訊息需簡述新增了哪些語料或訓練變更;push 到 GitHub 後 Vercel 會自動重新部署。

執行 `python`、`pip` 前記得把新安裝的 Python 加進當次 shell 的 PATH(此機器的 python.exe 位於
`C:\Users\Administrator\AppData\Local\Programs\Python\Python312`,winget 安裝後新開的
Bash session 預設抓不到,需要 `export PATH=".../Python312:.../Python312/Scripts:$PATH"`)。

torch 只用於本機訓練(不在 `requirements.txt` 內,因為 Vercel 的 serverless function
大小限制塞不下 torch),需要另外 `pip install torch`。若載入 torch 出現
`WinError 1114`(DLL 初始化失敗),代表機器上的 Microsoft Visual C++ Redistributable
過舊,需要 `winget install --id Microsoft.VCRedist.2015+.x64 -e` 更新。

`config.py` 的 `resume` 預設是 `False`,所以每次 `python train.py` 都是從頭重新訓練,
不是接續訓練;這是目前專案的預期行為(語料量還小,從頭訓練成本低)。

使用者已明確授權:資料異動 → 重新訓練 → 匯出 → commit → push 到 GitHub(觸發 Vercel
自動部署)這一整套流程,不需要每次都再另外確認,完成後跟使用者回報結果即可。
