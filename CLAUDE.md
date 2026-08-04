# CHATBOT 專案筆記

## 目前版本:NEXUX v1.0

正式站(ai.nexuxai.net)`provider == "own"` 現在服務的是 **NEXUX Model
v1.0**——微調 ckiplab/gpt2-base-chinese(中研院 CKIP Lab 預訓練繁體中文
GPT2)得到的版本,已通過測試、確認比舊版(從零訓練的 char-level 模型)
穩定,正式扶正為唯一的正式版本(不再有 `own`/`own_beta` 兩套並存)。
完整背景、開發原則見 **`docs/MODEL_MIGRATION.md`**。

**下面「模型再訓練與更新流程」這一節講的是舊版 char-level 模型的做法,
已經不是正式站在用的流程,只保留給需要回頭比較兩種做法時參考。**
NEXUX v1.0 實際的重新訓練流程是:

1. `python convert_pretrained.py`
   從 HuggingFace 下載 ckiplab/gpt2-base-chinese,轉成專案的 checkpoint
   格式(輸出 `checkpoint_pretrained.pt` + `vocab_pretrained.txt`,本機
   需要另外 `pip install transformers`,不在 requirements.txt 內)。
2. `python run_pretrained_sft.py`(可加 `--steps N` 自訂步數,預設讀
   `config.py` 的 `sft_max_iters`)
   接上 `data/*.jsonl` 語料做問答微調,更新 `checkpoint_pretrained.pt`。
3. `python export_pretrained.py`
   int8 量化匯出成 `weights_meta_pretrained.json` + `weights_pretrained.npz`
   (Vercel 上的 numpy 推理引擎讀這兩個檔案)。
4. 用 `python eval_open_ended.py --pretrained` 固定種子(seed=1337)跑一次
   固定 benchmark,確認這次改動是真的進步,不能只憑感覺判斷。
5. `git add`(只加 `data/*.jsonl` 變更、`weights_meta_pretrained.json`、
   `weights_pretrained.npz`,**不要**加 `checkpoint_pretrained.pt`,那個
   檔案很大且已經在 `.gitignore` 排除)、`git commit`、`git push`。

舊版模型(char-level 從零訓練)已經正式下架、不再部署,對應的
`weights_meta.json`/`weights.npz`/`tokenizer.json` 已經從 git 追蹤移除
(檔案仍在本機、git 歷史紀錄完整保留),不需要再上傳新版本。

## 語料格式:標準 messages JSONL

`data/` 底下的語料統一存成標準的 `messages` 格式,用 `.jsonl`(一行一筆 JSON,
不是單一大陣列),方便新增語料時直接在檔案後面加一行,不用整份重新解析寫入:

```jsonl
{"messages": [{"role": "user", "content": "嗨"}, {"role": "assistant", "content": "你好!今天過得怎麼樣?"}]}
{"messages": [{"role": "user", "content": "你叫什麼?"}, {"role": "assistant", "content": "我是AI助手。"}, {"role": "user", "content": "你可以做什麼?"}, {"role": "assistant", "content": "我可以回答問題。"}]}
```

一段對話可以只有一問一答,也可以是多輪(訊息陣列交替 user/assistant)。
讀取、驗證邏輯集中在 `messages_format.py`(`load_conversations` / `render_messages`)。
`train.py`(純接龍預訓練)跟 `prepare_sft_data.py`(展開成 SFT 訓練用的 jsonl)都是從這裡讀取,
內部會轉成「問:.../答:...」文字,跟 `inference.py` / `conversation.py` / `server.py`
推論時使用、`text_cleanup.py` 判斷生成該不該停止的標記保持一致,所以新增語料時
只要維持 `messages` 格式,不需要自己組「問:/答:」字串。

## 〔封存〕舊版 char-level 模型再訓練與更新流程

**這一節是 NEXUX v1.0 之前的做法,正式站已經不再使用,純粹保留給需要
回頭比較「從零訓練」跟「微調預訓練模型」兩種做法時參考,不要拿來更新
正式站。** 正式站現在的重新訓練流程見上面「目前版本:NEXUX v1.0」。

每當 `data/` 資料夾底下任何 `.jsonl` 語料檔案有新增或修改,或是進行了重新訓練,
都必須執行完整的「訓練 → 匯出 → 上傳」流程,讓部署到 Vercel 上的模型與最新語料保持一致:

1. `python train.py`
   讀取 `data/` 下所有 `.jsonl`,重新訓練 checkpoint(輸出 `checkpoint.pt`)。
   **注意**:如果語料改變導致詞表(vocab)變動,`train.py` 預設會沿用既有的
   `tokenizer.json`(靜默忽略新字元!),新增語料後如果不確定詞表有沒有變,
   訓練前先刪除 `tokenizer.json` 和 `checkpoint.pt`,強制重新建立詞表比較保險。
2. `python prepare_sft_data.py` 接著 `python train_sft.py`
   把 `data/` 下所有 `.jsonl` 對話展開成 SFT 訓練用的 `sft_data.jsonl`,在預訓練成果上做問答微調。
3. `python export_weights.py`
   把 `checkpoint.pt` 轉成 `weights.json`(Vercel 上的 numpy 推理引擎讀這個檔案)。
4. `git add -A`、`git commit`(說明是資料更新/重新訓練)、`git push`
   commit 訊息需簡述新增了哪些語料或訓練變更;push 到 GitHub 後 Vercel 會自動重新部署。

如果這台機器有 NVIDIA GPU(`nvidia-smi` 能看到裝置),訓練時優先用 CUDA 版 torch
(`pip install torch --index-url https://download.pytorch.org/whl/cu126`,依驅動版本調整
cu126 這個標籤),`config.py` 的 `device` 會自動偵測並優先用 GPU,速度比單執行緒 CPU
快非常多(實測同樣步數快了 30 倍以上),值得優先確認。

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
