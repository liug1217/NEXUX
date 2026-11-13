# 在 Kaggle 上訓練 NEXUX(階段 1:自製訓練迴圈 + 微調 Qwen2.5)

這份說明帶你把 [`kaggle/train_nexux.py`](train_nexux.py) 搬到 Kaggle 的免費 GPU 上,
用你整理好的 `data/` 語料訓練出第一個「真的會聊天」的 NEXUX。

> **為什麼是這個做法?**
> 訓練「迴圈」是我們自己寫的(資料處理、loss 遮罩、優化器、學習率排程、
> 混合精度、存檔全在 `train_nexux.py` 裡),只跟 transformers 借 Qwen2.5 的
> 底模型。這是「慢慢減少依賴別人」路線的第 1 步:先做出能用的版本,
> 之後階段再把底模型也換成自己的。

---

## 一、開 Notebook 並設定 GPU

1. 登入 [kaggle.com](https://www.kaggle.com) →右上 **Create → New Notebook**
2. 右側面板 **Settings**:
   - **Accelerator** 選 **GPU T4 x2**(或 P100)
   - **Internet** 打開 **On**(要連網下載 Qwen 模型、clone GitHub)
3. (建議)Kaggle 帳號先完成手機驗證,GPU 和 Internet 才開得了。

---

## 二、貼上並執行(三個 cell)

**Cell 1 — 裝套件**
```python
!pip install -q -U "transformers>=4.44" "accelerate>=0.33" "peft>=0.11"
```

**Cell 2 — 拉專案(程式碼 + 語料)**
```python
!rm -rf NEXUX && git clone --depth 1 https://github.com/liug1217/NEXUX.git
```

**Cell 3 — 開始訓練**
```python
# 正式預設:Qwen2.5-1.5B-Instruct + LoRA,3 epoch,依業界 SFT 共識調參
!cd NEXUX && python kaggle/train_nexux.py --output-dir /kaggle/working/nexux_sft_out
```

訓練時終端機會印 loss、learning rate、進度;跑完會自動用幾句測試問題
(你叫什麼名字?/ 你是不是 GPT? …)試生成,讓你肉眼看成效。

---

## 三、想調參數 / 換模型

所有設定都在 `train_nexux.py` 最上面的 `CFG`(已依業界 SFT 共識設好:
LoRA lr 2e-4、rank 32 / alpha 64、全線性層、cosine + warmup 0.03、3 epoch、
有效 batch 16),也可以用指令參數覆蓋:

```python
# 想更省資源 → 改用 0.5B 全參數微調(不用 peft、速度快)
!cd NEXUX && python kaggle/train_nexux.py \
        --model Qwen/Qwen2.5-0.5B-Instruct --full-finetune --no-grad-ckpt \
        --output-dir /kaggle/working/nexux_sft_out
```

常用參數:`--model`、`--epochs`、`--lr`、`--batch-size`、`--grad-accum`、
`--no-grad-ckpt`(關 gradient checkpointing 加速)、`--full-finetune`。

---

## 四、把訓練好的模型帶回來

訓練完模型會在 `/kaggle/working/nexux_sft_out`。兩種帶回方式:

- **直接下載**:Notebook 右側 **Output** 分頁,把 `nexux_sft_out` 資料夾下載下來。
- **存成 Kaggle Dataset**(建議,方便下次接續):Notebook 右上 **Save Version**
  跑完後,輸出會保留;再從 output 建立 Dataset。

> ⚠️ **先別急著 commit 回 GitHub repo。**
> 這個 Qwen 微調模型(資料夾好幾百 MB ~ 1GB+)跟現在 Vercel 上那套
> numpy 推理引擎**格式不相容、也放不進 git**。怎麼上線是下一個獨立階段
> (GPU 主機 / 推理 API / 自架),我們訓練出滿意的版本後再來處理。

---

## 五、本機先驗證資料管線(選用,不用 GPU)

在自己電腦上(不下載模型、不碰 GPU)確認語料讀得進、loss 遮罩對齊:

```bash
python kaggle/train_nexux.py --dry-run
```

會印出可用樣本數、長度分布、平均每筆要學的 token 數。
