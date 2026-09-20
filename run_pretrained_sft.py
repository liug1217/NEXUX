"""
run_pretrained_sft.py  (Unsloth 版)
-------------------------------------
用 Unsloth + Qwen2.5-1.5B-Instruct + LoRA + SFTTrainer 對 data/*.jsonl 做微調,
訓練完自動量化匯出為 GGUF(q4_k_m)。

用法:
    python run_pretrained_sft.py               # 預設 100 步
    python run_pretrained_sft.py --steps 500   # 指定步數
    python run_pretrained_sft.py --smoke       # 20 步快速確認
"""

import glob
import json
import os
import sys

# ======================================================
# 1. 安裝核心套件(若未安裝)
# ======================================================
try:
    import unsloth  # noqa: F401
except ImportError:
    os.system(
        'pip install --no-deps "unsloth_zoo>=2025.2.5" unsloth bitsandbytes xformers trl peft tensorflow -q'
    )

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

# ======================================================
# 2. 自動掃描所有語料檔案
# ======================================================
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
all_files = glob.glob(os.path.join(data_dir, "*.jsonl"))
if not all_files:
    raise ValueError(f"❌ 錯誤：在 {data_dir} 找不到任何 .jsonl 語料，請確認已上傳語料檔案！")

raw_rows = []
for path in all_files:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    raw_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

print(f"📂 系統地毯式掃描成功！共強行抓取到 {len(raw_rows)} 個訓練檔案物件！")

dataset = Dataset.from_list(raw_rows)

# ======================================================
# 3. 設定大模型(Qwen2.5-1.5B-Instruct)
# ======================================================
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

# ======================================================
# 4. 讓模型適應雙端雙顯卡，開啟 LoRA 微調模式
# ======================================================
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# ======================================================
# 5. 格式化函數：萬能格式自動對接器
# ======================================================
def formatting_prompts_func(examples):
    texts = []
    messages_batch = examples.get("messages", [None] * len(next(iter(examples.values()))))
    for i, msgs in enumerate(messages_batch):
        # 情況 A：標準 chat/messages 格式
        if isinstance(msgs, list):
            conv = ""
            for msg in msgs:
                role = "使用者" if msg.get("role") == "user" else "助手"
                conv += f"### {role}:\n{msg.get('content', '')}\n\n"
            texts.append(conv)
        # 情況 B：instruction/output 格式
        elif "instruction" in {k: v[i] for k, v in examples.items() if hasattr(v, '__getitem__')}:
            row = {k: v[i] for k, v in examples.items()}
            inst = row.get("instruction", "")
            inp  = row.get("input", "")
            out  = row.get("output", "")
            texts.append(f"### 指令:\n{inst}\n\n### 輸入:\n{inp}\n\n### 回答:\n{out}")
        # 情況 C：兜底方案
        else:
            row = {k: v[i] for k, v in examples.items()}
            texts.append(str(row))
    return texts  # 直接回傳符合 Unsloth 標準的純陣列

# ======================================================
# 6. 步數設定
# ======================================================
smoke = "--smoke" in sys.argv
custom_steps = None
if "--steps" in sys.argv:
    idx = sys.argv.index("--steps")
    custom_steps = int(sys.argv[idx + 1])

if smoke:
    max_steps = 20
elif custom_steps:
    max_steps = custom_steps
else:
    max_steps = 100

# ======================================================
# 7. 執行雲端高速訓練！
# ======================================================
print(f"🚀 正在啟動 GPU 進行高速雲端後台微調...（{max_steps} 步）")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    max_seq_length=max_seq_length,
    dataset_num_proc=1,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=max_steps,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        output_dir="outputs",
    ),
)

trainer_stats = trainer.train()

# ======================================================
# 8. 訓練完成後，自動將模型壓縮（量化）成 GGUF 格式！
# ======================================================
if not smoke:
    print("📦 訓練完成！正在進行 4-bit 量化壓縮，打包為 GGUF 格式...")
    model.save_pretrained_gguf("nexux_qwen25_model", tokenizer, quantization_method="q4_k_m")
    print("🎉 恭喜！模型訓練並成功壓縮完畢！GGUF 檔案：nexux_qwen25_model-Q4_K_M.gguf")
else:
    print("✅ Smoke test 完成（未匯出 GGUF）")
