# -*- coding: utf-8 -*-
"""
train_nexux.py —— NEXUX 正式訓練腳本(階段 1)
================================================

這是「減少依賴別人」路線圖的第 1 階段(見 docs/CLEANUP_REPORT.md 之後的規劃):

  * 訓練「迴圈」是我們自己寫的 PyTorch loop —— 不是照搬 Hugging Face 的
    Trainer / trl.SFTTrainer。資料處理、loss 遮罩、優化器、學習率排程、
    梯度累積、混合精度、存檔,全部在這個檔案裡看得到、改得動。
  * 只跟 transformers 借「底模型的架構與權重」(Qwen2.5)。這是我們談好的:
    站在別人的肩膀上先做出「能對話」的版本,之後階段再逐步把底模型也換成
    自己的。

資料來源:專案 data/ 底下的 *.jsonl(標準 messages 格式,一行一段對話)。
          這些語料剛好就是現代 chat SFT 要的格式,直接對應 Qwen 的 chat template。

產出:output_dir/ 底下一個標準的 HF 模型資料夾(可被 transformers 載入推論)。

在 Kaggle 跑法見 kaggle/README.md。本機可用 `--dry-run` 只驗證資料管線
(不需要 GPU、不下載模型)。
"""

import argparse
import glob
import json
import math
import os
import random
import time


# ======================================================================
# 設定區 —— 全部都可以改。你說「布數參數等等全部都可以改」,就是改這裡。
# ======================================================================
class CFG:
    # --- 底模型 ---
    # 預設 0.5B:Kaggle 單張 T4(16GB)可做「全參數微調」,外部套件最少。
    # 想要更強 → 改成 "Qwen/Qwen2.5-1.5B-Instruct" 並把 use_lora 設 True
    # (1.5B 全參數微調塞不下 T4,要用 LoRA)。
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    use_lora = False          # True 需要另外 pip install peft

    # --- 資料 ---
    data_dir = "data"
    system_prompt = (
        "你是 NEXUX,一個由台灣團隊開發的繁體中文 AI 聊天助理。"
        "你親切、誠實、實用,一律用繁體中文回答。"
        "當有人問你是誰、你叫什麼、你是不是 GPT 時,你會清楚說明自己叫 NEXUX。"
    )
    max_len = 1024            # 超過就截斷(我們的語料大多很短,很少觸發)

    # --- 訓練超參數 ---
    epochs = 3
    lr = 2e-5                 # 全參數微調用 2e-5;LoRA 可調高到 1e-4 ~ 2e-4
    weight_decay = 0.01
    warmup_ratio = 0.03       # 前 3% 步數做 learning rate warmup
    batch_size = 2            # 單張卡一次幾段對話
    grad_accum = 8            # 梯度累積 → 有效 batch = batch_size * grad_accum = 16
    max_grad_norm = 1.0
    seed = 1337

    # --- 存檔 / 輸出 ---
    output_dir = "nexux_sft_out"
    log_every = 10            # 每幾個 optimizer step 印一次 loss
    save_each_epoch = True

    # --- LoRA(只有 use_lora=True 時才用到)---
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    lora_target = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]

    # --- 訓練完順手測幾句,肉眼看看像不像話 ---
    sample_prompts = [
        "你叫什麼名字?",
        "你是不是 GPT?",
        "你好",
        "推薦台北一個好玩的地方",
        "幫我想三個週末可以做的事",
    ]


# ======================================================================
# 1) 讀語料:data/*.jsonl → 每段對話的 messages 列表
# ======================================================================
def load_conversations(data_dir):
    """讀 data_dir 底下所有 .jsonl,回傳 [[{role,content},...], ...]。
    跳過空訊息;每段至少要有一組 user→assistant 才有訓練價值。"""
    files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    if not files:
        raise FileNotFoundError(f"{data_dir} 底下找不到任何 .jsonl 語料檔")

    conversations = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msgs = item.get("messages") or []
                msgs = [
                    {"role": m.get("role"), "content": (m.get("content") or "").strip()}
                    for m in msgs
                    if (m.get("content") or "").strip()
                    and m.get("role") in ("user", "assistant")
                ]
                # 至少要有一句 assistant 才能當訓練目標
                if any(m["role"] == "assistant" for m in msgs) and len(msgs) >= 2:
                    conversations.append(msgs)
    return conversations


# ======================================================================
# 2) 把一段對話轉成 (input_ids, labels)
#    labels 只在「assistant 的回覆」上計算 loss,其餘(system/user)設 -100。
#    做法:用 chat template 逐輪累加,算出每一輪新增的 token,
#    assistant 輪的新增 token 才當作學習目標。
# ======================================================================
def build_example(tokenizer, conversation, system_prompt, max_len):
    running = [{"role": "system", "content": system_prompt}]
    ids = tokenizer.apply_chat_template(running, tokenize=True, add_generation_prompt=False)
    input_ids = list(ids)
    labels = [-100] * len(ids)          # system 不學

    for turn in conversation:
        running.append(turn)
        ids_now = tokenizer.apply_chat_template(running, tokenize=True, add_generation_prompt=False)
        delta = ids_now[len(input_ids):]
        if not delta:
            continue
        if turn["role"] == "assistant":
            labels.extend(delta)        # 這一輪(含收尾標記)就是要學的
        else:
            labels.extend([-100] * len(delta))
        input_ids = list(ids_now)

    # 截斷(保留前面,含 system 身分設定)
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]

    # 整段都沒有要學的 token 就不要了
    if all(l == -100 for l in labels):
        return None
    return {"input_ids": input_ids, "labels": labels}


# ======================================================================
# 3) Dataset / 批次整理(padding)
# ======================================================================
def make_dataset(tokenizer, conversations, cfg):
    examples = []
    skipped = 0
    for conv in conversations:
        ex = build_example(tokenizer, conv, cfg.system_prompt, cfg.max_len)
        if ex is None:
            skipped += 1
            continue
        examples.append(ex)
    print(f"[data] 可用訓練樣本: {len(examples)}(跳過 {skipped})")
    return examples


def collate(batch, pad_id):
    import torch
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = len(b["input_ids"])
        pad = maxlen - n
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * n + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


# ======================================================================
# 4) 我們自己的學習率排程:線性 warmup + cosine 衰減(不借 HF 的 scheduler)
# ======================================================================
def lr_at(step, total_steps, base_lr, warmup_steps):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


# ======================================================================
# 5) 主訓練流程
# ======================================================================
def train(cfg):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # T4 不支援 bfloat16,只有 Ampere(A100/3090…)以上才支援;自動偵測。
    bf16_ok = device == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
    print(f"[env] device={device} | bf16_supported={bf16_ok} | amp_dtype={amp_dtype}")

    # --- tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 資料 ---
    conversations = load_conversations(cfg.data_dir)
    print(f"[data] 讀到 {len(conversations)} 段對話")
    examples = make_dataset(tokenizer, conversations, cfg)
    random.shuffle(examples)
    loader = DataLoader(
        examples, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )

    # --- 模型 ---
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float32 if not cfg.use_lora else amp_dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    if cfg.use_lora:
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # --- 優化器:bias / LayerNorm 不做 weight decay ---
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    steps_per_epoch = math.ceil(len(loader) / cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))
    print(f"[plan] 樣本 {len(examples)} | 每 epoch optimizer step {steps_per_epoch} "
          f"| 總 step {total_steps} | warmup {warmup_steps}")

    # fp16 需要 GradScaler 防止梯度 underflow;bf16 不用。
    use_scaler = (amp_dtype == torch.float16)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    global_step = 0
    model.train()
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        optim.zero_grad(set_to_none=True)
        running_loss = 0.0
        micro = 0
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                 dtype=amp_dtype, enabled=(device == "cuda")):
                out = model(**batch)
                loss = out.loss / cfg.grad_accum
            scaler.scale(loss).backward()
            running_loss += out.loss.item()
            micro += 1

            if (i + 1) % cfg.grad_accum == 0 or (i + 1) == len(loader):
                # 套用這一步的學習率
                cur_lr = lr_at(global_step, total_steps, cfg.lr, warmup_steps)
                for g in optim.param_groups:
                    g["lr"] = cur_lr
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % cfg.log_every == 0:
                    avg = running_loss / max(1, micro)
                    elapsed = time.time() - t0
                    print(f"[train] epoch {epoch} step {global_step}/{total_steps} "
                          f"loss {avg:.4f} lr {cur_lr:.2e} ({elapsed:.0f}s)")
                    running_loss, micro = 0.0, 0

        if cfg.save_each_epoch:
            tag = os.path.join(cfg.output_dir, f"epoch{epoch}")
            save_model(model, tokenizer, tag, cfg)
            print(f"[save] 已存 epoch {epoch} → {tag}")

    save_model(model, tokenizer, cfg.output_dir, cfg)
    print(f"[done] 訓練完成,最終模型 → {cfg.output_dir}(總耗時 {time.time()-t0:.0f}s)")

    sample_generate(model, tokenizer, cfg, device)


def save_model(model, tokenizer, path, cfg):
    os.makedirs(path, exist_ok=True)
    if cfg.use_lora:
        # LoRA:存合併後的完整權重,之後推論不用再裝 peft
        merged = model.merge_and_unload()
        merged.save_pretrained(path)
    else:
        model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def sample_generate(model, tokenizer, cfg, device):
    import torch
    model.config.use_cache = True
    model.eval()
    print("\n===== 訓練後試生成 =====")
    for q in cfg.sample_prompts:
        msgs = [{"role": "system", "content": cfg.system_prompt},
                {"role": "user", "content": q}]
        ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                            return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=200, do_sample=True,
                                 temperature=0.7, top_p=0.9,
                                 repetition_penalty=1.1,
                                 pad_token_id=tokenizer.pad_token_id)
        reply = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        print(f"Q: {q}\nA: {reply}\n")


# ======================================================================
# dry-run:不載模型、不碰 GPU,只驗證資料管線(本機就能跑)
# ======================================================================
def dry_run(cfg):
    convs = load_conversations(cfg.data_dir)
    print(f"[dry-run] 讀到 {len(convs)} 段對話")
    # 統計角色分布、長度,用假 tokenizer(字元切分)驗證 build_example 邏輯
    class FakeTok:
        pad_token_id = 0
        def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=False):
            # 模擬 chat template:每則訊息 = 角色標記 + 內容字元
            s = ""
            for m in msgs:
                s += f"<|{m['role']}|>" + m["content"]
            if add_generation_prompt:
                s += "<|assistant|>"
            return [ord(c) % 1000 for c in s] if tokenize else s
    tok = FakeTok()
    ok, none = 0, 0
    total_label_tokens = 0
    lengths = []
    for conv in convs:
        ex = build_example(tok, conv, cfg.system_prompt, cfg.max_len)
        if ex is None:
            none += 1
            continue
        ok += 1
        lengths.append(len(ex["input_ids"]))
        total_label_tokens += sum(1 for l in ex["labels"] if l != -100)
        assert len(ex["input_ids"]) == len(ex["labels"]), "input/labels 長度不一致!"
    lengths.sort()
    print(f"[dry-run] 可用樣本 {ok} | 全遮罩被跳過 {none}")
    if lengths:
        print(f"[dry-run] 樣本長度(字元近似) min={lengths[0]} "
              f"median={lengths[len(lengths)//2]} max={lengths[-1]}")
        print(f"[dry-run] 平均每樣本要學的 token 數 ≈ {total_label_tokens/ok:.1f}")
    print("[dry-run] 資料管線 OK ✅(input_ids / labels 對齊、遮罩正常)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只驗證資料管線,不訓練")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = CFG()
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.model:
        cfg.model_name = args.model
    if args.epochs:
        cfg.epochs = args.epochs
    if args.lora:
        cfg.use_lora = True
    if args.output_dir:
        cfg.output_dir = args.output_dir

    if args.dry_run:
        dry_run(cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
