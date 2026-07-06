"""
server.py
---------
一個很輕量的本地伺服器,負責:
1. 提供 index.html 這個聊天介面網頁
2. 提供 /api/generate 這個 API,讓網頁把使用者輸入的文字傳過來,
   由這裡呼叫 model.py / inference.py 產生回覆,再傳回網頁顯示。

模型只會在第一次收到請求時載入一次,之後的請求都重複使用同一個模型,
不會每次都重新讀取 checkpoint,回應速度會快很多。

啟動方式:
    python server.py
啟動後,用瀏覽器打開 http://localhost:5000 即可使用。
"""

import os
import torch
from flask import Flask, request, jsonify, send_from_directory

from config import Config
from inference import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=None)

# ---- 模型快取,避免每次請求都重新載入 checkpoint ----
_cache = {"model": None, "tokenizer": None, "config": None, "is_sft": False}


def get_model_and_tokenizer():
    if _cache["model"] is None:
        cfg = Config()
        model, tokenizer, is_sft = load_model(cfg)  # 找不到 checkpoint 時,這裡會拋出 FileNotFoundError
        _cache["model"] = model
        _cache["tokenizer"] = tokenizer
        _cache["config"] = cfg
        _cache["is_sft"] = is_sft
        print(f"[server] 模型已載入並快取(SFT問答模式: {is_sft})")
    return _cache["config"], _cache["model"], _cache["tokenizer"], _cache["is_sft"]


@app.route("/")
def index():
    """提供聊天介面的 HTML 檔案。"""
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """
    接收 { "prompt": "使用者輸入的文字" },
    回傳 { "reply": "模型生成的回覆" } 或 { "error": "錯誤訊息" }。
    """
    payload = request.get_json(silent=True) or {}
    prompt = (payload.get("prompt") or "").strip()

    if not prompt:
        return jsonify({"error": "請輸入內容再送出。"}), 400

    try:
        config, model, tokenizer, is_sft = get_model_and_tokenizer()
    except FileNotFoundError:
        return jsonify({
            "error": "還沒有訓練好的模型。請先在終端機執行「python train.py」完成訓練,"
                     "再重新啟動 server.py。"
        }), 400

    try:
        # 只有模型「真的經過 SFT 訓練」時,才包裝成問答格式,
        # 否則模型從沒見過這種格式,硬套上去只會讓生成效果更差。
        wrapped_prompt = f"問:{prompt}\n答:" if is_sft else prompt

        idx = torch.tensor(
            [tokenizer.encode(wrapped_prompt)], dtype=torch.long, device=config.device
        )
        if idx.shape[1] == 0:
            return jsonify({"error": "輸入的文字包含詞表以外的字元,請換一句話試試。"}), 400

        out_idx = model.generate(
            idx,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
        )
        full_text = tokenizer.decode(out_idx[0].tolist())

        # 只取「新生成」的部分回覆給前端,不要把包裝用的文字也顯示給使用者
        reply = full_text[len(wrapped_prompt):] if full_text.startswith(wrapped_prompt) else full_text

        return jsonify({"reply": reply})

    except Exception as e:  # noqa: BLE001 - 這裡刻意攔截所有例外,回傳給前端顯示
        return jsonify({"error": f"生成時發生錯誤: {e}"}), 500


if __name__ == "__main__":
    print("[server] 啟動中,請用瀏覽器開啟 http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
    