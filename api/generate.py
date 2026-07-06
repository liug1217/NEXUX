"""
api/generate.py
----------------
部署到 Vercel 用的 Serverless Function。

注意:這個檔案刻意「不」使用 config.py、model.py、server.py,
因為那些檔案會 import torch,而 torch 太大,塞不進 Vercel 的大小限制。

這裡改用:
- numpy_gpt.py  (純 numpy 重新實作的推理引擎)
- tokenizer.py  (原本就沒有依賴 torch,可以直接沿用)
- weights.json  (用 export_weights.py 從 checkpoint.pt 轉出來的純數字權重)

本機開發(python server.py)走的是 torch 版本(server.py + model.py),
兩邊的生成結果理論上幾乎一致(誤差在小數點後 5、6 位,不影響生成內容)。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, request, jsonify  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402
from numpy_gpt import NumpyGPT  # noqa: E402

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

app = Flask(__name__)

# ---- 推理設定(對應原本 config.py 裡跟生成有關的參數) ----
MAX_NEW_TOKENS = 60
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.9
REPETITION_PENALTY = 1.3

_cache = {"model": None, "tokenizer": None}


def get_model_and_tokenizer():
    if _cache["model"] is None:
        weights_path = os.path.join(BASE_DIR, "weights.json")
        tokenizer_path = os.path.join(BASE_DIR, "tokenizer.json")

        if not os.path.exists(weights_path) or not os.path.exists(tokenizer_path):
            raise FileNotFoundError(
                "找不到 weights.json 或 tokenizer.json。請先在本機執行「python train.py」"
                "訓練模型,再執行「python export_weights.py」匯出權重,"
                "最後把 weights.json 和 tokenizer.json 一起 commit 上傳。"
            )

        _cache["model"] = NumpyGPT(weights_path)
        _cache["tokenizer"] = CharTokenizer.load(tokenizer_path)

    return _cache["model"], _cache["tokenizer"]


@app.route("/api/generate", methods=["POST"])
def api_generate():
    payload = request.get_json(silent=True) or {}
    prompt = (payload.get("prompt") or "").strip()

    if not prompt:
        return jsonify({"error": "請輸入內容再送出。"}), 400

    try:
        model, tokenizer = get_model_and_tokenizer()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400

    try:
        # 只有模型「真的經過 SFT 訓練」時,才包裝成問答格式,
        # 否則模型從沒見過這種格式,硬套上去只會讓生成效果更差。
        wrapped_prompt = f"問:{prompt}\n答:" if model.is_sft else prompt

        idx = tokenizer.encode(wrapped_prompt)
        if len(idx) == 0:
            return jsonify({"error": "輸入的文字包含詞表以外的字元,請換一句話試試。"}), 400

        out_idx = model.generate(
            idx,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
        )
        full_text = tokenizer.decode(out_idx)
        reply = full_text[len(wrapped_prompt):] if full_text.startswith(wrapped_prompt) else full_text

        return jsonify({"reply": reply})

    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"生成時發生錯誤: {e}"}), 500
    