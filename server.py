"""
server.py
---------
一個很輕量的本地伺服器,負責:
1. 提供 CHATBOT.html 這個聊天介面網頁
2. 提供 /api/generate 這個 API,讓網頁把使用者輸入的文字傳過來,
   由這裡呼叫 model.py / inference.py 產生回覆,再傳回網頁顯示。

模型只會在第一次收到請求時載入一次,之後的請求都重複使用同一個模型,
不會每次都重新讀取 checkpoint,回應速度會快很多。

啟動方式:
    python server.py
啟動後,用瀏覽器打開 http://localhost:5000 即可使用。
"""

import os

# 跟 train.py 同樣的道理:這台機器上 torch 用多執行緒跑 CPU 運算時偶爾會跟
# OpenMP/MKL 搶執行緒資源導致不穩定,推理(生成回覆)時也可能受影響,
# 在 import torch 之前先鎖定成單執行緒模式。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory, Response

torch.set_num_threads(1)

# 本機開發用 .env 檔案提供第三方 API 金鑰(OPENAI_API_KEY 等),
# 這個檔案不會被 commit 上傳(見 .gitignore)。
load_dotenv()

from config import Config
from inference import load_model
from text_cleanup import find_next_turn_marker
from providers import call_provider, ProviderError, SUPPORTED_PROVIDERS

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
    return send_from_directory(BASE_DIR, "CHATBOT.html")


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """
    接收 { "prompt": "使用者輸入的文字" },
    以串流(text/plain,chunked transfer)的方式把模型生成的文字逐字傳回去,
    讓前端可以邊生成邊顯示,不用等整段回覆生成完才看到文字,大幅縮短「等待感」。
    """
    payload = request.get_json(silent=True) or {}
    prompt = (payload.get("prompt") or "").strip()
    provider = payload.get("provider") or "own"

    if not prompt:
        return jsonify({"error": "請輸入內容再送出。"}), 400

    if provider not in SUPPORTED_PROVIDERS:
        return jsonify({"error": f"不支援的模型來源: {provider}"}), 400

    # 第三方 API(OpenAI / Anthropic / Google / Groq)只是暫時借來頂著用,
    # 沒有串接串流,一次生成完整回覆再一次回傳,前端會自己補上打字機效果。
    if provider != "own":
        try:
            reply = call_provider(provider, prompt)
        except ProviderError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:  # noqa: BLE001
            return jsonify({"error": f"呼叫 {provider} 時發生錯誤: {e}"}), 500
        return jsonify({"reply": reply})

    try:
        config, model, tokenizer, is_sft = get_model_and_tokenizer()
    except FileNotFoundError:
        return jsonify({
            "error": "還沒有訓練好的模型。請先在終端機執行「python train.py」完成訓練,"
                     "再重新啟動 server.py。"
        }), 400

    # 只有模型「真的經過 SFT 訓練」時,才包裝成問答格式,
    # 否則模型從沒見過這種格式,硬套上去只會讓生成效果更差。
    wrapped_prompt = f"問:{prompt}\n答:" if is_sft else prompt

    idx = torch.tensor(
        [tokenizer.encode(wrapped_prompt)], dtype=torch.long, device=config.device
    )
    if idx.shape[1] == 0:
        return jsonify({"error": "輸入的文字包含詞表以外的字元,請換一句話試試。"}), 400

    def stream():
        accumulated = ""
        sent_len = 0
        # 尾巴保留幾個字元先不送出,避免剛好把「換行標記」(例如 \nA:)送出一半,
        # 等累積的文字夠長、確定不是標記的開頭之後,才把安全的部分吐給前端。
        HOLD = 3

        try:
            for token_ids in model.generate_stream(
                idx,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
            ):
                accumulated += tokenizer.decode(token_ids)

                marker = find_next_turn_marker(accumulated)
                if marker:
                    final_text = accumulated[:marker.start()].rstrip()
                    if len(final_text) > sent_len:
                        yield final_text[sent_len:]
                    return

                safe_len = max(0, len(accumulated) - HOLD)
                if safe_len > sent_len:
                    yield accumulated[sent_len:safe_len]
                    sent_len = safe_len

            final_text = accumulated.rstrip()
            if len(final_text) > sent_len:
                yield final_text[sent_len:]

        except Exception as e:  # noqa: BLE001 - 這裡刻意攔截所有例外,回傳給前端顯示
            yield f"\n[生成時發生錯誤: {e}]"

    return Response(stream(), mimetype="text/plain; charset=utf-8")


if __name__ == "__main__":
    print("[server] 啟動中,請用瀏覽器開啟 http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
    