"""
railway_server.py
------------------
Railway 部署用的推理伺服器。跟 api/generate.py 用同一套 numpy_gpt.py
推理引擎，但以常駐容器運行——模型啟動時載入一次、常駐記憶體，
之後每個請求直接推理，沒有 Vercel serverless 的冷啟動和 timeout 問題。

Vercel 前端透過 OWN_INFERENCE_URL 環境變數把 own 模型的請求轉過來。
"""

import json
import os
import sys
import time

from flask import Flask, request, jsonify, Response

from bert_wordpiece_tokenizer import BertWordpieceTokenizer
from numpy_gpt import NumpyGPT
from text_cleanup import find_next_turn_marker
from conversation import build_context_prompt

app = Flask(__name__)

MAX_NEW_TOKENS = 100
TEMPERATURE = 0.5
TOP_K = 20
TOP_P = 0.8
REPETITION_PENALTY = 2.0

_model = None
_tokenizer = None


def get_model_and_tokenizer():
    global _model, _tokenizer
    if _model is None:
        print("[railway] 載入 LYNX 模型...", flush=True)
        t0 = time.time()
        _model = NumpyGPT("weights_meta_pretrained.json")
        _tokenizer = BertWordpieceTokenizer.load_from_vocab_txt("vocab_pretrained.txt")
        print(f"[railway] LYNX 載入完成 ({time.time()-t0:.1f}s) "
              f"n_layer={_model.n_layer} n_embd={_model.n_embd}", flush=True)
    return _model, _tokenizer


@app.route("/api/generate", methods=["POST"])
def api_generate():
    payload = request.get_json(silent=True) or {}
    prompt = (payload.get("prompt") or "").strip()
    history = payload.get("history") or []

    if not prompt:
        return jsonify({"error": "請輸入內容再送出。"}), 400

    model, tokenizer = get_model_and_tokenizer()

    if model.is_sft:
        wrapped_prompt = build_context_prompt(
            history, prompt, tokenizer, model.block_size, MAX_NEW_TOKENS
        )
    else:
        wrapped_prompt = prompt

    idx = tokenizer.encode(wrapped_prompt)
    if len(idx) == 0:
        return jsonify({"error": "輸入的文字包含詞表以外的字元,請換一句話試試。"}), 400

    def stream():
        accumulated = ""
        sent_len = 0
        HOLD = 3
        step = 0

        for token_id in model.generate_stream(
            idx,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            eos_id=getattr(tokenizer, "eos_id", None),
        ):
            step += 1
            accumulated += tokenizer.decode([token_id])

            marker = find_next_turn_marker(accumulated)
            if marker:
                final_text = accumulated[:marker.start()].rstrip()
                if len(final_text) > sent_len:
                    yield json.dumps({"delta": final_text[sent_len:], "n": step}, ensure_ascii=False) + "\n"
                break

            safe_len = max(0, len(accumulated) - HOLD)
            if safe_len > sent_len:
                yield json.dumps({"delta": accumulated[sent_len:safe_len], "n": step}, ensure_ascii=False) + "\n"
                sent_len = safe_len
        else:
            final_text = accumulated.rstrip()
            if len(final_text) > sent_len:
                yield json.dumps({"delta": final_text[sent_len:], "n": step}, ensure_ascii=False) + "\n"

        yield json.dumps({"done": True, "total_tokens": step}, ensure_ascii=False) + "\n"

    return Response(stream(), mimetype="application/x-ndjson; charset=utf-8")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "LYNX"})


print("[railway] 啟動中，預先載入模型...", flush=True)
get_model_and_tokenizer()
print("[railway] LYNX 推理伺服器就緒", flush=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
