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

provider == "own"      沿用原本從零訓練的 char-level 模型,行為不變。
provider == "own_beta" 微調過的預訓練模型(見 docs/MODEL_MIGRATION.md),
                        格式(懂得自己收尾)比舊模型好很多,但內容準確度
                        還在驗證中,故意先做成可切換的選項,不直接取代
                        預設模型,讓使用者自己選擇要不要試用,累積實際
                        使用情況後再考慮要不要扶正成預設。
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, request, jsonify  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402
from bert_wordpiece_tokenizer import BertWordpieceTokenizer  # noqa: E402
from numpy_gpt import NumpyGPT  # noqa: E402
from text_cleanup import truncate_at_next_turn  # noqa: E402
from providers import call_provider, ProviderError, SUPPORTED_PROVIDERS  # noqa: E402
from conversation import build_context_prompt  # noqa: E402
from smalltalk import match_smalltalk  # noqa: E402
from question_log import log_question  # noqa: E402
from qa_lookup import match_qa  # noqa: E402

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

app = Flask(__name__)

# NEXUX v1.0(見 docs/MODEL_MIGRATION.md):ckiplab/gpt2-base-chinese 微調
# 10000 步的版本正式定案為第一個穩定基準,之後的改進都從這個版本迭代。
NEXUX_VERSION = "v1.0"

# ---- 推理設定:原本的 char-level 模型 ----
MAX_NEW_TOKENS = 60
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.9
REPETITION_PENALTY = 1.3

# ---- 推理設定:預訓練微調模型(own_beta),參數是分開調的 ----
# repetition_penalty 從 1.3 調高到 2.0,是實測後解決重複退化問題
# (連續生成同一個字)的結果。
BETA_MAX_NEW_TOKENS = 100
BETA_TEMPERATURE = 0.8
BETA_TOP_K = 40
BETA_TOP_P = 0.9
BETA_REPETITION_PENALTY = 2.0

_cache = {"model": None, "tokenizer": None}
_beta_cache = {"model": None, "tokenizer": None}


def get_model_and_tokenizer():
    if _cache["model"] is None:
        weights_meta_path = os.path.join(BASE_DIR, "weights_meta.json")
        weights_npz_path = os.path.join(BASE_DIR, "weights.npz")
        tokenizer_path = os.path.join(BASE_DIR, "tokenizer.json")

        if (
            not os.path.exists(weights_meta_path)
            or not os.path.exists(weights_npz_path)
            or not os.path.exists(tokenizer_path)
        ):
            raise FileNotFoundError(
                "找不到 weights_meta.json / weights.npz 或 tokenizer.json。"
                "請先在本機執行「python train.py」訓練模型,"
                "再執行「python export_weights.py」匯出權重,"
                "最後把 weights_meta.json、weights.npz 和 tokenizer.json 一起 commit 上傳。"
            )

        _cache["model"] = NumpyGPT(weights_meta_path)
        _cache["tokenizer"] = CharTokenizer.load(tokenizer_path)

    return _cache["model"], _cache["tokenizer"]


def get_beta_model_and_tokenizer():
    if _beta_cache["model"] is None:
        weights_meta_path = os.path.join(BASE_DIR, "weights_meta_pretrained.json")
        weights_npz_path = os.path.join(BASE_DIR, "weights_pretrained.npz")
        vocab_path = os.path.join(BASE_DIR, "vocab_pretrained.txt")

        if (
            not os.path.exists(weights_meta_path)
            or not os.path.exists(weights_npz_path)
            or not os.path.exists(vocab_path)
        ):
            raise FileNotFoundError(
                "找不到 weights_meta_pretrained.json / weights_pretrained.npz "
                "或 vocab_pretrained.txt。請先執行「python convert_pretrained.py」"
                "「python run_pretrained_sft.py」「python export_pretrained.py」,"
                "再把這三個檔案一起 commit 上傳。"
            )

        _beta_cache["model"] = NumpyGPT(weights_meta_path, npz_filename="weights_pretrained.npz")
        _beta_cache["tokenizer"] = BertWordpieceTokenizer.load_from_vocab_txt(vocab_path)

    return _beta_cache["model"], _beta_cache["tokenizer"]


@app.route("/api/generate", methods=["POST"])
def api_generate():
    payload = request.get_json(silent=True) or {}
    prompt = (payload.get("prompt") or "").strip()
    provider = payload.get("provider") or "own"
    history = payload.get("history") or []
    debug = bool(payload.get("debug"))
    skip_rules = bool(payload.get("skipRules"))

    if not prompt:
        return jsonify({"error": "請輸入內容再送出。"}), 400

    if provider not in SUPPORTED_PROVIDERS:
        return jsonify({"error": f"不支援的模型來源: {provider}"}), 400

    # 記錄使用者實際輸入的問題(見 question_log.py),方便之後回顧真實
    # 使用情境、補強語料。如果沒有設定 Upstash 的環境變數,這裡會直接
    # 靜默略過,不影響任何回覆邏輯。
    log_question(prompt, provider)

    # 第三方 API(OpenAI / Anthropic / Google / Groq)只是暫時借來頂著用,
    # 金鑰要另外在 Vercel 專案的 Environment Variables 設定裡加好。
    if provider not in ("own", "own_beta"):
        try:
            reply = call_provider(provider, prompt)
        except ProviderError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:  # noqa: BLE001
            import traceback
            return jsonify({
                "error": f"呼叫 {provider} 時發生錯誤: {e}",
                "traceback": traceback.format_exc(),
            }), 500
        return jsonify({"reply": reply})

    # 短的問候/道別/道謝類輸入,直接用規則比對回覆,不經過模型生成,
    # 不管選的是哪個模型都適用(這一層跟底層生成模型無關)。
    # skip_rules 開關(NEXUX.html「略過規則」勾選框)讓使用者可以強制跳過
    # 這兩層保底機制,直接看選定的模型自己會怎麼生成——主要是為了實測
    # own_beta 面對這些簡單輸入時的真實能力,不被規則系統擋住。
    if not skip_rules:
        smalltalk_match = match_smalltalk(prompt, history)
        if smalltalk_match is not None:
            smalltalk_reply, smalltalk_category = smalltalk_match
            return jsonify({"reply": smalltalk_reply, "type": smalltalk_category, "version": NEXUX_VERSION})

        # 訓練語料裡「本來就有標準答案」的問題(qa.jsonl / html.jsonl / python.jsonl),
        # 直接比對回傳原始答案,不用冒險讓模型生成,同樣不管選哪個模型都適用。
        qa_reply = match_qa(prompt, data_dir=os.path.join(BASE_DIR, "data"))
        if qa_reply is not None:
            return jsonify({"reply": qa_reply, "type": "qa_lookup", "version": NEXUX_VERSION})

    is_beta = provider == "own_beta"
    try:
        if is_beta:
            model, tokenizer = get_beta_model_and_tokenizer()
        else:
            model, tokenizer = get_model_and_tokenizer()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400

    max_new_tokens = BETA_MAX_NEW_TOKENS if is_beta else MAX_NEW_TOKENS
    temperature = BETA_TEMPERATURE if is_beta else TEMPERATURE
    top_k = BETA_TOP_K if is_beta else TOP_K
    top_p = BETA_TOP_P if is_beta else TOP_P
    repetition_penalty = BETA_REPETITION_PENALTY if is_beta else REPETITION_PENALTY

    try:
        # 只有模型「真的經過 SFT 訓練」時,才包裝成問答格式,並帶入歷史對話當作 context;
        # 否則模型從沒見過這種格式,硬套上去只會讓生成效果更差。
        if model.is_sft:
            wrapped_prompt = build_context_prompt(
                history, prompt, tokenizer, model.block_size, max_new_tokens
            )
        else:
            wrapped_prompt = prompt

        idx = tokenizer.encode(wrapped_prompt)
        if len(idx) == 0:
            return jsonify({"error": "輸入的文字包含詞表以外的字元,請換一句話試試。"}), 400

        if debug:
            print(
                f"[inference-debug] provider={provider} context tokens: {len(idx)}/{model.block_size} "
                f"| history turns received: {len(history)} "
                f"| wrapped_prompt:\n{wrapped_prompt!r}"
            )

        start_time = time.time()
        out_idx = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_id=getattr(tokenizer, "eos_id", None),
        )
        # 用「token 數量」切開新生成的部分,而不是把整段解碼成文字後再用
        # 字串比對開頭:BertWordpieceTokenizer 的 decode() 會做小寫化、
        # 標點間距調整等正規化,解碼後的文字不保證跟原始輸入的 wrapped_prompt
        # 逐字一致(例如英文大小寫),字串比對法在新 tokenizer 下不可靠,
        # 用 token id 切割才是穩固的做法,不管哪種 tokenizer 都適用。
        reply = tokenizer.decode(out_idx[len(idx):])
        reply = truncate_at_next_turn(reply)

        if debug:
            print(
                f"[inference-debug] generated {len(out_idx) - len(idx)} tokens "
                f"in {time.time() - start_time:.2f}s"
            )

        return jsonify({"reply": reply, "version": NEXUX_VERSION})

    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"生成時發生錯誤: {e}"}), 500
