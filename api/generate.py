"""
api/generate.py
----------------
部署到 Vercel 用的 Serverless Function。

注意:這個檔案刻意「不」使用 config.py、model.py、server.py,
因為那些檔案會 import torch,而 torch 太大,塞不進 Vercel 的大小限制。

這裡改用:
- numpy_gpt.py               (純 numpy 重新實作的推理引擎)
- bert_wordpiece_tokenizer.py(純 Python 重新實作 WordPiece,不需要
                               transformers/tokenizers 套件)
- weights_pretrained.npz     (export_pretrained.py 從微調後的
                               checkpoint_pretrained.pt 轉出來的 int8
                               量化權重)

本機開發(python server.py)走的是 torch 版本(server.py + model.py +
inference.py 的 load_pretrained_model()),兩邊的生成結果理論上幾乎一致。

provider == "own"  NEXUX Model v1.0(見 docs/MODEL_MIGRATION.md):微調
                    ckiplab/gpt2-base-chinese 得到的版本,已通過測試、確認
                    比舊版(從零訓練的 char-level 模型)穩定,正式扶正為
                    唯一的正式版本。

舊版(從零訓練的 char-level 模型)不再部署、不再上傳權重檔案,程式碼與
匯出流程封存在 git 歷史(commit 之前的版本)與 inference.py 的
load_model()/train.py/export_weights.py 裡,方便未來回頭比較,但不會被
這支正式站服務載入。
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, request, jsonify, Response  # noqa: E402
from bert_wordpiece_tokenizer import BertWordpieceTokenizer  # noqa: E402
from numpy_gpt import NumpyGPT  # noqa: E402
from text_cleanup import find_next_turn_marker  # noqa: E402
from providers import call_provider, ProviderError, SUPPORTED_PROVIDERS  # noqa: E402
from conversation import build_context_prompt  # noqa: E402
from smalltalk import match_smalltalk  # noqa: E402
from question_log import log_question  # noqa: E402
from qa_lookup import match_qa  # noqa: E402
from weather_lookup import match_weather  # noqa: E402
import ai_roles  # noqa: E402
import quota_manager  # noqa: E402

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

app = Flask(__name__)

# ---- 推理設定(NEXUX v1.0):repetition_penalty 2.0 是實測後解決重複
# 退化問題(連續生成同一個字)的結果,見 docs/MODEL_MIGRATION.md。 ----
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8
TOP_K = 40
TOP_P = 0.9
REPETITION_PENALTY = 2.0

_cache = {"model": None, "tokenizer": None}


def get_model_and_tokenizer():
    if _cache["model"] is None:
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

        _cache["model"] = NumpyGPT(weights_meta_path, npz_filename="weights_pretrained.npz")
        _cache["tokenizer"] = BertWordpieceTokenizer.load_from_vocab_txt(vocab_path)

    return _cache["model"], _cache["tokenizer"]


def _generate_team_reply(text_prompt, model, tokenizer):
    """
    NexoraAI 團隊模式用:完整生成一次回覆,邏輯跟 server.py 的
    _generate_team_reply() 一致(兩邊的推理引擎不同——這裡是
    numpy_gpt.py,idx/token 是 list[int] 而不是 torch tensor——但停止
    邏輯、回傳格式刻意保持一樣,方便本機/正式站行為對齊)。
    """
    wrapped_prompt = build_context_prompt(None, text_prompt, tokenizer, model.block_size, MAX_NEW_TOKENS)
    idx = tokenizer.encode(wrapped_prompt)
    if len(idx) == 0:
        return "", 0

    accumulated = ""
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
            accumulated = accumulated[:marker.start()]
            break

    return accumulated.rstrip(), len(idx) + step


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
    if provider != "own":
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

    # NexoraAI 第一階段:composer 選了角色時走團隊模式,見 ai_roles.py 開頭的說明
    # ——本質上是同一個 own 模型用不同角色提示語各問一次,不是真正各自獨立思考
    # 的智慧體,角色之間也不會互相看到彼此的回覆再討論。跳過 smalltalk/qa_lookup/
    # weather_lookup 這些為單一問答設計的短路機制,也不用 NDJSON 串流(要等所有
    # 角色+整合都生成完才一次回傳)。跟 server.py 的團隊模式分支邏輯保持一致。
    roles = [r for r in (payload.get("roles") or []) if r in ai_roles.ROLES]
    if roles:
        try:
            model, tokenizer = get_model_and_tokenizer()
        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 400

        client_id = quota_manager.get_client_identifier(request, payload)
        if quota_manager.is_over_limit(client_id):
            status = quota_manager.get_status(client_id)
            minutes = (status["reset_in_seconds"] or 0) // 60
            return jsonify({"error": f"額度已用完,約 {minutes} 分鐘後自動恢復。"}), 429

        status = quota_manager.get_status(client_id)
        estimated_tokens = (len(roles) + 1) * 150
        if status["enabled"] and status["remaining"] < estimated_tokens:
            minutes = (status["reset_in_seconds"] or 0) // 60
            return jsonify({
                "error": f"團隊模式需要呼叫模型 {len(roles) + 1} 次,預估至少需要 "
                         f"{estimated_tokens} token,目前剩餘額度只有 {status['remaining']},"
                         f"約 {minutes} 分鐘後額度會重置,建議減少選取的角色數量或稍後再試。"
            }), 429

        try:
            responses = []
            role_replies_for_integration = []
            total_tokens = 0
            for role in roles:
                role_prompt = ai_roles.build_role_prompt(role, prompt)
                reply_text, tokens_used = _generate_team_reply(role_prompt, model, tokenizer)
                total_tokens += tokens_used
                role_info = ai_roles.ROLES[role]
                responses.append({
                    "role": role, "label": role_info["label"], "icon": role_info["icon"], "reply": reply_text,
                })
                role_replies_for_integration.append((role_info["label"], reply_text))

            integration_prompt = ai_roles.build_integration_prompt(prompt, role_replies_for_integration)
            integration_text, integration_tokens = _generate_team_reply(integration_prompt, model, tokenizer)
            total_tokens += integration_tokens
        except Exception as e:  # noqa: BLE001
            return jsonify({"error": f"團隊模式生成時發生錯誤: {e}"}), 500

        quota_manager.consume(client_id, total_tokens)

        return jsonify({"type": "team", "responses": responses, "integration": integration_text})

    # 短的問候/道別/道謝類輸入,直接用規則比對回覆,不經過模型生成。
    # skip_rules 開關(NEXUX.html「略過規則」勾選框)讓使用者可以強制跳過
    # 這兩層保底機制,直接看模型自己會怎麼生成,方便實測模型本身的能力。
    if not skip_rules:
        smalltalk_match = match_smalltalk(prompt, history)
        if smalltalk_match is not None:
            smalltalk_reply, smalltalk_category = smalltalk_match
            return jsonify({"reply": smalltalk_reply, "type": smalltalk_category})

        # 訓練語料裡「本來就有標準答案」的問題(qa.jsonl / html.jsonl / python.jsonl),
        # 直接比對回傳原始答案,不用冒險讓模型生成。
        qa_reply = match_qa(prompt, data_dir=os.path.join(BASE_DIR, "data"))
        if qa_reply is not None:
            return jsonify({"reply": qa_reply, "type": "qa_lookup"})

        # 問「現在天氣/氣溫/有沒有下雨」這類問題時,直接呼叫中央氣象署
        # API 拿真實觀測資料回答,不要讓模型自己編數字(見 weather_lookup.py
        # 的說明:這只回答得了「現在」的觀測狀況,答不了「未來預報」)。
        weather_reply = match_weather(prompt)
        if weather_reply is not None:
            return jsonify({"reply": weather_reply, "type": "weather_lookup"})

    try:
        model, tokenizer = get_model_and_tokenizer()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400

    # own 模型才做額度限制(見 quota_manager.py):跑在自己伺服器上會佔用
    # 運算資源,防止單一使用者無限制濫用;第三方 provider 不受影響,前面
    # 已經 return 掉了,不會執行到這裡。沒設定 Upstash 時 quota_manager
    # 永遠回傳 False,行為等同沒有限制。
    client_id = quota_manager.get_client_identifier(request, payload)
    if quota_manager.is_over_limit(client_id):
        status = quota_manager.get_status(client_id)
        minutes = (status["reset_in_seconds"] or 0) // 60
        return jsonify({"error": f"額度已用完,約 {minutes} 分鐘後自動恢復。"}), 429

    try:
        # 只有模型「真的經過 SFT 訓練」時,才包裝成問答格式,並帶入歷史對話當作 context;
        # 否則模型從沒見過這種格式,硬套上去只會讓生成效果更差。
        if model.is_sft:
            wrapped_prompt = build_context_prompt(
                history, prompt, tokenizer, model.block_size, MAX_NEW_TOKENS
            )
        else:
            wrapped_prompt = prompt

        idx = tokenizer.encode(wrapped_prompt)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"處理輸入時發生錯誤: {e}"}), 500

    if len(idx) == 0:
        return jsonify({"error": "輸入的文字包含詞表以外的字元,請換一句話試試。"}), 400

    if debug:
        print(
            f"[inference-debug] context tokens: {len(idx)}/{model.block_size} "
            f"| history turns received: {len(history)} "
            f"| wrapped_prompt:\n{wrapped_prompt!r}"
        )

    def stream():
        """
        以 NDJSON(一行一個 JSON 物件)串流回傳生成結果,每一行格式是
        {"delta": "這次新增的文字", "n": 目前已生成的token數},最後固定
        以 {"done": true, "total_tokens": 總token數} 結尾。前端(NEXUX.html)
        靠這個協定即時更新「正在生成... Output tokens: N」的顯示,同時
        還是能像以前一樣把 delta 逐段接起來組成完整回覆文字,不影響原本
        的生成速度(仍然靠 numpy_gpt.py 的 KV-cache 加速)。
        """
        accumulated = ""
        sent_len = 0
        # 尾巴保留幾個字元先不送出,避免剛好把「換行標記」(例如 \n答:)送出一半。
        HOLD = 3
        step = 0
        start_time = time.time()

        try:
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
                # 迴圈正常跑完(沒有被上面 marker 的 break 中斷),把剩餘尾巴吐出來
                final_text = accumulated.rstrip()
                if len(final_text) > sent_len:
                    yield json.dumps({"delta": final_text[sent_len:], "n": step}, ensure_ascii=False) + "\n"

            if debug:
                print(f"[inference-debug] generated {step} tokens in {time.time()-start_time:.2f}s")

            quota_manager.consume(client_id, len(idx) + step)
            yield json.dumps({"done": True, "total_tokens": step}, ensure_ascii=False) + "\n"

        except Exception as e:  # noqa: BLE001 - 攔截所有例外,回傳給前端顯示
            quota_manager.consume(client_id, len(idx) + step)
            yield json.dumps({"delta": f"\n[生成時發生錯誤: {e}]", "n": step}, ensure_ascii=False) + "\n"
            yield json.dumps({"done": True, "total_tokens": step}, ensure_ascii=False) + "\n"

    return Response(stream(), mimetype="application/x-ndjson; charset=utf-8")


@app.route("/api/quota", methods=["GET"])
def api_quota():
    """
    給前端(composer 旁的額度顯示)輪詢用,回傳目前的額度使用狀況,
    不用每次都跟著 /api/generate 一起回傳。見 quota_manager.py 的說明。
    """
    client_id = quota_manager.get_client_identifier(request)
    return jsonify(quota_manager.get_status(client_id))
