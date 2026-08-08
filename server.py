"""
server.py
---------
一個很輕量的本地伺服器,負責:
1. 提供 NEXUX.html 這個聊天介面網頁
2. 提供 /api/generate 這個 API,讓網頁把使用者輸入的文字傳過來,
   由這裡呼叫 model.py / inference.py 產生回覆,再傳回網頁顯示。

模型只會在第一次收到請求時載入一次,之後的請求都重複使用同一個模型,
不會每次都重新讀取 checkpoint,回應速度會快很多。

啟動方式:
    python server.py
啟動後,用瀏覽器打開 http://localhost:5000 即可使用。
"""

import json
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

import time

from config import Config
from inference import load_pretrained_model
from text_cleanup import find_next_turn_marker
from providers import call_provider, ProviderError, SUPPORTED_PROVIDERS
from conversation import build_context_prompt
from smalltalk import match_smalltalk
from qa_lookup import match_qa
from weather_lookup import match_weather
from bead_pattern import generate_pattern, DEFAULT_GRID, MIN_GRID, MAX_GRID
import ai_roles
import quota_manager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=None)

# ---- 模型快取,避免每次請求都重新載入 checkpoint ----
_cache = {"model": None, "tokenizer": None, "config": None, "is_sft": False}


def get_model_and_tokenizer():
    """
    NEXUX v1.0:本機開發伺服器現在載入的是微調過的預訓練模型(見
    docs/MODEL_MIGRATION.md),跟正式站(Vercel)服務的是同一個版本。
    需要先在本機執行過 convert_pretrained.py + run_pretrained_sft.py,
    產生 checkpoint_pretrained.pt 才能載入(這個檔案不會進 git)。
    """
    if _cache["model"] is None:
        model, tokenizer, is_sft = load_pretrained_model()  # 找不到 checkpoint 時,這裡會拋出 FileNotFoundError
        base = Config()
        # 沿用 api/generate.py 的 own_beta 生成參數(見那裡的說明:
        # repetition_penalty 2.0 是實測解決重複退化問題的結果)。
        cfg = Config(**{
            **base.__dict__,
            "block_size": 1024,
            "max_new_tokens": 100,
            "temperature": 0.8,
            "top_k": 40,
            "top_p": 0.9,
            "repetition_penalty": 2.0,
        })
        _cache["model"] = model
        _cache["tokenizer"] = tokenizer
        _cache["config"] = cfg
        _cache["is_sft"] = is_sft
        print(f"[server] 模型已載入並快取(SFT問答模式: {is_sft})")
    return _cache["config"], _cache["model"], _cache["tokenizer"], _cache["is_sft"]


def _generate_team_reply(text_prompt, config, model, tokenizer):
    """
    NexoraAI 團隊模式用:完整生成一次回覆(不像 /api/generate 一般模式那樣
    邊生成邊串流給前端——團隊模式要等所有角色+整合都生成完才一次回傳,
    所以這裡直接把 model.generate_stream() 產生的 token 在後端內部收集完,
    回傳完整文字跟這次呼叫實際用掉的 token 數(給 quota_manager 累加)。
    停止邏輯(遇到 find_next_turn_marker 就截斷)跟一般模式共用同一套,
    避免模型生成到下一輪的「問:/答:」標記還繼續往下寫。
    """
    wrapped_prompt = build_context_prompt(None, text_prompt, tokenizer, config.block_size, config.max_new_tokens)
    idx = torch.tensor([tokenizer.encode(wrapped_prompt)], dtype=torch.long, device=config.device)
    if idx.shape[1] == 0:
        return "", 0

    accumulated = ""
    step = 0
    for token_ids in model.generate_stream(
        idx,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
        eos_id=getattr(tokenizer, "eos_id", None),
    ):
        step += 1
        accumulated += tokenizer.decode(token_ids)
        marker = find_next_turn_marker(accumulated)
        if marker:
            accumulated = accumulated[:marker.start()]
            break

    return accumulated.rstrip(), idx.shape[1] + step


@app.route("/")
def index():
    """提供聊天介面的 HTML 檔案。"""
    return send_from_directory(BASE_DIR, "NEXUX.html")


@app.route("/NEXUX.png")
def favicon():
    """提供瀏覽器分頁圖示 / logo 用的圖片。"""
    return send_from_directory(BASE_DIR, "NEXUX.png")


@app.route("/download.html")
def download_page():
    """獨立的桌面版下載頁面,只放 NEXUX_desktop.exe 一個下載按鈕。"""
    return send_from_directory(BASE_DIR, "download.html")


@app.route("/local_python_agent.py")
def local_python_agent_download():
    """
    提供本機執行代理程式(local_python_agent.py)下載,讓使用者的電腦
    有真正的 Python 時,程式碼編輯器面板可以改用它執行,而不是只能用
    Pyodide(瀏覽器端 WASM,功能較受限)。詳見該檔案開頭的說明。
    """
    return send_from_directory(BASE_DIR, "local_python_agent.py", mimetype="text/x-python")


@app.route("/start_local_python_agent.bat")
def local_python_agent_launcher_download():
    """
    跟 local_python_agent.py 放同一個資料夾、雙擊就能啟動的批次檔,
    讓使用者不用自己打開終端機、手動輸入指令(Windows 專用)。
    """
    return send_from_directory(BASE_DIR, "start_local_python_agent.bat", mimetype="application/bat")


@app.route("/local_python_agent.exe")
def local_python_agent_exe_download():
    """
    local_python_agent.py 用 PyInstaller 打包成的單一 exe(Windows 專用),
    雙擊就能直接執行,不用先裝 Python 才能「開啟這個程式本身」——但
    執行使用者送來的程式碼時,還是會去找電腦上真正安裝的 Python
    (見 local_python_agent.py 的 _resolve_python_executable() 說明),
    才能用到使用者自己裝的套件/GPU。想直接看原始碼再自己執行的人,
    可以改用 local_python_agent.py + start_local_python_agent.bat 這組。
    """
    return send_from_directory(BASE_DIR, "local_python_agent.exe", mimetype="application/octet-stream")


@app.route("/NEXUX_desktop.exe")
def nexux_desktop_download():
    """
    整個 NEXUX 聊天網站包成的桌面版捷徑(pywebview 原生視窗載入
    https://ai.nexuxai.net,需要網路連線,裡面沒有內建 AI 模型,
    見 desktop_app.py 的說明)。
    """
    return send_from_directory(BASE_DIR, "NEXUX_desktop.exe", mimetype="application/octet-stream")


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
    history = payload.get("history") or []
    debug = bool(payload.get("debug"))
    skip_rules = bool(payload.get("skipRules"))

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

    # NexoraAI 第一階段:composer 選了角色時走團隊模式,見 ai_roles.py 開頭的說明
    # ——本質上是同一個 own 模型用不同角色提示語各問一次,不是真正各自獨立思考
    # 的智慧體,角色之間也不會互相看到彼此的回覆再討論。跳過 smalltalk/qa_lookup/
    # weather_lookup 這些為單一問答設計的短路機制(團隊模式的意義在於「不同角色
    # 的觀點」,直接查表回同一句罐頭答案沒有意義),也不用 NDJSON 串流(要等所有
    # 角色+整合都生成完才一次回傳)。
    roles = [r for r in (payload.get("roles") or []) if r in ai_roles.ROLES]
    # 自訂角色(composer 角色選單裡的「+ 自訂角色」):使用者自己輸入角色
    # 名稱,沒有預先寫好的專屬提示語,見 ai_roles.build_custom_role_prompt()。
    # sanitize_custom_role_labels() 是真正生效的數量/長度上限,不能只信
    # 前端已經做過的限制。
    custom_role_labels = ai_roles.sanitize_custom_role_labels(payload.get("customRoles"))
    if roles or custom_role_labels:
        try:
            config, model, tokenizer, is_sft = get_model_and_tokenizer()
        except FileNotFoundError:
            return jsonify({
                "error": "還沒有本機微調好的模型。請先在終端機依序執行"
                         "「python convert_pretrained.py」「python run_pretrained_sft.py」,"
                         "再重新啟動 server.py。"
            }), 400

        client_id = quota_manager.get_client_identifier(request, payload)
        if quota_manager.is_over_limit(client_id):
            status = quota_manager.get_status(client_id)
            minutes = (status["reset_in_seconds"] or 0) // 60
            return jsonify({"error": f"額度已用完,約 {minutes} 分鐘後自動恢復。"}), 429

        # 團隊模式會呼叫模型 (len(roles)+len(custom_role_labels))+1 次
        # (每個角色一次+核心AI整合一次),用粗略估計值在生成前先擋下明顯
        # 不足的額度,避免生成到一半才因為額度用完而中斷、浪費前面已經
        # 生成的角色回覆。
        total_role_count = len(roles) + len(custom_role_labels)
        status = quota_manager.get_status(client_id)
        estimated_tokens = (total_role_count + 1) * 150
        if status["enabled"] and status["remaining"] < estimated_tokens:
            minutes = (status["reset_in_seconds"] or 0) // 60
            return jsonify({
                "error": f"團隊模式需要呼叫模型 {total_role_count + 1} 次,預估至少需要 "
                         f"{estimated_tokens} token,目前剩餘額度只有 {status['remaining']},"
                         f"約 {minutes} 分鐘後額度會重置,建議減少選取的角色數量或稍後再試。"
            }), 429

        responses = []
        role_replies_for_integration = []
        total_tokens = 0
        for role in roles:
            role_prompt = ai_roles.build_role_prompt(role, prompt)
            reply_text, tokens_used = _generate_team_reply(role_prompt, config, model, tokenizer)
            total_tokens += tokens_used
            role_info = ai_roles.ROLES[role]
            responses.append({
                "role": role, "label": role_info["label"], "icon": role_info["icon"], "reply": reply_text,
            })
            role_replies_for_integration.append((role_info["label"], reply_text))

        for label in custom_role_labels:
            role_prompt = ai_roles.build_custom_role_prompt(label, prompt)
            reply_text, tokens_used = _generate_team_reply(role_prompt, config, model, tokenizer)
            total_tokens += tokens_used
            responses.append({
                "role": f"custom:{label}", "label": label, "icon": "custom", "reply": reply_text,
            })
            role_replies_for_integration.append((label, reply_text))

        integration_prompt = ai_roles.build_integration_prompt(prompt, role_replies_for_integration)
        integration_text, integration_tokens = _generate_team_reply(integration_prompt, config, model, tokenizer)
        total_tokens += integration_tokens

        quota_manager.consume(client_id, total_tokens)

        return jsonify({"type": "team", "responses": responses, "integration": integration_text})

    # 短的問候/道別/道謝類輸入,直接用規則比對回覆,不經過模型生成,
    # 因為目前模型規模太小,對這種短輸入常常分不清語境(見 smalltalk.py 說明)。
    # skip_rules 開關(NEXUX.html「略過規則」勾選框)讓使用者可以強制跳過
    # 這兩層保底機制,直接看模型自己會怎麼生成。
    if not skip_rules:
        smalltalk_match = match_smalltalk(prompt, history)
        if smalltalk_match is not None:
            smalltalk_reply, smalltalk_category = smalltalk_match
            return jsonify({"reply": smalltalk_reply, "type": smalltalk_category})

        # 訓練語料裡「本來就有標準答案」的問題(qa.jsonl / html.jsonl / python.jsonl),
        # 直接比對回傳原始答案,不用冒險讓模型生成——目前模型規模太小,連訓練資料裡
        # 出現過的問題都常常答錯,先確保這些「已知題目」一定答對(見 qa_lookup.py)。
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
        config, model, tokenizer, is_sft = get_model_and_tokenizer()
    except FileNotFoundError:
        return jsonify({
            "error": "還沒有本機微調好的模型。請先在終端機依序執行"
                     "「python convert_pretrained.py」「python run_pretrained_sft.py」,"
                     "再重新啟動 server.py。"
        }), 400

    # own 模型才做額度限制(見 quota_manager.py)。本機開發預設沒有設定
    # Upstash 環境變數,quota_manager 會自動回報沒有限制,行為上等於
    # 跟改動前一樣,不需要另外寫「本機停用」的特殊判斷。
    client_id = quota_manager.get_client_identifier(request, payload)
    if quota_manager.is_over_limit(client_id):
        status = quota_manager.get_status(client_id)
        minutes = (status["reset_in_seconds"] or 0) // 60
        return jsonify({"error": f"額度已用完,約 {minutes} 分鐘後自動恢復。"}), 429

    # 只有模型「真的經過 SFT 訓練」時,才包裝成問答格式,並帶入歷史對話當作 context;
    # 否則模型從沒見過這種格式,硬套上去只會讓生成效果更差。
    if is_sft:
        wrapped_prompt = build_context_prompt(
            history, prompt, tokenizer, config.block_size, config.max_new_tokens
        )
    else:
        wrapped_prompt = prompt

    idx = torch.tensor(
        [tokenizer.encode(wrapped_prompt)], dtype=torch.long, device=config.device
    )
    if idx.shape[1] == 0:
        return jsonify({"error": "輸入的文字包含詞表以外的字元,請換一句話試試。"}), 400

    if debug:
        print(
            f"[inference-debug] context tokens: {idx.shape[1]}/{config.block_size} "
            f"| history turns received: {len(history)} "
            f"| wrapped_prompt:\n{wrapped_prompt!r}"
        )

    def stream():
        """
        以 NDJSON(一行一個 JSON 物件)串流回傳生成結果,每一行格式是
        {"delta": "這次新增的文字", "n": 目前已生成的token數},最後固定
        以 {"done": true, "total_tokens": 總token數} 結尾,跟 api/generate.py
        (Vercel 正式站)用同一套協定,前端(NEXUX.html)可以共用一套解析邏輯。
        """
        accumulated = ""
        sent_len = 0
        # 尾巴保留幾個字元先不送出,避免剛好把「換行標記」(例如 \nA:)送出一半,
        # 等累積的文字夠長、確定不是標記的開頭之後,才把安全的部分吐給前端。
        HOLD = 3
        start_time = time.time()
        step = 0

        try:
            for token_ids in model.generate_stream(
                idx,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                eos_id=getattr(tokenizer, "eos_id", None),
            ):
                step += 1
                accumulated += tokenizer.decode(token_ids)

                if debug:
                    print(
                        f"[inference-debug] step {step:3d} | context_len {idx.shape[1] + step} "
                        f"| new token: {tokenizer.decode(token_ids)!r}"
                    )

                marker = find_next_turn_marker(accumulated)
                if marker:
                    final_text = accumulated[:marker.start()].rstrip()
                    if len(final_text) > sent_len:
                        yield json.dumps({"delta": final_text[sent_len:], "n": step}, ensure_ascii=False) + "\n"
                    if debug:
                        print(f"[inference-debug] stopped at turn marker, {step} tokens generated in {time.time()-start_time:.2f}s")
                    break
                else:
                    safe_len = max(0, len(accumulated) - HOLD)
                    if safe_len > sent_len:
                        yield json.dumps({"delta": accumulated[sent_len:safe_len], "n": step}, ensure_ascii=False) + "\n"
                        sent_len = safe_len
            else:
                final_text = accumulated.rstrip()
                if len(final_text) > sent_len:
                    yield json.dumps({"delta": final_text[sent_len:], "n": step}, ensure_ascii=False) + "\n"

                if debug:
                    print(f"[inference-debug] reached max_new_tokens, {step} tokens generated in {time.time()-start_time:.2f}s")

            quota_manager.consume(client_id, idx.shape[1] + step)
            yield json.dumps({"done": True, "total_tokens": step}, ensure_ascii=False) + "\n"

        except Exception as e:  # noqa: BLE001 - 這裡刻意攔截所有例外,回傳給前端顯示
            quota_manager.consume(client_id, idx.shape[1] + step)
            yield json.dumps({"delta": f"\n[生成時發生錯誤: {e}]", "n": step}, ensure_ascii=False) + "\n"
            yield json.dumps({"done": True, "total_tokens": step}, ensure_ascii=False) + "\n"

    return Response(stream(), mimetype="application/x-ndjson; charset=utf-8")


@app.route("/api/quota", methods=["GET"])
def api_quota():
    """給前端(composer 旁的額度顯示)輪詢用,見 quota_manager.py 的說明。"""
    client_id = quota_manager.get_client_identifier(request)
    return jsonify(quota_manager.get_status(client_id))


@app.route("/api/bead_pattern", methods=["POST"])
def api_bead_pattern():
    """
    本機開發用的拼豆對照圖端點,邏輯跟 api/bead_pattern.py 一致
    (共用同一個 bead_pattern.py),純圖片處理,不會用到語言模型。
    """
    import base64

    payload = request.get_json(silent=True) or {}
    image_b64 = payload.get("image") or ""
    grid_width = payload.get("gridWidth", DEFAULT_GRID)
    grid_height = payload.get("gridHeight")

    if not image_b64:
        return jsonify({"error": "沒有收到圖片,請重新上傳一次。"}), 400

    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception:  # noqa: BLE001
        return jsonify({"error": "圖片格式無法解析,請換一張圖片試試。"}), 400

    try:
        grid_width = int(grid_width)
    except (TypeError, ValueError):
        grid_width = DEFAULT_GRID
    grid_width = max(MIN_GRID, min(MAX_GRID, grid_width))

    if grid_height is not None:
        try:
            grid_height = int(grid_height)
        except (TypeError, ValueError):
            grid_height = None

    try:
        result = generate_pattern(image_bytes, grid_width=grid_width, grid_height=grid_height)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"圖片處理失敗,請換一張圖片試試。({e})"}), 500

    output_b64 = base64.b64encode(result.png_bytes).decode("ascii")
    color_table = [
        {"code": code, "name": name, "rgb": list(rgb), "count": count}
        for code, name, rgb, count in result.color_counts
    ]

    return jsonify({
        "image": f"data:image/png;base64,{output_b64}",
        "gridWidth": result.grid_width,
        "gridHeight": result.grid_height,
        "colors": color_table,
        "totalBeads": sum(c["count"] for c in color_table),
    })


if __name__ == "__main__":
    print("[server] 啟動中,請用瀏覽器開啟 http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
    
