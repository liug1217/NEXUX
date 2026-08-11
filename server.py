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
            "temperature": 0.5,
            "top_k": 20,
            "top_p": 0.8,
            "repetition_penalty": 2.0,
        })
        _cache["model"] = model
        _cache["tokenizer"] = tokenizer
        _cache["config"] = cfg
        _cache["is_sft"] = is_sft
        print(f"[server] 模型已載入並快取(SFT問答模式: {is_sft})")
    return _cache["config"], _cache["model"], _cache["tokenizer"], _cache["is_sft"]


def _generate_team_reply_stream(text_prompt, config, model, tokenizer):
    """
    NexoraAI 團隊模式用:generator版本,逐段yield {"delta": "..."} 讓呼叫端
    可以邊生成邊往前端送(團隊模式原本是等這個角色生成完才回傳一整段
    文字,使用者要等好幾次生成都跑完才第一次看到任何內容;改成這個
    generator 之後,單一角色的文字一樣能像一般問答模式那樣逐字出現)。
    最後固定yield一個 {"tokens": N} 收尾,呼叫端用有沒有 "tokens" 這個key
    分辨是不是最後一個chunk。停止邏輯(遇到 find_next_turn_marker 就截斷、
    HOLD尾巴避免把換行標記送出一半)跟 api_generate() 的 stream() 共用
    同一套規則。
    """
    wrapped_prompt = build_context_prompt(None, text_prompt, tokenizer, config.block_size, config.max_new_tokens)
    idx = torch.tensor([tokenizer.encode(wrapped_prompt)], dtype=torch.long, device=config.device)
    if idx.shape[1] == 0:
        yield {"tokens": 0}
        return

    accumulated = ""
    sent_len = 0
    HOLD = 3
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
            final_text = accumulated[:marker.start()].rstrip()
            if len(final_text) > sent_len:
                yield {"delta": final_text[sent_len:]}
            break
        safe_len = max(0, len(accumulated) - HOLD)
        if safe_len > sent_len:
            yield {"delta": accumulated[sent_len:safe_len]}
            sent_len = safe_len
    else:
        final_text = accumulated.rstrip()
        if len(final_text) > sent_len:
            yield {"delta": final_text[sent_len:]}

    yield {"tokens": idx.shape[1] + step}


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
    # 的觀點」,直接查表回同一句罐頭答案沒有意義)。跟一般單一問答模式一樣改用
    # NDJSON 串流:每個角色的文字一生成出來就送給前端,不用等所有角色+整合都
    # 生成完才一次看到內容(之前的版本是等全部跑完才一次回傳,使用者要空等
    # 好幾次生成的時間才第一次看到任何文字)。
    # roleRequests:composer「AI人員」選單裡目前選取的角色,每個都是
    # {id, label, icon, promptTemplate} ——不管是預設角色(可能已經被
    # 使用者右鍵編輯過)還是使用者自訂角色,前端一律送出組好的完整
    # promptTemplate,後端不需要再知道這個角色原本的出處(見 ai_roles.py
    # 開頭的說明)。sanitize_role_requests() 是真正生效的數量/長度驗證,
    # 不能只信前端已經做過的限制。
    role_requests = ai_roles.sanitize_role_requests(payload.get("roleRequests"))
    if role_requests:
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

        # 團隊模式會呼叫模型 len(role_requests)+1 次(每個角色一次+核心AI
        # 整合一次),用粗略估計值在生成前先擋下明顯不足的額度,避免生成
        # 到一半才因為額度用完而中斷、浪費前面已經生成的角色回覆。這個
        # 檢查必須在開始串流之前做完——一旦 Response(stream(), ...) 開始
        # 送出第一個位元組,狀態碼就已經定案是200,沒辦法半路改成429錯誤。
        status = quota_manager.get_status(client_id)
        estimated_tokens = (len(role_requests) + 1) * 150
        if status["enabled"] and status["remaining"] < estimated_tokens:
            minutes = (status["reset_in_seconds"] or 0) // 60
            return jsonify({
                "error": f"團隊模式需要呼叫模型 {len(role_requests) + 1} 次,預估至少需要 "
                         f"{estimated_tokens} token,目前剩餘額度只有 {status['remaining']},"
                         f"約 {minutes} 分鐘後額度會重置,建議減少選取的角色數量或稍後再試。"
            }), 429

        role_entries = [
            (r["id"], r["label"], r["icon"], ai_roles.build_prompt_from_template(r["promptTemplate"], prompt))
            for r in role_requests
        ]

        def team_stream():
            """
            NDJSON,每行一個JSON物件:
            {"role":id,"label":..,"icon":..,"role_start":true}  這個角色開始生成
            {"role":id,"delta":"..."}                            這個角色的部分文字
            {"role":id,"role_done":true,"reply":"完整文字"}       這個角色生成完了
            (核心AI整合是最後一個角色,role固定是"integration")
            {"done":true,"total_tokens":N}                        全部結束
            """
            total_tokens = 0
            role_replies_for_integration = []

            try:
                for role_id, label, icon, role_prompt in role_entries:
                    yield json.dumps({"role": role_id, "label": label, "icon": icon, "role_start": True}, ensure_ascii=False) + "\n"
                    full_text = ""
                    for chunk in _generate_team_reply_stream(role_prompt, config, model, tokenizer):
                        if "delta" in chunk:
                            full_text += chunk["delta"]
                            yield json.dumps({"role": role_id, "delta": chunk["delta"]}, ensure_ascii=False) + "\n"
                        else:
                            total_tokens += chunk["tokens"]
                    full_text = full_text.rstrip()
                    yield json.dumps({"role": role_id, "role_done": True, "reply": full_text}, ensure_ascii=False) + "\n"
                    role_replies_for_integration.append((label, full_text))

                integration_prompt = ai_roles.build_integration_prompt(prompt, role_replies_for_integration)
                yield json.dumps({"role": "integration", "label": "核心AI整合建議", "icon": "integration", "role_start": True}, ensure_ascii=False) + "\n"
                integration_text = ""
                for chunk in _generate_team_reply_stream(integration_prompt, config, model, tokenizer):
                    if "delta" in chunk:
                        integration_text += chunk["delta"]
                        yield json.dumps({"role": "integration", "delta": chunk["delta"]}, ensure_ascii=False) + "\n"
                    else:
                        total_tokens += chunk["tokens"]
                integration_text = integration_text.rstrip()
                yield json.dumps({"role": "integration", "role_done": True, "reply": integration_text}, ensure_ascii=False) + "\n"

                quota_manager.consume(client_id, total_tokens)
                yield json.dumps({"done": True, "total_tokens": total_tokens}, ensure_ascii=False) + "\n"
            except Exception as e:  # noqa: BLE001 - 攔截所有例外,回傳給前端顯示
                # 不管是在哪個角色生成時出錯,都補一個獨立的"error"角色泡泡
                # (先送role_start再送delta),前端才一定看得到這個錯誤訊息
                # ——如果直接對著還沒送過role_start的角色id送delta,前端會
                # 因為找不到對應的泡泡元素而把這則錯誤靜默丟掉。
                quota_manager.consume(client_id, total_tokens)
                yield json.dumps({"role": "error", "label": "錯誤", "icon": "custom", "role_start": True}, ensure_ascii=False) + "\n"
                yield json.dumps({"role": "error", "delta": f"生成時發生錯誤: {e}"}, ensure_ascii=False) + "\n"
                yield json.dumps({"role": "error", "role_done": True, "reply": f"生成時發生錯誤: {e}"}, ensure_ascii=False) + "\n"
                yield json.dumps({"done": True, "total_tokens": total_tokens}, ensure_ascii=False) + "\n"

        return Response(team_stream(), mimetype="application/x-ndjson; charset=utf-8")

    # [已停用] smalltalk 規則攔截——模型已經訓練過寒暄語料,交給模型生成。
    # smalltalk_match = match_smalltalk(prompt, history)
    # if smalltalk_match is not None:
    #     smalltalk_reply, smalltalk_category = smalltalk_match
    #     return jsonify({"reply": smalltalk_reply, "type": smalltalk_category})

    # [已停用] qa_lookup 規則攔截——模型已經訓練過這些語料,交給模型生成。
    # qa_reply = match_qa(prompt, data_dir=os.path.join(BASE_DIR, "data"))
    # if qa_reply is not None:
    #     return jsonify({"reply": qa_reply, "type": "qa_lookup"})

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


@app.route("/api/roles", methods=["GET"])
def api_roles():
    """
    給 composer 角色選單(AI人員)頁面載入時取得預設角色清單用,見
    ai_roles.py 開頭的說明。前端會把這份清單當作「預設角色」的初始值/
    右鍵「恢復預設」的還原目標,實際編輯/自訂角色都存在瀏覽器
    localStorage,這個端點只提供唯一真相來源(ai_roles.ROLES),
    不會受使用者的編輯影響。
    """
    return jsonify({"roles": ai_roles.get_default_roles()})


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
    
