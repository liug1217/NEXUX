"""
api/bead_pattern.py
--------------------
獨立的 Vercel Serverless Function,處理拼豆對照圖的產生請求。
跟 api/generate.py 分開成獨立檔案,是因為 Vercel 的 Python 零設定
部署是「一個檔案對應一個 API 路徑」(api/bead_pattern.py -> /api/bead_pattern),
不是靠同一個 Flask app 裡多個 @app.route 就能對外開放多個路徑。

這條路徑完全不會用到語言模型(不會載入 weights.json / tokenizer.json),
純粹是圖片處理,所以獨立開一支輕量的 function,不會拖慢也不會被
模型推論那邊的邏輯影響。
"""

import base64
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, request, jsonify  # noqa: E402
from bead_pattern import generate_pattern, DEFAULT_GRID, MIN_GRID, MAX_GRID  # noqa: E402

app = Flask(__name__)

MAX_IMAGE_BYTES = 8 * 1024 * 1024  # 8MB,避免使用者傳超大圖片拖垮 serverless function


@app.route("/api/bead_pattern", methods=["POST"])
def api_bead_pattern():
    payload = request.get_json(silent=True) or {}
    image_b64 = payload.get("image") or ""
    grid_width = payload.get("gridWidth", DEFAULT_GRID)
    grid_height = payload.get("gridHeight")

    if not image_b64:
        return jsonify({"error": "沒有收到圖片,請重新上傳一次。"}), 400

    # 前端傳來的通常是 data URL(data:image/png;base64,xxxx),把前綴去掉。
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception:  # noqa: BLE001
        return jsonify({"error": "圖片格式無法解析,請換一張圖片試試。"}), 400

    if len(image_bytes) > MAX_IMAGE_BYTES:
        return jsonify({"error": "圖片檔案太大,請壓縮或換一張較小的圖片(上限 8MB)。"}), 400

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
