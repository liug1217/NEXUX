"""
bead_pattern.py
----------------
把使用者上傳的圖片,轉換成拼豆(perler beads)對照圖:先把圖片縮成
指定格數的網格,每一格取代表色後,量化成調色盤裡最接近的拼豆顏色,
輸出一張「每一格畫成一顆圓珠、標上色號」的對照圖,以及一份色號用量表。

注意:這裡用的是一組自訂的通用色盤(約 30 色,涵蓋常見的紅橙黃綠藍紫
及黑白灰階),色號是自訂編號(B01、B02...),不是對應任何特定廠牌
(例如 MARD、Hama、Perler)的官方色號,如果要精準對照特定廠牌的色卡,
之後需要換成該廠牌實際的色號與色票 RGB 值。
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

# ---- 拼豆色盤(自訂通用色號,不對應特定廠牌) ----
# 每一色: (色號, 顯示名稱, RGB)
BEAD_PALETTE: list[tuple[str, str, tuple[int, int, int]]] = [
    ("B01", "純白", (255, 255, 255)),
    ("B02", "淺灰", (200, 200, 200)),
    ("B03", "中灰", (140, 140, 140)),
    ("B04", "純黑", (20, 20, 20)),
    ("B05", "正紅", (220, 30, 40)),
    ("B06", "深紅", (150, 20, 30)),
    ("B07", "粉紅", (245, 150, 180)),
    ("B08", "橘色", (240, 120, 30)),
    ("B09", "淺橘", (250, 180, 100)),
    ("B10", "正黃", (250, 220, 40)),
    ("B11", "淺黃", (250, 240, 150)),
    ("B12", "土黃", (200, 160, 60)),
    ("B13", "草綠", (100, 180, 60)),
    ("B14", "深綠", (30, 110, 60)),
    ("B15", "淺綠", (170, 220, 140)),
    ("B16", "青綠", (40, 160, 140)),
    ("B17", "天藍", (90, 170, 230)),
    ("B18", "正藍", (30, 90, 200)),
    ("B19", "深藍", (20, 40, 120)),
    ("B20", "淺藍", (180, 220, 245)),
    ("B21", "紫色", (140, 70, 190)),
    ("B22", "淺紫", (200, 170, 230)),
    ("B23", "咖啡", (120, 80, 50)),
    ("B24", "淺咖啡", (180, 140, 100)),
    ("B25", "膚色", (240, 200, 160)),
    ("B26", "米色", (230, 215, 190)),
    ("B27", "桃紅", (230, 60, 130)),
    ("B28", "銀灰藍", (150, 170, 190)),
    ("B29", "橄欖綠", (130, 140, 60)),
    ("B30", "亮黃綠", (190, 220, 60)),
]

DEFAULT_GRID = 29  # 最常見的單片拼豆板尺寸(29x29 pegs)
MAX_GRID = 100      # 避免使用者亂輸入超大格數,拖垮運算或圖片大小
MIN_GRID = 5

CELL_PX = 24         # 輸出圖裡每一顆珠子畫多少像素大小
PADDING_PX = 12


def _nearest_palette_color(rgb: tuple[int, int, int]) -> tuple[str, str, tuple[int, int, int]]:
    best = None
    best_dist = None
    for code, name, palette_rgb in BEAD_PALETTE:
        dist = sum((a - b) ** 2 for a, b in zip(rgb, palette_rgb))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = (code, name, palette_rgb)
    return best


@dataclass
class BeadPatternResult:
    png_bytes: bytes
    grid_width: int
    grid_height: int
    color_counts: list[tuple[str, str, tuple[int, int, int], int]]  # code, name, rgb, count


def generate_pattern(
    image_bytes: bytes,
    grid_width: int = DEFAULT_GRID,
    grid_height: int | None = None,
) -> BeadPatternResult:
    """
    把輸入圖片轉成拼豆對照圖。grid_width/grid_height 是拼豆板的格數
    (寬 x 高),沒有指定高度時,依照原圖比例自動算出高度。
    """
    grid_width = max(MIN_GRID, min(MAX_GRID, int(grid_width)))

    src = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if grid_height is None:
        grid_height = max(MIN_GRID, round(grid_width * src.height / src.width))
    grid_height = max(MIN_GRID, min(MAX_GRID, int(grid_height)))

    # 先縮小成「一格一像素」的小圖,用平均取樣讓每一格代表色更準確,
    # 不是單純挑左上角那個像素。
    small = src.resize((grid_width, grid_height), Image.LANCZOS)

    grid_colors: list[list[tuple[str, str, tuple[int, int, int]]]] = []
    counts: dict[str, list] = {}  # code -> [name, rgb, count]

    for y in range(grid_height):
        row = []
        for x in range(grid_width):
            rgb = small.getpixel((x, y))
            code, name, palette_rgb = _nearest_palette_color(rgb)
            row.append((code, name, palette_rgb))
            if code not in counts:
                counts[code] = [name, palette_rgb, 0]
            counts[code][2] += 1
        grid_colors.append(row)

    # 畫輸出圖:每一格畫一顆圓珠(色塊 + 深色描邊 + 小字色號),
    # 方便使用者一格一格照著拼。
    out_w = grid_width * CELL_PX + PADDING_PX * 2
    out_h = grid_height * CELL_PX + PADDING_PX * 2
    out_img = Image.new("RGB", (out_w, out_h), (250, 250, 250))
    draw = ImageDraw.Draw(out_img)

    try:
        font = ImageFont.load_default()
    except Exception:  # noqa: BLE001
        font = None

    for y in range(grid_height):
        for x in range(grid_width):
            code, name, rgb = grid_colors[y][x]
            x0 = PADDING_PX + x * CELL_PX
            y0 = PADDING_PX + y * CELL_PX
            x1 = x0 + CELL_PX
            y1 = y0 + CELL_PX
            draw.ellipse(
                [x0 + 2, y0 + 2, x1 - 2, y1 - 2],
                fill=rgb,
                outline=(90, 90, 90),
                width=1,
            )
            if font is not None:
                # 亮色底用深字、暗色底用亮字,確保色號看得清楚。
                brightness = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                text_color = (30, 30, 30) if brightness > 140 else (245, 245, 245)
                short_label = code[-2:]  # 只顯示編號後兩碼(例如 B01 -> 01),格子太小放不下全碼
                draw.text(
                    (x0 + CELL_PX / 2, y0 + CELL_PX / 2),
                    short_label,
                    fill=text_color,
                    font=font,
                    anchor="mm",
                )

    buf = io.BytesIO()
    out_img.save(buf, format="PNG")

    color_counts = sorted(
        ((code, info[0], info[1], info[2]) for code, info in counts.items()),
        key=lambda item: -item[3],
    )

    return BeadPatternResult(
        png_bytes=buf.getvalue(),
        grid_width=grid_width,
        grid_height=grid_height,
        color_counts=color_counts,
    )
