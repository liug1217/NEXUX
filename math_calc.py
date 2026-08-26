"""
math_calc.py
------------
偵測數學算式，用 Python 安全計算後回傳結果。
不走模型，直接回傳正確答案。
"""

import re
import ast
import operator


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)

        if left is None or right is None:
            return None

        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            return None

        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            return None

        try:
            return _OPS[type(node.op)](left, right)
        except (OverflowError, ZeroDivisionError):
            return None

    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        value = _safe_eval(node.operand)
        if value is None:
            return None
        return _OPS[type(node.op)](value)

    return None


def match_math(text: str):
    text = text.strip()

    if not text:
        return None

    # 中文數學轉成 Python 算式
    expr = text
    expr = expr.replace("×", "*")
    expr = expr.replace("÷", "/")
    expr = expr.replace("＝", "=")
    expr = expr.replace("^", "**")

    replacements = [
        (r"乘以", "*"),
        (r"除以", "/"),
        (r"加上", "+"),
        (r"減去", "-"),
        (r"加", "+"),
        (r"減", "-"),
        (r"乘", "*"),
        (r"除", "/"),
        (r"等於多少", ""),
        (r"等於什麼", ""),
        (r"等於幾", ""),
        (r"是多少", ""),
        (r"是什麼", ""),
        (r"是幾", ""),
        (r"多少", ""),
        (r"幾", ""),
        (r"？", ""),
        (r"\?", ""),
    ]

    for pattern, replacement in replacements:
        expr = re.sub(pattern, replacement, expr)

    # 移除空白與等號
    expr = re.sub(r"[=\s]", "", expr)

    # 只允許數字與安全運算符
    if not re.fullmatch(r"[\d.+\-*/()%]+", expr):
        return None

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    result = _safe_eval(tree)

    if result is None:
        return None

    # 例如 5.0 → 5
    if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
        result = int(result)

    formatted = f"{result:.10g}" if isinstance(result, float) else str(result)

    return f"{text.rstrip('？? ')} = {formatted}"
