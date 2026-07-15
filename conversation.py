"""
conversation.py
----------------
把「多輪對話歷史」組成模型看得懂的單一段 prompt 文字。

背景:
之前每次請求只會把使用者「這一句話」丟給模型,模型完全不知道
前面聊過什麼,所以沒辦法做到「我叫小明」->「我叫什麼?」這種
需要記住上文的對話。

做法:
沿用 qa.txt / chat.txt 訓練時用過的「問:...\n答:...」格式,把最近幾輪
對話接在新問題前面,一起送進模型當作 context。因為模型的 block_size
(一次能看的字數上限)有限,所以要先幫最新的這一輪保留位置,再從
最近的歷史往回加,加到快超過長度上限就停止,越舊的對話會被優先捨棄。

不需要 torch,server.py(本機 Flask)和 api/generate.py(Vercel numpy 版本)
都可以直接共用這份邏輯。
"""

from typing import Optional


def build_context_prompt(
    history: Optional[list[dict]],
    prompt: str,
    tokenizer,
    block_size: int,
    max_new_tokens: int,
) -> str:
    """
    history: [{"role": "user" | "assistant", "text": "..."}, ...],由舊到新排序,
             不包含這次的新輸入。
    prompt:  這次使用者輸入的新訊息。
    回傳:    組好的「問:...\n答:...\n...\n問:{prompt}\n答:」字串,
             總長度會控制在 block_size - max_new_tokens 個 token 以內,
             確保模型還有空間可以生成回覆。
    """
    budget = max(block_size - max_new_tokens, 8)

    tail = f"問:{prompt}\n答:"
    used = len(tokenizer.encode(tail))

    kept_pieces = []
    for turn in reversed(history or []):
        text = (turn.get("text") or "").strip()
        if not text:
            continue
        tag = "問:" if turn.get("role") == "user" else "答:"
        piece = f"{tag}{text}\n"
        piece_len = len(tokenizer.encode(piece))
        if used + piece_len > budget:
            break
        kept_pieces.append(piece)
        used += piece_len

    kept_pieces.reverse()
    return "".join(kept_pieces) + tail
