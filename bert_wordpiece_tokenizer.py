"""
bert_wordpiece_tokenizer.py
-----------------------------
給預訓練模型(ckiplab/gpt2-base-chinese)用的 tokenizer,純 Python 實作
BERT 的 WordPiece 演算法,不依賴 transformers/tokenizers 套件(那兩個對
Vercel Serverless Function 來說太大了,跟原本捨棄 torch 改用 numpy_gpt.py
的理由一樣)。

跟 tokenizer.py 的 CharTokenizer 保持相同介面(vocab_size / encode / decode /
save / load),train_sft.py、numpy 推理端(之後會接上)都可以直接把這個當
CharTokenizer 的替代品使用。

對中文來說,這個 tokenizer 幾乎等於逐字元切分(每個中文字都會被空白隔開,
變成獨立的 token),只有英文單字、數字這類非中文內容才會真的用到
WordPiece 的「最長前綴比對 + ## 延續片段」演算法。
"""

from __future__ import annotations

import json
import os
import unicodedata


def _is_cjk_char(ch: str) -> bool:
    """判斷是不是中日韓表意文字(CJK Unified Ideographs 及常見擴充區段)。"""
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2A6DF
        or 0xF900 <= cp <= 0xFAFF
        or 0x2F800 <= cp <= 0x2FA1F
    )


def _is_punctuation(ch: str) -> bool:
    cp = ord(ch)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    return unicodedata.category(ch).startswith("P")


def _is_control(ch: str) -> bool:
    if ch in ("\t", "\n", "\r"):
        return False
    return unicodedata.category(ch).startswith("C")


def _is_whitespace(ch: str) -> bool:
    if ch in (" ", "\t", "\n", "\r"):
        return True
    return unicodedata.category(ch) == "Zs"


def _basic_tokenize(text: str) -> list[str]:
    """
    對應 BERT 的 BasicTokenizer:清掉控制字元、統一空白、每個 CJK 字元
    左右加空白讓它自成一個 token、標點符號也拆成獨立 token,最後用空白
    切分、轉小寫(bert-base-chinese 預設 do_lower_case=True)。
    """
    cleaned = []
    for ch in text:
        if ord(ch) == 0 or ord(ch) == 0xFFFD or _is_control(ch):
            continue
        cleaned.append(" " if _is_whitespace(ch) else ch)
    text = "".join(cleaned)

    spaced = []
    for ch in text:
        if _is_cjk_char(ch):
            spaced.append(" ")
            spaced.append(ch)
            spaced.append(" ")
        else:
            spaced.append(ch)
    text = "".join(spaced)

    tokens = text.strip().split()
    output = []
    for token in tokens:
        token = token.lower()
        token = "".join(c for c in unicodedata.normalize("NFD", token) if unicodedata.category(c) != "Mn")

        # 標點符號各自獨立成一個 token
        chars = list(token)
        sub_tokens = []
        current = []
        for ch in chars:
            if _is_punctuation(ch):
                if current:
                    sub_tokens.append("".join(current))
                    current = []
                sub_tokens.append(ch)
            else:
                current.append(ch)
        if current:
            sub_tokens.append("".join(current))
        output.extend(sub_tokens)

    return [t for t in output if t]


class BertWordpieceTokenizer:
    def __init__(self, vocab: list[str] | None = None, unk_token: str = "[UNK]"):
        self.vocab = vocab or []
        self.stoi = {tok: i for i, tok in enumerate(self.vocab)}
        self.itos = {i: tok for i, tok in enumerate(self.vocab)}
        self.unk_token = unk_token
        self.unk_id = self.stoi.get(unk_token, 0)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def eos_id(self) -> int | None:
        """
        [SEP] 在 BERT 系列詞表裡本來是「句子分隔」符號,這裡借用當作
        回答的結束符號:SFT 訓練時接在答案後面,推論時生成到這個 id
        就代表模型判斷自己回答完了,可以停止生成。
        """
        return self.stoi.get("[SEP]")

    def _wordpiece_tokenize(self, word: str, max_chars: int = 100) -> list[str]:
        """對單一個(已經過 basic tokenize 的)word,做最長前綴比對 + ## 延續片段。"""
        if len(word) > max_chars:
            return [self.unk_token]

        sub_tokens = []
        start = 0
        while start < len(word):
            end = len(word)
            found = None
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = "##" + substr
                if substr in self.stoi:
                    found = substr
                    break
                end -= 1
            if found is None:
                return [self.unk_token]
            sub_tokens.append(found)
            start = end
        return sub_tokens

    def tokenize(self, text: str) -> list[str]:
        tokens = []
        for word in _basic_tokenize(text):
            tokens.extend(self._wordpiece_tokenize(word))
        return tokens

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(tok, self.unk_id) for tok in self.tokenize(text)]

    def decode(self, ids: list[int]) -> str:
        """
        id 序列 -> 文字。中文字元(basic tokenize 時被空白隔開)重新黏回去,
        英文 ## 延續片段直接接在前一個詞後面,其餘用空白分隔,
        是簡化版還原,不保證跟原文完全一致(例如原本的大小寫、標點間距),
        但對聊天機器人的輸出來說已經足夠自然。
        """
        pieces = [self.itos.get(i, self.unk_token) for i in ids]
        out = []
        for piece in pieces:
            if piece in ("[CLS]", "[SEP]", "[PAD]", "[MASK]"):
                continue
            if piece.startswith("##"):
                out.append(piece[2:])
            elif out and (_is_cjk_char(piece[0]) if piece else False):
                out.append(piece)
            elif out and _is_cjk_char(out[-1][-1] if out[-1] else " "):
                out.append(piece)
            else:
                if out:
                    out.append(" " + piece)
                else:
                    out.append(piece)
        return "".join(out)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "BertWordpieceTokenizer":
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到 tokenizer 檔案: {path}")
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab=vocab)

    @classmethod
    def load_from_vocab_txt(cls, path: str) -> "BertWordpieceTokenizer":
        """從 convert_pretrained.py 匯出的逐行 vocab.txt 建立 tokenizer。"""
        with open(path, "r", encoding="utf-8") as f:
            vocab = [line.rstrip("\n") for line in f]
        return cls(vocab=vocab)


if __name__ == "__main__":
    tok = BertWordpieceTokenizer.load_from_vocab_txt("vocab_pretrained.txt")
    samples = ["為什麼天空是藍色的?", "幫我寫一個 Python 範例 hello world", "哈囉,你好嗎?"]
    for s in samples:
        ids = tok.encode(s)
        decoded = tok.decode(ids)
        print(f"原文: {s!r}")
        print(f"tokens: {tok.tokenize(s)}")
        print(f"ids: {ids}")
        print(f"解碼: {decoded!r}")
        print()
