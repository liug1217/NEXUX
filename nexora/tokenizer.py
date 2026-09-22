"""
nexora/tokenizer.py
-------------------
Nexora Native 自有 character-level tokenizer。

設計原則：
  - vocabulary 完全從我們自己的語料掃出，不引用任何外部詞表。
  - 固定 4 個 special tokens（PAD / UNK / BOS / EOS），ID 0-3 保留。
  - vocab 最多 4096 個 token（special tokens + 最高頻的字元）。
  - 存檔格式加入 version 欄位，避免未來詞表重建後 checkpoint 對應錯誤。
  - encode / decode / save / load 介面跟現有 CharTokenizer 相容（方便對比）。
"""

import json
import os
import glob
import collections


class NexoraTokenizer:
    VERSION = "nexora-char-v1"

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3

    def __init__(self, vocab: list[str]):
        """
        vocab: 完整詞表，index 即 token ID。
        vocab[0:4] 必須是 SPECIAL_TOKENS。
        """
        assert vocab[:4] == self.SPECIAL_TOKENS, (
            f"vocab 前 4 個必須是 {self.SPECIAL_TOKENS}，實際是 {vocab[:4]}"
        )
        self.vocab = vocab
        self.stoi = {t: i for i, t in enumerate(vocab)}

    # ── 基本屬性 ────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def eos_id(self) -> int:
        return self.EOS_ID

    @property
    def bos_id(self) -> int:
        return self.BOS_ID

    # ── encode / decode ─────────────────────────────────────────────────

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """
        文字 → token ID 列表。
        詞表外的字元映射到 UNK_ID（不靜默丟棄，保留位置資訊）。
        """
        ids: list[int] = []
        if add_bos:
            ids.append(self.BOS_ID)
        for ch in text:
            ids.append(self.stoi.get(ch, self.UNK_ID))
        if add_eos:
            ids.append(self.EOS_ID)
        return ids

    def decode(
        self,
        ids: list[int],
        skip_special: bool = True,
    ) -> str:
        """
        token ID 列表 → 文字。
        skip_special=True 時略過 PAD / UNK / BOS / EOS。
        """
        parts: list[str] = []
        special_set = set(self.SPECIAL_TOKENS)
        for i in ids:
            if 0 <= i < len(self.vocab):
                t = self.vocab[i]
                if skip_special and t in special_set:
                    continue
                parts.append(t)
        return "".join(parts)

    # ── 建立詞表 ────────────────────────────────────────────────────────

    @classmethod
    def build_from_data(
        cls,
        data_dir: str,
        max_vocab: int = 4096,
    ) -> "NexoraTokenizer":
        """
        掃描 data_dir 底下所有 .jsonl 語料，統計字元頻率，
        選出最高頻的 (max_vocab - 4) 個字元加上 4 個 special tokens，
        建立詞表。
        """
        counter: collections.Counter = collections.Counter()
        files = glob.glob(os.path.join(data_dir, "*.jsonl"))
        if not files:
            raise FileNotFoundError(f"在 {data_dir} 找不到任何 .jsonl 語料檔")

        total_lines = 0
        for path in sorted(files):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        import json as _json
                        obj = _json.loads(line)
                        for msg in obj.get("messages", []):
                            counter.update(msg.get("content", ""))
                        total_lines += 1
                    except Exception:
                        pass

        print(f"[NexoraTokenizer] 掃描 {len(files)} 個檔案，{total_lines} 筆對話")
        print(f"[NexoraTokenizer] 唯一字元數: {len(counter)}")

        # 統計語言分佈
        cjk = sum(v for ch, v in counter.items() if "一" <= ch <= "鿿")
        ascii_cnt = sum(v for ch, v in counter.items() if ord(ch) < 128)
        total_chars = sum(counter.values())
        print(f"[NexoraTokenizer] 字元統計："
              f" CJK {cjk/total_chars:.1%},"
              f" ASCII {ascii_cnt/total_chars:.1%},"
              f" 其他 {(total_chars-cjk-ascii_cnt)/total_chars:.1%}")

        # 選最高頻字元（保留 4 個 special token 槽位）
        n_char_slots = max_vocab - len(cls.SPECIAL_TOKENS)
        selected = [ch for ch, _ in counter.most_common(n_char_slots)]

        vocab = cls.SPECIAL_TOKENS + selected
        print(f"[NexoraTokenizer] 詞表大小: {len(vocab)}"
              f"（special={len(cls.SPECIAL_TOKENS)}, chars={len(selected)}）")
        return cls(vocab=vocab)

    # ── 儲存 / 載入 ─────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "version": self.VERSION,
            "vocab_size": self.vocab_size,
            "special_tokens": self.SPECIAL_TOKENS,
            "vocab": self.vocab,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[NexoraTokenizer] 已儲存詞表至 {path}（{self.vocab_size} tokens）")

    @classmethod
    def load(cls, path: str) -> "NexoraTokenizer":
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到 tokenizer 檔案: {path}")
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        version = payload.get("version", "unknown")
        if version != cls.VERSION:
            raise ValueError(
                f"tokenizer 版本不相符（檔案: {version!r}，預期: {cls.VERSION!r}）。"
                "版本不同代表詞表格式可能已更改，請重新建立 tokenizer。"
            )
        return cls(vocab=payload["vocab"])

    # ── 統計工具 ────────────────────────────────────────────────────────

    def coverage(self, text: str) -> float:
        """回傳文字中有多少比例的字元在詞表內（UNK 率的反面）。"""
        if not text:
            return 1.0
        known = sum(1 for ch in text if ch in self.stoi and self.stoi[ch] >= 4)
        return known / len(text)


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    tok = NexoraTokenizer.build_from_data(data_dir, max_vocab=4096)

    # 存檔再重新載入，確認一致
    test_path = "nexora/nexora_vocab_test.json"
    tok.save(test_path)
    tok2 = NexoraTokenizer.load(test_path)

    sample = "你好，台灣！Hello, world! 123"
    ids1 = tok.encode(sample, add_bos=True, add_eos=True)
    ids2 = tok2.encode(sample, add_bos=True, add_eos=True)
    assert ids1 == ids2, "存載後 encode 結果不一致"
    decoded = tok2.decode(ids2)
    assert decoded == sample, f"decode 不一致: {decoded!r}"
    print(f"[Test] encode: {ids1[:10]}...")
    print(f"[Test] decode: {decoded!r}")
    print("[Test] tokenizer 儲存/載入/encode/decode 全部通過 ✅")
    os.remove(test_path)
