"""
rag_engine.py — TF-IDF 極簡 RAG 檢索引擎
==========================================
零外部依賴（純 Python + json + math），不需要 numpy / torch / faiss。

原理：
1. 讀取 data/*.jsonl 所有語料，每筆 Q&A 存成一份「文件」
2. 用字元雙字組（character bigram）做 TF-IDF 特徵
3. 查詢時計算餘弦相似度，取 top-k 最相關的 Q&A
4. 格式化成「問:/答:」格式注入 prompt，讓模型看到相關範例再生成

用法：
    # 建立索引（幾秒鐘）
    python rag_engine.py --build

    # 測試查詢
    python rag_engine.py --query "地震怎麼辦"

    # 在 server.py / app.py 中使用
    from rag_engine import RAGEngine
    rag = RAGEngine.load()
    context = rag.retrieve("地震怎麼辦", top_k=2)
"""

import argparse
import glob
import json
import math
import os
from collections import Counter


def _bigrams(text: str) -> list[str]:
    chars = [c for c in text if not c.isspace()]
    if len(chars) < 2:
        return list(chars)
    return [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]


class RAGEngine:

    def __init__(self, docs, idf, doc_tfidf_vecs):
        self.docs = docs
        self.idf = idf
        self.doc_vecs = doc_tfidf_vecs

    @classmethod
    def build_from_data(cls, data_dir: str = "data", save_path: str = "rag_index.json"):
        docs = []
        jsonl_files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))

        for path in jsonl_files:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    messages = obj.get("messages", [])
                    pairs = []
                    for i in range(0, len(messages) - 1, 2):
                        if messages[i].get("role") == "user" and messages[i + 1].get("role") == "assistant":
                            q = (messages[i].get("content") or "").strip()
                            a = (messages[i + 1].get("content") or "").strip()
                            if q and a:
                                pairs.append({"q": q, "a": a})
                    docs.extend(pairs)

        print(f"[RAG] 從 {len(jsonl_files)} 個檔案讀取 {len(docs)} 筆 Q&A")

        n = len(docs)
        df = Counter()
        q_bigrams = []
        for doc in docs:
            bg = _bigrams(doc["q"])
            tf = Counter(bg)
            q_bigrams.append(tf)
            for term in tf:
                df[term] += 1

        idf = {term: math.log(n / (1 + freq)) for term, freq in df.items()}

        doc_vecs = []
        for tf in q_bigrams:
            total = sum(tf.values()) or 1
            vec = {term: (count / total) * idf.get(term, 0) for term, count in tf.items()}
            doc_vecs.append(vec)

        engine = cls(docs, idf, doc_vecs)

        index_data = {"docs": docs, "idf": idf}
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False)
        size_kb = os.path.getsize(save_path) / 1024
        print(f"[RAG] 索引已存檔: {save_path} ({size_kb:.0f} KB, {len(docs)} 筆)")
        return engine

    @classmethod
    def load(cls, index_path: str = "rag_index.json"):
        if not os.path.exists(index_path):
            return None
        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        docs = data["docs"]
        idf = data["idf"]

        doc_vecs = []
        for doc in docs:
            bg = _bigrams(doc["q"])
            tf = Counter(bg)
            total = sum(tf.values()) or 1
            vec = {term: (count / total) * idf.get(term, 0) for term, count in tf.items()}
            doc_vecs.append(vec)

        return cls(docs, idf, doc_vecs)

    def _query_vec(self, query: str) -> dict:
        bg = _bigrams(query)
        tf = Counter(bg)
        total = sum(tf.values()) or 1
        return {term: (count / total) * self.idf.get(term, 0) for term, count in tf.items()}

    @staticmethod
    def _cosine(a: dict, b: dict) -> float:
        common = set(a) & set(b)
        if not common:
            return 0.0
        dot = sum(a[t] * b[t] for t in common)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def retrieve(self, query: str, top_k: int = 2, threshold: float = 0.05) -> list[dict]:
        qvec = self._query_vec(query)
        scored = []
        for i, dvec in enumerate(self.doc_vecs):
            score = self._cosine(qvec, dvec)
            if score >= threshold:
                scored.append((score, i))
        scored.sort(key=lambda x: -x[0])
        return [{"q": self.docs[i]["q"], "a": self.docs[i]["a"], "score": s}
                for s, i in scored[:top_k]]

    def retrieve_context(self, query: str, top_k: int = 2) -> str:
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return ""
        lines = []
        for r in results:
            lines.append(f"問:{r['q']}\n答:{r['a']}")
        return "\n".join(lines) + "\n"

    def direct_answer(self, query: str, threshold: float = 0.25) -> str | None:
        """高分命中時直接回傳語料庫答案，不經過模型生成。"""
        results = self.retrieve(query, top_k=1, threshold=threshold)
        if not results:
            return None
        return results[0]["a"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TF-IDF RAG 索引管理")
    parser.add_argument("--build", action="store_true", help="從 data/ 建立索引")
    parser.add_argument("--query", type=str, help="測試查詢")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--index", default="rag_index.json")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    if args.build:
        engine = RAGEngine.build_from_data(args.data_dir, args.index)
    else:
        engine = RAGEngine.load(args.index)
        if engine is None:
            print("索引不存在，請先執行 --build")
            exit(1)

    if args.query:
        results = engine.retrieve(args.query, top_k=args.top_k)
        print(f"\n查詢: {args.query}")
        for r in results:
            print(f"  [{r['score']:.3f}] Q: {r['q']}")
            print(f"         A: {r['a']}")
        if not results:
            print("  （無相關結果）")
        print(f"\n格式化 context:\n{engine.retrieve_context(args.query)}")
