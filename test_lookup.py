"""
test_lookup.py
---------------
驗證 qa_lookup.py 的查表機制,對 _LOOKUP_FILES 裡列出的每一類語料
(qa/html/python/css/javascript/logic/troubleshooting)都真的能命中、
而且回傳的答案內容正確,不是「檔案有加進清單但實際上比對邏輯有問題
沒真的生效」。

用法:
    python test_lookup.py

原理:
從每個 data/*.jsonl 檔案裡抽樣幾筆題目,直接把「訓練資料裡原始的問題
文字」丟給 qa_lookup.match_qa(),驗證:
1. 有沒有命中(回傳值不是 None)
2. 命中的答案內容,有沒有包含原始訓練資料裡的答案(html/python 這類
   會被包上```語言標籤```,前面加開場白的自然語言類則檢查答案結尾
   是否吻合)

另外也測試一題「完全不存在於任何語料庫」的問題,確認沒有被誤判命中
(避免比對邏輯過於寬鬆、什麼都判定成有命中)。
"""

import json
import os
import random

import qa_lookup

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLE_SIZE = 5
random.seed(42)  # 固定種子,每次跑抽到的樣本一致,方便比對結果


def _load_samples(filename, n=SAMPLE_SIZE):
    path = os.path.join(DATA_DIR, filename)
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            messages = item.get("messages") or []
            if len(messages) != 2:
                continue
            question = messages[0]["content"].strip()
            answer = messages[1]["content"].strip()
            entries.append((question, answer))
    return random.sample(entries, min(n, len(entries)))


def _check_hit(question, expected_answer, category):
    result = qa_lookup.match_qa(question, data_dir=DATA_DIR)
    if result is None:
        return False, "沒有命中(回傳None)"
    # html.jsonl / python.jsonl 的答案會被包成```語言\n答案\n```,
    # 其餘類別是「隨機開場白 + 原始答案」,兩種都用「原始答案是不是
    # 結果的子字串」來檢查,不管有沒有被包fence或加開場白都適用。
    if expected_answer not in result:
        return False, f"命中但內容對不上,實際回傳: {result[:80]}..."
    return True, "OK"


def main():
    categories = [
        ("qa.jsonl", "QA"),
        ("html.jsonl", "HTML"),
        ("python.jsonl", "Python"),
        ("css.jsonl", "CSS"),
        ("javascript.jsonl", "JavaScript"),
        ("logic.jsonl", "Logic"),
        ("troubleshooting.jsonl", "Troubleshooting"),
    ]

    total_pass = 0
    total_fail = 0

    for filename, label in categories:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"[{label:15s}] 找不到 {filename},略過")
            continue

        samples = _load_samples(filename)
        passed = 0
        for question, answer in samples:
            ok, message = _check_hit(question, answer, label)
            if ok:
                passed += 1
                total_pass += 1
            else:
                total_fail += 1
                print(f"[{label:15s}] FAIL: {question!r}")
                print(f"                 -> {message}")
        print(f"[{label:15s}] {passed}/{len(samples)} 通過")

    # 額外測試:完全不存在的問題,不應該被誤判命中。
    unrelated_question = "宇宙的盡頭到底有沒有邊界,這個問題目前科學界有結論了嗎"
    result = qa_lookup.match_qa(unrelated_question, data_dir=DATA_DIR)
    if result is None:
        print("[未命中測試   ] OK: 不相關的問題正確回傳 None,沒有被誤判")
    else:
        total_fail += 1
        print(f"[未命中測試   ] FAIL: 不相關的問題不應該命中,但回傳了: {result[:80]}...")

    print()
    print(f"總計: {total_pass} 通過, {total_fail} 失敗")
    if total_fail > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
