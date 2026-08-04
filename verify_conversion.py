"""
verify_conversion.py
---------------------
驗證 convert_pretrained.py 轉換出來的權重是否正確:
1. 把 checkpoint_pretrained.pt 匯出成 numpy_gpt.py 看得懂的格式
2. 分別用 HuggingFace 原始模型(torch)跟 NumpyGPT 對同一段輸入做前向運算
3. 比較兩邊的 logits 數值,確認轉換沒有出錯(允許浮點誤差)

只在本機驗證用,跑完就可以刪除,不是專案正式流程的一部分。
"""

import json

import numpy as np
import torch

from numpy_gpt import NumpyGPT

_KEY_SEP_ORIGINAL = "."
_KEY_SEP_NPZ = "__"


def export_pretrained_for_numpy(checkpoint_path, meta_path, npz_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    arch = checkpoint["architecture"]

    meta = {
        "config": {
            "vocab_size": checkpoint["vocab_size"],
            "n_embd": arch["n_embd"],
            "n_head": arch["n_head"],
            "n_layer": arch["n_layer"],
            "block_size": arch["block_size"],
        },
        "sft_applied": checkpoint.get("sft_applied", False),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    arrays = {
        name.replace(_KEY_SEP_ORIGINAL, _KEY_SEP_NPZ): tensor.numpy().astype(np.float32)
        for name, tensor in state_dict.items()
    }
    np.savez(npz_path, **arrays)
    print(f"[verify] 已匯出驗證用權重: {meta_path}, {npz_path}")


def main():
    # NumpyGPT 固定會去 meta_path 同一個資料夾底下找檔名叫 weights.npz 的檔案,
    # 為了不跟正式模型的 weights.npz 打架,獨立放一個測試用資料夾。
    import os
    os.makedirs("pretrained_test", exist_ok=True)
    export_pretrained_for_numpy(
        "checkpoint_pretrained.pt",
        "pretrained_test/weights_meta.json",
        "pretrained_test/weights.npz",
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_model = AutoModelForCausalLM.from_pretrained(
        "ckiplab/gpt2-base-chinese", cache_dir=r"C:\mlcache"
    )
    hf_model.eval()
    hf_tokenizer = AutoTokenizer.from_pretrained(
        "ckiplab/gpt2-base-chinese", cache_dir=r"C:\mlcache"
    )

    text = "為什麼天空是藍色的"
    ids = hf_tokenizer.encode(text, add_special_tokens=False)
    print(f"[verify] 測試輸入: {text!r} -> ids={ids}")

    with torch.no_grad():
        hf_out = hf_model(torch.tensor([ids]))
        hf_logits = hf_out.logits[0].numpy()  # (T, vocab_size)

    numpy_model = NumpyGPT("pretrained_test/weights_meta.json")
    numpy_logits = numpy_model.forward(ids)  # (T, vocab_size)

    print(f"[verify] HF logits shape: {hf_logits.shape}, Numpy logits shape: {numpy_logits.shape}")

    diff = np.abs(hf_logits.astype(np.float64) - numpy_logits.astype(np.float64))
    print(f"[verify] 最大絕對誤差: {diff.max():.6f}")
    print(f"[verify] 平均絕對誤差: {diff.mean():.6f}")

    # 比較兩邊各自預測下一個字最可能的 token 是否一致(比逐一數字比對更直觀)
    hf_next = hf_logits[-1].argmax()
    numpy_next = numpy_logits[-1].argmax()
    print(f"[verify] HF 預測下一字 id={hf_next} ({hf_tokenizer.decode([int(hf_next)])!r})")
    print(f"[verify] Numpy 預測下一字 id={numpy_next} ({hf_tokenizer.decode([int(numpy_next)])!r})")

    if diff.max() < 0.05:
        print("[verify] 通過:誤差在合理浮點範圍內,轉換正確。")
    else:
        print("[verify] 警告:誤差偏大,轉換可能有問題,需要檢查。")


if __name__ == "__main__":
    main()
