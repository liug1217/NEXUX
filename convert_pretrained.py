"""
convert_pretrained.py
----------------------
一次性的轉換腳本:把 HuggingFace 上的 ckiplab/gpt2-base-chinese(中研院 CKIP Lab
訓練的繁體中文 GPT2 預訓練模型)轉換成這個專案 model.py / numpy_gpt.py 認得的
checkpoint 格式,取代原本「從零訓練」的起始點。

只在本機執行一次(需要暫時安裝 transformers,不會進 requirements.txt,
不影響 Vercel 部署),之後 train_sft.py 就能接著這個 checkpoint 繼續微調。

命名對照(HuggingFace GPT2 -> 這個專案的 model.py):
    transformer.wte.weight              -> token_emb.weight
    transformer.wpe.weight              -> pos_emb.weight
    transformer.h.{i}.ln_1.*            -> blocks.{i}.ln1.*
    transformer.h.{i}.attn.c_attn.*     -> blocks.{i}.attn.qkv_proj.*(weight要轉置,
                                            HF 的 Conv1D 是 (in,out),
                                            nn.Linear 是 (out,in))
    transformer.h.{i}.attn.c_proj.*     -> blocks.{i}.attn.out_proj.*(轉置)
    transformer.h.{i}.ln_2.*            -> blocks.{i}.ln2.*
    transformer.h.{i}.mlp.c_fc.*        -> blocks.{i}.ff.net.0.*(轉置)
    transformer.h.{i}.mlp.c_proj.*      -> blocks.{i}.ff.net.2.*(轉置)
    transformer.ln_f.*                  -> ln_f.*
    lm_head.weight(== wte.weight,tied)  -> head.weight(轉換時完整存一份,
                                            微調過程會讓兩者逐漸產生差異;
                                            numpy_gpt.py 仍保留「找不到
                                            head.weight 時 fallback 用
                                            token_emb.weight」的邏輯,只是
                                            這個轉換流程目前用不到那個分支)

用法:
    python convert_pretrained.py
"""

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

_CACHE_DIR = r"C:\mlcache"
_MODEL_NAME = "ckiplab/gpt2-base-chinese"

OUT_CHECKPOINT = "checkpoint_pretrained.pt"
OUT_VOCAB = "vocab_pretrained.txt"


def convert():
    print(f"[convert] 載入 {_MODEL_NAME} ...")
    hf_model = AutoModelForCausalLM.from_pretrained(_MODEL_NAME, cache_dir=_CACHE_DIR)
    hf_tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME, cache_dir=_CACHE_DIR)
    hf_sd = hf_model.state_dict()
    cfg = hf_model.config

    n_layer = cfg.n_layer
    new_sd = {}

    new_sd["token_emb.weight"] = hf_sd["transformer.wte.weight"].clone()
    new_sd["pos_emb.weight"] = hf_sd["transformer.wpe.weight"].clone()

    for i in range(n_layer):
        hf_prefix = f"transformer.h.{i}"
        new_prefix = f"blocks.{i}"

        new_sd[f"{new_prefix}.ln1.weight"] = hf_sd[f"{hf_prefix}.ln_1.weight"].clone()
        new_sd[f"{new_prefix}.ln1.bias"] = hf_sd[f"{hf_prefix}.ln_1.bias"].clone()

        # HF 的 Conv1D 權重是 (in_features, out_features),
        # nn.Linear 是 (out_features, in_features),要轉置。
        new_sd[f"{new_prefix}.attn.qkv_proj.weight"] = hf_sd[f"{hf_prefix}.attn.c_attn.weight"].t().clone()
        new_sd[f"{new_prefix}.attn.qkv_proj.bias"] = hf_sd[f"{hf_prefix}.attn.c_attn.bias"].clone()
        new_sd[f"{new_prefix}.attn.out_proj.weight"] = hf_sd[f"{hf_prefix}.attn.c_proj.weight"].t().clone()
        new_sd[f"{new_prefix}.attn.out_proj.bias"] = hf_sd[f"{hf_prefix}.attn.c_proj.bias"].clone()

        new_sd[f"{new_prefix}.ln2.weight"] = hf_sd[f"{hf_prefix}.ln_2.weight"].clone()
        new_sd[f"{new_prefix}.ln2.bias"] = hf_sd[f"{hf_prefix}.ln_2.bias"].clone()

        new_sd[f"{new_prefix}.ff.net.0.weight"] = hf_sd[f"{hf_prefix}.mlp.c_fc.weight"].t().clone()
        new_sd[f"{new_prefix}.ff.net.0.bias"] = hf_sd[f"{hf_prefix}.mlp.c_fc.bias"].clone()
        new_sd[f"{new_prefix}.ff.net.2.weight"] = hf_sd[f"{hf_prefix}.mlp.c_proj.weight"].t().clone()
        new_sd[f"{new_prefix}.ff.net.2.bias"] = hf_sd[f"{hf_prefix}.mlp.c_proj.bias"].clone()

    new_sd["ln_f.weight"] = hf_sd["transformer.ln_f.weight"].clone()
    new_sd["ln_f.bias"] = hf_sd["transformer.ln_f.bias"].clone()

    # lm_head 跟 token_emb 現在數值上完全相同(tied embedding),但 model.py
    # 的 GPTModel 目前是「輸入 embedding」跟「輸出 head」各自獨立的參數
    # (沒有做真正的權重共享),微調過程中兩者會各自更新、逐漸產生差異,
    # 所以這裡還是把 head.weight 存一份完整的初始值,讓 model.py 不用改
    # 架構就能直接 load_state_dict。之後如果要在部署時省空間,再由
    # export_weights.py/numpy_gpt.py 決定「微調後兩者夠接近就只存一份」。
    assert torch.equal(hf_sd["lm_head.weight"], hf_sd["transformer.wte.weight"]), \
        "lm_head 跟 wte 權重不相同,不是預期的 tied embedding,需要另外處理 head.weight"
    new_sd["head.weight"] = hf_sd["lm_head.weight"].clone()

    architecture = {
        "n_embd": cfg.n_embd,
        "n_head": cfg.n_head,
        "n_layer": cfg.n_layer,
        "block_size": cfg.n_positions,
    }

    checkpoint = {
        "model_state_dict": new_sd,
        "vocab_size": cfg.vocab_size,
        "architecture": architecture,
        "sft_applied": False,
        "pretrained_source": _MODEL_NAME,
    }
    torch.save(checkpoint, OUT_CHECKPOINT)
    print(f"[convert] 已存檔: {OUT_CHECKPOINT}")
    print(f"[convert] 架構: {architecture}, vocab_size={cfg.vocab_size}")

    # 匯出 vocab.txt(逐行一個 token,索引即 token id),
    # 之後可以直接用簡單的 lookup-based tokenizer 讀取,不需要完整的
    # transformers/tokenizers 套件。
    vocab = hf_tokenizer.get_vocab()
    id_to_token = [None] * len(vocab)
    for tok, idx in vocab.items():
        id_to_token[idx] = tok
    with open(OUT_VOCAB, "w", encoding="utf-8") as f:
        for tok in id_to_token:
            f.write(tok + "\n")
    print(f"[convert] 已存檔: {OUT_VOCAB}({len(id_to_token)} 個 token)")

    return checkpoint, hf_tokenizer, hf_model


if __name__ == "__main__":
    convert()
