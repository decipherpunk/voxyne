"""Coherence comparison under quantization: Voxyne vs a token chat baseline.
Voxyne identity_v2 (fp32, naive int4) vs SmolLM2-135M-Instruct (fp16, naive int4, GPTQ int4),
same chat prompts. Question: does Voxyne stay coherent under naive int4 while the token chat
model degrades and only recovers with GPTQ? Metric: repetition rate, garbled rate."""
import json
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from voxyne_pipeline.encoder import RahuKetuEncoder
from voxyne_pipeline.generate import generate as vgen
from voxyne_pipeline.model import VoxyneLM

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GROUP = 128
PROMPTS = ["hi", "how are you?", "tell me about yourself", "what do you like to do?",
           "i'm feeling sad today", "can you help me with something?", "tell me a short story",
           "what is your favorite season?", "i had a long day", "cheer me up",
           "what should i cook for dinner?", "goodbye"]


def q_sym(W, cd, group, bits):
    n = 2 ** (bits - 1)
    div = max(n - 1, 1)
    moved = cd != W.dim() - 1
    Wt = W.transpose(cd, -1).contiguous() if moved else W.clone()
    shp = Wt.shape
    g = group if shp[-1] % group == 0 else shp[-1]
    Wg = Wt.reshape(-1, shp[-1] // g, g)
    s = (Wg.abs().amax(2, keepdim=True) / div).clamp(min=1e-8)
    Wq = ((Wg / s).round().clamp(-n, n - 1) * s).reshape(shp)
    return Wq.transpose(cd, -1).contiguous() if moved else Wq


def fake_byte(model, bits):
    for mod in model.modules():
        if isinstance(mod, (nn.Linear, nn.Embedding)):
            mod.weight.data = q_sym(mod.weight.data, 1, GROUP, bits)


def fake_hf(model, bits):
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            mod.weight.data = q_sym(mod.weight.data, 1, GROUP, bits)
        elif isinstance(mod, nn.Embedding):
            mod.weight.data = q_sym(mod.weight.data, 1, GROUP, bits)


def has_rep(t, n=3, thr=3):
    w = t.split()
    if len(w) < n:
        return False
    g = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return bool(g) and Counter(g).most_common(1)[0][1] >= thr


def garbled(t):
    if not t.strip():
        return True
    good = sum(c.isalnum() or c.isspace() or c in ".,!?'\"-:;()" for c in t)
    return good / max(len(t), 1) < 0.6


# Voxyne
ck = torch.load("/workspace/runs/identity_v2/voxyne-best.pt", map_location=DEV, weights_only=False)
cfg = ck["config"]
cg = lambda k, d: getattr(cfg, k) if hasattr(cfg, k) else (cfg.get(k, d) if isinstance(cfg, dict) else d)
CFG = dict(alphabet_size=256, d_model=cg("d_model", 768), n_heads=cg("n_heads", 12),
           n_layers=cg("n_layers", 18), max_len=cg("model_max_len", 1024), aux_dim=1, use_aux=True)
enc = RahuKetuEncoder(alphabet_size=256, use_sigma_k=True, use_features=False).to(DEV).eval()


def load_vox():
    m = VoxyneLM(**CFG).to(DEV).eval()
    m.load_state_dict(ck["model_state"])
    return m


def vrun(m):
    return {p: vgen(m, enc, p, max_new=56, temperature=0.0, device=DEV, max_len=1024) for p in PROMPTS}


res = {}
res["voxyne_fp32"] = vrun(load_vox())
vi4 = load_vox()
fake_byte(vi4, 4)
res["voxyne_int4_naive"] = vrun(vi4)
del vi4

# SmolLM2-135M-Instruct
MID = "HuggingFaceTB/SmolLM2-135M-Instruct"
tok = AutoTokenizer.from_pretrained(MID)


def sgen(model, p):
    enc_in = tok.apply_chat_template([{"role": "user", "content": p}], return_tensors="pt",
                                     add_generation_prompt=True, return_dict=True).to(model.device)
    out = model.generate(**enc_in, max_new_tokens=56, do_sample=False, pad_token_id=tok.eos_token_id)
    n = enc_in["input_ids"].shape[1]
    return tok.decode(out[0][n:], skip_special_tokens=True).strip()


def srun(model):
    return {p: sgen(model, p) for p in PROMPTS}


base = AutoModelForCausalLM.from_pretrained(MID, torch_dtype=torch.float16).to(DEV).eval()
res["smollm_instruct_fp16"] = srun(base)
del base
torch.cuda.empty_cache()
m = AutoModelForCausalLM.from_pretrained(MID, torch_dtype=torch.float16).to(DEV).eval()
fake_hf(m, 4)
res["smollm_instruct_int4_naive"] = srun(m)
del m
torch.cuda.empty_cache()
try:
    from gptqmodel import GPTQModel, QuantizeConfig
    CALIB = [t for t in pd.read_parquet("https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/train-00000-of-00001.parquet")["text"].tolist() if len(t.strip()) > 200][:128]
    gm = GPTQModel.load(MID, QuantizeConfig(bits=4, group_size=GROUP))
    gm.quantize(CALIB, batch_size=4)
    gm.save("/workspace/smol-inst-gptq")
    del gm
    torch.cuda.empty_cache()
    gq = GPTQModel.load("/workspace/smol-inst-gptq")
    res["smollm_instruct_gptq"] = srun(getattr(gq, "model", gq).eval())
except Exception as e:
    res["smollm_instruct_gptq"] = {"error": str(e)[:150]}


def summ(k):
    o = res[k]
    if "error" in o:
        return o
    return {"repetition": sum(has_rep(o[p]) for p in PROMPTS),
            "garbled": sum(garbled(o[p]) for p in PROMPTS),
            "n": len(PROMPTS),
            "avg_len": round(sum(len(o[p]) for p in PROMPTS) / len(PROMPTS), 1)}


summary = {k: summ(k) for k in res}
print("=== SUMMARY (lower repetition/garbled = more coherent) ===", flush=True)
print(json.dumps(summary, indent=2), flush=True)
print("=== SAMPLES ===", flush=True)
for p in PROMPTS[:6]:
    print(f"Q: {p!r}", flush=True)
    for k in ["voxyne_int4_naive", "smollm_instruct_int4_naive", "smollm_instruct_gptq"]:
        v = res[k].get(p, res[k]) if isinstance(res[k], dict) else res[k]
        print(f"   {k}: {v!r}", flush=True)
json.dump({"prompts": PROMPTS, "summary": summary, "outputs": res},
          open("/workspace/artifacts/coherence_compare.json", "w"), indent=2)
print("COHERENCE COMPARE DONE", flush=True)
