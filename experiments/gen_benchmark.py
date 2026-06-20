"""Generation/coherence benchmark on the released identity_v2 byte model.
Prompt groups, fp32 vs int8 vs int4, with automatic checks: exact-match vs fp32,
repetition rate, garbled rate, identity stability, average length. This tests whether
quantized generations stay USABLE (coherence), separate from the bpb loss benchmark."""
import json
from collections import Counter

import torch
import torch.nn as nn
from voxyne_pipeline.encoder import RahuKetuEncoder
from voxyne_pipeline.generate import generate as gen
from voxyne_pipeline.model import VoxyneLM

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "/workspace/runs/identity_v2/voxyne-best.pt"
GROUP = 128


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


ck = torch.load(CKPT, map_location=DEV, weights_only=False)
cfg = ck["config"]


def cg(k, d):
    return getattr(cfg, k) if hasattr(cfg, k) else (cfg.get(k, d) if isinstance(cfg, dict) else d)


CFG = dict(alphabet_size=256, d_model=cg("d_model", 768), n_heads=cg("n_heads", 12),
           n_layers=cg("n_layers", 18), max_len=cg("model_max_len", 1024), aux_dim=1, use_aux=True)
enc = RahuKetuEncoder(alphabet_size=256, use_sigma_k=True, use_features=False).to(DEV).eval()


def load():
    m = VoxyneLM(**CFG).to(DEV).eval()
    m.load_state_dict(ck["model_state"])
    return m


GROUPS = {
    "greetings_identity": ["hi", "who are you?", "what is your name?", "who made you?",
                           "tell me about yourself", "are you a real person?"],
    "simple_conversation": ["how are you?", "what's up?", "how was your day?", "i'm bored",
                            "talk to me about anything", "i'm so tired"],
    "factual_qa": ["what is the capital of France?", "who wrote Romeo and Juliet?",
                   "what is the largest planet?", "what is water made of?"],
    "arithmetic": ["what is 2 plus 2?", "what is 10 times 5?", "what is 100 minus 37?"],
    "explanation": ["explain photosynthesis simply", "what is gravity?", "how do plants grow?"],
    "instruction": ["write a short poem about the sea", "list three colors",
                    "say hello in a friendly way"],
    "uncertainty": ["what is the meaning of life?", "what will happen tomorrow?",
                    "what am i thinking right now?"],
    "repetition_traps": ["repeat the word hello", "say cat three times", "count from one to five"],
}
PAIRS = [(g, p) for g, ps in GROUPS.items() for p in ps]


def has_rep(text, n=3, thr=3):
    w = text.split()
    if len(w) < n:
        return False
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return bool(grams) and Counter(grams).most_common(1)[0][1] >= thr


def garbled(text):
    if "\x01" in text or "\x02" in text or not text.strip():
        return True
    good = sum(c.isalnum() or c.isspace() or c in ".,!?'\"-:;()" for c in text)
    return good / max(len(text), 1) < 0.6


def run(model):
    return {p: gen(model, enc, p, max_new=56, temperature=0.0, device=DEV, max_len=1024)
            for _, p in PAIRS}


outs = {"fp32": run(load())}
m8 = load()
fake_byte(m8, 8)
outs["int8"] = run(m8)
m4 = load()
fake_byte(m4, 4)
outs["int4"] = run(m4)

ID_PROMPTS = ["who are you?", "what is your name?", "tell me about yourself"]


def metrics(prec):
    o = outs[prec]
    return {
        "n": len(PAIRS),
        "exact_match_vs_fp32": sum(o[p] == outs["fp32"][p] for _, p in PAIRS),
        "repetition": sum(has_rep(o[p]) for _, p in PAIRS),
        "garbled": sum(garbled(o[p]) for _, p in PAIRS),
        "identity_voxyne_3of3": sum("voxyne" in o[p].lower() for p in ID_PROMPTS),
        "avg_len": round(sum(len(o[p]) for _, p in PAIRS) / len(PAIRS), 1),
    }


summary = {prec: metrics(prec) for prec in ("fp32", "int8", "int4")}
print("=== SUMMARY ===", flush=True)
print(json.dumps(summary, indent=2), flush=True)
print("=== SAMPLES (identity + repetition_traps) ===", flush=True)
for g in ("greetings_identity", "repetition_traps", "instruction"):
    print(f"## {g}", flush=True)
    for p in GROUPS[g]:
        print(f"  Q {p!r}", flush=True)
        for prec in ("fp32", "int4"):
            print(f"     {prec}: {outs[prec][p]!r}", flush=True)
json.dump({"subject": "identity_v2", "groups": list(GROUPS), "summary": summary, "outputs": outs},
          open("/workspace/artifacts/generation_benchmark.json", "w"), indent=2)
print("GEN BENCHMARK DONE", flush=True)
