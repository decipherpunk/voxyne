"""Consolidated claim artifacts for CLAIMS.md C1-C6 + the int2 probe.
Writes /workspace/artifacts/results.json + manifest.json. Run on the GPU pod."""
import json
import math
import os
import platform
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = "/workspace/artifacts"
os.makedirs(OUT, exist_ok=True)
SEED = 0
torch.manual_seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BYTE_CKPT = "/workspace/voxyne-rk-infer.pt"
GROUP = 128

TEXTS = [
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France, named after the engineer Gustave Eiffel.",
    "Photosynthesis is the process by which green plants use sunlight to synthesize food from carbon dioxide and water, releasing oxygen.",
    "The Roman Empire was one of the largest empires in ancient history; at its height it held sway over some seventy million people.",
    "A transistor is a semiconductor device used to amplify or switch electrical signals, a basic building block of modern electronics.",
    "The Amazon rainforest is the largest tropical rainforest in the world, covering much of northwestern Brazil and neighbouring countries.",
]


def _grp(W, cd, group):
    moved = cd != W.dim() - 1
    Wt = W.transpose(cd, -1).contiguous() if moved else W
    shp = Wt.shape
    inn = shp[-1]
    g = group if inn % group == 0 else inn
    return Wt, shp, g, moved


def q_sym(W, cd, group=GROUP, bits=4):
    n = 2 ** (bits - 1)
    div = max(n - 1, 1)
    Wt, shp, g, moved = _grp(W.clone(), cd, group)
    Wg = Wt.reshape(-1, shp[-1] // g, g)
    s = (Wg.abs().amax(2, keepdim=True) / div).clamp(min=1e-8)
    Wq = ((Wg / s).round().clamp(-n, n - 1) * s).reshape(shp)
    return Wq.transpose(cd, -1).contiguous() if moved else Wq


def q_asym(W, cd, group=GROUP):
    Wt, shp, g, moved = _grp(W.clone(), cd, group)
    Wg = Wt.reshape(-1, shp[-1] // g, g)
    mn = Wg.amin(2, keepdim=True)
    mx = Wg.amax(2, keepdim=True)
    s = ((mx - mn) / 15.0).clamp(min=1e-8)
    Wq = (((Wg - mn) / s).round().clamp(0, 15) * s + mn).reshape(shp)
    return Wq.transpose(cd, -1).contiguous() if moved else Wq


def q_zerofree2(W, cd, group=GROUP):
    Wt, shp, g, moved = _grp(W.clone(), cd, group)
    Wg = Wt.reshape(-1, shp[-1] // g, g)
    s = (Wg.abs().amax(2, keepdim=True) / 2).clamp(min=1e-8)
    qn = Wg / s
    q = qn.round().clamp(-2, 2)
    q = torch.where(q == 0, torch.where(qn >= 0, torch.ones_like(q), -torch.ones_like(q)), q)
    Wq = (q * s).reshape(shp)
    return Wq.transpose(cd, -1).contiguous() if moved else Wq


def q_resid2(W, cd, group=GROUP, stages=2):
    out = torch.zeros_like(W)
    resid = W.clone()
    for _ in range(stages):
        qv = q_sym(resid, cd, group, 2)
        out = out + qv
        resid = resid - qv
    return out


def fake_byte(model, fn):
    for mod in model.modules():
        if isinstance(mod, (nn.Linear, nn.Embedding)):
            mod.weight.data = fn(mod.weight.data, 1)


def fake_hf(model, fn, body_only=False):
    """body_only=True spares the vocab table: both the input Embedding AND the
    vocab-sized output projection (lm_head, often tied to the embedding)."""
    try:
        from transformers.pytorch_utils import Conv1D
    except Exception:
        Conv1D = ()
    vocab = getattr(getattr(model, "config", None), "vocab_size", None)
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            if body_only and vocab and mod.weight.shape[0] == vocab:
                continue  # lm_head / output vocab projection -- this is the table, spare it
            mod.weight.data = fn(mod.weight.data, 1)
        elif Conv1D and isinstance(mod, Conv1D):
            if body_only and vocab and mod.weight.shape[1] == vocab:
                continue  # Conv1D weight is (in, out); vocab output -- spare it
            mod.weight.data = fn(mod.weight.data, 0)
        elif isinstance(mod, nn.Embedding) and not body_only:
            mod.weight.data = fn(mod.weight.data, 1)


results = {"byte_sigmaK": {}, "int2_probe": {}, "token_models": {}}

# ===== BYTE + sigma_K (rk model) =====
from voxyne_pipeline.loop_infer import load
from voxyne_pipeline.generate import generate as gen_text

mb, enc, _ = load(BYTE_CKPT, DEV)


@torch.no_grad()
def byte_ppl(model):
    ps = []
    for t in TEXTS:
        d = t.encode()
        x = torch.tensor([[b + 1 for b in d]], device=DEV)
        s = torch.tensor([[1.0] * len(d)], device=DEV)
        _, aux = enc(x, sigma_K=s)
        dl, _, _ = model(x, aux)
        ps.append(F.cross_entropy(dl[0, :-1], torch.tensor(list(d[1:]), device=DEV)).item())
    return math.exp(sum(ps) / len(ps))


b0 = byte_ppl(mb)
results["byte_sigmaK"]["fp32_ppl"] = round(b0, 4)
print(f"byte fp32 ppl={b0:.3f}", flush=True)
for name, fn in [("int8", lambda W, cd: q_sym(W, cd, GROUP, 8)),
                 ("int4_naive", lambda W, cd: q_sym(W, cd, GROUP, 4))]:
    m, _, _ = load(BYTE_CKPT, DEV)
    fake_byte(m, fn)
    p = byte_ppl(m)
    results["byte_sigmaK"][name] = {"ppl": round(p, 4), "pct_increase": round(100 * (p / b0 - 1), 2)}
    print(f"byte {name}: +{100*(p/b0-1):.1f}%", flush=True)
    del m

# C2: int4 preserves greedy output (determinism / bit-exactness of behavior)
mb_i4, _, _ = load(BYTE_CKPT, DEV)
fake_byte(mb_i4, lambda W, cd: q_sym(W, cd, GROUP, 4))
c2 = {}
for p in ["who are you?", "hello"]:
    a = gen_text(mb, enc, p, max_new=40, temperature=0.0, device=DEV, max_len=1024)
    bb = gen_text(mb_i4, enc, p, max_new=40, temperature=0.0, device=DEV, max_len=1024)
    c2[p] = {"identical": a == bb}
results["byte_sigmaK"]["int4_greedy_identical"] = c2
del mb_i4

# int2 probe
for name, fn in [("int2_naive", lambda W, cd: q_sym(W, cd, GROUP, 2)),
                 ("int2_zerofree", q_zerofree2),
                 ("int2_residual_x2", lambda W, cd: q_resid2(W, cd, GROUP, 2)),
                 ("int2_residual_x3", lambda W, cd: q_resid2(W, cd, GROUP, 3))]:
    m, _, _ = load(BYTE_CKPT, DEV)
    fake_byte(m, fn)
    p = byte_ppl(m)
    results["int2_probe"][name] = {"ppl": round(p, 4), "pct_increase": round(100 * (p / b0 - 1), 2)}
    print(f"byte {name}: +{100*(p/b0-1):.0f}%", flush=True)
    del m

# ===== TOKEN MODELS =====
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

WT_URL = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet"
WT = pd.read_parquet(WT_URL)["text"].tolist()
EVAL = "\n\n".join(t for t in WT if t.strip())
CALIB = [t for t in WT if len(t.strip()) > 200][:128]


@torch.no_grad()
def tok_ppl_wt(model, tok, maxlen=1024, limit=40):
    ids = tok(EVAL, return_tensors="pt").input_ids.to(model.device)
    nll, n, c = 0.0, 0, 0
    for i in range(0, ids.size(1) - 1, maxlen):
        ch = ids[:, i:i + maxlen]
        if ch.size(1) < 2:
            break
        nll += model(ch, labels=ch).loss.item() * (ch.size(1) - 1)
        n += ch.size(1) - 1
        c += 1
        if c >= limit:
            break
    return math.exp(nll / n)


@torch.no_grad()
def tok_ppl_passages(model, tok):
    ls = []
    for t in TEXTS:
        ids = tok(t, return_tensors="pt").input_ids.to(model.device)
        ls.append(model(ids, labels=ids).loss.item())
    return math.exp(sum(ls) / len(ls))


MODELS = [("gpt2", "gpt2", "conv1d"),
          ("HuggingFaceTB/SmolLM2-135M", "smollm2-135m", "llama"),
          ("HuggingFaceTB/SmolLM2-360M", "smollm2-360m", "llama")]

for mid, label, arch in MODELS:
    tok = AutoTokenizer.from_pretrained(mid)
    base = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float16).to(DEV).eval()
    fp_wt = tok_ppl_wt(base, tok)
    fp_ps = tok_ppl_passages(base, tok)
    r = {"arch": arch, "fp16_wikitext_ppl": round(fp_wt, 4), "fp16_passages_ppl": round(fp_ps, 4),
         "schemes": {}}
    del base
    torch.cuda.empty_cache()
    for nm, fn, bo in [("naive_int4", lambda W, cd: q_sym(W, cd, GROUP, 4), False),
                       ("asym_int4", q_asym, False),
                       ("table_fp16_body_int4", lambda W, cd: q_sym(W, cd, GROUP, 4), True)]:
        m = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float16).to(DEV).eval()
        fake_hf(m, fn, body_only=bo)
        p = tok_ppl_wt(m, tok)
        r["schemes"][nm] = {"wikitext_ppl": round(p, 4), "pct_increase": round(100 * (p / fp_wt - 1), 1)}
        print(f"{label} {nm}: +{100*(p/fp_wt-1):.0f}%", flush=True)
        del m
        torch.cuda.empty_cache()
    if arch == "llama":
        try:
            from gptqmodel import GPTQModel, QuantizeConfig
            gm = GPTQModel.load(mid, QuantizeConfig(bits=4, group_size=GROUP))
            gm.quantize(CALIB, batch_size=4)
            gpath = f"/workspace/{label}-gptq"
            gm.save(gpath)
            del gm
            torch.cuda.empty_cache()
            gq = GPTQModel.load(gpath)
            inner = getattr(gq, "model", gq)
            inner.eval()
            p = tok_ppl_wt(inner, tok)
            r["schemes"]["gptq_int4"] = {"wikitext_ppl": round(p, 4),
                                         "pct_increase": round(100 * (p / fp_wt - 1), 1)}
            print(f"{label} gptq_int4: +{100*(p/fp_wt-1):.1f}%", flush=True)
            del gq, inner
            torch.cuda.empty_cache()
        except Exception as e:
            r["schemes"]["gptq_int4"] = {"error": str(e)[:200]}
            print(f"{label} gptq FAILED: {str(e)[:120]}", flush=True)
    results["token_models"][label] = r

# ===== MANIFEST =====
manifest = {
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "seed": SEED,
    "device": DEV,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "python": platform.python_version(),
    "torch": torch.__version__,
    "byte_checkpoint": BYTE_CKPT,
    "byte_params": int(sum(p.numel() for p in mb.parameters())),
    "token_models": [m[0] for m in MODELS],
    "eval_corpora": {"passages": len(TEXTS),
                     "wikitext2": "Salesforce/wikitext wikitext-2-raw-v1 test (parquet)"},
    "quant_settings": {"group_size": GROUP, "naive": "symmetric RTN", "gptq_bits": 4,
                       "gptq_group": GROUP, "metric": "relative perplexity increase, % over fp32/fp16"},
    "notes": "GPT-2 Conv1D not supported by GPTQ tooling; uses naive/asym/table-fp16 only.",
}
try:
    import transformers
    manifest["transformers"] = transformers.__version__
except Exception:
    pass
try:
    import gptqmodel
    manifest["gptqmodel"] = gptqmodel.__version__
except Exception:
    pass

json.dump(results, open(f"{OUT}/results.json", "w"), indent=2)
json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=2)
print("ARTIFACTS DONE ->", OUT, flush=True)
print(json.dumps(results, indent=2), flush=True)
