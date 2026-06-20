"""Benchmark v2 (paper-grade). Subject: released identity_v2 byte model.
Fixes over v1: full WikiText-2 test corpus, token-weighted bits-per-byte (fair across
byte and token models), GPTQ calibrated on the TRAIN split (no eval leakage), same
corpus for byte and token models, zero-free vs ordinary residual int2 ablation.
Metric: bits per byte (bpb) and percent increase vs each model's own fp baseline."""
import hashlib
import json
import math
import os
import platform
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from voxyne_pipeline.encoder import RahuKetuEncoder
from voxyne_pipeline.model import VoxyneLM

OUT = "/workspace/artifacts"
os.makedirs(OUT, exist_ok=True)
SEED = 0
torch.manual_seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LN2 = math.log(2)
GROUP = 128
BYTE_CKPT = "/workspace/runs/identity_v2/voxyne-best.pt"
BASE = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1"

TEST = "\n\n".join(t for t in pd.read_parquet(f"{BASE}/test-00000-of-00001.parquet")["text"].tolist() if t.strip())
TRAIN = [t for t in pd.read_parquet(f"{BASE}/train-00000-of-00001.parquet")["text"].tolist() if t.strip()]
CALIB = [t for t in TRAIN if len(t.strip()) > 200][:128]
TEST_BYTES = len(TEST.encode("utf-8"))


def _grp(W, cd, group):
    moved = cd != W.dim() - 1
    Wt = W.transpose(cd, -1).contiguous() if moved else W.clone()
    shp = Wt.shape
    g = group if shp[-1] % group == 0 else shp[-1]
    return Wt, shp, g, moved


def q_sym(W, cd, group=GROUP, bits=4):
    n = 2 ** (bits - 1)
    div = max(n - 1, 1)
    Wt, shp, g, moved = _grp(W, cd, group)
    Wg = Wt.reshape(-1, shp[-1] // g, g)
    s = (Wg.abs().amax(2, keepdim=True) / div).clamp(min=1e-8)
    Wq = ((Wg / s).round().clamp(-n, n - 1) * s).reshape(shp)
    return Wq.transpose(cd, -1).contiguous() if moved else Wq


def q_asym(W, cd, group=GROUP):
    Wt, shp, g, moved = _grp(W, cd, group)
    Wg = Wt.reshape(-1, shp[-1] // g, g)
    mn, mx = Wg.amin(2, keepdim=True), Wg.amax(2, keepdim=True)
    s = ((mx - mn) / 15.0).clamp(min=1e-8)
    Wq = (((Wg - mn) / s).round().clamp(0, 15) * s + mn).reshape(shp)
    return Wq.transpose(cd, -1).contiguous() if moved else Wq


def q_zerofree2(W, cd, group=GROUP):
    Wt, shp, g, moved = _grp(W, cd, group)
    Wg = Wt.reshape(-1, shp[-1] // g, g)
    s = (Wg.abs().amax(2, keepdim=True) / 2).clamp(min=1e-8)
    qn = Wg / s
    q = qn.round().clamp(-2, 2)
    q = torch.where(q == 0, torch.where(qn >= 0, torch.ones_like(q), -torch.ones_like(q)), q)
    return (q * s).reshape(shp).transpose(cd, -1).contiguous() if moved else (q * s).reshape(shp)


def q_resid(W, cd, group=GROUP, stages=2, mode="ordinary"):
    out = torch.zeros_like(W)
    resid = W.clone()
    for _ in range(stages):
        qv = q_zerofree2(resid, cd, group) if mode == "zerofree" else q_sym(resid, cd, group, 2)
        out = out + qv
        resid = resid - qv
    return out


def fake_byte(model, fn):
    for mod in model.modules():
        if isinstance(mod, (nn.Linear, nn.Embedding)):
            mod.weight.data = fn(mod.weight.data, 1)


def fake_hf(model, fn, body_only=False):
    try:
        from transformers.pytorch_utils import Conv1D
    except Exception:
        Conv1D = ()
    vocab = getattr(getattr(model, "config", None), "vocab_size", None)
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            if body_only and vocab and mod.weight.shape[0] == vocab:
                continue
            mod.weight.data = fn(mod.weight.data, 1)
        elif Conv1D and isinstance(mod, Conv1D):
            if body_only and vocab and mod.weight.shape[1] == vocab:
                continue
            mod.weight.data = fn(mod.weight.data, 0)
        elif isinstance(mod, nn.Embedding) and not body_only:
            mod.weight.data = fn(mod.weight.data, 1)


# ---------- byte model (identity_v2) ----------
ck = torch.load(BYTE_CKPT, map_location=DEV, weights_only=False)
cfg = ck["config"]


def cg(k, d):
    return getattr(cfg, k) if hasattr(cfg, k) else (cfg.get(k, d) if isinstance(cfg, dict) else d)


CFG = dict(alphabet_size=256, d_model=cg("d_model", 768), n_heads=cg("n_heads", 12),
           n_layers=cg("n_layers", 18), max_len=cg("model_max_len", 1024), aux_dim=1, use_aux=True)
enc = RahuKetuEncoder(alphabet_size=256, use_sigma_k=True, use_features=False).to(DEV).eval()


def load_byte():
    m = VoxyneLM(**CFG).to(DEV).eval()
    m.load_state_dict(ck["model_state"])
    return m


@torch.no_grad()
def byte_bpb(model, maxlen=1024):
    data = TEST.encode("utf-8")
    tot_nll, tot_n = 0.0, 0
    for i in range(0, len(data) - 1, maxlen):
        ch = data[i:i + maxlen + 1]
        if len(ch) < 2:
            break
        x = torch.tensor([[b + 1 for b in ch[:-1]]], device=DEV)
        s = torch.ones_like(x, dtype=torch.float32)
        _, aux = enc(x, sigma_K=s)
        dl, _, _ = model(x, aux)
        y = torch.tensor(list(ch[1:]), device=DEV)
        tot_nll += F.cross_entropy(dl[0], y, reduction="sum").item()
        tot_n += len(ch) - 1
    return (tot_nll / tot_n) / LN2


results = {"corpus": "wikitext-2-raw test", "metric": "bits_per_byte", "byte_identity_v2": {},
           "int2_ablation": {}, "token_models": {}}

mb = load_byte()
bb = byte_bpb(mb)
results["byte_identity_v2"]["fp32_bpb"] = round(bb, 4)
print(f"byte fp32 bpb={bb:.4f}", flush=True)
for name, fn in [("int8", lambda W, c: q_sym(W, c, GROUP, 8)),
                 ("int4_naive", lambda W, c: q_sym(W, c, GROUP, 4))]:
    m = load_byte()
    fake_byte(m, fn)
    v = byte_bpb(m)
    results["byte_identity_v2"][name] = {"bpb": round(v, 4), "pct": round(100 * (v / bb - 1), 2)}
    print(f"byte {name}: bpb={v:.4f} +{100*(v/bb-1):.2f}%", flush=True)
    del m

# int2 ablation: ordinary vs zero-free residual at same bit budget
for name, fn in [("int2_naive", lambda W, c: q_sym(W, c, GROUP, 2)),
                 ("int2_resid_ordinary_x2", lambda W, c: q_resid(W, c, GROUP, 2, "ordinary")),
                 ("int2_resid_zerofree_x2", lambda W, c: q_resid(W, c, GROUP, 2, "zerofree")),
                 ("int2_resid_ordinary_x3", lambda W, c: q_resid(W, c, GROUP, 3, "ordinary")),
                 ("int2_resid_zerofree_x3", lambda W, c: q_resid(W, c, GROUP, 3, "zerofree"))]:
    m = load_byte()
    fake_byte(m, fn)
    v = byte_bpb(m)
    results["int2_ablation"][name] = {"bpb": round(v, 4), "pct": round(100 * (v / bb - 1), 2)}
    print(f"byte {name}: +{100*(v/bb-1):.1f}%", flush=True)
    del m

# ---------- token models (same corpus, bpb normalized by bytes) ----------
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def token_bpb(model, tok, maxlen=1024):
    ids = tok(TEST, return_tensors="pt").input_ids.to(model.device)
    tot_nll = 0.0
    for i in range(0, ids.size(1) - 1, maxlen):
        ch = ids[:, i:i + maxlen + 1]
        if ch.size(1) < 2:
            break
        out = model(ch[:, :-1])
        logits = out.logits if hasattr(out, "logits") else out[0]
        tot_nll += F.cross_entropy(logits[0], ch[0, 1:], reduction="sum").item()
    return (tot_nll / TEST_BYTES) / LN2


for mid, label, arch in [("gpt2", "gpt2", "conv1d"),
                         ("HuggingFaceTB/SmolLM2-135M", "smollm2-135m", "llama"),
                         ("HuggingFaceTB/SmolLM2-360M", "smollm2-360m", "llama")]:
    tok = AutoTokenizer.from_pretrained(mid)
    base = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float16).to(DEV).eval()
    fp = token_bpb(base, tok)
    r = {"arch": arch, "fp16_bpb": round(fp, 4), "schemes": {}}
    del base
    torch.cuda.empty_cache()
    for nm, fn, bo in [("naive_int4", lambda W, c: q_sym(W, c, GROUP, 4), False),
                       ("asym_int4", q_asym, False),
                       ("table_fp16_body_int4", lambda W, c: q_sym(W, c, GROUP, 4), True)]:
        m = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float16).to(DEV).eval()
        fake_hf(m, fn, body_only=bo)
        v = token_bpb(m, tok)
        r["schemes"][nm] = {"bpb": round(v, 4), "pct": round(100 * (v / fp - 1), 1)}
        print(f"{label} {nm}: +{100*(v/fp-1):.0f}%", flush=True)
        del m
        torch.cuda.empty_cache()
    if arch == "llama":
        try:
            from gptqmodel import GPTQModel, QuantizeConfig
            gm = GPTQModel.load(mid, QuantizeConfig(bits=4, group_size=GROUP))
            gm.quantize(CALIB, batch_size=4)  # CALIB is the TRAIN split (no eval leakage)
            gp = f"/workspace/{label}-gptq-v2"
            gm.save(gp)
            del gm
            torch.cuda.empty_cache()
            gq = GPTQModel.load(gp)
            inner = getattr(gq, "model", gq).eval()
            v = token_bpb(inner, tok)
            r["schemes"]["gptq_int4_train_calib"] = {"bpb": round(v, 4), "pct": round(100 * (v / fp - 1), 1)}
            print(f"{label} gptq(train-calib): +{100*(v/fp-1):.1f}%", flush=True)
            del gq, inner
            torch.cuda.empty_cache()
        except Exception as e:
            r["schemes"]["gptq_int4_train_calib"] = {"error": str(e)[:200]}
    results["token_models"][label] = r

# ---------- runtime (byte fp32 decode, GPU) ----------
mb.eval()
x = torch.tensor([[66]], device=DEV)
s = torch.ones_like(x, dtype=torch.float32)
with torch.no_grad():
    _, aux = enc(x, sigma_K=s)
    for _ in range(5):
        mb(x, aux)
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    N = 100
    for _ in range(N):
        mb(x, aux)
    if DEV == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
results["runtime_byte_fp32"] = {"device": DEV, "steps_per_sec": round(N / dt, 1),
                                "ms_per_step": round(1000 * dt / N, 2)}

# ---------- manifest ----------
sha = hashlib.sha256(open(BYTE_CKPT, "rb").read()).hexdigest()
manifest = {
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "subject": "identity_v2 (released voxyne)",
    "byte_checkpoint": BYTE_CKPT,
    "byte_checkpoint_sha256": sha,
    "byte_params": int(sum(p.numel() for p in mb.parameters())),
    "seed": SEED, "device": DEV,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "python": platform.python_version(), "torch": torch.__version__,
    "corpus": "Salesforce/wikitext wikitext-2-raw-v1 test (full)",
    "corpus_bytes": TEST_BYTES,
    "metric": "bits per byte; pct = increase vs each model's own fp baseline",
    "gptq_calibration": "wikitext-2-raw TRAIN split (disjoint from eval test) -- no leakage",
    "quant": {"group_size": GROUP},
    "notes": "byte and token models evaluated on the same text; bpb normalizes token NLL by bytes.",
}
try:
    import transformers
    manifest["transformers"] = transformers.__version__
    import gptqmodel
    manifest["gptqmodel"] = gptqmodel.__version__
except Exception:
    pass

json.dump(results, open(f"{OUT}/results_v2.json", "w"), indent=2)
json.dump(manifest, open(f"{OUT}/manifest_v2.json", "w"), indent=2)
print("BENCHMARK V2 DONE", flush=True)
print(json.dumps(results, indent=2), flush=True)
