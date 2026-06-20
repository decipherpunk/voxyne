"""Export the voxyne release (identity_v2 step-500): bf16 PT base + ONNX fp32/int8/int4.
int4 is the default deployable. Verifies the int4 ONNX still chats with identity."""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
from voxyne_pipeline.model import VoxyneLM
from voxyne_pipeline.encoder import RahuKetuEncoder

CKPT = "/workspace/runs/identity_v2/voxyne-best.pt"
OUT = "/workspace/release"
os.makedirs(OUT, exist_ok=True)
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = ck["config"]


def cg(k, d):
    return getattr(cfg, k) if hasattr(cfg, k) else (cfg.get(k, d) if isinstance(cfg, dict) else d)


D, NH, NL, ML = cg("d_model", 768), cg("n_heads", 12), cg("n_layers", 18), cg("model_max_len", 1024)
HD = D // NH
m = VoxyneLM(alphabet_size=256, d_model=D, n_heads=NH, n_layers=NL, max_len=ML,
             aux_dim=1, use_aux=True).eval()
m.load_state_dict(ck["model_state"])
enc = RahuKetuEncoder(alphabet_size=256, use_sigma_k=True, use_sigma_r=False,
                      use_features=False).eval()

# 1. bf16 PT base
state_bf16 = {k: v.to(torch.bfloat16) for k, v in m.state_dict().items()}
cfg_out = {"alphabet_size": 256, "d_model": D, "n_heads": NH, "n_layers": NL,
           "model_max_len": ML, "aux_dim": 1, "use_aux": True, "use_sigma_k": True,
           "use_features": False}
torch.save({"model_state": state_bf16, "config": cfg_out}, f"{OUT}/voxyne-v0.1.pt")
print("PT base (bf16):", round(os.path.getsize(f"{OUT}/voxyne-v0.1.pt") / 1e6, 1), "MB", flush=True)


# 2. ONNX decode-step graph
class Step(nn.Module):
    def __init__(self, mm):
        super().__init__()
        self.m = mm

    def forward(self, h, pk, pv):
        newk, newv = [], []
        for li, blk in enumerate(self.m.blocks):
            hn = blk.ln1(h)
            qkv = blk.attn.qkv(hn).view(1, 1, 3, NH, HD).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            K = torch.cat([pk[li], k], dim=2)
            V = torch.cat([pv[li], v], dim=2)
            y = F.scaled_dot_product_attention(q, K, V)
            h = h + blk.attn.proj(y.transpose(1, 2).reshape(1, 1, D))
            h = h + blk.mlp(blk.ln2(h))
            newk.append(K)
            newv.append(V)
        return self.m.out_digit(self.m.norm(h)), torch.stack(newk), torch.stack(newv)


def embed(digit, sigma, pos):
    x = torch.tensor([[digit + 1]])
    s = torch.tensor([[float(sigma)]])
    _, aux = enc(x, sigma_K=s)
    h = m.byte_embed(x)
    if m.byte_up is not None:
        h = m.byte_up(h)
    if m.use_aux:
        h = h + m.aux_proj(aux)
    return (h + m.pos_embed(torch.tensor([[pos]]))).detach()


FP32, INT8, INT4 = f"{OUT}/voxyne-fp32.onnx", f"{OUT}/voxyne-int8.onnx", f"{OUT}/voxyne-int4.onnx"
step = Step(m).eval()
h0 = embed(66, 1, 0)
pk0 = torch.zeros(NL, 1, NH, 1, HD)
pv0 = torch.zeros(NL, 1, NH, 1, HD)
torch.onnx.export(step, (h0, pk0, pv0), FP32, opset_version=17, dynamo=False,
                  input_names=["h", "pk", "pv"], output_names=["logits", "npk", "npv"],
                  dynamic_axes={"pk": {3: "T"}, "pv": {3: "T"}, "npk": {3: "T1"}, "npv": {3: "T1"}})
quantize_dynamic(FP32, INT8, weight_type=QuantType.QInt8)
q = MatMulNBitsQuantizer(onnx.load(FP32), bits=4, block_size=32, is_symmetric=True)
q.process()
om = q.model
for sv in (lambda: om.save_model_to_file(INT4, False), lambda: onnx.save(om, INT4),
           lambda: __import__("onnx_ir").save(om, INT4), lambda: om.save(INT4)):
    try:
        sv()
        break
    except Exception:
        pass
for f in (FP32, INT8, INT4):
    print(os.path.basename(f), round(os.path.getsize(f) / 1e6, 1), "MB", flush=True)


def gen(sess, prompt, max_new=60):
    pre = b"\x01bos\x02\x01user\x02" + prompt.encode() + b"\x01endturn\x02\x01assistant\x02"
    sig = [-1] * (len(pre) - len(b"\x01assistant\x02")) + [1] * len(b"\x01assistant\x02")
    pk = np.zeros((NL, 1, NH, 0, HD), np.float32)
    pv = pk.copy()
    logits, pos = None, 0
    for b, sg in zip(pre, sig):
        logits, pk, pv = sess.run(None, {"h": embed(b, sg, pos).numpy(), "pk": pk, "pv": pv})
        pos += 1
    out = bytearray()
    while len(out) < max_new and pos < 256:
        nb = int(logits[0, -1].argmax())
        out.append(nb)
        if out.endswith(b"\x01endturn\x02") or out.endswith(b"\x01eos\x02"):
            break
        logits, pk, pv = sess.run(None, {"h": embed(nb, 1, pos).numpy(), "pk": pk, "pv": pv})
        pos += 1
    return bytes(out).split(b"\x01")[0].decode("utf-8", "replace")


so = ort.SessionOptions()
so.intra_op_num_threads = 8
s4 = ort.InferenceSession(INT4, so, providers=["CPUExecutionProvider"])
for p in ["who are you?", "what is your name?", "hi"]:
    print(f"int4 verify  {p!r} -> {gen(s4, p)!r}", flush=True)
print("EXPORT DONE -> /workspace/release", flush=True)
