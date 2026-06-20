# Experiments

Reproducibility artifacts for the paper "Calibration-Free Int4 Robustness in a
Byte-Level Conversational Model".

## Environment

NVIDIA A40, Python 3.12, PyTorch 2.8.0+cu128, Transformers 5.12.1, GPTQModel 7.1.0,
seed 0. Subject checkpoint: Voxyne identity_v2,
SHA256 e8864b5130f7d517f8bab553b879646bfc284db7bb6b26d72ab8331a7a94247a,
released at https://huggingface.co/decipherpunk/voxyne.

## Metric

Bits per byte (bpb) is the negative log likelihood in nats divided by the byte count
and by ln 2, on the full WikiText-2 raw test corpus (1,293,436 bytes). Degradation is
the percent increase versus each model's own full-precision baseline. GPTQ is calibrated
on the WikiText-2 train split, disjoint from the eval test split.

## Artifacts (used by the paper)

- artifacts/results_v2.json        bits-per-byte robustness for byte and token models (naive int4, asym, table-fp16, GPTQ, int2 ablation), plus runtime
- artifacts/manifest_v2.json       run provenance: seed, hardware, library versions, corpus, checkpoint SHA256, GPTQ calibration split
- artifacts/generation_benchmark.json  Voxyne fp32/int8/int4 generation regression, 31 prompts, repetition, garbled, identity
- artifacts/coherence_compare.json     Voxyne vs SmolLM2-135M-Instruct coherence under quantization, 12 prompts

## Scripts

- benchmark_v2.py        bits-per-byte benchmark, produces results_v2.json and manifest_v2.json
- gen_benchmark.py       Voxyne generation regression, produces generation_benchmark.json
- coherence_compare.py   Voxyne vs token chat baseline, produces coherence_compare.json
- export_voxyne.py       exports the released checkpoint to bf16 and ONNX fp32/int8/int4

## Preliminary run (superseded)

artifacts/results.json and artifacts/manifest.json, with run_all_claims.py, are an
earlier 5-passage probe. They are kept for history. The paper uses the v2 artifacts above,
which use the full corpus, bits per byte, and a train-calibrated GPTQ baseline.
