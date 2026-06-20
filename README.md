# voxyne

A tiny byte-level conversational language model with the RahuKetu `sigma_K`
control channel. Loads on CPU and runs offline.

> The **code** is Apache-2.0. The released **research weights** are hosted on
> Hugging Face under a separate non-commercial model license. See the model card.

## Project

Voxyne is a small byte-level conversational model package.

- **rahuketu** - the zero-free encoding framework behind the `sigma_K` control channel.
- **voxyne** - this repository contains the model code, marker format, `sigma_K`
  encoder, generation helpers, and tests. It does not include a training loop,
  retrieval system, mutable memory, or autonomous learning runtime.
- Released research weights and quantized artifacts are published separately on
  Hugging Face with their own model card and license.

## Install

```
pip install voxyne
```

## Usage

Smoke test with random weights:

```python
import torch
from voxyne import smoke_config, build

model, enc = build(smoke_config())
x = torch.tensor([[1, 2, 3, 4]])
_, aux = enc(x, sigma_K=torch.tensor([[1.0, 1.0, -1.0, 1.0]]))
print(model(x, aux)[0].shape)        # -> next-byte logits
```

To generate real responses, load a trained checkpoint. The package does not
bundle weights. Download `voxyne-v0.1.pt` from the Hugging Face model repo,
review the license there, and pass the local path to `load_weights`:

```python
from voxyne import VoxyneConfig, build, load_weights, generate

model, enc = build(VoxyneConfig())   # 128.8M base with sigma_K aux
load_weights(model, "/path/to/voxyne-v0.1.pt")
print(generate(model, enc, "hello", device="cpu"))
```

## Components

- `VoxyneConfig` - architecture + encoder settings.
- `RahuKetuEncoder` - byte digits -> `(digits, aux)` with the `sigma_K` channel.
- `VoxyneLM` - transformer with `out_digit`, `out_sigma`, and KV cache.
- `generate` / `generate_cached` - greedy or top-k decoding. Cached generation uses KV cache.
- `build(config)` - construct a matching `(model, encoder)` pair.

## Weights

The package ships the **architecture only**. No model weights are bundled in this
repository.

Released research weights and ONNX artifacts:

- Model repo: `https://huggingface.co/decipherpunk/voxyne`
- PyTorch checkpoint: `voxyne-v0.1.pt`
- ONNX artifacts: `voxyne-int4.onnx`, `voxyne-int8.onnx`, `voxyne-fp32.onnx`
- Weight license: CC BY-NC 4.0 / non-commercial, per the model card

The Apache-2.0 license in this repository covers code only. Weight files remain
under the model-card license.

## Tests

`pytest` covers the package API, tensor shapes, loading, and KV-cache path.
Model quality depends on the released weights.

## Provenance / AI assistance

Built by Ramakrishnan (ORCID 0009-0006-0905-7275). The idea, architecture, and
research direction are by Ramakrishnan. AI tools helped with tests, automation,
and QA.

## License

Apache-2.0 (code). See `LICENSE` and `NOTICE`. Weights are licensed separately on
the model card.
