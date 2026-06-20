import numpy as np
import pytest
import torch

from voxyne import (
    RahuKetuEncoder,
    VoxyneConfig,
    VoxyneLM,
    build,
    build_prompt,
    generate,
    generate_cached,
    load_weights,
    smoke_config,
)
from voxyne.generate import _sample
from voxyne.markers import ASSISTANT, BOS, ENDTURN, EOS, PAD, SYSTEM, USER, _tag, sigma_k_for_dialogue_bytes
from voxyne.model import Block, CausalSelfAttention


def test_config_defaults_and_smoke_config_are_consistent():
    cfg = VoxyneConfig()
    assert cfg.alphabet_size == 256
    assert cfg.d_model == 768
    assert cfg.n_heads == 12
    assert cfg.n_layers == 18
    assert cfg.inference_max_len == cfg.model_max_len
    smoke = smoke_config()
    assert smoke.d_model == 128
    assert smoke.n_heads == 4
    assert smoke.n_layers == 2
    assert smoke.model_max_len == 64
    assert smoke.dropout == 0.0


def test_markers_are_control_wrapped_and_unique():
    markers = [BOS, EOS, USER, ASSISTANT, SYSTEM, ENDTURN, PAD]
    assert len(set(markers)) == len(markers)
    for marker in markers:
        assert marker.startswith(b"\x01")
        assert marker.endswith(b"\x02")
    assert _tag("custom") == b"\x01custom\x02"
    with pytest.raises(ValueError):
        _tag("\x01bad")


def test_sigma_k_marks_user_payload_negative_and_markers_positive():
    data = BOS + USER + b"hi" + ENDTURN + ASSISTANT + b"ok" + ENDTURN
    sig = np.frombuffer(sigma_k_for_dialogue_bytes(data), dtype=np.int8)
    assert sig.shape == (len(data),)
    user_start = len(BOS) + len(USER)
    assert sig[user_start:user_start + 2].tolist() == [-1, -1]
    assistant_start = data.index(ASSISTANT) + len(ASSISTANT)
    assert sig[assistant_start:assistant_start + 2].tolist() == [1, 1]
    assert set(sig[: len(BOS)].tolist()) == {1}


def test_build_prompt_wraps_user_turn_and_builds_matching_sigma():
    prompt, sigma = build_prompt("hello")
    assert prompt.startswith(BOS + USER)
    assert prompt.endswith(ENDTURN + ASSISTANT)
    assert b"hello" in prompt
    assert sigma.shape == (len(prompt),)
    assert sigma.dtype == np.float32
    payload_start = len(BOS) + len(USER)
    assert sigma[payload_start:payload_start + 5].tolist() == [-1.0] * 5


def test_encoder_sigma_only_aux_default_and_explicit_values():
    enc = RahuKetuEncoder(use_sigma_k=True, use_sigma_r=False)
    x = torch.tensor([[1, 2, 256]])
    digits, aux = enc(x)
    direct_digits, direct_aux = enc.forward(x)
    assert torch.equal(digits, x)
    assert torch.equal(direct_digits, x)
    assert aux.shape == (1, 3, 1)
    assert torch.equal(aux, torch.ones(1, 3, 1))
    assert torch.equal(direct_aux, aux)
    sigma = torch.tensor([[1.0, -1.0, 1.0]])
    _, aux = enc(x, sigma_K=sigma)
    assert torch.equal(aux.squeeze(-1), sigma)


def test_encoder_can_include_sigma_r_and_rejects_empty_aux():
    enc = RahuKetuEncoder(use_sigma_k=True, use_sigma_r=True)
    x = torch.tensor([[1, 2]])
    _, aux = enc(x, sigma_K=torch.tensor([[1.0, -1.0]]), sigma_R=torch.tensor([[-1.0, 1.0]]))
    assert aux.shape == (1, 2, 2)
    assert aux[0, :, 0].tolist() == [-1.0, 1.0]
    assert aux[0, :, 1].tolist() == [1.0, -1.0]
    with pytest.raises(ValueError):
        RahuKetuEncoder(use_sigma_k=False, use_sigma_r=False)


def test_encoder_rejects_digits_outside_zero_free_range():
    enc = RahuKetuEncoder()
    with pytest.raises(ValueError):
        enc(torch.tensor([[0]]))
    with pytest.raises(ValueError):
        enc(torch.tensor([[257]]))


def test_build_constructs_model_and_encoder_with_matching_aux_dim():
    cfg = smoke_config()
    model, enc = build(cfg)
    assert isinstance(model, VoxyneLM)
    assert isinstance(enc, RahuKetuEncoder)
    assert model.aux_dim == enc.aux_dim
    assert model.max_len == cfg.model_max_len


def test_model_forward_shapes_sigma_head_and_recon_head():
    model = VoxyneLM(alphabet_size=256, d_model=64, n_heads=4, n_layers=2, max_len=16, aux_dim=2, dropout=0.0, recon=True)
    x = torch.tensor([[1, 2, 3]])
    aux = torch.ones(1, 3, 2)
    digit_logits, sigma_logits, recon_logits = model(x, aux)
    assert digit_logits.shape == (1, 3, 256)
    assert sigma_logits.shape == (1, 3, 2)
    assert recon_logits.shape == (1, 3, 256)
    assert model.param_count() > 0


def test_attention_and_block_forward_shapes():
    attn = CausalSelfAttention(d_model=32, n_heads=4, dropout=0.0)
    x = torch.randn(2, 3, 32)
    y, past = attn(x, use_cache=True)
    y_direct, past_direct = attn.forward(x, use_cache=True)
    assert y.shape == x.shape
    assert y_direct.shape == x.shape
    assert past[0].shape == (2, 4, 3, 8)
    assert past_direct[0].shape == (2, 4, 3, 8)
    block = Block(d_model=32, n_heads=4, dropout=0.0)
    y, past = block(x, use_cache=True)
    y_direct, past_direct = block.forward(x, use_cache=True)
    assert y.shape == x.shape
    assert y_direct.shape == x.shape
    assert past[0].shape == (2, 4, 3, 8)
    assert past_direct[0].shape == (2, 4, 3, 8)


def test_model_requires_aux_when_enabled_and_rejects_too_long_sequences():
    model = VoxyneLM(d_model=32, n_heads=4, n_layers=1, max_len=2, aux_dim=1, dropout=0.0)
    with pytest.raises(ValueError):
        model.forward(torch.tensor([[1]]))
    with pytest.raises(ValueError):
        model(torch.tensor([[1, 2, 3]]), torch.ones(1, 3, 1))


def test_model_supports_no_aux_and_byte_embedding_bottleneck():
    model = VoxyneLM(d_model=32, n_heads=4, n_layers=1, max_len=8, aux_dim=1, dropout=0.0, use_aux=False, byte_embed_bottleneck=8)
    digit_logits, sigma_logits, recon_logits = model(torch.tensor([[1, 2]]))
    assert digit_logits.shape == (1, 2, 256)
    assert sigma_logits.shape == (1, 2, 2)
    assert recon_logits is None


def test_model_cache_prefill_and_single_step_extend_sequence():
    model = VoxyneLM(d_model=32, n_heads=4, n_layers=2, max_len=8, aux_dim=1, dropout=0.0)
    x = torch.tensor([[1, 2]])
    aux = torch.ones(1, 2, 1)
    digit_logits, sigma_logits, recon_logits, past = model(x, aux, use_cache=True)
    assert digit_logits.shape == (1, 2, 256)
    assert sigma_logits.shape == (1, 2, 2)
    assert recon_logits is None
    assert len(past) == 2
    y, _, _, past = model(torch.tensor([[3]]), torch.ones(1, 1, 1), past_key_values=past, use_cache=True)
    assert y.shape == (1, 1, 256)
    assert past[0][0].shape[2] == 3


def test_load_weights_accepts_raw_and_wrapped_state_dicts(tmp_path):
    model = VoxyneLM(d_model=32, n_heads=4, n_layers=1, max_len=8, aux_dim=1, dropout=0.0)
    raw_path = tmp_path / "raw.pt"
    wrapped_path = tmp_path / "wrapped.pt"
    torch.save(model.state_dict(), raw_path)
    torch.save({"model": model.state_dict()}, wrapped_path)
    fresh = VoxyneLM(d_model=32, n_heads=4, n_layers=1, max_len=8, aux_dim=1, dropout=0.0)
    assert load_weights(fresh, raw_path) is fresh
    newer = VoxyneLM(d_model=32, n_heads=4, n_layers=1, max_len=8, aux_dim=1, dropout=0.0)
    assert load_weights(newer, wrapped_path) is newer


def test_sample_returns_argmax_for_zero_temperature():
    logits = torch.tensor([0.0, 1.0, 3.0, 2.0])
    assert _sample(logits, temperature=0.0, top_k=2) == 2
    assert _sample(logits, temperature=None, top_k=None) == 2


def test_generate_and_cached_generate_return_strings_and_match_greedy():
    cfg = smoke_config()
    model, enc = build(cfg)
    a = generate(model, enc, "hello", max_new=12, device="cpu", max_len=cfg.model_max_len)
    b = generate_cached(model, enc, "hello", max_new=12, device="cpu", max_len=cfg.model_max_len)
    assert isinstance(a, str)
    assert isinstance(b, str)
    assert a == b
