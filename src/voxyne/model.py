"""Byte-level RahuKetu transformer for Voxyne.

A decoder-only Transformer backbone over a byte-level zero-free representation.
It uses standard building blocks: causal self-attention, learned positions,
pre-norm, and GELU MLP. It adds the RahuKetu sigma_K control channel and an
out_sigma head. It does not use a BPE or GPT tokenizer stack.

Shape: d768 / 18L / 12h / FFN 4x, about 128.8M parameters.
Runtime path: SDPA / FlashAttention and KV cache.

Per position:
    digit in [1, 256]      byte identity. byte+1, with 0 as pad
    aux   in R^aux_dim     sigma_K (role) direction channel
Input:  x = ByteEmbed[digit] (+ AuxProj(aux)) + PosEmbed[pos]
Heads:  out_digit -> 256 (next byte), out_sigma -> 2 (next sigma_K), recon (optional)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x, past_kv=None, use_cache=False):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_kv = (k, v) if use_cache else None
        causal = past_kv is None
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), new_kv


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, past_kv=None, use_cache=False):
        a, new_kv = self.attn(self.ln1(x), past_kv, use_cache)
        x = x + self.drop(a)
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x, new_kv


class VoxyneLM(nn.Module):
    def __init__(
        self,
        alphabet_size: int = 256,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 18,
        max_len: int = 1024,
        aux_dim: int = 1,
        dropout: float = 0.1,
        use_aux: bool = True,
        byte_embed_bottleneck: int = 0,
        recon: bool = False,
    ):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.d_model = d_model
        self.max_len = max_len
        self.aux_dim = aux_dim
        self.use_aux = use_aux

        if byte_embed_bottleneck and byte_embed_bottleneck > 0:
            self.byte_embed = nn.Embedding(alphabet_size + 1, byte_embed_bottleneck, padding_idx=0)
            self.byte_up = nn.Linear(byte_embed_bottleneck, d_model, bias=False)
        else:
            self.byte_embed = nn.Embedding(alphabet_size + 1, d_model, padding_idx=0)
            self.byte_up = None
        self.aux_proj = nn.Linear(aux_dim, d_model, bias=False) if use_aux else None
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([Block(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

        self.out_digit = nn.Linear(d_model, alphabet_size)
        self.out_sigma = nn.Linear(d_model, 2)
        self.recon_head = nn.Linear(aux_dim, alphabet_size) if (recon and use_aux) else None

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, digits, aux=None, past_key_values=None, use_cache=False):
        B, T = digits.shape
        past_len = past_key_values[0][0].shape[2] if past_key_values else 0
        if past_len + T > self.max_len:
            raise ValueError(f"sequence length {past_len + T} exceeds max_len {self.max_len}")

        x = self.byte_embed(digits)
        if self.byte_up is not None:
            x = self.byte_up(x)
        recon_logits = None
        if self.use_aux:
            if aux is None:
                raise ValueError("use_aux=True but aux not provided")
            x = x + self.aux_proj(aux)
            if self.recon_head is not None:
                recon_logits = self.recon_head(aux)
        pos = torch.arange(past_len, past_len + T, device=digits.device).unsqueeze(0).expand(B, T)
        x = self.drop(x + self.pos_embed(pos))

        new_past = [] if use_cache else None
        for i, blk in enumerate(self.blocks):
            pkv = past_key_values[i] if past_key_values is not None else None
            x, kv = blk(x, pkv, use_cache)
            if use_cache:
                new_past.append(kv)
        x = self.norm(x)

        out = (self.out_digit(x), self.out_sigma(x), recon_logits)
        if use_cache:
            return out + (new_past,)
        return out
