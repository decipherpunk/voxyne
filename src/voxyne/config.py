"""VoxyneLM configuration for architecture and encoder settings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VoxyneConfig:
    # architecture (full base model ~128.8M params)
    alphabet_size: int = 256
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 18
    model_max_len: int = 1024
    dropout: float = 0.1

    # aux / RahuKetu channels. The released path is sigma_K-only.
    use_aux: bool = True
    use_sigma_k: bool = True
    use_sigma_r: bool = False
    use_features: bool = False  # True needs a rahuketu build with a features module
    polar_depth: int = 8        # only used when use_features=True
    shred_width: int = 8
    byte_embed_bottleneck: int = 0

    @property
    def inference_max_len(self) -> int:
        return self.model_max_len


def smoke_config() -> "VoxyneConfig":
    """Tiny CPU-friendly config for smoke tests."""
    return VoxyneConfig(d_model=128, n_heads=4, n_layers=2, model_max_len=64, dropout=0.0)
