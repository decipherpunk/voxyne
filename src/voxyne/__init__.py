"""Voxyne: a tiny byte-level conversational language model with the RahuKetu
sigma_K control channel. Loads on CPU, runs offline."""

from .config import VoxyneConfig, smoke_config
from .encoder import RahuKetuEncoder
from .generate import build_prompt, generate, generate_cached
from .model import VoxyneLM

__version__ = "0.1.0"


def build(config: VoxyneConfig):
    """Construct (model, encoder) with matching aux_dim from a VoxyneConfig."""
    encoder = RahuKetuEncoder(
        alphabet_size=config.alphabet_size,
        polar_depth=config.polar_depth,
        shred_width=config.shred_width,
        use_sigma_k=config.use_sigma_k,
        use_sigma_r=config.use_sigma_r,
        use_features=config.use_features,
    )
    model = VoxyneLM(
        alphabet_size=config.alphabet_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        max_len=config.model_max_len,
        aux_dim=encoder.aux_dim,
        dropout=config.dropout,
        use_aux=config.use_aux,
        byte_embed_bottleneck=config.byte_embed_bottleneck,
    )
    return model, encoder


def load_weights(model, path, *, map_location="cpu", strict=False):
    """Load released VoxyneLM weights into `model` from a local checkpoint.

    Download the weights from the Hugging Face model repo first (see the README /
    model card). Accepts a raw state_dict or a {"model": state_dict, ...}
    checkpoint. strict=False tolerates auxiliary-head differences across releases.
    """
    import torch

    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict):
        for key in ("model_state", "model", "state_dict"):
            if key in state:
                state = state[key]
                break
    model.load_state_dict(state, strict=strict)
    return model


__all__ = [
    "VoxyneConfig",
    "smoke_config",
    "RahuKetuEncoder",
    "VoxyneLM",
    "build",
    "load_weights",
    "generate",
    "generate_cached",
    "build_prompt",
]
