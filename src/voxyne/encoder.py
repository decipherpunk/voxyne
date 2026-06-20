"""RahuKetuEncoder for byte digits and auxiliary channels.

The aux tensor carries sigma_K, the role and direction channel. It provides
information that the bytes themselves do not contain, such as role when markers
are stripped. This is the released path when use_features=False. Optional
recursive-polar byte features require a rahuketu build with a `features` module
and are reserved for ablations.
"""

from __future__ import annotations

import torch
from torch import nn


class RahuKetuEncoder(nn.Module):
    def __init__(
        self,
        alphabet_size: int = 256,
        polar_depth: int = 8,
        shred_width: int = 8,
        use_sigma_k: bool = True,
        use_sigma_r: bool = False,
        use_features: bool = False,
    ):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.use_sigma_k = use_sigma_k
        self.use_sigma_r = use_sigma_r
        self.use_features = use_features

        if use_features:
            import numpy as np

            try:
                from rahuketu import features
            except ImportError as e:
                raise ImportError(
                    "use_features=True needs a rahuketu build that provides recursive-polar "
                    "byte features through rahuketu.features. This is not available in v0.1"
                ) from e
            table = features.all_byte_feature_table(polar_depth, shred_width)
            self.feature_dim = int(table.shape[1])
            self.register_buffer(
                "feature_table", torch.from_numpy(table.astype(np.float32)), persistent=False
            )
        else:
            self.feature_dim = 0

        self.aux_dim = self.feature_dim + int(use_sigma_r) + int(use_sigma_k)
        if self.aux_dim == 0:
            raise ValueError("aux has no channels: enable sigma_k, sigma_r, or features")

    def forward(self, digits, sigma_K=None, sigma_R=None):
        if int(digits.min()) < 1 or int(digits.max()) > self.alphabet_size:
            raise ValueError("digits must be in [1, alphabet_size]")
        parts = []
        if self.use_features:
            parts.append(self.feature_table[digits - 1])
        if self.use_sigma_r:
            if sigma_R is None:
                sigma_R = torch.ones_like(digits, dtype=torch.float32)
            parts.append(sigma_R.unsqueeze(-1).float())
        if self.use_sigma_k:
            if sigma_K is None:
                sigma_K = torch.ones_like(digits, dtype=torch.float32)
            parts.append(sigma_K.unsqueeze(-1).float())
        return digits, torch.cat(parts, dim=-1)
