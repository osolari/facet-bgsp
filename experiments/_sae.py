"""Minimal tied-decoder sparse autoencoder for M4.

Standard interpretability recipe (Anthropic 2023/2024):
    encoder: h -> z = ReLU(W_e h + b_e)
    decoder: z -> h_hat = W_d z + b_d
    loss   : ||h - h_hat||^2 + lambda * ||z||_1

Tied-decoder means ``W_d = W_e^T``. Dict columns are unit-normalized.

``torch`` is imported lazily inside the training functions so this module
imports without ``facet[torch]`` installed; the smoke tests that need the
trained SAE call ``pytest.importorskip('torch')`` first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SAEConfig:
    d_in: int
    dict_size: int
    l1_coef: float
    lr: float
    steps: int
    batch_size: int
    seed: int
    tie_weights: bool = True


def _build_tied_sae(cfg: SAEConfig) -> Any:
    """Construct the torch module. Deferred so this file imports without torch."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class TiedSAE(nn.Module):
        def __init__(self, config: SAEConfig) -> None:
            super().__init__()
            self.cfg = config
            torch.manual_seed(config.seed)
            self.W_e = nn.Parameter(
                torch.randn(config.dict_size, config.d_in) * (1.0 / config.d_in**0.5)
            )
            self.b_e = nn.Parameter(torch.zeros(config.dict_size))
            self.b_d = nn.Parameter(torch.zeros(config.d_in))
            if not config.tie_weights:
                self.W_d = nn.Parameter(
                    torch.randn(config.d_in, config.dict_size) * (1.0 / config.dict_size**0.5)
                )

        def encode(self, h: Any) -> Any:
            return F.relu(h @ self.W_e.T + self.b_e)

        def decode(self, z: Any) -> Any:
            W = self.W_e if self.cfg.tie_weights else self.W_d.T
            return z @ W + self.b_d

        def forward(self, h: Any) -> tuple[Any, Any]:
            z = self.encode(h)
            return self.decode(z), z

        @torch.no_grad()
        def unit_normalize_dict(self) -> None:
            norms = self.W_e.norm(dim=1, keepdim=True).clamp_min(1e-8)
            self.W_e.data = self.W_e.data / norms

    return TiedSAE(cfg)


def train_sae(H: np.ndarray, cfg: SAEConfig) -> tuple[Any, dict[str, float]]:
    """Train on activations ``H`` of shape ``(d_in, N)``; return the model and final metrics."""
    import torch

    model = _build_tied_sae(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    H_t = torch.from_numpy(H.T.astype(np.float32))
    N = H_t.shape[0]
    rng = np.random.default_rng(cfg.seed + 1)
    for _step in range(cfg.steps):
        idx = rng.integers(0, N, size=cfg.batch_size)
        batch = H_t[idx]
        recon, z = model(batch)
        mse = (recon - batch).pow(2).mean()
        l1 = z.abs().mean()
        loss = mse + cfg.l1_coef * l1
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.unit_normalize_dict()

    with torch.no_grad():
        recon_full, z_full = model(H_t)
        final_mse = float((recon_full - H_t).pow(2).mean())
        final_l0 = float((z_full > 0).float().mean())
    return model, {"final_mse": final_mse, "final_l0_frac": final_l0}


def dict_directions(model: Any) -> np.ndarray:
    """Return the dictionary columns as a ``(d_in, dict_size)`` numpy array."""
    return model.W_e.detach().cpu().numpy().T.astype(np.float64)
