"""Elhage-style toy superposition model (helper for Tier-2).

A compressed autoencoder with hidden dimension ``d`` smaller than the number
of input features ``n`` is forced to represent ``n`` features in ``d``
directions, provided inputs are sparse (``k`` active features per sample).
After training, hidden activations ``H in R^(d x N)`` encode the planted
sparse features. The ground truth is the decoder matrix ``W in R^(d x n)``:
column ``i`` is the planted direction for feature ``i``.

Model:  h = W x,   y_hat = ReLU(W^T h + b)
Loss:   sum_i I_i * (x_i - y_hat_i)^2   (importance-weighted MSE)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SuperpositionConfig:
    n: int
    d: int
    sparsity: float
    importance_decay: float
    batch_size: int
    steps: int
    lr: float
    seed: int


def _sample_inputs(rng: np.random.Generator, n: int, batch: int, sparsity: float) -> np.ndarray:
    mask = (rng.random((batch, n)) < sparsity).astype(np.float32)
    values = rng.uniform(0.0, 1.0, size=(batch, n)).astype(np.float32)
    return mask * values


def _importance(n: int, decay: float) -> np.ndarray:
    return np.array([decay**i for i in range(n)], dtype=np.float32)


def train_superposition(cfg: SuperpositionConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train the toy and return ``(W, b, planted_codes)``.

    ``W`` has shape ``(d, n)``, ``b`` has shape ``(n,)``, and ``planted_codes``
    is an ``(n, N)`` array of fresh sparse inputs sampled *after* training.
    The hidden activations used by Tier-2 are then ``H = W @ planted_codes``.
    Returning the codes (not the activations) lets the caller recompute
    activations deterministically from ``W`` and provides the ground-truth
    sparse support each FACET component is supposed to recover.
    """
    import torch

    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    W = torch.nn.Parameter(torch.randn(cfg.d, cfg.n) * 0.1)
    b = torch.nn.Parameter(torch.zeros(cfg.n))
    importance = torch.from_numpy(_importance(cfg.n, cfg.importance_decay))

    opt = torch.optim.Adam([W, b], lr=cfg.lr)
    for _step in range(cfg.steps):
        x_np = _sample_inputs(rng, cfg.n, cfg.batch_size, cfg.sparsity)
        x = torch.from_numpy(x_np)
        h = x @ W.T  # (B, d)
        y = torch.relu(h @ W + b)
        err = (x - y).pow(2) * importance
        loss = err.sum(dim=1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    W_np = W.detach().cpu().numpy().astype(np.float64)
    b_np = b.detach().cpu().numpy().astype(np.float64)

    # Fresh evaluation codes; one sparse input per column.
    eval_N = cfg.n * 20
    codes = _sample_inputs(rng, cfg.n, eval_N, cfg.sparsity).T.astype(np.float64)  # (n, N)
    return W_np, b_np, codes


def _normalize_cols(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=0, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return mat / norms


def feature_recovery_cosines(components: np.ndarray, planted_directions: np.ndarray) -> np.ndarray:
    """For each planted direction, return the max absolute cosine to any component.

    ``components`` is ``(d, m)`` and ``planted_directions`` is ``(d, n)``.
    This is the lenient metric: one component can "cover" multiple planted
    features. Use :func:`hungarian_recovery_cosines` for the strict
    bipartite-matched metric.
    """
    C = _normalize_cols(components)
    P = _normalize_cols(planted_directions)
    sims = np.abs(P.T @ C)  # (n, m)
    return sims.max(axis=1)


def hungarian_recovery_cosines(
    components: np.ndarray, planted_directions: np.ndarray
) -> np.ndarray:
    """Strict bipartite assignment: each component used at most once.

    Returns a length-``n`` array where entry ``i`` is ``|cos(w_i, x_{\\sigma(i)})|``
    for the Hungarian-optimal assignment ``\\sigma``. Any planted feature left
    unmatched (because ``m < n``) gets a cosine of ``0``.
    """
    from scipy.optimize import linear_sum_assignment

    C = _normalize_cols(components)
    P = _normalize_cols(planted_directions)
    sims = np.abs(P.T @ C)  # (n, m)
    n, m = sims.shape
    # Hungarian maximizes by minimizing the negated cost; pad to square.
    size = max(n, m)
    cost = np.zeros((size, size))
    cost[:n, :m] = -sims
    row_ind, col_ind = linear_sum_assignment(cost)
    out = np.zeros(n)
    for r, c in zip(row_ind, col_ind, strict=True):
        if r < n and c < m:
            out[r] = sims[r, c]
    return out


def unique_planted_directions(planted_directions: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    """Collapse antipodal / near-duplicate planted directions.

    Returns a ``(d, n_unique)`` matrix. Two columns are considered duplicates
    if ``|cos| > 1 - tol``.
    """
    P = _normalize_cols(planted_directions)
    kept: list[np.ndarray] = []
    for i in range(P.shape[1]):
        v = P[:, i]
        if all(abs(float(v @ u)) < 1.0 - tol for u in kept):
            kept.append(v)
    if not kept:
        return np.zeros((P.shape[0], 0))
    return np.stack(kept, axis=1)
