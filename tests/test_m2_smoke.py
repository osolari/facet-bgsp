"""Smoke tests for the M2 monosemanticity driver.

Pin the pure-numpy helpers (mean-pairwise-cosine, top-K index selection) ,
no transformers / no model download, so CI stays model-free.
"""

from __future__ import annotations

import numpy as np

from experiments import m2_monosemanticity as m2


def test_coherence_perfect_on_identical_embeddings() -> None:
    emb = np.tile(np.array([[1.0, 0.0, 0.0]]), (5, 1))
    assert m2._mean_pairwise_cosine(emb) == 1.0


def test_coherence_zero_on_orthogonal_embeddings() -> None:
    emb = np.eye(4)[:3]  # three mutually orthogonal unit vectors
    assert abs(m2._mean_pairwise_cosine(emb)) < 1e-12


def test_coherence_is_in_interval() -> None:
    rng = np.random.default_rng(0)
    for _ in range(5):
        emb = rng.normal(size=(10, 8))
        val = m2._mean_pairwise_cosine(emb)
        assert -1.0 <= val <= 1.0, val


def test_top_k_token_indices_shape() -> None:
    rng = np.random.default_rng(1)
    d, N, m, k = 20, 200, 6, 10
    H = rng.normal(size=(d, N))
    comps = rng.normal(size=(d, m))
    idx = m2._top_k_token_indices_per_component(comps, H, k)
    assert idx.shape == (m, k)
    # Entries must be valid token column indices.
    assert idx.min() >= 0
    assert idx.max() < N


def test_facet_fallback_returns_shape() -> None:
    rng = np.random.default_rng(2)
    H = rng.normal(size=(10, 50))
    comps, _ = m2._facet_components(H, m=3, gammas=[0.1, 0.3], seed=0)
    assert comps.shape == (10, 3)
