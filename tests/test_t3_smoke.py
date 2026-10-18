"""Smoke tests for the Tier-3 LLM-activations driver.

Pins the pure-numpy helpers (reconstruction R^2, score sparsity, baselines)
without requiring the ``transformers`` library or an actual model download
,  CI stays model-free. The end-to-end GPT-2 run is exercised manually via
``configs/t3_llm_activations.yaml``.
"""

from __future__ import annotations

import numpy as np

from experiments import t3_llm_activations as t3


def _planted_activations(rng: np.random.Generator, d: int, N: int, m_true: int) -> np.ndarray:
    directions = np.linalg.qr(rng.normal(size=(d, m_true)))[0]
    scores = rng.normal(size=(m_true, N))
    noise = rng.normal(scale=0.05, size=(d, N))
    return directions @ scores + noise


def test_reconstruction_r2_matches_pca_on_rank_m_data() -> None:
    rng = np.random.default_rng(0)
    d, N, m = 32, 400, 8
    H = _planted_activations(rng, d=d, N=N, m_true=m)
    pca = t3._pca_components(H, m)
    r2 = t3._reconstruction_r2(H, pca)
    # PCA on rank-m + small noise should recover > 99% of the variance.
    assert r2 > 0.99, r2


def test_reconstruction_r2_random_baseline_is_near_m_over_d() -> None:
    rng = np.random.default_rng(1)
    d, N, m = 48, 500, 6
    H = rng.normal(size=(d, N))
    rand = t3._random_orthogonal(d, m, seed=2)
    r2 = t3._reconstruction_r2(H, rand)
    # For isotropic noise a random m-plane captures ~ m/d of the energy.
    assert abs(r2 - m / d) < 0.05, (r2, m / d)


def test_hoyer_in_unit_interval() -> None:
    rng = np.random.default_rng(2)
    for _ in range(10):
        z = rng.normal(size=200)
        hoy = t3._hoyer(z)
        assert 0.0 <= hoy <= 1.0, hoy


def test_kurtosis_matches_scipy_for_normal() -> None:
    rng = np.random.default_rng(3)
    z = rng.normal(size=5000)
    kurt = t3._kurtosis(z)
    # Fisher kurtosis of a standard normal is 3; we're close within sampling noise.
    assert abs(kurt - 3.0) < 0.3, kurt


def test_per_component_metrics_shape_and_range() -> None:
    rng = np.random.default_rng(4)
    d, N, m = 16, 300, 5
    H = _planted_activations(rng, d=d, N=N, m_true=m)
    comps = t3._pca_components(H, m)
    hoy, kurt = t3._per_component_metrics(comps, H)
    assert 0.0 <= hoy <= 1.0
    assert kurt > 0.0


def test_top_components_by_score_variance_selects_k_cols() -> None:
    rng = np.random.default_rng(5)
    d, N, m = 16, 150, 6
    H = _planted_activations(rng, d=d, N=N, m_true=m)
    comps = t3._pca_components(H, m)
    top3 = t3._top_components_by_score_variance(comps, H, k=3)
    assert top3.shape == (d, 3)


def test_facet_components_shape_regardless_of_outcome() -> None:
    # The driver must always return a (d, m) frame, either a successful fit
    # or a zero-matrix fallback, never raise, so the comparison bar always
    # has a numerical height even if every candidate gamma stalls.
    rng = np.random.default_rng(0)
    H = rng.normal(size=(10, 50))
    comps, _ = t3._facet_components(H, m=4, gammas=[0.1, 0.3], seed=0)
    assert comps.shape == (10, 4)
