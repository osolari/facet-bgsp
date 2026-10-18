"""Smoke tests for the M1 multi-layer sweep driver.

Pin the pure-numpy helpers (gap arithmetic + metric functions reused from T3)
without requiring an actual GPT-2 download; CI stays model-free.
"""

from __future__ import annotations

import numpy as np

from experiments import m1_multilayer as m1


def test_facet_components_shape_regardless_of_outcome() -> None:
    rng = np.random.default_rng(0)
    H = rng.normal(size=(16, 80))
    comps, _ = m1._facet_components(H, m=4, gammas=[0.1, 0.3], seed=1)
    assert comps.shape == (16, 4)


def test_module_reuses_t3_metric_values() -> None:
    # m1 imports _pca_components, _per_component_metrics, _reconstruction_r2 from t3.
    # Verify equivalent numerical behavior; the imports keep the definitions in one place.
    from experiments import t3_llm_activations as t3

    rng = np.random.default_rng(0)
    H = rng.normal(size=(12, 100))
    comps_m1 = m1._pca_components(H, 4)
    comps_t3 = t3._pca_components(H, 4)
    np.testing.assert_allclose(comps_m1, comps_t3)
    assert m1._reconstruction_r2(H, comps_m1) == t3._reconstruction_r2(H, comps_t3)


def test_pca_recovers_rank_m_on_planted_data() -> None:
    rng = np.random.default_rng(1)
    d, N, m_true = 32, 600, 6
    dirs = np.linalg.qr(rng.normal(size=(d, m_true)))[0]
    scores = rng.normal(size=(m_true, N))
    H = dirs @ scores + 0.02 * rng.normal(size=(d, N))
    pca = m1._pca_components(H, m_true)
    r2 = m1._reconstruction_r2(H, pca)
    assert r2 > 0.99, r2
