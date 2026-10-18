"""Smoke tests for the M4 FACET-vs-SAE driver (pure-numpy / torch helpers)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from experiments import m4_facet_vs_sae as m4
from experiments._sae import SAEConfig, dict_directions, train_sae


def test_sae_dictionary_rows_are_unit_norm_after_training() -> None:
    rng = np.random.default_rng(0)
    H = rng.normal(size=(16, 400))
    model, _ = train_sae(
        H,
        SAEConfig(
            d_in=16,
            dict_size=8,
            l1_coef=0.01,
            lr=0.01,
            steps=50,
            batch_size=64,
            seed=0,
        ),
    )
    W = dict_directions(model)
    norms = np.linalg.norm(W, axis=0)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_sae_dictionary_shape() -> None:
    rng = np.random.default_rng(1)
    H = rng.normal(size=(12, 200))
    model, _ = train_sae(
        H,
        SAEConfig(
            d_in=12,
            dict_size=24,
            l1_coef=0.01,
            lr=0.01,
            steps=20,
            batch_size=32,
            seed=1,
        ),
    )
    W = dict_directions(model)
    assert W.shape == (12, 24)


def test_sae_training_reduces_reconstruction_mse() -> None:
    rng = np.random.default_rng(2)
    H = rng.normal(size=(10, 300))
    cfg = SAEConfig(
        d_in=10,
        dict_size=20,
        l1_coef=0.001,
        lr=0.01,
        steps=200,
        batch_size=64,
        seed=2,
    )
    _, metrics = train_sae(H, cfg)
    # Trained SAE should reconstruct the data to below the input variance baseline.
    assert metrics["final_mse"] < float(H.var())


def test_facet_components_fallback_returns_shape() -> None:
    rng = np.random.default_rng(3)
    H = rng.normal(size=(10, 50))
    comps, _ = m4._facet_components(H, m=3, gammas=[0.1, 0.3], seed=0)
    assert comps.shape == (10, 3)
