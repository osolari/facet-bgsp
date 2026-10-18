"""Smoke tests for the M6 Pitprops sparse-PCA benchmark driver."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import yaml

from experiments import m6_pitprops as m6
from experiments._pitprops import pitprops_correlation, pitprops_data_matrix


def test_pitprops_correlation_is_symmetric_pd() -> None:
    R = pitprops_correlation()
    assert R.shape == (13, 13)
    np.testing.assert_allclose(R, R.T, atol=1e-12)
    evals = np.linalg.eigvalsh(R)
    assert evals.min() > 0, evals.min()


def test_pitprops_data_sample_covariance_approaches_correlation() -> None:
    A = pitprops_data_matrix(seed=0, n_samples=10000)
    assert A.shape == (13, 10000)
    S = (A @ A.T) / (A.shape[1] - 1)
    R = pitprops_correlation()
    # Monte Carlo tolerance at 10k samples
    np.testing.assert_allclose(S, R, atol=0.05)


def test_adjusted_ev_uses_subspace_projection_not_rayleigh_sum() -> None:
    """Shen-Huang adjusted EV must drop below the raw-Rayleigh sum when the
    sparse loadings have overlapping support (double-counting)."""
    p = 10
    cov = np.eye(p) + 0.5 * np.ones((p, p))  # positive, non-identity
    # Two overlapping sparse loadings (not orthogonal in variable space)
    loadings = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        + [[0.0, 0.0]] * (p - 4),
        dtype=float,
    )
    nnz, adj_ev = m6._loading_metrics(loadings, cov)
    assert nnz == 6
    # Raw-Rayleigh sum would be strictly greater; adjusted EV is bounded by
    # the subspace trace.
    v0 = loadings[:, 0] / np.linalg.norm(loadings[:, 0])
    v1 = loadings[:, 1] / np.linalg.norm(loadings[:, 1])
    raw_sum = float(v0 @ cov @ v0 + v1 @ cov @ v1)
    assert adj_ev < raw_sum
    assert adj_ev > 0.0


def test_driver_schema_and_dense_pca_point(tmp_path: Path) -> None:
    cfg = {
        "name": "m6_smoke",
        "seed": 0,
        "data": {"n_samples": 180},
        "decomposition": {"m": 3},
        "facet": {"gammas_sq": [0.1, 0.5], "gammas_sum": [0.1, 0.5]},
        "sparse_pca": {"alphas": [0.5, 2.0]},
        "output": {"csv": "m6.csv", "figure": "m6.png"},
    }
    cfg_path = tmp_path / "m6.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    rc = m6.main(["--config", str(cfg_path), "--output-dir", str(tmp_path), "--no-figure"])
    assert rc == 0
    rows = list(csv.DictReader((tmp_path / "m6.csv").open()))
    methods = {r["method"] for r in rows}
    assert methods == {"pca", "facet_asq", "facet_asum", "sparse_pca_zou"}
    # Dense PCA row should be present exactly once.
    pca_rows = [r for r in rows if r["method"] == "pca"]
    assert len(pca_rows) == 1
    assert int(pca_rows[0]["total_nnz"]) <= 13 * 3  # at most p*m
