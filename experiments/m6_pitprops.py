"""M6, classical sparse-PCA benchmark on Jeffers (1967) Pitprops.

Pitprops is the canonical 13-variable sparse-PCA benchmark (Zou et al. 2006,
Journée et al. 2010). The data correlation matrix is positive definite; the
first six PCs explain roughly 87% of the variance. Sparse variants target
"interpretable" loadings with many exact zeros while losing little explained
variance.

This driver evaluates FACET A-SQ, FACET A-SUM, PCA, and sklearn's SparsePCA
on an explained-variance-vs-sparsity curve. For each component ``j``:

  - ``nnz_j``  = number of nonzero entries of the loading
  - ``ev_j``   = explained variance of the component on the empirical cov

Each method is run at several sparsity levels (γ grid for FACET; alpha grid
for SparsePCA). The headline figure plots the Pareto frontier of
(total nonzeros across m, total explained variance across m) for each
method.

See plan-2026-10-13T1908Z §M6.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses as dc
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _pitprops import pitprops_data_matrix  # noqa: E402

from facet.papers.block_sparse import BlockSparseEstimator  # noqa: E402
from facet.supports import soft_threshold  # noqa: E402


@dc.dataclass(frozen=True)
class SparseRow:
    method: str
    hyperparam: float  # gamma (FACET) or alpha (SparsePCA); nan for dense PCA
    total_nnz: int  # sum of nonzeros across the m loadings (variable space)
    total_ev: float  # sum of per-component variable-space Rayleigh v^T C v
    m: int


def _dense_pca_variable_loadings(X_data: np.ndarray, m: int) -> np.ndarray:
    """Classical dense PCA loadings in variable space.

    ``X_data`` is ``(n_samples, p_features)``; returns ``(p_features, m)``.
    """
    cov = (X_data.T @ X_data) / (X_data.shape[0] - 1)
    _, vecs = np.linalg.eigh(cov)
    return vecs[:, -m:]


def _facet_variable_loadings(
    X_data: np.ndarray, m: int, gamma: float, objective: str, seed: int
) -> np.ndarray:
    """FACET dual-loading z vectors in variable space.

    For a Zou-style sparse-PCA comparison we want sparsity in VARIABLE space
    (the ``p_features = 13`` axis for Pitprops). FACET's dual loading
    ``z_j = soft_threshold(A^T x_j, gamma) / ||.||`` is sparse by construction.
    To put ``z`` on the variable axis we feed FACET the data with samples on
    the row axis and variables on the column axis: ``A = X_data`` has shape
    ``(n_samples, p_features)``, so FACET fits ``X in R^(n_samples x m)``
    (dense Stiefel scores in sample space) and the dual loadings
    ``Soft(A^T X)`` live in R^(p_features x m), exactly what we want sparse.
    """
    est = BlockSparseEstimator(
        objective=objective,
        gamma=gamma,
        solver="safeguarded",
        max_iterations=400,
        tolerance=1e-10,
        refit="none",
        random_state=seed,
    )
    try:
        fit = est.fit(X_data, m=m)
    except RuntimeError:
        return np.zeros((X_data.shape[1], m))

    X_scores = np.asarray(fit.raw.factors)  # (n_samples, m)
    AtX = X_data.T @ X_scores  # (p_features, m)
    loadings = np.zeros_like(AtX)
    for j in range(m):
        z = soft_threshold(AtX[:, j], gamma)
        norm = float(np.linalg.norm(z))
        if norm > 1e-12:
            loadings[:, j] = z / norm
    return loadings


def _sparse_pca_variable_loadings(
    X_data: np.ndarray, m: int, alpha: float, seed: int
) -> np.ndarray:
    """sklearn SparsePCA components in variable space, returned as (p, m)."""
    from sklearn.decomposition import SparsePCA

    spca = SparsePCA(
        n_components=m,
        alpha=alpha,
        max_iter=500,
        tol=1e-6,
        random_state=seed,
    )
    spca.fit(X_data)
    return spca.components_.T  # (p_features, m)


def _loading_metrics(loadings: np.ndarray, cov: np.ndarray) -> tuple[int, float]:
    """Return (total_nnz, adjusted_explained_variance) in variable space.

    Uses the Shen & Huang (2008) adjusted EV: project the feature-space
    covariance onto the subspace spanned by the (non-orthogonal) sparse
    loadings via QR, then report ``tr(Q^T C Q)``. This avoids the
    double-counting pitfall of summing per-component Rayleigh quotients when
    sparse loadings overlap in support.

    ``nnz`` counts entries above 1e-6 (sklearn SparsePCA floors small
    coefficients exactly to zero; FACET's soft-threshold leaves zeros at
    machine precision).
    """
    nnz = int(np.sum(np.abs(loadings) > 1e-6))
    # Drop zero columns before QR.
    col_norms = np.linalg.norm(loadings, axis=0)
    nonzero = loadings[:, col_norms > 1e-12]
    if nonzero.shape[1] == 0:
        return nnz, 0.0
    q, _ = np.linalg.qr(nonzero)
    adjusted_ev = float(np.trace(q.T @ cov @ q))
    return nnz, adjusted_ev


def run(config: dict[str, Any]) -> list[SparseRow]:
    m = int(config["decomposition"]["m"])
    n_samples = int(config["data"]["n_samples"])
    seed = int(config["seed"])
    # pitprops_data_matrix returns (p_features, n_samples) by convention;
    # for a Zou-style benchmark we want (n_samples, p_features).
    X_data = pitprops_data_matrix(seed=seed, n_samples=n_samples).T  # (n_samples, 13)
    cov = (X_data.T @ X_data) / (X_data.shape[0] - 1)

    rows: list[SparseRow] = []

    # Dense PCA, a single point.
    pca = _dense_pca_variable_loadings(X_data, m)
    nnz, ev = _loading_metrics(pca, cov)
    rows.append(SparseRow("pca", float("nan"), nnz, ev, m))

    for gamma in config["facet"]["gammas_sq"]:
        comps = _facet_variable_loadings(X_data, m, float(gamma), "squared", seed)
        nnz, ev = _loading_metrics(comps, cov)
        rows.append(SparseRow("facet_asq", float(gamma), nnz, ev, m))

    for gamma in config["facet"]["gammas_sum"]:
        comps = _facet_variable_loadings(X_data, m, float(gamma), "sum_norm", seed)
        nnz, ev = _loading_metrics(comps, cov)
        rows.append(SparseRow("facet_asum", float(gamma), nnz, ev, m))

    for alpha in config["sparse_pca"]["alphas"]:
        comps = _sparse_pca_variable_loadings(X_data, m, float(alpha), seed)
        nnz, ev = _loading_metrics(comps, cov)
        rows.append(SparseRow("sparse_pca_zou", float(alpha), nnz, ev, m))

    return rows


def write_csv(rows: list[SparseRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dc.fields(SparseRow)])
        writer.writeheader()
        for r in rows:
            writer.writerow(dc.asdict(r))


def write_figure(rows: list[SparseRow], path: Path) -> None:
    from _style import apply_style, full_spines

    apply_style()
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame([dc.asdict(r) for r in rows])

    method_style = {
        "pca": {"color": "#1f77b4", "marker": "s", "label": "PCA (dense)"},
        "facet_asq": {"color": "#d62728", "marker": "o", "label": "FACET A-SQ"},
        "facet_asum": {"color": "#ff7f0e", "marker": "^", "label": "FACET A-SUM"},
        "sparse_pca_zou": {"color": "#2ca02c", "marker": "D", "label": "sklearn SparsePCA (Zou)"},
    }

    fig, ax = plt.subplots(1, 1, figsize=(8.6, 5.6), constrained_layout=True)

    for method, style in method_style.items():
        sub = df[df["method"] == method].sort_values("total_nnz")
        if sub.empty:
            continue
        ax.plot(
            sub["total_nnz"],
            sub["total_ev"],
            color=style["color"],
            marker=style["marker"],
            markersize=9,
            markeredgecolor="black",
            markeredgewidth=0.5,
            lw=1.6,
            alpha=0.9,
            label=style["label"],
        )
        if method != "pca":
            for hp, nnz, ev in zip(
                sub["hyperparam"], sub["total_nnz"], sub["total_ev"], strict=True
            ):
                ax.annotate(
                    f"{hp:.2g}",
                    xy=(nnz, ev),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    color=style["color"],
                    alpha=0.8,
                )

    m_val = int(df["m"].iloc[0]) if not df.empty else 0
    p_vars = 13  # Pitprops
    total_vars_m = p_vars * m_val
    ax.axvline(
        total_vars_m, color="black", lw=0.8, linestyle=":", label=f"dense fill ({total_vars_m} nnz)"
    )
    ax.set_xlabel("total nonzeros across all m components")
    ax.set_ylabel("adjusted explained variance (Shen--Huang)")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.4)
    full_spines(ax)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    rows = run(cfg)
    out = Path(args.output_dir)
    write_csv(rows, out / cfg["output"]["csv"])
    if not args.no_figure:
        write_figure(rows, out / cfg["output"]["figure"])
    print(f"wrote {len(rows)} rows across {len({r.method for r in rows})} methods")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
