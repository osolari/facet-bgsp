"""M8: sparse risk factors on Fama-French 48 industry daily returns.

Downloads the Ken French 48 Industry Portfolios daily returns (public,
redistributable with attribution), filters to the modern era (2000-),
standardises per-industry, and fits FACET A-SQ, dense PCA, and sklearn
SparsePCA in variable (industry) space at m=6 components. The hero figure
shows (a) the top-K FACET components as per-industry loading bars coloured
by GICS-style sector grouping, and (b) the Pareto frontier of (total nnz,
Shen-Huang adjusted explained variance).
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

from _finance import SECTOR_PALETTE, FinanceData, download_ff48  # noqa: E402

from facet.papers.block_sparse import BlockSparseEstimator  # noqa: E402
from facet.supports import soft_threshold  # noqa: E402


@dc.dataclass(frozen=True)
class M8Row:
    method: str
    hyperparam: float
    total_nnz: int
    total_ev: float
    m: int
    n_industries: int
    n_days: int


def _standardize(returns: np.ndarray) -> np.ndarray:
    mu = returns.mean(axis=0, keepdims=True)
    sd = returns.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (returns - mu) / sd


def _dense_pca_loadings(X_data: np.ndarray, m: int) -> np.ndarray:
    cov = (X_data.T @ X_data) / (X_data.shape[0] - 1)
    _, vecs = np.linalg.eigh(cov)
    return vecs[:, -m:]


def _facet_variable_loadings(
    X_data: np.ndarray, m: int, gamma: float, objective: str, seed: int
) -> np.ndarray:
    est = BlockSparseEstimator(
        objective=objective,
        gamma=gamma,
        solver="safeguarded",
        max_iterations=600,
        tolerance=1e-10,
        refit="none",
        random_state=seed,
    )
    try:
        fit = est.fit(X_data, m=m)
    except RuntimeError:
        return np.zeros((X_data.shape[1], m))
    X_scores = np.asarray(fit.raw.factors)
    AtX = X_data.T @ X_scores
    out = np.zeros_like(AtX)
    for j in range(m):
        z = soft_threshold(AtX[:, j], gamma)
        norm = float(np.linalg.norm(z))
        if norm > 1e-12:
            out[:, j] = z / norm
    return out


def _sparse_pca_loadings(X_data: np.ndarray, m: int, alpha: float, seed: int) -> np.ndarray:
    from sklearn.decomposition import SparsePCA

    spca = SparsePCA(n_components=m, alpha=alpha, max_iter=400, tol=1e-6, random_state=seed)
    spca.fit(X_data)
    return spca.components_.T


def _loading_metrics(loadings: np.ndarray, cov: np.ndarray) -> tuple[int, float]:
    nnz = int(np.sum(np.abs(loadings) > 1e-6))
    col_norms = np.linalg.norm(loadings, axis=0)
    nonzero = loadings[:, col_norms > 1e-12]
    if nonzero.shape[1] == 0:
        return nnz, 0.0
    q, _ = np.linalg.qr(nonzero)
    return nnz, float(np.trace(q.T @ cov @ q))


def run(config: dict[str, Any]) -> tuple[list[M8Row], dict[str, Any]]:
    cache_path = str(config["data"].get("cache_path", ""))
    source_zip = str(config["data"].get("source_zip_path", ""))
    data: FinanceData = download_ff48(
        min_year=int(config["data"].get("min_year", 2000)),
        cache_path=cache_path or None,
        source_zip_path=source_zip or None,
    )
    X_std = _standardize(data.returns)
    cov = (X_std.T @ X_std) / (X_std.shape[0] - 1)
    m = int(config["decomposition"]["m"])
    seed = int(config["seed"])
    n_ind = len(data.industries)

    rows: list[M8Row] = []
    pca_loadings = _dense_pca_loadings(X_std, m)
    nnz, ev = _loading_metrics(pca_loadings, cov)
    rows.append(M8Row("pca", float("nan"), nnz, ev, m, n_ind, X_std.shape[0]))

    facet_loadings_by_gamma: dict[float, np.ndarray] = {}
    for gamma in config["facet"]["gammas_sq"]:
        g = float(gamma)
        comps = _facet_variable_loadings(X_std, m, g, "squared", seed)
        facet_loadings_by_gamma[g] = comps
        nnz, ev = _loading_metrics(comps, cov)
        rows.append(M8Row("facet_asq", g, nnz, ev, m, n_ind, X_std.shape[0]))

    for alpha in config["sparse_pca"]["alphas"]:
        a = float(alpha)
        comps = _sparse_pca_loadings(X_std, m, a, seed)
        nnz, ev = _loading_metrics(comps, cov)
        rows.append(M8Row("sparse_pca_zou", a, nnz, ev, m, n_ind, X_std.shape[0]))

    extras = {
        "data": data,
        "pca_loadings": pca_loadings,
        "facet_loadings_by_gamma": facet_loadings_by_gamma,
        "cov": cov,
        "hero_facet_gamma": float(config["facet"]["hero_gamma"]),
        "hero_top_k_components": int(config["figure"]["hero_top_k_components"]),
        "hero_top_k_industries_per_component": int(
            config["figure"]["hero_top_k_industries_per_component"]
        ),
    }
    return rows, extras


def write_csv(rows: list[M8Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dc.fields(M8Row)])
        writer.writeheader()
        for r in rows:
            writer.writerow(dc.asdict(r))


def write_hero_figure(rows: list[M8Row], extras: dict[str, Any], path: Path) -> None:
    from _style import apply_style, full_spines

    apply_style()
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    data: FinanceData = extras["data"]
    facet_loadings = extras["facet_loadings_by_gamma"][extras["hero_facet_gamma"]]
    top_k_comp = extras["hero_top_k_components"]
    top_k_ind = extras["hero_top_k_industries_per_component"]

    comp_mass = np.abs(facet_loadings).sum(axis=0)
    comp_order = np.argsort(-comp_mass)[:top_k_comp]

    fig = plt.figure(figsize=(14.0, 7.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.1])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, :])

    industries = np.array(data.industries)
    sector_of = np.array(data.sector_of)
    sectors = list(SECTOR_PALETTE.keys())

    strip_height = 0.78
    for row_idx, j in enumerate(comp_order):
        loadings = facet_loadings[:, j]
        order_by_abs = np.argsort(-np.abs(loadings))
        top_idx = order_by_abs[:top_k_ind]
        y_center = top_k_comp - 1 - row_idx
        for xi, ti in enumerate(top_idx):
            colour = SECTOR_PALETTE[sector_of[ti]]
            value = loadings[ti]
            x_center = xi + 0.5
            ax_a.bar(
                x_center,
                abs(value),
                width=0.9,
                bottom=y_center - strip_height / 2,
                color=colour,
                edgecolor="black",
                linewidth=0.4,
                alpha=0.9 if value >= 0 else 0.55,
            )
            ax_a.text(
                x_center,
                y_center - strip_height / 2 - 0.07,
                industries[ti],
                ha="center",
                va="top",
                fontsize=8,
                rotation=90,
                color="0.25",
            )
            if value < 0:
                ax_a.text(
                    x_center,
                    y_center + strip_height / 2 + 0.03,
                    "-",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    color=colour,
                    weight="bold",
                )
        ax_a.text(
            -0.6,
            y_center,
            f"comp. {row_idx + 1}",
            ha="right",
            va="center",
            fontsize=11,
            weight="bold",
        )

    ax_a.set_xlim(-1.2, top_k_ind)
    ax_a.set_ylim(-0.6, top_k_comp - 0.2)
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    ax_a.set_xlabel(
        "top-|loading| FF48 industries (bar height = |loading|; '-' marks negative loading)"
    )
    full_spines(ax_a)
    ax_a.text(
        0.005,
        0.98,
        "(a)",
        transform=ax_a.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        weight="bold",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.3"},
    )
    sector_handles = [
        mlines.Line2D(
            [],
            [],
            marker="s",
            linestyle="",
            color=SECTOR_PALETTE[s],
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=10,
            label=s,
        )
        for s in sectors
    ]
    ax_a.legend(
        handles=sector_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.35),
        ncol=4,
        fontsize=9,
        frameon=True,
        title="GICS sector",
        title_fontsize=10,
    )

    style = {
        "pca": ("PCA (dense)", "#1f77b4", "s"),
        "facet_asq": ("FACET A-SQ", "#d62728", "o"),
        "sparse_pca_zou": ("sklearn SparsePCA (Zou)", "#2ca02c", "D"),
    }
    for method, (label, colour, marker) in style.items():
        sub = sorted(
            (r for r in rows if r.method == method),
            key=lambda r: r.total_nnz,
        )
        if not sub:
            continue
        xs = [r.total_nnz for r in sub]
        ys = [r.total_ev for r in sub]
        ax_b.plot(
            xs,
            ys,
            color=colour,
            marker=marker,
            markersize=9,
            markeredgecolor="black",
            markeredgewidth=0.5,
            lw=1.6,
            alpha=0.9,
            label=label,
        )
    pca_ev = next((r.total_ev for r in rows if r.method == "pca"), None)
    if pca_ev is not None:
        ax_b.axhline(pca_ev, color="black", linestyle=":", lw=1.0, label="dense PCA ceiling")

    m_val = next((r.m for r in rows), 0)
    n_ind = next((r.n_industries for r in rows), 0)
    ax_b.axvline(
        m_val * n_ind,
        color="0.6",
        linestyle=":",
        lw=0.9,
        label=f"dense fill ({m_val * n_ind} nnz)",
    )
    ax_b.set_xlabel("total nonzeros across all $m$ components")
    ax_b.set_ylabel("adjusted explained variance (Shen--Huang)")
    ax_b.grid(True, linestyle=":", alpha=0.4)
    ax_b.legend(loc="lower right", fontsize=10)
    full_spines(ax_b)
    ax_b.text(
        0.005,
        0.98,
        "(b)",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        weight="bold",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.3"},
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--hero", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    rows, extras = run(cfg)
    out = Path(args.output_dir)
    write_csv(rows, out / cfg["output"]["csv"])
    if not args.no_figure and args.hero:
        fig_name = Path(cfg["output"]["figure"]).stem + "_hero.png"
        write_hero_figure(rows, extras, out / fig_name)
    data: FinanceData = extras["data"]
    print(
        f"wrote {len(rows)} rows across {len({r.method for r in rows})} methods; "
        f"{len(data.industries)} FF48 industries, {data.returns.shape[0]} trading days, "
        f"{data.start_date}..{data.end_date}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
