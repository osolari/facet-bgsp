"""M9 - single-cell RNA-seq: sparse gene programs on PBMC 3k.

Loads the 10x Genomics PBMC 3k dataset (2{,}700 peripheral-blood mononuclear
cells, 32{,}738 genes) applies standard scRNA-seq preprocessing (per-cell
total-count normalisation, log1p, 2{,}000 highly-variable genes, per-gene
$z$-score) and sweeps three decomposition methods at matched $m$:

  - FACET A-SQ swept over a dense $\\gamma$ grid.
  - Truncated PCA: dense PCA loadings hard-thresholded to the top-$K$
    gene-magnitudes per column, swept over $K$. The matched-sparsity
    baseline that the FACET story has to beat.
  - sklearn SparsePCA swept over $\\alpha$.

Ground-truth cluster labels come from PCA (20 PCs) + k-means, named by
canonical PBMC marker-gene programs (T CD4+/CD8+, NK, B, mono CD14+,
mono FCGR3A+, DC, platelet). The hero plots the Pareto frontier of
(total nonzeros, cluster-recovery AMI) across methods and the top gene
loadings of the hero FACET fit, so the components are both competitive at
matched sparsity and directly interpretable as marker-gene programs.
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

from _scrna import MARKER_GENES, ScRNAData, load_pbmc3k  # noqa: E402

from facet.papers.block_sparse import BlockSparseEstimator  # noqa: E402
from facet.supports import soft_threshold  # noqa: E402


@dc.dataclass(frozen=True)
class M9Row:
    method: str
    hyperparam: float
    total_nnz: int
    adjusted_mutual_info: float
    silhouette_2d: float
    reconstruction_r2: float
    m: int
    n_cells: int
    n_hvg: int


def _dense_pca_loadings(X_cells_x_hvg: np.ndarray, m: int) -> np.ndarray:
    cov = (X_cells_x_hvg.T @ X_cells_x_hvg) / (X_cells_x_hvg.shape[0] - 1)
    _, vecs = np.linalg.eigh(cov)
    return vecs[:, -m:]


def _facet_gene_loadings(X_cells_x_hvg: np.ndarray, m: int, gamma: float, seed: int) -> np.ndarray:
    """Return gene_loadings (n_hvg, m); all-zeros if the Armijo guard trips."""
    est = BlockSparseEstimator(
        objective="squared",
        gamma=float(gamma),
        solver="safeguarded",
        max_iterations=500,
        tolerance=1e-10,
        refit="none",
        random_state=seed,
    )
    try:
        fit = est.fit(X_cells_x_hvg, m=m)
    except RuntimeError:
        return np.zeros((X_cells_x_hvg.shape[1], m))
    X_scores = np.asarray(fit.raw.factors)  # (n_cells, m)
    AtX = X_cells_x_hvg.T @ X_scores  # (n_hvg, m)
    out = np.zeros_like(AtX)
    for j in range(m):
        z = soft_threshold(AtX[:, j], float(gamma))
        norm = float(np.linalg.norm(z))
        if norm > 1e-12:
            out[:, j] = z / norm
    return out


def _truncated_pca_loadings(dense_pca: np.ndarray, top_k_per_component: int) -> np.ndarray:
    """Zero every entry outside the top-K by magnitude in each column, renormalise.

    This is the natural matched-sparsity baseline the FACET story has to beat:
    just hard-threshold the dense PCA loadings instead of solving for them.
    """
    out = np.zeros_like(dense_pca)
    for j in range(dense_pca.shape[1]):
        col = dense_pca[:, j]
        keep = np.argsort(-np.abs(col))[:top_k_per_component]
        v = np.zeros_like(col)
        v[keep] = col[keep]
        n = float(np.linalg.norm(v))
        if n > 1e-12:
            out[:, j] = v / n
    return out


def _sparse_pca_loadings(X_cells_x_hvg: np.ndarray, m: int, alpha: float, seed: int) -> np.ndarray:
    from sklearn.decomposition import SparsePCA

    spca = SparsePCA(n_components=m, alpha=float(alpha), max_iter=400, tol=1e-6, random_state=seed)
    spca.fit(X_cells_x_hvg)
    return spca.components_.T


def _total_nnz(loadings: np.ndarray) -> int:
    return int(np.sum(np.abs(loadings) > 1e-6))


def _dominant_cluster_assignment(cell_scores: np.ndarray) -> np.ndarray:
    return np.argmax(np.abs(cell_scores), axis=1)


def _top_two_variance_components(loadings: np.ndarray, X: np.ndarray) -> np.ndarray:
    cell_scores = X @ loadings
    order = np.argsort(-cell_scores.var(axis=0))[:2]
    return loadings[:, order]


def _marker_gene_enrichment(loadings: np.ndarray, hvg_symbols: list[str], top_k: int) -> np.ndarray:
    """Return per-component fraction of top-K $|$loading$|$ HVGs that appear
    in any canonical PBMC marker-gene list. Higher = more interpretable."""
    marker_set: set[str] = set()
    for genes in MARKER_GENES.values():
        marker_set.update(genes)
    out = np.zeros(loadings.shape[1], dtype=np.float64)
    for j in range(loadings.shape[1]):
        col = np.abs(loadings[:, j])
        if col.sum() <= 1e-12:
            out[j] = 0.0
            continue
        top_idx = np.argsort(-col)[:top_k]
        hits = sum(1 for i in top_idx if hvg_symbols[i] in marker_set)
        out[j] = hits / float(top_k)
    return out


def _marker_gene_enrichment_curve(
    loadings: np.ndarray, hvg_symbols: list[str], k_values: list[int]
) -> np.ndarray:
    """Return ``(m, len(k_values))`` matrix; entry (j, i) = fraction of the
    top-K_i $|$loading$|$ HVGs of component j that are canonical PBMC markers."""
    marker_set: set[str] = set()
    for genes in MARKER_GENES.values():
        marker_set.update(genes)
    m = loadings.shape[1]
    out = np.zeros((m, len(k_values)), dtype=np.float64)
    for j in range(m):
        col = np.abs(loadings[:, j])
        if col.sum() <= 1e-12:
            continue
        order = np.argsort(-col)
        for i, k in enumerate(k_values):
            top_idx = order[:k]
            hits = sum(1 for g in top_idx if hvg_symbols[g] in marker_set)
            out[j, i] = hits / float(k)
    return out


def _cell_type_purity(
    loadings: np.ndarray, expr: np.ndarray, labels: np.ndarray, n_clusters: int
) -> np.ndarray:
    """Per (component j, cluster c), mean $|$score$|$ on that cluster's cells;
    then z-score each row so a 'pure' component shows a single hot cluster."""
    scores = expr @ loadings
    m = scores.shape[1]
    mat = np.zeros((m, n_clusters), dtype=np.float64)
    for c in range(n_clusters):
        mask = labels == c
        if mask.any():
            mat[:, c] = np.abs(scores[mask]).mean(axis=0)
    row_mean = mat.mean(axis=1, keepdims=True)
    row_std = mat.std(axis=1, keepdims=True)
    row_std = np.where(row_std < 1e-12, 1.0, row_std)
    return (mat - row_mean) / row_std


def _reconstruction_r2(X: np.ndarray, loadings: np.ndarray) -> float:
    if loadings.size == 0:
        return 0.0
    col_norms = np.linalg.norm(loadings, axis=0)
    nz = loadings[:, col_norms > 1e-12]
    if nz.shape[1] == 0:
        return 0.0
    q, _ = np.linalg.qr(nz)
    proj = X @ q @ q.T
    num = float(np.linalg.norm(X - proj, "fro") ** 2)
    den = float(np.linalg.norm(X, "fro") ** 2)
    return 1.0 - num / den if den > 0 else 0.0


def _score_one(
    method: str,
    hyperparam: float,
    loadings: np.ndarray,
    expr: np.ndarray,
    labels: np.ndarray,
    m: int,
) -> M9Row:
    from sklearn.metrics import adjusted_mutual_info_score, silhouette_score

    cell_scores = expr @ loadings
    assignment = _dominant_cluster_assignment(cell_scores)
    try:
        ami = float(adjusted_mutual_info_score(labels, assignment))
    except Exception:
        ami = float("nan")
    top2 = _top_two_variance_components(loadings, expr)
    top2_scores = expr @ top2
    try:
        sil = float(silhouette_score(top2_scores, labels))
    except Exception:
        sil = float("nan")
    return M9Row(
        method=method,
        hyperparam=float(hyperparam),
        total_nnz=_total_nnz(loadings),
        adjusted_mutual_info=ami,
        silhouette_2d=sil,
        reconstruction_r2=_reconstruction_r2(expr, loadings),
        m=m,
        n_cells=int(expr.shape[0]),
        n_hvg=int(expr.shape[1]),
    )


def run(config: dict[str, Any]) -> tuple[list[M9Row], dict[str, Any]]:
    data: ScRNAData = load_pbmc3k(
        mtx_dir=str(config["data"]["mtx_dir"]),
        cache_path=str(config["data"].get("cache_path", "")) or None,
        n_hvg=int(config["data"]["n_hvg"]),
        n_clusters=int(config["data"]["n_clusters"]),
        seed=int(config["seed"]),
    )
    expr = data.expression
    labels = data.cluster_labels
    m = int(config["decomposition"]["m"])
    seed = int(config["seed"])

    rows: list[M9Row] = []
    best_facet: tuple[M9Row, np.ndarray] | None = None
    facet_fits: list[tuple[M9Row, np.ndarray]] = []

    # Dense PCA, one point and the ancestor of the truncated-PCA sweep.
    pca_dense = _dense_pca_loadings(expr, m)
    pca_row = _score_one("pca", float("nan"), pca_dense, expr, labels, m)
    rows.append(pca_row)

    # Truncated PCA at a range of top-K-per-component values.
    for k in config["truncated_pca"]["top_k_per_component"]:
        comps = _truncated_pca_loadings(pca_dense, int(k))
        rows.append(_score_one("pca_truncated", float(k), comps, expr, labels, m))

    # FACET A-SQ sweep.
    for gamma in config["facet"]["gamma_candidates"]:
        comps = _facet_gene_loadings(expr, m, float(gamma), seed)
        if _total_nnz(comps) == 0:
            continue
        row = _score_one("facet_asq", float(gamma), comps, expr, labels, m)
        rows.append(row)
        facet_fits.append((row, comps))
        if best_facet is None or row.adjusted_mutual_info > best_facet[0].adjusted_mutual_info:
            best_facet = (row, comps)

    # sklearn SparsePCA sweep.
    for alpha in config["sparse_pca"]["alphas"]:
        comps = _sparse_pca_loadings(expr, m, float(alpha), seed)
        if _total_nnz(comps) == 0:
            continue
        rows.append(_score_one("sparse_pca_zou", float(alpha), comps, expr, labels, m))

    interp_cfg = config.get("interpretability", {}) or {}
    ami_tol = float(interp_cfg.get("ami_tolerance", 0.05))
    reference = str(interp_cfg.get("reference", "best_facet"))
    ref_ami = (
        pca_row.adjusted_mutual_info
        if reference == "pca"
        else (best_facet[0].adjusted_mutual_info if best_facet else float("nan"))
    )
    threshold = ref_ami - ami_tol
    qualifying = [
        (row, comps) for row, comps in facet_fits if row.adjusted_mutual_info >= threshold
    ]
    interp_facet = min(qualifying, key=lambda rc: rc[0].total_nnz) if qualifying else best_facet

    extras = {
        "data": data,
        "pca_loadings": pca_dense,
        "pca_row": pca_row,
        "best_facet_row": best_facet[0] if best_facet else None,
        "best_facet_loadings": best_facet[1] if best_facet else None,
        "interp_facet_row": interp_facet[0] if interp_facet else None,
        "interp_facet_loadings": interp_facet[1] if interp_facet else None,
        "interp_ami_tolerance": ami_tol,
        "interp_reference": reference,
        "hero_top_k_components": int(config["figure"]["hero_top_k_components"]),
        "hero_top_k_genes": int(config["figure"]["hero_top_k_genes"]),
    }
    return rows, extras


def write_csv(rows: list[M9Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dc.fields(M9Row)])
        writer.writeheader()
        for r in rows:
            writer.writerow(dc.asdict(r))


def write_provenance_json(
    *,
    path: Path,
    name: str,
    config_path: str,
    extras: dict[str, Any],
    extra_fields: dict[str, Any] | None = None,
) -> None:
    """Sidecar JSON with run metadata next to the CSV. Captures seed, data
    shape, best / interp FACET + PCA rows, interp selector settings, and
    an absolute wall-clock timestamp. Deterministic, hand-readable, and
    stable enough to diff across runs."""
    import datetime as _dt
    import json as _json

    best = extras.get("best_facet_row")
    interp = extras.get("interp_facet_row")
    pca = extras.get("pca_row")
    data = extras.get("data")
    expr_shape = tuple(getattr(data, "expression", np.empty((0, 0))).shape) if data else (0, 0)
    payload: dict[str, Any] = {
        "name": name,
        "config_path": config_path,
        "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "n_cells": int(expr_shape[0]),
        "n_hvg": int(expr_shape[1]),
        "interp_ami_tolerance": extras.get("interp_ami_tolerance"),
        "interp_reference": extras.get("interp_reference"),
        "best_facet": dc.asdict(best) if best else None,
        "interp_facet": dc.asdict(interp) if interp else None,
        "pca": dc.asdict(pca) if pca else None,
    }
    if extra_fields:
        payload.update(extra_fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        _json.dump(payload, f, indent=2, sort_keys=True, default=str)


def write_hero_figure(rows: list[M9Row], extras: dict[str, Any], path: Path) -> None:
    """Hero: Pareto frontier of (nnz, AMI) across methods + top genes per
    FACET component at the best-AMI gamma."""
    from _style import apply_style, full_spines

    apply_style()
    import matplotlib.pyplot as plt

    data: ScRNAData = extras["data"]
    hvg_symbols = data.hvg_symbols
    best_row: M9Row | None = extras["best_facet_row"]
    best_loadings: np.ndarray | None = extras["best_facet_loadings"]

    fig = plt.figure(figsize=(14.4, 9.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9], width_ratios=[1.0, 1.0])
    ax_ami = fig.add_subplot(gs[0, 0])
    ax_sil = fig.add_subplot(gs[0, 1])
    ax_genes = fig.add_subplot(gs[1, :])

    style = {
        "pca": ("PCA (dense)", "#1f77b4", "s", 10),
        "pca_truncated": ("truncated PCA", "#17becf", "x", 9),
        "facet_asq": ("FACET A-SQ", "#d62728", "o", 9),
        "sparse_pca_zou": ("sklearn SparsePCA (Zou)", "#2ca02c", "D", 9),
    }

    for ax, col, ylabel, panel_tag in (
        (ax_ami, "adjusted_mutual_info", "AMI(dominant comp., cluster label)", "(a)"),
        (ax_sil, "silhouette_2d", "silhouette in top-2 comp. plane", "(b)"),
    ):
        for method, (label, colour, marker, size) in style.items():
            sub = [r for r in rows if r.method == method]
            if not sub:
                continue
            sub = sorted(sub, key=lambda r: r.total_nnz)
            xs = [r.total_nnz for r in sub]
            ys = [getattr(r, col) for r in sub]
            if method == "pca":
                # single dense point; render as a horizontal reference bar
                ax.scatter(
                    xs,
                    ys,
                    color=colour,
                    marker=marker,
                    s=size * 10,
                    edgecolor="black",
                    linewidth=0.5,
                    label=label,
                    zorder=4,
                )
            else:
                ax.plot(
                    xs,
                    ys,
                    color=colour,
                    marker=marker,
                    markersize=size,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    lw=1.6,
                    alpha=0.9,
                    label=label,
                )
        ax.set_xlabel("total nonzeros across all $m$ components")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.grid(True, linestyle=":", alpha=0.4, which="both")
        ax.legend(loc="lower right", fontsize=10)
        full_spines(ax)
        ax.text(
            0.02,
            0.98,
            panel_tag,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            weight="bold",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.3"},
        )

    # Panel (c): FACET top-K genes per component at the best-AMI gamma.
    k_components = extras["hero_top_k_components"]
    k_genes = extras["hero_top_k_genes"]

    if best_loadings is not None and best_row is not None:
        comp_mass = np.abs(best_loadings).sum(axis=0)
        comp_order = np.argsort(-comp_mass)[:k_components]
        strip_height = 0.78
        for row_idx, j in enumerate(comp_order):
            col = best_loadings[:, j]
            order_by_abs = np.argsort(-np.abs(col))[:k_genes]
            y_center = k_components - 1 - row_idx
            for xi, gi in enumerate(order_by_abs):
                value = col[gi]
                x_center = xi + 0.5
                ax_genes.bar(
                    x_center,
                    abs(value),
                    width=0.9,
                    bottom=y_center - strip_height / 2,
                    color="#d62728" if value >= 0 else "#1f77b4",
                    edgecolor="black",
                    linewidth=0.4,
                    alpha=0.9,
                )
                ax_genes.text(
                    x_center,
                    y_center - strip_height / 2 - 0.07,
                    hvg_symbols[gi],
                    ha="center",
                    va="top",
                    fontsize=8,
                    rotation=90,
                    color="0.25",
                )
            ax_genes.text(
                -0.6,
                y_center,
                f"FACET\ncomp. {row_idx + 1}",
                ha="right",
                va="center",
                fontsize=10,
                weight="bold",
            )
        ax_genes.set_xlim(-1.2, k_genes)
        ax_genes.set_ylim(-0.6, k_components - 0.2)
        ax_genes.set_xticks([])
        ax_genes.set_yticks([])
        ax_genes.set_xlabel(
            f"top-$|$loading$|$ HVGs per component at the hero FACET gamma "
            f"($\\gamma={best_row.hyperparam:.2g}$, "
            f"nnz={best_row.total_nnz}, AMI={best_row.adjusted_mutual_info:.3f})"
        )
        full_spines(ax_genes)
        ax_genes.text(
            0.005,
            0.98,
            "(c)",
            transform=ax_genes.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            weight="bold",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.3"},
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


FACET_COLOUR = "#c82727"
PCA_COLOUR = "#1b4f85"


def _shorten(name: str, max_len: int = 24) -> str:
    """Truncate long cell-type names with an ellipsis so x-axis labels stay
    inside their allotted width in the heatmap panels."""
    if len(name) <= max_len:
        return name
    return name[: max_len - 1].rstrip() + "…"


def render_interpretability_hero(
    *,
    path: Path,
    dataset_title: str,
    interp_loadings: np.ndarray,
    pca_loadings: np.ndarray,
    interp_row: M9Row,
    pca_row: M9Row | None,
    expr: np.ndarray,
    labels: np.ndarray,
    cluster_names: list[str],
    hvg_symbols: list[str],
    marker_set: set[str],
    marker_source_label: str,
    label_axis_title: str,
    k_values: tuple[int, ...] = (5, 10, 15, 25, 50, 100),
) -> None:
    """Publication-quality interpretability hero shared by the four M9
    drivers. Three panels: (a) enrichment-vs-K curves with per-component
    IQR band + chance line + gene-economy annotation; (b) FACET per-
    component cell-type purity heatmap with dominant-type labels on each
    row; (c) matching PCA purity heatmap on a shared colour scale."""
    from _style import apply_style, full_spines

    apply_style()
    import matplotlib.pyplot as plt
    import seaborn as sns

    m = interp_loadings.shape[1]
    n_clusters = len(cluster_names)

    def _curve(loadings: np.ndarray) -> np.ndarray:
        out = np.zeros((loadings.shape[1], len(k_values)), dtype=np.float64)
        for j in range(loadings.shape[1]):
            col = np.abs(loadings[:, j])
            if col.sum() <= 1e-12:
                continue
            order = np.argsort(-col)
            for i, k in enumerate(k_values):
                top_idx = order[:k]
                out[j, i] = sum(1 for g in top_idx if hvg_symbols[g] in marker_set) / float(k)
        return out

    facet_curve = _curve(interp_loadings)
    pca_curve = _curve(pca_loadings)
    facet_purity = _cell_type_purity(interp_loadings, expr, labels, n_clusters)
    pca_purity = _cell_type_purity(pca_loadings, expr, labels, n_clusters)
    chance_rate = len(marker_set) / max(len(hvg_symbols), 1)
    facet_nnz_per = interp_row.total_nnz / float(m)
    pca_nnz_per = (pca_row.total_nnz if pca_row else len(hvg_symbols) * m) / float(m)
    ratio = pca_nnz_per / max(facet_nnz_per, 1)

    short_names = [_shorten(name) for name in cluster_names]
    facet_dominant = [cluster_names[int(np.argmax(facet_purity[j]))] for j in range(m)]

    panel_width = max(10.0, 0.42 * n_clusters + 4.8)
    fig = plt.figure(figsize=(panel_width, 10.0), constrained_layout=True)
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[0.22, 1.0, 1.25],
        width_ratios=[1.0, 1.0],
    )
    ax_title = fig.add_subplot(gs[0, :])
    ax_enrich = fig.add_subplot(gs[1, :])
    ax_facet = fig.add_subplot(gs[2, 0])
    ax_pca = fig.add_subplot(gs[2, 1], sharey=ax_facet)

    # Dataset banner (acts as a figure title that lives inside the figure
    # rectangle so savefig/bbox_inches="tight" doesn't trim it).
    ax_title.set_axis_off()
    ami_delta = ""
    if pca_row is not None:
        delta = interp_row.adjusted_mutual_info - pca_row.adjusted_mutual_info
        sign = "+" if delta >= 0 else ""
        winner = "FACET" if delta >= 0 else "PCA"
        ami_delta = (
            f"\nFACET AMI {interp_row.adjusted_mutual_info:.3f}, "
            f"PCA AMI {pca_row.adjusted_mutual_info:.3f}  "
            f"(Δ = {sign}{delta:.3f}, {winner} wins)"
        )
    ax_title.text(
        0.5,
        0.65,
        dataset_title,
        ha="center",
        va="center",
        transform=ax_title.transAxes,
        fontsize=16,
        weight="bold",
    )
    ax_title.text(
        0.5,
        0.08,
        f"n = {expr.shape[0]:,} cells  |  {expr.shape[1]:,} HVGs  |  "
        f"m = {m} components  |  {n_clusters} cell types" + ami_delta,
        ha="center",
        va="center",
        transform=ax_title.transAxes,
        fontsize=11,
        color="0.25",
    )

    # Panel (a): enrichment-vs-K curves.
    x = np.asarray(k_values, dtype=np.float64)
    for curve, colour, label in (
        (
            facet_curve,
            FACET_COLOUR,
            f"FACET A-SQ ($\\gamma={interp_row.hyperparam:.2g}$, "
            f"nnz={interp_row.total_nnz:,}, AMI={interp_row.adjusted_mutual_info:.3f})",
        ),
        (
            pca_curve,
            PCA_COLOUR,
            "PCA (dense"
            + (
                f", nnz={pca_row.total_nnz:,}, AMI={pca_row.adjusted_mutual_info:.3f})"
                if pca_row
                else ")"
            ),
        ),
    ):
        mean = curve.mean(axis=0)
        q1 = np.quantile(curve, 0.25, axis=0)
        q3 = np.quantile(curve, 0.75, axis=0)
        ax_enrich.fill_between(x, q1, q3, color=colour, alpha=0.18, linewidth=0)
        ax_enrich.plot(
            x,
            mean,
            color=colour,
            marker="o",
            markersize=8,
            lw=2.2,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label=label,
        )
    ax_enrich.axhline(
        y=chance_rate,
        color="0.4",
        linestyle="--",
        lw=1.0,
        label=f"chance (uniform random, {chance_rate:.3f})",
    )
    ax_enrich.set_xscale("log")
    ax_enrich.set_xticks(x)
    ax_enrich.set_xticklabels([str(k) for k in k_values])
    ax_enrich.set_xlabel("$K$ (top-$K$ loadings per component considered)", fontsize=12)
    ax_enrich.set_ylabel(f"mean fraction in {marker_source_label}", fontsize=12)
    y_hi = max(0.3, 1.15 * max(facet_curve.max(), pca_curve.max()))
    ax_enrich.set_ylim(0.0, y_hi)
    ax_enrich.grid(True, which="both", linestyle=":", alpha=0.4)
    ax_enrich.tick_params(axis="both", labelsize=11)
    ax_enrich.legend(loc="upper right", fontsize=11, framealpha=0.92)
    full_spines(ax_enrich)
    ax_enrich.text(
        0.005,
        0.97,
        "(a) marker-gene enrichment vs top-$K$",
        transform=ax_enrich.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        weight="bold",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.3"},
    )
    ax_enrich.text(
        0.005,
        0.04,
        f"active gene budget / component: FACET {facet_nnz_per:,.0f}, "
        f"PCA {pca_nnz_per:,.0f} ({ratio:.1f}x denser)",
        transform=ax_enrich.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        style="italic",
        color="0.15",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#fff3e0", "edgecolor": "0.4"},
    )

    # Panels (b)+(c): purity heatmaps.
    vmax = float(np.nanmax(np.abs(np.concatenate([facet_purity.ravel(), pca_purity.ravel()]))))
    heatmap_tick_fontsize = max(7, 10 - max(0, n_clusters - 10) // 4)
    for ax, mat, title, panel_tag, show_rhs in (
        (ax_facet, facet_purity, "FACET A-SQ", "(b)", True),
        (ax_pca, pca_purity, "PCA (dense)", "(c)", False),
    ):
        sns.heatmap(
            mat,
            ax=ax,
            cmap="RdBu_r",
            center=0.0,
            vmin=-vmax,
            vmax=vmax,
            cbar=True,
            cbar_kws={"label": "mean $|$score$|$, $z$-scored per component"},
            xticklabels=short_names,
            yticklabels=[f"c{i + 1}" for i in range(m)],
            linewidths=0.0,
        )
        ax.set_xlabel(label_axis_title, fontsize=12)
        ax.set_ylabel("component", fontsize=12)
        ax.set_xticklabels(
            short_names,
            rotation=35,
            ha="right",
            fontsize=heatmap_tick_fontsize,
        )
        ax.tick_params(axis="y", labelsize=10)
        full_spines(ax)
        ax.text(
            0.005,
            1.03,
            f"{panel_tag} cell-type purity: {title}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=13,
            weight="bold",
        )
        if show_rhs:
            for j in range(m):
                ax.text(
                    n_clusters + 0.4,
                    j + 0.5,
                    "→ " + _shorten(facet_dominant[j], 20),
                    transform=ax.transData,
                    ha="left",
                    va="center",
                    fontsize=9,
                    color="0.2",
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_interpretability_figure(
    extras: dict[str, Any], path: Path, *, top_k_marker: int = 15
) -> None:
    """PBMC3K specialisation of ``render_interpretability_hero`` using the
    k-means cluster labels + MARKER_GENES defined in ``_scrna``."""
    data: ScRNAData = extras["data"]
    interp_loadings: np.ndarray | None = extras["interp_facet_loadings"]
    interp_row: M9Row | None = extras["interp_facet_row"]
    if interp_loadings is None or interp_row is None:
        return
    marker_set: set[str] = set()
    for genes in MARKER_GENES.values():
        marker_set.update(genes)
    render_interpretability_hero(
        path=path,
        dataset_title="M9 PBMC 3k (10x Genomics) — interpretability hero",
        interp_loadings=interp_loadings,
        pca_loadings=extras["pca_loadings"],
        interp_row=interp_row,
        pca_row=extras.get("pca_row"),
        expr=data.expression,
        labels=data.cluster_labels,
        cluster_names=data.cluster_names,
        hvg_symbols=data.hvg_symbols,
        marker_set=marker_set,
        marker_source_label="PBMC marker lists",
        label_axis_title="pseudo-label (PCA + $k$-means + marker programs)",
    )
    _ = top_k_marker


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--hero", action="store_true")
    parser.add_argument("--interpretability", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    rows, extras = run(cfg)
    out = Path(args.output_dir)
    write_csv(rows, out / cfg["output"]["csv"])
    provenance_name = Path(cfg["output"]["csv"]).with_suffix(".provenance.json").name
    write_provenance_json(
        path=out / provenance_name,
        name=str(cfg.get("name", "m9_scrna")),
        config_path=str(args.config),
        extras=extras,
        extra_fields={"dataset": "pbmc3k", "label_source": "pca+kmeans+markers"},
    )
    if not args.no_figure and args.hero:
        fig_name = Path(cfg["output"]["figure"]).stem + "_hero.png"
        write_hero_figure(rows, extras, out / fig_name)
    if not args.no_figure and args.interpretability:
        top_k_marker = int(cfg.get("figure", {}).get("interpretability_top_k_genes", 15))
        interp_name = cfg["output"].get(
            "interpretability_figure", "m9_scrna_interpretability_hero.png"
        )
        write_interpretability_figure(
            extras,
            out / interp_name,
            top_k_marker=top_k_marker,
        )
    data: ScRNAData = extras["data"]
    print(
        f"wrote {len(rows)} rows across {len({r.method for r in rows})} methods; "
        f"{data.expression.shape[0]} cells x {data.expression.shape[1]} HVGs; "
        f"best FACET: {extras['best_facet_row']}; "
        f"interp FACET (sparsest with AMI within {extras['interp_ami_tolerance']} of "
        f"{extras['interp_reference']}): {extras['interp_facet_row']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
