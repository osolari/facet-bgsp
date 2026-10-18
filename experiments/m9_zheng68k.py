"""M9-Zheng-68k: sparse gene programs on the fresh 68k PBMC donor-A matrix
with the paper-published FACS-correlated per-barcode cell-type labels.

Native labels are from
10XGenomics/single-cell-3prime-paper/pbmc68k_analysis/68k_pbmc_barcodes_annotation.tsv
(correlation with FACS-sorted reference profiles), so AMI vs this label
is a legitimate cross-method benchmark. Pipeline reuses helpers from
m9_scrna.py (preprocessing differs; sweep + metrics identical).
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

from _scrna import MARKER_GENES  # noqa: E402
from _zheng68k import Zheng68kData, load_zheng68k  # noqa: E402
from m9_scrna import (  # noqa: E402
    M9Row,
    _dense_pca_loadings,
    _facet_gene_loadings,
    _score_one,
    _sparse_pca_loadings,
    _total_nnz,
    _truncated_pca_loadings,
    render_interpretability_hero,
)
from m9_scrna import (  # noqa: E402
    write_csv as _write_csv_shared,
)
from m9_scrna import (  # noqa: E402
    write_provenance_json as _write_provenance_json,
)


def _marker_set() -> set[str]:
    out: set[str] = set()
    for genes in MARKER_GENES.values():
        out.update(genes)
    return out


def run(config: dict[str, Any]) -> tuple[list[M9Row], dict[str, Any]]:
    data: Zheng68kData = load_zheng68k(
        data_root=str(config["data"]["data_root"]),
        labels_path=str(config["data"]["labels_path"]),
        cache_path=str(config["data"].get("cache_path", "")) or None,
        n_hvg=int(config["data"]["n_hvg"]),
    )
    expr = data.expression
    labels = data.cell_type_labels
    m = int(config["decomposition"]["m"])
    seed = int(config["seed"])

    rows: list[M9Row] = []
    facet_fits: list[tuple[M9Row, np.ndarray]] = []
    best_facet: tuple[M9Row, np.ndarray] | None = None

    pca_dense = _dense_pca_loadings(expr, m)
    pca_row = _score_one("pca", float("nan"), pca_dense, expr, labels, m)
    rows.append(pca_row)

    for k in config["truncated_pca"]["top_k_per_component"]:
        comps = _truncated_pca_loadings(pca_dense, int(k))
        rows.append(_score_one("pca_truncated", float(k), comps, expr, labels, m))

    for gamma in config["facet"]["gamma_candidates"]:
        comps = _facet_gene_loadings(expr, m, float(gamma), seed)
        if _total_nnz(comps) == 0:
            continue
        row = _score_one("facet_asq", float(gamma), comps, expr, labels, m)
        rows.append(row)
        facet_fits.append((row, comps))
        if best_facet is None or row.adjusted_mutual_info > best_facet[0].adjusted_mutual_info:
            best_facet = (row, comps)

    for alpha in config["sparse_pca"]["alphas"]:
        comps = _sparse_pca_loadings(expr, m, float(alpha), seed)
        if _total_nnz(comps) == 0:
            continue
        rows.append(_score_one("sparse_pca_zou", float(alpha), comps, expr, labels, m))

    interp_cfg = config.get("interpretability", {}) or {}
    ami_tol = float(interp_cfg.get("ami_tolerance", 0.025))
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


def write_hero_figure(rows: list[M9Row], extras: dict[str, Any], path: Path) -> None:
    from _style import apply_style, full_spines

    apply_style()
    import matplotlib.pyplot as plt

    data: Zheng68kData = extras["data"]
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
        (ax_ami, "adjusted_mutual_info", "AMI(dominant comp., FACS-correlated label)", "(a)"),
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


def write_interpretability_figure(
    extras: dict[str, Any], path: Path, *, top_k_marker: int = 15
) -> None:
    """Zheng17 68k specialisation of the shared interp hero: 68k donor-A
    PBMC matrix with FACS-correlated per-barcode labels."""
    data: Zheng68kData = extras["data"]
    interp_loadings: np.ndarray | None = extras["interp_facet_loadings"]
    interp_row: M9Row | None = extras["interp_facet_row"]
    if interp_loadings is None or interp_row is None:
        return
    render_interpretability_hero(
        path=path,
        dataset_title="M9 Zheng17 68k PBMC (donor A) — interpretability hero",
        interp_loadings=interp_loadings,
        pca_loadings=extras["pca_loadings"],
        interp_row=interp_row,
        pca_row=extras.get("pca_row"),
        expr=data.expression,
        labels=data.cell_type_labels,
        cluster_names=data.cell_type_names,
        hvg_symbols=data.hvg_symbols,
        marker_set=_marker_set(),
        marker_source_label="PBMC marker lists",
        label_axis_title="FACS-correlated cell type (Zheng17 68k, native)",
    )
    _ = top_k_marker


def write_csv(rows: list[M9Row], path: Path) -> None:
    _write_csv_shared(rows, path)


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
    _write_provenance_json(
        path=out / provenance_name,
        name=str(cfg.get("name", "m9_zheng68k")),
        config_path=str(args.config),
        extras=extras,
        extra_fields={
            "dataset": "zheng17_68k_pbmc_donor_a",
            "label_source": "FACS-correlated (paper annotation TSV)",
            "n_cell_types": len(extras["data"].cell_type_names),
        },
    )
    if not args.no_figure and args.hero:
        fig_name = Path(cfg["output"]["figure"]).stem + "_hero.png"
        write_hero_figure(rows, extras, out / fig_name)
    if not args.no_figure and args.interpretability:
        top_k_marker = int(cfg.get("figure", {}).get("interpretability_top_k_genes", 15))
        interp_name = cfg["output"].get(
            "interpretability_figure", "m9_zheng68k_interpretability_hero.png"
        )
        write_interpretability_figure(extras, out / interp_name, top_k_marker=top_k_marker)
    data: Zheng68kData = extras["data"]
    print(
        f"wrote {len(rows)} rows across {len({r.method for r in rows})} methods; "
        f"{data.expression.shape[0]} cells x {data.expression.shape[1]} HVGs; "
        f"{len(data.cell_type_names)} FACS-correlated cell types; "
        f"best FACET: {extras['best_facet_row']}; "
        f"interp FACET (sparsest with AMI within {extras['interp_ami_tolerance']} of "
        f"{extras['interp_reference']}): {extras['interp_facet_row']}"
    )
    _ = dc.fields(M9Row), csv.excel
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
