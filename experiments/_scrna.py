"""scRNA-seq helpers for M9: load PBMC 3k, preprocess, and label clusters.

Reads the 10x Genomics PBMC 3k filtered gene/barcode matrix (``.mtx`` plus
``genes.tsv`` and ``barcodes.tsv``) applies a minimal scanpy-style
preprocessing pipeline (per-cell total-count normalisation, ``log1p``,
highly-variable-gene selection, per-gene $z$-score) and produces an
``(n_cells, n_hvg)`` matrix together with gene symbols. ``cluster_cells``
runs PCA + $k$-means to produce cluster labels usable as ground-truth
proxies for downstream decomposition comparisons.

No ``scanpy`` dependency; everything runs on ``scipy.sparse`` + ``sklearn``.
"""

from __future__ import annotations

import gzip
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

PBMC3K_URL = (
    "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/"
    "pbmc3k_filtered_gene_bc_matrices.tar.gz"
)


@dataclass(frozen=True)
class ScRNAData:
    expression: np.ndarray  # (n_cells, n_hvg) log-normalised + z-scored
    cell_barcodes: list[str]
    hvg_symbols: list[str]
    hvg_ids: list[str]
    cluster_labels: np.ndarray  # (n_cells,) int
    cluster_names: list[str]


def _load_10x_mtx(mtx_dir: str) -> tuple[Any, list[str], list[str], list[str]]:
    """Return (sparse_count_matrix[cells, genes], barcodes, gene_ids, gene_symbols)."""
    from scipy.io import mmread
    from scipy.sparse import csr_matrix

    mtx_path = os.path.join(mtx_dir, "matrix.mtx")
    bc_path = os.path.join(mtx_dir, "barcodes.tsv")
    gene_path = os.path.join(mtx_dir, "genes.tsv")

    opener = gzip.open if mtx_path.endswith(".gz") else open
    with opener(mtx_path, "rt" if mtx_path.endswith(".gz") else "r") as f:
        counts_coo = mmread(f).tocsr()  # (genes, cells) in the 10x convention
    counts = csr_matrix(counts_coo.T)  # transpose to (cells, genes)

    with open(bc_path) as f:
        barcodes = [line.strip() for line in f if line.strip()]
    gene_ids: list[str] = []
    gene_symbols: list[str] = []
    with open(gene_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                gene_ids.append(parts[0])
                gene_symbols.append(parts[1])
    return counts, barcodes, gene_ids, gene_symbols


def preprocess_pbmc3k(
    mtx_dir: str,
    *,
    min_genes_per_cell: int = 200,
    min_cells_per_gene: int = 3,
    n_hvg: int = 2000,
    target_total: float = 1e4,
    cache_path: str | None = None,
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """Full preprocessing pipeline returning (expr, barcodes, hvg_symbols, hvg_ids)."""
    if cache_path and os.path.exists(cache_path):
        payload = np.load(cache_path, allow_pickle=True)
        return (
            payload["expr"],
            list(payload["barcodes"]),
            list(payload["hvg_symbols"]),
            list(payload["hvg_ids"]),
        )

    counts, barcodes, gene_ids, gene_symbols = _load_10x_mtx(mtx_dir)

    # Filter cells with too few expressed genes
    genes_per_cell = np.asarray((counts > 0).sum(axis=1)).ravel()
    keep_cells = genes_per_cell >= min_genes_per_cell
    counts = counts[keep_cells]
    barcodes = [b for b, k in zip(barcodes, keep_cells, strict=True) if k]

    # Filter genes present in too few cells
    cells_per_gene = np.asarray((counts > 0).sum(axis=0)).ravel()
    keep_genes = cells_per_gene >= min_cells_per_gene
    counts = counts[:, keep_genes]
    gene_ids = [g for g, k in zip(gene_ids, keep_genes, strict=True) if k]
    gene_symbols = [g for g, k in zip(gene_symbols, keep_genes, strict=True) if k]

    # Total-count normalise per cell, then log1p.
    cell_totals = np.asarray(counts.sum(axis=1)).ravel()
    scale = target_total / np.maximum(cell_totals, 1.0)
    counts_scaled = counts.multiply(scale[:, None])
    log_counts = counts_scaled.log1p().toarray().astype(np.float64)

    # Highly-variable-gene selection via Seurat-flavour dispersion ranking.
    mean_expr = log_counts.mean(axis=0)
    var_expr = log_counts.var(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dispersion = np.where(mean_expr > 1e-6, var_expr / mean_expr, 0.0)
    hvg_rank = np.argsort(-dispersion)[:n_hvg]
    hvg_rank = np.sort(hvg_rank)
    hvg_symbols = [gene_symbols[i] for i in hvg_rank]
    hvg_ids = [gene_ids[i] for i in hvg_rank]
    expr = log_counts[:, hvg_rank]

    # z-score each HVG across cells (standard sparse-PCA preprocessing).
    mu = expr.mean(axis=0, keepdims=True)
    sd = expr.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    expr = (expr - mu) / sd

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez_compressed(
            cache_path,
            expr=expr,
            barcodes=np.array(barcodes),
            hvg_symbols=np.array(hvg_symbols),
            hvg_ids=np.array(hvg_ids),
        )
    return expr, barcodes, hvg_symbols, hvg_ids


# Canonical PBMC marker genes; used only to NAME each k-means cluster via its
# average marker-gene expression. The clustering itself is unsupervised.
MARKER_GENES: dict[str, list[str]] = {
    "T (CD4+)": ["IL7R", "CD4", "CCR7"],
    "T (CD8+)": ["CD8A", "CD8B", "GZMK"],
    "NK": ["GNLY", "NKG7", "KLRD1"],
    "B": ["MS4A1", "CD79A", "CD79B"],
    "Mono (CD14+)": ["LYZ", "CD14", "S100A9"],
    "Mono (FCGR3A)": ["FCGR3A", "MS4A7", "CDKN1C"],
    "DC": ["FCER1A", "CST3", "CLEC10A"],
    "Platelet": ["PPBP", "PF4", "GP9"],
}


def cluster_cells(
    expr: np.ndarray,
    hvg_symbols: list[str],
    n_clusters: int = 8,
    pca_dim: int = 20,
    seed: int = 2044,
) -> tuple[np.ndarray, list[str]]:
    """Run PCA then KMeans; label every resulting cluster by the marker-gene
    program whose average expression it most strongly carries."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    pca_scores = PCA(n_components=pca_dim, random_state=seed).fit_transform(expr)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(pca_scores)

    # Map each cluster to the marker group whose mean per-cell expression on
    # that cluster's cells is largest (and remove duplicates by iterative
    # assignment: strongest cluster for each marker first).
    name_lookup = {sym: i for i, sym in enumerate(hvg_symbols)}
    marker_mean = {}
    for name, genes in MARKER_GENES.items():
        gene_cols = [name_lookup[g] for g in genes if g in name_lookup]
        if not gene_cols:
            marker_mean[name] = -np.inf * np.ones(n_clusters)
            continue
        gene_mat = expr[:, gene_cols]
        marker_mean[name] = np.array(
            [
                gene_mat[labels == c].mean() if (labels == c).any() else -np.inf
                for c in range(n_clusters)
            ]
        )

    cluster_names: list[str] = ["(unk)"] * n_clusters
    unassigned = list(range(n_clusters))
    for marker_name, scores in sorted(
        marker_mean.items(), key=lambda kv: -max(kv[1]) if kv[1].size else 0
    ):
        if not unassigned:
            break
        for cand in np.argsort(-scores):
            cand = int(cand)
            if cand in unassigned:
                cluster_names[cand] = marker_name
                unassigned.remove(cand)
                break
    # Any clusters still unassigned get a fallback.
    for cand in unassigned:
        cluster_names[cand] = f"cluster {cand}"
    return labels, cluster_names


def load_pbmc3k(
    mtx_dir: str,
    *,
    cache_path: str | None = None,
    n_hvg: int = 2000,
    n_clusters: int = 8,
    seed: int = 2044,
) -> ScRNAData:
    expr, barcodes, hvg_symbols, hvg_ids = preprocess_pbmc3k(
        mtx_dir,
        n_hvg=n_hvg,
        cache_path=cache_path,
    )
    labels, cluster_names = cluster_cells(expr, hvg_symbols, n_clusters=n_clusters, seed=seed)
    return ScRNAData(
        expression=expr,
        cell_barcodes=barcodes,
        hvg_symbols=hvg_symbols,
        hvg_ids=hvg_ids,
        cluster_labels=labels.astype(np.int32),
        cluster_names=cluster_names,
    )
