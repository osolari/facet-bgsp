"""Zheng17 FACS-sorted PBMC subpopulations (native cell-type labels).

Downloads 10x Genomics bead-purified subpopulations from the 2017 Zheng
massively parallel transcriptomics paper and concatenates them into a
single ``(n_cells, n_hvg)`` matrix with per-cell NATIVE cell-type labels
(the FACS gate, not a clustering output). Three available populations
return 403 on the 10x CDN at the sample names we tried
(cd19_b, cd8_t_cytotoxic, cd8_naive_t); the 8 that resolve are used
instead. Each is capped at ``cells_per_population`` cells to keep the
preprocessing matrix a manageable size and to balance the native-label
distribution.

Preprocessing matches ``_scrna.preprocess_pbmc3k`` so the same
decomposition sweep applies without changes: per-cell total-count
normalisation, log1p, Seurat-style HVG selection, per-gene z-score.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tarfile
import urllib.request
from dataclasses import dataclass
from typing import Any

import numpy as np

ZHENG_FACS_URL = (
    "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/"
    "{sample}/{sample}_filtered_gene_bc_matrices.tar.gz"
)

# The 8 FACS populations whose matrices resolve on the 10x CDN under the
# default sample name scheme (verified 2026-10-16). Native labels are the
# dict keys; the right-hand strings are the human-readable display names.
POPULATIONS: dict[str, str] = {
    "cd14_monocytes": "CD14+ Monocyte",
    "cd34": "CD34+ (HSPC)",
    "cd4_t_helper": "CD4+ Helper T",
    "cd56_nk": "CD56+ NK",
    "memory_t": "Memory T",
    "naive_cytotoxic": "Naive Cytotoxic",
    "naive_t": "Naive T",
    "regulatory_t": "Regulatory T",
}

# Canonical marker genes for the 8 FACS populations; used only to colour
# the loading bars and compute marker-gene enrichment. Overlaps with the
# PBMC3K list where populations coincide; extended for CD34 / Treg /
# memory + naive T subsets.
MARKER_GENES_ZHENG_FACS: dict[str, list[str]] = {
    "CD14+ Monocyte": ["LYZ", "CD14", "S100A9", "FCN1"],
    "CD34+ (HSPC)": ["CD34", "KIT", "PROM1", "SPINK2"],
    "CD4+ Helper T": ["CD4", "IL7R", "CCR7"],
    "CD56+ NK": ["GNLY", "NKG7", "KLRD1", "NCAM1"],
    "Memory T": ["CD3E", "CD3D", "GZMK", "CCR7"],
    "Naive Cytotoxic": ["CD8A", "CD8B", "CCR7", "SELL"],
    "Naive T": ["CD4", "CCR7", "SELL", "LEF1"],
    "Regulatory T": ["FOXP3", "IL2RA", "CTLA4", "TNFRSF18"],
}


@dataclass(frozen=True)
class ZhengFacsData:
    expression: np.ndarray  # (n_cells, n_hvg) log + HVG + z-scored
    cell_barcodes: list[str]
    population_labels: np.ndarray  # (n_cells,) int, index into population_names
    population_names: list[str]  # ordered display names
    hvg_symbols: list[str]
    hvg_ids: list[str]


def _fetch_url(url: str, timeout: int = 180) -> bytes:
    """Download ``url`` into memory. Tries urllib first; on SSL failure
    (common on networks with cert interception) falls back to the system
    ``curl`` binary, which uses the OS cert chain."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read()
    except Exception:
        curl = shutil.which("curl")
        if curl is None:
            raise
        proc = subprocess.run(
            [curl, "-sSL", "--max-time", str(timeout), url],
            check=True,
            capture_output=True,
        )
        return proc.stdout


def _download_and_extract(sample: str, dest_dir: str) -> str:
    """Fetch one FACS-sorted population tarball and extract to ``dest_dir``.
    Returns the path to the inner matrix dir containing matrix.mtx."""
    url = ZHENG_FACS_URL.format(sample=sample)
    os.makedirs(dest_dir, exist_ok=True)
    marker = next(
        (root for root, _d, files in os.walk(dest_dir) if "matrix.mtx" in files),
        None,
    )
    if marker is not None:
        return marker
    payload = _fetch_url(url)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tf:
        tf.extractall(dest_dir)
    for root, _dirs, files in os.walk(dest_dir):
        if "matrix.mtx" in files:
            return root
    raise FileNotFoundError(f"no matrix.mtx under {dest_dir}")


def _load_single_population(mtx_dir: str) -> tuple[Any, list[str], list[str], list[str]]:
    """Return (sparse (cells, genes), barcodes, gene_ids, gene_symbols)."""
    from scipy.io import mmread
    from scipy.sparse import csr_matrix

    counts = csr_matrix(mmread(os.path.join(mtx_dir, "matrix.mtx")).tocsr().T)
    with open(os.path.join(mtx_dir, "barcodes.tsv")) as f:
        barcodes = [line.strip() for line in f if line.strip()]
    gene_ids, gene_symbols = [], []
    with open(os.path.join(mtx_dir, "genes.tsv")) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                gene_ids.append(parts[0])
                gene_symbols.append(parts[1])
    return counts, barcodes, gene_ids, gene_symbols


def load_zheng_facs(
    *,
    data_root: str,
    cache_path: str | None = None,
    cells_per_population: int = 2000,
    min_genes_per_cell: int = 200,
    min_cells_per_gene: int = 3,
    n_hvg: int = 2000,
    target_total: float = 1e4,
    seed: int = 2046,
) -> ZhengFacsData:
    """Download, concatenate, and preprocess the Zheng17 FACS-sorted
    PBMC subpopulations. The matrices live under
    ``<data_root>/<sample>/`` and are downloaded on first use."""
    if cache_path and os.path.exists(cache_path):
        payload = np.load(cache_path, allow_pickle=True)
        return ZhengFacsData(
            expression=payload["expression"],
            cell_barcodes=list(payload["cell_barcodes"]),
            population_labels=payload["population_labels"],
            population_names=list(payload["population_names"]),
            hvg_symbols=list(payload["hvg_symbols"]),
            hvg_ids=list(payload["hvg_ids"]),
        )

    rng = np.random.default_rng(seed)
    population_names: list[str] = []
    gene_id_master: list[str] | None = None
    gene_symbol_master: list[str] | None = None
    sampled_counts: list[Any] = []
    sampled_barcodes: list[str] = []
    sampled_labels: list[int] = []

    for label_idx, (sample, display) in enumerate(POPULATIONS.items()):
        mtx_dir = _download_and_extract(sample, os.path.join(data_root, sample))
        counts, barcodes, gene_ids, gene_symbols = _load_single_population(mtx_dir)
        if gene_id_master is None:
            gene_id_master = gene_ids
            gene_symbol_master = gene_symbols
        else:
            if gene_ids != gene_id_master:
                # 10x matrices across these samples share the GRCh38 reference
                # and should have identical gene lists; defend anyway.
                raise ValueError(
                    f"gene list mismatch between {sample} and the first "
                    "population; cannot concatenate without a gene-index merge"
                )
        if counts.shape[0] > cells_per_population:
            keep = rng.choice(counts.shape[0], size=cells_per_population, replace=False)
            keep.sort()
            counts = counts[keep]
            barcodes = [barcodes[i] for i in keep]
        sampled_counts.append(counts)
        sampled_barcodes.extend([f"{sample}:{b}" for b in barcodes])
        sampled_labels.extend([label_idx] * counts.shape[0])
        population_names.append(display)

    from scipy.sparse import vstack

    assert gene_id_master is not None and gene_symbol_master is not None
    merged = vstack(sampled_counts).tocsr()
    labels = np.asarray(sampled_labels, dtype=np.int32)

    # Filter cells with too few expressed genes.
    genes_per_cell = np.asarray((merged > 0).sum(axis=1)).ravel()
    keep_cells = genes_per_cell >= min_genes_per_cell
    merged = merged[keep_cells]
    sampled_barcodes = [b for b, k in zip(sampled_barcodes, keep_cells, strict=True) if k]
    labels = labels[keep_cells]

    # Filter genes present in too few cells.
    cells_per_gene = np.asarray((merged > 0).sum(axis=0)).ravel()
    keep_genes = cells_per_gene >= min_cells_per_gene
    merged = merged[:, keep_genes]
    gene_ids = [g for g, k in zip(gene_id_master, keep_genes, strict=True) if k]
    gene_symbols = [g for g, k in zip(gene_symbol_master, keep_genes, strict=True) if k]

    # Total-count normalise per cell, then log1p.
    cell_totals = np.asarray(merged.sum(axis=1)).ravel()
    scale = target_total / np.maximum(cell_totals, 1.0)
    merged_scaled = merged.multiply(scale[:, None])
    log_counts = merged_scaled.log1p().toarray().astype(np.float64)

    # Highly-variable-gene selection (Seurat-style dispersion ranking).
    mean_expr = log_counts.mean(axis=0)
    var_expr = log_counts.var(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dispersion = np.where(mean_expr > 1e-6, var_expr / mean_expr, 0.0)
    hvg_rank = np.sort(np.argsort(-dispersion)[:n_hvg])
    hvg_symbols = [gene_symbols[i] for i in hvg_rank]
    hvg_ids = [gene_ids[i] for i in hvg_rank]
    expr = log_counts[:, hvg_rank]

    # z-score each HVG across cells.
    mu = expr.mean(axis=0, keepdims=True)
    sd = expr.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    expr = (expr - mu) / sd

    data = ZhengFacsData(
        expression=expr,
        cell_barcodes=sampled_barcodes,
        population_labels=labels,
        population_names=population_names,
        hvg_symbols=hvg_symbols,
        hvg_ids=hvg_ids,
    )

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez_compressed(
            cache_path,
            expression=expr,
            cell_barcodes=np.array(sampled_barcodes),
            population_labels=labels,
            population_names=np.array(population_names),
            hvg_symbols=np.array(hvg_symbols),
            hvg_ids=np.array(hvg_ids),
        )
    return data
