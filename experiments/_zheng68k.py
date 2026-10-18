"""Zheng17 68k PBMC (fresh donor A) with paper-published cell-type labels.

Pairs the 10x Genomics ``fresh_68k_pbmc_donor_a`` filtered matrix with
the FACS-correlated per-barcode cell-type assignments published in the
Zheng et al. 2017 analysis repository (10XGenomics/single-cell-3prime-
paper). Labels come from correlation of each cell with the FACS-sorted
reference profiles, NOT from any clustering of the 68k matrix, so AMI
against this label is a legitimate cross-method benchmark.
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

ZHENG68K_MATRIX_URL = (
    "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/fresh_68k_pbmc_donor_a/"
    "fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz"
)
ZHENG68K_LABELS_URL = (
    "https://raw.githubusercontent.com/10XGenomics/single-cell-3prime-paper/"
    "master/pbmc68k_analysis/68k_pbmc_barcodes_annotation.tsv"
)


@dataclass(frozen=True)
class Zheng68kData:
    expression: np.ndarray
    cell_barcodes: list[str]
    cell_type_labels: np.ndarray  # (n_cells,) int
    cell_type_names: list[str]
    hvg_symbols: list[str]
    hvg_ids: list[str]


def _fetch(url: str, timeout: int = 180) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read()
    except Exception:
        curl = shutil.which("curl")
        if curl is None:
            raise
        return subprocess.run(
            [curl, "-sSL", "--max-time", str(timeout), url],
            check=True,
            capture_output=True,
        ).stdout


def _extract(dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    for root, _d, files in os.walk(dest_dir):
        if "matrix.mtx" in files:
            return root
    with tarfile.open(fileobj=io.BytesIO(_fetch(ZHENG68K_MATRIX_URL)), mode="r:gz") as tf:
        tf.extractall(dest_dir)
    for root, _d, files in os.walk(dest_dir):
        if "matrix.mtx" in files:
            return root
    raise FileNotFoundError(f"no matrix.mtx under {dest_dir}")


def _load_mtx(mtx_dir: str) -> tuple[Any, list[str], list[str], list[str]]:
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


def _load_labels(labels_path: str) -> dict[str, str]:
    """Parse the TSV; return barcode -> celltype."""
    out: dict[str, str] = {}
    with open(labels_path) as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            bc_idx = header.index("barcodes")
            ct_idx = header.index("celltype")
        except ValueError as exc:
            raise RuntimeError(f"unexpected header {header}") from exc
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(bc_idx, ct_idx):
                continue
            out[parts[bc_idx]] = parts[ct_idx]
    return out


def load_zheng68k(
    *,
    data_root: str,
    labels_path: str,
    cache_path: str | None = None,
    min_genes_per_cell: int = 200,
    min_cells_per_gene: int = 3,
    n_hvg: int = 2000,
    target_total: float = 1e4,
) -> Zheng68kData:
    """Download + preprocess the fresh 68k donor-A PBMC matrix with the
    paper-published FACS-correlated per-barcode labels. Keeps only cells
    whose barcode has a published label."""
    if cache_path and os.path.exists(cache_path):
        payload = np.load(cache_path, allow_pickle=True)
        return Zheng68kData(
            expression=payload["expression"],
            cell_barcodes=list(payload["cell_barcodes"]),
            cell_type_labels=payload["cell_type_labels"],
            cell_type_names=list(payload["cell_type_names"]),
            hvg_symbols=list(payload["hvg_symbols"]),
            hvg_ids=list(payload["hvg_ids"]),
        )

    if not os.path.exists(labels_path):
        os.makedirs(os.path.dirname(labels_path) or ".", exist_ok=True)
        with open(labels_path, "wb") as f:
            f.write(_fetch(ZHENG68K_LABELS_URL))
    bc_to_type = _load_labels(labels_path)

    mtx_dir = _extract(data_root)
    counts, barcodes, gene_ids, gene_symbols = _load_mtx(mtx_dir)

    # Match barcodes against the label table; drop unlabeled cells.
    labeled = np.array([b in bc_to_type for b in barcodes], dtype=bool)
    counts = counts[labeled]
    barcodes = [b for b, k in zip(barcodes, labeled, strict=True) if k]
    cell_type_strings = [bc_to_type[b] for b in barcodes]

    # Standard QC filters.
    genes_per_cell = np.asarray((counts > 0).sum(axis=1)).ravel()
    keep_cells = genes_per_cell >= min_genes_per_cell
    counts = counts[keep_cells]
    barcodes = [b for b, k in zip(barcodes, keep_cells, strict=True) if k]
    cell_type_strings = [c for c, k in zip(cell_type_strings, keep_cells, strict=True) if k]

    cells_per_gene = np.asarray((counts > 0).sum(axis=0)).ravel()
    keep_genes = cells_per_gene >= min_cells_per_gene
    counts = counts[:, keep_genes]
    gene_ids = [g for g, k in zip(gene_ids, keep_genes, strict=True) if k]
    gene_symbols = [g for g, k in zip(gene_symbols, keep_genes, strict=True) if k]

    # log1p normalisation.
    cell_totals = np.asarray(counts.sum(axis=1)).ravel()
    scale = target_total / np.maximum(cell_totals, 1.0)
    counts_scaled = counts.multiply(scale[:, None])
    log_counts = counts_scaled.log1p().toarray().astype(np.float64)

    mean_expr = log_counts.mean(axis=0)
    var_expr = log_counts.var(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dispersion = np.where(mean_expr > 1e-6, var_expr / mean_expr, 0.0)
    hvg_rank = np.sort(np.argsort(-dispersion)[:n_hvg])
    hvg_symbols = [gene_symbols[i] for i in hvg_rank]
    hvg_ids = [gene_ids[i] for i in hvg_rank]
    expr = log_counts[:, hvg_rank]

    mu = expr.mean(axis=0, keepdims=True)
    sd = expr.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    expr = (expr - mu) / sd

    cell_type_names = sorted(set(cell_type_strings))
    name_to_idx = {name: i for i, name in enumerate(cell_type_names)}
    cell_type_labels = np.array([name_to_idx[s] for s in cell_type_strings], dtype=np.int32)

    data = Zheng68kData(
        expression=expr,
        cell_barcodes=barcodes,
        cell_type_labels=cell_type_labels,
        cell_type_names=cell_type_names,
        hvg_symbols=hvg_symbols,
        hvg_ids=hvg_ids,
    )

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez_compressed(
            cache_path,
            expression=expr,
            cell_barcodes=np.array(barcodes),
            cell_type_labels=cell_type_labels,
            cell_type_names=np.array(cell_type_names),
            hvg_symbols=np.array(hvg_symbols),
            hvg_ids=np.array(hvg_ids),
        )
    return data
