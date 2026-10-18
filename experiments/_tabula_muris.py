"""Tabula Muris FACS tissue counts loader (mouse scRNA-seq with native
cell_ontology_class labels).

Downloads the per-tissue count CSVs from the Tabula Muris FACS v2
figshare record (10.6084/m9.figshare.5829687) and the matching
``annotations_facs.csv`` with per-cell ``cell_ontology_class`` labels.
Reads one or more tissues, concatenates them with native labels, and
runs the same log1p + HVG + z-score pipeline as ``_scrna.py`` so the
decomposition sweep applies without changes.

Smart-seq2 counts are gene-length-biased relative to UMI 10x counts, but
the dispersion-based HVG selection + z-score normalisation remain the
canonical preprocessing choices in the Tabula Muris paper and work
without modification here.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import urllib.request
import zipfile
from dataclasses import dataclass

import numpy as np

TABULA_MURIS_ZIP_URL = "https://ndownloader.figshare.com/files/10700143"
TABULA_MURIS_ANNOTATIONS_URL = "https://ndownloader.figshare.com/files/13088129"

# Mouse immune / stromal marker genes keyed by canonical cell type.
# Used only for marker-gene enrichment metrics; clustering labels come
# from the Tabula Muris annotations.csv (cell_ontology_class). Mouse
# gene symbols are title-case (Cd3e, not CD3E).
MARKER_GENES_TABULA_MURIS: dict[str, list[str]] = {
    "B cell": ["Cd19", "Cd79a", "Cd79b", "Ms4a1", "Cd22"],
    "T cell": ["Cd3d", "Cd3e", "Cd3g", "Cd4", "Cd8a"],
    "NK cell": ["Nkg7", "Klrb1c", "Ncr1", "Gnly"],
    "Macrophage": ["Cd68", "Lyz2", "Fcgr3", "Cd14", "Adgre1"],
    "Dendritic cell": ["Itgax", "Cd83", "H2-Ab1", "Cd86"],
    "Granulocyte": ["S100a8", "S100a9", "Ly6g", "Mpo"],
    "Hematopoietic stem": ["Kit", "Cd34", "Mpl", "Flt3"],
    "Erythrocyte": ["Hbb-bs", "Hbb-bt", "Hba-a1", "Hba-a2", "Alas2"],
}


@dataclass(frozen=True)
class TabulaMurisData:
    expression: np.ndarray  # (n_cells, n_hvg)
    cell_barcodes: list[str]
    cell_type_labels: np.ndarray  # (n_cells,) int into cell_type_names
    cell_type_names: list[str]
    tissue_labels: np.ndarray  # (n_cells,) int into tissue_names
    tissue_names: list[str]
    hvg_symbols: list[str]
    hvg_ids: list[str]


def _fetch_to(url: str, dest_path: str, timeout: int = 600) -> None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(dest_path, "wb") as f:
            shutil.copyfileobj(resp, f)
    except Exception:
        curl = shutil.which("curl")
        if curl is None:
            raise
        subprocess.run([curl, "-sSL", "--max-time", str(timeout), "-o", dest_path, url], check=True)


def _ensure_tissue_csv(tissue: str, data_root: str) -> str:
    """Return the path to ``<data_root>/FACS/<tissue>-counts.csv``, downloading
    + extracting from the figshare zip on demand."""
    tissue_csv = os.path.join(data_root, "FACS", f"{tissue}-counts.csv")
    if os.path.exists(tissue_csv):
        return tissue_csv
    zip_path = os.path.join(data_root, "FACS.zip")
    if not os.path.exists(zip_path):
        os.makedirs(data_root, exist_ok=True)
        _fetch_to(TABULA_MURIS_ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        inner = f"FACS/{tissue}-counts.csv"
        if inner not in zf.namelist():
            raise FileNotFoundError(f"{inner} not in {zip_path}")
        zf.extract(inner, data_root)
    return tissue_csv


def _ensure_annotations(data_root: str) -> str:
    annotations = os.path.join(data_root, "annotations_facs.csv")
    if not os.path.exists(annotations):
        os.makedirs(data_root, exist_ok=True)
        _fetch_to(TABULA_MURIS_ANNOTATIONS_URL, annotations)
    return annotations


def _load_annotations(path: str) -> dict[str, tuple[str, str]]:
    """Return cell -> (tissue, cell_ontology_class)."""
    out: dict[str, tuple[str, str]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cell = row.get("cell", "").strip()
            cell_type = row.get("cell_ontology_class", "").strip()
            tissue = row.get("tissue", "").strip()
            if cell and cell_type and tissue:
                out[cell] = (tissue, cell_type)
    return out


def _load_tissue_csv(csv_path: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Return (counts[cells, genes] (float32), cell_ids, gene_symbols).
    Memory: the Tabula Muris Smart-seq2 tissues range from ~500 to ~5000
    cells x ~24000 genes; dense float32 fits in <1 GB for any one tissue."""
    gene_symbols: list[str] = []
    cell_ids: list[str] = []
    # First pass: read header for cell IDs.
    with open(csv_path, newline="") as f:
        header = next(csv.reader(f))
        cell_ids = header[1:]
        rows: list[np.ndarray] = []
        for parts in csv.reader(f):
            if not parts:
                continue
            gene_symbols.append(parts[0].strip('"'))
            rows.append(np.asarray(parts[1:], dtype=np.float32))
    counts_genes_x_cells = np.vstack(rows)
    return counts_genes_x_cells.T, cell_ids, gene_symbols


def load_tabula_muris(
    *,
    data_root: str,
    tissues: list[str],
    cache_path: str | None = None,
    cells_per_tissue: int | None = None,
    min_genes_per_cell: int = 200,
    min_cells_per_gene: int = 3,
    n_hvg: int = 2000,
    target_total: float = 1e4,
    seed: int = 2048,
) -> TabulaMurisData:
    """Download, concatenate, and preprocess one or more Tabula Muris
    FACS tissues. Keeps only cells whose barcode has a published
    ``cell_ontology_class`` annotation. If ``cells_per_tissue`` is set,
    each tissue is subsampled to that cap (per-tissue, before merge)."""
    if cache_path and os.path.exists(cache_path):
        payload = np.load(cache_path, allow_pickle=True)
        return TabulaMurisData(
            expression=payload["expression"],
            cell_barcodes=list(payload["cell_barcodes"]),
            cell_type_labels=payload["cell_type_labels"],
            cell_type_names=list(payload["cell_type_names"]),
            tissue_labels=payload["tissue_labels"],
            tissue_names=list(payload["tissue_names"]),
            hvg_symbols=list(payload["hvg_symbols"]),
            hvg_ids=list(payload["hvg_ids"]),
        )

    rng = np.random.default_rng(seed)
    annotations_path = _ensure_annotations(data_root)
    bc_to_label = _load_annotations(annotations_path)

    master_gene_symbols: list[str] | None = None
    counts_blocks: list[np.ndarray] = []
    barcodes: list[str] = []
    cell_type_strings: list[str] = []
    tissue_strings: list[str] = []

    for tissue in tissues:
        csv_path = _ensure_tissue_csv(tissue, data_root)
        tissue_counts, tissue_cells, tissue_genes = _load_tissue_csv(csv_path)
        if master_gene_symbols is None:
            master_gene_symbols = tissue_genes
        else:
            if tissue_genes != master_gene_symbols:
                # Fall back to intersect-based alignment if any tissue has a
                # slightly different gene list (shouldn't happen in TM FACS v2,
                # all tissues share the same gencode M14 reference).
                raise ValueError(
                    f"{tissue} gene list differs from the first tissue's list; "
                    "need gene-index alignment to concatenate"
                )
        labeled_mask = np.array(
            [c in bc_to_label and bc_to_label[c][0] == tissue for c in tissue_cells],
            dtype=bool,
        )
        tissue_counts = tissue_counts[labeled_mask]
        tissue_cells = [c for c, k in zip(tissue_cells, labeled_mask, strict=True) if k]
        if cells_per_tissue is not None and tissue_counts.shape[0] > cells_per_tissue:
            keep = rng.choice(tissue_counts.shape[0], size=cells_per_tissue, replace=False)
            keep.sort()
            tissue_counts = tissue_counts[keep]
            tissue_cells = [tissue_cells[i] for i in keep]
        counts_blocks.append(tissue_counts)
        barcodes.extend(tissue_cells)
        for c in tissue_cells:
            _tissue, ct = bc_to_label[c]
            cell_type_strings.append(ct)
            tissue_strings.append(tissue)

    assert master_gene_symbols is not None
    merged = np.vstack(counts_blocks).astype(np.float64)

    # QC: cells with too few expressed genes.
    genes_per_cell = (merged > 0).sum(axis=1)
    keep_cells = genes_per_cell >= min_genes_per_cell
    merged = merged[keep_cells]
    barcodes = [b for b, k in zip(barcodes, keep_cells, strict=True) if k]
    cell_type_strings = [c for c, k in zip(cell_type_strings, keep_cells, strict=True) if k]
    tissue_strings = [t for t, k in zip(tissue_strings, keep_cells, strict=True) if k]

    # QC: genes present in too few cells.
    cells_per_gene = (merged > 0).sum(axis=0)
    keep_genes = cells_per_gene >= min_cells_per_gene
    merged = merged[:, keep_genes]
    gene_symbols = [g for g, k in zip(master_gene_symbols, keep_genes, strict=True) if k]

    # log1p normalisation + HVG + z-score.
    cell_totals = merged.sum(axis=1)
    scale = target_total / np.maximum(cell_totals, 1.0)
    log_counts = np.log1p(merged * scale[:, None])

    mean_expr = log_counts.mean(axis=0)
    var_expr = log_counts.var(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dispersion = np.where(mean_expr > 1e-6, var_expr / mean_expr, 0.0)
    hvg_rank = np.sort(np.argsort(-dispersion)[:n_hvg])
    hvg_symbols = [gene_symbols[i] for i in hvg_rank]
    hvg_ids = list(hvg_symbols)  # TM CSVs don't carry gene IDs separately
    expr = log_counts[:, hvg_rank]

    mu = expr.mean(axis=0, keepdims=True)
    sd = expr.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    expr = (expr - mu) / sd

    cell_type_names = sorted(set(cell_type_strings))
    tissue_names = sorted(set(tissue_strings))
    ct_to_idx = {name: i for i, name in enumerate(cell_type_names)}
    tissue_to_idx = {name: i for i, name in enumerate(tissue_names)}
    cell_type_labels = np.array([ct_to_idx[s] for s in cell_type_strings], dtype=np.int32)
    tissue_labels = np.array([tissue_to_idx[t] for t in tissue_strings], dtype=np.int32)

    data = TabulaMurisData(
        expression=expr,
        cell_barcodes=barcodes,
        cell_type_labels=cell_type_labels,
        cell_type_names=cell_type_names,
        tissue_labels=tissue_labels,
        tissue_names=tissue_names,
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
            tissue_labels=tissue_labels,
            tissue_names=np.array(tissue_names),
            hvg_symbols=np.array(hvg_symbols),
            hvg_ids=np.array(hvg_ids),
        )
    return data
