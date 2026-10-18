"""Smoke tests for the M9-Tabula-Muris driver.

Primes the ``_tabula_muris`` cache with a tiny synthetic matrix so the
test does not download the 290 MB FACS.zip or the annotations CSV.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import yaml

from experiments import m9_tabula_muris as m9tm


def _write_cache(cache_path: Path) -> None:
    rng = np.random.default_rng(2)
    cell_type_names = ["B cell", "T cell", "NK cell", "Macrophage", "Granulocyte"]
    tissue_names = ["Spleen", "Thymus"]
    per_type = 8
    n_cells = per_type * len(cell_type_names)
    expression = rng.standard_normal((n_cells, 40))
    labels = np.repeat(np.arange(len(cell_type_names)), per_type).astype(np.int32)
    tissues = np.tile(np.arange(len(tissue_names)), n_cells // len(tissue_names) + 1).astype(
        np.int32
    )[:n_cells]
    for p in range(len(cell_type_names)):
        expression[labels == p, p] += 2.0
    hvg_symbols = ["Cd19", "Cd3e", "Nkg7", "Cd68", "S100a8"] + [f"g{i}" for i in range(35)]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        expression=expression,
        cell_barcodes=np.array([f"bc_{i}" for i in range(n_cells)]),
        cell_type_labels=labels,
        cell_type_names=np.array(cell_type_names),
        tissue_labels=tissues,
        tissue_names=np.array(tissue_names),
        hvg_symbols=np.array(hvg_symbols),
        hvg_ids=np.array(hvg_symbols),
    )


def _write_config(tmp_path: Path, cache_path: Path) -> Path:
    cfg = {
        "name": "m9_tabula_muris_smoke",
        "seed": 13,
        "data": {
            "data_root": str(tmp_path / "raw"),
            "cache_path": str(cache_path),
            "tissues": ["Spleen", "Thymus"],
            "cells_per_tissue": None,
            "n_hvg": 40,
        },
        "decomposition": {"m": 3},
        "facet": {"gamma_candidates": [0.3, 1.0]},
        "truncated_pca": {"top_k_per_component": [5, 20]},
        "sparse_pca": {"alphas": [0.5]},
        "figure": {
            "hero_top_k_components": 2,
            "hero_top_k_genes": 4,
            "interpretability_top_k_genes": 5,
        },
        "interpretability": {"reference": "best_facet", "ami_tolerance": 0.1},
        "output": {
            "csv": "m9_tabula_muris.csv",
            "figure": "m9_tabula_muris.png",
            "interpretability_figure": "m9_tabula_muris_interp.png",
        },
    }
    cfg_path = tmp_path / "m9_tabula_muris.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


def test_driver_runs_end_to_end_from_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.npz"
    _write_cache(cache_path)
    cfg_path = _write_config(tmp_path, cache_path)
    rc = m9tm.main(["--config", str(cfg_path), "--output-dir", str(tmp_path), "--no-figure"])
    assert rc == 0
    rows = list(csv.DictReader((tmp_path / "m9_tabula_muris.csv").open()))
    assert rows
    methods = {r["method"] for r in rows}
    assert {"pca", "pca_truncated", "facet_asq", "sparse_pca_zou"} <= methods
    for r in rows:
        assert int(r["n_cells"]) == 40
        assert int(r["n_hvg"]) == 40
        assert int(r["m"]) == 3


def test_mouse_marker_set_includes_canonical_programs() -> None:
    s = m9tm._marker_set()
    assert {"Cd3e", "Cd19", "Nkg7", "Cd68"} <= s
