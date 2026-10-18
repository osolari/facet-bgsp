"""Smoke tests for the M9-Zheng-FACS driver.

Does not hit the 10x Genomics CDN: the loader's ``cache_path`` branch is
primed with a tiny synthetic ``(n_cells, n_hvg)`` matrix and the driver
runs end-to-end on that fixture.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import yaml

from experiments import m9_zheng_facs as m9zf


def _write_cache(cache_path: Path, *, n_per_pop: int = 12, n_hvg: int = 40) -> None:
    """Prime the ``_zheng_facs`` cache with a synthetic planted-cell dataset
    of 4 FACS populations x ``n_per_pop`` cells x ``n_hvg`` genes."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    population_names = ["CD14+ Monocyte", "CD56+ NK", "CD4+ Helper T", "Regulatory T"]
    population_labels = np.repeat(np.arange(len(population_names)), n_per_pop).astype(np.int32)
    # Marker-rich genes: first 4 HVGs each spike one population.
    hvg_symbols = ["LYZ", "GNLY", "IL7R", "FOXP3"] + [f"g{i}" for i in range(n_hvg - 4)]
    expression = rng.standard_normal((n_per_pop * len(population_names), n_hvg))
    for p in range(len(population_names)):
        expression[population_labels == p, p] += 3.0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        expression=expression,
        cell_barcodes=np.array([f"cell_{i}" for i in range(expression.shape[0])]),
        population_labels=population_labels,
        population_names=np.array(population_names),
        hvg_symbols=np.array(hvg_symbols),
        hvg_ids=np.array(hvg_symbols),
    )


def _write_config(tmp_path: Path, cache_path: Path) -> Path:
    cfg = {
        "name": "m9_zheng_facs_smoke",
        "seed": 7,
        "data": {
            "data_root": str(tmp_path / "raw"),
            "cache_path": str(cache_path),
            "cells_per_population": 12,
            "n_hvg": 40,
        },
        "decomposition": {"m": 3},
        "facet": {"gamma_candidates": [0.2, 0.8]},
        "truncated_pca": {"top_k_per_component": [4, 20]},
        "sparse_pca": {"alphas": [0.5]},
        "figure": {
            "hero_top_k_components": 2,
            "hero_top_k_genes": 4,
            "interpretability_top_k_genes": 5,
        },
        "interpretability": {"reference": "best_facet", "ami_tolerance": 0.1},
        "output": {
            "csv": "m9_zheng_facs.csv",
            "figure": "m9_zheng_facs.png",
            "interpretability_figure": "m9_zheng_facs_interp.png",
        },
    }
    cfg_path = tmp_path / "m9_zheng_facs.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


def test_driver_runs_end_to_end_from_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.npz"
    _write_cache(cache_path)
    cfg_path = _write_config(tmp_path, cache_path)
    rc = m9zf.main(["--config", str(cfg_path), "--output-dir", str(tmp_path), "--no-figure"])
    assert rc == 0
    rows = list(csv.DictReader((tmp_path / "m9_zheng_facs.csv").open()))
    assert rows, "driver produced no CSV rows"
    methods = {r["method"] for r in rows}
    # Every sweep arm must contribute at least one row.
    assert {"pca", "pca_truncated", "facet_asq", "sparse_pca_zou"} <= methods, methods
    for r in rows:
        assert int(r["n_cells"]) == 48
        assert int(r["n_hvg"]) == 40
        assert int(r["m"]) == 3
        total_nnz = int(r["total_nnz"])
        assert 0 < total_nnz <= 3 * 40


def test_marker_set_covers_all_populations() -> None:
    s = m9zf._marker_set()
    for genes in m9zf.MARKER_GENES_ZHENG_FACS.values():
        for g in genes:
            assert g in s
