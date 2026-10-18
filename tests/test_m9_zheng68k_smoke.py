"""Smoke tests for the M9-Zheng-68k driver.

Primes the ``_zheng68k`` cache with a tiny synthetic matrix so the test
runs without downloading the 124 MB fresh-68k tarball or the paper
annotation TSV.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import yaml

from experiments import m9_zheng68k as m9z68


def _write_cache(cache_path: Path) -> None:
    rng = np.random.default_rng(1)
    cell_type_names = [
        "CD14+ Monocyte",
        "CD19+ B",
        "CD4+ Helper1",
        "CD56+ NK",
        "Dendritic",
    ]
    per_type = 10
    expression = rng.standard_normal((per_type * len(cell_type_names), 50))
    labels = np.repeat(np.arange(len(cell_type_names)), per_type).astype(np.int32)
    for p in range(len(cell_type_names)):
        expression[labels == p, p] += 2.5
    hvg_symbols = ["CD14", "MS4A1", "IL7R", "GNLY", "FCER1A"] + [f"g{i}" for i in range(45)]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        expression=expression,
        cell_barcodes=np.array([f"bc_{i}" for i in range(expression.shape[0])]),
        cell_type_labels=labels,
        cell_type_names=np.array(cell_type_names),
        hvg_symbols=np.array(hvg_symbols),
        hvg_ids=np.array(hvg_symbols),
    )


def _write_config(tmp_path: Path, cache_path: Path) -> Path:
    cfg = {
        "name": "m9_zheng68k_smoke",
        "seed": 11,
        "data": {
            "data_root": str(tmp_path / "raw"),
            "labels_path": str(tmp_path / "labels.tsv"),
            "cache_path": str(cache_path),
            "n_hvg": 50,
        },
        "decomposition": {"m": 3},
        "facet": {"gamma_candidates": [0.3, 1.0]},
        "truncated_pca": {"top_k_per_component": [5, 25]},
        "sparse_pca": {"alphas": [0.5]},
        "figure": {
            "hero_top_k_components": 2,
            "hero_top_k_genes": 4,
            "interpretability_top_k_genes": 5,
        },
        "interpretability": {"reference": "best_facet", "ami_tolerance": 0.1},
        "output": {
            "csv": "m9_zheng68k.csv",
            "figure": "m9_zheng68k.png",
            "interpretability_figure": "m9_zheng68k_interp.png",
        },
    }
    cfg_path = tmp_path / "m9_zheng68k.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


def test_driver_runs_end_to_end_from_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.npz"
    _write_cache(cache_path)
    cfg_path = _write_config(tmp_path, cache_path)
    rc = m9z68.main(["--config", str(cfg_path), "--output-dir", str(tmp_path), "--no-figure"])
    assert rc == 0
    rows = list(csv.DictReader((tmp_path / "m9_zheng68k.csv").open()))
    assert rows
    methods = {r["method"] for r in rows}
    assert {"pca", "pca_truncated", "facet_asq", "sparse_pca_zou"} <= methods
    for r in rows:
        assert int(r["n_cells"]) == 50
        assert int(r["n_hvg"]) == 50
        assert int(r["m"]) == 3


def test_marker_set_includes_pbmc_markers() -> None:
    s = m9z68._marker_set()
    assert {"CD14", "MS4A1", "IL7R", "GNLY"} <= s
