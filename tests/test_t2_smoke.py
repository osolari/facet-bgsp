"""Smoke test for the Tier-2 superposition driver.

Pins the sparsity-sweep schema (plan-2026-10-13T1908Z §T2):
- CSV + figure outputs on a tiny config,
- three methods x three regimes appear in the output,
- Hungarian-matched FACET recovery dominates PCA when the planted features
  fit in the hidden dim (very-sparse regime).
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest
import yaml

pytest.importorskip("torch")

from experiments import t2_superposition as t2


def _cfg() -> dict:
    return {
        "name": "t2_smoke",
        "seed_base": 1001,
        "replicates": 2,
        "model": {"n": 8, "d": 4, "importance_decay": 0.95},
        "sparsity_regimes": [
            {"label": "sparse", "sparsity": 0.12},
            {"label": "dense", "sparsity": 0.6},
        ],
        "training": {"batch_size": 256, "steps": 400, "lr": 0.02},
        "facet": {"gamma_candidates": [0.15, 0.20, 0.25, 0.12, 0.30]},
        "output": {"csv": "t2.csv", "figure": "t2.png"},
    }


def test_csv_contains_three_methods_per_regime(tmp_path: Path) -> None:
    cfg_path = tmp_path / "t2.yaml"
    cfg_path.write_text(yaml.safe_dump(_cfg()))
    rc = t2.main(["--config", str(cfg_path), "--output-dir", str(tmp_path), "--no-figure"])
    assert rc == 0
    with (tmp_path / "t2.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 * 2 * 3  # replicates x regimes x methods
    methods = {r["method"] for r in rows}
    regimes = {r["regime"] for r in rows}
    assert methods == {"native_neuron", "pca", "facet_asq"}
    assert regimes == {"sparse", "dense"}


def test_facet_hungarian_dominates_pca_in_sparse_regime(tmp_path: Path) -> None:
    cfg_path = tmp_path / "t2.yaml"
    cfg_path.write_text(yaml.safe_dump(_cfg()))
    rc = t2.main(["--config", str(cfg_path), "--output-dir", str(tmp_path), "--no-figure"])
    assert rc == 0
    with (tmp_path / "t2.csv").open() as f:
        rows = list(csv.DictReader(f))
    pca_vals = [
        float(r["mean_hungarian"]) for r in rows if r["method"] == "pca" and r["regime"] == "sparse"
    ]
    facet_vals = [
        float(r["mean_hungarian"])
        for r in rows
        if r["method"] == "facet_asq" and r["regime"] == "sparse"
    ]
    assert max(facet_vals) > 0.0, "FACET returned all-zeros on both replicates"
    assert (sum(facet_vals) / len(facet_vals)) >= (sum(pca_vals) / len(pca_vals)), (
        f"facet mean {facet_vals} vs pca mean {pca_vals}"
    )


@pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None, reason="matplotlib is optional")
def test_driver_renders_figure(tmp_path: Path) -> None:
    cfg_path = tmp_path / "t2.yaml"
    cfg_path.write_text(yaml.safe_dump(_cfg()))
    rc = t2.main(["--config", str(cfg_path), "--output-dir", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "t2.png").exists()
