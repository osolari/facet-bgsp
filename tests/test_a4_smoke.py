"""Smoke test for the A4 raw-vs-refit driver.

Pins:
- CSV + TeX + figure outputs on a tiny config,
- the A4 dominance invariant (refit >= raw) across all surviving cells.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest
import yaml

from experiments import a4_raw_vs_refit as a4

EXPECTED = {
    "seed",
    "noise_std",
    "margin_fraction",
    "support_matches",
    "raw_restricted_objective",
    "refit_restricted_objective",
    "refit_minus_raw",
}


def _cfg() -> dict:
    return {
        "name": "a4_smoke",
        "seed_base": 2026,
        "problem": {"p": 8, "n": 18, "k": 4, "singular_values": [2.0]},
        "grid": {"noise_std": [0.0, 0.1], "margin_fraction": [0.4], "replicates": 3},
        "rank2_cells": [{"lam1": 4.0, "lam2": 2.0, "label_noise": -0.5, "replicates": 3}],
        "solver": {"max_iterations": 60, "tolerance": 1.0e-5, "initial_angle_rad": 0.5},
        "objective": "squared",
        "output": {
            "csv": "results.csv",
            "figure": "figure.png",
            "table_tex": "table.tex",
        },
    }


def test_csv_schema_and_refit_dominates_raw(tmp_path: Path) -> None:
    cfg_path = tmp_path / "a4.yaml"
    cfg_path.write_text(yaml.safe_dump(_cfg()))
    rc = a4.main(
        ["--config", str(cfg_path), "--output-dir", str(tmp_path), "--no-figure", "--no-table"]
    )
    assert rc == 0
    with (tmp_path / "results.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert set(rows[0].keys()) == EXPECTED
    assert len(rows) > 0
    for r in rows:
        raw = float(r["raw_restricted_objective"])
        ref = float(r["refit_restricted_objective"])
        # A4 dominance: refit must be at least raw (modulo numerical noise).
        assert ref >= raw - 1e-10, (r, raw, ref)


def test_latex_table_written(tmp_path: Path) -> None:
    cfg_path = tmp_path / "a4.yaml"
    cfg_path.write_text(yaml.safe_dump(_cfg()))
    rc = a4.main(["--config", str(cfg_path), "--output-dir", str(tmp_path), "--no-figure"])
    assert rc == 0
    tex = (tmp_path / "table.tex").read_text()
    assert "\\begin{table}" in tex and "\\end{table}" in tex
    assert "\\label{tab:a4-raw-vs-refit}" in tex


@pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None, reason="matplotlib is optional")
def test_driver_renders_figure(tmp_path: Path) -> None:
    cfg_path = tmp_path / "a4.yaml"
    cfg_path.write_text(yaml.safe_dump(_cfg()))
    rc = a4.main(["--config", str(cfg_path), "--output-dir", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "figure.png").exists()
