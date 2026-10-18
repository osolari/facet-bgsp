"""Smoke test for the Q3 margin-identification experiment driver.

Verifies that:
- the driver loads a YAML config,
- runs the planted-spike x BlockSparseEstimator grid,
- writes a CSV with the expected columns,
- optionally renders a PNG without crashing.

Uses a tiny grid so the whole test completes in well under a second.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest
import yaml

from experiments import q3_margin_identification as q3

EXPECTED_COLUMNS = {
    "seed",
    "noise_std",
    "margin_fraction",
    "initial_angle_rad",
    "gamma",
    "achieved_margin",
    "iterations",
    "support_matches",
    "final_objective",
    "refit_objective",
}


def _tiny_config() -> dict:
    return {
        "name": "q3_smoke",
        "seed_base": 7,
        "problem": {
            "p": 8,
            "n": 16,
            "k": 3,
            "singular_values": [5.0],
            "noise_std": 0.0,
        },
        "grid": {
            "margin_fraction": [0.3, 0.7],
            "initial_angle_rad": [0.2, 0.5],
            "replicates": 2,
        },
        "solver": {"name": "safeguarded", "max_iterations": 200, "tolerance": 1.0e-8},
        "objective": "squared",
        "output": {"csv": "results.csv", "figure": "figure.png"},
    }


def test_driver_writes_csv_with_expected_schema(tmp_path: Path) -> None:
    config_path = tmp_path / "q3.yaml"
    config_path.write_text(yaml.safe_dump(_tiny_config()))

    rc = q3.main(["--config", str(config_path), "--output-dir", str(tmp_path), "--no-figure"])
    assert rc == 0

    csv_path = tmp_path / "results.csv"
    assert csv_path.exists()
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert set(rows[0].keys()) == EXPECTED_COLUMNS
    # 2 replicates * 2 margins * 2 angles = 8 runs
    assert len(rows) == 8
    for row in rows:
        assert row["support_matches"] in {"True", "False"}
        assert float(row["gamma"]) > 0
        assert int(row["iterations"]) >= 0


@pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None, reason="matplotlib is optional")
def test_driver_renders_figure(tmp_path: Path) -> None:
    config_path = tmp_path / "q3.yaml"
    config_path.write_text(yaml.safe_dump(_tiny_config()))
    rc = q3.main(["--config", str(config_path), "--output-dir", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "figure.png").exists()
    assert (tmp_path / "figure.png").stat().st_size > 1024
