"""Smoke test for the Q2 convergence-trajectories driver.

Pins CSV schema (one row per solver iteration) and figure render on a tiny grid.
"""

from __future__ import annotations

import csv
import importlib.util
import itertools
from pathlib import Path

import pytest
import yaml

from experiments import q2_convergence_trajectories as q2

EXPECTED = {"seed", "iteration", "objective", "step_norm", "residual", "support_size"}


def _cfg() -> dict:
    return {
        "name": "q2_smoke",
        "seed_base": 11,
        "problem": {"p": 8, "n": 18, "k": 4, "singular_values": [5.0], "noise_std": 0.0},
        "margin_fraction": 0.5,
        "initial_angle_rad": 0.5,
        "replicates": 2,
        "solver": {"max_iterations": 150, "tolerance": 1.0e-9},
        "objective": "squared",
        "output": {"csv": "results.csv", "figure": "figure.png"},
    }


def test_csv_schema_and_monotone_objective(tmp_path: Path) -> None:
    cfg_path = tmp_path / "q2.yaml"
    cfg_path.write_text(yaml.safe_dump(_cfg()))
    rc = q2.main(["--config", str(cfg_path), "--output-dir", str(tmp_path), "--no-figure"])
    assert rc == 0
    with (tmp_path / "results.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert set(rows[0].keys()) == EXPECTED
    assert len(rows) > 0

    # A6, objective non-decreasing within each seed's trajectory
    seeds = {int(r["seed"]) for r in rows}
    for seed in seeds:
        series = sorted(
            (r for r in rows if int(r["seed"]) == seed), key=lambda r: int(r["iteration"])
        )
        values = [float(r["objective"]) for r in series]
        for prev, curr in itertools.pairwise(values):
            assert curr >= prev - 1e-12, (seed, prev, curr)


@pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None, reason="matplotlib is optional")
def test_driver_renders_figure(tmp_path: Path) -> None:
    cfg_path = tmp_path / "q2.yaml"
    cfg_path.write_text(yaml.safe_dump(_cfg()))
    rc = q2.main(["--config", str(cfg_path), "--output-dir", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "figure.png").exists()
    assert (tmp_path / "figure.png").stat().st_size > 1024
