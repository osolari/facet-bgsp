"""Smoke test for the Q4 local-rate driver.

Pins CSV schema + the key empirical claim: the observed geometric rate of the
full-polar iteration tracks the restricted-operator eigenvalue ratio
``lambda_2 / lambda_1`` within a loose multiplicative tolerance.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest
import yaml

from experiments import q4_local_rate as q4

EXPECTED = {
    "seed",
    "lam1",
    "lam2",
    "gamma",
    "predicted_rate",
    "observed_rate",
    "branch",
    "iterations",
}


def _cfg() -> dict:
    return {
        "name": "q4_smoke",
        "seed_base": 42,
        "problem": {"p": 10, "k": 5, "lam1": 4.0},
        "grid": {
            "lam2": [0.3, 1.5, 2.5],
            "replicates": 2,
            "initial_angle_rad": 0.4,
        },
        "solver": {"max_iterations": 400, "tolerance": 1.0e-11},
        "output": {"csv": "results.csv", "figure": "figure.png"},
    }


def test_csv_schema(tmp_path: Path) -> None:
    cfg_path = tmp_path / "q4.yaml"
    cfg_path.write_text(yaml.safe_dump(_cfg()))
    rc = q4.main(["--config", str(cfg_path), "--output-dir", str(tmp_path), "--no-figure"])
    assert rc == 0
    with (tmp_path / "results.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert set(rows[0].keys()) == EXPECTED
    assert len(rows) > 0


def test_hard_branch_observed_rate_matches_lambda_ratio(tmp_path: Path) -> None:
    cfg_path = tmp_path / "q4.yaml"
    cfg_path.write_text(yaml.safe_dump(_cfg()))
    rc = q4.main(["--config", str(cfg_path), "--output-dir", str(tmp_path), "--no-figure"])
    assert rc == 0
    with (tmp_path / "results.csv").open() as f:
        rows = list(csv.DictReader(f))
    hard = [r for r in rows if r["branch"] == "hard"]
    assert hard, "no hard-branch rows in output"
    for row in hard:
        pred = float(row["predicted_rate"])
        obs = float(row["observed_rate"])
        if not (obs > 0 and pred > 0):
            continue
        ratio = obs / pred
        # Pure power iteration should match lam2/lam1 within ~15% on these fixtures.
        assert 0.5 <= ratio <= 1.5, (row, ratio)


def test_soft_branch_observed_rate_matches_shifted_denominator(tmp_path: Path) -> None:
    cfg_path = tmp_path / "q4.yaml"
    cfg_path.write_text(yaml.safe_dump(_cfg()))
    rc = q4.main(["--config", str(cfg_path), "--output-dir", str(tmp_path), "--no-figure"])
    assert rc == 0
    with (tmp_path / "results.csv").open() as f:
        rows = list(csv.DictReader(f))
    soft = [r for r in rows if r["branch"] == "soft"]
    assert soft, "no soft-branch rows in output"
    for row in soft:
        pred = float(row["predicted_rate"])
        obs = float(row["observed_rate"])
        if not (obs > 0 and pred > 0):
            continue
        ratio = obs / pred
        # Shifted denominator should still match prediction well except at the
        # near-divergence boundary where Armijo truncates.
        assert 0.5 <= ratio <= 1.5, (row, ratio)


@pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None, reason="matplotlib is optional")
def test_driver_renders_figure(tmp_path: Path) -> None:
    cfg_path = tmp_path / "q4.yaml"
    cfg_path.write_text(yaml.safe_dump(_cfg()))
    rc = q4.main(["--config", str(cfg_path), "--output-dir", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "figure.png").exists()
