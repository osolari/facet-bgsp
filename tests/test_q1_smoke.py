"""Smoke test for the Q1 A-SUM-vs-A-SQ ordering-reversal experiment driver.

Verifies the driver loads the deterministic A3 fixture from YAML, produces a
CSV with the expected columns, renders the figure, and that the argmax
reversal (A3 proposition) is reproduced on the canonical instance.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

from experiments import q1_sum_vs_sq as q1

EXPECTED_COLUMNS = {"angle_rad", "f_sum", "f_sq"}


def _config() -> dict:
    return {
        "name": "q1_smoke",
        "instance": {
            "A": [
                [
                    1.42993853,
                    -0.15647532,
                    -0.67375915,
                    -0.6390601,
                    -0.06136133,
                    -0.39278492,
                    2.28990995,
                ],
                [
                    -0.71818115,
                    0.03260774,
                    0.0280499,
                    0.02827212,
                    0.05534586,
                    -0.48156286,
                    -0.5834075,
                ],
            ],
            "gammas": [0.6845716431159987, 0.5315438759871329],
        },
        "sweep": {"angle_min_rad": 0.0, "angle_max_rad": 3.141592653589793, "angle_samples": 181},
        "output": {"csv": "results.csv", "figure": "figure.png"},
    }


def test_driver_writes_csv_with_expected_schema(tmp_path: Path) -> None:
    cfg_path = tmp_path / "q1.yaml"
    cfg_path.write_text(yaml.safe_dump(_config()))
    rc = q1.main(["--config", str(cfg_path), "--output-dir", str(tmp_path), "--no-figure"])
    assert rc == 0
    csv_path = tmp_path / "results.csv"
    assert csv_path.exists()
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert set(rows[0].keys()) == EXPECTED_COLUMNS
    assert len(rows) == 181


def test_canonical_a3_reversal_is_reproduced(tmp_path: Path) -> None:
    """The known reversal: argmax(F_sum) ≈ 30°, argmax(F_sq) ≈ 72.5°, tested
    against the same fixture used by test_a3 so the two assertions stay
    aligned."""
    cfg_path = tmp_path / "q1.yaml"
    cfg_path.write_text(yaml.safe_dump(_config()))
    rc = q1.main(["--config", str(cfg_path), "--output-dir", str(tmp_path), "--no-figure"])
    assert rc == 0
    with (tmp_path / "results.csv").open() as f:
        rows = list(csv.DictReader(f))
    angles = np.array([float(r["angle_rad"]) for r in rows])
    f_sum = np.array([float(r["f_sum"]) for r in rows])
    f_sq = np.array([float(r["f_sq"]) for r in rows])
    i_sum = int(np.argmax(f_sum))
    i_sq = int(np.argmax(f_sq))
    assert abs(angles[i_sum] - angles[i_sq]) > 0.5  # at least ~30° apart
    # The ordering reverses: each objective's argmax wins under its own measure.
    assert f_sum[i_sum] - f_sum[i_sq] > 0.05
    assert f_sq[i_sq] - f_sq[i_sum] > 0.5


@pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None, reason="matplotlib is optional")
def test_driver_renders_figure(tmp_path: Path) -> None:
    cfg_path = tmp_path / "q1.yaml"
    cfg_path.write_text(yaml.safe_dump(_config()))
    rc = q1.main(["--config", str(cfg_path), "--output-dir", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "figure.png").exists()
    assert (tmp_path / "figure.png").stat().st_size > 1024
