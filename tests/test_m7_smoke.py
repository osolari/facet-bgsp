"""Smoke tests for the M7 safe-screening payoff driver."""

from __future__ import annotations

import csv
from pathlib import Path

import yaml

from experiments import m7_safe_screening as m7


def test_run_one_produces_monotone_screening_approaching_ceiling() -> None:
    rows = m7._run_one(
        p=10,
        n=40,
        k=4,
        seed=123,
        gamma_mult=0.4,
        solver_cfg={"max_iterations": 100, "tolerance": 1.0e-9},
    )
    assert rows, "no rows produced for a valid instance"
    # Final fraction screened should hit the ceiling (n-k)/n exactly.
    final = rows[-1]
    ceiling = (final.n - final.k) / final.n
    assert abs(final.frac_screened - ceiling) < 1e-9, (final.frac_screened, ceiling)
    # Final radius should be smaller than the initial.
    assert rows[-1].radius < rows[0].radius + 1e-9


def test_driver_schema_and_sparsity_coverage(tmp_path: Path) -> None:
    cfg = {
        "name": "m7_smoke",
        "seed_base": 7,
        "replicates": 2,
        "problem": {"p": 8, "n": 30, "k_values": [3, 10], "gamma_margin_fraction": 0.4},
        "solver": {"max_iterations": 80, "tolerance": 1.0e-8},
        "output": {"csv": "m7.csv", "figure": "m7.png"},
    }
    cfg_path = tmp_path / "m7.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    rc = m7.main(["--config", str(cfg_path), "--output-dir", str(tmp_path), "--no-figure"])
    assert rc == 0
    rows = list(csv.DictReader((tmp_path / "m7.csv").open()))
    assert rows, "driver produced no rows"
    assert {int(r["k"]) for r in rows} == {3, 10}
    # Every row has a sensible active + screened = n.
    for r in rows:
        total = int(r["n_active"]) + int(r["n_screened"])
        assert total == int(r["n"]), r


def test_payoff_csv_has_zero_objective_gap_and_reaches_ceiling(tmp_path: Path) -> None:
    cfg = {
        "name": "m7_payoff_smoke",
        "seed_base": 7,
        "replicates": 2,
        "problem": {"p": 10, "n": 40, "k_values": [4, 16], "gamma_margin_fraction": 0.4},
        "solver": {"max_iterations": 80, "tolerance": 1.0e-8},
        "output": {"csv": "m7.csv", "figure": "m7.png"},
    }
    cfg_path = tmp_path / "m7.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    rc = m7.main(["--config", str(cfg_path), "--output-dir", str(tmp_path), "--no-figure"])
    assert rc == 0
    rows = list(csv.DictReader((tmp_path / "m7_payoff.csv").open()))
    assert rows, "payoff csv was empty"
    for r in rows:
        full = float(r["final_obj_full"])
        screened = float(r["final_obj_screened"])
        # ScreenedDenseOperator must produce objective values matching the full
        # solve to machine precision (A10 safety invariant at the solver level).
        assert abs(full - screened) < 1e-10, (r, full, screened)
        # Work reduction should be at least (n - k) / n * 0.9 (10% slack for
        # the warm-start iterations in the full solve).
        kn = int(r["k"]) / int(r["n"])
        ceiling = 1.0 - kn
        assert float(r["work_reduction_frac"]) >= 0.9 * ceiling, r
