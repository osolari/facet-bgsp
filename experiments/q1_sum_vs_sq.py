"""Q1, do A-SUM and A-SQ actually differ under multi-component aggregations?

Evaluates ``F_sum(X(θ))`` and ``F_sq(X(θ))`` on the deterministic 2x7 A-matrix
counterexample from the BGSP paper's A3 proposition, as θ sweeps through [0, π).
Produces a CSV of the two curves and a two-panel PNG:

- panel 1: the two objective values vs rotation angle, with argmax markers;
- panel 2: the implied ordering (sign of F_sum(θ_sum) − F_sum(θ_sq) etc.),
  demonstrating that each objective's arg-max dominates its own measure yet
  loses on the other, the A3 reversal.

The 2x7 instance is embedded in the YAML config verbatim so the experiment is
self-contained and matches the fixture used by
``tests/test_paper_a_identities.py::test_a3_sum_vs_squared_aggregation_rank_two_candidates_differently``.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses as dc
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from facet.supports import soft_threshold

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


@dc.dataclass(frozen=True)
class SweepRow:
    angle_rad: float
    f_sum: float
    f_sq: float


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _f_sum(A: np.ndarray, X: np.ndarray, gammas: tuple[float, ...]) -> float:
    total = 0.0
    for j, gamma in enumerate(gammas):
        total += float(np.linalg.norm(soft_threshold(A.T @ X[:, j], gamma)))
    return total


def _f_sq(A: np.ndarray, X: np.ndarray, gammas: tuple[float, ...]) -> float:
    total = 0.0
    for j, gamma in enumerate(gammas):
        s = soft_threshold(A.T @ X[:, j], gamma)
        total += float(s @ s)
    return total


def sweep(config: dict[str, Any]) -> tuple[list[SweepRow], np.ndarray, tuple[float, ...]]:
    inst = config["instance"]
    A = np.array(inst["A"], dtype=float)
    gammas = tuple(float(g) for g in inst["gammas"])
    s = config["sweep"]
    angles = np.linspace(
        float(s["angle_min_rad"]), float(s["angle_max_rad"]), int(s["angle_samples"])
    )
    rows: list[SweepRow] = []
    for theta in angles:
        X = _rot(float(theta))
        rows.append(
            SweepRow(
                angle_rad=float(theta),
                f_sum=_f_sum(A, X, gammas),
                f_sq=_f_sq(A, X, gammas),
            )
        )
    return rows, angles, gammas


def write_csv(rows: list[SweepRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dc.fields(SweepRow)])
        writer.writeheader()
        for r in rows:
            writer.writerow(dc.asdict(r))


def write_figure(rows: list[SweepRow], angles: np.ndarray, path: Path) -> None:
    from _style import METHOD_COLORS, METHOD_LABELS, apply_style, full_spines

    apply_style()
    import matplotlib.pyplot as plt

    f_sum = np.array([r.f_sum for r in rows])
    f_sq = np.array([r.f_sq for r in rows])
    i_sum = int(np.argmax(f_sum))
    i_sq = int(np.argmax(f_sq))
    angles_deg = angles * 180.0 / np.pi

    fig, (ax, ax_bars) = plt.subplots(1, 2, figsize=(12.0, 4.4), constrained_layout=True)

    ax.plot(angles_deg, f_sum, label=METHOD_LABELS["sum"], color=METHOD_COLORS["sum"], lw=2.0)
    ax.plot(
        angles_deg,
        f_sq / f_sq.max() * f_sum.max(),
        label=METHOD_LABELS["sq"] + " (rescaled)",
        color=METHOD_COLORS["sq"],
        lw=2.0,
        linestyle="--",
    )
    ax.axvline(angles_deg[i_sum], color=METHOD_COLORS["sum"], lw=1.0, ls=":")
    ax.axvline(angles_deg[i_sq], color=METHOD_COLORS["sq"], lw=1.0, ls=":")
    ax.set_xlabel(r"rotation angle $\theta$ [deg]")
    ax.set_ylabel("objective value")
    ax.set_title(
        rf"argmax $F_{{\mathrm{{sum}}}}$ at ${angles_deg[i_sum]:.1f}^\circ$, "
        rf"argmax $F_{{\mathrm{{sq}}}}$ at ${angles_deg[i_sq]:.1f}^\circ$"
    )
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.4)
    full_spines(ax)

    tick_labels = [
        r"$F_{\mathrm{sum}}(\theta_{\mathrm{sum}})$",
        r"$F_{\mathrm{sum}}(\theta_{\mathrm{sq}})$",
        r"$F_{\mathrm{sq}}(\theta_{\mathrm{sum}})$",
        r"$F_{\mathrm{sq}}(\theta_{\mathrm{sq}})$",
    ]
    heights = [f_sum[i_sum], f_sum[i_sq], f_sq[i_sum], f_sq[i_sq]]
    colors = [METHOD_COLORS["sum"], METHOD_COLORS["sum"], METHOD_COLORS["sq"], METHOD_COLORS["sq"]]
    xs = range(len(tick_labels))
    bars = ax_bars.bar(xs, heights, color=colors, edgecolor="black")
    for b in (bars[1], bars[2]):
        b.set_hatch("//")
        b.set_alpha(0.6)
    ax_bars.set_xticks(list(xs))
    ax_bars.set_xticklabels(tick_labels)
    ax_bars.set_ylabel("objective value")
    ax_bars.set_title(
        "A3 ordering reversal\n(solid = argmax, hatched = evaluated at the OTHER argmax)"
    )
    ax_bars.tick_params(axis="x", rotation=15)
    ax_bars.grid(axis="y", linestyle=":", alpha=0.4)
    full_spines(ax_bars)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    rows, angles, _ = sweep(config)
    out = Path(args.output_dir)
    write_csv(rows, out / config["output"]["csv"])
    if not args.no_figure:
        write_figure(rows, angles, out / config["output"]["figure"])

    f_sum = np.array([r.f_sum for r in rows])
    f_sq = np.array([r.f_sq for r in rows])
    i_sum, i_sq = int(np.argmax(f_sum)), int(np.argmax(f_sq))
    reversal_rad = abs(angles[i_sum] - angles[i_sq])
    print(
        f"wrote {len(rows)} rows; argmax(F_sum)={angles[i_sum]:.4f} rad, "
        f"argmax(F_sq)={angles[i_sq]:.4f} rad, Δ={reversal_rad:.4f} rad"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
