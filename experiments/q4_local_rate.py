"""Q4, local-rate prediction versus observation (§14 required figure #6).

Hand-constructed rank-2 operator with KNOWN top-two restricted eigenvalues
(``lambda_1``, ``lambda_2``). Two branches:

  (a) Pure power iteration (``gamma = 0``, hard support): classical sphere
      power iteration on ``A_S A_S^T``; the observed geometric decay rate of
      ``sin(angle_t)`` matches ``lambda_2 / lambda_1`` exactly.
  (b) A-SQ with the standard soft-threshold shrinkage: the gradient
      becomes a *shifted* power iteration, ``x_{k+1} ∝ A_S (A_S^T x -
      gamma * sign(c))``. In the invariant 2D subspace this makes the
      effective denominator ``lambda_1 - gamma * sigma_1 * sqrt(k)``, so
      the predicted rate is ``lambda_2 / (lambda_1 - gamma * sigma_1 *
      sqrt(k))``.

The driver sweeps ``lambda_2`` with ``lambda_1`` fixed and records
(predicted_hard, predicted_soft, observed_hard, observed_soft) for each
cell. See
``.claude/skills/facet-paper-a/plan-2026-10-13T1908Z_...md`` §Q4 for the
derivation of the shifted-denominator formula.
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

from facet.geometry import SphereGeometry
from facet.objectives import SquaredThresholdObjective
from facet.operators import DenseOperator

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


@dc.dataclass(frozen=True)
class RateRow:
    seed: int
    lam1: float
    lam2: float
    gamma: float
    predicted_rate: float
    observed_rate: float
    branch: str  # "hard" (gamma=0) or "soft" (A-SQ with gamma>0)
    iterations: int


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _controlled_instance(
    *, p: int, k: int, lam1: float, lam2: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank-2 operator with known top-two restricted eigenvalues on an identified support.

    Matches the construction in ``tests/test_paper_a_local_rate.py`` so the
    manuscript figure and the pinned test share a single fixture.
    """
    rng = np.random.default_rng(seed)
    sigma_1 = float(np.sqrt(lam1))
    sigma_2 = float(np.sqrt(lam2))

    base = rng.normal(size=p)
    x_star = base / np.linalg.norm(base)
    perp = rng.normal(size=p)
    perp -= (perp @ x_star) * x_star
    w = perp / np.linalg.norm(perp)

    r_raw = rng.normal(size=k)
    r_raw -= r_raw.mean()
    r = r_raw / np.linalg.norm(r_raw)

    ones = np.ones(k)
    A_S = sigma_1 * np.outer(x_star, ones) / np.sqrt(k) + sigma_2 * np.outer(w, r)
    n = k + 15
    A = np.zeros((p, n))
    A[:, :k] = A_S
    # Keep the off-support block tiny (1e-6) so the fixed-point floor doesn't
    # cut the geometric decay off after 1-2 iterations. Larger off-support noise
    # perturbs the top eigenvector by O(noise/eigengap) and masks the linear
    # rate on the plot, see plan-2026-10-13T1908Z §Q4.
    A[:, k:] = 1e-6 * rng.normal(size=(p, n - k))
    support = np.arange(k)
    return A, x_star, support


def _perturbed_init(rng: np.random.Generator, x_star: np.ndarray, angle: float) -> np.ndarray:
    perp = rng.normal(size=x_star.shape)
    perp -= (perp @ x_star) * x_star
    perp /= np.linalg.norm(perp)
    x = np.cos(angle) * x_star + np.sin(angle) * perp
    return x / np.linalg.norm(x)


def _observed_rate(obj, geom, init: np.ndarray, x_star: np.ndarray, cfg: dict) -> tuple[float, int]:
    """Fit the geometric decay rate of sin(angle_t) over the strictly-decaying
    prefix of the trajectory.

    The iteration plateaus at a non-zero floor (near x_star, slightly
    perturbed by off-support operator entries). Fitting across the plateau
    collapses the slope and inflates the estimated rate, see
    plan-2026-10-13T1908Z §Q4 for the diagnostic. We cut the fit as soon as
    successive ratios exceed 0.99 (plateau onset) or deviation falls below
    1e-9.
    """
    max_iters = int(cfg["max_iterations"])
    x = np.asarray(init, dtype=float).copy()
    devs = []
    for _ in range(max_iters):
        g = obj.gradient(x)
        x = geom.linear_oracle(g)
        cos = abs(float(x @ x_star))
        dev = float(np.sqrt(max(0.0, 1.0 - cos**2)))
        devs.append(dev)
        if dev < 1e-10:
            break

    devs = np.array(devs)
    if devs.size < 2:
        return float("nan"), int(devs.size)

    # Find the strictly-decaying prefix: stop at the first index where either
    # (a) devs[i] >= 0.99 * devs[i-1] (plateau onset), or
    # (b) devs[i] < 1e-9 (below numerical floor).
    cutoff = devs.size
    for i in range(1, devs.size):
        if devs[i] < 1e-9:
            cutoff = i
            break
        if devs[i] >= 0.99 * devs[i - 1]:
            cutoff = i
            break
    decaying = devs[:cutoff]
    if decaying.size < 3:
        # Not enough points for a stable slope; use the single-step ratio
        if devs.size >= 2 and devs[0] > 0:
            return float(devs[1] / devs[0]), int(devs.size)
        return float("nan"), int(devs.size)

    log_dev = np.log(decaying)
    slope = float(np.polyfit(np.arange(len(log_dev)), log_dev, deg=1)[0])
    return float(np.exp(slope)), int(devs.size)


def run(config: dict[str, Any]) -> list[RateRow]:
    problem = config["problem"]
    grid = config["grid"]
    solver_cfg = config["solver"]
    lam1 = float(problem["lam1"])
    sigma_1 = float(np.sqrt(lam1))
    p = int(problem["p"])
    k = int(problem["k"])

    records: list[RateRow] = []
    for lam2 in grid["lam2"]:
        for rep in range(int(grid["replicates"])):
            seed = int(config["seed_base"]) + rep * 100 + round(float(lam2) * 10)
            A, x_star, support = _controlled_instance(
                p=p, k=k, lam1=lam1, lam2=float(lam2), seed=seed
            )
            op = DenseOperator(A)
            c_star = A.T @ x_star
            on_min = float(np.abs(c_star[support]).min())
            off_max = float(np.abs(np.delete(c_star, support)).max())
            if on_min <= off_max:
                continue
            rng = np.random.default_rng(seed + 7)
            init = _perturbed_init(rng, x_star, float(grid["initial_angle_rad"]))
            geom = SphereGeometry(radius=1.0)

            # (a) Pure power iteration, gamma=0 (no shrinkage shift).
            obj_hard = SquaredThresholdObjective(op, gamma=0.0)
            obs_hard, iters_hard = _observed_rate(obj_hard, geom, init, x_star, solver_cfg)
            records.append(
                RateRow(
                    seed=seed,
                    lam1=lam1,
                    lam2=float(lam2),
                    gamma=0.0,
                    predicted_rate=float(lam2) / lam1,
                    observed_rate=obs_hard,
                    branch="hard",
                    iterations=iters_hard,
                )
            )

            # (b) A-SQ with the standard shrinkage gamma, shifted power
            # iteration; the effective denominator is lam1 - gamma*sigma_1*sqrt(k).
            gamma_soft = 0.5 * (on_min + off_max)
            denom = lam1 - gamma_soft * sigma_1 * float(np.sqrt(k))
            predicted_soft = float(lam2) / denom if denom > 0.0 else float("nan")
            obj_soft = SquaredThresholdObjective(op, gamma=gamma_soft)
            obs_soft, iters_soft = _observed_rate(obj_soft, geom, init, x_star, solver_cfg)
            records.append(
                RateRow(
                    seed=seed,
                    lam1=lam1,
                    lam2=float(lam2),
                    gamma=gamma_soft,
                    predicted_rate=predicted_soft,
                    observed_rate=obs_soft,
                    branch="soft",
                    iterations=iters_soft,
                )
            )
    return records


def write_csv(rows: list[RateRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dc.fields(RateRow)])
        writer.writeheader()
        for r in rows:
            writer.writerow(dc.asdict(r))


def write_figure(rows: list[RateRow], path: Path) -> None:
    from _style import METHOD_COLORS, apply_style, full_spines

    apply_style()
    import matplotlib.pyplot as plt

    def _branch(name: str) -> tuple[np.ndarray, np.ndarray]:
        pred = np.array([r.predicted_rate for r in rows if r.branch == name])
        obs = np.array([r.observed_rate for r in rows if r.branch == name])
        valid = np.isfinite(pred) & np.isfinite(obs) & (obs > 0) & (pred > 0)
        return pred[valid], obs[valid]

    hard_pred, hard_obs = _branch("hard")
    soft_pred, soft_obs = _branch("soft")

    all_pts = np.concatenate([hard_pred, hard_obs, soft_pred, soft_obs])
    lo = max(1e-3, float(all_pts.min()) * 0.7) if all_pts.size else 1e-3
    hi = max(1.5, float(all_pts.max()) * 1.4) if all_pts.size else 1.5
    ref = np.array([lo, hi])

    fig, (ax_hard, ax_soft) = plt.subplots(1, 2, figsize=(11.6, 5.0), constrained_layout=True)

    for ax, pred, obs, title, color in (
        (
            ax_hard,
            hard_pred,
            hard_obs,
            r"(a) Pure power iteration ($\gamma = 0$)",
            METHOD_COLORS["observed"],
        ),
        (
            ax_soft,
            soft_pred,
            soft_obs,
            r"(b) A-SQ with shrinkage shift ($\gamma = \gamma^\star$)",
            METHOD_COLORS["facet_asq"],
        ),
    ):
        ax.loglog(ref, ref, "--", color=METHOD_COLORS["theory"], lw=1.0, label=r"$y = x$")
        if pred.size:
            ax.loglog(
                pred,
                obs,
                "o",
                color=color,
                markeredgecolor="k",
                markeredgewidth=0.5,
                label="observed",
            )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("predicted rate")
        ax.set_ylabel("observed geometric rate")
        ax.set_title(title, fontsize=12)
        ax.legend(loc="lower right")
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        full_spines(ax)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    rows = run(cfg)
    out = Path(args.output_dir)
    write_csv(rows, out / cfg["output"]["csv"])
    if not args.no_figure:
        write_figure(rows, out / cfg["output"]["figure"])
    valid = [r for r in rows if np.isfinite(r.observed_rate) and r.observed_rate > 0]
    print(f"wrote {len(rows)} rows ({len(valid)} valid observed-rates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
