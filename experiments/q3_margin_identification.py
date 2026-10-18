"""Q3, threshold margin predicts finite support identification.

Reads a YAML config (see ``configs/q3_margin_identification.yaml``),
sweeps a grid of (margin_fraction, initial_angle_rad, replicate) over planted-
spike instances, fits ``BlockSparseEstimator`` with the configured objective
and solver, and records per-run metrics in a CSV. Optionally renders a two-
panel figure summarizing identification rate and iteration count vs margin.

Usage
-----
    python -m papers.a.experiments.q3_margin_identification \\
        --config configs/q3_margin_identification.yaml \\
        --output-dir runs/paper-a/q3

Add ``--no-figure`` to skip the matplotlib step (useful in headless CI).
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

from facet.generators import planted_spike
from facet.papers.block_sparse import BlockSparseEstimator

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


@dc.dataclass(frozen=True)
class RunRecord:
    seed: int
    noise_std: float
    margin_fraction: float
    initial_angle_rad: float
    gamma: float
    achieved_margin: float
    iterations: int
    support_matches: bool
    final_objective: float
    refit_objective: float


# --------------------------------------------------------------------------- #
# Configuration loading


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# Grid driver


def _gamma_from_margin_fraction(instance, margin_fraction: float) -> tuple[float, float]:
    """Pick gamma on the segment [off_max, on_min] parameterized by margin_fraction.

    ``margin_fraction=0`` sits at ``off_max`` (zero margin on the support);
    ``margin_fraction=1`` sits at ``on_min`` (zero slack off the support).
    """
    c_star = instance.data.T @ instance.scores[:, 0]
    on_min = float(np.abs(c_star[instance.supports[0]]).min())
    off_max = float(np.abs(np.delete(c_star, instance.supports[0])).max())
    gamma = (1.0 - margin_fraction) * on_min + margin_fraction * off_max
    achieved = on_min - gamma
    return gamma, achieved


def _perturbed_init(rng: np.random.Generator, x_star: np.ndarray, angle: float) -> np.ndarray:
    perp = rng.normal(size=x_star.shape)
    perp -= (perp @ x_star) * x_star
    perp /= np.linalg.norm(perp)
    x = np.cos(angle) * x_star + np.sin(angle) * perp
    return x / np.linalg.norm(x)


def _stabilization_iteration(trace) -> int:
    sizes = [r.support_size for r in trace.records if r.support_size is not None]
    if not sizes:
        return len(trace.records)
    final = sizes[-1]
    stabilized_from = next(
        (i for i in range(len(sizes)) if all(s == final for s in sizes[i:])),
        len(sizes),
    )
    return stabilized_from


def run_grid(config: dict[str, Any]) -> list[RunRecord]:
    problem = config["problem"]
    grid = config["grid"]
    solver_cfg = config["solver"]

    noise_grid = [float(x) for x in grid.get("noise_std", [float(problem.get("noise_std", 0.0))])]

    records: list[RunRecord] = []
    for rep in range(int(grid["replicates"])):
        seed = int(config["seed_base"]) + rep
        for noise_std in noise_grid:
            instance = planted_spike(
                p=int(problem["p"]),
                n=int(problem["n"]),
                k=int(problem["k"]),
                m=1,
                singular_values=tuple(problem["singular_values"]),
                noise_std=float(noise_std),
                seed=seed,
            )
            planted_support = set(int(i) for i in instance.supports[0])

            for mf in grid["margin_fraction"]:
                gamma, achieved = _gamma_from_margin_fraction(instance, float(mf))
                est = BlockSparseEstimator(
                    objective=config["objective"],
                    gamma=gamma,
                    solver=solver_cfg["name"],
                    max_iterations=int(solver_cfg["max_iterations"]),
                    tolerance=float(solver_cfg["tolerance"]),
                    random_state=seed,
                )

                for angle in grid["initial_angle_rad"]:
                    rng = np.random.default_rng(
                        seed * 1000 + round(float(angle) * 1e4) + int(noise_std * 1e3)
                    )
                    init = _perturbed_init(rng, instance.scores[:, 0], float(angle))
                    try:
                        fit = est.fit(instance.data, init=init)
                    except RuntimeError:
                        # Record a failed run as a non-match with NaN objectives
                        records.append(
                            RunRecord(
                                seed=seed,
                                noise_std=float(noise_std),
                                margin_fraction=float(mf),
                                initial_angle_rad=float(angle),
                                gamma=gamma,
                                achieved_margin=achieved,
                                iterations=int(solver_cfg["max_iterations"]),
                                support_matches=False,
                                final_objective=float("nan"),
                                refit_objective=float("nan"),
                            )
                        )
                        continue
                    iters = _stabilization_iteration(fit.trace)
                    support_matches = set(fit.support.tolist()) == planted_support
                    final_obj = fit.raw.objective if fit.raw.objective is not None else float("nan")
                    refit_obj = (
                        fit.refit.objective
                        if fit.refit is not None and fit.refit.objective is not None
                        else float("nan")
                    )
                    records.append(
                        RunRecord(
                            seed=seed,
                            noise_std=float(noise_std),
                            margin_fraction=float(mf),
                            initial_angle_rad=float(angle),
                            gamma=gamma,
                            achieved_margin=achieved,
                            iterations=iters,
                            support_matches=bool(support_matches),
                            final_objective=float(final_obj),
                            refit_objective=float(refit_obj),
                        )
                    )
    return records


# --------------------------------------------------------------------------- #
# Output


def write_csv(records: list[RunRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dc.fields(RunRecord)])
        writer.writeheader()
        for r in records:
            writer.writerow(dc.asdict(r))


def write_figure(records: list[RunRecord], path: Path) -> None:
    from _style import apply_style, full_spines

    apply_style()
    import matplotlib.pyplot as plt
    import seaborn as sns

    mfs = sorted({r.margin_fraction for r in records})
    noises = sorted({r.noise_std for r in records})

    rate_grid = np.full((len(noises), len(mfs)), np.nan)
    iter_grid = np.full((len(noises), len(mfs)), np.nan)
    for i, ns in enumerate(noises):
        for j, mf in enumerate(mfs):
            subset = [r for r in records if r.noise_std == ns and r.margin_fraction == mf]
            if not subset:
                continue
            rate_grid[i, j] = float(np.mean([r.support_matches for r in subset]))
            matched = [r.iterations for r in subset if r.support_matches]
            iter_grid[i, j] = float(np.median(matched)) if matched else float("nan")

    fig, (ax_rate, ax_iter) = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)

    sns.heatmap(
        rate_grid,
        ax=ax_rate,
        annot=True,
        fmt=".2f",
        vmin=0.0,
        vmax=1.0,
        cmap="RdYlGn",
        xticklabels=[f"{m:.2f}" for m in mfs],
        yticklabels=[f"{n:.1f}" for n in noises],
        cbar_kws={"label": "support recovery rate"},
        linewidths=0.5,
        linecolor="w",
    )
    ax_rate.set_title("A9: exact support recovery rate")
    ax_rate.set_xlabel("margin fraction")
    ax_rate.set_ylabel("noise std")
    ax_rate.invert_yaxis()
    full_spines(ax_rate)

    with np.errstate(invalid="ignore"):
        annot = np.where(np.isfinite(iter_grid), iter_grid, 0)
    sns.heatmap(
        iter_grid,
        ax=ax_iter,
        annot=annot.astype(int),
        fmt="d",
        cmap="viridis_r",
        xticklabels=[f"{m:.2f}" for m in mfs],
        yticklabels=[f"{n:.1f}" for n in noises],
        cbar_kws={"label": "iters to support stabilization (median)"},
        linewidths=0.5,
        linecolor="w",
    )
    ax_iter.set_title("safeguarded solver cost (recovered runs only)")
    ax_iter.set_xlabel("margin fraction")
    ax_iter.set_ylabel("noise std")
    ax_iter.invert_yaxis()
    full_spines(ax_iter)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Entrypoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    records = run_grid(config)
    out = Path(args.output_dir)
    csv_path = out / config["output"]["csv"]
    write_csv(records, csv_path)

    if not args.no_figure:
        fig_path = out / config["output"]["figure"]
        write_figure(records, fig_path)

    print(f"wrote {len(records)} records to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
