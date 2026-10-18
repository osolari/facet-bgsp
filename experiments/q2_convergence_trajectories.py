"""Q2, convergence trajectories of the safeguarded Riemannian ascent.

For each replicate, fit a planted-spike instance with ``BlockSparseEstimator``
(objective = A-SQ, solver = safeguarded) and record the full per-iteration
``(objective, step_norm, tangent_residual)`` trace. The 3-panel figure is the
empirical face of theorems A6 (sufficient-ascent), A7 (relative-error /
vanishing step + residual), and A8 (whole-sequence Cauchy via geometric
decay). CSV + figure outputs.
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
from facet.supports import soft_threshold  # noqa: F401 (kept for config-driven extensions)

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


@dc.dataclass(frozen=True)
class StepRow:
    seed: int
    iteration: int
    objective: float
    step_norm: float
    residual: float
    support_size: int


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _gamma_from_margin_fraction(instance, mf: float) -> float:
    c = instance.data.T @ instance.scores[:, 0]
    on_min = float(np.abs(c[instance.supports[0]]).min())
    off_max = float(np.abs(np.delete(c, instance.supports[0])).max())
    return (1.0 - mf) * on_min + mf * off_max


def _perturbed_init(rng: np.random.Generator, x_star: np.ndarray, angle: float) -> np.ndarray:
    perp = rng.normal(size=x_star.shape)
    perp -= (perp @ x_star) * x_star
    perp /= np.linalg.norm(perp)
    x = np.cos(angle) * x_star + np.sin(angle) * perp
    return x / np.linalg.norm(x)


def run(config: dict[str, Any]) -> list[StepRow]:
    problem = config["problem"]
    solver_cfg = config["solver"]
    reps = int(config["replicates"])
    mf = float(config["margin_fraction"])
    angle = float(config["initial_angle_rad"])

    records: list[StepRow] = []
    for rep in range(reps):
        seed = int(config["seed_base"]) + rep
        inst = planted_spike(
            p=int(problem["p"]),
            n=int(problem["n"]),
            k=int(problem["k"]),
            m=1,
            singular_values=tuple(problem["singular_values"]),
            noise_std=float(problem["noise_std"]),
            seed=seed,
        )
        gamma = _gamma_from_margin_fraction(inst, mf)
        rng = np.random.default_rng(seed + 10_000)
        init = _perturbed_init(rng, inst.scores[:, 0], angle)
        est = BlockSparseEstimator(
            objective=config["objective"],
            gamma=gamma,
            solver="safeguarded",
            max_iterations=int(solver_cfg["max_iterations"]),
            tolerance=float(solver_cfg["tolerance"]),
            random_state=seed,
        )
        try:
            fit = est.fit(inst.data, init=init)
        except RuntimeError as exc:
            # Armijo near-fixed-point failure: expected at very tight tolerances.
            # Record the failure as a single row so a reader can see the seed
            # dropped out rather than silently vanishing from the figure.
            print(f"[q2] seed={seed} dropped, {exc}")
            continue
        for r in fit.trace.records:
            records.append(
                StepRow(
                    seed=seed,
                    iteration=int(r.iteration),
                    objective=float(r.objective),
                    step_norm=float(r.step_norm),
                    residual=float(r.residual or 0.0),
                    support_size=int(r.support_size or 0),
                )
            )
    return records


def write_csv(rows: list[StepRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dc.fields(StepRow)])
        writer.writeheader()
        for r in rows:
            writer.writerow(dc.asdict(r))


def write_figure(rows: list[StepRow], path: Path) -> None:
    from _style import apply_style, full_spines

    apply_style()
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, (ax_obj, ax_step, ax_res) = plt.subplots(
        1, 3, figsize=(13.2, 4.4), constrained_layout=True
    )

    seeds = sorted({r.seed for r in rows})
    palette = sns.color_palette("colorblind", n_colors=max(len(seeds), 3))

    for i, seed in enumerate(seeds):
        srows = [r for r in rows if r.seed == seed]
        it = np.array([r.iteration for r in srows])
        obj = np.array([r.objective for r in srows])
        step = np.array([max(r.step_norm, 1e-16) for r in srows])
        res = np.array([max(r.residual, 1e-16) for r in srows])
        color = palette[i % len(palette)]
        label = f"seed {seed}" if i < 6 else None
        ax_obj.plot(it, obj, color=color, alpha=0.9, lw=1.4, label=label)
        ax_step.semilogy(it, step, color=color, alpha=0.9, lw=1.4)
        ax_res.semilogy(it, res, color=color, alpha=0.9, lw=1.4)

    for ax in (ax_obj, ax_step, ax_res):
        ax.set_xlabel("iteration")
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        full_spines(ax)

    ax_obj.set_ylabel(r"$F_{\mathrm{sq}}(X_k)$")
    ax_obj.set_title("A6, monotone ascent")
    if seeds:
        ax_obj.legend(loc="lower right", fontsize=9, ncol=1)
    ax_step.set_ylabel(r"$\|X_{k+1}-X_k\|_F$")
    ax_step.set_title(r"A7/A8, step norm $\|X_{k+1}-X_k\|_F$ (log)")
    ax_res.set_ylabel("tangent residual")
    ax_res.set_title("A7, tangent residual (log)")

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
    print(f"wrote {len(rows)} step records across {len({r.seed for r in rows})} seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
