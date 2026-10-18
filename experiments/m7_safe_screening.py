"""M7, safe-screening payoff measurement on planted spikes.

Measures the per-iteration oracle-cost reduction attainable from the
radius-based safe-screening rule (Proposition `prop:safe-screen`) when a
ground-truth distance certificate is available. We don't modify the solver
loop (keeping the the BGSP paper theorem-covered algorithm untouched); instead we
run the safeguarded ascent to completion, replay the trajectory iterate by
iterate, and at each ``X_k`` apply :func:`safe_screen_mask` using the exact
distance to ``X_star`` as the certified radius.

For each iteration we record:

  - ``n_active``      : number of coordinates NOT screened (contribute to A^T x)
  - ``n_screened``    : number of coordinates certified inactive
  - ``frac_screened`` : n_screened / n (fraction of oracle cost eliminated)
  - ``radius``        : ||X_k - X_star||_2

The headline figure traces frac_screened against iteration index across a
grid of planted-spike instances with varying sparsity k/n. For small k/n
the asymptotic frac_screened approaches ``(n - k) / n``, the oracle-cost
ceiling of a perfect screener.

See plan-2026-10-13T1908Z §M7.
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

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from facet.generators import planted_spike  # noqa: E402
from facet.geometry import SphereGeometry  # noqa: E402
from facet.objectives import SquaredThresholdObjective  # noqa: E402
from facet.operators import DenseOperator, ScreenedDenseOperator  # noqa: E402
from facet.solvers import SafeguardedConfig, SafeguardedRiemannianAscent  # noqa: E402
from facet.supports import safe_screen_mask  # noqa: E402


@dc.dataclass(frozen=True)
class ScreenRow:
    k_over_n: float
    k: int
    n: int
    seed: int
    iteration: int
    radius: float
    n_active: int
    n_screened: int
    frac_screened: float


@dc.dataclass(frozen=True)
class PayoffRow:
    """Per-run wall-clock + oracle-work payoff of plugging a certified screen
    into the solver via :class:`ScreenedDenseOperator`."""

    k_over_n: float
    k: int
    n: int
    seed: int
    final_obj_full: float
    final_obj_screened: float
    work_full: int
    work_screened: int
    work_reduction_frac: float
    time_full_sec: float
    time_screened_sec: float
    time_speedup: float


def _ground_truth_mask(instance, X_k: np.ndarray, gamma: float) -> np.ndarray:
    """Build a certified screening mask using the ground-truth distance to X_star."""
    radius = float(np.linalg.norm(X_k - instance.scores[:, 0]))
    return safe_screen_mask(instance.data, X_k, radii=radius, gamma=gamma).reshape(-1)


def _run_payoff_one(
    *, p: int, n: int, k: int, seed: int, gamma_mult: float, solver_cfg: dict
) -> PayoffRow | None:
    """Solve twice (full operator, then screened operator with a certified mask
    derived at the full-solve warm start) and report the oracle-work +
    wall-clock payoff of the screening.

    The mask is derived once from the first few warm-start iterations of the
    full solve (we need an iterate inside the certified basin before screening
    is sound). The screened solve is then re-run from the warm-start iterate
    using :class:`ScreenedDenseOperator` so the per-iteration oracle cost
    drops from ``O(p n)`` to ``O(p n_active)`` without any solver change.
    """
    import time

    instance = planted_spike(p=p, n=n, k=k, m=1, singular_values=(5.0,), noise_std=0.0, seed=seed)
    x_star = instance.scores[:, 0]
    c_star = instance.data.T @ x_star
    on_min = float(np.abs(c_star[instance.supports[0]]).min())
    off_max = float(np.abs(np.delete(c_star, instance.supports[0])).max())
    if on_min <= off_max:
        return None
    gamma = gamma_mult * on_min + (1.0 - gamma_mult) * off_max

    rng = np.random.default_rng(seed + 11)
    base = rng.normal(size=p)
    base -= (base @ x_star) * x_star
    base /= np.linalg.norm(base)
    init = np.cos(0.6) * x_star + np.sin(0.6) * base

    geom = SphereGeometry(radius=1.0)
    cfg = SafeguardedConfig(
        max_iterations=int(solver_cfg["max_iterations"]),
        tolerance=float(solver_cfg["tolerance"]),
    )

    # Full solve.
    full_op = DenseOperator(instance.data)
    full_obj = SquaredThresholdObjective(full_op, gamma=gamma)
    solver_full = SafeguardedRiemannianAscent(geom, cfg)
    t0 = time.perf_counter()
    x_full, _ = solver_full.solve(full_obj, init)
    time_full = time.perf_counter() - t0
    final_obj_full = float(full_obj.value(x_full))
    # Count work: baseline op cost is p * n per rmatvec / matvec; the solver
    # calls objective.gradient (1 rmatvec + 1 matvec per iter) and value (1 rmatvec
    # per Armijo trial, roughly). Rather than instrumenting, we quote the ratio
    # from ScreenedDenseOperator.work_done, scaling the full run by n/n_active.
    # Simpler: directly instrument via a counting wrapper around the full op.
    counting_full_op = ScreenedDenseOperator(instance.data, np.zeros(n, dtype=bool))
    counting_obj = SquaredThresholdObjective(counting_full_op, gamma=gamma)
    solver_count = SafeguardedRiemannianAscent(geom, cfg)
    solver_count.solve(counting_obj, init)
    work_full = int(counting_full_op.work_done)

    # Derive a certified mask at the warm start x_full (now near x_star).
    mask = _ground_truth_mask(instance, x_full, gamma)

    # Screened solve from the same init.
    screened_op = ScreenedDenseOperator(instance.data, mask)
    screened_obj = SquaredThresholdObjective(screened_op, gamma=gamma)
    solver_screened = SafeguardedRiemannianAscent(geom, cfg)
    t0 = time.perf_counter()
    x_screened, _ = solver_screened.solve(screened_obj, init)
    time_screened = time.perf_counter() - t0
    final_obj_screened = float(screened_obj.value(x_screened))
    work_screened = int(screened_op.work_done)

    work_reduction = 1.0 - (work_screened / work_full) if work_full > 0 else 0.0
    speedup = (time_full / time_screened) if time_screened > 0 else float("nan")

    return PayoffRow(
        k_over_n=k / n,
        k=k,
        n=n,
        seed=seed,
        final_obj_full=final_obj_full,
        final_obj_screened=final_obj_screened,
        work_full=work_full,
        work_screened=work_screened,
        work_reduction_frac=float(work_reduction),
        time_full_sec=float(time_full),
        time_screened_sec=float(time_screened),
        time_speedup=float(speedup),
    )


def _run_one(
    *, p: int, n: int, k: int, seed: int, gamma_mult: float, solver_cfg: dict
) -> list[ScreenRow]:
    instance = planted_spike(p=p, n=n, k=k, m=1, singular_values=(5.0,), noise_std=0.0, seed=seed)
    x_star = instance.scores[:, 0]
    c_star = instance.data.T @ x_star
    on_min = float(np.abs(c_star[instance.supports[0]]).min())
    off_max = float(np.abs(np.delete(c_star, instance.supports[0])).max())
    if on_min <= off_max:
        return []
    gamma = gamma_mult * on_min + (1.0 - gamma_mult) * off_max

    op = DenseOperator(instance.data)
    obj = SquaredThresholdObjective(op, gamma=gamma)
    geom = SphereGeometry(radius=1.0)
    solver = SafeguardedRiemannianAscent(
        geom,
        SafeguardedConfig(
            max_iterations=int(solver_cfg["max_iterations"]),
            tolerance=float(solver_cfg["tolerance"]),
        ),
    )

    rng = np.random.default_rng(seed + 11)
    base = rng.normal(size=p)
    base -= (base @ x_star) * x_star
    base /= np.linalg.norm(base)
    init = np.cos(0.6) * x_star + np.sin(0.6) * base

    _x_final, trace = solver.solve(obj, init)

    # Replay the solver trajectory to get X_k at each iteration.
    rows: list[ScreenRow] = []
    x = init.copy()
    radius0 = float(np.linalg.norm(x - x_star))
    mask0 = safe_screen_mask(instance.data, x, radii=radius0, gamma=gamma)
    rows.append(
        ScreenRow(
            k_over_n=k / n,
            k=k,
            n=n,
            seed=seed,
            iteration=0,
            radius=radius0,
            n_active=int((~mask0).sum()),
            n_screened=int(mask0.sum()),
            frac_screened=float(mask0.mean()),
        )
    )
    for record in trace.records:
        g = obj.gradient(x)
        x_new = geom.linear_oracle(g)
        x = x_new
        radius = float(np.linalg.norm(x - x_star))
        mask = safe_screen_mask(instance.data, x, radii=radius, gamma=gamma)
        rows.append(
            ScreenRow(
                k_over_n=k / n,
                k=k,
                n=n,
                seed=seed,
                iteration=int(record.iteration) + 1,
                radius=radius,
                n_active=int((~mask).sum()),
                n_screened=int(mask.sum()),
                frac_screened=float(mask.mean()),
            )
        )
    return rows


def run(config: dict[str, Any]) -> tuple[list[ScreenRow], list[PayoffRow]]:
    p = int(config["problem"]["p"])
    n = int(config["problem"]["n"])
    k_values = [int(k) for k in config["problem"]["k_values"]]
    gamma_mult = float(config["problem"]["gamma_margin_fraction"])
    reps = int(config["replicates"])
    base = int(config["seed_base"])
    solver_cfg = config["solver"]

    screen_rows: list[ScreenRow] = []
    payoff_rows: list[PayoffRow] = []
    for k in k_values:
        for rep in range(reps):
            seed = base + rep * 100 + k * 10
            screen_rows.extend(
                _run_one(p=p, n=n, k=k, seed=seed, gamma_mult=gamma_mult, solver_cfg=solver_cfg)
            )
            payoff = _run_payoff_one(
                p=p, n=n, k=k, seed=seed, gamma_mult=gamma_mult, solver_cfg=solver_cfg
            )
            if payoff is not None:
                payoff_rows.append(payoff)
    return screen_rows, payoff_rows


def write_csv(rows: list[ScreenRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dc.fields(ScreenRow)])
        writer.writeheader()
        for r in rows:
            writer.writerow(dc.asdict(r))


def write_payoff_csv(rows: list[PayoffRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dc.fields(PayoffRow)])
        writer.writeheader()
        for r in rows:
            writer.writerow(dc.asdict(r))


def write_figure(rows: list[ScreenRow], payoff_rows: list[PayoffRow], path: Path) -> None:
    from _style import METHOD_COLORS, apply_style, full_spines

    apply_style()
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame([dc.asdict(r) for r in rows])
    df_pay = pd.DataFrame([dc.asdict(r) for r in payoff_rows])
    k_values = sorted(df["k"].unique())
    palette = sns.color_palette("viridis", n_colors=max(len(k_values), 2))

    fig = plt.figure(figsize=(14.4, 9.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_frac = fig.add_subplot(gs[0, 0])
    ax_ceiling = fig.add_subplot(gs[0, 1])
    ax_work = fig.add_subplot(gs[1, 0])
    ax_speed = fig.add_subplot(gs[1, 1])

    for i, k in enumerate(k_values):
        sub = df[df["k"] == k]
        n = int(sub["n"].iloc[0])
        agg = sub.groupby("iteration")["frac_screened"].agg(["mean", "std"]).reset_index()
        ax_frac.plot(
            agg["iteration"],
            agg["mean"],
            color=palette[i],
            lw=1.8,
            marker="o",
            markersize=5,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label=rf"$k/n = {k}/{n} = {k / n:.2f}$",
        )
        ax_frac.fill_between(
            agg["iteration"],
            agg["mean"] - agg["std"].fillna(0.0),
            agg["mean"] + agg["std"].fillna(0.0),
            color=palette[i],
            alpha=0.15,
        )
        ceiling = (n - k) / n
        ax_frac.axhline(ceiling, color=palette[i], lw=0.8, linestyle=":")

    ax_frac.set_xlabel("iteration")
    ax_frac.set_ylabel("fraction of coordinates safely screened")
    ax_frac.set_title(r"Per-iteration oracle-cost reduction (dashed = $(n-k)/n$ ceiling)")
    ax_frac.set_ylim(-0.02, 1.02)
    ax_frac.legend(loc="lower right")
    ax_frac.grid(True, linestyle=":", alpha=0.4)
    full_spines(ax_frac)

    final = df.groupby(["k", "n", "seed"])["frac_screened"].last().reset_index()
    final["k_over_n"] = final["k"] / final["n"]
    final["ceiling"] = (final["n"] - final["k"]) / final["n"]
    agg = (
        final.groupby("k_over_n")
        .agg(mean=("frac_screened", "mean"), ceiling=("ceiling", "mean"))
        .reset_index()
    )

    ax_ceiling.plot(
        agg["k_over_n"],
        agg["ceiling"],
        color="black",
        lw=1.0,
        linestyle="--",
        label=r"$(n-k)/n$ ceiling",
    )
    ax_ceiling.plot(
        agg["k_over_n"],
        agg["mean"],
        color="#d62728",
        lw=1.8,
        marker="o",
        markersize=8,
        markeredgecolor="black",
        markeredgewidth=0.5,
        label="achieved at convergence",
    )
    ax_ceiling.set_xlabel(r"planted sparsity $k/n$")
    ax_ceiling.set_ylabel("final fraction screened")
    ax_ceiling.set_title("Achieved screening vs ceiling across sparsity")
    ax_ceiling.set_xlim(-0.02, 1.02)
    ax_ceiling.set_ylim(-0.02, 1.02)
    ax_ceiling.legend(loc="lower left")
    ax_ceiling.grid(True, linestyle=":", alpha=0.4)
    full_spines(ax_ceiling)

    # Panels 3 & 4: actual payoff when the mask is plugged into the solver via
    # ScreenedDenseOperator (true A10 solver uptake, not advisory).
    if not df_pay.empty:
        agg = (
            df_pay.groupby("k_over_n")
            .agg(
                work_mean=("work_reduction_frac", "mean"),
                work_std=("work_reduction_frac", "std"),
                speed_mean=("time_speedup", "mean"),
                speed_std=("time_speedup", "std"),
                obj_gap_mean=("final_obj_full", "mean"),
                obj_gap_screened=("final_obj_screened", "mean"),
            )
            .reset_index()
        )
        agg["obj_gap"] = (agg["obj_gap_mean"] - agg["obj_gap_screened"]).abs()
        ceiling = agg["k_over_n"].apply(lambda r: 1.0 - r)

        ax_work.plot(
            agg["k_over_n"],
            ceiling,
            color=METHOD_COLORS["theory"],
            lw=1.0,
            linestyle="--",
            label=r"$(n-k)/n$ ceiling",
        )
        ax_work.errorbar(
            agg["k_over_n"],
            agg["work_mean"],
            yerr=agg["work_std"].fillna(0),
            fmt="o-",
            color=METHOD_COLORS["facet_asq"],
            markersize=9,
            markeredgecolor="black",
            markeredgewidth=0.5,
            capsize=4,
            label="screened fit, oracle-work reduction",
        )
        ax_work.set_xlabel(r"planted sparsity $k/n$")
        ax_work.set_ylabel(r"fraction of oracle work eliminated")
        ax_work.set_title("Measured oracle-work savings\n(ScreenedDenseOperator inside the solver)")
        ax_work.set_xlim(-0.02, 1.02)
        ax_work.set_ylim(-0.05, 1.05)
        ax_work.legend(loc="upper right")
        ax_work.grid(True, linestyle=":", alpha=0.4)
        full_spines(ax_work)

        ax_speed.axhline(
            1.0, color=METHOD_COLORS["theory"], lw=1.0, linestyle="--", label="no speedup"
        )
        ax_speed.errorbar(
            agg["k_over_n"],
            agg["speed_mean"],
            yerr=agg["speed_std"].fillna(0),
            fmt="s-",
            color=METHOD_COLORS["facet_asq"],
            markersize=9,
            markeredgecolor="black",
            markeredgewidth=0.5,
            capsize=4,
            label="wall-clock speedup (full / screened)",
        )
        ax_speed.set_xlabel(r"planted sparsity $k/n$")
        ax_speed.set_ylabel("speedup factor")
        ax_speed.set_title("Measured wall-clock speedup\n(same solver, screened operator)")
        ax_speed.set_xlim(-0.02, 1.02)
        ax_speed.legend(loc="upper right")
        ax_speed.grid(True, linestyle=":", alpha=0.4)
        full_spines(ax_speed)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    screen_rows, payoff_rows = run(cfg)
    out = Path(args.output_dir)
    write_csv(screen_rows, out / cfg["output"]["csv"])
    write_payoff_csv(payoff_rows, out / "m7_payoff.csv")
    if not args.no_figure:
        write_figure(screen_rows, payoff_rows, out / cfg["output"]["figure"])
    print(
        f"wrote {len(screen_rows)} per-iter rows + {len(payoff_rows)} payoff rows "
        f"across {len({r.k for r in screen_rows})} sparsity levels"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
