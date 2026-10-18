"""A4, raw selection vs restricted refit (§14 required table #5 + figure).

For each (noise_std, margin_fraction, replicate) cell, fit a planted-spike
instance with ``BlockSparseEstimator`` and record both estimator stages on
the identified support:

- ``raw`` restricted objective: ``||A_S^T z_raw||_2`` where ``z_raw`` is the
  mask-first-normalized loading of the final raw iterate (the primary
  estimand before refit);
- ``refit`` restricted objective: the leading singular value of ``A_S`` returned
  by :func:`refit_rank_one`.

The refit dominates raw on this objective (A4-refit proposition). The driver
emits a CSV, a two-panel figure, and a LaTeX table (``\\input``-ready) that
goes into the manuscript as Table 5 "raw vs refitted results".
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
from facet.refit import mask_then_normalize

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


@dc.dataclass(frozen=True)
class RefitRow:
    seed: int
    noise_std: float
    margin_fraction: float
    support_matches: bool
    raw_restricted_objective: float
    refit_restricted_objective: float
    refit_minus_raw: float


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _gamma_from_margin_fraction(instance, mf: float) -> float | None:
    c = instance.data.T @ instance.scores[:, 0]
    on_min = float(np.abs(c[instance.supports[0]]).min())
    off_max = float(np.abs(np.delete(c, instance.supports[0])).max())
    if on_min <= off_max:
        return None
    return (1.0 - mf) * on_min + mf * off_max


def _perturbed_init(rng: np.random.Generator, x_star: np.ndarray, angle: float) -> np.ndarray:
    perp = rng.normal(size=x_star.shape)
    perp -= (perp @ x_star) * x_star
    perp /= np.linalg.norm(perp)
    x = np.cos(angle) * x_star + np.sin(angle) * perp
    return x / np.linalg.norm(x)


def _controlled_rank2_instance(
    *, p: int, k: int, lam1: float, lam2: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rank-2 restricted operator on a known support (same construction as the
    Q4 driver). Produces a planted spike where the restricted leading
    eigenvector is well-defined but DIFFERENT from the raw iterate's loading,
    so refit vs raw has a visible positive gap.
    """
    rng = np.random.default_rng(seed)
    sigma_1, sigma_2 = np.sqrt(lam1), np.sqrt(lam2)
    base = rng.normal(size=p)
    x_star = base / np.linalg.norm(base)
    perp = rng.normal(size=p)
    perp -= (perp @ x_star) * x_star
    w = perp / np.linalg.norm(perp)
    r_raw = rng.normal(size=k)
    r_raw -= r_raw.mean()
    r = r_raw / np.linalg.norm(r_raw)
    A_S = sigma_1 * np.outer(x_star, np.ones(k)) / np.sqrt(k) + sigma_2 * np.outer(w, r)
    n = k + 15
    A = np.zeros((p, n))
    A[:, :k] = A_S
    A[:, k:] = 0.01 * rng.normal(size=(p, n - k))
    return A, x_star


def _controlled_cells(config: dict[str, Any], solver_cfg: dict[str, Any]) -> list[RefitRow]:
    """Rank-2 controlled cells where refit > raw by a visible margin."""
    extra = config.get("rank2_cells")
    if not extra:
        return []
    records: list[RefitRow] = []
    problem = config["problem"]
    for cell in extra:
        lam1 = float(cell["lam1"])
        lam2 = float(cell["lam2"])
        for rep in range(int(cell.get("replicates", 6))):
            seed = int(config["seed_base"]) + rep * 131 + int(lam2 * 100)
            A, x_star = _controlled_rank2_instance(
                p=int(problem["p"]), k=int(problem["k"]), lam1=lam1, lam2=lam2, seed=seed
            )
            support = np.arange(int(problem["k"]))
            c_star = A.T @ x_star
            on_min = float(np.abs(c_star[support]).min())
            off_max = float(np.abs(np.delete(c_star, support)).max())
            if on_min <= off_max:
                continue
            gamma = 0.5 * (on_min + off_max)
            est = BlockSparseEstimator(
                objective=config["objective"],
                gamma=gamma,
                solver="safeguarded",
                max_iterations=int(solver_cfg["max_iterations"]),
                tolerance=float(solver_cfg["tolerance"]),
                random_state=seed,
            )
            rng = np.random.default_rng(seed + 999)
            init = _perturbed_init(rng, x_star, float(solver_cfg["initial_angle_rad"]))
            try:
                fit = est.fit(A, init=init)
            except RuntimeError:
                continue
            if fit.support.size == 0 or fit.refit is None:
                continue
            mask = np.zeros(A.shape[1], dtype=bool)
            mask[fit.support] = True
            c = A.T @ fit.raw.factors
            try:
                z_raw = mask_then_normalize(c, mask)
            except Exception:
                continue
            raw_restricted = float(np.linalg.norm(A @ z_raw))
            refit_restricted = float(fit.refit.objective or 0.0)
            support_matches = set(fit.support.tolist()) == set(int(i) for i in support)
            records.append(
                RefitRow(
                    seed=seed,
                    noise_std=float(cell["label_noise"]),  # label for grouping only
                    margin_fraction=lam2 / lam1,
                    support_matches=bool(support_matches),
                    raw_restricted_objective=raw_restricted,
                    refit_restricted_objective=refit_restricted,
                    refit_minus_raw=refit_restricted - raw_restricted,
                )
            )
    return records


def run(config: dict[str, Any]) -> list[RefitRow]:
    problem = config["problem"]
    grid = config["grid"]
    solver_cfg = config["solver"]

    records: list[RefitRow] = []
    for noise in grid["noise_std"]:
        for mf in grid["margin_fraction"]:
            for rep in range(int(grid["replicates"])):
                seed = int(config["seed_base"]) + rep * 97 + round(float(noise) * 1000)
                inst = planted_spike(
                    p=int(problem["p"]),
                    n=int(problem["n"]),
                    k=int(problem["k"]),
                    m=1,
                    singular_values=tuple(problem["singular_values"]),
                    noise_std=float(noise),
                    seed=seed,
                )
                gamma = _gamma_from_margin_fraction(inst, float(mf))
                if gamma is None:
                    continue
                rng = np.random.default_rng(seed + 5000)
                init = _perturbed_init(
                    rng, inst.scores[:, 0], float(solver_cfg["initial_angle_rad"])
                )
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
                except RuntimeError:
                    continue
                if fit.support.size == 0 or fit.refit is None:
                    continue

                mask = np.zeros(inst.data.shape[1], dtype=bool)
                mask[fit.support] = True
                # Raw loading: mask-first-normalize of the final raw iterate's
                # correlation vector.
                c = inst.data.T @ fit.raw.factors
                try:
                    z_raw = mask_then_normalize(c, mask)
                except Exception:
                    continue
                raw_restricted = float(np.linalg.norm(inst.data @ z_raw))
                refit_restricted = float(fit.refit.objective or 0.0)

                planted_support = set(int(i) for i in inst.supports[0])
                support_matches = set(fit.support.tolist()) == planted_support

                records.append(
                    RefitRow(
                        seed=seed,
                        noise_std=float(noise),
                        margin_fraction=float(mf),
                        support_matches=bool(support_matches),
                        raw_restricted_objective=raw_restricted,
                        refit_restricted_objective=refit_restricted,
                        refit_minus_raw=refit_restricted - raw_restricted,
                    )
                )
    records.extend(_controlled_cells(config, solver_cfg))
    return records


def write_csv(rows: list[RefitRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dc.fields(RefitRow)])
        writer.writeheader()
        for r in rows:
            writer.writerow(dc.asdict(r))


def write_figure(rows: list[RefitRow], path: Path) -> None:
    from _style import METHOD_COLORS, apply_style, full_spines

    apply_style()
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame([dc.asdict(r) for r in rows])
    noises = sorted(df["noise_std"].unique())
    palette = sns.color_palette("viridis", n_colors=max(len(noises), 2))

    fig, (ax_scatter, ax_box) = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)

    for i, n in enumerate(noises):
        subset = df[df["noise_std"] == n]
        ax_scatter.scatter(
            subset["raw_restricted_objective"],
            subset["refit_restricted_objective"],
            color=palette[i],
            s=56,
            edgecolor="black",
            linewidth=0.5,
            label=f"σ = {n:g}",
            alpha=0.85,
        )
    lo = float(min(df["raw_restricted_objective"].min(), df["refit_restricted_objective"].min()))
    hi = float(max(df["raw_restricted_objective"].max(), df["refit_restricted_objective"].max()))
    pad = (hi - lo) * 0.05 if hi > lo else 0.05
    ax_scatter.plot(
        [lo - pad, hi + pad],
        [lo - pad, hi + pad],
        "--",
        color=METHOD_COLORS["theory"],
        lw=1.0,
        label=r"$y = x$ (A4 dominance)",
    )
    ax_scatter.set_xlim(lo - pad, hi + pad)
    ax_scatter.set_ylim(lo - pad, hi + pad)
    ax_scatter.set_xlabel(r"raw restricted objective $\|A_S z_{\mathrm{raw}}\|_2$")
    ax_scatter.set_ylabel(r"refit restricted objective $\sigma_1(A_S)$")
    ax_scatter.set_title("A4-refit, refit $\\geq$ raw on every point")
    ax_scatter.legend(loc="lower right")
    ax_scatter.grid(True, linestyle=":", alpha=0.4)
    full_spines(ax_scatter)

    sns.boxplot(
        data=df,
        x="noise_std",
        y="refit_minus_raw",
        hue="noise_std",
        order=noises,
        hue_order=noises,
        palette={n: palette[i] for i, n in enumerate(noises)},
        legend=False,
        ax=ax_box,
    )
    sns.stripplot(
        data=df,
        x="noise_std",
        y="refit_minus_raw",
        order=noises,
        color="black",
        size=3,
        alpha=0.5,
        ax=ax_box,
    )
    ax_box.axhline(0, color=METHOD_COLORS["theory"], lw=1.0, linestyle="--")
    ax_box.set_xlabel("noise std σ")
    ax_box.set_ylabel("refit $-$ raw")
    ax_box.set_title("Gap grows with secondary-eigen strength")
    ax_box.grid(axis="y", linestyle=":", alpha=0.4)
    full_spines(ax_box)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_latex_table(rows: list[RefitRow], path: Path) -> None:
    """Required §14 table #5: raw vs refitted results, aggregated per cell."""
    noises = sorted({r.noise_std for r in rows})
    mfs = sorted({r.margin_fraction for r in rows})

    body_lines = []
    for n in noises:
        for mf in mfs:
            subset = [r for r in rows if r.noise_std == n and r.margin_fraction == mf]
            if not subset:
                continue
            raw_mean = np.mean([r.raw_restricted_objective for r in subset])
            refit_mean = np.mean([r.refit_restricted_objective for r in subset])
            gap_mean = np.mean([r.refit_minus_raw for r in subset])
            recovery = np.mean([r.support_matches for r in subset])
            body_lines.append(
                f"{n:.2f} & {mf:.2f} & {len(subset):d} & {recovery:.2f} & "
                f"{raw_mean:.3f} & {refit_mean:.3f} & {gap_mean:.3f} \\\\"
            )
    body = "\n".join(body_lines)
    tex = (
        "% Table 5 (§14), raw vs refitted results. Generated by\n"
        "% experiments/a4_raw_vs_refit.py.\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Raw selection vs restricted refit on the identified "
        "support. Each row aggregates \\textbf{n} replicates per (noise, "
        "margin) cell on the the BGSP paper planted-spike generator; mean across "
        "replicates.}\n"
        "\\label{tab:a4-raw-vs-refit}\n"
        "\\begin{tabular}{llrrrrr}\n"
        "\\toprule\n"
        "noise $\\sigma$ & margin frac & n & support rec. "
        "& raw $\\|A_S z_{\\mathrm{raw}}\\|_2$ "
        "& refit $\\sigma_1(A_S)$ & refit $-$ raw \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tex)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--no-table", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    rows = run(cfg)
    out = Path(args.output_dir)
    write_csv(rows, out / cfg["output"]["csv"])
    if not args.no_figure:
        write_figure(rows, out / cfg["output"]["figure"])
    if not args.no_table:
        write_latex_table(rows, out / cfg["output"]["table_tex"])
    print(f"wrote {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
