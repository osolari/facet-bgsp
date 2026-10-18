"""Tier-2, superposition toy: FACET vs native-neuron and PCA.

Trains an Elhage-style compressed autoencoder with ``n`` planted sparse
features squeezed into ``d < n`` hidden dimensions. Sparsity is swept
across three regimes:

  - trivial (sparsity ~ 1/n): the model learns the antipodal arrangement
    (6 unique directions in R^d), so "recovering n>d features" is
    cosmetic. Both FACET and PCA can succeed if they find the unique
    directions.
  - genuine superposition (sparsity ~ 3/n): the model is forced to
    arrange n>d features non-orthogonally. This is the real test.
  - dense (sparsity ~ 0.6): recovery breaks for everyone, negative
    control.

After training, hidden activations ``H = W @ Z`` are decomposed by
native-neuron, PCA, and FACET A-SQ with ``m = d`` components. We report
TWO recovery metrics:

  1. ``max_j |cos(w_i, x_j)|`` (lenient; one component can match
     multiple planted features).
  2. Hungarian bipartite-matched ``|cos|`` (strict; each component is
     assigned to at most one planted feature), the honest metric.

We also report the number of UNIQUE planted directions after collapsing
antipodal pairs, so the reader can see whether the "n > d" claim is
real at each sparsity level.
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

from _toy_superposition import (  # noqa: E402
    SuperpositionConfig,
    feature_recovery_cosines,
    hungarian_recovery_cosines,
    train_superposition,
    unique_planted_directions,
)

from facet.papers.block_sparse import BlockSparseEstimator  # noqa: E402


@dc.dataclass(frozen=True)
class RecoveryRow:
    sparsity: float
    regime: str
    method: str
    replicate: int
    mean_lenient: float
    mean_hungarian: float
    frac_hungarian_above_070: float
    frac_hungarian_above_090: float
    n_planted: int
    n_unique_planted: int
    d_hidden: int
    gamma_used: float


def _pca_components(H: np.ndarray, m: int) -> np.ndarray:
    cov = H @ H.T
    _, vecs = np.linalg.eigh(cov)
    return vecs[:, -m:]


def _native_neuron_components(d: int) -> np.ndarray:
    return np.eye(d)


def _facet_components(
    H: np.ndarray, m: int, gammas: list[float], seed: int
) -> tuple[np.ndarray, float | None]:
    for gamma in gammas:
        est = BlockSparseEstimator(
            objective="squared",
            gamma=gamma,
            solver="safeguarded",
            max_iterations=400,
            tolerance=1e-10,
            refit="none",
            random_state=seed,
        )
        try:
            fit = est.fit(H, m=m)
        except RuntimeError:
            continue
        return np.asarray(fit.raw.factors), gamma
    return np.zeros((H.shape[0], m)), None


def _row(
    method: str,
    rep: int,
    sparsity: float,
    regime: str,
    W: np.ndarray,
    unique_W: np.ndarray,
    components: np.ndarray,
    gamma: float,
) -> RecoveryRow:
    lenient = feature_recovery_cosines(components, W)
    hungarian = hungarian_recovery_cosines(components, W)
    return RecoveryRow(
        sparsity=float(sparsity),
        regime=regime,
        method=method,
        replicate=rep,
        mean_lenient=float(np.mean(lenient)),
        mean_hungarian=float(np.mean(hungarian)),
        frac_hungarian_above_070=float(np.mean(hungarian >= 0.70)),
        frac_hungarian_above_090=float(np.mean(hungarian >= 0.90)),
        n_planted=int(W.shape[1]),
        n_unique_planted=int(unique_W.shape[1]),
        d_hidden=int(W.shape[0]),
        gamma_used=float(gamma),
    )


def run(config: dict[str, Any]) -> list[RecoveryRow]:
    rows: list[RecoveryRow] = []
    base = int(config["seed_base"])
    gamma_candidates = [float(g) for g in config["facet"]["gamma_candidates"]]

    for regime_cfg in config["sparsity_regimes"]:
        regime = str(regime_cfg["label"])
        sparsity = float(regime_cfg["sparsity"])
        for rep in range(int(config["replicates"])):
            seed = base + hash((regime, rep)) % 1_000_000
            cfg = SuperpositionConfig(
                n=int(config["model"]["n"]),
                d=int(config["model"]["d"]),
                sparsity=sparsity,
                importance_decay=float(config["model"]["importance_decay"]),
                batch_size=int(config["training"]["batch_size"]),
                steps=int(config["training"]["steps"]),
                lr=float(config["training"]["lr"]),
                seed=seed,
            )
            W, _b, codes = train_superposition(cfg)
            H = W @ codes
            d, _n = W.shape
            m = d
            unique_W = unique_planted_directions(W)

            native = _native_neuron_components(d)
            pca = _pca_components(H, m)
            facet, gamma_used = _facet_components(H, m, gammas=gamma_candidates, seed=seed + 1)

            rows.append(
                _row("native_neuron", rep, sparsity, regime, W, unique_W, native, float("nan"))
            )
            rows.append(_row("pca", rep, sparsity, regime, W, unique_W, pca, float("nan")))
            rows.append(
                _row(
                    "facet_asq",
                    rep,
                    sparsity,
                    regime,
                    W,
                    unique_W,
                    facet,
                    gamma_used if gamma_used is not None else float("nan"),
                )
            )
    return rows


def write_csv(rows: list[RecoveryRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dc.fields(RecoveryRow)])
        writer.writeheader()
        for r in rows:
            writer.writerow(dc.asdict(r))


def write_figure(rows: list[RecoveryRow], path: Path) -> None:
    from _style import METHOD_COLORS, METHOD_LABELS, apply_style, full_spines

    apply_style()
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame([dc.asdict(r) for r in rows])
    regimes = list(df["regime"].unique())
    methods = ["native_neuron", "pca", "facet_asq"]
    palette = {m: METHOD_COLORS[m] for m in methods}
    labels = [METHOD_LABELS[m] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.8), constrained_layout=True)

    for ax, metric, title in (
        (axes[0], "mean_hungarian", "Mean Hungarian-matched |cos|  (strict)"),
        (
            axes[1],
            "frac_hungarian_above_090",
            r"Fraction of planted features with Hungarian |cos| $\geq$ 0.9",
        ),
    ):
        sns.barplot(
            data=df,
            x="regime",
            y=metric,
            hue="method",
            hue_order=methods,
            order=regimes,
            palette=palette,
            errorbar="sd",
            edgecolor="black",
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("sparsity regime")
        ax.set_ylabel("value")
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        full_spines(ax)
        legend = ax.get_legend()
        if legend is not None:
            for text, label in zip(legend.get_texts(), labels, strict=True):
                text.set_text(label)
            legend.set_title("")

    # Overlay a text annotation per regime showing n_unique vs n_planted
    unique_counts = df.groupby("regime")[["n_planted", "n_unique_planted"]].mean().astype(int)
    for ax in axes:
        for i, regime in enumerate(regimes):
            u = int(unique_counts.loc[regime, "n_unique_planted"])
            n = int(unique_counts.loc[regime, "n_planted"])
            ax.text(
                i,
                1.02,
                f"n={n}, unique={u}",
                ha="center",
                va="bottom",
                fontsize=10,
                color="0.3",
            )

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
    rows = run(cfg)
    out = Path(args.output_dir)
    write_csv(rows, out / cfg["output"]["csv"])
    if not args.no_figure:
        write_figure(rows, out / cfg["output"]["figure"])
    print(f"wrote {len(rows)} rows across {len({r.method for r in rows})} methods")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
