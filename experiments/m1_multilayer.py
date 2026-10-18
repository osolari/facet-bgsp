"""M1, multi-layer FACET sweep on GPT-2 small.

For each (block index L, hook site s in {resid_post, attn_out, mlp_out})
pair, extract activations from a short public-domain corpus and fit
m=32-component FACET A-SQ (safeguarded) and PCA. Record per-cell:

  - reconstruction R^2 (projection onto component subspace)
  - mean Hoyer sparsity per component
  - mean score kurtosis per component

The figure shows (FACET − PCA) for each metric as a layer × site heatmap,
so one can read the layers / sites where FACET's threshold operator pays
off relative to dense PCA on the same data.

See plan-2026-10-13T1908Z §M1.
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

from _llm_activations import extract_residual_activations, load_gpt2  # noqa: E402
from t3_llm_activations import (  # noqa: E402
    _pca_components,
    _per_component_metrics,
    _reconstruction_r2,
)

from facet.papers.block_sparse import BlockSparseEstimator  # noqa: E402


@dc.dataclass(frozen=True)
class M1Row:
    layer: int
    site: str
    method: str
    reconstruction_r2: float
    mean_hoyer: float
    mean_kurtosis: float
    gamma_used: float
    n_tokens: int


def _facet_components(
    H: np.ndarray, m: int, gammas: list[float], seed: int
) -> tuple[np.ndarray, float | None]:
    for gamma in gammas:
        est = BlockSparseEstimator(
            objective="squared",
            gamma=gamma,
            solver="safeguarded",
            max_iterations=300,
            tolerance=1e-9,
            refit="none",
            random_state=seed,
        )
        try:
            fit = est.fit(H, m=m)
        except RuntimeError:
            continue
        return np.asarray(fit.raw.factors), gamma
    return np.zeros((H.shape[0], m)), None


def run(config: dict[str, Any]) -> list[M1Row]:
    model_name = str(config["model"]["name"])
    layers = [int(L) for L in config["model"]["layers"]]
    sites = [str(s) for s in config["model"]["sites"]]
    max_tokens = int(config["model"]["max_tokens_per_text"])
    texts = list(config["texts"])
    m = int(config["decomposition"]["m"])
    seed = int(config["seed"])
    gammas = [float(g) for g in config["facet"]["gamma_candidates"]]

    model, tokenizer = load_gpt2(model_name)

    rows: list[M1Row] = []
    for layer in layers:
        for site in sites:
            H = extract_residual_activations(
                model,
                tokenizer,
                texts,
                layer=layer,
                max_tokens_per_text=max_tokens,
                site=site,
            )
            N = H.shape[1]
            # PCA
            pca = _pca_components(H, m)
            pca_hoy, pca_kurt = _per_component_metrics(pca, H)
            rows.append(
                M1Row(
                    layer=layer,
                    site=site,
                    method="pca",
                    reconstruction_r2=_reconstruction_r2(H, pca),
                    mean_hoyer=pca_hoy,
                    mean_kurtosis=pca_kurt,
                    gamma_used=float("nan"),
                    n_tokens=N,
                )
            )
            # FACET
            facet, gamma_used = _facet_components(H, m, gammas=gammas, seed=seed + layer * 10)
            fac_hoy, fac_kurt = _per_component_metrics(facet, H)
            rows.append(
                M1Row(
                    layer=layer,
                    site=site,
                    method="facet_asq",
                    reconstruction_r2=_reconstruction_r2(H, facet),
                    mean_hoyer=fac_hoy,
                    mean_kurtosis=fac_kurt,
                    gamma_used=gamma_used if gamma_used is not None else float("nan"),
                    n_tokens=N,
                )
            )
    return rows


def write_csv(rows: list[M1Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dc.fields(M1Row)])
        writer.writeheader()
        for r in rows:
            writer.writerow(dc.asdict(r))


def write_figure(rows: list[M1Row], path: Path) -> None:
    from _style import apply_style, full_spines

    apply_style()
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame([dc.asdict(r) for r in rows])

    def _gap(metric: str) -> pd.DataFrame:
        facet = df[df["method"] == "facet_asq"].pivot(index="layer", columns="site", values=metric)
        pca = df[df["method"] == "pca"].pivot(index="layer", columns="site", values=metric)
        return facet - pca

    metrics = [
        ("reconstruction_r2", r"$\Delta R^2$ (FACET $-$ PCA)", "RdBu_r"),
        ("mean_hoyer", r"$\Delta$ Hoyer sparsity", "RdBu_r"),
        ("mean_kurtosis", r"$\Delta$ kurtosis", "RdBu_r"),
    ]
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(5.0 * len(metrics), 4.6), constrained_layout=True
    )

    for ax, (metric, title, cmap) in zip(axes, metrics, strict=True):
        gap = _gap(metric)
        vmax = float(np.max(np.abs(gap.values))) if gap.size else 1.0
        vmax = max(vmax, 1e-9)
        sns.heatmap(
            gap,
            ax=ax,
            annot=True,
            fmt=".3f" if metric != "mean_kurtosis" else ".2f",
            cmap=cmap,
            center=0.0,
            vmin=-vmax,
            vmax=vmax,
            linewidths=0.5,
            linecolor="w",
            cbar_kws={"label": title},
        )
        ax.set_title(title)
        ax.set_xlabel("hook site")
        ax.set_ylabel("transformer block")
        ax.invert_yaxis()
        full_spines(ax)

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
    layers = sorted({r.layer for r in rows})
    sites = sorted({r.site for r in rows})
    print(f"wrote {len(rows)} rows ({len(layers)} layers × {len(sites)} sites × 2 methods)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
