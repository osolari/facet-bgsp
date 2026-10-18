"""Tier-3, GPT-2 residual-stream activations: FACET vs PCA vs random (§14 figure #7).

See plan-2026-10-13T1908Z §T3 for the audit that motivated this rewrite.

Loads GPT-2 small, extracts residual-stream activations at a chosen
transformer block over a longer public-domain text corpus, and fits three
decompositions with the same number of components ``m``:

  (1) FACET A-SQ (safeguarded Riemannian ascent on the squared threshold objective)
  (2) PCA (top-``m`` eigenvectors of ``H H^T``)
  (3) Random orthogonal frame (sanity baseline)

Metrics per method:

  - **Reconstruction R²** : ``1 - ||H - P_m H||_F² / ||H||_F²``.
  - **Hoyer sparsity**    : per-component ``(sqrt(N) - L1/L2) / (sqrt(N) - 1)``
    averaged over components. Higher = more concentrated scores.
  - **Mean |kurtosis|**   : per-component Fisher kurtosis of the score
    distribution; higher = heavier-tailed = sparser.
  - **Causal-ablation KL**: multi-probe × top-k components. For each (probe,
    component) ablate the component from the residual stream and record
    KL(clean || ablated) of next-token logits. Null is a matched-norm random
    orthogonal direction, repeated per probe.
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

from _llm_activations import (  # noqa: E402
    ablation_kl_bits,
    extract_residual_activations,
    load_gpt2,
)

from facet.papers.block_sparse import BlockSparseEstimator  # noqa: E402


@dc.dataclass(frozen=True)
class ScalarRow:
    method: str
    m: int
    reconstruction_r2: float
    mean_hoyer: float
    mean_kurtosis: float
    gamma_used: float
    d_model: int
    n_tokens: int
    layer: int


@dc.dataclass(frozen=True)
class CausalRow:
    method: str
    probe_idx: int
    component_rank: int
    kl_bits: float


def _pca_components(H: np.ndarray, m: int) -> np.ndarray:
    cov = H @ H.T
    _, vecs = np.linalg.eigh(cov)
    return vecs[:, -m:]


def _random_orthogonal(d: int, m: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(d, m)))
    return q[:, :m]


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


def _reconstruction_r2(H: np.ndarray, components: np.ndarray) -> float:
    if np.allclose(components, 0.0):
        return 0.0
    q, _ = np.linalg.qr(components)
    proj = q @ (q.T @ H)
    num = float(np.linalg.norm(H - proj, "fro") ** 2)
    den = float(np.linalg.norm(H, "fro") ** 2)
    return 1.0 - num / den if den > 0 else 0.0


def _hoyer(z: np.ndarray) -> float:
    N = int(z.size)
    if N < 2:
        return 0.0
    l1 = float(np.abs(z).sum())
    l2 = float(np.linalg.norm(z))
    if l2 < 1e-12:
        return 0.0
    return float((np.sqrt(N) - l1 / l2) / (np.sqrt(N) - 1.0))


def _kurtosis(z: np.ndarray) -> float:
    m = float(z.mean())
    s = float(z.std())
    if s < 1e-12:
        return 0.0
    return float(np.mean(((z - m) / s) ** 4))


def _per_component_metrics(components: np.ndarray, H: np.ndarray) -> tuple[float, float]:
    scores = components.T @ H  # (m, N)
    hoyers = [_hoyer(s) for s in scores]
    kurts = [_kurtosis(s) for s in scores]
    return float(np.mean(hoyers)), float(np.mean(kurts))


def _top_components_by_score_variance(components: np.ndarray, H: np.ndarray, k: int) -> np.ndarray:
    scores = components.T @ H
    ranked = np.argsort(-scores.var(axis=1))
    return components[:, ranked[:k]]


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


def run(config: dict[str, Any]) -> tuple[list[ScalarRow], list[CausalRow], dict[str, Any]]:
    model_name = str(config["model"]["name"])
    layer = int(config["model"]["layer"])
    max_tokens = int(config["model"]["max_tokens_per_text"])
    texts = list(config["texts"])
    probe_sentences = list(config["probe"]["sentences"])
    top_k = int(config["probe"]["top_k_components"])
    m = int(config["decomposition"]["m"])
    seed = int(config["seed"])
    gammas = [float(g) for g in config["facet"]["gamma_candidates"]]

    model, tokenizer = load_gpt2(model_name)
    H = extract_residual_activations(
        model, tokenizer, texts, layer=layer, max_tokens_per_text=max_tokens
    )
    d, N = H.shape

    facet, gamma_used = _facet_components(H, m=m, gammas=gammas, seed=seed)
    pca = _pca_components(H, m)
    rand = _random_orthogonal(d, m, seed=seed + 7)
    rng = np.random.default_rng(seed + 13)

    scalar_rows: list[ScalarRow] = []
    causal_rows: list[CausalRow] = []

    def _scalar(name: str, components: np.ndarray, gamma: float) -> ScalarRow:
        hoy, kurt = _per_component_metrics(components, H)
        return ScalarRow(
            method=name,
            m=m,
            reconstruction_r2=_reconstruction_r2(H, components),
            mean_hoyer=hoy,
            mean_kurtosis=kurt,
            gamma_used=gamma,
            d_model=d,
            n_tokens=N,
            layer=layer,
        )

    scalar_rows.append(
        _scalar("facet_asq", facet, gamma_used if gamma_used is not None else float("nan"))
    )
    scalar_rows.append(_scalar("pca", pca, float("nan")))
    scalar_rows.append(_scalar("random_orthogonal", rand, float("nan")))

    # Multi-probe causal battery.
    bundles = {
        "facet_asq": _top_components_by_score_variance(facet, H, top_k),
        "pca": _top_components_by_score_variance(pca, H, top_k),
        "random_orthogonal": _top_components_by_score_variance(rand, H, top_k),
    }
    for probe_idx, probe_text in enumerate(probe_sentences):
        for method, top_comps in bundles.items():
            for comp_rank in range(top_comps.shape[1]):
                direction = _unit(top_comps[:, comp_rank])
                if np.linalg.norm(direction) < 1e-9:
                    kl = float("nan")
                else:
                    kl = ablation_kl_bits(
                        model, tokenizer, probe_text, layer=layer, direction=direction
                    )
                causal_rows.append(
                    CausalRow(
                        method=method,
                        probe_idx=probe_idx,
                        component_rank=comp_rank,
                        kl_bits=kl,
                    )
                )
        # Null: a fresh matched-norm random direction per probe, same top_k count.
        for comp_rank in range(top_k):
            direction = _unit(rng.normal(size=d))
            kl = ablation_kl_bits(model, tokenizer, probe_text, layer=layer, direction=direction)
            causal_rows.append(
                CausalRow(
                    method="null_random_direction",
                    probe_idx=probe_idx,
                    component_rank=comp_rank,
                    kl_bits=kl,
                )
            )

    meta = {"d_model": d, "n_tokens": N, "layer": layer, "m": m, "top_k": top_k}
    return scalar_rows, causal_rows, meta


def write_csv(
    scalar_rows: list[ScalarRow], causal_rows: list[CausalRow], paths: dict[str, Path]
) -> None:
    paths["scalar"].parent.mkdir(parents=True, exist_ok=True)
    with paths["scalar"].open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dc.fields(ScalarRow)])
        writer.writeheader()
        for r in scalar_rows:
            writer.writerow(dc.asdict(r))
    with paths["causal"].open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dc.fields(CausalRow)])
        writer.writeheader()
        for r in causal_rows:
            writer.writerow(dc.asdict(r))


def write_figure(
    scalar_rows: list[ScalarRow],
    causal_rows: list[CausalRow],
    meta: dict[str, Any],
    path: Path,
) -> None:
    from _style import METHOD_COLORS, METHOD_LABELS, apply_style, full_spines

    apply_style()
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    methods = ["random_orthogonal", "pca", "facet_asq"]
    palette = {m: METHOD_COLORS[m] for m in methods}
    palette["null_random_direction"] = METHOD_COLORS["random_orthogonal"]
    labels = {m: METHOD_LABELS[m] for m in methods}
    labels["null_random_direction"] = "null random dir"

    scalar_df = pd.DataFrame([dc.asdict(r) for r in scalar_rows])
    causal_df = pd.DataFrame([dc.asdict(r) for r in causal_rows])

    fig = plt.figure(figsize=(14.4, 8.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax_r2 = fig.add_subplot(gs[0, 0])
    ax_hoyer = fig.add_subplot(gs[0, 1])
    ax_kurt = fig.add_subplot(gs[0, 2])
    ax_box = fig.add_subplot(gs[1, :])

    def _scalar_bar(ax, column, title, ylabel):
        subset = scalar_df[scalar_df["method"].isin(methods)].copy()
        subset["label"] = subset["method"].map(labels)
        sns.barplot(
            data=subset,
            x="label",
            y=column,
            hue="label",
            order=[labels[m] for m in methods],
            hue_order=[labels[m] for m in methods],
            palette={labels[m]: palette[m] for m in methods},
            edgecolor="black",
            legend=False,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        full_spines(ax)

    _scalar_bar(ax_r2, "reconstruction_r2", r"Reconstruction $R^2$", r"$R^2$")
    _scalar_bar(ax_hoyer, "mean_hoyer", "Mean Hoyer sparsity", "Hoyer")
    _scalar_bar(ax_kurt, "mean_kurtosis", "Mean score kurtosis", "kurtosis")

    # Causal battery boxplot: KL(clean || ablated) across (probe, top_k component)
    causal_methods = ["random_orthogonal", "null_random_direction", "pca", "facet_asq"]
    causal_df_used = causal_df[causal_df["method"].isin(causal_methods)].copy()
    causal_df_used["label"] = causal_df_used["method"].map(labels)
    order = [labels[m] for m in causal_methods]

    sns.boxplot(
        data=causal_df_used,
        x="label",
        y="kl_bits",
        hue="label",
        hue_order=order,
        order=order,
        palette={labels[m]: palette[m] for m in causal_methods},
        showfliers=False,
        legend=False,
        ax=ax_box,
    )
    sns.stripplot(
        data=causal_df_used,
        x="label",
        y="kl_bits",
        order=order,
        color="black",
        size=2.5,
        alpha=0.4,
        ax=ax_box,
    )
    ax_box.set_title(
        rf"Multi-probe causal-ablation KL(clean $\|$ ablated), top-{meta['top_k']} components $\times$ {len(set(causal_df['probe_idx']))} probes"
    )
    ax_box.set_xlabel("")
    ax_box.set_ylabel("KL (bits)")
    ax_box.grid(axis="y", linestyle=":", alpha=0.4)
    full_spines(ax_box)

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
    scalar_rows, causal_rows, meta = run(cfg)
    out = Path(args.output_dir)
    write_csv(
        scalar_rows,
        causal_rows,
        {"scalar": out / cfg["output"]["csv"], "causal": out / "t3_causal.csv"},
    )
    if not args.no_figure:
        write_figure(scalar_rows, causal_rows, meta, out / cfg["output"]["figure"])
    print(
        f"wrote {len(scalar_rows)} scalar rows + {len(causal_rows)} causal rows "
        f"(d={meta['d_model']}, N={meta['n_tokens']}, layer={meta['layer']}, m={meta['m']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
