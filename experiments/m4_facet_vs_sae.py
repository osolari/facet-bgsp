"""M4, FACET vs sparse autoencoder, matched-dim head-to-head.

At matched ``m == dict_size``, compare FACET A-SQ, PCA, and a tied-decoder
SAE trained on the same GPT-2 small (layer, site) activations on:

  - reconstruction R² against the activation Frobenius energy
  - mean Hoyer sparsity of per-component scores
  - monosemanticity: mean pairwise cosine of top-10 token embeddings per
    component (M2-style)
  - mean causal-ablation KL across a probe battery (M3-style, scaled down)

The SAE's "components" are the dictionary columns ``W_e^T``; we treat them
as a (d_in, dict_size) basis identical in shape to the FACET and PCA
outputs so every comparison runs through the same metric routines.

See plan-2026-10-13T1908Z §M4.
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
    extract_residual_activations_with_token_ids,
    load_gpt2,
    token_input_embeddings,
)
from _sae import SAEConfig, dict_directions, train_sae  # noqa: E402
from m2_monosemanticity import (  # noqa: E402
    _mean_pairwise_cosine,
    _top_k_token_indices_per_component,
)
from t3_llm_activations import (  # noqa: E402
    _hoyer,
    _pca_components,
    _reconstruction_r2,
    _top_components_by_score_variance,
)

from facet.papers.block_sparse import BlockSparseEstimator  # noqa: E402


@dc.dataclass(frozen=True)
class M4Row:
    method: str
    reconstruction_r2: float
    mean_hoyer: float
    median_monosemanticity: float
    mean_ablation_kl_bits: float
    hyperparam: float
    m: int
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


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros_like(v)


def _measure(
    name: str,
    components: np.ndarray,
    H: np.ndarray,
    emb: np.ndarray,
    model: object,
    tokenizer: object,
    layer: int,
    probes: list[str],
    top_k: int,
    hyperparam: float,
) -> M4Row:
    # Reconstruction R^2 via orthonormalized basis of the components.
    r2 = _reconstruction_r2(H, components)
    # Hoyer sparsity per component on projection scores onto H.
    scores = components.T @ H
    hoy = float(np.mean([_hoyer(s) for s in scores]))
    # Monosemanticity (top-k token coherence per component).
    top_idx = _top_k_token_indices_per_component(components, H, top_k)
    coherences = [_mean_pairwise_cosine(emb[top_idx[j]]) for j in range(top_idx.shape[0])]
    monosem = float(np.median(coherences)) if coherences else 0.0
    # Causal probe: ablate the top-6 score-variance components across every probe.
    top6 = _top_components_by_score_variance(components, H, k=6)
    kls = []
    for probe in probes:
        for j in range(top6.shape[1]):
            direction = _unit(top6[:, j])
            if np.linalg.norm(direction) < 1e-9:
                continue
            kls.append(ablation_kl_bits(model, tokenizer, probe, layer=layer, direction=direction))
    mean_kl = float(np.mean(kls)) if kls else 0.0
    return M4Row(
        method=name,
        reconstruction_r2=r2,
        mean_hoyer=hoy,
        median_monosemanticity=monosem,
        mean_ablation_kl_bits=mean_kl,
        hyperparam=hyperparam,
        m=components.shape[1],
        n_tokens=H.shape[1],
    )


def run(config: dict[str, Any]) -> list[M4Row]:
    model_name = str(config["model"]["name"])
    layer = int(config["model"]["layer"])
    site = str(config["model"].get("site", "resid_post"))
    max_tokens = int(config["model"]["max_tokens_per_text"])
    texts = list(config["texts"])
    probes = list(config["probe"]["sentences"])
    top_k = int(config["probe"]["top_k_tokens"])
    m = int(config["decomposition"]["m"])
    seed = int(config["seed"])
    gammas = [float(g) for g in config["facet"]["gamma_candidates"]]
    sae_cfg = config["sae"]

    model, tokenizer = load_gpt2(model_name)
    H, token_ids = extract_residual_activations_with_token_ids(
        model,
        tokenizer,
        texts,
        layer=layer,
        max_tokens_per_text=max_tokens,
        site=site,
    )
    emb = token_input_embeddings(model, token_ids)
    d = H.shape[0]

    # FACET
    facet, gamma_used = _facet_components(H, m, gammas=gammas, seed=seed)
    # PCA
    pca = _pca_components(H, m)
    # SAE (matched dict size = m)
    sae_model, _diag = train_sae(
        H,
        SAEConfig(
            d_in=d,
            dict_size=m,
            l1_coef=float(sae_cfg["l1_coef"]),
            lr=float(sae_cfg["lr"]),
            steps=int(sae_cfg["steps"]),
            batch_size=int(sae_cfg["batch_size"]),
            seed=seed + 101,
        ),
    )
    sae_dict = dict_directions(sae_model)

    rows = [
        _measure(
            "facet_asq",
            facet,
            H,
            emb,
            model,
            tokenizer,
            layer,
            probes,
            top_k,
            gamma_used if gamma_used is not None else float("nan"),
        ),
        _measure("pca", pca, H, emb, model, tokenizer, layer, probes, top_k, float("nan")),
        _measure(
            "sae",
            sae_dict,
            H,
            emb,
            model,
            tokenizer,
            layer,
            probes,
            top_k,
            float(sae_cfg["l1_coef"]),
        ),
    ]
    return rows


def write_csv(rows: list[M4Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dc.fields(M4Row)])
        writer.writeheader()
        for r in rows:
            writer.writerow(dc.asdict(r))


def write_figure(rows: list[M4Row], path: Path) -> None:
    from _style import METHOD_COLORS, apply_style, full_spines

    apply_style()
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    palette = {
        "facet_asq": METHOD_COLORS["facet_asq"],
        "pca": METHOD_COLORS["pca"],
        "sae": "#9467bd",  # purple, reserve for SAE
    }
    labels = {"facet_asq": "FACET A-SQ", "pca": "PCA", "sae": "SAE (tied, dict=m)"}
    order = ["pca", "sae", "facet_asq"]

    df = pd.DataFrame([dc.asdict(r) for r in rows])
    df["label"] = df["method"].map(labels)
    label_order = [labels[m] for m in order]

    fig, axes = plt.subplots(1, 4, figsize=(16.8, 4.8), constrained_layout=True)

    panels = [
        ("reconstruction_r2", r"Reconstruction $R^2$"),
        ("mean_hoyer", "Mean Hoyer sparsity"),
        ("median_monosemanticity", r"Median monosemanticity (top-$K$ coherence)"),
        ("mean_ablation_kl_bits", "Mean ablation KL (bits)"),
    ]
    for ax, (col, title) in zip(axes, panels, strict=True):
        sns.barplot(
            data=df,
            x="label",
            y=col,
            hue="label",
            order=label_order,
            hue_order=label_order,
            palette={labels[m]: palette[m] for m in order},
            edgecolor="black",
            legend=False,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=15)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
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
    print(f"wrote {len(rows)} rows ({', '.join(r.method for r in rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
