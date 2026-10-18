"""M2, monosemanticity probe: do FACET components activate on semantically coherent tokens?

For each FACET and PCA component at a chosen GPT-2 (layer, site), we:
  1. Compute per-token projection scores ``s_j[t] = x_j^T h_t``.
  2. Take the top-K tokens by ``|s_j[t]|``.
  3. Measure semantic coherence of those K tokens via the mean pairwise cosine
     of their GPT-2 input (wte) embeddings.

A monosemantic component should concentrate on tokens whose embeddings cluster
in input space (synonyms, inflections, co-occurring vocabulary). The headline
statistic per method is the median across its m components of this coherence.
We also emit the top-K exemplar tokens of the top-6 FACET and top-6 PCA
components for the qualitative appendix.

See plan-2026-10-13T1908Z §M2.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses as dc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _llm_activations import (  # noqa: E402
    extract_residual_activations_with_token_ids,
    load_gpt2,
    token_input_embeddings,
)
from t3_llm_activations import _pca_components  # noqa: E402

from facet.papers.block_sparse import BlockSparseEstimator  # noqa: E402


@dc.dataclass(frozen=True)
class CoherenceRow:
    method: str
    component_rank: int
    top_k: int
    mean_pairwise_cosine: float
    gamma_used: float


def _unit_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return mat / norms


def _mean_pairwise_cosine(embeddings: np.ndarray) -> float:
    """Mean |cos| across all pairs of row-vectors in ``embeddings``."""
    if embeddings.shape[0] < 2:
        return 0.0
    unit = _unit_rows(embeddings)
    sims = unit @ unit.T
    k = sims.shape[0]
    # Off-diagonal only; signs matter for the direction, but synonyms live on the
    # same hemisphere and polysemantic components spread across hemispheres, so
    # we use signed cosine (not |cos|) as the coherence measure.
    iu = np.triu_indices(k, k=1)
    return float(np.mean(sims[iu]))


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


def _top_k_token_indices_per_component(
    components: np.ndarray, H: np.ndarray, top_k: int
) -> np.ndarray:
    """Return (m, top_k) array: token-column indices of the K largest |score| per component."""
    scores = components.T @ H  # (m, N)
    return np.argsort(-np.abs(scores), axis=1)[:, :top_k]


def _coherence_rows(
    method: str,
    gamma: float,
    top_idx: np.ndarray,
    emb: np.ndarray,
    top_k: int,
) -> list[CoherenceRow]:
    rows = []
    for j in range(top_idx.shape[0]):
        token_cols = top_idx[j]
        rows.append(
            CoherenceRow(
                method=method,
                component_rank=j,
                top_k=top_k,
                mean_pairwise_cosine=_mean_pairwise_cosine(emb[token_cols]),
                gamma_used=gamma,
            )
        )
    return rows


def run(config: dict[str, Any]) -> tuple[list[CoherenceRow], dict[str, Any]]:
    model_name = str(config["model"]["name"])
    layer = int(config["model"]["layer"])
    site = str(config["model"].get("site", "resid_post"))
    max_tokens = int(config["model"]["max_tokens_per_text"])
    texts = list(config["texts"])
    m = int(config["decomposition"]["m"])
    top_k = int(config["probe"]["top_k_tokens"])
    exemplar_components = int(config["probe"]["exemplar_components"])
    seed = int(config["seed"])
    gammas = [float(g) for g in config["facet"]["gamma_candidates"]]

    model, tokenizer = load_gpt2(model_name)
    H, token_ids = extract_residual_activations_with_token_ids(
        model,
        tokenizer,
        texts,
        layer=layer,
        max_tokens_per_text=max_tokens,
        site=site,
    )
    emb = token_input_embeddings(model, token_ids)  # (N, d_model)

    facet, gamma_used = _facet_components(H, m, gammas=gammas, seed=seed)
    pca = _pca_components(H, m)

    facet_top = _top_k_token_indices_per_component(facet, H, top_k)
    pca_top = _top_k_token_indices_per_component(pca, H, top_k)

    rows: list[CoherenceRow] = []
    rows.extend(
        _coherence_rows(
            "facet_asq",
            gamma_used if gamma_used is not None else float("nan"),
            facet_top,
            emb,
            top_k,
        )
    )
    rows.extend(_coherence_rows("pca", float("nan"), pca_top, emb, top_k))

    # Decode top-K tokens for the top exemplar_components (ranked by score-variance)
    def _rank_components(components: np.ndarray) -> np.ndarray:
        scores = components.T @ H
        return np.argsort(-scores.var(axis=1))

    exemplars: dict[str, list[list[str]]] = {}
    for name, components, top_idx in (("facet_asq", facet, facet_top), ("pca", pca, pca_top)):
        ranked = _rank_components(components)[:exemplar_components]
        exemplars[name] = []
        for j in ranked:
            cols = top_idx[j]
            tokens = [tokenizer.decode([int(token_ids[c])]).strip() for c in cols]
            exemplars[name].append(tokens)

    meta = {
        "layer": layer,
        "site": site,
        "m": m,
        "top_k": top_k,
        "n_tokens": int(H.shape[1]),
        "d_model": int(H.shape[0]),
        "exemplars": exemplars,
    }
    return rows, meta


def write_csv(rows: list[CoherenceRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dc.fields(CoherenceRow)])
        writer.writeheader()
        for r in rows:
            writer.writerow(dc.asdict(r))


def write_exemplars_json(meta: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta["exemplars"], indent=2, ensure_ascii=False))


def write_figure(rows: list[CoherenceRow], meta: dict[str, Any], path: Path) -> None:
    from _style import METHOD_COLORS, METHOD_LABELS, apply_style, full_spines

    apply_style()
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame([dc.asdict(r) for r in rows])
    methods = ["pca", "facet_asq"]
    palette = {m: METHOD_COLORS[m] for m in methods}
    labels = {m: METHOD_LABELS[m] for m in methods}
    df["label"] = df["method"].map(labels)
    label_order = [labels[m] for m in methods]

    fig, (ax_box, ax_hist) = plt.subplots(1, 2, figsize=(12.4, 4.8), constrained_layout=True)

    sns.boxplot(
        data=df,
        x="label",
        y="mean_pairwise_cosine",
        hue="label",
        order=label_order,
        hue_order=label_order,
        palette={labels[m]: palette[m] for m in methods},
        showfliers=False,
        legend=False,
        ax=ax_box,
    )
    sns.stripplot(
        data=df,
        x="label",
        y="mean_pairwise_cosine",
        order=label_order,
        color="black",
        size=3,
        alpha=0.5,
        ax=ax_box,
    )
    ax_box.set_title(
        f"Component-wise coherence (top-{meta['top_k']} tokens, {meta['m']} components)"
    )
    ax_box.set_xlabel("")
    ax_box.set_ylabel("mean pairwise cosine of top-token embeddings")
    ax_box.grid(axis="y", linestyle=":", alpha=0.4)
    full_spines(ax_box)

    for method in methods:
        vals = df[df["method"] == method]["mean_pairwise_cosine"].to_numpy()
        ax_hist.hist(
            vals,
            bins=12,
            alpha=0.65,
            color=palette[method],
            edgecolor="black",
            label=labels[method],
        )
    ax_hist.set_title("Distribution of per-component coherence")
    ax_hist.set_xlabel("mean pairwise cosine")
    ax_hist.set_ylabel("number of components")
    ax_hist.legend()
    ax_hist.grid(axis="y", linestyle=":", alpha=0.4)
    full_spines(ax_hist)

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
    rows, meta = run(cfg)
    out = Path(args.output_dir)
    write_csv(rows, out / cfg["output"]["csv"])
    write_exemplars_json(meta, out / cfg["output"]["exemplars"])
    if not args.no_figure:
        write_figure(rows, meta, out / cfg["output"]["figure"])
    print(
        f"wrote {len(rows)} coherence rows "
        f"({sum(1 for r in rows if r.method == 'facet_asq')} FACET + "
        f"{sum(1 for r in rows if r.method == 'pca')} PCA)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
