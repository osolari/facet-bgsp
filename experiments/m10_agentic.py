"""M10 - agentic AI: multi-turn tool-use conversation activation decomposition.

Hand-written multi-turn conversations alternate through labelled phases
(greeting, planning, tool_call, tool_response, reasoning, answer, summary).
We render them end-to-end, extract GPT-2 small residual-stream activations
token by token at a chosen block, fit FACET A-SQ and PCA at matched m, and
measure the extent to which each method's dominant-component partition
tracks the ground-truth phase sequence.

Hero: a two-panel component-x-phase heatmap. For each (method, component,
phase) cell the mean magnitude of the component's score on tokens in that
phase, standardised per component. FACET components should show spiky,
phase-selective bands; PCA components should be comparatively flat.
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

from _agentic import CONVERSATIONS, PHASE_PALETTE, PHASES, build_corpus  # noqa: E402
from _llm_activations import load_gpt2  # noqa: E402
from t3_llm_activations import _pca_components, _reconstruction_r2  # noqa: E402

from facet.papers.block_sparse import BlockSparseEstimator  # noqa: E402


@dc.dataclass(frozen=True)
class M10Row:
    method: str
    adjusted_mutual_info: float
    reconstruction_r2: float
    mean_phase_selectivity: float
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


def _extract_residual_per_token(
    model: Any, token_ids: list[int], layer: int, site: str = "resid_post"
) -> np.ndarray:
    """Run GPT-2 on a flat token-ID stream and capture the per-token residual-stream
    activation at the chosen block. Returns a ``(d_model, n_tokens)`` matrix."""
    import torch

    block = model.transformer.h[layer]
    captured: list[Any] = []

    def _hook(_module: Any, _inputs: Any, outputs: Any) -> None:
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
        captured.append(hidden.detach())

    with torch.no_grad():
        target = (
            block
            if site == "resid_post"
            else getattr(block, {"attn_out": "attn", "mlp_out": "mlp"}[site])
        )
        handle = target.register_forward_hook(_hook)
        try:
            # Feed in chunks so we avoid blowing the context length on huge streams.
            chunk = 512
            for i in range(0, len(token_ids), chunk):
                ids = torch.as_tensor([token_ids[i : i + chunk]], dtype=torch.long)
                model(ids)
        finally:
            handle.remove()
    stacked = torch.cat([h.squeeze(0) for h in captured], dim=0)
    return stacked.T.cpu().numpy().astype(np.float64)


def _phase_selectivity(
    scores: np.ndarray, phase_ids: np.ndarray, n_phases: int
) -> tuple[np.ndarray, float]:
    """For every component return a vector of mean |score| per phase (shape
    (m, n_phases)) plus a scalar ``selectivity`` averaged across components,
    defined as (max_phase_mean - median_phase_mean) / max_phase_mean. Higher
    selectivity = component fires on one phase relative to the others."""
    m = scores.shape[0]
    out = np.zeros((m, n_phases), dtype=np.float64)
    for p in range(n_phases):
        mask = phase_ids == p
        if mask.any():
            out[:, p] = np.abs(scores[:, mask]).mean(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        row_max = out.max(axis=1)
        row_med = np.median(out, axis=1)
        sel = np.where(row_max > 1e-12, (row_max - row_med) / row_max, 0.0)
    return out, float(np.mean(sel))


def run(config: dict[str, Any]) -> tuple[list[M10Row], dict[str, Any]]:
    from sklearn.metrics import adjusted_mutual_info_score

    model_name = str(config["model"]["name"])
    layer = int(config["model"]["layer"])
    site = str(config["model"].get("site", "resid_post"))
    seed = int(config["seed"])
    gammas = [float(g) for g in config["facet"]["gamma_candidates"]]
    m = int(config["decomposition"]["m"])

    model, tokenizer = load_gpt2(model_name)
    token_ids, phase_ids, text_tokens = build_corpus(tokenizer, CONVERSATIONS)
    H = _extract_residual_per_token(model, token_ids, layer=layer, site=site)
    n_phases = len(PHASES)

    facet, gamma_used = _facet_components(H, m, gammas, seed)
    pca = _pca_components(H, m)

    rows: list[M10Row] = []
    scores_by_method: dict[str, np.ndarray] = {}
    phase_mean_by_method: dict[str, np.ndarray] = {}
    for name, comps, hp in (
        ("facet_asq", facet, gamma_used if gamma_used is not None else float("nan")),
        ("pca", pca, float("nan")),
    ):
        scores = comps.T @ H  # (m, n_tokens)
        scores_by_method[name] = scores
        phase_means, selectivity = _phase_selectivity(scores, phase_ids, n_phases)
        phase_mean_by_method[name] = phase_means
        assignment = np.argmax(np.abs(scores), axis=0)
        ami = float(adjusted_mutual_info_score(phase_ids, assignment))
        rows.append(
            M10Row(
                method=name,
                adjusted_mutual_info=ami,
                reconstruction_r2=_reconstruction_r2(H, comps),
                mean_phase_selectivity=selectivity,
                hyperparam=hp,
                m=m,
                n_tokens=int(H.shape[1]),
            )
        )

    extras = {
        "H": H,
        "token_ids": token_ids,
        "text_tokens": text_tokens,
        "phase_ids": phase_ids,
        "phase_mean_by_method": phase_mean_by_method,
        "scores_by_method": scores_by_method,
        "layer": layer,
        "site": site,
        "n_tokens": int(H.shape[1]),
    }
    return rows, extras


def write_csv(rows: list[M10Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dc.fields(M10Row)])
        writer.writeheader()
        for r in rows:
            writer.writerow(dc.asdict(r))


def write_hero_figure(rows: list[M10Row], extras: dict[str, Any], path: Path) -> None:
    """Hero: component-by-phase heatmap for PCA and FACET side-by-side."""
    from _style import apply_style, full_spines

    apply_style()
    import matplotlib.pyplot as plt
    import seaborn as sns

    phase_mean = extras["phase_mean_by_method"]
    ami_by_method = {r.method: r.adjusted_mutual_info for r in rows}
    sel_by_method = {r.method: r.mean_phase_selectivity for r in rows}

    def _standardise(mat: np.ndarray) -> np.ndarray:
        row_mean = mat.mean(axis=1, keepdims=True)
        row_std = mat.std(axis=1, keepdims=True)
        row_std = np.where(row_std < 1e-12, 1.0, row_std)
        return (mat - row_mean) / row_std

    # Order rows so phase-selective components bubble up; use the FACET order
    # for both panels so the rows are comparable.
    facet_mat = _standardise(phase_mean["facet_asq"])
    pca_mat = _standardise(phase_mean["pca"])
    facet_order = np.argsort(-facet_mat.max(axis=1))
    pca_order = np.argsort(-pca_mat.max(axis=1))
    facet_mat = facet_mat[facet_order]
    pca_mat = pca_mat[pca_order]

    vmax = float(max(np.abs(facet_mat).max(), np.abs(pca_mat).max()))

    fig, (ax_pca, ax_facet) = plt.subplots(
        1,
        2,
        figsize=(13.6, 7.2),
        constrained_layout=True,
        sharey=True,
    )

    for ax, mat, method, panel_tag in (
        (ax_pca, pca_mat, "pca", "(a) PCA"),
        (ax_facet, facet_mat, "facet_asq", "(b) FACET A-SQ"),
    ):
        sns.heatmap(
            mat,
            ax=ax,
            cmap="RdBu_r",
            center=0.0,
            vmin=-vmax,
            vmax=vmax,
            cbar=True,
            cbar_kws={"label": "mean $|$score$|$ (standardised per component)"},
            xticklabels=PHASES,
            yticklabels=[f"c{i + 1}" for i in range(mat.shape[0])],
            linewidths=0.0,
        )
        ax.set_xlabel("conversation phase")
        ax.set_ylabel("component (ranked by peak phase magnitude)")
        ax.set_xticklabels(PHASES, rotation=25, ha="right")
        full_spines(ax)
        ax.text(
            0.01,
            1.02,
            panel_tag,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=14,
            weight="bold",
        )
        ax.text(
            0.99,
            1.02,
            f"AMI = {ami_by_method[method]:.3f}   phase selectivity = {sel_by_method[method]:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=11,
        )

    # Thin coloured strip along the top of each panel labelling the phase palette.
    for ax in (ax_pca, ax_facet):
        for i, phase in enumerate(PHASES):
            ax.add_patch(
                plt.Rectangle(
                    (i, -0.4),
                    1,
                    0.3,
                    color=PHASE_PALETTE[phase],
                    transform=ax.transData,
                    clip_on=False,
                    zorder=5,
                )
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--hero", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    rows, extras = run(cfg)
    out = Path(args.output_dir)
    write_csv(rows, out / cfg["output"]["csv"])
    if not args.no_figure and args.hero:
        fig_name = Path(cfg["output"]["figure"]).stem + "_hero.png"
        write_hero_figure(rows, extras, out / fig_name)
    print(
        f"wrote {len(rows)} rows; {extras['n_tokens']} tokens across {len(CONVERSATIONS)} conversations"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
