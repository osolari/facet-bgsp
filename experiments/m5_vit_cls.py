"""M5, ViT CLS token decomposition: FACET vs PCA on class-conditional purity.

Generates ten synthetic image classes (solid colors, stripes, gradients,
disks, frames, see ``_vit.synthetic_classes``), runs a pretrained
ViT-base-patch16-224 to extract CLS tokens from the final transformer
block, and fits ``m=10`` components with FACET A-SQ and PCA on those
tokens. We score each method's components by how well their dominant-
component clustering of images matches the known class labels, measured by
adjusted mutual information (AMI). We also report silhouette score of
images in the component-score feature space.

See plan-2026-10-13T1908Z §M5.
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

from _vit import extract_vit_cls, generate_image_set, synthetic_classes  # noqa: E402
from t3_llm_activations import _pca_components, _reconstruction_r2  # noqa: E402

from facet.papers.block_sparse import BlockSparseEstimator  # noqa: E402


@dc.dataclass(frozen=True)
class VitRow:
    method: str
    reconstruction_r2: float
    adjusted_mutual_info: float
    silhouette: float
    hyperparam: float
    n_images: int
    m: int


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


def _dominant_component_assignment(components: np.ndarray, H: np.ndarray) -> np.ndarray:
    scores = components.T @ H  # (m, n_images)
    return np.argmax(np.abs(scores), axis=0)


def _measure(
    name: str,
    components: np.ndarray,
    H: np.ndarray,
    labels: np.ndarray,
    hp: float,
) -> VitRow:
    from sklearn.metrics import adjusted_mutual_info_score, silhouette_score

    assignment = _dominant_component_assignment(components, H)
    try:
        ami = float(adjusted_mutual_info_score(labels, assignment))
    except Exception:
        ami = float("nan")

    scores = (components.T @ H).T  # (n_images, m) feature matrix for silhouette
    # Silhouette needs at least 2 distinct cluster labels and >= n+1 samples.
    if len(np.unique(labels)) > 1 and scores.shape[0] > len(np.unique(labels)):
        try:
            sil = float(silhouette_score(scores, labels))
        except Exception:
            sil = float("nan")
    else:
        sil = float("nan")

    return VitRow(
        method=name,
        reconstruction_r2=_reconstruction_r2(H, components),
        adjusted_mutual_info=ami,
        silhouette=sil,
        hyperparam=hp,
        n_images=int(H.shape[1]),
        m=int(components.shape[1]),
    )


def run(config: dict[str, Any]) -> list[VitRow]:
    from transformers import AutoImageProcessor, AutoModel

    model_name = str(config["model"]["name"])
    per_class = int(config["data"]["per_class"])
    seed = int(config["seed"])
    gammas = [float(g) for g in config["facet"]["gamma_candidates"]]

    classes = synthetic_classes()
    images, labels = generate_image_set(classes, per_class=per_class, size=224, seed=seed)

    processor = AutoImageProcessor.from_pretrained(model_name)
    vit = AutoModel.from_pretrained(model_name)
    vit.eval()

    H = extract_vit_cls(vit, processor, images, layer=None)  # (d_model, n_images)
    m = int(config["decomposition"]["m"])

    facet, gamma_used = _facet_components(H, m=m, gammas=gammas, seed=seed)
    pca = _pca_components(H, m)

    rows = [
        _measure(
            "facet_asq", facet, H, labels, gamma_used if gamma_used is not None else float("nan")
        ),
        _measure("pca", pca, H, labels, float("nan")),
    ]
    return rows


def write_csv(rows: list[VitRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dc.fields(VitRow)])
        writer.writeheader()
        for r in rows:
            writer.writerow(dc.asdict(r))


def write_figure(rows: list[VitRow], path: Path) -> None:
    from _style import METHOD_COLORS, apply_style, full_spines

    apply_style()
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    palette = {"facet_asq": METHOD_COLORS["facet_asq"], "pca": METHOD_COLORS["pca"]}
    labels = {"facet_asq": "FACET A-SQ", "pca": "PCA"}
    order = ["pca", "facet_asq"]

    df = pd.DataFrame([dc.asdict(r) for r in rows])
    df["label"] = df["method"].map(labels)
    label_order = [labels[m] for m in order]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8), constrained_layout=True)
    panels = [
        ("reconstruction_r2", r"Reconstruction $R^2$"),
        ("adjusted_mutual_info", r"AMI(dominant-component, class label)"),
        ("silhouette", r"Silhouette score in component space"),
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
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=3, fontsize=10)
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
