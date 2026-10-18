"""Shared plot style for every the BGSP paper driver.

Enforces:
  - seaborn ``ticks`` theme with ``colorblind`` palette,
  - all four spines visible on every axis (full-spine plots),
  - larger default fonts suitable for a two-column paper figure,
  - a single method-color map so FACET / PCA / native / random / theory
    always render in the same color across Q-, T-, and A- figures.
"""

from __future__ import annotations

from typing import Any

METHOD_COLORS: dict[str, str] = {
    "facet_asq": "#d62728",  # red
    "facet_asum": "#ff7f0e",  # orange
    "pca": "#1f77b4",  # blue
    "native_neuron": "#7f7f7f",  # mid-gray
    "random_orthogonal": "#bcbcbc",  # light-gray
    "theory": "#000000",  # black (reference lines)
    "observed": "#2ca02c",  # green (data overlays)
    "sum": "#1f77b4",  # Q1 F_sum curve
    "sq": "#d62728",  # Q1 F_sq curve
}

METHOD_LABELS: dict[str, str] = {
    "facet_asq": "FACET A-SQ",
    "facet_asum": "FACET A-SUM",
    "pca": "PCA",
    "native_neuron": "native neurons",
    "random_orthogonal": "random orth.",
    "theory": "theory",
    "observed": "observed",
    "sum": r"$F_{\mathrm{sum}}$",
    "sq": r"$F_{\mathrm{sq}}$",
}


def apply_style() -> None:
    """Apply the shared seaborn theme + rcParams. Call once at driver start."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(
        context="notebook",
        style="ticks",
        palette="colorblind",
        font_scale=1.25,
        rc={
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.linewidth": 1.0,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "axes.labelweight": "normal",
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "legend.frameon": True,
            "legend.framealpha": 0.85,
            "legend.edgecolor": "0.5",
            "figure.titlesize": 14,
            "figure.titleweight": "bold",
            "lines.linewidth": 1.8,
            "lines.markersize": 7,
            "grid.alpha": 0.35,
            "grid.linewidth": 0.6,
        },
    )
    # ensure matplotlib honours the new rc even if a prior driver mutated it
    plt.rcParams.update(plt.rcParams)


def full_spines(ax: Any) -> None:
    """Explicitly re-enable all four spines (idempotent helper for subplots)."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.0)
