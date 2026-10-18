"""Jeffers (1967) Pitprops correlation matrix, canonical sparse-PCA benchmark.

Reproduced from Jeffers (1967) and reused by Zou et al. (2006) and Journée et
al. (2010) in Table-form comparisons. 13 variables measured on pine props:

    1  topdiam    top diameter
    2  length     prop length
    3  moist      moisture content
    4  testsg     specific gravity of the test prop
    5  ovensg     oven-dry specific gravity
    6  ringtop    rings per inch at the top
    7  ringbut    rings per inch at the butt
    8  bowmax     maximum bow
    9  bowdist    distance of maximum bow
    10 whorls     number of knot whorls
    11 clear      length of clear prop from the top
    12 knots      average knot diameter
    13 diaknot    maximum knot diameter

The matrix is positive definite by construction. The top-6 PCs explain
~87% of the variance in the full data.
"""

from __future__ import annotations

import numpy as np

_PITPROPS_UPPER = """
1.000  0.954  0.364  0.342 -0.129  0.313  0.496  0.424  0.592  0.545  0.084 -0.019  0.134
0.954  1.000  0.297  0.284 -0.118  0.291  0.503  0.419  0.648  0.569  0.076 -0.036  0.144
0.364  0.297  1.000  0.882 -0.148  0.153 -0.029 -0.054  0.125 -0.081  0.162  0.220  0.126
0.342  0.284  0.882  1.000  0.220  0.381  0.174 -0.059  0.137 -0.014  0.097  0.169  0.015
-0.129 -0.118 -0.148  0.220  1.000  0.364  0.296  0.004 -0.039  0.037 -0.091 -0.145 -0.208
0.313  0.291  0.153  0.381  0.364  1.000  0.813  0.090  0.211  0.274 -0.036  0.024 -0.329
0.496  0.503 -0.029  0.174  0.296  0.813  1.000  0.372  0.465  0.679 -0.113 -0.232 -0.424
0.424  0.419 -0.054 -0.059  0.004  0.090  0.372  1.000  0.482  0.557  0.061 -0.357 -0.202
0.592  0.648  0.125  0.137 -0.039  0.211  0.465  0.482  1.000  0.526  0.085 -0.127 -0.076
0.545  0.569 -0.081 -0.014  0.037  0.274  0.679  0.557  0.526  1.000 -0.319 -0.368 -0.291
0.084  0.076  0.162  0.097 -0.091 -0.036 -0.113  0.061  0.085 -0.319  1.000  0.029  0.007
-0.019 -0.036  0.220  0.169 -0.145  0.024 -0.232 -0.357 -0.127 -0.368  0.029  1.000  0.184
0.134  0.144  0.126  0.015 -0.208 -0.329 -0.424 -0.202 -0.076 -0.291  0.007  0.184  1.000
""".strip()


def pitprops_correlation() -> np.ndarray:
    """Return the 13x13 Jeffers Pitprops correlation matrix (symmetric PD)."""
    rows = []
    for line in _PITPROPS_UPPER.splitlines():
        rows.append([float(x) for x in line.split()])
    R = np.array(rows, dtype=float)
    # Enforce exact symmetry (minor float rounding may leave a 1e-16 asymmetry).
    return 0.5 * (R + R.T)


def pitprops_data_matrix(seed: int = 0, n_samples: int = 180) -> np.ndarray:
    """Sample a Gaussian data matrix with the Pitprops correlation.

    The classical sPCA literature uses the correlation matrix directly, but
    FACET's reference pipeline wants a (p, n) data matrix. We draw ``n_samples``
    i.i.d. standard-normal rows and transform them by the Cholesky factor of R
    so the sample covariance converges to Pitprops. ``n_samples=180`` is the
    original Jeffers sample size.
    """
    R = pitprops_correlation()
    L = np.linalg.cholesky(R)
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n_samples, R.shape[0]))
    return (Z @ L.T).T  # (p, n_samples)


def pitprops_variable_names() -> list[str]:
    return [
        "topdiam",
        "length",
        "moist",
        "testsg",
        "ovensg",
        "ringtop",
        "ringbut",
        "bowmax",
        "bowdist",
        "whorls",
        "clear",
        "knots",
        "diaknot",
    ]
