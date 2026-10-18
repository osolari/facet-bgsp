"""A3 local-rate tests on fixed-support dynamics (the BGSP paper).

After the active support stabilizes (A9), the solver's dynamics reduce to a
smooth sphere-constrained spectral iteration on the restricted operator
``A_S``. These tests pin two empirical facts the local-rate theorem must be
consistent with:

- the post-identification step norms decay geometrically (linear convergence);
- for the historical full-polar iteration on a controlled restricted problem
  with known top-two eigenvalues ``lambda_1 > lambda_2``, the observed
  per-step shrinkage of the deviation to the planted direction tracks the
  classical ``(lambda_2 / lambda_1)^2`` rate of sphere power iteration.

The analyzed safeguarded ascent is also required to converge linearly (rate
strictly below 1) but its exact rate constant is Armijo-dependent and is not
required to match a closed-form prediction.
"""

from __future__ import annotations

import numpy as np
import pytest

from facet.generators import planted_spike
from facet.geometry import SphereGeometry
from facet.objectives import SquaredThresholdObjective
from facet.operators import DenseOperator
from facet.solvers import (
    FullPolarSolver,
    PolarSolverConfig,
    SafeguardedConfig,
    SafeguardedRiemannianAscent,
)
from facet.supports import soft_threshold


def _gamma_with_strict_margin(instance) -> float:
    c = instance.data.T @ instance.scores[:, 0]
    on = np.abs(c[instance.supports[0]])
    off = np.abs(np.delete(c, instance.supports[0]))
    assert on.min() > off.max(), "fixture lacks a clean margin"
    return float(0.5 * (on.min() + off.max()))


def _perturbed_init(rng: np.random.Generator, x_star: np.ndarray, angle: float) -> np.ndarray:
    perp = rng.normal(size=x_star.shape)
    perp -= (perp @ x_star) * x_star
    perp /= np.linalg.norm(perp)
    x = np.cos(angle) * x_star + np.sin(angle) * perp
    return x / np.linalg.norm(x)


# ---------------------------------------------------------------------------
# Local linear convergence of the safeguarded solver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 2, 5, 11])
def test_a3_safeguarded_shows_linear_post_identification_decay(seed: int) -> None:
    """``log(step_norm)`` falls (strictly on average) across the tail, the
    empirical signature of linear convergence to a stationary point."""
    inst = planted_spike(p=10, n=20, k=4, m=1, singular_values=(5.0,), seed=seed)
    gamma = _gamma_with_strict_margin(inst)
    op = DenseOperator(inst.data)
    obj = SquaredThresholdObjective(op, gamma=gamma)

    def probe(x: np.ndarray) -> dict:
        c = op.rmatvec(x)
        s = soft_threshold(c, gamma)
        active = s != 0.0
        return {
            "support_size": int(active.sum()),
            "margin": float(np.abs(c[active]).min() - gamma) if active.any() else 0.0,
        }

    rng = np.random.default_rng(seed + 1000)
    init = _perturbed_init(rng, inst.scores[:, 0], angle=0.4)
    solver = SafeguardedRiemannianAscent(
        SphereGeometry(), SafeguardedConfig(max_iterations=400, tolerance=1e-10)
    )
    _, trace = solver.solve(obj, init, probe=probe)

    sizes = [r.support_size for r in trace.records]
    k = inst.supports[0].size
    first_stable = next(i for i, s in enumerate(sizes) if all(x == k for x in sizes[i:]))
    tail = [r.step_norm for r in trace.records[first_stable:] if r.step_norm > 0]
    assert len(tail) >= 5, f"only {len(tail)} post-identification steps"

    # Linear convergence ⇒ log(step) decreases at least linearly in t.
    log_tail = np.log(np.array(tail))
    t = np.arange(len(log_tail))
    slope = float(np.polyfit(t, log_tail, deg=1)[0])
    assert slope < -0.05, f"log step-norm slope {slope:.4f} is not negative enough"


@pytest.mark.parametrize("seed", [0, 1, 3, 7])
def test_a3_safeguarded_asymptotic_ratio_strictly_below_one(seed: int) -> None:
    inst = planted_spike(p=10, n=20, k=4, m=1, singular_values=(5.0,), seed=seed)
    gamma = _gamma_with_strict_margin(inst)
    op = DenseOperator(inst.data)
    obj = SquaredThresholdObjective(op, gamma=gamma)

    def probe(x: np.ndarray) -> dict:
        c = op.rmatvec(x)
        s = soft_threshold(c, gamma)
        active = s != 0.0
        return {
            "support_size": int(active.sum()),
            "margin": float(np.abs(c[active]).min() - gamma) if active.any() else 0.0,
        }

    rng = np.random.default_rng(seed + 2000)
    init = _perturbed_init(rng, inst.scores[:, 0], angle=0.5)
    _, trace = SafeguardedRiemannianAscent(
        SphereGeometry(), SafeguardedConfig(max_iterations=400, tolerance=1e-10)
    ).solve(obj, init, probe=probe)

    sizes = [r.support_size for r in trace.records]
    k = inst.supports[0].size
    first_stable = next(i for i, s in enumerate(sizes) if all(x == k for x in sizes[i:]))
    tail = [r.step_norm for r in trace.records[first_stable:] if r.step_norm > 1e-13]
    assert len(tail) >= 6
    # Drop the first few iterations of the tail (transient) before averaging.
    mid = tail[len(tail) // 4 : 3 * len(tail) // 4]
    ratios = [mid[i + 1] / mid[i] for i in range(len(mid) - 1)]
    med = float(np.median(ratios))
    assert 0.0 < med < 0.99, f"median asymptotic ratio {med:.4f} outside (0, 0.99)"


# ---------------------------------------------------------------------------
# Full-polar iteration on a controlled restricted problem
# ---------------------------------------------------------------------------


def _controlled_restricted_instance(
    *, p: int = 12, k: int = 5, lam1: float = 4.0, lam2: float = 1.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Build an instance where ``A_S A_S^T`` has prescribed top-two eigenvalues
    AND the planted top eigenvector ``x_star`` induces a uniform-magnitude
    correlation ``c_j = sigma_1 / sqrt(k)`` on the planted support.

    Construction: ``A_S = sigma_1 * x_star * 1_k^T / sqrt(k) + sigma_2 * w * r^T``
    with ``w _|_ x_star`` and ``1_k^T r = 0``. Then
    ``A_S A_S^T = sigma_1^2 * x_star x_star^T + sigma_2^2 * w w^T`` (clean
    rank-2), and ``A_S^T x_star = (sigma_1 / sqrt(k)) * 1_k`` (uniform).
    """
    rng = np.random.default_rng(seed)
    sigma_1 = float(np.sqrt(lam1))
    sigma_2 = float(np.sqrt(lam2))

    # Pick x_star and an orthogonal w in R^p.
    base = rng.normal(size=p)
    x_star = base / np.linalg.norm(base)
    perp = rng.normal(size=p)
    perp -= (perp @ x_star) * x_star
    w = perp / np.linalg.norm(perp)

    # Pick r in R^k with 1_k^T r = 0 and ||r|| = 1.
    r_raw = rng.normal(size=k)
    r_raw -= r_raw.mean()
    r = r_raw / np.linalg.norm(r_raw)

    ones = np.ones(k)
    A_S = sigma_1 * np.outer(x_star, ones) / np.sqrt(k) + sigma_2 * np.outer(w, r)

    n = k + 15
    A = np.zeros((p, n))
    A[:, :k] = A_S
    A[:, k:] = 0.01 * rng.normal(size=(p, n - k))  # negligible: keeps support = {0..k-1}
    support = np.arange(k)
    return A, x_star, support, lam1, lam2


def test_a3_full_polar_rate_tracks_restricted_eigengap() -> None:
    """Full-polar sphere iteration on A-SQ with a strict margin: once the
    support is identified the deviation ``sin(angle_t)`` shrinks like
    ``(lambda_2 / lambda_1)^t`` (classical power-iteration rate on the
    restricted operator). We check the observed geometric rate matches the
    prediction to within a loose multiplicative tolerance."""
    A, x_star, support, lam1, lam2 = _controlled_restricted_instance(
        p=12, k=5, lam1=4.0, lam2=1.2, seed=42
    )
    op = DenseOperator(A)
    # gamma well below the smallest on-support correlation and well above the
    # largest off-support correlation.
    c_star = A.T @ x_star
    on_min = float(np.abs(c_star[support]).min())
    off_max = float(np.abs(np.delete(c_star, support)).max())
    assert on_min > off_max
    gamma = 0.5 * (on_min + off_max)
    obj = SquaredThresholdObjective(op, gamma=gamma)

    rng = np.random.default_rng(7)
    init = _perturbed_init(rng, x_star, angle=0.4)
    _, trace = FullPolarSolver(
        SphereGeometry(), PolarSolverConfig(max_iterations=600, tolerance=1e-12)
    ).solve(obj, init)

    # Reconstruct the iterate sequence (full-polar's record is per-iteration;
    # read x via the induced correlation sign pattern + value identity isn't
    # straightforward, so recompute by replaying the oracle).
    xs = [init]
    for _ in range(len(trace.records)):
        g = obj.gradient(xs[-1])
        xs.append(SphereGeometry().linear_oracle(g))
    xs = xs[1:]  # drop init
    # Deviation ∝ sin(angle to x_star); |x_t · x_star| → 1 geometrically.
    overlaps = np.array([abs(float(x @ x_star)) for x in xs])
    deviations = np.sqrt(np.clip(1.0 - overlaps**2, 0.0, None))

    tail = deviations[deviations > 1e-10]
    if len(tail) < 5:
        pytest.skip("not enough tail samples for a rate estimate")
    log_dev = np.log(tail)
    slope = float(np.polyfit(np.arange(len(log_dev)), log_dev, deg=1)[0])
    observed_rate = float(np.exp(slope))
    predicted_rate = (lam2 / lam1) ** 2

    # The observed rate should be positive, strictly below 1, and within one
    # order of magnitude of the classical prediction.
    assert 0.0 < observed_rate < 0.99
    assert predicted_rate / 10 <= observed_rate <= 10 * predicted_rate, (
        observed_rate,
        predicted_rate,
    )
