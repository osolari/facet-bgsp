"""Solver-dynamics tests pinning the empirical content of the BGSP paper theorems
A6 (sufficient-ascent), A7 (relative-error / vanishing step + residual),
A8 (whole-sequence convergence via bounded total variation), and A9 (finite
support/sign identification under a strict margin).

The theorems themselves are target statements; these tests pin what the
analyzed solver (``SafeguardedRiemannianAscent``) is *empirically* required to
exhibit on planted-spike instances. A failure here falsifies the solver as
analyzed, not the theorem.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from facet.generators import planted_spike
from facet.geometry import SphereGeometry
from facet.objectives import SquaredThresholdObjective
from facet.operators import DenseOperator
from facet.solvers import SafeguardedConfig, SafeguardedRiemannianAscent
from facet.supports import soft_threshold


def _build(seed: int, *, p: int = 8, n: int = 20, k: int = 4) -> tuple:
    """Planted-spike instance + objective + sphere + probe for the BGSP paper diagnostics.

    Uses m=1 so the sphere oracle is appropriate; A6-A9 extend to Stiefel
    with the same probe shape.
    """
    instance = planted_spike(p=p, n=n, k=k, m=1, singular_values=(5.0,), noise_std=0.0, seed=seed)
    operator = DenseOperator(instance.data)
    # Choose gamma strictly between the planted-support correlation values and
    # the off-support ones at the planted score to guarantee a strict margin.
    c_star = instance.data.T @ instance.scores[:, 0]
    on_support = np.abs(c_star[instance.supports[0]])
    off_support = np.abs(np.delete(c_star, instance.supports[0]))
    gamma = float(0.5 * (on_support.min() + off_support.max()))
    assert on_support.min() > off_support.max(), "fixture lacks a clean margin"
    objective = SquaredThresholdObjective(operator, gamma=gamma)
    geometry = SphereGeometry(radius=1.0)

    def probe(x: np.ndarray) -> dict:
        c = operator.rmatvec(x)
        s = soft_threshold(c, gamma)
        active = s != 0.0
        margin = float(np.abs(c[active]).min() - gamma) if active.any() else 0.0
        return {"support_size": int(active.sum()), "margin": margin}

    return instance, objective, geometry, gamma, probe


def _perturbed_init(rng: np.random.Generator, x_star: np.ndarray, angle: float) -> np.ndarray:
    """Return a unit vector within ``angle`` radians of x_star."""
    perp = rng.normal(size=x_star.shape)
    perp -= (perp @ x_star) * x_star
    perp /= np.linalg.norm(perp)
    x = np.cos(angle) * x_star + np.sin(angle) * perp
    return x / np.linalg.norm(x)


# ---------------------------------------------------------------------------
# A6, Monotone ascent (sufficient-ascent lemma, empirical)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 7])
def test_a6_monotone_ascent_on_planted_spike(seed: int) -> None:
    instance, objective, geometry, _, probe = _build(seed)
    rng = np.random.default_rng(seed + 100)
    init = _perturbed_init(rng, instance.scores[:, 0], angle=np.pi / 6)
    solver = SafeguardedRiemannianAscent(geometry, SafeguardedConfig(max_iterations=200))
    _, trace = solver.solve(objective, init, probe=probe)

    values = [r.objective for r in trace.records]
    assert len(values) >= 2
    # Armijo guarantees strict non-decrease at every accepted iterate.
    for prev, curr in itertools.pairwise(values):
        assert curr >= prev - 1e-12, (prev, curr)


# ---------------------------------------------------------------------------
# A7, Vanishing step norm and tangent residual
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 3, 11])
def test_a7_step_and_residual_vanish(seed: int) -> None:
    instance, objective, geometry, _, probe = _build(seed)
    rng = np.random.default_rng(seed + 50)
    init = _perturbed_init(rng, instance.scores[:, 0], angle=np.pi / 8)
    solver = SafeguardedRiemannianAscent(geometry, SafeguardedConfig(max_iterations=500))
    _, trace = solver.solve(objective, init, probe=probe)
    tail = trace.records[-5:]
    assert len(tail) >= 2
    # The tail step norms and tangent residuals are both near zero: this is the
    # empirical signature of the relative-error / sufficient-ascent conclusion.
    assert max(r.step_norm for r in tail) < 1e-4
    assert max(float(r.residual or 0.0) for r in tail) < 1e-3


# ---------------------------------------------------------------------------
# A8, Bounded total iterate variation (whole-sequence Cauchy, empirical)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 4, 9])
def test_a8_bounded_total_variation(seed: int) -> None:
    instance, objective, geometry, _, probe = _build(seed)
    rng = np.random.default_rng(seed + 200)
    init = _perturbed_init(rng, instance.scores[:, 0], angle=np.pi / 7)
    solver = SafeguardedRiemannianAscent(geometry, SafeguardedConfig(max_iterations=500))
    _, trace = solver.solve(objective, init, probe=probe)
    # Sum of step norms (total variation) must be finite and in fact bounded by
    # a modest multiple of the sphere diameter on a correctly behaving run.
    total_variation = sum(r.step_norm for r in trace.records)
    assert np.isfinite(total_variation)
    assert total_variation < 10.0  # sphere diameter is 2.0; 10 is a loose but meaningful cap


# ---------------------------------------------------------------------------
# A9, Finite support (and sign) identification under a strict margin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 2, 5, 8, 13])
def test_a9_finite_support_identification_under_strict_margin(seed: int) -> None:
    instance, objective, geometry, gamma, probe = _build(seed)
    planted_support = set(int(i) for i in instance.supports[0])

    rng = np.random.default_rng(seed + 300)
    init = _perturbed_init(rng, instance.scores[:, 0], angle=np.pi / 10)
    solver = SafeguardedRiemannianAscent(geometry, SafeguardedConfig(max_iterations=400))
    x_final, trace = solver.solve(objective, init, probe=probe)

    # The last probe record must report a support matching the planted support
    # exactly, with a strictly positive margin (A9's strict-margin assumption).
    last = trace.records[-1]
    assert last.support_size is not None
    assert last.margin is not None and last.margin > 0.0

    c_final = instance.data.T @ x_final
    final_mask = (np.abs(c_final) - gamma) > 0
    assert set(np.flatnonzero(final_mask).tolist()) == planted_support

    # Finite identification: once the support stabilizes it stays stable.
    sizes = [r.support_size for r in trace.records if r.support_size is not None]
    stabilized_from = next(i for i in range(len(sizes)) if all(s == sizes[-1] for s in sizes[i:]))
    assert stabilized_from < len(sizes) - 1, "support never stabilized before termination"


def test_a9_sign_identification_matches_planted_sign_pattern() -> None:
    instance, objective, geometry, _gamma, probe = _build(seed=0)
    rng = np.random.default_rng(999)
    init = _perturbed_init(rng, instance.scores[:, 0], angle=np.pi / 12)
    solver = SafeguardedRiemannianAscent(geometry, SafeguardedConfig(max_iterations=400))
    x_final, _ = solver.solve(objective, init, probe=probe)

    planted = instance.scores[:, 0]
    orientation = 1.0 if float(x_final @ planted) >= 0.0 else -1.0
    c_final = orientation * (instance.data.T @ x_final)
    c_star = instance.data.T @ planted
    support = instance.supports[0]
    # On the planted support the sign pattern of the correlation vector
    # matches the sign pattern at the planted score (modulo global sign).
    assert np.array_equal(np.sign(c_final[support]), np.sign(c_star[support]))
