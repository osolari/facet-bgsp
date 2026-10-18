"""Primal/reduced identity and aggregation-nonequivalence tests for the BGSP paper.

Pins theorems A1 (A-SUM primal/reduced equivalence), A2 (A-SQ quadratic lift),
and A3 (sum-of-norms vs sum-of-squared-norms aggregation nonequivalence at m >= 2).
"""

from __future__ import annotations

import numpy as np
import pytest

from facet.geometry import StiefelGeometry
from facet.objectives import SquaredThresholdObjective, SumNormThresholdObjective
from facet.operators import DenseOperator
from facet.supports import soft_threshold


def _sum_norm_primal_value(a: np.ndarray, x: np.ndarray, gamma: float, mu: float = 1.0) -> float:
    """Closed-form optimum of max_{||z||_2 <= 1} mu * x^T A z - gamma * ||z||_1.

    When S_gamma(c) is zero, the optimum is zero. Otherwise the optimum equals
    ||S_gamma(c)||_2 attained at z* = S_gamma(c) / ||S_gamma(c)||_2.
    """
    c = mu * a.T @ x
    s = soft_threshold(c, gamma)
    norm = float(np.linalg.norm(s))
    if norm == 0.0:
        return 0.0
    z_star = s / norm
    return float(c @ z_star - gamma * np.linalg.norm(z_star, 1))


def _random_sphere(rng: np.random.Generator, dim: int) -> np.ndarray:
    v = rng.normal(size=dim)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# A1, A-SUM primal/reduced equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 7, 13, 42])
@pytest.mark.parametrize("gamma", [0.0, 0.1, 0.5, 1.5])
def test_a1_sum_norm_reduced_equals_primal_closed_form(seed: int, gamma: float) -> None:
    rng = np.random.default_rng(seed)
    p, n = 6, 11
    a = rng.normal(size=(p, n))
    x = _random_sphere(rng, p)

    reduced = SumNormThresholdObjective(DenseOperator(a), gamma=gamma).value(x)
    primal = _sum_norm_primal_value(a, x, gamma=gamma)
    assert np.isclose(reduced, primal, atol=1e-12)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_a1_sum_norm_closed_form_is_primal_optimum(seed: int) -> None:
    """Confirm z* dominates random feasible z under ||z||_2 <= 1 within noise."""
    rng = np.random.default_rng(seed)
    p, n = 5, 9
    gamma = 0.3
    a = rng.normal(size=(p, n))
    x = _random_sphere(rng, p)
    objective = SumNormThresholdObjective(DenseOperator(a), gamma=gamma)
    optimum = objective.value(x)

    c = a.T @ x
    for _ in range(100):
        z = rng.normal(size=n)
        nz = np.linalg.norm(z)
        if nz > 1.0:
            z = z / nz  # project onto the ball interior/boundary
        candidate = float(c @ z - gamma * np.linalg.norm(z, 1))
        assert candidate <= optimum + 1e-10


def test_a1_sum_norm_zero_support_returns_zero() -> None:
    """When soft-threshold kills every coordinate, the primal/reduced value is 0."""
    a = np.eye(3)
    x = np.array([1.0, 0.0, 0.0])
    objective = SumNormThresholdObjective(DenseOperator(a), gamma=10.0)
    assert objective.value(x) == 0.0
    assert np.allclose(objective.loading(x), 0.0)


# ---------------------------------------------------------------------------
# A2, A-SQ quadratic lift and gradient
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("gamma", [0.0, 0.2, 1.0])
def test_a2_squared_threshold_quadratic_lift(seed: int, gamma: float) -> None:
    """F_sq(x) = ||S_gamma(c)||_2^2 = max_z 2 c^T z - z^T z - 2 gamma ||z||_1."""
    rng = np.random.default_rng(seed)
    p, n = 4, 7
    a = rng.normal(size=(p, n))
    x = _random_sphere(rng, p)
    objective = SquaredThresholdObjective(DenseOperator(a), gamma=gamma)

    c = a.T @ x
    s = soft_threshold(c, gamma)
    lift_at_optimum = 2.0 * (c @ s) - float(s @ s) - 2.0 * gamma * float(np.linalg.norm(s, 1))

    assert np.isclose(objective.value(x), float(s @ s), atol=1e-12)
    assert np.isclose(lift_at_optimum, float(s @ s), atol=1e-12)


@pytest.mark.parametrize("seed", [11, 22, 33])
def test_a2_gradient_matches_finite_difference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    p, n = 5, 9
    gamma = 0.25
    a = rng.normal(size=(p, n))
    x = _random_sphere(rng, p)
    # Choose a direction that keeps us far from thresholding ties.
    direction = _random_sphere(rng, p)
    objective = SquaredThresholdObjective(DenseOperator(a), gamma=gamma)

    eps = 1e-6
    fd = (objective.value(x + eps * direction) - objective.value(x - eps * direction)) / (2 * eps)
    analytic = float(objective.gradient(x) @ direction)
    assert np.isclose(fd, analytic, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# A3, Aggregation nonequivalence: F_sum and F_sq rank orthonormal frames differently
# ---------------------------------------------------------------------------


def _sum_block(a: np.ndarray, x: np.ndarray, gammas: tuple[float, ...]) -> float:
    """Row-wise block A-SUM reduced objective: sum_j ||S_gamma_j(A^T x_j)||_2."""
    total = 0.0
    for j, gamma in enumerate(gammas):
        total += float(np.linalg.norm(soft_threshold(a.T @ x[:, j], gamma)))
    return total


def _sq_block(a: np.ndarray, x: np.ndarray, gammas: tuple[float, ...]) -> float:
    """Row-wise block A-SQ reduced objective: sum_j ||S_gamma_j(A^T x_j)||_2^2."""
    total = 0.0
    for j, gamma in enumerate(gammas):
        s = soft_threshold(a.T @ x[:, j], gamma)
        total += float(s @ s)
    return total


def test_a3_sum_vs_squared_aggregation_rank_two_candidates_differently() -> None:
    """A-SUM and A-SQ can reverse the ordering of two orthonormal frames.

    This is the m=2 counterexample underlying Proposition A3: the two block
    objectives are not equivalent, so treating them interchangeable is unsound.
    A deterministic (A, gamma, X_sum, X_sq) triple is checked below; the
    construction was found by a grid search over 2x2 rotation angles.
    """
    a = np.array(
        [
            [
                1.42993853,
                -0.15647532,
                -0.67375915,
                -0.6390601,
                -0.06136133,
                -0.39278492,
                2.28990995,
            ],
            [-0.71818115, 0.03260774, 0.0280499, 0.02827212, 0.05534586, -0.48156286, -0.5834075],
        ]
    )
    gammas = (0.6845716431159987, 0.5315438759871329)

    theta_sum = 0.5280  # near-arg-max of F_sum
    theta_sq = 1.2654  # near-arg-max of F_sq

    def rot(theta: float) -> np.ndarray:
        return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    frame_sum = rot(theta_sum)
    frame_sq = rot(theta_sq)

    stiefel = StiefelGeometry(2, 2)
    assert stiefel.is_feasible(frame_sum)
    assert stiefel.is_feasible(frame_sq)

    sum_at_sum = _sum_block(a, frame_sum, gammas)
    sum_at_sq = _sum_block(a, frame_sq, gammas)
    sq_at_sum = _sq_block(a, frame_sum, gammas)
    sq_at_sq = _sq_block(a, frame_sq, gammas)

    # Each frame wins decisively under its own objective.
    assert sum_at_sum - sum_at_sq > 0.05
    assert sq_at_sq - sq_at_sum > 0.5

    # The ordering is strictly reversed between the two objectives.
    assert (sum_at_sum > sum_at_sq) and (sq_at_sq > sq_at_sum)


def test_a3_identity_at_single_component() -> None:
    """For m=1 the two aggregations reduce to monotone transforms of each other:
    F_sq = F_sum^2. The nonequivalence is a strictly multi-component phenomenon.
    """
    rng = np.random.default_rng(0)
    p, n = 5, 11
    a = rng.normal(size=(p, n))
    x = _random_sphere(rng, p)
    gamma = 0.3

    sum_val = SumNormThresholdObjective(DenseOperator(a), gamma=gamma).value(x)
    sq_val = SquaredThresholdObjective(DenseOperator(a), gamma=gamma).value(x)
    assert np.isclose(sq_val, sum_val**2, atol=1e-12)
