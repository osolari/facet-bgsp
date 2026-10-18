"""Edge-case counterexamples for the BGSP paper: A4 (mask-before-normalize) and A5
(polar oracle under rank deficiency).

A4 shows that normalize-then-mask breaks ball feasibility and strictly reduces
the restricted objective; the correct order is mask-then-normalize (which the
``refit.mask_then_normalize`` primitive enforces).

A5 shows the Stiefel linear-oracle policies: ``canonical`` returns a feasible
Stiefel point even when the input matrix is rank-deficient, while ``raise``
rejects the input to flag an assumption violation.
"""

from __future__ import annotations

import numpy as np
import pytest

from facet.exceptions import RankDeficiencyError, ZeroDirectionError
from facet.geometry import StiefelGeometry
from facet.refit import mask_then_normalize


def _soft(c: np.ndarray, gamma: float) -> np.ndarray:
    return np.sign(c) * np.maximum(np.abs(c) - gamma, 0.0)


# ---------------------------------------------------------------------------
# A4, Mask-before-normalize is correct; normalize-then-mask is wrong.
# ---------------------------------------------------------------------------


def test_a4_normalize_then_mask_breaks_ball_constraint() -> None:
    """A vector normalized to the unit sphere, then masked, is no longer unit norm.

    Concrete counterexample: v is already a unit vector. Removing one nonzero
    coordinate strictly shrinks its L2 norm below 1, violating the loading-ball
    equality side used in the primal.
    """
    v = np.array([0.6, 0.8, 0.0])  # unit vector by construction
    assert np.isclose(np.linalg.norm(v), 1.0)
    mask = np.array([True, False, False])  # drop the 0.8 coordinate
    masked = np.where(mask, v, 0.0)
    assert np.linalg.norm(masked) == 0.6  # strictly below 1

    # Mask-then-normalize restores unit norm on the selected support.
    renorm = mask_then_normalize(v, mask)
    assert np.isclose(np.linalg.norm(renorm), 1.0)
    assert renorm[1] == 0.0 and renorm[2] == 0.0
    assert renorm[0] == 1.0  # single-coordinate support => sign-preserving indicator


def test_a4_mask_then_normalize_dominates_normalize_then_mask_on_quadratic_objective() -> None:
    """On the restricted quadratic c^T z / ||z|| objective (A-SUM optimum shape),
    mask-then-normalize is at least as large as normalize-then-mask with equality
    only when the dropped coordinates contribute zero correlation."""
    c = np.array([3.0, 4.0, 1.0, 0.0])
    mask = np.array([True, True, False, False])
    normalize_first = c / np.linalg.norm(c)
    normalized_then_masked = np.where(mask, normalize_first, 0.0)

    masked_then_normalized = mask_then_normalize(c, mask)

    def quad(z: np.ndarray) -> float:
        return float(c @ z)

    # The correctly ordered refit achieves a strictly larger inner product per unit norm.
    assert quad(masked_then_normalized) > quad(normalized_then_masked) + 1e-12


def test_a4_mask_then_normalize_rejects_zero_support() -> None:
    """A support that annihilates the vector raises a ZeroDirectionError rather
    than silently returning NaN; downstream code must handle the inactive case
    explicitly (ties into the zero/inactive-component policy A1)."""
    from facet.exceptions import ZeroDirectionError  # local re-import clarity

    with pytest.raises(ZeroDirectionError):
        mask_then_normalize(np.array([2.0, -1.0]), np.array([False, False]))


# ---------------------------------------------------------------------------
# A5, Stiefel polar oracle rank-deficiency policy
# ---------------------------------------------------------------------------


def test_a5_canonical_policy_returns_feasible_stiefel_point_for_rank_deficient_direction() -> None:
    """Direction has rank 1 but requires rank-2 output.

    Under the ``canonical`` policy the oracle still returns a Stiefel point
    (the SVD completes the frame with an orthogonal null-space direction),
    which is the exact behavior the analyzed solver depends on at degenerate
    corners.
    """
    direction = np.outer(np.array([1.0, 2.0, 2.0]), np.array([1.0, 0.0]))
    assert np.linalg.matrix_rank(direction) == 1

    stiefel = StiefelGeometry(3, 2, rank_policy="canonical")
    polar = stiefel.linear_oracle(direction)
    assert stiefel.is_feasible(polar)


def test_a5_raise_policy_rejects_rank_deficient_direction() -> None:
    """Under ``raise`` the oracle refuses to silently invent a direction; this
    is the mode theorems with strict-rank assumptions should be run under."""
    direction = np.outer(np.array([1.0, -1.0, 0.5]), np.array([1.0, 0.0]))
    stiefel = StiefelGeometry(3, 2, rank_policy="raise")
    with pytest.raises(RankDeficiencyError):
        stiefel.linear_oracle(direction)


def test_a5_zero_direction_raises_regardless_of_policy() -> None:
    """A zero direction has no polar factor; both policies must raise, since no
    convention can distinguish a chosen Stiefel point from an arbitrary one."""
    zeros = np.zeros((4, 2))
    for policy in ("canonical", "raise"):
        stiefel = StiefelGeometry(4, 2, rank_policy=policy)
        with pytest.raises(ZeroDirectionError):
            stiefel.linear_oracle(zeros)


def test_a5_canonical_polar_is_optimal_when_full_rank() -> None:
    """When the direction is full rank the canonical polar is the unique Stiefel
    maximizer of ``trace(X^T D)`` -- confirms policy does not alter the full-rank case."""
    rng = np.random.default_rng(2026)
    d = rng.normal(size=(6, 3))
    stiefel = StiefelGeometry(6, 3, rank_policy="canonical")
    polar = stiefel.linear_oracle(d)
    optimum = float(np.sum(polar * d))
    for _ in range(32):
        q, _ = np.linalg.qr(rng.normal(size=(6, 3)))
        assert optimum >= float(np.sum(q * d)) - 1e-10
