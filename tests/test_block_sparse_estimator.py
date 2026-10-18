"""the BGSP paper end-to-end tests for ``facet.papers.block_sparse.BlockSparseEstimator``.

Pins the sklearn-style composition (objective + geometry + solver + refit)
on planted-spike instances. Separate from the lower-level primitive tests in
``tests/test_paper_a_*.py``: those pin individual theorems; these pin the
end-to-end paper pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from facet.generators import planted_spike
from facet.papers.block_sparse import BlockSparseEstimator
from facet.results import EstimatorStage
from facet.supports import soft_threshold


def _instance(seed: int = 0, *, p: int = 8, n: int = 20, k: int = 4):
    return planted_spike(p=p, n=n, k=k, m=1, singular_values=(5.0,), noise_std=0.0, seed=seed)


def _gamma_with_strict_margin(instance) -> float:
    c_star = instance.data.T @ instance.scores[:, 0]
    on_support = np.abs(c_star[instance.supports[0]])
    off_support = np.abs(np.delete(c_star, instance.supports[0]))
    return float(0.5 * (on_support.min() + off_support.max()))


# ---------------------------------------------------------------------------
# Shape / lineage invariants
# ---------------------------------------------------------------------------


def test_fit_returns_raw_and_refit_stages_with_lineage() -> None:
    instance = _instance(seed=0)
    gamma = _gamma_with_strict_margin(instance)
    est = BlockSparseEstimator(objective="squared", gamma=gamma, random_state=0)
    fit = est.fit(instance.data)

    assert fit.raw.stage is EstimatorStage.RAW
    assert fit.raw.parent_stage is None
    assert fit.raw.factors.shape == (instance.data.shape[0],)
    assert fit.refit is not None
    assert fit.refit.stage is EstimatorStage.REFIT
    assert fit.refit.parent_stage is EstimatorStage.RAW
    assert fit.support.size > 0


def test_support_matches_planted_support_under_strict_margin() -> None:
    instance = _instance(seed=3)
    gamma = _gamma_with_strict_margin(instance)
    fit = BlockSparseEstimator(objective="squared", gamma=gamma, random_state=3).fit(instance.data)
    assert set(fit.support.tolist()) == set(int(i) for i in instance.supports[0])


def test_refit_rank_one_recovers_leading_restricted_singular_vector() -> None:
    instance = _instance(seed=5)
    gamma = _gamma_with_strict_margin(instance)
    fit = BlockSparseEstimator(objective="squared", gamma=gamma, random_state=5).fit(instance.data)
    assert fit.refit is not None
    loading = fit.refit.factors
    score = fit.refit.metadata["score"]
    assert np.isclose(np.linalg.norm(score), 1.0)
    # Loading has the restricted-eigenvector shape: supported exactly on the
    # identified support and unit-norm on it.
    off_support = np.setdiff1d(np.arange(loading.size), fit.support)
    assert np.allclose(loading[off_support], 0.0)
    assert np.isclose(np.linalg.norm(loading), 1.0)


# ---------------------------------------------------------------------------
# Objective wiring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["sum_norm", "squared"])
def test_both_objectives_fit_and_identify_the_planted_support(name: str) -> None:
    instance = _instance(seed=1)
    gamma = _gamma_with_strict_margin(instance)
    fit = BlockSparseEstimator(objective=name, gamma=gamma, random_state=1).fit(instance.data)
    assert set(fit.support.tolist()) == set(int(i) for i in instance.supports[0])


def test_refit_objective_is_at_least_as_large_as_raw_on_restricted_frobenius() -> None:
    """Mask-before-normalize refit cannot decrease the restricted leading
    singular value compared with the raw-iterate's reconstruction. This is
    the BGSP paper's refit-dominance claim localised to the single-component case."""
    instance = _instance(seed=7)
    gamma = _gamma_with_strict_margin(instance)
    fit = BlockSparseEstimator(objective="squared", gamma=gamma, random_state=7).fit(instance.data)
    assert fit.refit is not None
    raw_score = fit.raw.factors
    raw_c = instance.data.T @ raw_score
    raw_loading = soft_threshold(raw_c, gamma)
    raw_norm = float(np.linalg.norm(raw_loading))
    assert raw_norm > 0
    raw_restricted = float(
        np.linalg.norm(instance.data[:, fit.support] @ (raw_loading[fit.support] / raw_norm))
    )
    assert fit.refit.objective is not None
    assert fit.refit.objective >= raw_restricted - 1e-10


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------


def test_zero_components_rejected() -> None:
    with pytest.raises(ValueError):
        BlockSparseEstimator(objective="squared", gamma=0.3).fit(_instance().data, m=0)


def test_m_exceeds_p_rejected() -> None:
    with pytest.raises(ValueError):
        BlockSparseEstimator(objective="squared", gamma=0.3).fit(_instance(p=4).data, m=5)


# ---------------------------------------------------------------------------
# m >= 2 Stiefel block
# ---------------------------------------------------------------------------


def _two_component_instance(seed: int = 0):
    """A planted-spike instance with two orthonormal scores, disjoint supports,
    and tuned singular values so A-SUM / A-SQ both identify both planted
    components under a strict margin."""
    return planted_spike(
        p=10,
        n=24,
        k=4,
        m=2,
        support_overlap=0,
        singular_values=(5.0, 3.0),
        noise_std=0.0,
        seed=seed,
    )


def test_m_equals_two_returns_stiefel_frame_and_per_column_supports() -> None:
    instance = _two_component_instance(seed=0)
    c = instance.data.T @ instance.scores
    on = [np.abs(c[instance.supports[j], j]).min() for j in range(2)]
    off = [np.abs(np.delete(c[:, j], instance.supports[j])).max() for j in range(2)]
    gamma = 0.5 * (min(on) + max(off))
    fit = BlockSparseEstimator(objective="squared", gamma=gamma, random_state=0).fit(
        instance.data, m=2
    )
    assert fit.m == 2
    assert fit.raw.factors.shape == (10, 2)
    assert np.allclose(fit.raw.factors.T @ fit.raw.factors, np.eye(2), atol=1e-6)
    assert fit.refit is None  # multi-component refit intentionally deferred
    assert len(fit.supports) == 2
    for j in range(2):
        assert fit.supports[j].size > 0


def test_m_equals_two_support_recovery_across_both_objectives() -> None:
    instance = _two_component_instance(seed=1)
    c = instance.data.T @ instance.scores
    on = [np.abs(c[instance.supports[j], j]).min() for j in range(2)]
    off = [np.abs(np.delete(c[:, j], instance.supports[j])).max() for j in range(2)]
    gamma = 0.5 * (min(on) + max(off))
    for objective in ("sum_norm", "squared"):
        fit = BlockSparseEstimator(objective=objective, gamma=gamma, random_state=1).fit(
            instance.data, m=2
        )
        recovered = {frozenset(s.tolist()) for s in fit.supports}
        planted = {frozenset(int(i) for i in s) for s in instance.supports}
        # Up to component permutation each planted support should appear.
        assert planted <= recovered, (objective, recovered, planted)


def test_unknown_objective_raises() -> None:
    with pytest.raises(ValueError):
        BlockSparseEstimator(objective="bogus")  # type: ignore[arg-type]


def test_negative_gamma_raises() -> None:
    with pytest.raises(ValueError):
        BlockSparseEstimator(gamma=-0.1)


def test_init_vector_is_normalized_and_shape_checked() -> None:
    instance = _instance()
    gamma = _gamma_with_strict_margin(instance)
    # Correct shape, nonzero: accepted.
    fit = BlockSparseEstimator(objective="squared", gamma=gamma).fit(
        instance.data, init=np.random.default_rng(0).normal(size=instance.data.shape[0])
    )
    assert fit.raw.factors.shape == (instance.data.shape[0],)
    # Wrong shape: rejected.
    with pytest.raises(ValueError):
        BlockSparseEstimator(objective="squared", gamma=gamma).fit(instance.data, init=np.zeros(3))
    # Zero vector: rejected.
    with pytest.raises(ValueError):
        BlockSparseEstimator(objective="squared", gamma=gamma).fit(
            instance.data, init=np.zeros(instance.data.shape[0])
        )
