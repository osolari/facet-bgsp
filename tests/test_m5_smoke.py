"""Smoke tests for the M5 ViT CLS driver (image-free, ViT-free helpers)."""

from __future__ import annotations

import numpy as np

from experiments import m5_vit_cls as m5
from experiments._vit import generate_image_set, synthetic_classes


def test_synthetic_classes_has_ten_distinct_labels() -> None:
    classes = synthetic_classes()
    labels = {c.label for c in classes}
    assert labels == set(range(10))
    assert len({c.name for c in classes}) == 10


def test_generate_image_set_produces_right_shapes_and_labels() -> None:
    classes = synthetic_classes()
    images, labels = generate_image_set(classes, per_class=3, size=64, seed=0)
    assert len(images) == 30
    assert labels.shape == (30,)
    # Every image is a PIL Image of size 64x64.
    for img in images:
        assert img.size == (64, 64)


def test_dominant_component_assignment_shape() -> None:
    rng = np.random.default_rng(0)
    d, N, m = 32, 200, 10
    H = rng.normal(size=(d, N))
    comps = rng.normal(size=(d, m))
    assignment = m5._dominant_component_assignment(comps, H)
    assert assignment.shape == (N,)
    assert assignment.min() >= 0
    assert assignment.max() < m


def test_facet_components_fallback_returns_shape() -> None:
    rng = np.random.default_rng(2)
    H = rng.normal(size=(16, 60))
    comps, _ = m5._facet_components(H, m=4, gammas=[0.1, 0.3], seed=0)
    assert comps.shape == (16, 4)
