"""ViT CLS-token helpers for M5.

Synthetic image generation (ten class prototypes with controlled variations)
plus a ViT-base-patch16-224 feature extractor. We keep zero image-dataset
dependencies: images are generated deterministically from a per-class
template with pixel-space jitter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SyntheticClass:
    label: int
    name: str
    render: callable  # (rng, 224) -> np.ndarray (H, W, 3) in [0, 255]


def _solid(color: tuple[int, int, int]) -> callable:
    def render(rng: np.random.Generator, size: int) -> np.ndarray:
        base = np.zeros((size, size, 3), dtype=np.float32)
        c = np.array(color, dtype=np.float32) + rng.integers(-15, 16, size=3)
        base[:] = np.clip(c, 0.0, 255.0)
        return base.astype(np.uint8)

    return render


def _horizontal_stripes(a: tuple[int, int, int], b: tuple[int, int, int]) -> callable:
    def render(rng: np.random.Generator, size: int) -> np.ndarray:
        img = np.zeros((size, size, 3), dtype=np.float32)
        period = int(rng.integers(16, 48))
        for y in range(size):
            band = (y // period) % 2
            color = np.array(a if band == 0 else b, dtype=np.float32)
            color += rng.integers(-10, 11, size=3)
            img[y, :] = np.clip(color, 0.0, 255.0)
        return img.astype(np.uint8)

    return render


def _vertical_stripes(a: tuple[int, int, int], b: tuple[int, int, int]) -> callable:
    def render(rng: np.random.Generator, size: int) -> np.ndarray:
        img = np.zeros((size, size, 3), dtype=np.float32)
        period = int(rng.integers(16, 48))
        for x in range(size):
            band = (x // period) % 2
            color = np.array(a if band == 0 else b, dtype=np.float32)
            color += rng.integers(-10, 11, size=3)
            img[:, x] = np.clip(color, 0.0, 255.0)
        return img.astype(np.uint8)

    return render


def _diagonal_gradient(a: tuple[int, int, int], b: tuple[int, int, int]) -> callable:
    def render(rng: np.random.Generator, size: int) -> np.ndarray:
        xx, yy = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
        t = (xx + yy) / 2.0
        t = np.clip(t + rng.normal(scale=0.03), 0.0, 1.0)
        a_arr = np.array(a, dtype=np.float32)
        b_arr = np.array(b, dtype=np.float32)
        img = (1.0 - t)[..., None] * a_arr + t[..., None] * b_arr
        return np.clip(img, 0.0, 255.0).astype(np.uint8)

    return render


def _centered_disk(bg: tuple[int, int, int], fg: tuple[int, int, int]) -> callable:
    def render(rng: np.random.Generator, size: int) -> np.ndarray:
        img = np.tile(np.array(bg, dtype=np.float32), (size, size, 1))
        cx = size // 2 + int(rng.integers(-12, 13))
        cy = size // 2 + int(rng.integers(-12, 13))
        r = int(0.25 * size) + int(rng.integers(-8, 9))
        xx, yy = np.meshgrid(np.arange(size), np.arange(size))
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
        img[mask] = np.array(fg, dtype=np.float32)
        img += rng.normal(scale=5.0, size=img.shape)
        return np.clip(img, 0.0, 255.0).astype(np.uint8)

    return render


def _frame(bg: tuple[int, int, int], fg: tuple[int, int, int]) -> callable:
    def render(rng: np.random.Generator, size: int) -> np.ndarray:
        img = np.tile(np.array(bg, dtype=np.float32), (size, size, 1))
        thickness = int(0.1 * size) + int(rng.integers(-6, 7))
        fg_arr = np.array(fg, dtype=np.float32)
        img[:thickness] = fg_arr
        img[-thickness:] = fg_arr
        img[:, :thickness] = fg_arr
        img[:, -thickness:] = fg_arr
        img += rng.normal(scale=5.0, size=img.shape)
        return np.clip(img, 0.0, 255.0).astype(np.uint8)

    return render


def synthetic_classes() -> list[SyntheticClass]:
    return [
        SyntheticClass(0, "solid-red", _solid((200, 40, 40))),
        SyntheticClass(1, "solid-blue", _solid((40, 60, 200))),
        SyntheticClass(2, "solid-green", _solid((40, 180, 60))),
        SyntheticClass(3, "stripes-h-rb", _horizontal_stripes((200, 50, 50), (50, 60, 200))),
        SyntheticClass(4, "stripes-v-rg", _vertical_stripes((200, 50, 50), (50, 180, 60))),
        SyntheticClass(5, "diag-grad-by", _diagonal_gradient((20, 20, 200), (240, 220, 60))),
        SyntheticClass(6, "disk-red-wht", _centered_disk((240, 240, 240), (200, 40, 40))),
        SyntheticClass(7, "disk-blue-wht", _centered_disk((240, 240, 240), (40, 60, 200))),
        SyntheticClass(8, "frame-red-wht", _frame((240, 240, 240), (200, 40, 40))),
        SyntheticClass(9, "frame-wht-red", _frame((200, 40, 40), (240, 240, 240))),
    ]


def generate_image_set(
    classes: list[SyntheticClass], per_class: int, size: int, seed: int
) -> tuple[list[Image.Image], np.ndarray]:
    rng = np.random.default_rng(seed)
    images: list[Image.Image] = []
    labels: list[int] = []
    for cls in classes:
        for _ in range(per_class):
            arr = cls.render(rng, size)
            images.append(Image.fromarray(arr))
            labels.append(cls.label)
    return images, np.array(labels, dtype=np.int32)


def extract_vit_cls(
    model: Any, processor: Any, images: list[Image.Image], layer: int | None = None
) -> np.ndarray:
    """Return CLS activations from the last transformer block (or ``layer`` if given).

    If ``layer`` is None, returns the final CLS hidden state. Returns shape
    ``(d_model, n_images)``.
    """
    import torch

    with torch.no_grad():
        inputs = processor(images=images, return_tensors="pt")
        outputs = model(**inputs, output_hidden_states=True)
        if layer is None:
            cls = outputs.last_hidden_state[:, 0, :]
        else:
            cls = outputs.hidden_states[layer][:, 0, :]
        return cls.detach().cpu().numpy().T.astype(np.float64)
