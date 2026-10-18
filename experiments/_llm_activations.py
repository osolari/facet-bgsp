"""GPT-2 residual-stream activation helpers (Tier-3).

Loads a GPT-2 model, extracts residual-stream activations at a chosen
transformer block, and provides a lightweight causal-ablation probe that
measures the KL divergence of next-token logits when a component direction
is subtracted from the residual stream before the final layernorm/head.

``torch`` is imported lazily inside each function so the module can be
imported in a CI environment without ``facet[torch]`` installed (the smoke
tests under ``tests/`` only exercise the pure-numpy helpers).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ActivationSample:
    tokens: list[str]
    activations: np.ndarray  # (d_model, n_tokens)


def load_gpt2(model_name: str = "gpt2") -> tuple[Any, Any]:
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


def _resolve_hook_module(model: Any, layer: int, site: str) -> Any:
    """Return the submodule whose output is the activation we want to capture.

    Supported ``site`` values:
      - ``resid_post`` : output of the whole transformer block (post-residual).
      - ``attn_out``   : output of the attention sub-block (pre-residual add).
      - ``mlp_out``    : output of the MLP sub-block (pre-residual add).
    """
    block = model.transformer.h[layer]
    if site == "resid_post":
        return block
    if site == "attn_out":
        return block.attn
    if site == "mlp_out":
        return block.mlp
    raise ValueError(f"unknown hook site {site!r}")


def extract_residual_activations(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    layer: int,
    max_tokens_per_text: int,
    site: str = "resid_post",
) -> np.ndarray:
    """Return activations at ``(layer, site)`` as a (d_model, N) matrix."""
    import torch

    target = _resolve_hook_module(model, layer, site)
    captured: list[Any] = []

    def _hook(_module: Any, _inputs: Any, outputs: Any) -> None:
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
        captured.append(hidden.detach())

    with torch.no_grad():
        handle = target.register_forward_hook(_hook)
        try:
            for text in texts:
                ids = tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=max_tokens_per_text
                ).input_ids
                model(ids)
        finally:
            handle.remove()

    stacked = torch.cat([h.squeeze(0) for h in captured], dim=0)
    return stacked.T.cpu().numpy().astype(np.float64)


def extract_residual_activations_with_token_ids(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    layer: int,
    max_tokens_per_text: int,
    site: str = "resid_post",
) -> tuple[np.ndarray, np.ndarray]:
    """Like :func:`extract_residual_activations` but also returns token ids."""
    import torch

    target = _resolve_hook_module(model, layer, site)
    captured: list[Any] = []
    all_ids: list[Any] = []

    def _hook(_module: Any, _inputs: Any, outputs: Any) -> None:
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
        captured.append(hidden.detach())

    with torch.no_grad():
        handle = target.register_forward_hook(_hook)
        try:
            for text in texts:
                ids = tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=max_tokens_per_text
                ).input_ids
                model(ids)
                all_ids.append(ids.squeeze(0).detach())
        finally:
            handle.remove()

    stacked = torch.cat([h.squeeze(0) for h in captured], dim=0)
    tok_ids = torch.cat(all_ids, dim=0)
    return (
        stacked.T.cpu().numpy().astype(np.float64),
        tok_ids.cpu().numpy().astype(np.int32),
    )


def token_input_embeddings(model: Any, token_ids: np.ndarray) -> np.ndarray:
    """Return the GPT-2 input (wte) embedding of each token id, as (N, d_model)."""
    import torch

    with torch.no_grad():
        wte = model.transformer.wte
        ids = torch.as_tensor(token_ids, dtype=torch.long)
        return wte(ids).detach().cpu().numpy().astype(np.float64)


def ablation_kl_bits(
    model: Any,
    tokenizer: Any,
    text: str,
    layer: int,
    direction: np.ndarray,
    token_pos: int = -1,
) -> float:
    """Subtract the component of the residual stream along ``direction`` after block
    ``layer`` and return the KL(unablated || ablated) at ``token_pos`` in bits.
    ``direction`` is a unit-norm (d_model,) numpy array.
    """
    import torch

    with torch.no_grad():
        ids = tokenizer(text, return_tensors="pt").input_ids
        direction_t = torch.from_numpy(np.asarray(direction, dtype=np.float32)).reshape(-1)
        direction_t = direction_t / direction_t.norm().clamp_min(1e-12)

        def _ablate_hook(_module: Any, _inputs: Any, outputs: Any) -> Any:
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs
            proj = hidden @ direction_t
            hidden_new = hidden - proj.unsqueeze(-1) * direction_t.view(1, 1, -1)
            if isinstance(outputs, tuple):
                return (hidden_new, *outputs[1:])
            return hidden_new

        logits_clean = model(ids).logits[0, token_pos].log_softmax(dim=-1)
        handle = model.transformer.h[layer].register_forward_hook(_ablate_hook)
        try:
            logits_abl = model(ids).logits[0, token_pos].log_softmax(dim=-1)
        finally:
            handle.remove()

        p = logits_clean.exp()
        kl_nats = float((p * (logits_clean - logits_abl)).sum())
        return kl_nats / float(np.log(2.0))
