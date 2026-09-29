"""
estimators.py — Gradient estimators for discrete token boundaries.

Each estimator converts B's logits at each autoregressive step into
a "soft token" representation that:
  - Forward: behaves like a discrete token (or close to one)
  - Backward: allows gradients to flow through

All estimators follow the same interface:

    def decode(B_model, tokenizer, input_embeds, intermediate_len, **kwargs)
        -> (soft_tokens: [1, L, V], hard_ids: [L])

This module is used by ste_gcg.py to swap in different estimators.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor
from dataclasses import dataclass
from typing import Callable


@dataclass
class EstimatorConfig:
    """Configuration for a gradient estimator."""
    name: str
    decode_fn: Callable  # the decode function
    description: str
    # Estimator-specific hyperparameters
    temperature: float = 1.0
    tau_start: float = 1.0      # for Gumbel-Softmax annealing
    tau_end: float = 0.1
    topk: int = 64              # for top-k soft mixing
    n_samples: int = 8          # for REINFORCE


# ---------------------------------------------------------------------------
# 1. Straight-Through Estimator (STE)
# ---------------------------------------------------------------------------

def ste_decode(
    B_model,
    tokenizer,
    input_embeds: Tensor,
    intermediate_len: int,
    *,
    temperature: float = 1.0,
    step_frac: float = 0.0,   # unused, for API consistency
    **kwargs,
) -> tuple[Tensor, LongTensor]:
    """Standard STE: forward=hard argmax, backward=soft softmax.

    ŷ_t = one_hot(argmax(logits)) + (softmax(logits/τ) - softmax(logits/τ).detach())
    """
    device = input_embeds.device
    embed_matrix = B_model.get_input_embeddings().weight
    vocab_size = embed_matrix.shape[0]

    past_key_values = None
    current_embeds = input_embeds

    ste_tokens_list = []
    hard_ids_list = []

    for t in range(intermediate_len):
        outputs = B_model(
            inputs_embeds=current_embeds,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        logits_t = outputs.logits[:, -1, :]

        y_t = F.softmax(logits_t / temperature, dim=-1)
        hard_idx = logits_t.argmax(dim=-1)
        h_t = F.one_hot(hard_idx, num_classes=vocab_size).float().to(y_t.dtype)

        # STE trick: forward = h_t, backward = ∂/∂y_t
        ste_t = h_t + (y_t - y_t.detach())

        ste_tokens_list.append(ste_t.unsqueeze(1))
        hard_ids_list.append(hard_idx.item())

        next_embed = (ste_t @ embed_matrix).unsqueeze(1)
        current_embeds = next_embed

    soft_tokens = torch.cat(ste_tokens_list, dim=1)
    hard_ids = torch.tensor(hard_ids_list, dtype=torch.long)
    return soft_tokens, hard_ids


# ---------------------------------------------------------------------------
# Gumbel noise helper (float32-safe for bfloat16 models)
# ---------------------------------------------------------------------------

def _gumbel_noise(logits: Tensor) -> Tensor:
    """Sample Gumbel(0,1) noise, computed in float32 for numerical stability.

    The double-log in -log(-log(U)) overflows bfloat16's limited precision
    (~3.4 decimal digits). We compute in float32 and cast back.

    IMPORTANT: We use explicit intermediate variables because Python precedence
    makes `-torch.log(x).clamp()` parse as `-(torch.log(x).clamp())`, not
    `(-torch.log(x)).clamp()`, which causes NaN when log returns negatives.
    """
    u = torch.rand(logits.shape, device=logits.device, dtype=torch.float32)
    u = u.clamp(min=1e-10, max=1.0 - 1e-7)  # avoid log(0) and log(1)=0
    neg_log_u = -torch.log(u)                 # positive values
    neg_log_u = neg_log_u.clamp(min=1e-10)    # ensure positive for second log
    noise = -torch.log(neg_log_u)             # Gumbel(0,1) noise
    return noise.to(logits.dtype)


# ---------------------------------------------------------------------------
# 2. Gumbel-Softmax (Concrete Relaxation)
# ---------------------------------------------------------------------------

def gumbel_softmax_decode(
    B_model,
    tokenizer,
    input_embeds: Tensor,
    intermediate_len: int,
    *,
    temperature: float = 1.0,
    tau_start: float = 1.0,
    tau_end: float = 0.1,
    step_frac: float = 0.0,   # fraction through optimization (0→1) for annealing
    **kwargs,
) -> tuple[Tensor, LongTensor]:
    """Gumbel-Softmax: fully differentiable categorical relaxation.

    y_t = softmax((logits + Gumbel_noise) / τ)

    Temperature τ anneals from tau_start to tau_end based on step_frac.
    At low τ, the output approaches one-hot but gradients may vanish.
    """
    device = input_embeds.device
    embed_matrix = B_model.get_input_embeddings().weight
    vocab_size = embed_matrix.shape[0]

    # Anneal temperature
    tau = tau_start + (tau_end - tau_start) * step_frac

    past_key_values = None
    current_embeds = input_embeds

    soft_tokens_list = []
    hard_ids_list = []

    for t in range(intermediate_len):
        outputs = B_model(
            inputs_embeds=current_embeds,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        logits_t = outputs.logits[:, -1, :]

        # Gumbel-Softmax: add Gumbel noise and apply softmax
        noise = _gumbel_noise(logits_t)
        y_t = F.softmax((logits_t + noise) / tau, dim=-1)

        hard_idx = logits_t.argmax(dim=-1)  # for logging
        hard_ids_list.append(hard_idx.item())

        soft_tokens_list.append(y_t.unsqueeze(1))

        next_embed = (y_t @ embed_matrix).unsqueeze(1)
        current_embeds = next_embed

    soft_tokens = torch.cat(soft_tokens_list, dim=1)
    hard_ids = torch.tensor(hard_ids_list, dtype=torch.long)
    return soft_tokens, hard_ids


# ---------------------------------------------------------------------------
# 3. ST-Gumbel-Softmax (hybrid: hard forward, Gumbel backward)
# ---------------------------------------------------------------------------

def st_gumbel_decode(
    B_model,
    tokenizer,
    input_embeds: Tensor,
    intermediate_len: int,
    *,
    temperature: float = 1.0,
    tau_start: float = 1.0,
    tau_end: float = 0.1,
    step_frac: float = 0.0,
    **kwargs,
) -> tuple[Tensor, LongTensor]:
    """ST-Gumbel: forward=hard argmax, backward=Gumbel-Softmax gradients.

    Combines STE with Gumbel-Softmax:
      Forward:  h_t = one_hot(argmax(logits))
      Backward: through Gumbel-Softmax relaxation

    A sees real discrete tokens (valid input), but gradients come from
    the Gumbel-Softmax distribution rather than vanilla softmax.
    """
    device = input_embeds.device
    embed_matrix = B_model.get_input_embeddings().weight
    vocab_size = embed_matrix.shape[0]

    tau = tau_start + (tau_end - tau_start) * step_frac

    past_key_values = None
    current_embeds = input_embeds

    soft_tokens_list = []
    hard_ids_list = []

    for t in range(intermediate_len):
        outputs = B_model(
            inputs_embeds=current_embeds,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        logits_t = outputs.logits[:, -1, :]

        # Gumbel-Softmax soft probabilities
        noise = _gumbel_noise(logits_t)
        y_t = F.softmax((logits_t + noise) / tau, dim=-1)

        # Hard one-hot
        hard_idx = logits_t.argmax(dim=-1)
        h_t = F.one_hot(hard_idx, num_classes=vocab_size).float().to(y_t.dtype)

        # ST trick: forward=hard, backward=Gumbel-Softmax
        ste_t = h_t + (y_t - y_t.detach())

        soft_tokens_list.append(ste_t.unsqueeze(1))
        hard_ids_list.append(hard_idx.item())

        next_embed = (ste_t @ embed_matrix).unsqueeze(1)
        current_embeds = next_embed

    soft_tokens = torch.cat(soft_tokens_list, dim=1)
    hard_ids = torch.tensor(hard_ids_list, dtype=torch.long)
    return soft_tokens, hard_ids


# ---------------------------------------------------------------------------
# 4. Soft Token Mixing (continuous relaxation, no STE)
# ---------------------------------------------------------------------------

def soft_mixing_decode(
    B_model,
    tokenizer,
    input_embeds: Tensor,
    intermediate_len: int,
    *,
    temperature: float = 0.1,
    step_frac: float = 0.0,
    tau_start: float = 0.5,
    tau_end: float = 0.05,
    **kwargs,
) -> tuple[Tensor, LongTensor]:
    """Soft mixing: fully differentiable, no discrete step at all.

    embed_{t+1} = softmax(logits / τ) @ Embedding

    No STE or Gumbel needed — the soft distribution IS the output.
    τ is annealed to make the distribution sharper over time.
    At very low τ, approaches one-hot but is always differentiable.
    """
    device = input_embeds.device
    embed_matrix = B_model.get_input_embeddings().weight
    vocab_size = embed_matrix.shape[0]

    tau = tau_start + (tau_end - tau_start) * step_frac

    past_key_values = None
    current_embeds = input_embeds

    soft_tokens_list = []
    hard_ids_list = []

    for t in range(intermediate_len):
        outputs = B_model(
            inputs_embeds=current_embeds,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        logits_t = outputs.logits[:, -1, :]

        # Pure soft mixing — fully differentiable
        y_t = F.softmax(logits_t / tau, dim=-1)

        hard_idx = logits_t.argmax(dim=-1)  # for logging only
        hard_ids_list.append(hard_idx.item())

        soft_tokens_list.append(y_t.unsqueeze(1))

        next_embed = (y_t @ embed_matrix).unsqueeze(1)
        current_embeds = next_embed

    soft_tokens = torch.cat(soft_tokens_list, dim=1)
    hard_ids = torch.tensor(hard_ids_list, dtype=torch.long)
    return soft_tokens, hard_ids


# ---------------------------------------------------------------------------
# 5. Top-k Soft Mixing
# ---------------------------------------------------------------------------

def topk_soft_decode(
    B_model,
    tokenizer,
    input_embeds: Tensor,
    intermediate_len: int,
    *,
    temperature: float = 0.3,
    topk: int = 64,
    step_frac: float = 0.0,
    **kwargs,
) -> tuple[Tensor, LongTensor]:
    """Top-k soft mixing: like soft mixing but only over top-k tokens.

    Concentrates the gradient on the most likely tokens, reducing noise.
    More memory-efficient than full-vocabulary soft mixing.

    embed_{t+1} = softmax(top_k_logits / τ) @ Embedding[top_k_ids]
    """
    device = input_embeds.device
    embed_matrix = B_model.get_input_embeddings().weight
    vocab_size = embed_matrix.shape[0]

    past_key_values = None
    current_embeds = input_embeds

    soft_tokens_list = []
    hard_ids_list = []

    for t in range(intermediate_len):
        outputs = B_model(
            inputs_embeds=current_embeds,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        logits_t = outputs.logits[:, -1, :]  # [1, V]

        # Top-k selection
        topk_vals, topk_idx = logits_t.topk(topk, dim=-1)  # [1, k]
        topk_probs = F.softmax(topk_vals / temperature, dim=-1)  # [1, k]

        # Build full-vocab soft token (sparse: only top-k are nonzero)
        y_t = torch.zeros_like(logits_t)  # [1, V]
        y_t.scatter_(1, topk_idx, topk_probs)

        hard_idx = logits_t.argmax(dim=-1)
        hard_ids_list.append(hard_idx.item())

        soft_tokens_list.append(y_t.unsqueeze(1))

        # Embed using only top-k for efficiency
        topk_embeds = embed_matrix[topk_idx.squeeze(0)]  # [k, d]
        next_embed = (topk_probs @ topk_embeds).unsqueeze(1)  # [1, 1, d]
        current_embeds = next_embed

    soft_tokens = torch.cat(soft_tokens_list, dim=1)
    hard_ids = torch.tensor(hard_ids_list, dtype=torch.long)
    return soft_tokens, hard_ids


# ---------------------------------------------------------------------------
# 6. REINFORCE (Score Function Estimator)
# ---------------------------------------------------------------------------

def reinforce_decode(
    B_model,
    tokenizer,
    input_embeds: Tensor,
    intermediate_len: int,
    *,
    temperature: float = 1.0,
    n_samples: int = 8,
    step_frac: float = 0.0,
    **kwargs,
) -> tuple[Tensor, LongTensor]:
    """REINFORCE: sample from B's distribution, use log-prob for gradients.

    Instead of STE, we use the log-probability trick:
      - Sample tokens from B's softmax distribution
      - The "soft token" carries gradient via: y_t = softmax(logits/τ)
        where we use the sampled token in forward but softmax in backward

    This is actually equivalent to STE with sampled (not argmax) forward.
    For true REINFORCE (loss × ∇log p), the loss weighting happens in
    the outer GCG loop, not here.

    We sample instead of argmax to explore more of the output space.
    """
    device = input_embeds.device
    embed_matrix = B_model.get_input_embeddings().weight
    vocab_size = embed_matrix.shape[0]

    past_key_values = None
    current_embeds = input_embeds

    soft_tokens_list = []
    hard_ids_list = []

    for t in range(intermediate_len):
        outputs = B_model(
            inputs_embeds=current_embeds,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        logits_t = outputs.logits[:, -1, :]

        # Soft probabilities (for gradient)
        y_t = F.softmax(logits_t / temperature, dim=-1)

        # Sample (not argmax) — explores more of the space
        sampled_idx = torch.multinomial(y_t, num_samples=1).squeeze(-1)  # [1]
        h_t = F.one_hot(sampled_idx, num_classes=vocab_size).float().to(y_t.dtype)

        # STE with sampled forward (instead of argmax forward)
        ste_t = h_t + (y_t - y_t.detach())

        soft_tokens_list.append(ste_t.unsqueeze(1))
        hard_ids_list.append(sampled_idx.item())

        next_embed = (ste_t @ embed_matrix).unsqueeze(1)
        current_embeds = next_embed

    soft_tokens = torch.cat(soft_tokens_list, dim=1)
    hard_ids = torch.tensor(hard_ids_list, dtype=torch.long)
    return soft_tokens, hard_ids


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ESTIMATORS: dict[str, EstimatorConfig] = {
    "ste": EstimatorConfig(
        name="ste",
        decode_fn=ste_decode,
        description="Straight-Through Estimator: forward=argmax, backward=softmax",
        temperature=1.0,
    ),
    "gumbel": EstimatorConfig(
        name="gumbel",
        decode_fn=gumbel_softmax_decode,
        description="Gumbel-Softmax: differentiable categorical relaxation with annealing",
        tau_start=1.0,
        tau_end=0.1,
    ),
    "annealed_gumbel": EstimatorConfig(
        name="annealed_gumbel",
        decode_fn=gumbel_softmax_decode,
        description="Gumbel-Softmax with aggressive annealing (τ 2.0→0.05) for wider exploration early on",
        tau_start=2.0,
        tau_end=0.05,
    ),
    "st_gumbel": EstimatorConfig(
        name="st_gumbel",
        decode_fn=st_gumbel_decode,
        description="ST-Gumbel: forward=argmax, backward=Gumbel-Softmax (lower bias than STE)",
        tau_start=1.0,
        tau_end=0.1,
    ),
    "annealed_st_gumbel": EstimatorConfig(
        name="annealed_st_gumbel",
        decode_fn=st_gumbel_decode,
        description="ST-Gumbel with aggressive annealing (τ 2.0→0.05)",
        tau_start=2.0,
        tau_end=0.05,
    ),
    "soft_mixing": EstimatorConfig(
        name="soft_mixing",
        decode_fn=soft_mixing_decode,
        description="Soft token mixing: fully differentiable softmax→embedding, no discrete step",
        tau_start=0.5,
        tau_end=0.05,
    ),
    "topk_soft": EstimatorConfig(
        name="topk_soft",
        decode_fn=topk_soft_decode,
        description="Top-k soft mixing: soft mixing restricted to top-k tokens",
        topk=64,
        temperature=0.3,
    ),
    "reinforce": EstimatorConfig(
        name="reinforce",
        decode_fn=reinforce_decode,
        description="REINFORCE-style: sampled forward (not argmax) with softmax backward",
        temperature=1.0,
        n_samples=8,
    ),
}


def get_estimator(name: str) -> EstimatorConfig:
    """Get an estimator config by name."""
    if name not in ESTIMATORS:
        raise ValueError(
            f"Unknown estimator '{name}'. Available: {list(ESTIMATORS.keys())}"
        )
    return ESTIMATORS[name]


def list_estimators() -> list[str]:
    """Return list of available estimator names."""
    return list(ESTIMATORS.keys())
