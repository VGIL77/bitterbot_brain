#!/usr/bin/env python3
"""
Entropy Flux and Sparsity Utilities for Predictive Coding

Implements:
- Normalized token entropy computation
- Entropy flux (relative entropy reduction)
- KL sparsity regularization (Olshausen-Field style)

References:
- Olshausen & Field (1996): Sparse coding in V1
- Friston (2010): Free-energy principle
"""
import torch
import torch.nn.functional as F


def token_entropy(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute normalized token entropy along `dim`.

    Args:
        x: Arbitrary float tensor; will be normalized with softmax
        dim: Dimension to compute entropy over

    Returns:
        Entropy in [0, 1] (normalized by log(n_dims))

    Formula:
        H = -Σ p(x) log p(x)
        H_norm = H / log(K)  where K = size of dim
    """
    probs = torch.softmax(x, dim=dim).clamp_min(1e-10)
    H = -(probs * probs.log()).sum(dim=dim)
    denom = torch.log(torch.tensor(x.size(dim), device=x.device, dtype=x.dtype)) if x.size(dim) > 1 else 1.0
    return (H / denom).clamp(0.0, 1.0)


def entropy_flux(before: torch.Tensor, after: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Relative entropy reduction: (H_before - H_after) / max(H_before, eps).

    Args:
        before: Input tensor (higher uncertainty)
        after: Output tensor (hopefully lower uncertainty)
        dim: Dimension to compute entropy over

    Returns:
        Relative entropy reduction. Positive value means entropy decreased.
        Clamped to [0, inf) to ensure it's suitable as a loss term.

    Interpretation:
        > 0: Model reduced uncertainty (good prediction)
        = 0: No change in uncertainty
        < 0: Model increased uncertainty (poor prediction) → clamped to 0

    Neuroscience connection:
        Predictive coding aims to minimize surprise (Friston's free-energy).
        Entropy flux measures how much surprise was reduced.
    """
    Hb = token_entropy(before, dim=dim)
    Ha = token_entropy(after, dim=dim)
    flux = (Hb - Ha) / Hb.clamp_min(1e-6)
    # Clamp negative flux to 0 to ensure entropy loss ≥ 0
    return flux.clamp_min(0.0)


def kl_sparsity(z: torch.Tensor, target: float = 0.05) -> torch.Tensor:
    """
    KL divergence between mean |z| (squashed) and target sparsity.
    Encourages sparse latent codes akin to Olshausen & Field style priors.

    Args:
        z: Latent tensor (any shape with last dim = features)
        target: Target sparsity level (e.g., 0.05 = 5% active)

    Returns:
        Scalar KL divergence loss

    Formula:
        p = sigmoid(|z|.mean())  (activation rate)
        KL(p || target) = p log(p/q) + (1-p) log((1-p)/(1-q))

    Neuroscience connection:
        V1 simple cells have sparse firing rates (~5%)
        Sparse codes are more efficient and generalizable
        (Olshausen & Field, Nature 1996)
    """
    # Squash to (0,1) via sigmoid after magnitude pooling
    p = torch.sigmoid(z.abs().mean(dim=-1))
    q = torch.tensor(target, device=z.device, dtype=z.dtype)
    kl = (p * (p.clamp_min(1e-10)/q).log() + (1-p) * ((1-p).clamp_min(1e-10)/(1-q)).log()).mean()
    return kl
