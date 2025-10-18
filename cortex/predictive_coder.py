#!/usr/bin/env python3
"""
Predictive Coding Modules for Cortical Sheet

Implements hierarchical predictive coding blocks inspired by:
- Rao & Ballard (1999): Predictive coding in visual cortex
- Friston (2010): Free-energy principle

Each PC block:
- Encodes input → latent z
- Decodes latent → prediction x_hat
- Computes error e = x - x_hat
- Propagates error gradients cleanly (fully differentiable)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .entropy_flux import entropy_flux, kl_sparsity


class PCBlock(nn.Module):
    """
    A small predictive coding block:
    - encoder: x -> z
    - decoder: z -> x_hat
    - residual error: e = x - x_hat (propagates gradients cleanly)

    Neuroscience inspiration:
        Cortical microcircuit with feedback (decoder) and feedforward (encoder).
        Error neurons (e) carry prediction residuals up the hierarchy.
    """
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_in),
        )
        # Lightweight skip-conditioning for stability
        self.skip = nn.Linear(d_in, d_hidden, bias=False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with predictive coding dynamics.

        Args:
            x: Input tensor [B, T, D] or [B, D]

        Returns:
            Dict with:
                z: Latent encoding [B, T, D_hidden]
                x_hat: Reconstruction [B, T, D]
                e: Prediction error [B, T, D]
        """
        z = self.encoder(x) + self.skip(x)
        x_hat = self.decoder(z)
        e = x - x_hat
        return {"z": z, "x_hat": x_hat, "e": e}


class PredictiveCoder(nn.Module):
    """
    Multi-block predictive coder with optional temporal prediction (1-step).
    Exposes differentiable reconstruction and entropy-flux losses.

    Stack of PC blocks creates a hierarchy:
        x -> [PC1] -> z1 -> [PC2] -> z2 -> ... -> z_final

    Each level predicts the level below, minimizing prediction error.
    """
    def __init__(self, d_in: int, d_hidden: int, depth: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([
            PCBlock(d_in if i == 0 else d_hidden, d_hidden)
            for i in range(depth)
        ])
        # Head to fold final z back to residual (for residual fusion upstream)
        self.residual_head = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_in),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run hierarchical predictive coding.

        Args:
            x: [B, T, D] tokens (T may be 1)

        Returns:
            Dict with:
                residual: [B, D] - Mean-pooled processed representation
                losses: Dict with recon, entropy
                z: [B, D_hidden] - Final latent encoding
        """
        B, T, D = x.shape
        h = x
        losses = {}
        recon_losses = []
        ent_flux = []

        for blk in self.blocks:
            out = blk(h)
            # Compute block-local losses
            recon = (out["e"] ** 2).mean()
            recon_losses.append(recon)
            ent_flux.append(entropy_flux(h, out["x_hat"], dim=-1).mean())
            # Feed next block with latent
            h = out["z"]

        # Aggregate latent by mean across tokens
        z_final = h.mean(dim=1) if h.dim() == 3 else h  # [B, D_hidden]
        residual = self.residual_head(z_final)  # [B, D]

        losses["recon"] = torch.stack(recon_losses).mean()
        # Keep raw flux as diagnostic (can be negative)
        raw_flux = torch.stack(ent_flux).mean()
        losses["entropy_flux_raw"] = raw_flux
        # Clamp negative flux so pc_entropy ≥ 0 for loss aggregation
        losses["entropy"] = raw_flux.clamp_min(0.0)

        return {
            "residual": residual,
            "losses": losses,
            "z": z_final
        }
