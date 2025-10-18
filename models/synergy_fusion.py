#!/usr/bin/env python3
"""
SynergyFusion: Fuses Cortex (predictive coder), RelMem (associative),
and BrainGraph (structured relations) into unified gradient-carrying priors.

Architecture inspired by:
- Predictive coding (Rao & Ballard, 1999; Friston, 2010)
- Associative memory as Hopfield attention (Ramsauer et al., 2020)
- Trust-weighted ensemble learning (precision-weighted prediction errors)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SynergyFusion(nn.Module):
    """
    Fuse three sources of prior knowledge into actionable signals:

    Inputs:
        1. Cortex: Predictive coding residuals + prior scales (φ, κ, CGE)
        2. RelMem: Concept embeddings + operation biases
        3. BrainGraph: Structural task embeddings

    Outputs:
        1. op_prior_logits [B, num_ops]: DSL operation priors for search
        2. resid_nudge [B, D]: Residual to enrich brain latent

    Key Features:
        - Trust-gated blending: Use memory confidence to modulate gradients
        - Reparameterization trick: Stochastic trust for smooth optimization
        - GPU-first: All ops on CUDA, device-safe projections
    """

    def __init__(self,
                 brain_dim: int,
                 op_dim: int,
                 concept_dim: int = 256,
                 trust_temp: float = 1.0,
                 device: str = "cuda"):
        """
        Args:
            brain_dim: Dimension of TOPAS brain latent (ctrl_dim, typically 768)
            op_dim: Number of DSL operations (typically 41)
            concept_dim: RelMem concept embedding dimension (default 256)
            trust_temp: Temperature for trust gating (higher = more exploration)
            device: Target device for all modules
        """
        super().__init__()
        self.brain_dim = brain_dim
        self.op_dim = op_dim
        self.concept_dim = concept_dim

        # Register device buffer for dynamic tracking (follows .to() calls)
        self.register_buffer("_dev", torch.empty(0, device=torch.device(device)))

        # === Operation Prior Head ===
        self.op_proj_ctx = nn.Sequential(
            nn.Linear(brain_dim + concept_dim, 256),
            nn.GELU(),
            nn.Linear(256, op_dim)
        ).to(self._dev.device)

        # === Residual Nudge Head ===
        self.resid_proj = nn.Sequential(
            nn.Linear(brain_dim + concept_dim, 256),
            nn.GELU(),
            nn.Linear(256, brain_dim)
        ).to(self._dev.device)

        # === Trust Gate ===
        self.trust_gate = nn.Sequential(
            nn.Linear(concept_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 2)  # (mu, log_sigma)
        ).to(self._dev.device)

        # Layer norms
        self.op_norm = nn.LayerNorm(op_dim).to(self._dev.device)
        self.brain_norm = nn.LayerNorm(brain_dim).to(self._dev.device)

        # Temperature for trust gating
        self.trust_temp = trust_temp

    def forward(self,
                brain: torch.Tensor,
                cortex_prior_scales: Optional[Dict[str, float]],
                relmem: Dict[str, torch.Tensor],
                braingraph_emb: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass: fuse priors and generate operation biases + residual nudge.

        Args:
            brain: [B, D] brain latent from TOPAS
            cortex_prior_scales: Dict with {'phi': float, 'kappa': float, 'cge': float}
                                 or None if cortex disabled
            relmem: Dict with:
                - 'concept_emb': [B, concept_dim] concept embeddings
                - 'op_bias': [B, op_dim] operation biases
                - 'confidence': [B] confidence scores
            braingraph_emb: Optional [B, emb_dim] task embedding from BrainGraph

        Returns:
            Dict with:
                - op_prior_logits: [B, op_dim] fused operation priors
                - resid_nudge: [B, D] brain enrichment vector
                - trust_weight: [B, 1] trust scores for logging
        """
        try:
            # === GPU-FIRST device propagation (using registered buffer) ===
            dev = self._dev.device
            brain = brain.to(dev)
            B, D = brain.shape

            concept_emb = relmem['concept_emb'].to(dev)
            op_bias = relmem['op_bias'].to(dev)
            confidence = relmem['confidence'].to(dev)

            if not torch.isfinite(concept_emb).all():
                raise RuntimeError("[SynergyFusion] concept_emb contains NaN/Inf")
            if not torch.isfinite(op_bias).all():
                raise RuntimeError("[SynergyFusion] op_bias contains NaN/Inf")

            # Handle BrainGraph embedding if provided (device-aware)
            if braingraph_emb is not None and torch.is_tensor(braingraph_emb):
                braingraph_emb = braingraph_emb.to(dev)
                # Concatenate with concept embedding
                if braingraph_emb.shape[-1] != concept_emb.shape[-1]:
                    # Project to concept_dim if needed
                    if not hasattr(self, '_graph_proj'):
                        proj = nn.Linear(braingraph_emb.shape[-1], self.concept_dim, device=dev)
                        # Register as submodule so .to(device) propagates
                        self.add_module('_graph_proj', proj)
                    braingraph_emb = self._graph_proj(braingraph_emb)
                # Weighted average with concept_emb
                concept_emb = 0.7 * concept_emb + 0.3 * braingraph_emb

            # === Compute trust weight (stochastic, reparameterized) ===
            trust_params = self.trust_gate(concept_emb)  # [B, 2]
            mu, logsig = trust_params.chunk(2, dim=-1)  # Each [B, 1]
            logsig = logsig.clamp(min=-5.0, max=1.5)  # Prevent exponential overflow

            # Reparameterization trick for smooth gradients
            if self.training:
                eps = torch.randn_like(mu)  # Inherits mu's device
                sigma = torch.exp(logsig).clamp(1e-5, 10.0) * self.trust_temp  # Prevent exp() overflow
                trust_logit = mu + eps * sigma
            else:
                # Deterministic at eval
                trust_logit = mu

            trust_weight = torch.sigmoid(trust_logit)  # [B, 1]

            # Modulate trust by RelMem confidence (soft gating)
            trust_weight = trust_weight * confidence.unsqueeze(-1).clamp(0, 1)

            # === Fuse features ===
            fused = torch.cat([brain, concept_emb], dim=-1)  # [B, D + concept_dim]

            # === Generate operation priors ===
            op_logits_learned = self.op_proj_ctx(fused)  # [B, op_dim]

            # Add Cortex prior scales if available
            if cortex_prior_scales is not None:
                # Cortex provides scalar scales for φ, κ, CGE
                # Use as global multipliers (broadcast across ops)
                phi_scale = cortex_prior_scales.get('phi', 1.0)
                kappa_scale = cortex_prior_scales.get('kappa', 1.0)
                cge_scale = cortex_prior_scales.get('cge', 1.0)

                # Simple weighted average of scales
                cortex_mult = (phi_scale + kappa_scale + cge_scale) / 3.0
                op_logits_learned = op_logits_learned * cortex_mult

            # Fuse: learned logits + RelMem bias (both contribute)
            op_prior_logits = self.op_norm(op_logits_learned + 0.35 * op_bias)
            assert torch.isfinite(op_prior_logits).all(), "[SynergyFusion] op_prior_logits has NaN/Inf"

            # === Generate residual nudge ===
            resid_nudge = self.brain_norm(self.resid_proj(fused))  # [B, D]
            assert torch.isfinite(resid_nudge).all(), "[SynergyFusion] resid_nudge has NaN/Inf"

            # === Trust-weighted blending ===
            # High trust → use fusion; Low trust → fallback to RelMem only
            # Stop gradient on trust to prevent collapse
            trust_detached = trust_weight.detach()

            op_prior_logits = (
                trust_detached * op_prior_logits +
                (1 - trust_detached) * op_bias
            )

            resid_nudge = trust_weight * resid_nudge  # Scale residual by trust

            assert torch.isfinite(op_prior_logits).all(), "[SynergyFusion] Final op_prior has NaN/Inf"
            assert torch.isfinite(resid_nudge).all(), "[SynergyFusion] Final resid_nudge has NaN/Inf"

            return {
                'op_prior_logits': op_prior_logits,  # [B, op_dim]
                'resid_nudge': resid_nudge,          # [B, D]
                'trust_weight': trust_weight.squeeze(-1),  # [B]
                'confidence': confidence                    # [B] (passthrough for logging)
            }

        except Exception as e:
            logger.warning(f"[SynergyFusion] Failed: {e}")
            # Safe fallback: return zeros on module device
            dev = self._dev.device
            return {
                'op_prior_logits': torch.zeros(B, self.op_dim, device=dev),
                'resid_nudge': torch.zeros(B, self.brain_dim, device=dev),
                'trust_weight': torch.zeros(B, device=dev),
                'confidence': torch.zeros(B, device=dev)
            }

    def get_stats(self) -> Dict[str, float]:
        """Return module statistics for logging/debugging."""
        stats = {}

        # Parameter norms
        for name, param in self.named_parameters():
            if param.requires_grad:
                stats[f'synergy_{name}_norm'] = float(param.data.norm().item())

        # Gradient norms (if available)
        if self.training:
            for name, param in self.named_parameters():
                if param.requires_grad and param.grad is not None:
                    stats[f'synergy_{name}_grad_norm'] = float(param.grad.norm().item())

        return stats
