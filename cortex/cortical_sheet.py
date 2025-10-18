#!/usr/bin/env python3
"""
Andromeda Cortical Sheet - Predictive Coding MoE Layer

A minimal-yet-powerful "cortex rail" that provides:
1. Residual enrichment of TOPAS brain latent
2. DSL operation bias for symbolic search
3. EBR prior scale hints (φ/κ/CGE weighting)

Architecture:
- Mixture of predictive-coding columns (experts)
- Sparse gating via softmax + top-k mask
- Three differentiable output heads

Neuroscience inspiration:
- Cortical columns (Mountcastle, 1978)
- Predictive coding (Rao & Ballard, 1999; Friston, 2010)
- Sparse coding (Olshausen & Field, 1996)
- Dendritic computation (London & Häusser, 2005)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .cortex_config import CortexConfig
from .predictive_coder import PredictiveCoder
from .entropy_flux import kl_sparsity


class CorticalSheet(nn.Module):
    """
    Andromeda Cortical Sheet: Sparse MoE of predictive coding columns.

    Produces three signals:
    1. **Residual:** [B, D] vector to enrich TOPAS 'brain' latent
    2. **Op bias:** Dict[str, float] - DSL operation priors
    3. **Prior scales:** Dict[str, float] - EBR weighting hints {φ, κ, CGE}

    Training-first design: All operations differentiable for joint training with TOPAS.
    """
    def __init__(self, cfg: CortexConfig, dsl_vocab_size: int = 41):
        super().__init__()
        self.cfg = cfg
        D = cfg.state_dim

        # === Predictive Coding Columns (Experts) ===
        self.columns = nn.ModuleList([
            PredictiveCoder(D, cfg.column_dim, cfg.depth)
            for _ in range(cfg.columns)
        ])

        # === Gating Network (Dendritic Routing) ===
        self.gate = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, cfg.columns)
        )

        # === Output Heads ===

        # Operation bias head: Maps latent → DSL operation logits
        self.op_head = nn.Sequential(
            nn.LayerNorm(cfg.column_dim),
            nn.Linear(cfg.column_dim, 128),
            nn.GELU(),
            nn.Linear(128, dsl_vocab_size)
        )

        # Prior scale head: Maps latent → EBR weighting hints
        # Outputs [φ, κ, CGE] scales in range [1/max, max]
        self.prior_head = nn.Sequential(
            nn.LayerNorm(cfg.column_dim),
            nn.Linear(cfg.column_dim, 16),
            nn.GELU(),
            nn.Linear(16, 3)  # 3 scale hints for [phi, kappa, cge]
        )

    def forward(self, tokens: torch.Tensor, extras: Optional[Dict] = None) -> Dict[str, object]:
        """
        Forward pass through cortical sheet.

        Args:
            tokens: [B, T, D] input tokens (T can be 1)
                   Typically the TOPAS brain latent unsqueezed to [B, 1, D]
            extras: Optional dict for passing context (unused currently)

        Returns:
            Dict with:
                residual: [B, D] - Vector to add to brain latent
                op_bias: Dict[str, float] - DSL operation priors (index→score)
                prior_scales: Dict[str, float] - EBR scale hints {phi, kappa, cge}
                losses: Dict[str, Tensor] - Training losses {recon, entropy, sparsity}

        Architecture:
            1. Gate network selects top-k columns (sparse routing)
            2. Run all columns in parallel (efficient batching)
            3. Weighted combination using gate probs
            4. Output heads produce signals
        """
        assert tokens.dim() == 3, f"Expected [B,T,D], got {tokens.shape}"
        B, T, D = tokens.shape

        # === GATING (Dendritic Routing) ===
        # Use mean-pooled token for gating decision
        x = tokens.mean(dim=1)  # [B, D]
        logits = self.gate(x) / max(1e-4, float(self.cfg.gating_temp))

        # Sparse-ish selection via top-k mask (k ~ sqrt(columns))
        # Keep gradients via softmax, but bias toward sparse routing
        k = max(1, int(self.cfg.columns ** 0.5))
        topk = torch.topk(logits, k=k, dim=-1).indices
        mask = torch.zeros_like(logits).scatter(1, topk, 1.0)
        gate_probs = torch.softmax(logits, dim=-1) * mask
        gate_probs = gate_probs / gate_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)  # [B, C]

        # === RUN ALL COLUMNS (Parallel Execution) ===
        col_out = []
        col_losses = {"recon": [], "entropy": []}

        for idx, col in enumerate(self.columns):
            out = col(tokens)  # [B, T, D] → {residual, z, losses}
            col_out.append(out)
            col_losses["recon"].append(out["losses"]["recon"])
            col_losses["entropy"].append(out["losses"]["entropy"])

        # === WEIGHTED COMBINATION (Gated Mixing) ===
        # Stack residuals and latents for gating
        residuals = torch.stack([o["residual"] for o in col_out], dim=1)  # [B, C, D]
        latents = torch.stack([o["z"] for o in col_out], dim=1)           # [B, C, H]

        # Weighted combine using gate probabilities
        gate_probs_exp = gate_probs.unsqueeze(-1)
        residual = (gate_probs_exp * residuals).sum(dim=1)  # [B, D]
        latent = (gate_probs_exp * latents).sum(dim=1)      # [B, H]

        # === OUTPUT HEADS ===

        # 1. DSL Operation Bias
        op_logits = self.op_head(latent)  # [B, V]
        op_probs = torch.softmax(op_logits, dim=-1)  # [B, V]

        # 2. EBR Prior Scales
        prior_raw = torch.sigmoid(self.prior_head(latent))  # [B, 3] in (0,1)
        # Scale from [1/max, max] range
        scale_min = 1.0 / float(self.cfg.prior_scale_max)
        scale_max = float(self.cfg.prior_scale_max)
        prior_scales = scale_min + (scale_max - scale_min) * prior_raw

        # Assemble dict with names (mean across batch for simplicity)
        prior_dict = {
            "phi": float(prior_scales[:, 0].mean().item()),
            "kappa": float(prior_scales[:, 1].mean().item()),
            "cge": float(prior_scales[:, 2].mean().item())
        }

        # === TRAINING LOSSES ===
        # - Mixture reconstruction + entropy flux + latent sparsity
        losses = {
            "pc_recon": torch.stack(col_losses["recon"]).mean(),
            "pc_entropy": torch.stack(col_losses["entropy"]).mean(),
            "kl_sparsity": kl_sparsity(latent, target=0.05)
        }

        # === EXPORT OP BIAS ===
        # Export as python dict (keep tensors for potential backprop)
        # Index-based for now (will be mapped to DSL op names upstream)
        if B == 1:
            op_bias = {str(i): float(op_probs[0, i].item()) for i in range(op_probs.size(-1))}
        else:
            # Batch mode: return mean probs
            op_bias = {str(i): float(op_probs[:, i].mean().item()) for i in range(op_probs.size(-1))}

        return {
            "residual": residual,          # [B, D]
            "op_bias": op_bias,            # Dict[str, float] (index-based)
            "prior_scales": prior_dict,    # Dict[str, float] {phi, kappa, cge}
            "losses": losses,              # Dict[str, Tensor]
            "gate_probs": gate_probs       # [B, C] - Which columns activated
        }
