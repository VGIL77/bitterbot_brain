#!/usr/bin/env python3
"""
Cortex Configuration for Andromeda Cortical Sheet
Hyperparameters for predictive coding MoE layer
"""
from dataclasses import dataclass


@dataclass
class CortexConfig:
    """
    Configuration for the Andromeda Cortical Sheet.

    Args:
        state_dim: Dimension of the "brain" latent fed by TOPAS (ctrl_dim in TopasARC60M)
        columns: Number of cortical columns (experts) used in the mixture
        column_dim: Internal hidden dimension per column
        depth: Number of residual predictive coding blocks per column
        pred_steps: Number of prediction steps for temporal PC (set to 1 for static tokens)
        sparsity_beta: KL sparsity strength for latent code
        entropy_beta: Weight for entropy flux regularization
        gating_temp: Softmax temperature for MoE gating
        prior_scale_max: Maximum multiplicative scale applied to EBR priors from cortex

    Neuroscience inspiration:
        - columns: Cortical columns (Mountcastle, 1978)
        - PC blocks: Predictive coding (Rao & Ballard, 1999; Friston, 2010)
        - Sparsity: Sparse coding in V1 (Olshausen & Field, 1996)
        - MoE gating: Dendritic routing (London & Häusser, 2005)
    """
    state_dim: int
    columns: int = 8
    column_dim: int = 256
    depth: int = 2
    pred_steps: int = 1
    sparsity_beta: float = 1e-3
    entropy_beta: float = 5e-4
    gating_temp: float = 0.7
    prior_scale_max: float = 1.5
