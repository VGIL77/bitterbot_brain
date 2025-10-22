"""
DSLHead: lightweight symbolic policy/value head that reuses PolicyNet + ValueNet encoders.
Shares feature backbone to preserve latent alignment.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class DSLHead(nn.Module):
    """
    DSLHead: lightweight symbolic policy/value head that reuses PolicyNet + ValueNet encoders.
    Shares feature backbone to preserve latent alignment.
    """

    def __init__(self, policy_net, value_net, vocab_size=128):
        super().__init__()
        # Reuse existing neural features
        self.feature_extractor = policy_net.feature_extractor \
            if hasattr(policy_net, "feature_extractor") else (
                policy_net.encoder if hasattr(policy_net, "encoder") else policy_net
            )

        # Small adapter if needed
        in_dim = getattr(policy_net, "hidden_dim", 512)
        self.policy_head = nn.Linear(in_dim, vocab_size)
        self.value_head = nn.Linear(in_dim, 1)
        self.vocab_size = vocab_size

        # Freeze shared encoder by default to preserve trained features
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

        logger.info(f"[DSLHead] Initialized with vocab_size={vocab_size}, in_dim={in_dim}")

    def forward(self, x):
        """
        Forward pass for policy/value prediction

        Args:
            x: Input features [B, ...] or grid [B, H, W]

        Returns:
            dict with 'policy' (probs) and 'value' (scalar estimate)
        """
        feats = self.feature_extractor(x)
        logits = self.policy_head(feats)
        probs = F.softmax(logits, dim=-1)
        value = torch.tanh(self.value_head(feats))
        return {"policy": probs, "value": value}

    def apply_program(self, grid: torch.Tensor, program) -> torch.Tensor:
        """
        Apply DSL program to grid (delegates to dsl_search.apply_program)

        Args:
            grid: Input grid tensor
            program: DSLProgram or list of (op, params) tuples

        Returns:
            Transformed grid
        """
        from models.dsl_search import apply_program
        return apply_program(grid, program)

    @property
    def ops(self) -> Dict[str, Any]:
        """
        Provide ops dictionary for direct operation access
        Returns dict mapping op_name -> callable
        """
        from models.dsl_search import (
            _rotate90, _rotate180, _rotate270, _flip_h, _flip_v,
            _color_map, _crop_bbox, _flood_fill, _outline, _translate,
            _scale, _tile, _resize_nn, _identity
        )

        # Return operation callables
        ops_dict = {
            'rotate90': lambda g, **kw: torch.rot90(g, k=-1, dims=(0, 1)),
            'rotate180': lambda g, **kw: torch.rot90(g, k=2, dims=(0, 1)),
            'rotate270': lambda g, **kw: torch.rot90(g, k=1, dims=(0, 1)),
            'flip_h': lambda g, **kw: torch.flip(g, dims=[1]),
            'flip_v': lambda g, **kw: torch.flip(g, dims=[0]),
            'identity': lambda g, **kw: g.clone(),
            # Add more ops as needed - can delegate to apply_program for complex ones
        }

        return ops_dict
