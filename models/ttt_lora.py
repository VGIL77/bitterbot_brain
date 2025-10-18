# models/ttt_lora.py
import math
import torch
import torch.nn as nn
from typing import List, Dict

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, lr_ratio=1.0):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r
        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.lr_ratio = float(lr_ratio)  # LoRA+ (different LR for A/B)

    def forward(self, x):
        return self.base(x) + self.B(self.A(x)) * self.scale

    # Expose base module attributes for compatibility with nn.MultiheadAttention
    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def parameters_grouped(self, base_lr):
        return [
            {'params': self.A.parameters(), 'lr': base_lr * self.lr_ratio},
            {'params': self.B.parameters(), 'lr': base_lr},
        ]

class LoRAConv1x1(nn.Module):
    def __init__(self, base: nn.Conv2d, r=8, alpha=16, lr_ratio=1.0):
        super().__init__()
        assert base.kernel_size == (1,1)
        self.base = base
        self.r, self.alpha = r, alpha
        self.scale = alpha / r
        self.A = nn.Conv2d(base.in_channels, r, kernel_size=1, bias=False)
        self.B = nn.Conv2d(r, base.out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.lr_ratio = float(lr_ratio)

    def forward(self, x):
        return self.base(x) + self.B(self.A(x)) * self.scale

    # Expose base module attributes for compatibility
    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_channels(self):
        return self.base.in_channels

    @property
    def out_channels(self):
        return self.base.out_channels

    @property
    def kernel_size(self):
        return self.base.kernel_size

    def parameters_grouped(self, base_lr):
        return [
            {'params': self.A.parameters(), 'lr': base_lr * self.lr_ratio},
            {'params': self.B.parameters(), 'lr': base_lr},
        ]

def attach_lora(model, targets: List[str], r=8, alpha=16, lr_ratio=1.0) -> Dict[str, nn.Module]:
    """Wrap named modules with LoRA adapters; returns mapping name->wrapper."""
    wrapped = {}

    # First pass: collect all modules to wrap (avoid iterator corruption)
    to_wrap = []
    for name, module in list(model.named_modules()):
        # Skip if already wrapped
        if isinstance(module, (LoRALinear, LoRAConv1x1)):
            continue

        # Skip recursion-prone or heavy modules
        if any(skip in name for skip in [
            "dream_engine", "relmem", "forward_pretraining",
            "evaluate_with_", "cycle_offline", "dream"
        ]):
            continue

        if any(name.endswith(t) for t in targets):
            if isinstance(module, nn.Linear):
                to_wrap.append((name, module, 'linear'))
            elif isinstance(module, nn.Conv2d) and module.kernel_size == (1,1):
                to_wrap.append((name, module, 'conv1x1'))

    # Second pass: create wrappers and rebind
    for name, module, mod_type in to_wrap:
        if mod_type == 'linear':
            wrapper = LoRALinear(module, r=r, alpha=alpha, lr_ratio=lr_ratio)
        else:  # conv1x1
            wrapper = LoRAConv1x1(module, r=r, alpha=alpha, lr_ratio=lr_ratio)

        # Move wrapper to same device as base module
        if hasattr(module, 'weight') and module.weight is not None:
            wrapper = wrapper.to(module.weight.device)

        # Find parent and rebind (use name split to get parent path)
        parts = name.split('.')
        if len(parts) == 1:
            # Top-level module
            setattr(model, parts[0], wrapper)
        else:
            # Navigate to parent
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], wrapper)

        wrapped[name] = wrapper

    return wrapped
