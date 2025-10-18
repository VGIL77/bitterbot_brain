"""
TTT (Test-Time Training) Methods for TOPAS

These methods should be added to the TopasARC60M class to enable
LoRA-based test-time adaptation.
"""

def _discover_ttt_targets(self):
    """
    Auto-discover small projection heads/attention layers for LoRA wrapping

    Returns:
        List of module name suffixes to target
    """
    targets = []

    # Small projection heads (good candidates for adaptation)
    for name, module in self.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Target small projections (< 2048 params)
            if module.in_features * module.out_features < 2048:
                suffix = name.split('.')[-1]
                if suffix not in targets:
                    targets.append(suffix)

        # Also target attention query/key/value projections
        if 'attn' in name.lower() and any(x in name.lower() for x in ['q_proj', 'k_proj', 'v_proj', 'out_proj']):
            suffix = name.split('.')[-1]
            if suffix not in targets:
                targets.append(suffix)

    return targets[:24]  # Limit to 24 modules for efficiency


def _ensure_ttt_ready(self):
    """
    Lazy initialization of TTT LoRA wrappers

    Call this before first adaptation to attach LoRA layers
    """
    if hasattr(self, '_ttt_ready') and self._ttt_ready:
        return  # Already initialized

    from models.ttt_lora import attach_lora

    # Discover or use pre-configured targets
    if not hasattr(self, '_ttt_targets'):
        self._ttt_targets = self._discover_ttt_targets()

    # Attach LoRA wrappers
    self._ttt_wrapped = attach_lora(
        self,
        targets=self._ttt_targets,
        r=self.config.ttt_r,
        alpha=self.config.ttt_alpha,
        lr_ratio=self.config.ttt_lr_ratio
    )

    self._ttt_ready = True

    import logging
    logging.info(f"[TTT] Initialized: {len(self._ttt_wrapped)} modules wrapped with LoRA")


def _freeze_all_but_lora(self):
    """Freeze base model, enable only LoRA parameters"""
    # Freeze everything
    for param in self.parameters():
        param.requires_grad = False

    # Unfreeze LoRA adapters
    if hasattr(self, '_ttt_wrapped'):
        for wrapper in self._ttt_wrapped.values():
            for param in wrapper.parameters():
                if 'A.weight' in str(param) or 'B.weight' in str(param):
                    param.requires_grad = True


def _ttt_reset(self):
    """Reset LoRA weights to zero (B) and random (A)"""
    if not hasattr(self, '_ttt_wrapped'):
        return

    from models.ttt_lora import LoRALinear, LoRAConv1x1
    import math

    for wrapper in self._ttt_wrapped.values():
        if isinstance(wrapper, (LoRALinear, LoRAConv1x1)):
            # Reset B to zeros, A to kaiming
            torch.nn.init.zeros_(wrapper.B.weight)
            torch.nn.init.kaiming_uniform_(wrapper.A.weight, a=math.sqrt(5))


def _ttt_adapt(self, demos, device):
    """
    Adapt model to task using demo pairs (TTT)

    Args:
        demos: List of (input_grid, output_grid) pairs
        device: torch device

    Returns:
        dict with adaptation stats
    """
    if not self.config.ttt_enable or not demos:
        return {}

    # Ensure LoRA ready
    self._ensure_ttt_ready()

    # Reset LoRA weights for fresh adaptation
    self._ttt_reset()

    # Freeze base, enable LoRA
    self._freeze_all_but_lora()

    # Create optimizer for LoRA params only
    lora_params = []
    for wrapper in self._ttt_wrapped.values():
        lora_params.extend(wrapper.parameters())

    optimizer = torch.optim.Adam(lora_params, lr=self.config.ttt_lr)

    # Adaptation loop
    stats = {'steps': 0, 'loss': 0.0}

    with torch.enable_grad():  # Enable gradients for adaptation
        for step in range(self.config.ttt_steps):
            # Sample random demo
            demo_input, demo_output = demos[step % len(demos)]

            # Ensure tensors
            if not torch.is_tensor(demo_input):
                demo_input = torch.as_tensor(demo_input, device=device)
            if not torch.is_tensor(demo_output):
                demo_output = torch.as_tensor(demo_output, device=device)

            # Add batch dim
            if demo_input.dim() == 2:
                demo_input = demo_input.unsqueeze(0)
            if demo_output.dim() == 2:
                demo_output = demo_output.unsqueeze(0)

            # Forward pass
            outputs = self.forward_pretraining(demo_input, eval_mode=True)
            logits = outputs['logits']

            # Compute loss against demo output
            H, W = demo_output.shape[-2:]
            target = demo_output.reshape(-1).long()
            pred_logits = logits.reshape(-1, logits.shape[-1])[:target.shape[0]]

            loss = torch.nn.functional.cross_entropy(pred_logits, target)

            # Backward and update LoRA
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()

            stats['steps'] += 1
            stats['loss'] += loss.item()

    stats['loss'] /= stats['steps']

    # Unfreeze all for normal operation
    for param in self.parameters():
        param.requires_grad = True

    import logging
    logging.info(f"[TTT] Adapted on {len(demos)} demos: {stats['steps']} steps, loss={stats['loss']:.4f}")

    return stats
