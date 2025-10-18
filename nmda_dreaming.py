#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import math
from dataclasses import dataclass
import logging

@dataclass
class Experience:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool = False
    valence: float = 0.7
    arousal: float = 0.5
    timestamp: int = 0
    phase: float = 0.0

class NMDAGatedDreaming(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, g_max: float = 1.0, mg_conc: float = 1.0, valence_thresh: float = 0.5, learning_rate: float = 0.0005, gamma: float = 0.99, tau: float = 0.005, buffer_size: int = 30000, batch_size: int = 128, device: str = "cuda:0", phase_lock: bool = True):
        super().__init__()
        self.device = torch.device(device)
        # ensure dims are stored for later checks
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)
        self.target_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.g_max = g_max
        self.mg_conc = mg_conc
        self.valence_thresh = valence_thresh
        self.phase_lock = phase_lock

        # strict mode for NMDA paths (fail fast instead of silent skip)
        self.STRICT_NMDA = True
        self.logger = logging.getLogger(__name__)

    def _ensure(self, cond: bool, msg: str):
        if not cond:
            raise RuntimeError(msg)

    # --- Readiness gate ---
    def is_ready(self) -> bool:
        """Ready when we have enough real experiences to form a batch."""
        need = getattr(self, "min_batch_size", self.batch_size)
        return len(self.buffer) >= need

    def compute_conductance(self, voltage: float, synaptic_act: float = 1.0) -> float:
        B = 1 / (1 + self.mg_conc * math.exp(-0.062 * voltage) / 3.57)
        return self.g_max * B * synaptic_act

    def compute_gate(self, valence: float, arousal: float) -> float:
        self._ensure(0.0 <= valence <= 1.0, f"NMDA.compute_gate: valence out of range {valence}")
        self._ensure(0.0 <= arousal <= 1.0, f"NMDA.compute_gate: arousal out of range {arousal}")
        voltage = arousal * 40 - 20
        conductance = self.compute_conductance(voltage)
        gate = 1 / (1 + math.exp(-10 * (valence - self.valence_thresh)))
        return gate * conductance
    
    def wrap_dist(self, phi: float, peak: float = 0.0) -> float:
        """
        Compute circular distance between two phase values.
        
        Args:
            phi: Phase value to compare
            peak: Peak phase value (default 0.0)
            
        Returns:
            float: Wrap-around aware distance on circle
        """
        d = abs((phi - peak + math.pi) % (2*math.pi) - math.pi)
        return d

    def store_dream_memory(self, state, action, reward, next_state, phase=0.0):
        self._ensure(torch.is_tensor(state), "NMDA.store: state must be tensor")
        self._ensure(torch.is_tensor(next_state), "NMDA.store: next_state must be tensor")
        state = state.detach().to(self.device)
        next_state = next_state.detach().to(self.device)
        self._ensure(state.ndim == 1 and next_state.ndim == 1,
                     f"NMDA.store: states must be 1D vectors; got {state.shape}, {next_state.shape}")
        self._ensure(state.shape[-1] == self.state_dim and next_state.shape[-1] == self.state_dim,
                     f"NMDA.store: expected dim={self.state_dim}, got {state.shape[-1]}/{next_state.shape[-1]}")

        # Bounds check action to prevent CUDA device assertions
        a = int(action)
        if a < 0 or a >= self.action_dim:
            msg = f"[NMDA][STORE] action {a} out of bounds [0,{self.action_dim-1}]; clamping"
            if hasattr(self, 'logger') and self.logger:
                self.logger.warning(msg)
            a = max(0, min(a, self.action_dim - 1))

        self.buffer.append(Experience(state, a, reward, next_state, phase=phase))

    def emit_signals(self, ctx=None):
        """Emit structured signals about NMDA rail status"""
        if ctx is not None:
            signals = ctx.setdefault('signals', {})
            metrics = ctx.setdefault('metrics', {})

            # Buffer status
            signals['nmda_ready'] = self.is_ready()
            metrics['nmda_buffer_level'] = len(self.buffer)
            metrics['nmda_min_threshold'] = getattr(self, "min_batch_size", self.batch_size)

            # Track consolidation events
            if hasattr(self, '_consolidation_count'):
                metrics['nmda_consolidations'] = self._consolidation_count
            else:
                metrics['nmda_consolidations'] = 0

            # Q-value statistics (from last consolidation)
            if hasattr(self, '_last_q_stats'):
                metrics['nmda_q_mean'] = self._last_q_stats.get('mean', 0.0)
                metrics['nmda_q_min'] = self._last_q_stats.get('min', 0.0)
                metrics['nmda_q_max'] = self._last_q_stats.get('max', 0.0)
                metrics['nmda_q_var'] = self._last_q_stats.get('var', 0.0)

            # Action distribution (from last consolidation)
            if hasattr(self, '_last_action_counts'):
                metrics['nmda_action_counts'] = self._last_action_counts

            # Gate value (from last consolidation)
            if hasattr(self, '_last_gate_value'):
                metrics['nmda_gate_value'] = self._last_gate_value

    def dream_consolidation(self, valence: float, arousal: float, curiosity_module, ripple_ctx=None, ctx=None, backprop: bool = True):
        """
        NMDA-gated dream consolidation with ripple-aware amplification.

        Args:
            valence: Emotional valence (0-1)
            arousal: Arousal level (0-1)
            ripple_ctx: Optional ripple context with gain multiplier
            curiosity_module: REQUIRED curiosity module for priority-driven consolidation

        Returns:
            float: Loss value after consolidation
        """
        if not self.is_ready():
            raise RuntimeError(f"[NMDA] inactive: buffer {len(self.buffer)} below threshold")
            
        # Apply phase-locked replay ordering if ripple context provided
        batch = self.sample_batch(ripple_ctx) if ripple_ctx else random.sample(self.buffer, self.batch_size)
        
        states, actions, rewards, next_states = zip(*[(e.state, e.action, e.reward, e.next_state) for e in batch])
        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device)

        # Compute NMDA gate
        gate = self.compute_gate(valence, arousal)
        if gate < 0.1:
            raise RuntimeError(f"[NMDA] gate too low ({gate:.3f}) for consolidation")

        # Bounds check actions before gather to prevent CUDA assertions
        if (actions < 0).any() or (actions >= self.action_dim).any():
            bad_vals = actions[(actions < 0) | (actions >= self.action_dim)][:16].tolist()
            raise RuntimeError(f"[NMDA] sampled actions out of bounds min={int(actions.min())} "
                             f"max={int(actions.max())} action_dim={self.action_dim} examples={bad_vals}")

        # Q-network forward (keep autograd for q_values)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Detach only the target branch
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q

        # MSE loss with graph preserved through q_values
        loss_per_example = F.mse_loss(q_values, target_q, reduction='none')

        # STRICT: no fallbacks, curiosity weighting is mandatory
        self._ensure(curiosity_module is not None, "[NMDA] curiosity_module is required")
        # Expect curiosity_module to provide a stable score_weights API
        if not hasattr(curiosity_module, "score_states"):
            error_msg = f"[NMDA] curiosity_module {type(curiosity_module)} must implement score_states(states, loss_per_example) -> (weights_tensor, info_dict)"
            raise RuntimeError(error_msg)

        # Ask curiosity module for per-example weights (fail-fast if it misbehaves)
        try:
            weights, info = curiosity_module.score_states(states=states, loss_per_example=loss_per_example)
        except Exception as e:
            raise RuntimeError(f"[NMDA] curiosity_module.score_states failed: {e}") from e
        # Validate weights
        if not torch.is_tensor(weights) or weights.shape[0] != loss_per_example.shape[0]:
            raise RuntimeError("[NMDA] curiosity_module.score_states returned invalid weights")

        # Compute weighted loss (mean of weighted per-example losses)
        # Clip weights for numeric stability
        weights_clipped = torch.clamp(weights.to(loss_per_example.device), min=0.0, max=10.0)
        weighted_loss = (loss_per_example * weights_clipped).mean()
        loss = weighted_loss

        # Apply ripple-aware gain with biological consistency
        # Both NMDA gate and ripple must be active for maximum effect
        if ripple_ctx and hasattr(ripple_ctx, 'gain') and ripple_ctx.gain > 1.0:
            # Combine ripple gain with NMDA gating
            # mult = 1.0 + (ripple_gain - 1.0) * nmda_gate
            # This ensures ripple only amplifies when NMDA allows plasticity
            ripple_mult = 1.0 + (ripple_ctx.gain - 1.0) * gate
            loss = loss * gate * ripple_mult
        else:
            # Standard NMDA gating without ripple amplification
            loss = weighted_loss * gate

        # Guard against insane loss spikes
        if not torch.isfinite(loss):
            raise RuntimeError(f"[NMDA] Non-finite loss detected: {loss}")

        # Backprop properly — gradients flow into q_net
        # Only backprop if requested (micro-ticks are read-only, full cycles learn)
        if backprop:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
            self.optimizer.step()

            # Soft update target network
            for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Track metrics for monitoring
        if not hasattr(self, '_consolidation_count'):
            self._consolidation_count = 0
        self._consolidation_count += 1

        # Q-value statistics
        with torch.no_grad():
            self._last_q_stats = {
                'mean': float(q_values.mean().item()),
                'min': float(q_values.min().item()),
                'max': float(q_values.max().item()),
                'var': float(q_values.var().item())
            }

        # Action histogram
        action_counts = {}
        for a in actions.cpu().tolist():
            action_counts[a] = action_counts.get(a, 0) + 1
        self._last_action_counts = action_counts

        # Gate value
        self._last_gate_value = float(gate)

        self._ensure(torch.isfinite(loss), "[NMDA] non-finite consolidation loss")
        return float(loss.item())
        
    def sample_batch(self, ripple_ctx=None):
        """
        Sample a batch of experiences with optional phase-locked ordering.
        
        During ripple bursts, order batch by phase distance to peak for
        biologically plausible replay sequences.
        
        Args:
            ripple_ctx: Optional ripple context with active flag and phase_peak
            
        Returns:
            list: Batch of experiences, potentially ordered by phase
        """
        if len(self.buffer) < self.batch_size:
            if self.STRICT_NMDA:
                raise RuntimeError(f"[NMDA] sample_batch shortfall: {len(self.buffer)} < {self.batch_size}")
            return list(self.buffer)
            
        # Random sampling by default
        batch = random.sample(self.buffer, self.batch_size)
        
        # Apply phase-locked ordering during active ripples
        if (ripple_ctx and 
            hasattr(ripple_ctx, 'active') and ripple_ctx.active and 
            self.phase_lock and
            hasattr(ripple_ctx, 'phase_peak')):
            
            # Quick check if phases are diverse enough to benefit from sorting
            phases = []
            for e in batch:
                if not hasattr(e, 'phase'):
                    raise RuntimeError("[NMDA] Experience missing required 'phase' attribute")
                phases.append(e.phase)
            phase_std = torch.std(torch.tensor(phases))
            
            # Only sort if there's meaningful phase diversity (> 0.5 radians std)
            if phase_std > 0.5:
                if not hasattr(ripple_ctx, 'phase_peak'):
                    raise RuntimeError("[NMDA] ripple_ctx missing required 'phase_peak' attribute")
                phase_peak = ripple_ctx.phase_peak
                batch.sort(key=lambda e: self.wrap_dist(e.phase, phase_peak))
                
                # Log only occasionally to avoid spam
                if random.random() < 0.1:  # Log 10% of the time
                    self.logger.debug(f"[NMDA] phase-locked replay reordered (N={len(batch)})")
            
        return batch
    
    def to(self, device):
        """Move the module and buffer contents to the specified device."""
        # Call parent class to() to move nn.Module parameters
        super().to(device)
        self.device = torch.device(device)
        
        # Move buffer experiences to the new device
        new_buffer = deque(maxlen=self.buffer.maxlen)
        for exp in self.buffer:
            # Create new Experience with tensors on the new device
            new_exp = Experience(
                state=exp.state.to(device) if torch.is_tensor(exp.state) else exp.state,
                action=exp.action,
                reward=exp.reward,
                next_state=exp.next_state.to(device) if torch.is_tensor(exp.next_state) else exp.next_state,
                done=exp.done,
                valence=exp.valence,
                arousal=exp.arousal,
                timestamp=exp.timestamp,
                phase=exp.phase
            )
            new_buffer.append(new_exp)
        self.buffer = new_buffer
        
        return self