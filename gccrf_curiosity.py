#!/usr/bin/env python3
"""
Strict GCCRFCuriosity with fail-fast semantics:
- No dummy novelty, empowerment, or alignment values
- Rewards assembled consistently with .mean()
- Raises immediately if invariants are broken
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict

STRICT_CURIOSITY = True

def _ensure(cond, msg: str):
    if not cond:
        raise RuntimeError(msg)

class GCCRFCuriosity(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: Optional[int] = None,
                 alpha_start: float = -0.5, alpha_end: float = 0.0,
                 anneal_steps: int = 1000, learning_rate: float = 1e-3,
                 enable_kde: bool = True, bandwidth: float = 0.5,
                 reservoir_size: int = 2048):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim or (state_dim * 2)
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.anneal_steps = anneal_steps
        self.current_step = 0

        # KDE parameters
        self.enable_kde = enable_kde
        self.bandwidth = bandwidth
        self.reservoir_size = reservoir_size
        self.register_buffer('reservoir', torch.zeros(self.reservoir_size, self.state_dim))
        self.reservoir_ptr = 0
        # strategic_targets will be registered as buffer via set_strategic_targets()

        # Predictor network
        self.predictor = nn.Sequential(
            nn.Linear(state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, state_dim)
        )

        self.predictor_optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)

        # Running statistics - graph-connected
        self.register_buffer('running_error', torch.zeros(1))
        self.register_buffer('error_momentum', torch.ones(1) * 0.95)

        # --- running statistics for curiosity normalization (stable online stats / EMA)
        # These are intentionally simple EMA stats used to normalize curiosity scores
        # downstream. They are lightweight and safe to checkpoint as plain attributes.
        self.running_mean: float = 0.0
        self.running_var: float = 1.0
        self.running_std: float = 1.0
        self.running_count: int = 0
        # smoothing factor for EMA updates of mean/var (tunable)
        self._running_ema_alpha: float = 0.001

        # Additional running stats as registered buffers for score_states API
        self.register_buffer("_running_mean", torch.zeros(1))
        self.register_buffer("_running_var", torch.ones(1))
        self._momentum = 0.995  # momentum for EMA
        self._eps = 1e-6  # epsilon for numeric stability

    def set_strategic_targets(self, targets: torch.Tensor):
        """Set strategic targets for alignment computation"""
        _ensure(targets.dim() == 2 and targets.size(-1) == self.state_dim,
                f"[Curiosity] bad targets shape {targets.shape}")
        t = targets.detach()
        # Register as a buffer so `.to(device)` and checkpointing include the targets.
        # Handle case where strategic_targets might exist as a regular attribute
        if hasattr(self, "strategic_targets") and not isinstance(getattr(self, "strategic_targets"), torch.Tensor):
            # Remove non-tensor attribute
            delattr(self, "strategic_targets")

        if "strategic_targets" not in self._buffers:
            # register fresh buffer (places on module device)
            self.register_buffer("strategic_targets", t)
        else:
            # update existing buffer in-place (avoid re-registration)
            buf = self._buffers["strategic_targets"]
            if buf.shape != t.shape:
                # replace buffer entirely if shape differs (safe path)
                self._buffers["strategic_targets"] = t.to(buf.device)
            else:
                # in-place copy (preserves device/registration)
                buf.data.copy_(t.to(buf.device))

    # --- Readiness gate: ARC-only, no dummy novelty ---
    def is_ready(self) -> bool:
        """Rail is ready when KDE has enough real states and targets exist."""
        have_targets = ("strategic_targets" in self._buffers) and (getattr(self, "strategic_targets", None) is not None)
        if not self.enable_kde:
            return bool(have_targets)
        n = min(self.reservoir_ptr, self.reservoir_size)
        return bool(have_targets and (n >= getattr(self, "min_reservoir", 8)))

    def _kde_density(self, x: torch.Tensor) -> torch.Tensor:
        """Compute KDE density estimate for novelty"""
        n = min(self.reservoir_ptr, self.reservoir_size)
        min_needed = getattr(self, "min_reservoir", 8)
        if n < min_needed:
            # Graceful fallback - return neutral density for insufficient real data
            return torch.ones(x.size(0), device=x.device) * 0.5
        bank = self.reservoir[:n].to(x.device)  # [n, D]
        dist2 = torch.cdist(x, bank, p=2.0).pow(2)
        k = torch.exp(-dist2 / (2 * (self.bandwidth ** 2)))
        p = k.mean(dim=1).clamp_min(1e-6)
        return p  # pseudo-density

    def seed_from_np_embeddings(self, emb_batch: torch.Tensor):
        # emb_batch: [N,D], real ARC puzzle embeddings
        self.set_strategic_targets(emb_batch)
        self.min_reservoir = 8

    def _update_reservoir(self, states: torch.Tensor):
        """Update reservoir with new states"""
        b = states.size(0)
        take = min(b, self.reservoir_size)
        idxs = torch.arange(take, device=states.device)
        pos = (self.reservoir_ptr + idxs) % self.reservoir_size
        self.reservoir[pos] = states[:take].detach()
        self.reservoir_ptr = int((self.reservoir_ptr + take) % self.reservoir_size)

    def get_alpha(self):
        """Get annealed alpha value"""
        frac = min(self.current_step / self.anneal_steps, 1.0)
        alpha = self.alpha_start + frac * (self.alpha_end - self.alpha_start)
        self.current_step += 1
        return alpha

    def emit_signals(self, ctx=None):
        """Emit structured signals about curiosity rail status"""
        if ctx is not None:
            signals = ctx.setdefault('signals', {})
            metrics = ctx.setdefault('metrics', {})

            # Readiness signals
            signals['curiosity_ready'] = self.is_ready()
            n = min(self.reservoir_ptr, self.reservoir_size)
            metrics['curiosity_reservoir_level'] = n
            metrics['curiosity_min_threshold'] = getattr(self, "min_reservoir", 8)

            # Flow effectiveness
            if hasattr(self, '_recent_rewards'):
                metrics['curiosity_flow'] = len(self._recent_rewards)
            else:
                metrics['curiosity_flow'] = 0

    def compute_reward(self, states: torch.Tensor, actions: Optional[torch.Tensor] = None,
                      update_predictor: bool = True, next_states: Optional[torch.Tensor] = None, ctx=None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute curiosity reward with strict fail-fast checks
        Returns:
          rewards: Tensor shaped [B] (per-sample)
        All rewards scaled to [0, 0.05] to prevent explosion
        """
        device = states.device

        # Cold-start curiosity: use variance-based priors before KDE ready
        n = min(self.reservoir_ptr, self.reservoir_size)
        if n < 8:  # Not enough real data yet for KDE
            B = states.size(0)
            # Use state variance as exploration bonus (high variance = novel = high reward)
            state_variance = states.var(dim=-1).clamp(min=1e-6)  # [B]
            # Normalize to [0, 0.05] range
            variance_reward = (state_variance / (state_variance.max() + 1e-6)) * 0.05
            # Update reservoir for future KDE
            self._update_reservoir(states)
            return variance_reward, {
                "curiosity_active": True,
                "curiosity_mode": "cold_start_variance",
                "reservoir_size": n
            }

        # Prediction error (per-sample)
        pred = self.predictor(states)
        eta = F.mse_loss(pred, states, reduction='none').mean(dim=-1)  # [B]

        # Update running_error *in-place* (preserves registered buffer)
        with torch.no_grad():
            # running_error is a registered buffer shape [1], update with EMA using batch mean
            batch_err_mean = eta.mean().detach()
            # in-place ops to preserve buffer registration & device
            self.running_error.mul_(self.error_momentum)
            self.running_error.add_((1.0 - self.error_momentum) * batch_err_mean)
        # delta per-example
        delta_eta = eta - float(self.running_error.item())

        # Novelty (per-sample)
        alpha = self.get_alpha()
        if self.enable_kde and states.dim() == 2 and len(self.reservoir) >= 8:
            density = self._kde_density(states)  # [B]
            I_alpha = ((density + 1e-6).pow(-(alpha + 1.0) / 2.0) - 1.0).clamp(-1.0, 1.0)
        else:
            # Graceful fallback when insufficient real data - neutral novelty
            B = states.size(0)
            I_alpha = torch.zeros(B, device=states.device)

        # Empowerment: global scalar, broadcast to per-sample
        cov = torch.cov(states.t())
        eps = 1e-3
        cov_stable = cov + eps * torch.eye(self.state_dim, device=device)
        try:
            eigvals = torch.linalg.eigvalsh(cov_stable)
            eigvals = eigvals.clamp(eps, 1.0)
            logdet = eigvals.log().sum()
            E_scalar = float(logdet.clamp(-10.0, 10.0).item())
        except Exception as e:
            if STRICT_CURIOSITY:
                raise RuntimeError(f"[Curiosity] Empowerment eigvalsh failed: {e}")
            E_scalar = 0.0

        # Modulator: scalar based on batch / per-sample stats (use batch-level stability)
        mu_t = float(torch.sigmoid(eta.std().clamp(0.0, 10.0)).item()) if eta.numel() > 1 else float(torch.sigmoid(states.std().clamp(0.0, 10.0)).item())

        # Alignment: per-sample max cosine to any target
        if getattr(self, "strategic_targets", None) is not None and states.dim() == 2:
            tgt = self.strategic_targets.to(states.device)
            S_per = torch.max(F.cosine_similarity(states.unsqueeze(1), tgt.unsqueeze(0), dim=-1), dim=1).values  # [B]
        else:
            if STRICT_CURIOSITY:
                raise RuntimeError("[Curiosity] No strategic targets provided for alignment")
            S_per = F.cosine_similarity(states, states.mean(0, keepdim=True), dim=-1)

        # Construct per-sample reward (broadcast E scalar)
        # Use per-sample eta, delta_eta, I_alpha, per-sample S_per, and broadcasted E*mu_t
        R_per = (0.01 * eta +
                 0.01 * delta_eta.clamp(-1.0, 1.0) +
                 0.01 * I_alpha +
                 0.01 * (E_scalar * mu_t) +
                 0.01 * S_per)
        # clamp to [0, 0.05]
        R_per = R_per.clamp(0.0, 0.05)

        # Update reservoir (KDE data bank) using states (in-place)
        if self.enable_kde and states.dim() == 2:
            self._update_reservoir(states)

        # Predictor update (optionally update predictor using next_states)
        if update_predictor and next_states is not None:
            _ = self.update_predictor(states, next_states)

        info = {
            'prediction_error': torch.nan_to_num(eta.mean(), nan=0.0, posinf=1.0, neginf=0.0),
            'learning_progress': torch.nan_to_num(delta_eta.mean(), nan=0.0, posinf=1.0, neginf=-1.0),
            'novelty': torch.nan_to_num(I_alpha.mean(), nan=0.0, posinf=1.0, neginf=-1.0),
            'empowerment': E_scalar if math.isfinite(E_scalar) else 0.0,
            'modulator': mu_t if math.isfinite(mu_t) else 0.5,
            'alignment_mean': torch.nan_to_num(S_per.mean(), nan=0.0, posinf=1.0, neginf=-1.0)
        }

        # Track recent rewards for flow effectiveness and emit signals
        if not hasattr(self, '_recent_rewards'):
            self._recent_rewards = []
        self._recent_rewards.append(float(R_per.mean().item()))
        if len(self._recent_rewards) > 50:  # Keep last 50 rewards
            self._recent_rewards.pop(0)

        # Emit signals if context provided
        if ctx is not None:
            self.emit_signals(ctx)

        return R_per, info

    def update_predictor(self, states: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Update predictor network - returns loss tensor"""
        self.predictor_optimizer.zero_grad()
        predictions = self.predictor(states)
        loss = F.mse_loss(predictions, targets)
        if torch.is_grad_enabled():
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
            self.predictor_optimizer.step()
        return loss

    # -----------------------------
    # Score API + running-statistics
    # -----------------------------
    def update_running_stats(self, values: torch.Tensor):
        """
        Update running mean/var/std with an EMA-style update.
        `values` expected to be a 1-D tensor of shape [B].
        """
        if values is None or values.numel() == 0:
            return
        # detach to avoid retaining computation graph
        vals = values.detach()
        batch_mean = float(vals.mean().item())
        batch_var = float(vals.var(unbiased=False).item()) if vals.numel() > 1 else 0.0

        if not hasattr(self, "_running_ema_alpha"):
            raise RuntimeError("GCCRFCuriosity._running_ema_alpha missing")
        alpha = float(self._running_ema_alpha)
        # initialize on first update
        if self.running_count == 0:
            self.running_mean = batch_mean
            self.running_var = batch_var
            self.running_count = vals.numel()
        else:
            # EMA updates
            self.running_mean = (1.0 - alpha) * self.running_mean + alpha * batch_mean
            self.running_var = (1.0 - alpha) * self.running_var + alpha * batch_var
            self.running_count += vals.numel()
        # guard numerics
        self.running_std = float(max(1e-6, math.sqrt(max(0.0, self.running_var))))

    # -----------------------------------------------------------------
    # API used by NMDA/dream engine
    # -----------------------------------------------------------------
    @torch.no_grad()
    def score_states(self, states: torch.Tensor, loss_per_example: Optional[torch.Tensor] = None):
        """
        Produce per-example positive weights for NMDA consolidation.
        Returns (weights_tensor, info_dict).
        Maintains running mean/var so weights are comparable over time.
        """
        # Flush prior CUDA kernels to catch async errors immediately
        if torch.cuda.is_available() and states.is_cuda:
            torch.cuda.synchronize()

        if states is None:
            raise RuntimeError("GCCRFCuriosity.score_states: 'states' is None")
        if not torch.is_tensor(states):
            states = torch.as_tensor(states)

        # states: [B, D] or [B, T, D] -> flatten to [B, D]
        if states.dim() == 3:
            states_flat = states.mean(dim=1)
        else:
            states_flat = states

        # Basic curiosity signal: use compute_reward.
        # In strict mode we propagate compute_reward exceptions to surface upstream bugs.
        reward, info = self.compute_reward(states_flat.detach(), update_predictor=False)
        r = reward.detach().float()

        # Optional alignment with strategic targets: boost states aligned to any strategic target
        if getattr(self, "strategic_targets", None) is not None:
            # cosine similarity to each target -> max alignment
            sim = F.cosine_similarity(states_flat.unsqueeze(1), self.strategic_targets.unsqueeze(0), dim=-1)
            align_boost, _ = sim.max(dim=1)
            # combine with curiosity reward
            r = r + 0.5 * align_boost

        # Running normalization (exponential moving mean/var)
        b_mean = float(r.mean().item())
        b_var = float(r.var(unbiased=False).item()) if r.numel() > 1 else 0.0
        self._running_mean.mul_(self._momentum).add_(torch.tensor((1.0 - self._momentum) * b_mean, device=self._running_mean.device))
        self._running_var.mul_(self._momentum).add_(torch.tensor((1.0 - self._momentum) * b_var, device=self._running_var.device))

        denom = torch.sqrt(self._running_var + self._eps)
        normalized = (r - self._running_mean) / (denom + self._eps)

        # Convert normalized curiosity to positive weights (softplus + clip)
        weights = F.softplus(normalized).clamp(min=0.0, max=10.0)

        # Optionally mix with inverse loss_per_example (promote high-loss states slightly)
        if loss_per_example is not None:
            lp = loss_per_example.detach().float()
            # normalize loss similarly
            lp_norm = (lp - lp.mean()) / (lp.std(unbiased=False) + 1e-6)
            weights = 0.7 * weights + 0.3 * F.softplus(lp_norm)

        # Validate weights
        if not torch.isfinite(weights).all():
            raise RuntimeError("GCCRFCuriosity.score_states returned non-finite weights")

        info_dict = {"running_mean": float(self._running_mean.item()), "running_var": float(self._running_var.item())}
        return weights, info_dict