#!/usr/bin/env python3
"""
DreamEngine: Unified dream system with FSHO oscillator, CIO meta-learning,
NMDA gating, GCCRF curiosity, theme synthesis, and wormhole mining.
"""
import math
import random
import time
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from nmda_dreaming import NMDAGatedDreaming
from gccrf_curiosity import GCCRFCuriosity
from emergent_theme_synthesis import EmergentThemeSynthesis
from wormhole_offline import WormholeTemplateMiner
from phi_metrics_neuro import phi_synergy_features, kappa_floor, cge_boost
from ripple_substrate import RippleSubstrate, RippleConfig

# -----------------------------
# STRICT DREAM ENGINE UTILITIES
# -----------------------------
STRICT_DREAM = True  # wire to a CLI/Config later if needed

def _ensure(cond: bool, msg: str):
    if not cond:
        raise RuntimeError(msg)

def _strict_or(default, msg: str):
    raise RuntimeError(msg)

@dataclass
class DreamMotif:
    """TTL Dream Motif with usage tracking and entropy measures"""
    pattern: torch.Tensor  # The pattern/template
    ttl: int  # Time to live (decreases each cycle)
    usage_count: int = 0
    success_rate: float = 0.0
    entropy_reduction: float = 0.0
    created_time: float = 0.0
    last_used: float = 0.0
    
    def tick(self) -> bool:
        """Decrease TTL and return True if should be kept"""
        self.ttl -= 1
        # Keep if TTL > 0 and success rate is reasonable OR very recent
        keep_condition = (self.ttl > 0 and self.success_rate > 0.3) or \
                        (time.time() - self.created_time < 60.0)  # Keep new motifs for 1 minute
        return keep_condition
        
    def update_success(self, success: bool, entropy_delta: float = 0.0):
        """Update success tracking"""
        self.usage_count += 1
        self.last_used = time.time()
        
        # Exponential moving average for success rate
        alpha = 0.2  # Learning rate
        self.success_rate = (1 - alpha) * self.success_rate + alpha * float(success)
        
        # Track entropy reduction
        if entropy_delta < 0:  # Negative delta means entropy reduced (good)
            self.entropy_reduction = (1 - alpha) * self.entropy_reduction + alpha * abs(entropy_delta)
            
class MetaLearner:
    """Meta-learning for dream strategy selection"""
    def __init__(self):
        self.strategy_success = {}  # strategy â†’ success_rate list
        self.task_features = {}  # task â†’ feature vector
        self.feature_dim = 20  # Fixed feature dimension
        
    def update(self, task_id: str, strategy: str, success: bool, features: torch.Tensor = None):
        """Update strategy success rates and task features"""
        # Update strategy success rates
        if strategy not in self.strategy_success:
            self.strategy_success[strategy] = []
        self.strategy_success[strategy].append(success)
        
        # Keep only recent history
        if len(self.strategy_success[strategy]) > 100:
            self.strategy_success[strategy] = self.strategy_success[strategy][-100:]
            
        # Store task features if provided
        if features is not None and features.numel() > 0:
            # Pad or truncate to fixed dimension
            if features.numel() >= self.feature_dim:
                self.task_features[task_id] = features.flatten()[:self.feature_dim]
            else:
                padded = torch.zeros(self.feature_dim, device=features.device, dtype=features.dtype)
                padded[:features.numel()] = features.flatten()
                self.task_features[task_id] = padded
                
    def recommend_strategy(self, task_features: torch.Tensor = None) -> str:
        """Recommend best strategy based on similarity to successful tasks"""
        if not self.strategy_success:
            return 'default'
            
        best_strategy = 'default'
        best_score = -1
        
        for strategy, successes in self.strategy_success.items():
            if not successes:
                continue
                
            base_success = np.mean(successes)
            
            # Add task similarity bonus if we have features
            similarity_bonus = 0.0
            if task_features is not None and self.task_features:
                # Find most similar successful task
                max_similarity = 0.0
                for task_id, stored_features in self.task_features.items():
                    if stored_features.device != task_features.device:
                        stored_features = stored_features.to(task_features.device)
                    similarity = F.cosine_similarity(
                        task_features.flatten()[:self.feature_dim],
                        stored_features.flatten()[:self.feature_dim],
                        dim=0
                    )
                    max_similarity = max(max_similarity, similarity.item())
                similarity_bonus = max_similarity * 0.1
                
            total_score = base_success + similarity_bonus
            
            if total_score > best_score:
                best_score = total_score
                best_strategy = strategy
                
        return best_strategy

@dataclass
class DreamConfig:
    state_dim: int
    action_dim: int = 4
    device: str = "cuda:0"  # GPU-only for EM enhancement
    # Determinism control
    deterministic: bool = False
    cio_seed: int = 1337
    # Logging control
    verbose: bool = False
    # FSHO params
    fsho_K: float = 0.2
    fsho_eta: float = 0.1
    fsho_alpha: float = 1.6  # Levy alpha (1<Î±<=2), 2 => Gaussian
    fsho_H: float = 0.7      # Hurst memory (0<H<1)
    fsho_fgn_scale: float = 0.1   # FGN noise scale
    fsho_levy_scale: float = 0.05  # Levy noise scale
    # CIO meta
    cio_lr: float = 0.02
    cio_hist: int = 128
    # NMDA gate
    valence_default: float = 0.7
    arousal_default: float = 0.5
    # Budgets
    micro_ticks: int = 1    # micro-dream steps per forward
    offline_iters: int = 50 # deep-dream steps per cycle
    # Ripple substrate params
    ripple_rate_hz: float = 0.8
    stdp_gain: float = 3.0
    micro_dt_ms: float = 5.0  # Time step in milliseconds

class DreamEngine:
    """
    Unified dream engine: FSHO + CIO + NMDA + GCCRF + Themes + Wormhole.
    Enhanced with TTL motifs, selective updates, and meta-learning.
    """

    # --- Retrofit wrapper for conductor compatibility ---
    def run_dream_cycle(self, tokens: torch.Tensor, demos_programs=None,
                        valence: float = 0.7, arousal: float = 0.3,
                        max_wall_time_s: float = 2.0, ctx: Optional[dict] = None):
        """Strict-but-safe wrapper: fail-loud in rails, catch at boundary.
        Enforces a wall-time budget and emits novelty/overtime signals.
        Any rail exception is logged and treated as **skip this cycle** (never injects neutral bias).
        """
        import time, logging
        start = time.time()
        try:
            stats = self.cycle_offline(tokens, demos_programs,
                                       valence=valence, arousal=arousal)
        except Exception as e:
            logging.warning(f"[DreamEngine] cycle_offline failed (skipped this cycle): {e}")
            stats = {}
        elapsed = time.time() - start

        if ctx is not None:
            m = ctx.setdefault("metrics", {})
            if elapsed > max_wall_time_s:
                m["dream_overtime"] = m.get("dream_overtime", 0) + 1
            motif_count = int(stats.get("dream_motifs", 0)) if stats else 0
            theme_synths = 0
            if hasattr(self, "theme") and hasattr(self.theme, "synthesis_count"):
                theme_synths = int(getattr(self.theme, "synthesis_count", 0))
            novelty = motif_count + theme_synths
            if novelty > 0:
                m["dream_novelty"] = m.get("dream_novelty", 0) + novelty

        return stats
    def __init__(self, cfg: DreamConfig):
        self.cfg = cfg
        # one-time log keys to prevent spam
        self._once_keys = set()

        # Robust device handling with validation
        self.device = self._validate_device(cfg.device)

        # Update config with validated device
        self.cfg.device = str(self.device)

        # Continue with remaining initialization
        self._continue_init(cfg)

    def _once_log(self, key: str, level: str, msg: str):
        if key in self._once_keys:
            return
        self._once_keys.add(key)
        log = logging.getLogger(__name__)
        getattr(log, level.lower(), log.info)(msg)

    # ---------- Unified rail readiness ----------
    def rails_status(self, demos_programs=None) -> dict:
        st = {}
        # Curiosity: KDE reservoir >= 16 and strategic targets loaded
        st["curiosity"] = (hasattr(self, "curiosity")
                           and hasattr(self.curiosity, "is_ready")
                           and self.curiosity.is_ready())
        # NMDA: enough buffer and gate could open (cheap pre-check: buffer size only)
        st["nmda"] = (hasattr(self, "nmda")
                      and hasattr(self.nmda, "is_ready")
                      and self.nmda.is_ready())
        # Ripple: enough phase samples to compute coherence
        st["ripple"] = (hasattr(self, "ripple")
                        and hasattr(self.ripple, "is_ready")
                        and self.ripple.is_ready())
        # Themes: at least two themes or can synthesize from tokens
        st["themes"] = (hasattr(self, "theme")
                        and hasattr(self.theme, "is_ready")
                        and self.theme.is_ready())
        # Wormhole: we only run miner if we have real programs
        have_programs = bool(demos_programs) and len(demos_programs) > 0
        st["wormhole"] = (have_programs and hasattr(self, "wormhole")
                          and (not hasattr(self.wormhole, "is_ready") or self.wormhole.is_ready(demos_programs)))
        # RelMem: at least one active concept before op-bias integration
        st["relmem"] = (hasattr(self, "_relmem")
                        and self._relmem is not None
                        and hasattr(self._relmem, "is_ready")
                        and self._relmem.is_ready())
        # EBR (EnergyRefiner) is gated at call time by prior tensors presence; tracked elsewhere
        return st

    def _continue_init(self, cfg: DreamConfig):
        """Continue initialization after rails_status method."""

        # safety: ensure small internal MLPs are created if missing
        # (we'll lazily create minimal trainable heads used by train_step())
        if not hasattr(self, "_dream_color_head"):
            import torch.nn as nn
            self._dream_color_head = nn.Sequential(
                nn.Linear(getattr(cfg, "state_dim", 64), 64),
                nn.ReLU(),
                nn.Linear(64, 10)  # predict 10 colors
            )
            self._dream_opbias_head = nn.Sequential(
                nn.Linear(getattr(cfg, "state_dim", 64), 64),
                nn.ReLU(),
                nn.Linear(64, getattr(cfg, "action_dim", 41))
            )
            # place on device
            self._dream_color_head.to(self.device)
            self._dream_opbias_head.to(self.device)

        # Set up deterministic behavior if requested
        if cfg.deterministic:
            torch.use_deterministic_algorithms(True)
            torch.manual_seed(cfg.cio_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(cfg.cio_seed)
                torch.cuda.manual_seed_all(cfg.cio_seed)
        
        # Create seeded generator for reproducible random numbers
        # Always ensure generator is on the same device as DreamEngine
        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(cfg.cio_seed)
        
        # Log device and deterministic configuration
        if cfg.verbose:
            pass  # Device and CIO configuration initialized
            
            pass  # DreamEngine config initialized
        
        # Optional attached relational memory (set by model)
        self._relmem = None

        # TTL Dream Motifs storage - initialize early to prevent AttributeError
        self.dream_motifs = []  # List[DreamMotif]
        self.max_motifs = 50  # Maximum number of motifs to keep
        self.motif_entropy_threshold = 0.70  # More sensitive pattern detection for human-level learning - keep motifs with â‰¥30% entropy reduction

        # Initialize core Dream components (must be in _continue_init, not attach_relmem)
        self.nmda = NMDAGatedDreaming(
            state_dim=self.cfg.state_dim, action_dim=self.cfg.action_dim, device=self.device
        )
        self.curiosity = GCCRFCuriosity(state_dim=self.cfg.state_dim).to(self.device)
        # Initialize strategic targets for curiosity alignment (required in strict mode)
        self._init_strategic_targets()
        self.theme = EmergentThemeSynthesis()
        self.wormhole = WormholeTemplateMiner()

        # FSHO oscillator state (complex z = x + i y) - MUST be initialized here
        self.z = torch.randn(2, device=self.device, generator=self._rng) * 0.1  # [Re, Im]

        # CIO Meta-Learner memory buffers
        self._cio_X = []  # Feature vectors
        self._cio_y = []  # Retention gains
        self._cio_max_hist = 512  # Maximum history size
        self._ridge_lambda = 1e-2  # Ridge regression regularization

        # Meta-learning for strategy selection
        self.meta_learner = MetaLearner()

        # Selective update tracking
        self.beam_entropy_history = []
        self.template_performance = {}  # template_id â†’ performance history

        # Initialize ripple substrate
        # Ripple substrate requires center_freq_hz in range [120.0, 250.0] Hz
        # Need dt_ms small enough so that Nyquist > 120Hz: fs = 1000/dt_ms > 240 => dt_ms < 4.17ms
        # Use dt_ms = 2.0ms to be safe, which gives fs = 500Hz, Nyquist = 225Hz (with 0.45 factor)
        ripple_dt_ms = 2.0  # Override dream engine dt for ripple substrate

        ripple_config = RippleConfig(
            event_rate_hz=self.cfg.ripple_rate_hz,
            stdp_gain=self.cfg.stdp_gain,
            center_freq_hz=170.0,  # Use default 170Hz which is in valid range
            phase_lock=True,
            dt_ms=ripple_dt_ms  # Use faster sampling for ripple substrate
        )
        self.ripple = RippleSubstrate(ripple_config)

        # Centralized ripple time management - single source of truth
        self._ripple_time = 0.0

    def _advance_ripple_time(self, dt_ms: float = None):
        """Centralized ripple time advancement - single source of truth"""
        dt = (dt_ms or self.cfg.micro_dt_ms) / 1000.0  # Convert ms to seconds
        self._ripple_time += dt
        self.ripple.update(self._ripple_time)
        return self._ripple_time

    def to(self, device: str):
        """Move DreamEngine and RNG to a new device (e.g., cuda)"""
        self.device = torch.device(device)
        self.cfg.device = str(self.device)
        # Recreate RNG on the new device
        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(self.cfg.cio_seed)
        # Move any nn.Module subcomponents
        if hasattr(self, "_dream_color_head"):
            self._dream_color_head.to(self.device)
        if hasattr(self, "_dream_opbias_head"):
            self._dream_opbias_head.to(self.device)
        if hasattr(self, "nmda"):
            self.nmda.to(self.device)
        if hasattr(self, "curiosity"):
            self.curiosity.to(self.device)
            # Normalize curiosity buffers to device
            for buf_name in ["reservoir", "running_error", "error_momentum"]:
                if hasattr(self.curiosity, buf_name):
                    setattr(self.curiosity, buf_name,
                            getattr(self.curiosity, buf_name).to(self.device))
        if hasattr(self, "theme"):
            # Move theme synthesis module if it has device-dependent components
            if hasattr(self.theme, "to"):
                self.theme.to(self.device)
        if hasattr(self, "wormhole"):
            # Move wormhole miner if it has device-dependent components
            if hasattr(self.wormhole, "to"):
                self.wormhole.to(self.device)
        # Move oscillator state if it exists
        if hasattr(self, "z"):
            self.z = self.z.to(self.device)
        return self

    def attach_relmem(self, relmem):
        """Attach a relational memory module for dream-gated plasticity."""
        self._relmem = relmem

        # Add projection layer if dimensions mismatch
        if hasattr(relmem, 'D') and relmem.D != self.cfg.state_dim:
            import logging
            logging.info(f"[DreamEngine] Creating projection: state_dim={self.cfg.state_dim} â†’ RelMem.D={relmem.D}")
            self._relmem_proj = nn.Linear(self.cfg.state_dim, relmem.D).to(self.device)
            nn.init.xavier_uniform_(self._relmem_proj.weight)
        else:
            self._relmem_proj = None

        # All Dream components are now initialized in _continue_init
        # Just initialize strategic targets for curiosity
        self._init_strategic_targets()
    
    def _validate_device(self, device_str: str) -> torch.device:
        """
        Validate and convert device string to torch.device with proper fallbacks.
        
        Args:
            device_str: Device specification (e.g., "cuda", "cpu", "cuda:0")
            
        Returns:
            torch.device: Validated device object
            
        Fallback chain:
        1. Try to create torch.device from input string
        2. If CUDA requested but not available, fall back to CPU
        3. If any error, fall back to CPU
        """
        try:
            device = torch.device(device_str)
            
            # Special handling for CUDA devices
            if device.type == 'cuda':
                if not torch.cuda.is_available():
                    raise RuntimeError(f"[DreamEngine] CUDA required for EM enhancement but not available - device: {device_str}")
                    # Force CUDA - no CPU fallback for EM gains
                elif device.index is not None and device.index >= torch.cuda.device_count():
                    print(f"[DreamEngine] CUDA device {device.index} not available, falling back to cuda:0")
                    return torch.device('cuda:0')
            
            return device
            
        except Exception as e:
            raise RuntimeError(f"[DreamEngine] Invalid device '{device_str}': {e} - CUDA required for EM enhancement")
            # No CPU fallback - force GPU for EM gains

    def _init_strategic_targets(self):
        """Initialize strategic targets for curiosity alignment (required in strict mode)"""
        # Create initial strategic targets as normalized random vectors
        # These represent desired directions in the token space for alignment
        num_targets = 8  # Multiple strategic directions
        targets = torch.randn(num_targets, self.cfg.state_dim, device=self.device, generator=self._rng)
        targets = F.normalize(targets, dim=1)  # Normalize to unit vectors
        self.curiosity.set_strategic_targets(targets)

        # NO synthetic reservoir initialization - reservoir fills naturally with real training data
        # Components will activate gracefully when sufficient real data accumulates

        # Track high-performing states for target updates
        self._target_update_buffer = []
        self._target_update_threshold = 0.02  # Curiosity reward threshold for good states

    def _update_strategic_targets(self, tokens_flat: torch.Tensor, curiosity_reward: float):
        """Update strategic targets based on high-performing token states"""
        if curiosity_reward > self._target_update_threshold:
            # Store high-performing states (up to 100 recent ones)
            self._target_update_buffer.append(tokens_flat.mean(0).detach())
            if len(self._target_update_buffer) > 100:
                self._target_update_buffer.pop(0)

            # Update targets every 20 good states
            if len(self._target_update_buffer) >= 20 and len(self._target_update_buffer) % 20 == 0:
                # Cluster high-performing states into strategic targets
                buffer_stack = torch.stack(self._target_update_buffer[-20:])  # Recent 20
                targets = F.normalize(buffer_stack[:8], dim=1)  # Take first 8 as targets
                self.curiosity.set_strategic_targets(targets)

    # --------- FSHO dynamics (fractional + Levy-ish noise, no external deps) ----------
    def _stable_noise(self, alpha: float, size: Tuple[int, ...]) -> torch.Tensor:
        """Chambers-Mallows-Stuck sampling for symmetric alpha-stable (Î²=0)"""
        # For Î±=2 => Gaussian
        U = (torch.rand(size, device=self.device, generator=self._rng) - 0.5) * math.pi
        W = -torch.log(torch.rand(size, device=self.device, generator=self._rng).clamp_min(1e-10))
        if abs(alpha - 2.0) < 1e-5:
            return torch.randn(size, device=self.device, generator=self._rng)
        const = math.tan(math.pi * alpha / 2.0)
        X = (torch.sin(alpha * U) / (torch.cos(U) ** (1.0 / alpha))) * \
            ((torch.cos(U - alpha * U) / W) ** ((1.0 - alpha) / alpha))
        return X

    def _fgn_davies_harte(self, L: int, H: float) -> torch.Tensor:
        """
        Generate Fractional Gaussian Noise using corrected Davies-Harte FFT method.
        Target PSD slope: Î² = 1 - 2H
        
        Args:
            L: Length of FGN sequence to generate
            H: Hurst parameter (0 < H < 1)
            
        Returns:
            torch.Tensor: FGN sequence of length L with correct spectral characteristics
        """
        # Find next power of 2 for efficient FFT
        n = 1
        while n < 2*L:
            n <<= 1
        
        # Compute autocovariance for FBM increments (FGN)
        # Î³(k) = ÏƒÂ²/2 * (|k+1|^(2H) - 2|k|^(2H) + |k-1|^(2H))
        gamma = torch.zeros(n, dtype=torch.float64, device=self.device)
        
        # For k=0: variance = 1
        gamma[0] = 1.0
        
        # For k > 0: use correct FGN autocovariance formula
        for k in range(1, L):
            gamma[k] = 0.5 * ((k+1)**(2*H) - 2*(k**(2*H)) + abs(k-1)**(2*H))
        
        # Create circulant embedding by mirroring (excluding gamma[0] from mirror)
        if L > 1:
            gamma[n-L+1:n] = gamma[1:L].flip(0)
        
        # FFT to get eigenvalues
        eigenvalues = torch.fft.fft(gamma).real
        
        # Check for negative eigenvalues and fix if needed
        min_eig = eigenvalues.min().item()
        if min_eig < 0:
            # Add small correction to ensure positive definiteness
            eigenvalues = eigenvalues - min_eig + 1e-10
        
        # Generate complex Gaussian noise with correct variance
        # Each complex component should be N(0, 0.5) for total variance 1
        Z_real = torch.randn(n, dtype=torch.float64, device=self.device, generator=self._rng) * math.sqrt(0.5)
        Z_imag = torch.randn(n, dtype=torch.float64, device=self.device, generator=self._rng) * math.sqrt(0.5)
        Z = torch.complex(Z_real, Z_imag)
        
        # Apply square root of eigenvalues
        Y = torch.sqrt(eigenvalues + 1e-10) * Z
        
        # IFFT and extract first L samples
        y = torch.fft.ifft(Y).real
        fgn = y[:L]
        
        # Normalize to unit variance
        fgn = (fgn - fgn.mean()) / (fgn.std() + 1e-10)
        
        return fgn.float()  # Return as float32

    def _fractional_innovation(self, H: float, size: Tuple[int, ...]) -> torch.Tensor:
        """Cheap proxy: AR(1)-like persistence to mimic H>0.5 vs anti-persistence for H<0.5"""
        rho = (H - 0.5) * 1.6  # map Hâˆˆ(0,1) to rhoâˆˆ(-0.8,0.8)
        eps = torch.randn(size, device=self.device, generator=self._rng)
        out = torch.zeros_like(eps)
        for t in range(size[0]):
            out[t] = (rho * out[t-1] if t > 0 else 0.0) + eps[t]
        return out

    def fsho_step(self, steps: int = 1):
        """Stuart-Landau oscillator with fractional Gaussian noise + stable innovations"""
        K = self.cfg.fsho_K
        eta = self.cfg.fsho_eta
        alpha = self.cfg.fsho_alpha
        H = self.cfg.fsho_H
        fgn_scale = self.cfg.fsho_fgn_scale
        levy_scale = self.cfg.fsho_levy_scale
        
        for _ in range(steps):
            # z = x + i y
            x, y = self.z[0], self.z[1]
            r2 = x*x + y*y
            # Hopf-like drift
            dx = (1 - r2) * x - K * y
            dy = (1 - r2) * y + K * x
            
            # True Fractional Gaussian Noise using Davies-Harte
            fgn_sequence = self._fgn_davies_harte(L=4, H=H)  # Generate small sequence
            f = fgn_sequence.mean().item()  # Use mean as scalar innovation
            
            # Levy-stable jumps (independent for x, y components)
            Lx = self._stable_noise(alpha, ()).item()
            Ly = self._stable_noise(alpha, ()).item()
            
            # Update with scaled noise contributions
            x = x + eta * (dx + fgn_scale * f + levy_scale * Lx)
            y = y + eta * (dy + fgn_scale * f + levy_scale * Ly)
            
            # Update oscillator state
            self.z = torch.stack([x, y])
            
            # Stability check: ensure ||z|| stays reasonable
            z_norm = torch.norm(self.z).item()
            if z_norm > 5.0 or torch.isnan(self.z).any():
                # Reset if unstable
                self.z = torch.randn(2, device=self.device, generator=self._rng) * 0.1
                print(f"FSHO reset due to instability: ||z||={z_norm:.3f}")

    # -------------------- CIO meta-learning feature engineering ----------------------
    def _feat_vec(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Centralized feature extraction in pure Torch on model device.
        Returns tensor ready for conversion to numpy only when needed.
        """
        # Control knobs (first 4 features for gradient updates)
        K = torch.tensor(self.cfg.fsho_K, device=self.device, dtype=torch.float32)
        eta = torch.tensor(self.cfg.fsho_eta, device=self.device, dtype=torch.float32)
        alpha = torch.tensor(self.cfg.fsho_alpha, device=self.device, dtype=torch.float32)
        H = torch.tensor(self.cfg.fsho_H, device=self.device, dtype=torch.float32)
        
        # Ripple knobs
        ripple_rate = torch.tensor(self.cfg.ripple_rate_hz, device=self.device, dtype=torch.float32)
        stdp_gain = torch.tensor(self.cfg.stdp_gain, device=self.device, dtype=torch.float32)
        
        # Oscillator state
        z_re = self.z[0]
        z_im = self.z[1]
        z_norm = torch.norm(self.z)
        
        # Token statistics
        B, T, D = tokens.shape
        tokens_flat = tokens.reshape(-1, D)
        token_mean = tokens_flat.mean()
        token_std = tokens_flat.std()
        
        # Token entropy (approximate)
        token_prob = torch.softmax(tokens_flat.mean(0), dim=0)
        token_entropy = (-token_prob * torch.log(token_prob + 1e-10)).sum()
        
        # Curiosity metrics
        R_i, info_dict = self.curiosity.compute_reward(tokens_flat.detach())
        R_i_scalar = R_i.mean() if R_i.numel() > 1 else R_i
        I_alpha_scalar = info_dict.get('prediction_error', torch.tensor(0.0, device=self.device))
        
        # Derived oscillator metrics
        eta_cur = eta * (1.0 + 0.1 * z_norm)  # Current effective eta
        delta_eta = eta_cur - eta  # Deviation from baseline
        
        # Empowerment proxy (token variance scaled by curiosity)
        empowerment = tokens.var() * R_i_scalar
        
        # Alignment (oscillator-token coherence proxy)
        alignment = z_norm * token_std
        
        # Ripple metrics (centralized time management)
        current_ripple_time = self._advance_ripple_time()
        
        # Check if ripple has accumulated sufficient samples for meaningful metrics
        if self.ripple.is_ready():
            # Use real ripple metrics - high fidelity signals
            ripple_coherence = torch.tensor(self.ripple.get_coherence_metrics()['current_coherence'],
                                           device=self.device, dtype=torch.float32)
            ripple_phase = torch.tensor(self.ripple.get_phase_info()['phase_normalized'],
                                       device=self.device, dtype=torch.float32)
            ripple_stdp = torch.tensor(self.ripple.get_stdp_gain(),
                                      device=self.device, dtype=torch.float32)
        else:
            # Ripple not ready yet - use neutral values until sufficient samples accumulate
            ripple_coherence = torch.tensor(0.5, device=self.device, dtype=torch.float32)  # Neutral coherence
            ripple_phase = torch.tensor(0.0, device=self.device, dtype=torch.float32)      # Zero phase
            ripple_stdp = torch.tensor(1.0, device=self.device, dtype=torch.float32)       # Unity gain
        ripple_active = torch.tensor(1.0 if self.ripple.is_ripple_active() else 0.0, 
                                    device=self.device, dtype=torch.float32)
        
        features = torch.stack([
            K, eta, alpha, H,  # Control knobs (first 4)
            ripple_rate, stdp_gain,  # Ripple knobs (indices 4-5)
            z_re, z_im, z_norm,  # Oscillator state  
            token_mean, token_std, token_entropy,  # Token stats
            eta_cur, delta_eta, I_alpha_scalar,  # Derived metrics
            empowerment, alignment,  # High-level features
            ripple_coherence, ripple_phase, ripple_stdp, ripple_active  # Ripple metrics
        ])
        
        return features
    
    def _extract_cio_features(self, tokens: torch.Tensor) -> np.ndarray:
        """
        Extract comprehensive feature vector for Ridge regression.
        Uses centralized _feat_vec() and converts to CPU numpy only at the end.
        """
        # Get features in Torch on device
        features_torch = self._feat_vec(tokens)
        
        # Convert to CPU numpy only once for Ridge regression
        features_numpy = features_torch.detach().cpu().numpy().astype(np.float32)
        
        return features_numpy
        
    def compute_beam_entropy(self) -> float:
        """Compute current beam entropy for selective updates"""
        if not hasattr(self, '_last_tokens') or self._last_tokens is None:
            return _strict_or(0.5, "[DreamEngine] beam entropy requested without cached tokens")
            
        try:
            # Use cached tokens from last computation
            tokens = self._last_tokens
            
            # Compute entropy across feature dimensions
            # H = -sum(p * log(p)) where p = softmax(features)
            B, T, D = tokens.shape
            
            # Average across batch and time, compute entropy over features
            mean_features = tokens.mean(dim=(0, 1))  # [D]
            probs = F.softmax(mean_features, dim=0) + 1e-10
            entropy = -(probs * torch.log(probs)).sum().item()
            
            # Normalize to [0, 1] range (log(D) is max entropy for D dimensions)
            normalized_entropy = entropy / math.log(D) if D > 1 else 0.0

            return float(normalized_entropy)

        except Exception as e:
            return _strict_or(None, f"[DreamEngine] beam entropy failed: {e}")
            
    def compute_beam_entropy_with_template(self, template: torch.Tensor) -> float:
        """Compute beam entropy if we were to add this template"""
        if not hasattr(self, '_last_tokens') or self._last_tokens is None:
            return _strict_or(0.5, "[DreamEngine] beam entropy (template) without cached tokens")
            
        try:
            tokens = self._last_tokens
            
            # Simulate adding template by modifying features
            # Simple approach: add template as guidance to features
            B, T, D = tokens.shape
            
            # Broadcast template to match token dimensions
            if template.dim() == 1:
                template_expanded = template.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
            else:
                template_expanded = template
                
            # Ensure compatible dimensions
            if template_expanded.shape[-1] != D:
                # Pad or truncate template to match feature dimension
                if template_expanded.shape[-1] > D:
                    template_expanded = template_expanded[..., :D]
                else:
                    padding = D - template_expanded.shape[-1]
                    template_expanded = F.pad(template_expanded, (0, padding))
                    
            # Mix template with tokens (small influence)
            mixed_tokens = 0.95 * tokens + 0.05 * template_expanded
            
            # Compute entropy of mixed features
            mean_features = mixed_tokens.mean(dim=(0, 1))
            probs = F.softmax(mean_features, dim=0) + 1e-10
            entropy = -(probs * torch.log(probs)).sum().item()
            
            normalized_entropy = entropy / math.log(D) if D > 1 else 0.0
            return float(normalized_entropy)

        except Exception as e:
            return _strict_or(None, f"[DreamEngine] beam entropy with template failed: {e}")
            
    def add_dream_motif(self, pattern: torch.Tensor, ttl: int = 10) -> bool:
        """Add a new dream motif with TTL and entropy checking"""
        try:
            # Test if motif reduces entropy
            initial_entropy = self.compute_beam_entropy()
            test_entropy = self.compute_beam_entropy_with_template(pattern)
            
            denom = max(float(initial_entropy), 1e-6)
            entropy_reduction = (initial_entropy - test_entropy) / denom
            
            # Only add if entropy reduction meets threshold
            if entropy_reduction >= (1.0 - self.motif_entropy_threshold):  # 15% reduction
                motif = DreamMotif(
                    pattern=pattern.detach().clone(),
                    ttl=ttl,
                    entropy_reduction=entropy_reduction,
                    created_time=time.time()
                )
                
                self.dream_motifs.append(motif)
                
                # Limit number of motifs
                if len(self.dream_motifs) > self.max_motifs:
                    # Remove worst performing motifs
                    self.dream_motifs.sort(key=lambda m: m.success_rate + m.entropy_reduction, reverse=True)
                    self.dream_motifs = self.dream_motifs[:self.max_motifs]
                    
                if self.cfg.verbose:
                    print(f"[DreamMotif] Added motif with {entropy_reduction:.3f} entropy reduction")
                    
                return True
            else:
                raise RuntimeError(f"[DreamMotif] insufficient entropy reduction ({entropy_reduction:.3f})")

        except Exception as e:
            raise

    def seed_motifs_from_latents(self, latents: torch.Tensor,
                                 ctx: Optional[dict] = None,
                                 wormhole=None,
                                 pretrain: bool = True) -> int:
        """
        Adapter: mine motifs directly from Dream latents and route them into Wormhole.
        Adds detailed diagnostics and threshold relaxation during pretrain.
        - Extract candidate templates from latent tokens
        - Add them to Dream motif buffer
        - Optionally consolidate into Wormhole with MDL scoring (tagged as dream_latent)
        Returns number of motifs added.
        """
        added = 0
        if latents is None or latents.numel() == 0:
            import logging
            logging.debug("[Dreamâ†’Motif] No latents provided")
            return 0
        try:
            templates = self.extract_selective_templates(latents)
            if not templates:
                import logging
                logging.debug("[Dreamâ†’Motif] No templates passed entropy/novelty thresholds")
            for tpl in templates:
                if self.add_dream_motif(tpl, ttl=15):
                    added += 1
                    if wormhole is not None and hasattr(wormhole, "consolidator"):
                        try:
                            program = [(op, {}) for op in tpl]
                            sig = wormhole.consolidator._compute_signature(program)
                            if sig not in wormhole.consolidator.library.templates:
                                from wormhole_offline import Template
                                template = Template(
                                    ops=program,
                                    support=1 if not pretrain else 0,  # weak support if pretrain
                                    score=0.1 if not pretrain else 0.01,  # low score if pretrain
                                    signature=sig
                                )
                                setattr(template, "source", "dream_latent")
                                wormhole.library.add_template(template)
                        except Exception as e:
                            import logging
                            logging.warning(f"[Dreamâ†’Wormhole] consolidation failed: {e}")
            if ctx is not None:
                m = ctx.setdefault("metrics", {})
                m["dream_novelty"] = m.get("dream_novelty", 0) + added
                m["dream_motif_candidates"] = len(templates)
                m["dream_motifs_added"] = added
        except Exception as e:
            import logging
            logging.error(f"[DreamEngine] motif seeding failed: {e}")
        return added

    def tick_motifs(self):
        """Update TTL for all motifs and remove expired ones"""
        before_count = len(self.dream_motifs)
        self.dream_motifs = [motif for motif in self.dream_motifs if motif.tick()]
        removed_count = before_count - len(self.dream_motifs)
        
        if removed_count > 0 and self.cfg.verbose:
            print(f"[DreamMotif] Removed {removed_count} expired motifs, {len(self.dream_motifs)} remaining")
            
    def extract_selective_templates(self, tokens: torch.Tensor, demos_programs=None) -> List[torch.Tensor]:
        """Extract templates only if they provide entropy reduction"""
        # Cache tokens for entropy computation
        self._last_tokens = tokens.detach()
        
        initial_entropy = self.compute_beam_entropy()
        
        # Extract candidate templates (simplified)
        templates = []
        
        B, T, D = tokens.shape
        # (no swallow) â€“ if anything below fails, let it raise
        # Pattern 1: High-variance features (indicating structure)
        feature_var = tokens.var(dim=(0, 1))  # [D]
        high_var_mask = feature_var > feature_var.mean() + feature_var.std()
        if high_var_mask.any():
            # Use ONLY masked values, normalized (no zero padding)
            pattern_values = tokens.mean(dim=(0, 1))[high_var_mask]
            # Create dense representation instead of sparse zeros
            pattern = F.normalize(pattern_values.unsqueeze(0), p=2, dim=-1).squeeze(0)
            templates.append(pattern)

        # Pattern 2: Dominant activation patterns
        mean_activation = tokens.mean(dim=(0, 1))  # [D]
        activation_threshold = mean_activation.mean() + mean_activation.std()
        dominant_mask = mean_activation > activation_threshold
        if dominant_mask.any():
            # Dense representation - no zero padding
            pattern_values = mean_activation[dominant_mask]
            pattern = F.normalize(pattern_values.unsqueeze(0), p=2, dim=-1).squeeze(0)
            templates.append(pattern)

        # Pattern 3: Temporal consistency patterns
        if T > 1:
            temporal_std = tokens.std(dim=1).mean(dim=0)  # [D] - low std = consistent
            consistent_mask = temporal_std < temporal_std.median()
            if consistent_mask.any():
                # Dense representation - no zero padding
                pattern_values = tokens.mean(dim=(0, 1))[consistent_mask]
                pattern = F.normalize(pattern_values.unsqueeze(0), p=2, dim=-1).squeeze(0)
                templates.append(pattern)
                
        # Filter templates by entropy reduction
        good_templates = []
        for template in templates:
            test_entropy = self.compute_beam_entropy_with_template(template)
            denom = max(float(initial_entropy), 1e-6)
            entropy_reduction = (initial_entropy - test_entropy) / denom
            
            if entropy_reduction >= (1.0 - self.motif_entropy_threshold):
                good_templates.append(template)
                if self.cfg.verbose:
                    print(f"[DreamEngine] Accepted template with {entropy_reduction:.3f} entropy reduction")
            
        return good_templates

    # -------------------- CIO meta-learning (perturb-and-learn) -----------------------
    def cio_perturb_and_learn(self, tokens: torch.Tensor, prior_retention: float) -> float:
        """
        CIO Meta-Learner with Ridge regression instead of heuristic updates.
        Collects (features, gain) pairs and uses Ridge regression to predict optimal knob gradients.
        """
        # Extract current feature vector
        current_features = self._extract_cio_features(tokens)
        
        # Compute current retention gain (same calculation as before)
        B, T, D = tokens.shape
        tokens_flat = tokens.reshape(B*T, D)
        R_i, _ = self.curiosity.compute_reward(tokens_flat.detach())
        R_i_scalar = R_i.mean() if R_i.numel() > 1 else R_i
        gain = float((R_i_scalar + tokens.var()).item())
        
        # Store (features, gain) pair in memory buffer
        self._cio_X.append(current_features.copy())
        self._cio_y.append(gain)
        
        # Maintain maximum history size
        if len(self._cio_X) > self._cio_max_hist:
            self._cio_X.pop(0)
            self._cio_y.pop(0)
            
        # Ridge regression learning (only if sufficient data)
        if len(self._cio_X) >= 32:
            try:
                # Prepare training data
                X = np.array(self._cio_X)  # [N, 15] features
                y = np.array(self._cio_y)  # [N,] gains
                
                # Ridge regression: (X^T X + Î»I)^{-1} X^T y
                XtX = X.T @ X
                lambda_I = self._ridge_lambda * np.eye(X.shape[1])
                try:
                    ridge_coeff = np.linalg.solve(XtX + lambda_I, X.T @ y)
                except np.linalg.LinAlgError:
                    # Fallback to pseudo-inverse if singular
                    ridge_coeff = np.linalg.pinv(XtX + lambda_I) @ (X.T @ y)
                
                # Gradient ascent on control knobs (first 6 coefficients)
                grad_K = ridge_coeff[0]
                grad_eta = ridge_coeff[1] 
                grad_alpha = ridge_coeff[2]
                grad_H = ridge_coeff[3]
                grad_ripple_rate = ridge_coeff[4]
                grad_stdp_gain = ridge_coeff[5]
                
                # Update control knobs with gradient ascent and clamping
                step_size = self.cfg.cio_lr
                self.cfg.fsho_K = np.clip(
                    self.cfg.fsho_K + step_size * grad_K, 0.05, 0.8
                )
                self.cfg.fsho_eta = np.clip(
                    self.cfg.fsho_eta + step_size * grad_eta, 0.02, 0.5
                )
                self.cfg.fsho_alpha = np.clip(
                    self.cfg.fsho_alpha + step_size * grad_alpha, 1.1, 2.0
                )
                self.cfg.fsho_H = np.clip(
                    self.cfg.fsho_H + step_size * grad_H, 0.1, 0.95
                )
                # Update ripple knobs
                self.cfg.ripple_rate_hz = np.clip(
                    self.cfg.ripple_rate_hz + step_size * grad_ripple_rate, 0.1, 5.0
                )
                self.cfg.stdp_gain = np.clip(
                    self.cfg.stdp_gain + step_size * grad_stdp_gain, 1.0, 10.0
                )
                
                # Update ripple substrate configuration
                self.ripple.config.event_rate_hz = self.cfg.ripple_rate_hz
                self.ripple.config.stdp_gain = self.cfg.stdp_gain
                
                # Log configuration changes for debugging
                if self.cfg.verbose:
                    print(f"CIO Ridge update: K={self.cfg.fsho_K:.4f}, eta={self.cfg.fsho_eta:.4f}, "
                          f"alpha={self.cfg.fsho_alpha:.4f}, H={self.cfg.fsho_H:.4f}, "
                          f"ripple_rate={self.cfg.ripple_rate_hz:.4f}, stdp_gain={self.cfg.stdp_gain:.4f}, "
                          f"gain={gain:.4f}")
                      
            except Exception as e:
                raise RuntimeError(f"[CIO] Ridge regression failed: {e}")
        else:
            # CIO not ready yet - return neutral gain until sufficient samples accumulate
            # This allows CIO to naturally collect real training data without crashes
            if self.cfg.verbose:
                print(f"[CIO] Waiting for samples: {len(self._cio_X)}/32")
            gain = prior_retention  # No change until ready
            
        return gain
    
    def _fallback_perturbation(self, gain: float, prior_retention: float):
        """Fallback perturbation method using seeded generator when Ridge regression is not applicable."""
        if gain > prior_retention:
            # Small random improvements to all knobs using seeded generator
            perturb_K = torch.rand(1, device=self.device, generator=self._rng).item() * 0.04 - 0.02  # [-0.02, 0.02]
            perturb_eta = torch.rand(1, device=self.device, generator=self._rng).item() * 0.02 - 0.01  # [-0.01, 0.01]
            perturb_alpha = torch.rand(1, device=self.device, generator=self._rng).item() * 0.1 - 0.05  # [-0.05, 0.05]
            perturb_H = torch.rand(1, device=self.device, generator=self._rng).item() * 0.04 - 0.02  # [-0.02, 0.02]
            perturb_ripple_rate = torch.rand(1, device=self.device, generator=self._rng).item() * 0.2 - 0.1  # [-0.1, 0.1]
            perturb_stdp = torch.rand(1, device=self.device, generator=self._rng).item() * 0.4 - 0.2  # [-0.2, 0.2]
            
            self.cfg.fsho_K = np.clip(
                self.cfg.fsho_K + perturb_K, 0.05, 0.8
            )
            self.cfg.fsho_eta = np.clip(
                self.cfg.fsho_eta + perturb_eta, 0.02, 0.5
            )
            self.cfg.fsho_alpha = np.clip(
                self.cfg.fsho_alpha + perturb_alpha, 1.1, 2.0
            )
            self.cfg.fsho_H = np.clip(
                self.cfg.fsho_H + perturb_H, 0.1, 0.95
            )
            # Also perturb ripple knobs in fallback
            self.cfg.ripple_rate_hz = np.clip(
                self.cfg.ripple_rate_hz + perturb_ripple_rate, 0.1, 5.0
            )
            self.cfg.stdp_gain = np.clip(
                self.cfg.stdp_gain + perturb_stdp, 1.0, 10.0
            )

    # -------------------- Public API used by model -----------------------------------
    @torch.no_grad()
    def compute_priors(self, tokens: torch.Tensor, H: int, W: int) -> Dict[str, torch.Tensor]:
        """
        Compute neuromorphic priors for EBR: phi, kappa, cge (hodge optional).
        tokens: [B, T, D]  (T = H*W if possible)
        """
        device = tokens.device
        try:
            phi   = phi_synergy_features(tokens)
            kappa = kappa_floor(tokens, H, W)
            cge   = cge_boost(tokens)
        except Exception as e:
            raise RuntimeError(f"[DreamEngine] compute_priors failed: {e}")
        return dict(phi=phi, kappa=kappa, cge=cge)

    def record_experience(self, latent_state: torch.Tensor, next_latent: Optional[torch.Tensor] = None,
                          action: int = 0, reward: float = 0.0, valence: Optional[float] = None,
                          arousal: Optional[float] = None):
        """Push one experience into NMDA buffer (latent space)."""
        v = self.cfg.valence_default if valence is None else float(valence)
        a = self.cfg.arousal_default if arousal is None else float(arousal)
        if next_latent is None:
            next_latent = latent_state
        self.nmda.store_dream_memory(latent_state, action, reward, next_latent)
        # NMDA gate checked during consolidation call

    # -------- Curiosity adapter for NMDA (strict interface) --------
    @torch.no_grad()
    def score_states(self, states: torch.Tensor, loss_per_example: Optional[torch.Tensor] = None):
        """
        Return (weights, info) for NMDA prioritization.
        - states: [B, D] on same device as DreamEngine
        - weights: [B] positive, clipped, finite
        """
        if states is None or not torch.is_tensor(states):
            raise RuntimeError("[NMDA] curiosity adapter: states must be a Tensor")
        states = states.to(self.device)

        info = {}
        # Prefer GCCRF.z-scores if available
        if hasattr(self, "curiosity") and hasattr(self.curiosity, "score_states"):
            result = self.curiosity.score_states(states)  # Returns (weights, info_dict)
            # Handle tuple return
            if isinstance(result, tuple):
                z, curiosity_info = result
                info.update(curiosity_info or {})
            else:
                z = result
            if not torch.is_tensor(z):
                z = torch.as_tensor(z, device=self.device, dtype=torch.float32)
            # standardize â†' positive weights
            m, s = float(z.mean().item()), float(z.std().item() + 1e-8)
            w = (z - m) / s
        elif hasattr(self, "curiosity") and hasattr(self.curiosity, "compute_reward"):
            R_i, info_c = self.curiosity.compute_reward(states)
            info.update({k: (float(v.item()) if torch.is_tensor(v) else float(v)) for k, v in (info_c or {}).items() if v is not None})
            if not torch.is_tensor(R_i):
                R_i = torch.as_tensor(R_i, device=self.device, dtype=torch.float32)
            m, s = float(R_i.mean().item()), float(R_i.std().item() + 1e-8)
            w = (R_i - m) / s
        else:
            raise RuntimeError("[NMDA] curiosity_module lacks score_states/compute_reward")

        # Map to positive, clipped weights
        weights = (1.0 + w).clamp(0.1, 5.0).to(torch.float32)
        if not torch.isfinite(weights).all():
            raise RuntimeError("[NMDA] curiosity weights non-finite")
        info["w_mean"] = float(weights.mean().item())
        info["w_std"] = float(weights.std().item())
        return weights, info

    def step_micro(self, valence: Optional[float], arousal: Optional[float]):
        """Tiny consolidation step (online-safe) with training signal extraction."""
        v = self.cfg.valence_default if valence is None else float(valence)
        a = self.cfg.arousal_default if arousal is None else float(arousal)

        st = self.rails_status()
        if not st.get("curiosity", False):
            self._once_log("curiosity_inactive", "DEBUG",
                           "[Curiosity] inactive: reservoir/targets below thresholds; deferring NMDA micro-tick")
            return None
        if not st.get("nmda", False):
            self._once_log("nmda_inactive", "DEBUG",
                           "[NMDA] inactive: buffer/gate thresholds not met; deferring NMDA micro-tick")
            return None

        # allow smaller NMDA buffers early
        self.nmda.min_batch_size = getattr(self.nmda, "min_batch_size", max(2, int(self.nmda.batch_size/2)))

        # Perform NMDA consolidation with curiosity weighting (read-only for micro-ticks)
        # Micro-ticks are retrieval-only (no backward pass)
        with torch.no_grad():
            try:
                nmda_loss = self.nmda.dream_consolidation(v, a, curiosity_module=self, backprop=False)
            except Exception as e:
                import logging
                logging.warning(f"Dream micro-tick failed: {e}")
                return 0.0

        # Extract actionable training signals from dream consolidation
        training_signals = self._extract_training_signals(v, a)

        # Store signals for main training to access
        self.last_training_signals = training_signals

        return nmda_loss

    def motif_seed(self, programs, relaxed=False):
        """Seed Wormhole miner + ThemeSynth with programs from DSL"""
        if not programs:
            return []
        try:
            # Relax thresholds early
            min_support = 1 if relaxed else 2
            min_length = 1 if relaxed else 2

            if hasattr(self, 'wormhole') and hasattr(self.wormhole, 'mine_from_programs'):
                templates = self.wormhole.mine_from_programs(programs, top_k=5)
                if templates and hasattr(self, 'theme') and hasattr(self.theme, 'synthesis_count'):
                    # Increment synthesis count to track motifs
                    self.theme.synthesis_count = getattr(self.theme, 'synthesis_count', 0) + len(templates)
                return templates
            else:
                logger.info(f"[DreamEngine] wormhole mining unavailable")
                return []
        except Exception as e:
            logger.warning(f"[DreamEngine] motif_seed error: {e}")
            return []

    # -----------------------------
    # Trainer Self-Reward Extension
    # -----------------------------
    def compute_trainer_self_reward(self, ctx, prev_metrics=None):
        """
        Meta-reward: how well the trainer is steering its own rails.
        ctx: dict with .signals and .metrics from rails
        prev_metrics: dict of metrics from previous step (to compute deltas)
        Returns: (reward: float, components: dict)
        """
        signals = ctx.get("signals", {})
        metrics = ctx.get("metrics", {})
        prev = prev_metrics or {}

        # Safe deltas - track improvement in core metrics
        d_em = metrics.get("exact_match", 0) - prev.get("exact_match", 0)
        d_acc = metrics.get("accuracy", 0) - prev.get("accuracy", 0)
        d_cur = metrics.get("curiosity_flow", 0) - prev.get("curiosity_flow", 0)

        # Positive signals - rails working effectively
        theme = metrics.get("theme_emergence", 0)
        wormhole_success = metrics.get("wormhole_success", 0)
        relmem_bindings = metrics.get("relmem_bindings", 0)

        # Negative signals - rails failing or going silent
        starv_wormhole = metrics.get("wormhole_starvation", 0)
        starv_curiosity = 1 if not signals.get("curiosity_ready", True) else 0
        starv_nmda = metrics.get("nmda_skipped", 0)
        starv_relmem = 1 if not signals.get("relmem_active", True) else 0
        fallbacks = metrics.get("silent_fallbacks", 0)

        # Total rail starvation
        rail_starvation = starv_wormhole + starv_curiosity + starv_nmda + starv_relmem

        # Weighted meta-reward formula
        reward = (
            +1.0 * d_em                    # Most important: EM improvement
            +0.5 * d_acc                   # Accuracy trends
            +0.3 * d_cur                   # Curiosity effectiveness
            +0.2 * theme                   # Theme/motif discovery
            +0.1 * wormhole_success        # Successful mining events
            +0.1 * relmem_bindings         # Concept binding events
            -0.5 * rail_starvation         # Penalize rail failures
            -0.3 * fallbacks               # Penalize silent fallbacks
        )

        components = {
            "d_em": d_em, "d_acc": d_acc, "d_cur": d_cur,
            "theme": theme, "wormhole_success": wormhole_success,
            "relmem_bindings": relmem_bindings, "rail_starvation": rail_starvation,
            "fallbacks": fallbacks, "total_reward": reward
        }

        return reward, components

    def update_trainer_self_score(self, ctx, prev_metrics=None):
        """
        Compute trainer reward and update moving average baseline.
        Stores self.trainer_self_score for logging/plotting.
        """
        r, comps = self.compute_trainer_self_reward(ctx, prev_metrics)

        # Initialize EMA baseline
        if not hasattr(self, "_ema_trainer"):
            self._ema_trainer = 0.0
            self._trainer_history = []

        # Update EMA baseline
        self._ema_trainer = 0.9 * self._ema_trainer + 0.1 * r
        advantage = r - self._ema_trainer

        # Store for access
        self.trainer_self_score = float(advantage)
        self.trainer_self_comps = comps
        self.trainer_raw_score = float(r)

        # Track history for adaptation
        self._trainer_history.append((r, advantage, comps))
        if len(self._trainer_history) > 100:  # Keep last 100 scores
            self._trainer_history.pop(0)

        return advantage, comps

    def should_adapt_parameters(self, window_size=10):
        """
        Determine if trainer should adapt parameters based on recent performance.
        Returns: (should_adapt: bool, adaptation_type: str, confidence: float)
        """
        if not hasattr(self, "_trainer_history") or len(self._trainer_history) < window_size:
            return False, "insufficient_data", 0.0

        recent_advantages = [adv for _, adv, _ in self._trainer_history[-window_size:]]
        recent_rewards = [r for r, _, _ in self._trainer_history[-window_size:]]

        avg_advantage = sum(recent_advantages) / len(recent_advantages)
        avg_reward = sum(recent_rewards) / len(recent_rewards)

        # Strong negative trend - urgent adaptation needed
        if avg_advantage < -0.5 and avg_reward < -0.3:
            return True, "urgent_adapt", 0.9

        # Moderate negative trend - gradual adaptation
        elif avg_advantage < -0.2:
            return True, "gradual_adapt", 0.6

        # Positive trend - maybe tighten parameters
        elif avg_advantage > 0.3 and avg_reward > 0.2:
            return True, "optimize_tighten", 0.7

        return False, "stable", avg_advantage

    def _extract_training_signals(self, valence: float, arousal: float) -> Dict[str, torch.Tensor]:
        """Extract meaningful training signals from dream state for main network"""
        signals = {}

        try:
            # 1) DSL Operation Confidence from NMDA Q-values
            if hasattr(self.nmda, 'q_net') and len(self.nmda.buffer) > 0:
                # Sample recent experiences to get operation preferences
                recent_experiences = list(self.nmda.buffer)[-20:]  # Last 20 experiences
                if recent_experiences:
                    # Compute average Q-values for each action (DSL operation)
                    states = torch.stack([e.state for e in recent_experiences]).to(self.device)
                    with torch.no_grad():
                        q_values = self.nmda.q_net(states)  # [batch, num_actions]
                        # Convert Q-values to operation confidence scores
                        op_confidence = torch.softmax(q_values.mean(dim=0), dim=0)  # Average across batch
                        signals['dsl_operation_priors'] = op_confidence.detach()

            # 2) Pattern Attention Weights from Ripple Coherence
            if hasattr(self, 'ripple') and self.ripple.is_ripple_active():
                ripple_ctx = self.ripple.get_current_context()
                # Higher coherence â†’ stronger attention signal
                attention_strength = torch.tensor(ripple_ctx.coherence * ripple_ctx.amplitude, device=self.device)
                signals['attention_boost'] = attention_strength.detach()

            # 3) Memory Consolidation Bias from Successful Patterns
            if hasattr(self, 'theme') and hasattr(self.theme, 'synthesis_buffer'):
                # Extract pattern features from successful syntheses
                recent_themes = getattr(self.theme, 'synthesis_buffer', [])[-5:]
                if recent_themes:
                    # Create bias vector from successful theme patterns
                    pattern_bias = torch.zeros(self.cfg.state_dim, device=self.device)
                    for theme in recent_themes:
                        if hasattr(theme, 'pattern') and torch.is_tensor(theme.pattern):
                            # Weight by success rate and recency
                            weight = getattr(theme, 'success_rate', 0.5)
                            pattern_bias += weight * theme.pattern.flatten()[:self.cfg.state_dim].to(self.device)

                    if pattern_bias.norm() > 0:
                        signals['memory_bias'] = pattern_bias.detach() / pattern_bias.norm()

            # 4) Curiosity Direction from curiosity module
            if hasattr(self, 'curiosity') and hasattr(self.curiosity, 'get_curiosity_vector'):
                curiosity_vec = self.curiosity.get_curiosity_vector()
                if curiosity_vec is not None and torch.is_tensor(curiosity_vec):
                    signals['curiosity_direction'] = curiosity_vec.detach()

        except Exception as e:
            if STRICT_DREAM:
                raise RuntimeError(f"[DreamEngine] training signal extraction failed: {e}")
            import logging
            logging.debug(f"Dream training signal extraction failed (non-strict): {e}")
        return signals

    def cycle_offline(self, tokens: torch.Tensor, demos_programs: Optional[List[List[Tuple[str, dict]]]] = None,
                      valence: float = 0.7, arousal: float = 0.3) -> Dict[str, float]:
        """
        Deep dream consolidation: oscillator rollouts, CIO perturb-learn, theme synthesis,
        wormhole motif mining, NMDA consolidation sweeps with ripple substrate integration.
        """
        import time
        
        # Always enforce tokens on DreamEngine device for consistency
        tokens = tokens.to(self.device)
        B, T, D = tokens.shape
        
        # Tokens should now always have the correct state_dim since we're using brain tokens
        # If dimension mismatch, it's a bug that should be fixed at the source
        if D != self.cfg.state_dim:
            raise ValueError(f"DreamEngine received tokens with dimension {D}, expected {self.cfg.state_dim}. "
                           f"Ensure brain tokens are being passed with ctrl_dim={self.cfg.state_dim}")
        
        # Cache last tokens for entropy/motif computations
        self._last_tokens = tokens.detach()

        # Synthesize themes from tokens to populate motifs and enable high-fidelity signals
        if hasattr(self, 'theme') and hasattr(self.theme, 'process_dream_themes'):
            # Create synthetic labels based on token variance patterns
            labels = torch.arange(min(8, B), device=self.device)[:B]  # Use up to 8 unique labels
            themes = self.theme.process_dream_themes(tokens.mean(dim=1), labels)
            # Trigger synthesis to create motifs
            if len(themes) >= 2:
                self.theme.synthesize_emergent_themes(themes)

        energy_scalar = 1.0 + float(tokens.var().item()) * 0.01
        
        # Adaptive offline iters based on NMDA buffer
        iters = self.cfg.offline_iters
        try:
            if hasattr(self, "nmda") and hasattr(self.nmda, "buffer"):
                buf_len = len(getattr(self.nmda, "buffer", []))
                if buf_len < 100:
                    iters = max(5, iters // 4)
                elif buf_len < 500:
                    iters = max(10, iters // 2)
        except Exception:
            pass

        # Reset ripple time for new cycle - ensures monotonic timing
        self._ripple_time = 0.0

        # 0) FSHO roll with ripple integration
        for i in range(iters):
            self.fsho_step()

            # Update ripple substrate with centralized time management
            current_ripple_time = self._advance_ripple_time()
            
            # Get active ripple context
            ripple_ctx = self.ripple.get_current_context() if self.ripple.is_ripple_active() else None
            
            # Store real training experiences in NMDA buffer - NO synthetic data
            # This will be populated from actual brain state transitions during real training

            # Perform NMDA consolidation with ripple context and curiosity weighting every few steps
            if i % 10 == 0 and self.nmda.is_ready():
                # Allow NMDA Q-net to learn even if outer context is no_grad()
                with torch.enable_grad():
                    loss = self.nmda.dream_consolidation(valence, arousal, self, ripple_ctx=ripple_ctx)

        # 1) CIO meta-learn a small step using tokens
        prior = 0.0
        gain = self.cio_perturb_and_learn(tokens, prior)
        
        # Tick motifs to update TTL
        self.tick_motifs()

        # 2) Theme synthesis from tokens (mock labels as cluster ids)
        labels = torch.arange(T, device=tokens.device) % max(2, T//4)
        themes = self.theme.process_dream_themes(tokens.mean(1), labels)
        self.theme.synthesize_emergent_themes(themes)

        # 3) Extract selective templates and add as motifs
        selective_templates = self.extract_selective_templates(tokens, demos_programs)
        for template in selective_templates:
            self.add_dream_motif(template, ttl=15)  # Longer TTL for good templates
            
        # 4) Wormhole motif mining (if programs recorded)
        mined = []
        if demos_programs:
            mined = self.wormhole.mine_from_programs(demos_programs, top_k=5)
            _ensure(isinstance(mined, list), "[DreamEngine] wormhole miner returned non-list")
            # normalize mined templates to DreamEngine device
            mined = [tpl.to(self.device) if hasattr(tpl, "to") else tpl for tpl in mined]

        # 5) Several NMDA consolidation passes with ripple context (only when ready)
        losses = []
        if self.nmda.is_ready():
            # Allow NMDA Q-net to learn even if outer context is no_grad()
            with torch.enable_grad():
                for i in range(3):
                    # Use centralized time advancement
                    current_ripple_time = self._advance_ripple_time()
                    ripple_ctx = self.ripple.get_current_context() if self.ripple.is_ripple_active() else None
                    loss_i = self.nmda.dream_consolidation(valence, arousal, self, ripple_ctx=ripple_ctx)
                    _ensure(loss_i is not None, "[DreamEngine] nmda consolidation returned None")
                    losses.append(loss_i)
        else:
            # NMDA not ready - skip consolidation passes until buffer fills with real data
            losses = [torch.tensor(0.0, device=self.device)]

        # Diagnostics: NMDA buffer length and consolidation stats
        nmda_buf = len(getattr(self.nmda, "buffer", []))
        nmda_passes = len(losses) if isinstance(losses, list) else 0

        # Reinforcement hook: if consolidation produced a favorable signal, reinforce nearest themes
        try:
            # map loss -> reinforcement_strength (example mapping, tuneable)
            reinforce_strength = 0.0
            if losses and len(losses) > 0:
                loss_i = losses[-1]  # Use last loss
                if isinstance(loss_i, (float, int)):
                    # lower loss -> stronger reinforcement; clamp to [0,1]
                    reinforce_strength = float(max(0.0, min(1.0, 1.0 - float(loss_i))))
            # Use last dream features or brain_latent if available
            dream_emb = getattr(self, "last_dream_features", None)
            if dream_emb is None and hasattr(self, "last_brain_latent"):
                dream_emb = self.last_brain_latent.mean(dim=0) if getattr(self, "last_brain_latent").dim() > 1 else self.last_brain_latent
            if dream_emb is not None and reinforce_strength > 0.01 and hasattr(self, "theme"):
                matches = self.theme.find_closest_themes(dream_emb.detach(), top_k=3)
                for theme, sim in matches:
                    # delta scales with similarity and reinforce_strength
                    delta = 0.2 * sim * reinforce_strength
                    self.theme.reinforce_theme(theme.theme_id, delta_freq=delta, emergence_bonus=reinforce_strength)
        except Exception as e:
                # Strict mode: re-raise so trainer sees it; if you prefer non-fatal, convert to log
                if STRICT_DREAM:
                    raise
                else:
                    import logging
                    logging.getLogger(__name__).warning(f"[DreamEngine] reinforcement after NMDA failed: {e}")

        # --- Dream-gated RelMem plasticity (phase-locked) ---
        try:
            if self._relmem is not None and ripple_ctx is not None:
                coh = ripple_ctx.coherence
                phase_bin = ripple_ctx.phase_bin
                scale = float(self.cfg.stdp_gain) * float(self.cfg.ripple_rate_hz)

                # Hebbian updates only during coherent ripples in early replay phases
                if coh > 0.7 and phase_bin in [0, 1, 2]:
                    self._relmem.apply_hebbian()

                # WTA inhibition during late-phase high-coherence ripples
                if coh > 0.8 and phase_bin in [5, 6, 7]:
                    reps = int(min(3, max(1, round(scale))))
                    for _ in range(reps):
                        self._relmem.apply_wta()
        except Exception as e:
            if self.cfg.verbose:
                print(f"[Dreamâ†’RelMem] plasticity skipped: {e}")

        # Get ripple metrics if available
        ripple_metrics = {}
        if self.ripple.is_ready():
            ripple_metrics = {
                "coherence": self.ripple.get_coherence_metrics().get('current_coherence', 0.0),
                "phase": self.ripple.get_phase_info().get('phase_normalized', 0.0)
            }
        else:
            ripple_metrics = {"coherence": 0.0, "phase": 0.0}
        
        # Update meta-learner with cycle results
        strategy_success = gain > prior * 1.1  # 10% improvement threshold
        self.meta_learner.update('dream_cycle', 'cio_perturb', strategy_success, tokens.mean(dim=(0, 1)))

        # Theme → RelMem seeding (use synthesized themes as concepts)
        themes_bound = 0
        try:
            if hasattr(self, "theme") and hasattr(self.theme, "themes") and self._relmem is not None:
                for th in self.theme.themes[-5:]:  # recent few
                    emb = getattr(th, "embedding", None)
                    if torch.is_tensor(emb) and emb.numel() > 0:
                        self._relmem.bind_concept_by_vector(
                            emb.detach(),
                            op_name="emergent_theme",
                            meta={"theme_id": getattr(th, "theme_id", "unknown"), "source": "dream_theme"},
                            alpha=0.3
                        )
                        themes_bound += 1
                        if self.cfg.verbose:
                            print(f"[RelMem] Bound emergent theme → concept_id (theme_id={getattr(th, 'theme_id', 'unknown')})")
        except Exception as e:
            if self.cfg.verbose:
                print(f"[RelMem] Theme binding failed: {e}")

        # Consolidation metrics logging
        if self.cfg.verbose:
            print(f"[Consolidation] NMDA.len={nmda_buf}, passes={nmda_passes}, wormhole_mined={len(mined)}, themes_bound={themes_bound}")

        return {
            "cio_gain": gain,
            "nmda_loss_mean": float(sum(losses)/max(1, len(losses))),
            "nmda_buffer": int(nmda_buf),
            "nmda_passes": int(nmda_passes),
            "motifs": float(len(mined)),
            "dream_motifs": len(self.dream_motifs),
            "active_motifs": len([m for m in self.dream_motifs if m.success_rate > 0.5]),
            "avg_motif_entropy_reduction": np.mean([m.entropy_reduction for m in self.dream_motifs]) if self.dream_motifs else 0.0,
            "ripple_events": ripple_metrics.get('ripple_events', 0),
            "ripple_active_steps": ripple_metrics.get('ripple_active_steps', 0),
            "ripple_phase_coherence": ripple_metrics.get('ripple_phase_coherence', 0.0)
        }

    # ======================== NEW: Online Memory Utilization ========================
    @torch.no_grad()
    def micro_dream(self) -> None:
        """
        A single online 'tick' to keep motifs fresh and ripple substrate advancing.
        Safe no-op if submodules are absent.
        """
        try:
            if hasattr(self, 'curiosity') and self.curiosity is not None:
                if hasattr(self.curiosity, 'step_micro'):
                    _ = self.curiosity.step_micro()
        except Exception:
            pass
        try:
            if hasattr(self, 'nmda') and self.nmda is not None:
                if hasattr(self.nmda, 'tick_micro'):
                    _ = self.nmda.tick_micro()
        except Exception:
            pass
        try:
            # Decay motif TTL & evict dead motifs
            if hasattr(self, 'dream_motifs'):
                self.dream_motifs = [m for m in self.dream_motifs if hasattr(m, 'ttl') and m.ttl > 0]
                for m in self.dream_motifs:
                    if hasattr(m, 'ttl'):
                        m.ttl -= 1
        except Exception:
            pass

    @torch.no_grad()
    def retrieve_and_bias(self, demos, test_grid, relmem=None, topk_templates:int=5, brain_emb=None) -> dict:
        """
        Return a dict with:
          - 'op_bias':  {op_name: weight} aggregated from episodic memory + Wormhole templates
          - 'ebr_prior_scale' (optional): scalar scaling for EBR priors based on current theme
          - 'similar_experiences': list of retrieved episodes (for logging/analysis)
        All lookups are conservative and fail-safe.
        """
        out: dict = {"op_bias": {}, "similar_experiences": []}

        # 1) Episodic Memory Retrieval â†’ bias ops from similar past tasks
        # NO FALLBACKS - fail loudly if signals are broken
        if relmem is not None and brain_emb is not None and hasattr(relmem, 'get_similar_experiences'):
            similar_eps = relmem.get_similar_experiences(brain_emb, k=5)
            out["similar_experiences"] = similar_eps

            # Bias operations based on what worked in similar episodes
            if similar_eps:
                import logging
                logging.info(f"[DreamEngine][EpisodicRetrieval] Retrieved {len(similar_eps)} experiences, "
                           f"top_sim={similar_eps[0]['similarity']:.3f}, cid={similar_eps[0]['cid']}")

                bias_applied = []
                for episode in similar_eps:
                    sim = episode['similarity']
                    ops_used = episode.get('ops_used', {})

                    # Weight by similarity (high sim = strong bias)
                    for op, success in ops_used.items():
                        if isinstance(success, (int, float)) and success > 0:
                            bias_strength = sim * success * 0.5  # Scale by similarity and success
                            if bias_strength > 0.01:  # Only apply meaningful biases
                                out["op_bias"][op] = max(out["op_bias"].get(op, 0.0), bias_strength)
                                bias_applied.append(f"{op}={bias_strength:.3f}")

                if bias_applied:
                    logging.info(f"[DreamEngine][EpisodicBias] Applied: {', '.join(bias_applied[:5])}")
            else:
                import logging
                logging.info("[DreamEngine][EpisodicRetrieval] No similar experiences found (memory empty or low similarity)")

        # 2) Wormhole template macro-ops â†’ soft op bias (+0.25 like existing rail)
        try:
            macro_templates = []
            if self.wormhole is not None and hasattr(self.wormhole, 'library'):
                # Use current episodic op bias as context signal
                context_weights = dict(out["op_bias"])  # Already seeded by episodic retrieval above

                # Try context-aware retrieval first, fallback to score-based
                if hasattr(self.wormhole.library, 'get_templates_by_overlap'):
                    macro_templates = self.wormhole.library.get_templates_by_overlap(context_weights, max_count=topk_templates)
                elif hasattr(self.wormhole.library, 'get_templates_by_score'):
                    macro_templates = self.wormhole.library.get_templates_by_score(max_count=topk_templates)
                elif hasattr(self.wormhole, 'get_templates_by_score'):
                    macro_templates = self.wormhole.get_templates_by_score(max_count=topk_templates)
                else:
                    macro_templates = []

                # Apply context-modulated boost
                for t in macro_templates:
                    ov = float(getattr(t, "context_overlap", 0.0))
                    # Normalize overlap into [0.10, 0.40] to avoid overpowering episodic priors
                    boost = 0.10 + 0.30 * max(0.0, min(1.0, ov))

                    # Extract ops from template
                    if hasattr(t, 'get_primitive_ops'):
                        ops = t.get_primitive_ops()
                    elif hasattr(t, 'ops'):
                        ops = t.ops
                    else:
                        ops = []

                    for op in ops:
                        op_name = op[0] if isinstance(op, tuple) else op
                        out["op_bias"][op_name] = max(out["op_bias"].get(op_name, 0.0), boost)

                # Logging
                if self.cfg.verbose and macro_templates:
                    top_tpl = [(tpl.ops if hasattr(tpl, 'ops') else [], round(float(getattr(tpl, "context_overlap", 0.0)), 3))
                              for tpl in macro_templates[:3]]
                    print(f"[DreamEngine][Wormhole] Contextual templates: {top_tpl}")

        except Exception as e:
            import logging
            logging.warning(f"[DreamEngine] Wormhole retrieval failed: {e}")

        # 3) Theme â†’ optional global EBR prior scaling (ETS strength)
        try:
            if hasattr(self, 'theme') and self.theme is not None:
                if hasattr(self.theme, 'get_strength'):
                    s = float(self.theme.get_strength())
                    # Keep in a conservative range [0.6, 1.4]
                    out["ebr_prior_scale"] = max(0.6, min(1.4, s))
                elif hasattr(self.theme, 'themes') and len(self.theme.themes) > 0:
                    # Fallback: use theme count as rough strength proxy
                    strength = min(1.4, 0.8 + len(self.theme.themes) * 0.1)
                    out["ebr_prior_scale"] = strength
        except Exception:
            pass

        return out

    def get_strategy_recommendation(self, brain_emb: torch.Tensor = None) -> Dict[str, Any]:
        """
        Get strategy recommendation from MetaLearner.

        Returns dict with:
        - 'strategy': recommended strategy ('dsl', 'ebr', 'painter', or 'default')
        - 'confidence': confidence in recommendation (0-1)
        """
        try:
            if brain_emb is not None:
                strategy = self.meta_learner.recommend_strategy(brain_emb)
            else:
                strategy = self.meta_learner.recommend_strategy()

            # Calculate confidence based on success rate difference
            if strategy in self.meta_learner.strategy_success:
                successes = self.meta_learner.strategy_success[strategy]
                if successes:
                    confidence = np.mean(successes)
                else:
                    confidence = 0.5
            else:
                confidence = 0.5

            return {'strategy': strategy, 'confidence': float(confidence)}
        except Exception as e:
            import logging
            logging.debug(f"[DreamEngine] Strategy recommendation failed: {e}")
            return {'strategy': 'default', 'confidence': 0.5}

    def should_trigger_dream(self, novelty_score: float = 0.0, uncertainty: float = 0.0) -> bool:
        """
        Determine if dream cycle should be triggered based on curiosity signals.

        Args:
            novelty_score: Novelty/surprise measure (0-1)
            uncertainty: Solver uncertainty/entropy (0-1)

        Returns:
            True if dream should be triggered
        """
        # Thresholds for triggering
        novelty_threshold = 0.7
        uncertainty_threshold = 0.8

        # Trigger if either signal is high
        if novelty_score > novelty_threshold:
            return True
        if uncertainty > uncertainty_threshold:
            return True

        return False

    # ======================== End Online Memory Utilization ========================

    # ========================================================================
    # BIDIRECTIONAL DREAM â†” RELMEM SYNCHRONIZATION
    # ========================================================================

    def sync_concept_ecosystem(self, relmem, em_delta: float, step: int):
        """
        Bidirectional flow: successful Dream motifs â†’ RelMem concepts
                            strong RelMem concepts â†’ Dream motifs
        Creates unified semantic memory across both systems.

        Args:
            relmem: RelationalMemoryNeuro instance
            em_delta: EM change for adaptive synchronization
            step: Current training step
        """
        if relmem is None or not relmem.is_ready():
            return

        # === FLOW 1: Dream Motifs â†’ RelMem Concepts ===
        # Promote high-performing motifs to persistent concepts
        promoted_count = 0
        for motif in self.dream_motifs:
            if motif.success_rate > 0.7 and motif.usage_count > 5:
                # Convert dream pattern to concept
                pattern_normalized = F.normalize(motif.pattern, dim=-1)

                # Infer operation from pattern characteristics (semantic tagging)
                op_name = self._infer_operation_from_pattern(pattern_normalized)

                meta = {
                    'source': 'dream_motif',
                    'success_score': float(motif.success_rate),
                    'entropy_reduction': float(motif.entropy_reduction),
                    'usage_count': motif.usage_count,
                    'step': step,
                    'has_real_metrics': False  # Mark as inferred
                }

                cid = relmem.bind_concept_by_vector(
                    pattern_normalized,
                    op_name=op_name,
                    meta=meta,
                    alpha=0.6
                )

                if cid >= 0:
                    # Boost motif TTL if successfully bound
                    motif.ttl += 10
                    promoted_count += 1
                    if self.cfg.verbose:
                        logging.info(f"[Dreamâ†’RelMem] Promoted motif to concept {cid} ({op_name}, success={motif.success_rate:.3f})")

        # === FLOW 2: RelMem Concepts â†’ Dream Motifs ===
        # Extract top concepts and seed as dream templates
        seeded_count = 0
        if hasattr(relmem, 'concepts') and relmem.concepts:
            concept_scores = []
            for cid, concept_data in relmem.concepts.items():
                if relmem.concept_used[cid]:
                    ops_meta = concept_data.get('meta', {}).get('operations', {})
                    avg_success = np.mean(list(ops_meta.values())) if ops_meta else 0.0
                    concept_scores.append((cid, avg_success))

            # Top 5 concepts become dream motifs
            concept_scores.sort(key=lambda x: x[1], reverse=True)
            for cid, score in concept_scores[:5]:
                if score > 0.5:
                    concept_vec = relmem.concept_proto[cid]

                    # Check if already exists as motif (avoid duplicates)
                    already_exists = any(
                        F.cosine_similarity(m.pattern, concept_vec, dim=0).item() > 0.95
                        for m in self.dream_motifs
                    )

                    if not already_exists:
                        # Project to dream state_dim if needed
                        if concept_vec.shape[0] != self.cfg.state_dim:
                            if self._relmem_proj is not None:
                                concept_vec = self._relmem_proj(concept_vec.unsqueeze(0)).squeeze(0)
                            else:
                                continue  # Skip if can't project

                        success = self.add_dream_motif(concept_vec, ttl=int(score * 30))
                        if success:
                            seeded_count += 1
                            if self.cfg.verbose:
                                logging.info(f"[RelMemâ†’Dream] Seeded concept {cid} as motif (score={score:.3f})")

        if promoted_count > 0 or seeded_count > 0:
            logging.info(f"[Concept Sync] Promoted {promoted_count} motifs, seeded {seeded_count} concepts (step={step})")

    def _infer_operation_from_pattern(self, pattern: torch.Tensor) -> str:
        """Infer likely DSL operation from pattern activation profile"""
        # Simple heuristic - could be learned
        pattern_cpu = pattern.detach().cpu()

        # Check for symmetry patterns (balanced activations)
        if torch.abs(pattern_cpu - pattern_cpu.flip(0)).mean() < 0.1:
            return "symmetry"

        # Check for rotation patterns (circular structure in activation)
        if pattern_cpu.std() > 0.5:
            return "transform"

        # Check for color patterns (sparse activations)
        if (pattern_cpu > pattern_cpu.mean() + pattern_cpu.std()).sum() < len(pattern_cpu) * 0.2:
            return "color"

        # Default
        return "pattern"

    def train_step(self, slot_vecs, target=None):
        """
        One training step for Dream pretraining.
        Expects slot_vecs of shape [B, K, D] or [B, D]; returns a scalar loss tensor with grad.

        BitterBot SOTA Patch 01: robust dream pretrain + NMDA seeding + real gradients
        """
        import torch.nn.functional as F
        import torch.nn as nn
        device = self.device if hasattr(self, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')

        if slot_vecs is None:
            raise RuntimeError("[DreamEngine] train_step called with slot_vecs=None")
        if not torch.is_tensor(slot_vecs):
            slot_vecs = torch.as_tensor(slot_vecs, device=device)

        # Normalize to [B, K, D]
        if slot_vecs.dim() == 2:
            slot_vecs = slot_vecs.unsqueeze(1)
        B, K, D = slot_vecs.shape

        # --- BitterBot: robust loss composition (always tensor, never .item()) ---
        losses = []

        # Color prediction head (auxiliary supervision from grid target if provided)
        if target is not None and torch.is_tensor(target):
            if target.dim() == 2:
                target = target.unsqueeze(0)
            target = target.to(device)
            pooled = slot_vecs.mean(dim=1)  # [B, D]
            color_logits = self._dream_color_head(pooled)  # [B, 10]
            # Weak supervision: dominant color of target grid as label
            with torch.no_grad():
                # histogram over 10 ARC colors
                H, W = target.shape[-2], target.shape[-1]
                flat = target.view(B, -1).clamp_min(0) % 10
                mode_color = torch.mode(flat, dim=1)[0].long()
            ce = F.cross_entropy(color_logits, mode_color)
            losses.append(ce)

        # Operation-bias entropy regularizer (maximize structural signal; reduce entropy)
        pooled = slot_vecs.mean(dim=1)  # [B, D]
        op_logits = self._dream_opbias_head(pooled)  # [B, 41]
        probs = F.softmax(op_logits, dim=-1)
        ent = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        # Minimize entropy -> add as negative (maximize -H)
        op_bias_entropy_loss = -0.01 * ent
        losses.append(op_bias_entropy_loss)

        # Aggregate
        loss_value = torch.stack([l if torch.is_tensor(l) else torch.tensor(float(l), device=device)
                                  for l in losses]).sum()

        # --- BitterBot: seed NMDA buffer during pretrain (200+ experiences) ---
        try:
            with torch.no_grad():
                # Project to NMDA state_dim if needed
                # Use last layer before heads when available; otherwise pooled slot_vecs
                state_vec = pooled  # [B, D]
                # Ensure dimensionality matches NMDA state_dim
                if hasattr(self, 'nmda') and hasattr(self.nmda, 'state_dim'):
                    sd = int(self.nmda.state_dim)
                    if state_vec.shape[-1] != sd:
                        # Use a small on-the-fly projection buffer (created once)
                        if not hasattr(self, '_nmda_proj'):
                            self._nmda_proj = nn.Linear(state_vec.shape[-1], sd, device=state_vec.device)
                            nn.init.xavier_uniform_(self._nmda_proj.weight)
                        state_vec = self._nmda_proj(state_vec)

                # Add diversity to synthetic experiences
                noise = 0.05 * torch.randn_like(state_vec)  # Increased from 0.01
                next_state = (state_vec + noise).detach()
                for b in range(state_vec.size(0)):
                    # Varied action and reward for diversity
                    action = random.randint(0, 10)
                    reward = 0.01 + 0.1 * random.random()
                    self.nmda.store_dream_memory(
                        state=state_vec[b].detach(),
                        action=action,
                        reward=reward,
                        next_state=next_state[b].detach(),
                        phase=(random.random()*math.tau) if hasattr(math, 'tau') else random.random()*6.283
                    )
        except Exception as e:
            if STRICT_DREAM:
                raise

        # --- BitterBot: force early themes & wormhole seeding from latents ---
        try:
            if hasattr(self, "theme"):
                # Build tiny faux sequence: repeat pooled -> [B, T=K, D]
                lat_seq = slot_vecs if slot_vecs.dim() == 3 else slot_vecs.unsqueeze(1)
                added = self.seed_motifs_from_latents(lat_seq, ctx=None, wormhole=getattr(self, 'wormhole', None), pretrain=True)
                # If themes empty, synthesize from average tokens
                if getattr(self.theme, 'synthesis_count', 0) < 1:
                    fake_labels = torch.arange(min(8, lat_seq.size(1)), device=device)
                    pooled_seq = lat_seq.mean(dim=1, keepdim=True)  # [B,1,D]
                    self.theme.process_dream_themes(pooled_seq, fake_labels)
                    self.theme.synthesize_emergent_themes()
        except Exception as e:
            # non-fatal in pretrain
            pass

        return loss_value

    def parameters(self):
        """Yield parameters from internal nn modules and discovered submodules."""
        seen = set()
        def walk(obj):
            if id(obj) in seen: return
            seen.add(id(obj))
            import torch.nn as nn
            if isinstance(obj, nn.Module):
                for p in obj.parameters():
                    yield p
                for name, sub in vars(obj).items():
                    for q in walk(sub):
                        yield q
            else:
                for name, sub in getattr(obj, "__dict__", {}).items():
                    if id(sub) in seen: continue
                    if hasattr(sub, "parameters") and callable(sub.parameters):
                        try:
                            for p in sub.parameters():
                                yield p
                        except Exception:
                            for q in walk(sub):
                                yield q
                    else:
                        for q in walk(sub):
                            yield q
        for p in walk(self):
            yield p

    def save_state(self, path: str):
        """Save minimal state for pretrain (heads + themes if present)."""
        state = {}
        try:
            state["color_head"] = {k: v.cpu() for k, v in self._dream_color_head.state_dict().items()}
            state["opbias_head"] = {k: v.cpu() for k, v in self._dream_opbias_head.state_dict().items()}
        except Exception:
            pass
        try:
            if hasattr(self, "theme") and hasattr(self.theme, "state_dict"):
                state["themes"] = {k: v.cpu() for k, v in self.theme.state_dict().items()}
        except Exception:
            pass
        import torch
        torch.save(state, path)

    def load_state(self, path: str):
        import os, torch
        if not os.path.exists(path): return
        state = torch.load(path, map_location=self.device, weights_only=False)
        if "color_head" in state:
            try:
                self._dream_color_head.load_state_dict(state["color_head"])
            except Exception:
                pass
        if "opbias_head" in state:
            try:
                self._dream_opbias_head.load_state_dict(state["opbias_head"])
            except Exception:
                pass