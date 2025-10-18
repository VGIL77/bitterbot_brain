#!/usr/bin/env python3
"""
Simplified Direct HRM-TOPAS Training (robust version)
- GradScaler()
- device-aware autocast
- optional HRM->TOPAS best-effort bridge
- skip steps if logits missing (no dummy grads)
"""

import torch

# GPU-FIRST global default (prevents stray CPU tensor allocations)
if torch.cuda.is_available():
    torch.set_default_device("cuda")

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import sys, os, logging
import argparse
import threading
import time
import json
import traceback
import numpy as np
import yaml
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from trainers.arc_dataset_loader import ARCDataset
try:
    from arc2_dataset_loader import ARC2Dataset
except ImportError:
    ARC2Dataset = None
from models.topas_arc_60M import TopasARC60M, ModelConfig
from models.brain_graph import BrainGraph
from neuroplanner import NeuroPlanner
import logging
import torch.nn.functional as F
from collections import defaultdict

# Set up logger for NeuroPlanner wrapper
logger = logging.getLogger(__name__)

# Global skip tracker for signal purity monitoring
class SkipTracker:
    """Track all training skips for transparency and debugging."""
    def __init__(self):
        self.skips = defaultdict(int)
        self.skip_details = []

    def log_skip(self, reason: str, details: str = "", global_step: int = 0):
        """Log a skip event with full transparency."""
        self.skips[reason] += 1
        self.skip_details.append({
            'reason': reason,
            'details': details,
            'step': global_step,
            'timestamp': time.time()
        })
        # Always log to console for transparency
        logger.warning(f"[SKIP] {reason}: {details} (step={global_step}, total={self.skips[reason]})")
        # Only keep last 1000 skip details in memory
        if len(self.skip_details) > 1000:
            self.skip_details = self.skip_details[-1000:]

    def get_summary(self) -> dict:
        """Get summary of all skips."""
        return {
            'total_skips': sum(self.skips.values()),
            'skip_counts': dict(self.skips),
            'recent_skips': self.skip_details[-10:]  # Last 10
        }

SKIP_TRACKER = SkipTracker()


class NeuroPlannerWrapper(NeuroPlanner):
    """Wrapper for NeuroPlanner to provide TOPAS-compatible interface."""

    def __init__(self, config_dict: dict):
        super().__init__(config_dict)
        self._carry = None
        self._fallback_count = 0
        self._total_calls = 0
        # Task ID mapping for unique puzzle identification
        self._task_id_map = {}  # str -> int mapping
        self._next_task_id = 0
        # Metrics tracking for signal purity
        self._metrics = {
            'total_calls': 0,
            'fallback_calls': 0,
            'missing_task_id': 0,
            'successful_calls': 0
        }

    def grid_to_tokens(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Convert ARC grid to tokens (synchronized with TOPAS grid_to_tokens).
        Accepts float grids in [0,1] or integer grids in [0..9].
        Input: [B,H,W]  â†'  Output: [B,H*W] (long)
        """
        if torch.is_floating_point(grid):
            grid = torch.round(grid * 9.0)  # 9 = NUM_COLORS - 1
        return grid.reshape(grid.size(0), -1).clamp(0, 9).long()

    def _get_task_id(self, task_id_str: str) -> int:
        """Convert string task ID to unique integer ID for NeuroPlanner."""
        if task_id_str not in self._task_id_map:
            self._task_id_map[task_id_str] = self._next_task_id
            self._next_task_id += 1
        return self._task_id_map[task_id_str]

    def encode(self, input_grid: torch.Tensor, task_id: str = None) -> torch.Tensor:
        """
        Extract latent representations from NeuroPlanner for TOPAS integration.

        Args:
            input_grid: [B, H, W] or [B, C, H, W] input grid
            task_id: String task identifier (required for proper puzzle embeddings)

        Returns:
            hrm_latents: [B, hidden_size] tensor for TOPAS FiLM conditioning
        """
        self._total_calls += 1
        self._metrics['total_calls'] += 1

        if input_grid is None:
            self._metrics['fallback_calls'] += 1
            logger.warning(f"[NeuroPlanner] Encode received None input_grid (call #{self._total_calls})")
            return self._create_fallback_latents(1, torch.device('cuda'))

        try:
            # Handle different input shapes
            if input_grid.dim() == 4:  # [B, C, H, W]
                input_grid = input_grid.squeeze(1)  # Remove channel dim, assume C=1
            elif input_grid.dim() == 2:  # [H, W]
                input_grid = input_grid.unsqueeze(0)  # Add batch dim

            batch_size = input_grid.shape[0]
            device = input_grid.device
            H, W = input_grid.shape[-2:]

            # Convert real grid to tokens instead of using dummy data
            tokens = self.grid_to_tokens(input_grid)  # [B, H*W]

            # Handle sequence length - account for puzzle embedding length
            current_seq_len = tokens.shape[1]
            # NeuroPlanner expects seq_len, but puzzle embeddings DON'T add to input sequence
            expected_seq_len = self.config.seq_len

            if current_seq_len != expected_seq_len:
                if current_seq_len < expected_seq_len:
                    # Pad with zeros (empty color)
                    padding = expected_seq_len - current_seq_len
                    tokens = F.pad(tokens, (0, padding), mode='constant', value=0)
                else:
                    # Crop to expected length
                    tokens = tokens[:, :expected_seq_len]

                logger.debug(f"[NeuroPlanner] Adjusted sequence length from {current_seq_len} to {expected_seq_len} "
                            f"for grid shape {H}x{W} (call #{self._total_calls})")

            # Create proper batch for NeuroPlanner with real data
            # Use real task IDs for proper puzzle-specific learning
            if task_id is not None:
                task_id_int = self._get_task_id(task_id)
                puzzle_ids = torch.full((batch_size,), task_id_int, dtype=torch.long, device=device)
            else:
                # Fallback for backward compatibility, but log warning and track
                self._metrics['missing_task_id'] += 1
                logger.warning(f"[NeuroPlanner] No task_id provided (call #{self._total_calls}) - using zero fallback (total: {self._metrics['missing_task_id']})")
                puzzle_ids = torch.zeros(batch_size, dtype=torch.long, device=device)

            batch = {
                "inputs": tokens,  # Real grid tokens
                "puzzle_identifiers": puzzle_ids  # Real task IDs
            }

            # Initialize carry if needed
            if self._carry is None or self._carry.inner_carry.z_H.shape[0] != batch_size:
                self._carry = self.initial_carry(batch)

            # Run NeuroPlanner forward pass
            new_carry, outputs = self.forward(self._carry, batch)
            self._carry = new_carry

            # Extract useful latents from carry state
            # Use z_H (high-level reasoning state) as the primary latent
            z_H = new_carry.inner_carry.z_H  # [B, seq_len, hidden_size]

            # Pool across sequence dimension to get [B, hidden_size]
            if z_H.dim() == 3:
                hrm_latents = z_H.mean(dim=1)  # Mean pooling
            else:
                hrm_latents = z_H

            self._metrics['successful_calls'] += 1
            # CRITICAL: Keep gradients flowing to HRM! Let NeuroPlanner learn from TOPAS feedback
            return hrm_latents  # Gradient flow enabled - HRM can now learn task patterns!

        except Exception as e:
            # Comprehensive logging for fallback scenario
            self._fallback_count += 1
            self._metrics['fallback_calls'] += 1
            logger.warning(f"[NeuroPlanner] Encode failed (call #{self._total_calls}, fallback #{self._fallback_count}): {e}")
            logger.warning(f"[NeuroPlanner] Input details - shape: {input_grid.shape if input_grid is not None else 'None'}, "
                          f"dtype: {input_grid.dtype if input_grid is not None else 'None'}, "
                          f"device: {input_grid.device if input_grid is not None else 'None'}")

            # Log fallback statistics every 100 calls
            if self._total_calls % 100 == 0:
                fallback_rate = (self._fallback_count / self._total_calls) * 100
                logger.info(f"[NeuroPlanner] Fallback statistics: {self._fallback_count}/{self._total_calls} "
                           f"({fallback_rate:.1f}%) calls used fallback")

            # Return meaningful fallback instead of None - ensure device is GPU
            fallback_device = device if device.type == 'cuda' else torch.device('cuda')
            return self._create_fallback_latents(batch_size, fallback_device)

    def _create_fallback_latents(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """REMOVED: Fallback latents pollute training with zero gradients."""
        # Fail loudly instead of returning zeros - forces upstream to handle properly
        raise RuntimeError(f"[NeuroPlanner] Encoding failed - no fallback allowed (would pollute gradients with zeros)")

    def get_signal_purity_metrics(self) -> dict:
        """Get metrics for signal purity analysis."""
        metrics = self._metrics.copy()
        if metrics['total_calls'] > 0:
            metrics['fallback_rate'] = metrics['fallback_calls'] / metrics['total_calls']
            metrics['success_rate'] = metrics['successful_calls'] / metrics['total_calls']
            metrics['missing_task_id_rate'] = metrics['missing_task_id'] / metrics['total_calls']
        else:
            metrics['fallback_rate'] = 0.0
            metrics['success_rate'] = 0.0
            metrics['missing_task_id_rate'] = 0.0
        metrics['unique_tasks'] = len(self._task_id_map)
        return metrics

    def get_padded_tokens(self, input_grid: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Get properly padded tokens that match NeuroPlanner's internal processing.
        Returns both padded tokens and original sequence length for supervision.

        Args:
            input_grid: [B, H, W] input grid

        Returns:
            tuple of (padded_tokens, original_seq_len) for supervision alignment
        """
        if input_grid is None:
            return None, 0

        # Handle different input shapes
        if input_grid.dim() == 4:  # [B, C, H, W]
            input_grid = input_grid.squeeze(1)
        elif input_grid.dim() == 2:  # [H, W]
            input_grid = input_grid.unsqueeze(0)

        # Convert to tokens
        tokens = self.grid_to_tokens(input_grid)  # [B, H*W]
        original_seq_len = tokens.shape[1]

        # Apply same padding logic as encode() method
        expected_seq_len = self.config.seq_len
        if original_seq_len != expected_seq_len:
            if original_seq_len < expected_seq_len:
                padding = expected_seq_len - original_seq_len
                tokens = F.pad(tokens, (0, padding), mode='constant', value=0)
            else:
                tokens = tokens[:, :expected_seq_len]
                original_seq_len = expected_seq_len

        return tokens, original_seq_len
from trainers.self_play import SelfPlayBuffer  # used for storing dopamine rewards
from trainers.self_critique.counterexamples import CounterexampleGenerator, Task  # Task wrapper + counterexamples
from trainers.self_critique.star_bootstrapper import STaRBootstrapper           # STaR trace gen + verification
from trainers.self_critique.consistency import ConsistencyEnforcer               # enforce consistency across valid traces
from trainers.augmentation.deep_program_discoverer import mine_deep_programs                  # deep DSL programs miner (6â€“10 ops)
from models.policy_nets import OpPolicyNet, op_logits_to_bias                              # policy-guided search
from collections import Counter, deque
from typing import Callable
import math, statistics, time
import numpy as np
import random
import hashlib   # NEW

# Alpha-ARC X additions
try:
    from trainers.puct_search import puct_search
except Exception:
    puct_search = None
try:
    from trainers.replay import PrioritizedReplay
except Exception:
    PrioritizedReplay = None
try:
    from trainers.near_miss import near_miss_repair
except Exception:
    near_miss_repair = None

def _canonical_puzzle_id(task_id, modulo: int = 1000) -> int:
    """
    Map any task identifier (int/str/uuid) to a stable int in [0, modulo).
    Matches HRM config num_puzzle_identifiers=1000.
    """
    try:
        return int(task_id) % modulo
    except Exception:
        h = int(hashlib.sha1(str(task_id).encode("utf-8")).hexdigest()[:8], 16)
        return h % modulo

# =========================
# Dopamine & Nightmare Core
# =========================

# Global state shared across training
op_success_count = Counter()          # track operations in successful traces (for planner op_bias)
recent_failures: List[Any] = []       # queue of failed counterexamples for nightmares
rolling_em = deque(maxlen=200)        # rolling window of EM to estimate failure pressure

# =========================
# Additional Dopamine Helpers (Production-Grade)
# =========================

def _extract_entropy_reduction(dream_stats) -> float:
    try:
        if isinstance(dream_stats, dict):
            if "entropy_reduction" in dream_stats:
                return float(dream_stats["entropy_reduction"])
            if "entropy_before" in dream_stats and "entropy_after" in dream_stats:
                eb = float(dream_stats["entropy_before"]); ea = float(dream_stats["entropy_after"])
                return max(0.0, eb - ea)
    except Exception:
        pass
    return 0.0

def _extract_mdl_gain(mined_templates) -> float:
    try:
        if not mined_templates:
            return 0.0
        gains = []
        for t in mined_templates:
            if isinstance(t, dict) and "mdl_gain" in t:
                gains.append(float(t["mdl_gain"]))
            else:
                gains.append(1.0)
        return float(sum(gains))
    except Exception:
        return 0.0

def _novelty_estimate(enc_inp: Tuple[Tuple[int,int], Tuple[int,...]], buffer, k: int = 64) -> float:
    try:
        (_, fa) = enc_inp
        fa = list(fa)
        if not hasattr(buffer, "buffer") or len(buffer.buffer) == 0:
            return 1.0
        sample = buffer.buffer[-k:] if len(buffer.buffer) > k else buffer.buffer
        def sim(a, b):
            enc_b = b[0]   # (enc_inp, enc_out) OR (enc_inp, enc_out, score)
            (_, fb) = enc_b
            n = min(len(fa), len(fb))
            if n == 0: return 0.0
            eq = sum(1 for i in range(n) if fa[i] == fb[i])
            return eq / n
        mx = 0.0
        for item in sample:
            try:
                mx = max(mx, sim(enc_inp, item))
            except Exception:
                continue
        return max(0.0, 1.0 - mx)
    except Exception:
        return 0.5

def _squash(x: float, temp: float = 1.0) -> float:
    try:
        return math.tanh(x / max(1e-6, temp))
    except Exception:
        return 0.0

# Old dopamine function REMOVED - replaced by euphoric version below (line ~567)

def _stringify_ops(ops: Any) -> List[str]:
    out = []
    if not ops:
        return out
    for op in ops:
        if isinstance(op, dict):
            try:
                key = f"composite_{hash(str(sorted(op.items())))}"
            except Exception:
                key = f"composite_{hash(str(op))}"
            out.append(key)
        else:
            out.append(str(op))
    return out

def _as_program(op_seq):
    """
    Canonicalize into List[Tuple[str, Dict]].
    Accepts: List[str], List[Tuple[str, Dict]], DSLProgram, or None.
    """
    if not op_seq:
        return []
    try:
        # DSLProgram
        if hasattr(op_seq, "ops") and hasattr(op_seq, "params"):
            return [(str(o), dict(p or {})) for o, p in zip(op_seq.ops, op_seq.params)]
        # List[Tuple[str, Dict]]
        if isinstance(op_seq[0], (tuple, list)) and len(op_seq[0]) == 2:
            return [(str(op), dict(params or {})) for op, params in op_seq]
        # List[str]
        return [(str(op), {}) for op in op_seq]
    except Exception:
        return []

def _hamming(a: torch.Tensor, b: torch.Tensor) -> int:
    """
    Compute Hamming distance between two tensors.
    """
    if a is None or b is None or a.shape != b.shape:
        return 10**9
    return (a.view(-1) != b.view(-1)).sum().item()

def _sc_run_star(star_bootstrapper, task, planner_bias, n: int) -> list:
    """
    Self-consistency booster: run STaR N times with small jitter in priors.
    Return list of valid traces.
    """
    traces_all = []
    import copy, random
    for _ in range(max(1, n)):
        pb = None
        if isinstance(planner_bias, dict):
            # tiny jitter (root-Dirichlet-like) for diversity
            pb = {k: max(1e-6, v * (0.9 + 0.2 * random.random())) for k, v in planner_bias.items()}
        tr = star_bootstrapper.generate_diverse_traces(task, n_traces=8, planner_op_bias=pb or planner_bias)
        traces_all.extend(tr)
    return traces_all

def _puct_plan_stepwise(model, demos, test_input, target_grid, cli_args, device) -> list:
    """
    Compose a program by running PUCT selection sequentially for puct_depth steps.
    Each step: run small PUCT on current grid to pick next op, apply via DSL shim.
    """
    if puct_search is None:
        return []
    state_grid = test_input.clone()
    program_ops = []

    # === CEGIS: Mine certificates from demos AND WGO ===
    from trainers.certificates import mine_certificates, certificate_penalty
    certs = mine_certificates(demos)

    # Merge WGO certificates (from pretrainer predictions) if available
    try:
        out = model.forward_pretraining(test_input.unsqueeze(0), target_shape=target_grid.shape[-2:], demos=demos)
        extras = out.get("extras", {}) if isinstance(out, dict) else {}
        wgo_certs = extras.get("wgo_certificates", {})
        if wgo_certs:
            # Merge WGO certificates with demo-based ones (WGO takes precedence as "teacher")
            certs = {**certs, **wgo_certs}
            logger.info(f"[CEGIS-WGO] Merged {len(wgo_certs)} WGO certificates with {len(certs)} total")
    except Exception as e:
        logger.debug(f"[CEGIS-WGO] Failed to extract WGO certificates: {e}")

    if certs:
        logger.info(f"[CEGIS] Using {len(certs)} certificates: {list(certs.keys())}")

    def priors_fn(grid_s):
        # one forward to get policy priors (dsl_op_logits) on this state
        try:
            out = model.forward_pretraining(grid_s.unsqueeze(0), target_shape=target_grid.shape[-2:], demos=demos)
            extras = out.get("extras", {}) if isinstance(out, dict) else {}
            logits = extras.get("dsl_op_logits") or extras.get("policy_logits")
            if logits is None: return {}
            probs = torch.softmax(logits[0], dim=-1).detach().cpu().numpy().tolist()
            from models.dsl_registry import DSL_OPS
            prior = {op: float(p) for op, p in zip(DSL_OPS, probs)}
            # root Dirichlet with deterministic seeding for reproducibility
            import numpy as np
            if len(prior) and cli_args.root_dirichlet_eps > 0:
                eps = cli_args.root_dirichlet_eps
                alpha = cli_args.root_dirichlet_alpha
                # FIXED: Use seeded RNG for determinism (inherits global seed from _seed_everything)
                # If full determinism is critical, create local RNG with fixed seed
                noise = np.random.dirichlet([alpha]*len(prior))
                keys = list(prior.keys())
                for i, k in enumerate(keys):
                    prior[k] = float((1-eps)*prior[k] + eps*noise[i])
            return prior
        except Exception:
            return {}

    def value_fn(grid_s):
        try:
            out = model.forward_pretraining(grid_s.unsqueeze(0), target_shape=target_grid.shape[-2:], demos=demos)
            ex = out.get("extras", {}) if isinstance(out, dict) else {}

            # === FUSE THREE VALUE SIGNALS ===
            # 1. Bridge value_logit (HRM-guided)
            bridge_value = ex.get("value_logit")
            v_bridge = torch.sigmoid(bridge_value)[0].item() if bridge_value is not None else 0.1

            # 2. WGO critic (pretrainer teacher)
            wgo_value = ex.get("wgo_value")
            v_wgo = float(wgo_value) if wgo_value is not None else 0.1

            # 3. TOPAS critic head (if available in extras)
            topas_value = ex.get("topas_critic")
            v_topas = torch.sigmoid(topas_value)[0].item() if topas_value is not None else 0.1

            # Calibrated ensemble: logit-averaging for sharper value
            # Weights: bridge=0.5 (main), wgo=0.3 (teacher), topas=0.2 (execution)
            ensemble_value = 0.5 * v_bridge + 0.3 * v_wgo + 0.2 * v_topas

            # CEGIS: soft penalty for certificate violations
            pen = certificate_penalty(certs, grid_s)
            certificate_mode = getattr(cli_args, 'certificates', 'soft')
            if certificate_mode in ['soft', 'hard']:
                adjusted = max(0.0, ensemble_value - 0.5 * pen)  # Soft penalty
                return adjusted
            return ensemble_value
        except Exception:
            return 0.1

    def expand_fn(grid_s, op):
        try:
            # Sample parameters using param_priors if available
            params = {}
            try:
                out = model.forward_pretraining(grid_s.unsqueeze(0), target_shape=target_grid.shape[-2:], demos=demos)
                ex = out.get("extras", {}) if isinstance(out, dict) else {}
                ppri = ex.get("param_priors", {})

                # Sample params based on operation type using param_priors
                if ppri:
                    # For translate ops: sample dx/dy from logits
                    if op in ['translate'] and 'dx_logits' in ppri and 'dy_logits' in ppri:
                        dx_logits = ppri['dx_logits']
                        dy_logits = ppri['dy_logits']
                        if dx_logits.numel() > 0 and dy_logits.numel() > 0:
                            dx_idx = torch.argmax(dx_logits[0]).item()
                            dy_idx = torch.argmax(dy_logits[0]).item()
                            params['dx'] = dx_idx - 2  # Map from [0,3] to [-2,1]
                            params['dy'] = dy_idx - 2

                    # For color_map ops: sample target color from histogram
                    elif op in ['color_map', 'recolor'] and 'color_map_logits' in ppri:
                        color_logits = ppri['color_map_logits']
                        if color_logits.numel() > 0:
                            color_idx = torch.argmax(color_logits[0]).item()
                            params['color'] = color_idx

                    # For flip ops: sample axis from flip_axis_logits
                    elif op in ['flip_h', 'flip_v'] and 'flip_axis_logits' in ppri:
                        axis_logits = ppri['flip_axis_logits']
                        if axis_logits.numel() > 0:
                            axis_idx = torch.argmax(axis_logits[0]).item()
                            params['axis'] = 'h' if axis_idx == 0 else 'v'

                    # For crop ops: sample bbox from crop_bbox_logits
                    elif op in ['crop_bbox'] and 'crop_bbox_logits' in ppri:
                        bbox_logits = ppri['crop_bbox_logits']
                        if bbox_logits.numel() >= 4:
                            # Normalize to grid dimensions
                            H, W = grid_s.shape[-2:]
                            y0 = int(torch.sigmoid(bbox_logits[0, 0]).item() * H)
                            x0 = int(torch.sigmoid(bbox_logits[0, 1]).item() * W)
                            y1 = int(torch.sigmoid(bbox_logits[0, 2]).item() * H)
                            x1 = int(torch.sigmoid(bbox_logits[0, 3]).item() * W)
                            params['bbox'] = (y0, x0, y1, x1)

                    # For scale ops: use scale_logits
                    elif op in ['scale', 'resize_nn'] and 'scale_logits' in ppri:
                        scale_logits = ppri['scale_logits']
                        if scale_logits.numel() >= 2:
                            sx = torch.sigmoid(scale_logits[0, 0]).item() * 3 + 0.5  # Range [0.5, 3.5]
                            sy = torch.sigmoid(scale_logits[0, 1]).item() * 3 + 0.5
                            params['scale_x'] = sx
                            params['scale_y'] = sy
            except Exception as e:
                # Graceful fallback if param sampling fails
                pass

            # Apply operation with sampled params
            if params:
                ng = model.dsl.apply((op, params), grid_s)
            else:
                ng = model.dsl.apply(op, grid_s)

            # CEGIS: hard prune if enabled and grid totally violates certificates
            certificate_mode = getattr(cli_args, 'certificates', 'soft')
            if certificate_mode == 'hard':
                if certificate_penalty(certs, ng) >= 1.0:
                    return grid_s  # Neutral no-move (keeps PUCT stable)

            return ng
        except Exception:
            return grid_s

    # === ADAPTIVE SEARCH PARAMETERS FROM BRIDGE ===
    # Use bridge control_signals to adapt depth/beam dynamically
    base_depth = int(cli_args.puct_depth)
    base_nodes = int(cli_args.puct_nodes)

    # Try to get adaptive params from bridge if available
    adapted_depth = base_depth
    adapted_nodes = base_nodes
    try:
        if hasattr(model, 'hrm_bridge') and model.hrm_bridge is not None:
            # Get control signals from a forward pass
            out = model.forward_pretraining(test_input.unsqueeze(0), target_shape=target_grid.shape[-2:], demos=demos)
            ex = out.get("extras", {}) if isinstance(out, dict) else {}
            control_signals = ex.get("control_signals", {})

            if control_signals:
                # Use bridge's compute_adaptive_search_params
                adapted_depth, adapted_beam = model.hrm_bridge.compute_adaptive_search_params(
                    control_signals, base_depth, base_beam_width=10  # Base beam (not directly used in PUCT but for reference)
                )

                # Map beam width to PUCT nodes (more beam = more PUCT simulations)
                adapted_nodes = int(base_nodes * (adapted_beam / 10.0))  # Scale nodes proportionally
                adapted_nodes = max(100, min(adapted_nodes, 5000))  # Clamp to reasonable range

                if cli_args.verbose:
                    logger.info(f"[Adaptive-PUCT] depth: {base_depth}→{adapted_depth}, nodes: {base_nodes}→{adapted_nodes}")
    except Exception as e:
        logger.debug(f"[Adaptive-PUCT] Failed to get adaptive params: {e}")

    for _ in range(max(1, adapted_depth)):
        best_op, _ = puct_search(
            state_grid, priors_fn, value_fn, expand_fn,
            max_nodes=adapted_nodes,
            c_puct=float(cli_args.c_puct)
        )
        try:
            state_grid = model.dsl.apply(best_op, state_grid)
            program_ops.append(best_op)
            if torch.is_tensor(target_grid) and (state_grid.shape == target_grid.shape) and (state_grid == target_grid).all():
                break
        except Exception:
            break
    return program_ops

# =========================
# Dopamine state (EMA + refractory)
# =========================
class _EMA:
    def __init__(self, beta=0.9, init=0.0):
        self.beta = beta
        self.m = init
        self.initialized = False
    def update(self, x: float) -> float:
        old_m = self.m  # Save old value BEFORE update for advantage calculation
        if not self.initialized:
            self.m = x
            self.initialized = True
        else:
            self.m = self.beta * self.m + (1 - self.beta) * x
        return old_m  # Return PRE-update value so advantage = R - old_ema works correctly
    def value(self) -> float:
        return self.m

_dopamine_ema = _EMA(beta=0.9, init=0.0)
_last_dopamine_step = -10**9  # refractory tracking

# =========================
# Enhanced Dopamine Helper Functions
# =========================
def _extract_program_len(programs: List) -> int:
    """Extract representative program length from operations list."""
    if not programs:
        return 0
    total_len = 0
    count = 0
    for prog in programs:
        if isinstance(prog, (list, tuple)):
            total_len += len(prog)
            count += 1
        elif hasattr(prog, '__len__'):
            total_len += len(prog)
            count += 1
    return total_len // max(1, count)

def _safe_iou(pred_grid: torch.Tensor, target_grid: torch.Tensor) -> float:
    """Safely compute IoU between prediction and target grids."""
    try:
        if pred_grid.shape != target_grid.shape:
            return 0.0
        pred_flat = pred_grid.view(-1)
        target_flat = target_grid.view(-1)
        intersection = (pred_flat == target_flat).sum().float()
        union = pred_flat.numel()
        return (intersection / union).item() if union > 0 else 0.0
    except Exception:
        return 0.0

def _extract_entropy_reduction(dream_stats: Optional[Dict]) -> float:
    """Extract entropy reduction metric from dream statistics."""
    if not isinstance(dream_stats, dict):
        return 0.0
    return float(dream_stats.get('entropy_reduction', 0.0))

def _extract_mdl_gain(mined_templates: List) -> float:
    """Extract MDL gain from mined program templates."""
    if not mined_templates:
        return 0.0
    # Simple heuristic: more complex templates = higher MDL gain
    total_complexity = sum(len(str(t)) for t in mined_templates)
    return min(1.0, total_complexity / 100.0)

def _novelty_estimate(encoded_input: tuple, buffer: Any, k: int = 64) -> float:
    """Estimate novelty of input relative to buffer contents."""
    if not hasattr(buffer, 'buffer') or len(buffer.buffer) < k:
        return 1.0  # High novelty if buffer sparse
    try:
        # Count similar patterns in recent buffer
        recent = buffer.buffer[-k:]
        matches = sum(1 for inp, _ in recent if inp == encoded_input)
        return max(0.0, 1.0 - matches / k)
    except Exception:
        return 0.5

def _dopamine_score_euphoric(em: float, acc: float, iou: float,
                            prev_acc: float = None, ema_acc: float = None,
                            program_len: int = 0, entropy_red: float = 0.0,
                            mdl_gain: float = 0.0, novelty: float = 0.0,
                            Lmax: int = 12) -> tuple:
    """
    EUPHORIC DOPAMINE: Non-linear reward with surprise, velocity, and "nut-bust" dynamics.

    Returns:
        (total_reward, components_dict) where reward can be negative (pain!)
    """
    # === EXPECTATION BASELINE (what did we expect?) ===
    expectation = ema_acc if ema_acc is not None else 0.5
    surprise = acc - expectation  # Positive = exceeded expectations

    # === ABSOLUTE PERFORMANCE EUPHORIA CURVE ===
    if acc < 0.70:
        base_reward = acc  # Linear below threshold (meh)
    elif acc < 0.85:
        # Quadratic warm-up: 70%â†’1.0, 85%â†’3.0
        normalized = (acc - 0.70) / 0.15
        base_reward = 1.0 + (normalized ** 2) * 2.0
    else:
        # Exponential EUPHORIA: 85%â†’3.0, 100%â†’10.0
        normalized = min(1.0, (acc - 0.85) / 0.15)
        base_reward = 3.0 + (10.0 - 3.0) * (normalized ** 2)

    # === SURPRISE MODULATION (beat or miss expectations?) ===
    if surprise > 0.05:  # Exceeded expectations!
        surprise_mult = 1.0 + surprise * 10.0  # Up to 2.5x boost
    elif surprise < -0.05:  # DISAPPOINTMENT
        surprise_mult = max(0.1, 1.0 + surprise * 5.0)  # Down to 0.1x (feels like shit!)
    else:
        surprise_mult = 1.0  # Met expectations, neutral

    # === VELOCITY TERM (momentum vs stagnation) ===
    if prev_acc is not None:
        velocity = acc - prev_acc
        if velocity > 0.05:  # Sharp improvement
            velocity_factor = 1.5  # Momentum feels good!
        elif velocity < -0.05:  # Sharp drop
            velocity_factor = 0.5  # PAIN from regression
        elif abs(velocity) < 0.01:  # Stagnation
            velocity_factor = 0.7  # "This feels like shit, try something new!"
        else:
            velocity_factor = 1.0
    else:
        velocity_factor = 1.0

    # === EM "NUT BUST" EXPLOSION ===
    if em >= 0.999:
        em_factor = 20.0  # ðŸŽ† MAXIMUM EUPHORIA ðŸŽ†
    elif em > 0.8:
        em_factor = 5.0 + (em - 0.8) / 0.2 * 15.0  # Building to explosion
    else:
        em_factor = 1.0

    # === REGRET TERM (almost perfect but missed EM) ===
    # ATLAS ULTRA-UNLEASHED: MAXIMUM EM obsession!
    regret_penalty = 0.0
    if acc > 0.80 and em < 0.6 and iou > 0.70:  # VERY low thresholds (was 0.90, 0.8)
        regret_penalty = -10.0 * (0.95 - em)  # BRUTAL penalty (was -3.0)

    # === COMBINE WITH NON-LINEAR INTERACTION ===
    reward = base_reward * surprise_mult * velocity_factor * em_factor
    reward *= (1.0 + iou * 2.0)  # IoU quality boost (up to 3x)
    reward += regret_penalty

    # Clip for numerical stability
    reward = max(-10.0, min(50.0, reward))

    # Legacy components for logging
    components = {
        'base': base_reward, 'surprise': surprise_mult, 'velocity': velocity_factor,
        'em_factor': em_factor, 'regret': regret_penalty, 'final_reward': reward,
        'acc': acc, 'em': em, 'iou': iou, 'expectation': expectation
    }

    return reward, components

# Keep old function for backward compatibility
def _dopamine_score(em: float, acc: float, iou: float, program_len: int,
                   entropy_red: float, mdl_gain: float, novelty: float, Lmax: int = 12) -> tuple:
    """Legacy dopamine scoring (calls euphoric version)."""
    return _dopamine_score_euphoric(em, acc, iou, prev_acc=None, ema_acc=None)

# =========================
# Adaptive Euphoria Gate
# =========================
class EuphoriaGate:
    """
    Adaptive controller for heavy features (STaR, PUCT, replay, dream).
    Triggers 'active windows' on ACC upticks or dopamine spikes,
    and scales budgets by step time & VRAM pressure.
    """
    def __init__(self, base_interval=50, window=50, acc_uptick=0.03, euphoria_thresh=3.0,
                 puct_min=40, puct_max=120, mem_cap_mb=7000):
        self.base_interval = int(base_interval)
        self.window = int(window)
        self.acc_uptick = float(acc_uptick)
        self.euphoria_thresh = float(euphoria_thresh)
        self.puct_min = int(puct_min)
        self.puct_max = int(puct_max)
        self.mem_cap_mb = int(mem_cap_mb)

        self.active_until = -1
        self.last_heavy_step = -10**9
        self.step_time_ema = None
        self.prev_acc = None

    def _update_time(self, dt):
        # Simple EMA on step time
        beta = 0.9
        if self.step_time_ema is None:
            self.step_time_ema = dt
        else:
            self.step_time_ema = beta*self.step_time_ema + (1-beta)*dt

    def update(self, global_step, acc, R_euphoric, step_dt):
        self._update_time(step_dt)
        uptick = 0.0 if self.prev_acc is None else (acc - self.prev_acc)
        self.prev_acc = acc

        # Trigger activation if ACC uptick or euphoric dopamine
        if (uptick >= self.acc_uptick) or (R_euphoric >= self.euphoria_thresh):
            self.active_until = max(self.active_until, global_step + self.window)

    def active(self, global_step):
        return global_step <= self.active_until

    def sim_budget(self):
        # Dynamic PUCT budget from step time and VRAM pressure
        import torch
        mem_mb = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
        slow = (self.step_time_ema or 0.0) > 0.12  # ~<8-9 it/s â†’ 'slow'
        pressured = mem_mb > self.mem_cap_mb
        if slow or pressured:
            return self.puct_min
        return self.puct_max

    def allow_heavy_this_step(self, global_step):
        # Baseline rate-limit when not active
        if self.active(global_step):
            return True
        return (global_step - self.last_heavy_step) >= self.base_interval

    def mark_heavy(self, global_step):
        self.last_heavy_step = global_step

# =========================
# Canonical grid encoding
# =========================
def _encode_grid_tensor(grid: torch.Tensor) -> tuple:
    """
    Canonical, hashable encoding for ARC grids.
    Returns: ((H, W), tuple(flat_int_values))
    Fixed: Ensure complete detachment from GPU and any unhashable references
    """
    if isinstance(grid, torch.Tensor):
        # GPU-FIRST: Keep on GPU, only detach for hashing (10-30x faster than .cpu())
        g = grid.detach().long()
        if g.dim() == 3 and g.size(0) == 1:
            g = g.squeeze(0)
        assert g.dim() == 2, f"Expected [H,W], got {tuple(g.shape)}"
        H, W = g.shape
        # GPU-FIRST: Use tolist() instead of loop with .item() (much faster)
        flat_values = g.flatten().tolist()  # Direct conversion to Python list on GPU
        
        # Create the final tuple and verify it's hashable before returning
        result = ((int(H), int(W)), tuple(flat_values))
        try:
            hash(result)  # Verify hashability
        except TypeError as e:
            raise ValueError(f"Generated unhashable grid encoding: H={H}, W={W}, values_type={type(flat_values)}, error={e}")
        
        return result
    # already encoded?
    if isinstance(grid, tuple) and len(grid) == 2 and isinstance(grid[0], tuple):
        return grid
    raise TypeError(f"Unsupported grid type for encoding: {type(grid)}")

def _decode_grid(enc: tuple) -> torch.Tensor:
    """
    Decode canonical grid encoding back to torch.LongTensor [H,W].
    """
    (H, W), flat = enc
    arr = np.array(flat, dtype=np.int64).reshape(H, W)
    return torch.from_numpy(arr).long()

def build_op_bias(temp: float = 0.7):
    """
    Convert op_success_count to a softmax prior (democratic â†’ data-driven).
    STaR will accept planner_op_bias for roughly half the traces.
    """
    ops = list(op_success_count.keys())
    if not ops:
        # If no data yet, keep a democratic prior over 41 ops
        ops = [f"op_{i}" for i in range(41)]
        vals = [1.0] * len(ops)
    else:
        vals = [op_success_count.get(op, 1.0) for op in ops]
    mx = max(vals) if vals else 1.0
    exps = [math.exp((v - mx) / max(1e-6, temp)) for v in vals]
    Z = sum(exps) if exps else 1.0
    return {op: (e / Z) for op, e in zip(ops, exps)}

def build_policy_guided_bias(grid_in: torch.Tensor, grid_out: torch.Tensor, 
                           op_policy: Optional[Any], device, temp: float = 0.7):
    """
    Enhanced op-bias that combines historical success counts with policy net predictions.
    Returns hybrid bias: 50% historical + 50% policy-guided.
    """
    # Start with historical bias (proven successful)
    historical_bias = build_op_bias(temp)
    
    # Add policy-guided bias if available
    if op_policy is not None:
        try:
            # Build minimal context for policy prediction  
            # Ensure grid is properly formatted [B, H, W] for policy
            if grid_in.dim() == 4:  # [B, C, H, W] â†’ remove channel dim if present
                policy_grid = grid_in.squeeze(1) if grid_in.size(1) == 1 else grid_in[:, 0]
            elif grid_in.dim() == 3:  # [B, H, W] or [C, H, W]
                if grid_in.size(0) == 1:  # [1, H, W]
                    policy_grid = grid_in
                else:  # [C, H, W] â†’ [1, H, W]  
                    policy_grid = grid_in[0:1]
            elif grid_in.dim() == 2:  # [H, W] â†’ [1, H, W]
                policy_grid = grid_in.unsqueeze(0)
            else:
                raise ValueError(f"Unexpected grid dimensions: {grid_in.shape}")
                
            B, H, W = policy_grid.shape
            rel_features = None  # Will use zeros fallback in policy
            size_oracle = torch.tensor([[H, W, H, W]], device=device).float()
            theme_priors = torch.zeros(1, 10, device=device)
            
            # Get policy prediction with properly shaped grid
            with torch.no_grad():
                pred = op_policy(policy_grid, rel_features, size_oracle, theme_priors, program_ops=[])
                raw_policy_bias = op_logits_to_bias(pred.op_logits)
                
                # Ensure policy_bias is a proper dict (robust type checking)
                if isinstance(raw_policy_bias, dict):
                    policy_bias = raw_policy_bias
                elif hasattr(raw_policy_bias, 'keys'):
                    policy_bias = dict(raw_policy_bias)  # Convert dict-like to dict
                else:
                    # Fallback: create uniform policy bias if conversion fails
                    policy_bias = {f"op_{i}": 1.0/41 for i in range(41)}
            
            # Hybrid: 50% historical + 50% policy-guided (with safe key access)
            hybrid_bias = {}
            all_ops = set(historical_bias.keys()) | set(policy_bias.keys())
            for op in all_ops:
                hist_val = historical_bias.get(op, 1.0 / len(all_ops))
                policy_val = policy_bias.get(op, 1.0 / len(all_ops))
                hybrid_bias[op] = 0.5 * hist_val + 0.5 * policy_val
                
            return hybrid_bias
        except Exception as e:
            logging.getLogger(__name__).warning(f"[Policy] guided bias failed: {e}")
    
    # Fallback to historical only
    return historical_bias

def dopamine_reward(task, buffer, logger, global_step, score: float = 1.0, components: Dict = None):
    """
    Enhanced dopamine capture with importance scoring:
    - Always store as canonical hashable tuples: ((H,W), tuple(flattened))
    - Enriched buffer with (enc_in, enc_out, score) for importance replay
    - Bypass ALL unhashable dict/Tensor issues
    - Log buffer growth and reward details
    """
    # CRITICAL: Detach score from autograd graph to prevent gradient leaks into RelMem
    score = float(score.detach().item() if torch.is_tensor(score) else score)

    if buffer is None:
        return
    try:
        inp_t = task['input'] if isinstance(task, dict) else getattr(task, 'input', None)
        out_t = task['output'] if isinstance(task, dict) else getattr(task, 'output', None)
        if inp_t is None or out_t is None:
            raise ValueError("dopamine_reward: task missing input/output")
        # Ensure [H,W] tensors for encoding
        if isinstance(inp_t, torch.Tensor) and inp_t.dim() == 3 and inp_t.size(0) == 1:
            inp_t = inp_t.squeeze(0)
        if isinstance(out_t, torch.Tensor) and out_t.dim() == 3 and out_t.size(0) == 1:
            out_t = out_t.squeeze(0)
        enc_inp = _encode_grid_tensor(inp_t)
        enc_out = _encode_grid_tensor(out_t)
        
        # Verify encodings are truly hashable before storage
        try:
            hash(enc_inp)
            hash(enc_out)
            hash(score)  # Ensure score is also hashable
        except TypeError as hash_err:
            raise ValueError(f"Generated unhashable encoding: enc_inp={type(enc_inp)}, enc_out={type(enc_out)}, score={type(score)}, error={hash_err}")
        
        # ALWAYS store score (even if zero/negative - it's information!)
        buffer.buffer.append((enc_inp, enc_out, float(score)))
            
        if logger:
            # Safely format components to avoid any unhashable issues in logging
            comp_str = ""
            if components:
                try:
                    # Convert components dict to a safe string representation
                    safe_components = {k: float(v) if hasattr(v, '__float__') else str(v) 
                                     for k, v in components.items() if v is not None}
                    comp_str = f" components={safe_components}"
                except Exception as comp_err:
                    comp_str = f" components=<error: {comp_err}>"
            logger.info(f"[Dopamine] Stored pair (score={score:.3f}) â†’ buffer size={len(buffer.buffer)} at step {global_step}{comp_str}")
    except Exception as e:
        if logger:
            logger.warning(f"[Dopamine] capture pipeline skipped at step {global_step}: {e}")

def _as_star_task(task):
    """
    Normalize any input into a proper Task dataclass.
    Guarantees .input/.output attributes for downstream (STaR, dopamine, Wormhole).
    """
    if task is None:
        return None

    # Already Task
    if isinstance(task, Task):
        return task

    # Dict-like
    if isinstance(task, dict):
        return Task(
            input=task.get("input") or task.get("test"),
            output=task.get("output") or task.get("target"),
            constraints=task.get("constraints", {}),
            metadata=task.get("metadata", {})
        )

    # Tuple/list
    if isinstance(task, (list, tuple)) and len(task) >= 2:
        return Task(input=task[0], output=task[1], constraints={}, metadata={})

    raise TypeError(f"Unsupported task type: {type(task)}")

def nightmare_prune(model, failures: List[Any], optimizer, scaler, device, logger, global_step: int,
                    alpha: float = 0.08, max_failures: int = 8):
    """
    Negative replay: apply small negative gradients on recent failed counterexamples.
    Uses model.evaluate_with_ebr() logits pathway like compute_metrics() for stability.
    """
    if not failures:
        return
    # Limit the number per cycle for stability
    batch_failures = failures[:max_failures]
    del failures[:max_failures]

    model.train()
    optimizer.zero_grad(set_to_none=True)
    device_type = 'cuda' if str(device).startswith('cuda') else 'cpu'

    neg_terms = []  # collect per-failure negative CE terms to ensure a real graph
    with torch.amp.autocast(device_type, enabled=(device.type == 'cuda')):
        for f in batch_failures:
            try:
                # Counterexample objects expose .task with .input/.output in Phase 3 code.
                task = getattr(f, "task", None)
                if task is None:
                    continue
                inp = task.input.to(device).unsqueeze(0)   # [1, H, W]
                tgt = task.output.to(device).unsqueeze(0)  # [1, H, W]

                logits = None
                # Prefer a grad-enabled path
                try:
                    out = model.forward_pretraining(inp, target_shape=tgt.shape[-2:])
                    logits = out.get('logits', None) if isinstance(out, dict) else None
                except Exception:
                    logits = None

                if logits is None:
                    # Fallback to evaluate_with_ebr (may be no-grad depending on implementation)
                    try:
                        eval_out = model.evaluate_with_ebr(inp, tgt)
                        logits = eval_out.get('logits', None)
                    except Exception:
                        logits = None

                if logits is None or (isinstance(logits, torch.Tensor) and not logits.requires_grad):
                    # Can't use this item for gradient-based pruning
                    continue

                # Negative CE: move away from these bad solutions
                ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                     tgt.view(-1).long().clamp(0, 9))
                neg_terms.append(-alpha * ce)
            except Exception as e:
                logger.debug(f"[Nightmare] skip one failure: {e}")
                continue

    if len(neg_terms) == 0:
        logger.info(f"[Nightmare] No grad-capable failures to prune at step {global_step} (skipped)")
        return

    # Aggregate (mean) to keep magnitude small and stable
    total_neg = torch.stack(neg_terms).mean()

    if torch.isfinite(total_neg) and total_neg.requires_grad:
        scaler.scale(total_neg).backward()
        # modest step; grads are small
        scaler.step(optimizer)
        scaler.update()
        logger.info(
            f"[Nightmare] Applied negative replay on {len(neg_terms)}/{len(batch_failures)} "
            f"failures at step {global_step} (loss={float(total_neg.item()):.4f})"
        )
    else:
        logger.warning("[Nightmare] Skipped due to non-finite or no-grad loss")

def load_config_with_overrides(config_path: str = None, cli_args = None) -> Dict[str, Any]:
    """
    Load YAML config and override with CLI arguments.

    Args:
        config_path: Path to YAML config file
        cli_args: Parsed argparse Namespace

    Returns:
        Merged config dictionary (CLI takes precedence over YAML)
    """
    config = {}

    # 1. Load YAML if provided
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)

            # Flatten nested YAML structure to match argparse
            def flatten_dict(d, parent_key=''):
                items = []
                for k, v in d.items():
                    new_key = f"{parent_key}_{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.extend(flatten_dict(v, new_key).items())
                    else:
                        items.append((new_key, v))
                return dict(items)

            config = flatten_dict(yaml_config)
            print(f"âœ… Loaded config from {config_path}")
            print(f"   YAML keys: {list(config.keys())[:10]}...")
        except Exception as e:
            print(f"âš ï¸  Failed to load YAML config {config_path}: {e}")
            print(f"   Falling back to CLI args only")

    # 2. Override with CLI args (argparse takes precedence)
    if cli_args:
        for key, val in vars(cli_args).items():
            # Only override if CLI arg was explicitly set (not default)
            # For simplicity, we override all non-None values
            if val is not None:
                config[key] = val

    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Train TOPAS+HRM (with DreamEngine controls)")

    # YAML config file support (CLI args override YAML)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (CLI arguments override YAML values)")

    # Dream engine controls
    parser.add_argument("--enable-dream", action="store_true", default=False,
                        help="Enable DreamEngine (micro ticks during forward) and offline cycles.")
    parser.add_argument("--dream-micro-ticks", type=int, default=1,
                        help="Number of micro dream ticks to run during each forward_pretraining call.")
    parser.add_argument("--dream-full-every", type=int, default=10,
                        help="Run a full offline dream consolidation every N epochs. 0 disables.")
    parser.add_argument("--dream-full-timeout", type=int, default=600,
                        help="Log timeout threshold (seconds) for full dream cycle; used for warnings only.")
    parser.add_argument("--dream-background", action="store_true", default=False,
                        help="If set, run full dream cycles in a background daemon thread (risky if dream touches model GPU state).")
    parser.add_argument("--dream-force-cpu", action="store_true", default=False,
                        help="Hint: prefer CPU for offline dream cycle if supported (not enforced here).")

    # Andromeda Cortex arguments
    parser.add_argument("--enable-cortex", action="store_true", default=False,
                        help="Enable Andromeda Cortex (predictive coding MoE layer)")
    parser.add_argument("--cortex-columns", type=int, default=8,
                        help="Number of expert columns in cortical sheet")
    parser.add_argument("--cortex-column-dim", type=int, default=256,
                        help="Hidden dimension per column")
    parser.add_argument("--cortex-depth", type=int, default=2,
                        help="Number of PC blocks per column")
    parser.add_argument("--cortex-gating-temp", type=float, default=0.7,
                        help="Softmax temperature for MoE gating")
    parser.add_argument("--lambda-cortex-recon", type=float, default=0.5,
                        help="Weight for cortex reconstruction loss")
    parser.add_argument("--lambda-cortex-entropy", type=float, default=0.5,
                        help="Weight for cortex entropy flux loss")
    parser.add_argument("--lambda-cortex-sparsity", type=float, default=0.25,
                        help="Weight for cortex KL sparsity loss")

    # SynergyFusion (ON by default, disable with --no-synergy-fusion)
    parser.add_argument("--no-synergy-fusion", action="store_false", dest="synergy_fusion_enabled",
                        help="Disable SynergyFusion (Cortex+RelMem+BrainGraph fusion, enabled by default)")
    parser.add_argument("--synergy-trust-temp", type=float, default=1.0, dest="synergy_fusion_trust_temp",
                        help="SynergyFusion trust temperature (default: 1.0)")
    parser.add_argument("--synergy-lambda-kl", type=float, default=1.0, dest="synergy_fusion_lambda_kl",
                        help="SynergyFusion KL loss weight (default: 1.0)")
    parser.add_argument("--synergy-resid-weight", type=float, default=0.2, dest="synergy_fusion_resid_weight",
                        help="SynergyFusion brain residual weight (default: 0.2)")
    parser.add_argument("--synergy-op-prior-weight", type=float, default=0.5, dest="synergy_fusion_op_prior_weight",
                        help="SynergyFusion operation prior blend weight (default: 0.5)")

    # Stage-6: Refinement Loop Configuration
    parser.add_argument("--no-refine-loop", action="store_true", default=False,
                        help="Disable Stage-6 iterative refinement loop")
    parser.add_argument("--refine-iters", type=int, default=3,
                        help="Maximum refinement iterations per task")
    parser.add_argument("--refine-depth", type=int, default=2,
                        help="PUCT search depth for refinement")
    parser.add_argument("--refine-simulations", type=int, default=100,
                        help="PUCT simulation budget per refinement iteration")
    parser.add_argument("--refine-c-puct", type=float, default=1.4,
                        help="PUCT exploration constant for refinement")
    parser.add_argument("--verbose-refine", action="store_true", default=False,
                        help="Enable verbose logging for refinement loop")
    # Energy aux losses (φ/κ/CGE/Hodge)
    parser.add_argument("--lambda-phi", type=float, default=0.02, help="Weight for phi_integration aux loss")
    parser.add_argument("--lambda-kappa", type=float, default=0.02, help="Weight for kappa_depth aux loss")
    parser.add_argument("--lambda-cge", type=float, default=0.02, help="Weight for cge_composition aux loss")
    parser.add_argument("--lambda-hodge", type=float, default=0.01, help="Weight for hodge_structure aux loss")
    parser.add_argument("--kappa-target", type=float, default=0.50, help="Target assembly depth (0..1)")
    parser.add_argument("--energy-ramp-epochs", type=int, default=20, help="Ramp-in duration for energy lambdas")
    parser.add_argument("--approx-steps-per-epoch", type=int, default=400, help="Used for ramp calc from global_step")
    parser.add_argument("--energy-every-n-steps", type=int, default=4, help="Compute energy losses every N steps")
    parser.add_argument("--energy-on-replay", action="store_true", default=False, help="Also apply during dopamine replay")
    # RelMem configuration
    parser.add_argument("--relmem-reg-alpha", type=float, default=1e-3,
                        help="Weight for inverse_loss_safe regularization")
    parser.add_argument("--relmem-reg-beta", type=float, default=5e-4,
                        help="Weight for inheritance_pass regularization")
    parser.add_argument("--relmem-bind-iou", type=float, default=0.25,
                        help="IoU threshold for RelMem concept binding on success")
    parser.add_argument("--relmem-log-interval", type=int, default=200,
                        help="Log RelMem stats every N steps")
    
    # Progressive RelMem bias ramping parameters
    parser.add_argument("--relmem-bias-ramp-start", type=int, default=0,
                        help="Epoch to start ramping up RelMem bias")
    parser.add_argument("--relmem-bias-max", type=float, default=0.5,
                        help="Maximum RelMem bias weight after ramping")
    
    # Dream pretrain args
    parser.add_argument("--dream-pretrain-epochs", type=int, default=0,
                        help="Number of epochs for Dream/ETS pretraining (default 0 = disabled)")
    parser.add_argument("--dream-pretrain-lr", type=float, default=1e-4,
                        help="Learning rate for Dream pretraining")
    parser.add_argument("--dream-pretrain-batches", type=int, default=200,
                        help="Number of batches per Dream pretrain epoch")
    parser.add_argument("--dream-pretrain-freeze-model", action="store_true", default=False,
                        help="Freeze main model during Dream pretraining")
    
    # Self-play args
    parser.add_argument("--selfplay-enable", action="store_true", default=False,
                        help="Enable self-play with template-guided puzzles")
    parser.add_argument("--selfplay-interval", type=int, default=250,
                        help="Generate self-play puzzles every N steps")
    parser.add_argument("--selfplay-weight", type=float, default=0.1,
                        help="Weight for self-play loss")
    parser.add_argument("--selfplay-topk", type=int, default=3,
                        help="Number of puzzles to generate per self-play round")
    parser.add_argument("--selfplay-buffer-size", type=int, default=200,
                        help="Maximum size of self-play buffer")
    
    # RelMem / UKS-lite
    parser.add_argument('--relmem-loss-weight', type=float, default=0.01,
                        help="Weight for RelMem auxiliary loss")
    parser.add_argument('--relmem-loss-interval', type=int, default=25,
                        help="Apply RelMem loss every N steps")
    parser.add_argument('--uks-save-path', type=str, default='checkpoints/uks_state.pt',
                        help="Path to save UKS state")
    parser.add_argument('--uks-load-path', type=str, default='',
                        help="Path to load UKS state from")
    
    # Eval args
    parser.add_argument("--eval-interval", type=int, default=5,
                        help="Run evaluation every N epochs")
    
    # Training args
    parser.add_argument("--max-steps", type=int, default=60000,
                        help="Maximum training steps")
    parser.add_argument("--dataset", type=str, default="arc1", choices=["arc1", "arc2"],
                        help="Dataset to use: arc1 (original training) or arc2 (ARC Prize 2025)")

    # ---- Model-size & dopaminergic/nightmare controls ----
    parser.add_argument("--model-width", type=int, default=512,
                        help="Hidden width for TOPAS conv backbone (default 512)")
    parser.add_argument("--model-slots", type=int, default=64,
                        help="Number of slots for concept vectors (default 64)")
    parser.add_argument("--breakthrough-threshold", type=float, default=0.33,
                        help="EM threshold to trigger dopamine capture (default 0.33)")
    parser.add_argument("--nightmare-alpha", type=float, default=0.08,
                        help="Negative reinforcement strength for nightmares (0.05â€“0.10 recommended)")
    parser.add_argument("--nightmare-min-interval", type=int, default=200,
                        help="Minimum steps between nightmare cycles (when failure low)")
    parser.add_argument("--nightmare-max-interval", type=int, default=1000,
                        help="Maximum steps between nightmare cycles (when failure high)")
    # Mind-voice controls
    parser.add_argument("--monologue-interval", type=int, default=200,
                        help="Every N steps, sample traces and compute reasoning consistency")
    parser.add_argument("--monologue-min-traces", type=int, default=4,
                        help="Min number of traces to consider for consistency")
    parser.add_argument("--monologue-consistency-target", type=float, default=0.85,
                        help="Target overall consistency; below this we increase pruning, above this we ramp bias")
    parser.add_argument("--monologue-selfplay-bonus", type=float, default=0.05,
                        help="Increase self-play weight by this when consistency is high")

    # Alpha-ARC X Neural-Guided Search 2.0 parameters
    parser.add_argument("--search-alg", type=str, default="puct", choices=["beam", "puct"],
                        help="Search algorithm: beam or puct")
    parser.add_argument("--puct-nodes", type=int, default=1000,
                        help="Number of PUCT search nodes")
    parser.add_argument("--c-puct", type=float, default=1.414,
                        help="PUCT exploration constant")
    parser.add_argument("--puct-depth", type=int, default=10,
                        help="Maximum PUCT search depth")
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.3,
                        help="Dirichlet noise alpha for root node")
    parser.add_argument("--root-dirichlet-eps", type=float, default=0.25,
                        help="Dirichlet noise epsilon for root node")
    parser.add_argument("--sc-star", action="store_true", default=False,
                        help="Enable Self-Critique STaR wrapper")
    parser.add_argument("--near-miss-hamming-pct", type=float, default=0.15,
                        help="Near-miss Hamming distance threshold percentage")
    parser.add_argument("--replay-cap", type=int, default=2000,
                        help="Replay buffer capacity")

    # Dream â†’ EM gradient path controls
    parser.add_argument("--use-dream-kl", type=float, default=1.0,
                        help="Weight for Dream-KL op-prior supervision loss (0=disabled)")
    parser.add_argument("--use-contrastive", type=float, default=0.05,
                        help="Weight for Dream feature alignment loss (0=disabled)")
    parser.add_argument("--use-demo-consistency", type=float, default=0.02,
                        help="Weight for cross-demo consistency loss (0=disabled)")
    parser.add_argument("--use-critic-head", action="store_true", default=True,
                        help="Enable critic head for EM likelihood prediction (HRM-aligned)")

    # Wormhole mining gates (lowered thresholds to increase template frequency)
    parser.add_argument("--wormhole-min-em", type=float, default=0.30,
                        help="Minimum EM to trigger wormhole mining (default 0.30)")
    parser.add_argument("--wormhole-min-iou", type=float, default=0.30,
                        help="Minimum mean IoU to trigger wormhole mining (default 0.30)")

    parser.add_argument("--replay-micro-every", type=int, default=30,
                        help="Run replay-only microstep every N steps (0=disabled)")
    parser.add_argument("--replay-micro-k", type=int, default=1,
                        help="Number of samples per replay microstep")
    parser.add_argument("--replay-micro-lambda", type=float, default=0.5,
                        help="Learning rate multiplier for replay microsteps")
    parser.add_argument("--mdl-penalty", type=float, default=0.02,
                        help="MDL penalty coefficient for program length in search")
    parser.add_argument("--max-concepts", type=int, default=64,
                        help="Maximum active concepts to keep in RelMem (prune/merge threshold)")
    parser.add_argument("--stagnation-patience", type=int, default=200,
                        help="Steps without improvement before enabling stagnation-gated losses")
    parser.add_argument("--seed", type=int, default=1337,
                        help="Random seed for reproducibility")

    # Orbit-CEGIS controls (Alpha-ARC Î©)
    parser.add_argument("--use-orbit-canon", action="store_true", default=True,
                        help="Enable Dâ‚„ + color canonicalization (default: enabled)")
    parser.add_argument("--orbit-loss-weight", type=float, default=0.03,
                        help="Weight for orbit-contrastive loss (OrthoNCE)")
    parser.add_argument("--certificates", type=str, default="soft", choices=["off", "soft", "hard"],
                        help="CEGIS certificate mode: off, soft (penalty only), hard (prune violations)")

    # TTT (Test-Time Training) controls
    parser.add_argument("--ttt-enable", action="store_true", default=False,
                        help="Enable test-time training with LoRA adaptation")
    parser.add_argument("--ttt-r", type=int, default=8,
                        help="LoRA rank for TTT adaptation")
    parser.add_argument("--ttt-alpha", type=float, default=16.0,
                        help="LoRA alpha scaling for TTT")
    parser.add_argument("--ttt-steps", type=int, default=10,
                        help="Number of adaptation steps per task during TTT")
    parser.add_argument("--ttt-lr", type=float, default=1e-3,
                        help="Learning rate for TTT adaptation")
    parser.add_argument("--ttt-lr-ratio", type=float, default=1.0,
                        help="LoRA+ ratio for A vs B learning rates")

    args, _unknown = parser.parse_known_args()
    return args

def run_dream_cycle_safe(model,
                         timeout_sec: int = 600,
                         background: bool = False,
                         force_cpu: bool = False,
                         logger=None) -> Optional[Dict[str, Any]]:
    """
    Run model.run_dream_cycle() robustly.

    - Checks model has run_dream_cycle
    - Skips if no cached tokens and model.run_dream_cycle requires tokens
    - Logs start/end, duration, returned stats (if any)
    - Optionally runs in a background daemon thread (default False).

    Returns the dict returned by run_dream_cycle() if called and returned; otherwise None.
    """
    logger = logger or logging.getLogger(__name__)

    if not hasattr(model, "run_dream_cycle"):
        logger.warning("[Dream] model has no run_dream_cycle() method; skipping.")
        return None

    # Best-effort: prefer to pass cached tokens if available
    tokens = getattr(model, "_dream_tokens", None)

    def _call_cycle():
        try:
            t0 = time.time()
            logger.info("[Dream] Full cycle started (force_cpu=%s) ...", force_cpu)
            # Call with tokens if available; wrap in try/except
            try:
                if tokens is not None:
                    stats = model.run_dream_cycle(tokens=tokens)
                else:
                    # Some implementations accept no args
                    stats = model.run_dream_cycle()
            except TypeError:
                # Older signature - call without tokens
                stats = model.run_dream_cycle()
            duration = time.time() - t0
            logger.info("[Dream] Full cycle finished in %.1f s. stats=%s", duration, repr(stats))
            return stats
        except Exception as e:
            logger.warning("[Dream] Full cycle failed: %s\n%s", repr(e), traceback.format_exc())
            return None

    if background:
        # Run asynchronously in a daemon thread and return immediately.
        logger.warning("[Dream] Running full dream cycle in background thread (thread-safety warning).")
        th = threading.Thread(target=_call_cycle, daemon=True, name="dream_cycle_thread")
        th.start()
        return None
    else:
        # Run synchronously (blocking) and return stats. Use a simple watchdog for long runs (warning only).
        t_start = time.time()
        stats = _call_cycle()
        t_total = time.time() - t_start
        if t_total > timeout_sec:
            logger.warning("[Dream] Full cycle exceeded timeout threshold %ds (took %.1fs)", timeout_sec, t_total)
        return stats

def setup_logging(verbose=True):
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    # Set up dual file logging with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename_clean = f'logs/training_CLEAN_{timestamp}.log'
    log_filename_verbose = f'logs/training_VERBOSE_{timestamp}.log'

    # Force logging configuration by removing existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set up formatters
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # CLEAN file handler (warnings/errors only)
    clean_handler = logging.FileHandler(log_filename_clean, mode='w')
    clean_handler.setLevel(logging.WARNING)
    clean_handler.setFormatter(formatter)

    # Console handler (keep clean)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)

    # Add base handlers
    root_logger.addHandler(clean_handler)
    root_logger.addHandler(console_handler)

    logger = logging.getLogger(__name__)

    if verbose:
        # VERBOSE file handler (everything: INFO, DEBUG, dopamine, relmem, gradients)
        log_filename_verbose = f'logs/training_VERBOSE_{timestamp}.log'
        verbose_handler = logging.FileHandler(log_filename_verbose, mode='w')
        verbose_handler.setLevel(logging.INFO)
        verbose_handler.setFormatter(formatter)
        root_logger.addHandler(verbose_handler)
        root_logger.setLevel(logging.INFO)
        logger.info(f"Dual logging enabled: CLEAN={log_filename_clean}, VERBOSE={log_filename_verbose}")
        print(f"Dual logging: CLEAN={log_filename_clean} | VERBOSE={log_filename_verbose}")
        return logger
    else:
        # SPEED MODE: Warnings only (5-10% faster!)
        root_logger.setLevel(logging.WARNING)
        print(f"⚡ SPEED MODE: Warnings-only logging → {log_filename_clean}")
        return logger


def _extract_puzzle_attributes(input_grid: torch.Tensor, target_grid: torch.Tensor) -> list:
    """
    Extract observable puzzle attributes for brain-graph learning.
    ARC-compliant: based on visual properties, not task-specific heuristics.
    Enhanced to align with brain-graph DSL vocabulary.
    """
    attrs = []

    # Color analysis
    inp_unique = torch.unique(input_grid)
    out_unique = torch.unique(target_grid)
    inp_colors = len(inp_unique)
    out_colors = len(out_unique)

    attrs.append(f"inp_colors_{inp_colors}")
    attrs.append(f"out_colors_{out_colors}")

    # Simple/complex palette classification
    if inp_colors <= 3:
        attrs.append("simple_palette")
    elif inp_colors >= 6:
        attrs.append("complex_palette")

    if inp_colors != out_colors:
        attrs.append("color_count_change")
        attrs.append("needs_color_map")  # Brain-graph keyword
    if out_colors < inp_colors:
        attrs.append("color_reduction")
    elif out_colors > inp_colors:
        attrs.append("color_expansion")

    # Shape analysis
    inp_h, inp_w = input_grid.shape[-2:]
    out_h, out_w = target_grid.shape[-2:]

    if inp_h == inp_w:
        attrs.append("square_input")
        attrs.append("square_grid")  # Brain-graph keyword
    if out_h == out_w:
        attrs.append("square_output")
    if inp_h == out_h and inp_w == out_w:
        attrs.append("shape_preserved")
    else:
        attrs.append("shape_changed")
        # Detect specific transformations
        if out_h < inp_h or out_w < inp_w:
            attrs.append("grid_crop")  # Brain-graph keyword
        elif out_h > inp_h or out_w > inp_w:
            attrs.append("scaling")  # Brain-graph keyword

    # Grid size categories
    grid_size = inp_h * inp_w
    if grid_size <= 100:
        attrs.append("small_grid")
    elif grid_size <= 400:
        attrs.append("medium_grid")
    else:
        attrs.append("large_grid")

    # Even/odd dimensions (correlate with symmetry patterns)
    if inp_h % 2 == 0 and inp_w % 2 == 0:
        attrs.append("even_dims")

    # Symmetry detection (input grid)
    if inp_h == inp_w:
        try:
            # Horizontal reflection
            if torch.allclose(input_grid, input_grid.flip(0), atol=0):
                attrs.append("symmetry_h")
                attrs.append("symmetry_quadrant")  # Brain-graph keyword
            # Vertical reflection
            if torch.allclose(input_grid, input_grid.flip(1), atol=0):
                attrs.append("symmetry_v")
                attrs.append("symmetry_quadrant")
            # 90-degree rotational
            if torch.allclose(input_grid, input_grid.rot90(1, [0, 1]), atol=0):
                attrs.append("symmetry_rot90")
                attrs.append("symmetry_quadrant")
        except Exception:
            pass

    # Pattern detection (tiling)
    if inp_h >= 4 and inp_w >= 4 and inp_h % 2 == 0 and inp_w % 2 == 0:
        try:
            quad1 = input_grid[:inp_h//2, :inp_w//2]
            quad2 = input_grid[:inp_h//2, inp_w//2:]
            quad3 = input_grid[inp_h//2:, :inp_w//2]
            quad4 = input_grid[inp_h//2:, inp_w//2:]

            if torch.allclose(quad1, quad2, atol=0) or torch.allclose(quad1, quad3, atol=0):
                attrs.append("tiling_pattern")
                attrs.append("pattern_fill")  # Brain-graph keyword
        except Exception:
            pass

    # Mass conservation check
    inp_nonzero = (input_grid != 0).sum().item()
    out_nonzero = (target_grid != 0).sum().item()
    if abs(inp_nonzero - out_nonzero) < 3:
        attrs.append("mass_conserved")

    # Object density
    if inp_nonzero < grid_size * 0.2:
        attrs.append("sparse_objects")
        attrs.append("object_manipulation")  # Brain-graph keyword
    elif inp_nonzero > grid_size * 0.8:
        attrs.append("dense_grid")

    # Translation detection (compare bounding boxes)
    try:
        inp_rows = (input_grid != 0).any(dim=1).nonzero().flatten()
        inp_cols = (input_grid != 0).any(dim=0).nonzero().flatten()
        out_rows = (target_grid != 0).any(dim=1).nonzero().flatten()
        out_cols = (target_grid != 0).any(dim=0).nonzero().flatten()

        if len(inp_rows) > 0 and len(out_rows) > 0:
            inp_bbox = (inp_rows.min().item(), inp_rows.max().item(),
                       inp_cols.min().item(), inp_cols.max().item())
            out_bbox = (out_rows.min().item(), out_rows.max().item(),
                       out_cols.min().item(), out_cols.max().item())

            # Check if object moved
            if (inp_bbox[1] - inp_bbox[0] == out_bbox[1] - out_bbox[0] and
                inp_bbox[3] - inp_bbox[2] == out_bbox[3] - out_bbox[2] and
                (inp_bbox[0] != out_bbox[0] or inp_bbox[2] != out_bbox[2])):
                attrs.append("translation")  # Brain-graph keyword
    except Exception:
        pass

    return attrs


def compute_metrics(model, input_grid, target_grid, hrm_latents=None):
    """Fast neural-only metrics for batch monitoring (every 30 steps)."""
    from trainers.arc_validator import validate_arc_grid

    with torch.no_grad():
        # ARC-II Validation: Ensure inputs conform to spec
        input_grid = validate_arc_grid(input_grid, "compute_metrics input", clamp=True)
        target_grid = validate_arc_grid(target_grid, "compute_metrics target", clamp=True)

        eval_outputs = model.evaluate_with_ebr(input_grid, target_grid, hrm_latents=hrm_latents)

        if eval_outputs.get('logits') is None:
            return {'exact_match': 0.0, 'accuracy': 0.0, 'mean_iou': 0.0, 'exact_match_refined': 0.0, 'batch_size': 1}

        logits = eval_outputs['logits']
        B = logits.size(0)
        preds = logits.argmax(dim=-1)  # [B, H*W]

        # ARC-II Validation: Clamp predictions to valid range [0, 9]
        preds = torch.clamp(preds, 0, 9)

        targets_flat = target_grid.reshape(B, -1)  # [B, H*W]

        # Basic metrics
        exact_match = (preds == targets_flat).all(dim=1).float().mean().item()
        accuracy = (preds == targets_flat).float().mean().item()

        # IoU per color
        ious = []
        for c in range(10):  # NUM_COLORS
            pred_c = (preds == c)
            target_c = (targets_flat == c)
            intersection = (pred_c & target_c).sum().float()
            union = (pred_c | target_c).sum().float()
            if union > 0:
                ious.append((intersection / union).item())
        mean_iou = sum(ious) / len(ious) if ious else 0.0

        # EBR refined exact match
        exact_match_refined = eval_outputs.get('exact_match_refined', 0.0)

        return {
            'exact_match': exact_match,
            'accuracy': accuracy,
            'mean_iou': mean_iou,
            'exact_match_refined': exact_match_refined,
            'batch_size': B  # Add batch size for weighted averaging
        }


def compute_metrics_with_dsl(model, input_grid, target_grid, hrm_latents=None, demos=None, task_id=None):
    """
    FULL neurosymbolic pipeline for epoch evaluations (HyLa + PUCT + CEGIS).

    WARNING: SLOW - only call during epoch evals (every 5 epochs), NOT batch metrics!
    Activates forward() → RAIL-DSL block with HyLa, PUCT, Certificate pruning.
    """
    with torch.no_grad():
        if demos is None or len(demos) == 0:
            # No demos - fall back to fast neural-only
            return compute_metrics(model, input_grid, target_grid, hrm_latents)

        # Convert demos to Dict format for forward()
        demos_forward = []
        for demo_input, demo_output in demos:
            demos_forward.append({
                'input': demo_input if demo_input.dim() == 2 else demo_input.squeeze(0),
                'output': demo_output if demo_output.dim() == 2 else demo_output.squeeze(0)
            })

        # Create test dict
        test_forward = {
            'input': input_grid if input_grid.dim() == 2 else input_grid.squeeze(0),
            'output': target_grid if target_grid.dim() == 2 else target_grid.squeeze(0)
        }

        # Call forward() - activates RAIL-DSL block (HyLa, PUCT, CEGIS)!
        try:
            grid_out, logits, size, extras = model.forward(
                demos=demos_forward,
                test=test_forward,
                eval_use_dsl=True,  # Enable HyLa + PUCT + CEGIS
                eval_use_ebr=True,  # Enable EBR refinement
                task_id=task_id,
                training_mode=False  # Eval mode
            )

            if logits is None:
                return {'exact_match': 0.0, 'accuracy': 0.0, 'mean_iou': 0.0, 'exact_match_refined': 0.0}

            # Compute metrics
            B = logits.size(0)
            preds = logits.argmax(dim=-1)
            targets_flat = target_grid.reshape(B, -1)

            exact_match = (preds == targets_flat).all(dim=1).float().mean().item()
            accuracy = (preds == targets_flat).float().mean().item()

            # IoU
            ious = []
            for c in range(10):
                pred_c = (preds == c)
                target_c = (targets_flat == c)
                intersection = (pred_c & target_c).sum().float()
                union = (pred_c | target_c).sum().float()
                if union > 0:
                    ious.append((intersection / union).item())
            mean_iou = sum(ious) / len(ious) if ious else 0.0

            # Grid-based exact match from DSL output
            if grid_out is not None:
                grid_em = (grid_out == target_grid).all().float().item()
                exact_match_refined = grid_em
            else:
                exact_match_refined = exact_match

            return {
                'exact_match': exact_match,
                'accuracy': accuracy,
                'mean_iou': mean_iou,
                'exact_match_refined': exact_match_refined
            }

        except Exception as e:
            print(f"[WARN] DSL metrics failed: {e}, using neural-only")
            import traceback
            traceback.print_exc()
            return compute_metrics(model, input_grid, target_grid, hrm_latents)

def create_models(device, cli_args=None, config_dict=None):
    print("🔧 Creating HRM-TOPAS integrated models...")
    topas_config = ModelConfig()

    # Use config_dict if available, otherwise empty dict
    config = config_dict if config_dict is not None else {}
    
    # Apply CLI dream settings to config before model creation
    if cli_args and getattr(cli_args, "enable_dream", False):
        topas_config.enable_dream = True
        if hasattr(cli_args, "dream_micro_ticks"):
            topas_config.dream_micro_ticks = cli_args.dream_micro_ticks
    # Allow moderate scaling to use GPU headroom safely
    topas_config.width = int(getattr(cli_args, "model_width", 512))
    topas_config.depth = 8
    topas_config.slots = int(getattr(cli_args, "model_slots", 64))  # Keep capacity for ARC-II
    topas_config.slot_dim = 256
    topas_config.max_dsl_depth = 4
    topas_config.use_ebr = True
    # Energy aux loss config (CLI → config)
    topas_config.lambda_phi = getattr(cli_args, "lambda_phi", topas_config.lambda_phi)
    topas_config.lambda_kappa = getattr(cli_args, "lambda_kappa", topas_config.lambda_kappa)
    topas_config.lambda_cge = getattr(cli_args, "lambda_cge", topas_config.lambda_cge)
    topas_config.lambda_hodge = getattr(cli_args, "lambda_hodge", topas_config.lambda_hodge)
    topas_config.kappa_target = getattr(cli_args, "kappa_target", topas_config.kappa_target)
    topas_config.energy_ramp_epochs = getattr(cli_args, "energy_ramp_epochs", topas_config.energy_ramp_epochs)
    topas_config.approx_steps_per_epoch = getattr(cli_args, "approx_steps_per_epoch", topas_config.approx_steps_per_epoch)
    topas_config.energy_every_n_steps = getattr(cli_args, "energy_every_n_steps", topas_config.energy_every_n_steps)
    topas_config.energy_on_replay = getattr(cli_args, "energy_on_replay", topas_config.energy_on_replay)

    # Andromeda Cortex config (CLI → config)
    if cli_args and getattr(cli_args, "enable_cortex", False):
        topas_config.enable_cortex = True
        topas_config.cortex_columns = getattr(cli_args, "cortex_columns", 8)
        topas_config.cortex_column_dim = getattr(cli_args, "cortex_column_dim", 256)
        topas_config.cortex_depth = getattr(cli_args, "cortex_depth", 2)
        topas_config.cortex_gating_temp = getattr(cli_args, "cortex_gating_temp", 0.7)
        topas_config.lambda_cortex_recon = getattr(cli_args, "lambda_cortex_recon", 0.5)
        topas_config.lambda_cortex_entropy = getattr(cli_args, "lambda_cortex_entropy", 0.5)
        topas_config.lambda_cortex_sparsity = getattr(cli_args, "lambda_cortex_sparsity", 0.25)

    # Stage-6: Refinement Loop Configuration
    topas_config.enable_refinement_loop = not getattr(cli_args, "no_refine_loop", False)
    topas_config.max_refine_iters = getattr(cli_args, "refine_iters", 3)
    topas_config.refine_search_depth = getattr(cli_args, "refine_depth", 2)
    topas_config.refine_simulations = getattr(cli_args, "refine_simulations", 100)
    topas_config.refine_c_puct = getattr(cli_args, "refine_c_puct", 1.4)
    topas_config.verbose_refine = getattr(cli_args, "verbose_refine", False)
    
    # Progressive RelMem bias ramping - conservative clean test
    base_bias_w = 0.2  # Conservative baseline for clean testing
    max_bias_w = getattr(cli_args, 'relmem_bias_max', 0.5)  # Standard maximum
    ramp_start = getattr(cli_args, 'relmem_bias_ramp_start', 10)
    
    # Boosted RelMem DSL influence parameters (will be ramped progressively)
    topas_config.relmem_op_bias_w = base_bias_w        # Start with proven stable
    topas_config.relmem_op_bias_scale = 1.0           # Double from 0.5 → full scaling
    topas_config.relmem_bias_beta = config.get('fusion_weights_relmem_bias_beta', 0.4)  # Read from YAML
    topas_config.theme_bias_alpha = config.get('fusion_weights_theme_bias_alpha', 0.2)  # Read from YAML

    # Hierarchical Abstraction (experimental - multi-level pattern extraction)
    topas_config.use_hierarchical_abstraction = config.get('model_use_hierarchical_abstraction', False)

    # Store ramping parameters for training loop
    topas_config._bias_ramp_start = ramp_start
    topas_config._bias_max = max_bias_w
    topas_config._bias_base = base_bias_w

    # TTT (Test-Time Training) configuration
    topas_config.ttt_enable = config.get('ttt_enable', getattr(cli_args, 'ttt_enable', False))
    topas_config.ttt_r = config.get('ttt_r', getattr(cli_args, 'ttt_r', 8))
    topas_config.ttt_alpha = config.get('ttt_alpha', getattr(cli_args, 'ttt_alpha', 16.0))
    topas_config.ttt_steps = config.get('ttt_steps', getattr(cli_args, 'ttt_steps', 10))
    topas_config.ttt_lr = config.get('ttt_lr', getattr(cli_args, 'ttt_lr', 1e-3))
    topas_config.ttt_lr_ratio = config.get('ttt_lr_ratio', getattr(cli_args, 'ttt_lr_ratio', 1.0))

    # Critic head configuration (HRM-aligned EM prediction)
    topas_config.use_critic_head = config.get('use_critic_head', getattr(cli_args, 'use_critic_head', True))

    # Don't override enable_dream - respect CLI setting
    topas_config.verbose = config.get('logging_verbose', True)  # Read from YAML
    topas_config.pretraining_mode = True

    topas_model = TopasARC60M(topas_config).to(device)

    # Initialize BrainGraph for structured symbolic reasoning (starts empty, learns during training)
    brain_graph = BrainGraph(device=device, puzzle_emb_dim=128)
    topas_model.brain_graph = brain_graph  # Attach to model for access in train_step
    print("[BrainGraph] Initialized (empty - will learn from training data)")

    # Ensure Dream system is also moved to the correct device
    if hasattr(topas_model, 'dream') and topas_model.dream is not None:
        if hasattr(topas_model.dream, 'to'):
            topas_model.dream.to(device)
        
        # Ensure Wormhole is initialized for template mining
        if not hasattr(topas_model.dream, 'wormhole') or topas_model.dream.wormhole is None:
            try:
                from wormhole_offline import WormholeTemplateMiner
                topas_model.dream.wormhole = WormholeTemplateMiner()
                print("[TOPAS] Wormhole template miner initialized")
            except ImportError as e:
                print(f"[TOPAS] Warning: Could not initialize Wormhole: {e}")
        else:
            print("[TOPAS] Wormhole template miner already initialized")
    
    print(f"âœ… TOPAS: {sum(p.numel() for p in topas_model.parameters()):,} parameters")

    # Policy/Value Networks
    from models.value_net import ValueNet
    ctrl_dim = topas_model.ctrl_dim
    policy_net = OpPolicyNet(input_dim=ctrl_dim, hidden_dim=512, num_layers=6, num_heads=8, dropout=0.1).to(device)
    value_net = ValueNet(context_dim=ctrl_dim, program_dim=128, hidden_dim=512, num_layers=4, dropout=0.1).to(device)
    topas_model.enable_policy_mode(policy_net=policy_net, value_net=value_net, ebr_direction_net=None)
    print(f"âœ… Policy Network: {sum(p.numel() for p in policy_net.parameters()):,} parameters")
    print(f"âœ… Value Network: {sum(p.numel() for p in value_net.parameters()):,} parameters")

    hrm_config = {
        "batch_size": 1,
        "seq_len": 900,  # ARC-II: Keep full coverage (handles all 30×30 grids)
        "vocab_size": 10,
        "num_puzzle_identifiers": 1000,
        "puzzle_emb_ndim": 0,  # ARC-II: Disable extra tokens; seq_len driven by actual grid size
        "H_cycles": 3,
        "L_cycles": 4,
        "H_layers": 4,
        "L_layers": 4,
        "hidden_size": 512,
        "expansion": 3.0,
        "num_heads": 8,
        "pos_encodings": "rope",
        "halt_max_steps": 6,
        "halt_exploration_prob": 0.1,
        "forward_dtype": "bfloat16",
    }
    hrm_model = NeuroPlannerWrapper(hrm_config).to(device)
    print(f"âœ… NeuroPlanner: {sum(p.numel() for p in hrm_model.parameters()):,} parameters")

    return topas_model, hrm_model

def dream_pretrain_loop(topas_model, dataset, cli_args, device, logger):
    """
    Tiny Dream-Pretrain phase: train Dream/ETS only for a few epochs
    """
    if not cli_args or cli_args.dream_pretrain_epochs <= 0:
        return
        
    logger.info(f"[Dream-Pretrain] Starting {cli_args.dream_pretrain_epochs} epochs")
    
    # Check if model has dream engine
    if not hasattr(topas_model, 'dream') or topas_model.dream is None:
        logger.warning("[Dream-Pretrain] No DreamEngine found, skipping pretrain")
        return
    
    dream = topas_model.dream
    # Move dream to the same device as the model
    if hasattr(dream, 'to'):
        # Use the proper to() method which handles device and generator
        dream.to(device)
    elif hasattr(dream, 'device'):
        # Fallback: Update device and move internal modules
        dream.device = device
        # Move internal modules if they exist
        if hasattr(dream, '_dream_color_head') and dream._dream_color_head is not None:
            dream._dream_color_head = dream._dream_color_head.to(device)
        if hasattr(dream, '_dream_theme_head') and dream._dream_theme_head is not None:
            dream._dream_theme_head = dream._dream_theme_head.to(device)
        if hasattr(dream, 'nmda') and dream.nmda is not None:
            for attr_name in dir(dream.nmda):
                attr = getattr(dream.nmda, attr_name)
                if isinstance(attr, torch.nn.Module):
                    setattr(dream.nmda, attr_name, attr.to(device))
    
    # robustly collect dream params
    dream_params = []
    if hasattr(dream, 'parameters') and callable(getattr(dream, 'parameters')):
        try:
            dream_params = [p for p in dream.parameters() if p is not None]
        except Exception:
            dream_params = []
    # if still empty, attempt recursive attribute scan (already implemented in dream.parameters())
    if len(dream_params) == 0:
        logging.getLogger(__name__).warning("[Dream-Pretrain] No dream params found (will attempt fallback or skip)")
        return
    
    if not dream_params:
        logger.warning("[Dream-Pretrain] No trainable Dream/ETS parameters found")
        return
    
    dream_optimizer = torch.optim.Adam(dream_params, lr=cli_args.dream_pretrain_lr)

    # === HIERARCHICAL ABSTRACTOR PRETRAIN (Warm-Start Strategy) ===
    # Train hierarchical abstractor on dream data to stabilize gradients before main training
    # NOTE: Projection layers will be initialized on-the-fly during pretrain loop
    hierarchical_params_list = []  # Will accumulate as projections are created
    hierarchical_optimizer = None

    if hasattr(topas_model, 'hierarchical_abstractor') and topas_model.hierarchical_abstractor is not None:
        hierarchical_params_list.extend(topas_model.hierarchical_abstractor.parameters())
        if hasattr(topas_model, 'hierarchical_gate'):
            hierarchical_params_list.extend(topas_model.hierarchical_gate.parameters())

        if hierarchical_params_list:
            # Create optimizer (will be updated as projection layers are added)
            hierarchical_optimizer = torch.optim.Adam(hierarchical_params_list, lr=cli_args.dream_pretrain_lr * 0.5)
            logger.info(f"[Dream-Pretrain] Hierarchical optimizer created with lr={cli_args.dream_pretrain_lr * 0.5:.6f}")

    # Track which params are in optimizer for dynamic addition
    hierarchical_params_set = set(id(p) for p in hierarchical_params_list)

    # Freeze main model if requested
    if cli_args.dream_pretrain_freeze_model:
        topas_model.eval()
        for param in topas_model.parameters():
            param.requires_grad = False
    
    # Use proven single-sample approach but benefit from larger Dream buffer (512)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)
    
    for epoch in range(cli_args.dream_pretrain_epochs):
        total_loss = 0.0
        motifs_added = 0
        buffer_len = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # ARCDataLoader returns (demos, test_inputs, test_outputs, task_id, seq_len)
            if len(batch) == 5:
                demos, test_inputs, test_outputs, task_id, seq_len = batch
            else:
                demos, test_inputs, test_outputs, task_id = batch
                seq_len = None
            if batch_idx >= cli_args.dream_pretrain_batches:
                break

            # Pick the first available test input/output (if any)
            # Handle both list and tensor formats
            if isinstance(test_inputs, (list, tuple)) and len(test_inputs) > 0:
                test_grid = test_inputs[0]
            elif torch.is_tensor(test_inputs):
                test_grid = test_inputs
            else:
                test_grid = None

            if isinstance(test_outputs, (list, tuple)) and len(test_outputs) > 0:
                target_grid = test_outputs[0]
            elif torch.is_tensor(test_outputs):
                target_grid = test_outputs
            else:
                target_grid = None

            # REMOVED: Zero-grid fallback that contaminated training
            # Fail-fast: Skip batches without real data
            if test_grid is None or target_grid is None:
                logger.warning(f"[Dream-Pretrain] Batch {batch_idx}: Missing test_grid or target_grid - SKIPPING batch to prevent contamination")
                continue  # Skip this batch entirely

            # Ensure tensors are on the right device
            test_grid = test_grid.to(device)
            target_grid = target_grid.to(device)
                
            # Get slot features from model (no grad if frozen)
            with torch.no_grad() if cli_args.dream_pretrain_freeze_model else torch.enable_grad():
                extras = {}
                if hasattr(topas_model, 'encoder'):
                    # Ensure batch dimension
                    if test_grid.dim() == 2:
                        test_grid = test_grid.unsqueeze(0)
                    enc_in = test_grid.float() / 9.0  # Normalize
                    feat, glob = topas_model.encoder(enc_in)
                    if hasattr(topas_model, 'slots'):
                        slot_vecs = topas_model.slots(feat)
                        if isinstance(slot_vecs, tuple):
                            slot_vecs = slot_vecs[0]
                        # Ensure slot_vecs has correct shape [B, num_slots, slot_dim]
                        if slot_vecs.dim() == 2:
                            slot_vecs = slot_vecs.unsqueeze(1)  # Add slot dimension
                        
                        # Concatenate global features with slot vectors to match state_dim
                        # DreamEngine expects state_dim = ctrl_dim (width + slot_dim + puzzle_emb if present)
                        B, K, D = slot_vecs.shape
                        glob_expanded = glob.unsqueeze(1).expand(B, K, -1)  # [B, K, width]
                        
                        # Check if we need to add puzzle_emb to match ctrl_dim
                        # Only add puzzle_emb if planner is present AND puzzle embeddings are enabled
                        puzzle_emb_enabled = (hasattr(topas_model, '_has_planner') and
                                            topas_model._has_planner and
                                            hasattr(topas_model, 'planner') and
                                            topas_model.planner.config.puzzle_emb_ndim > 0)

                        if puzzle_emb_enabled:
                            # Add puzzle_emb for dream pretraining
                            puzzle_emb_dim = topas_model.planner.config.puzzle_emb_ndim
                            puzzle_emb = torch.zeros(B, K, puzzle_emb_dim, device=device)
                            combined_state = torch.cat([glob_expanded, slot_vecs, puzzle_emb], dim=-1)  # [B, K, ctrl_dim]
                        else:
                            combined_state = torch.cat([glob_expanded, slot_vecs], dim=-1)  # [B, K, width+slot_dim]
                        
                        # Verify dimension matches DreamEngine's expectation
                        expected_dim = topas_model.ctrl_dim if hasattr(topas_model, 'ctrl_dim') else 768
                        if combined_state.shape[-1] != expected_dim:
                            logger.warning(f"[Dream-Pretrain] Dimension mismatch: latent has {combined_state.shape[-1]}D, expected {expected_dim}D")
                        
                        extras['latent'] = combined_state
            
            # Train Dream/ETS
            if hasattr(dream, 'train_step'):
                latent = extras.get('latent')
                if latent is not None:
                    # Log shape info only occasionally (every 100 batches)
                    if batch_idx % 100 == 0:
                        print(f"[Dream] latent: {latent.shape}, target: {target_grid.shape if target_grid is not None else None}")
                loss = dream.train_step(latent, target=target_grid)

                # ---- BitterBot: force-fill NMDA buffer and theme warmup ----
                try:
                    # Pooled [B, D] latent for buffer
                    pooled = latent.mean(dim=1) if latent.dim() == 3 else latent
                    pooled = pooled.detach()

                    # Store up to 8 quick observations per batch
                    # CRITICAL: NMDA buffer expects 1D state tensors [D], not batches!
                    max_store = min(8, pooled.shape[0] if pooled.dim() == 2 else 1)
                    for b in range(max_store):
                        s = pooled[b] if pooled.dim() == 2 else pooled

                        # Ensure 1D state vector [D]
                        if s.dim() > 1:
                            s = s.flatten()

                        # Verify state dimension matches NMDA expectations
                        if s.numel() != dream.nmda.state_dim:
                            # Skip if dimension mismatch
                            continue

                        # Use correct record_experience API
                        if hasattr(dream, 'record_experience'):
                            dream.record_experience(
                                latent_state=s,      # [D] 1D tensor
                                next_latent=s,       # Self-transition (observation)
                                action=0,            # Null action
                                reward=0.05,         # Small positive
                                valence=0.7,         # Neutral-positive
                                arousal=0.3          # Low arousal (calm observation)
                            )

                    # Periodic theme synthesis kick (if none exist yet)
                    if (batch_idx % 20 == 0) and hasattr(dream, "theme"):
                        labels = torch.arange(
                            min(max(2, pooled.shape[0] if pooled.dim() == 2 else 2), 8),
                            device=device
                        )
                        # Use a small slice to prevent OOM
                        seeds = pooled[:labels.numel()] if pooled.dim() == 2 else pooled.unsqueeze(0)
                        th = dream.theme.process_dream_themes(seeds, labels)
                        dream.theme.synthesize_emergent_themes(th)

                except Exception as e:
                    logger.warning(f"[Dream-Pretrain] Warmup buffer/theme failed: {e}")

            else:
                # REMOVED: Random weight fallback that contaminated gradients
                # Fail-fast: Dream pretraining requires proper train_step implementation
                logger.error("[Dream-Pretrain] Dream module lacks train_step method - skipping pretraining")
                logger.error("[Dream-Pretrain] Random weight fallback removed to prevent gradient contamination")
                break  # Exit pretraining loop

            if loss.requires_grad:
                dream_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dream_params, max_norm=1.0)
                dream_optimizer.step()

            total_loss += loss.item()

            # === TRAIN HIERARCHICAL ABSTRACTOR (Warm-Start on Dream Data) ===
            # Train hierarchical components to learn stable representations before main training
            if hierarchical_optimizer is not None and 'latent' in extras:
                try:
                    from trainers.hierarchical_abstraction import AbstractionLevel

                    # Get brain tokens (same as used for dream)
                    brain_tokens = extras['latent']  # [B, K, ctrl_dim]

                    # Pool to [B, ctrl_dim] for hierarchical input
                    brain_pooled = brain_tokens.mean(dim=1)  # [B, ctrl_dim]

                    # Normalize input for stability
                    brain_normalized = F.normalize(brain_pooled.unsqueeze(1), p=2, dim=-1)  # [B, 1, ctrl_dim]

                    # Extract hierarchical patterns
                    hierarchical_patterns = topas_model.hierarchical_abstractor.extract_hierarchical_patterns(
                        input_features=brain_normalized,
                        level=AbstractionLevel.GRID
                    )

                    # Reconstruction loss: Train ALL 4 projection layers to warm them up
                    hierarchical_losses = []

                    # Initialize all projection layers during pretrain
                    proj_inits = [
                        ('grid_proj', AbstractionLevel.GRID),
                        ('object_proj', AbstractionLevel.OBJECT),
                        ('abstract_proj', AbstractionLevel.ABSTRACT),
                        ('meta_proj', AbstractionLevel.META)
                    ]

                    for proj_name, level in proj_inits:
                        if level in hierarchical_patterns:
                            features = hierarchical_patterns[level]
                            pooled = features.mean(dim=1)  # [B, D]

                            # Initialize projection layer if not exists
                            if getattr(topas_model, proj_name, None) is None:
                                setattr(topas_model, proj_name, torch.nn.Linear(
                                    pooled.shape[-1], topas_model.ctrl_dim
                                ).to(device))
                                proj = getattr(topas_model, proj_name)
                                torch.nn.init.normal_(proj.weight, std=0.005)
                                torch.nn.init.zeros_(proj.bias)
                                logger.info(f"[Dream-Pretrain-Hierarchical] Initialized {proj_name}: {pooled.shape[-1]} → {topas_model.ctrl_dim}")

                                # Add new projection params to optimizer
                                for param in proj.parameters():
                                    if id(param) not in hierarchical_params_set:
                                        hierarchical_optimizer.add_param_group({'params': param, 'lr': cli_args.dream_pretrain_lr * 0.5})
                                        hierarchical_params_set.add(id(param))

                            # Reconstruct brain from this level's features
                            proj_layer = getattr(topas_model, proj_name)
                            reconstructed = proj_layer(pooled)  # [B, ctrl_dim]

                            # Reconstruction loss for this level
                            level_loss = F.mse_loss(reconstructed, brain_pooled)
                            hierarchical_losses.append(level_loss)

                    # Combined hierarchical loss
                    if hierarchical_losses:
                        hierarchical_loss = sum(hierarchical_losses) / len(hierarchical_losses)

                        # Train hierarchical components
                        hierarchical_optimizer.zero_grad()
                        hierarchical_loss.backward()
                        torch.nn.utils.clip_grad_norm_(hierarchical_params, max_norm=5.0)  # Tight clip
                        hierarchical_optimizer.step()

                        # Log occasionally
                        if batch_idx % 50 == 0:
                            logger.info(f"[Dream-Pretrain-Hierarchical] Batch {batch_idx}: loss={hierarchical_loss.item():.4f}, "
                                      f"levels_trained={len(hierarchical_losses)}")

                except Exception as e:
                    logger.debug(f"[Dream-Pretrain-Hierarchical] Training failed: {e}")

            # Track metrics
            if hasattr(dream, 'nmda') and hasattr(dream.nmda, 'buffer'):
                buffer_len = len(dream.nmda.buffer)
            if hasattr(dream, 'theme') and hasattr(dream.theme, 'synthesis_count'):
                motifs_added = dream.theme.synthesis_count
        
        avg_loss = total_loss / min(batch_idx + 1, cli_args.dream_pretrain_batches)

        # Check hierarchical gate values after epoch
        gate_status = "N/A"
        if hasattr(topas_model, 'hierarchical_gate') and topas_model.hierarchical_gate is not None:
            try:
                # Sample gates on a dummy input to see current values
                dummy_input = torch.randn(1, topas_model.ctrl_dim, device=device)
                with torch.no_grad():
                    sample_gates = topas_model.hierarchical_gate(dummy_input)
                    gate_status = f"[{', '.join([f'{v:.4f}' for v in sample_gates[0].tolist()])}]"
            except:
                gate_status = "error"

        logger.info(f"[Dream-Pretrain] Epoch {epoch+1}/{cli_args.dream_pretrain_epochs}: "
                   f"loss={avg_loss:.4f}, buffer_len={buffer_len}, motifs_added={motifs_added}, "
                   f"hierarchical_gates={gate_status}")
    
    # Save pretrained Dream/ETS
    try:
        dream.save_state('checkpoints/dream_pretrain.pth')
        logger.info("[Dream-Pretrain] saved checkpoints/dream_pretrain.pth")
    except Exception:
        logging.getLogger(__name__).exception("[Dream-Pretrain] Could not save dream state")

    # CRITICAL: Unfreeze model after dream pretraining if it was frozen
    if cli_args.dream_pretrain_freeze_model:
        logger.info("[Dream-Pretrain] Unfreezing model parameters for main training")
        topas_model.train()
        for param in topas_model.parameters():
            param.requires_grad = True

def train_step(topas_model, hrm_model, batch, optimizer, scaler, device, return_metrics=False, global_step=0):
    """Single training step with safer AMP, optional HRM->TOPAS bridge, and robust loss handling."""
    optimizer.zero_grad()
    device_type = 'cuda' if str(device).startswith('cuda') else 'cpu'

    try:
        # ARC-II: Unpack seq_len from dataset (5-tuple now)
        if len(batch) == 5:
            demos, test_inputs, test_outputs, task_id, seq_len = batch
        else:
            # Backward compatibility: 4-tuple
            demos, test_inputs, test_outputs, task_id = batch
            seq_len = None

        if not demos or len(demos) == 0:
            logging.warning("No demos in batch; skipping")
            return None

        input_grid = demos[0][0].to(device)
        target_grid = demos[0][1].to(device)

        # Normalize shapes
        if input_grid.dim() == 3 and input_grid.shape[0] == 1:
            input_grid = input_grid.squeeze(0)
        if target_grid.dim() == 3 and target_grid.shape[0] == 1:
            target_grid = target_grid.squeeze(0)

        input_grid = input_grid.unsqueeze(0)   # [1, H, W] or [1, C, H, W]
        target_grid = target_grid.unsqueeze(0)

        # === ORBIT CANONICALIZATION (Dâ‚„ + color permutations) ===
        # Initialize counters on first run
        if not hasattr(topas_model, "_orbit_canon_success"):
            topas_model._orbit_canon_success = 0
            topas_model._orbit_canon_total = 0

        canon_meta = None
        try:
            from trainers import orbits
            # Canonicalize per-sample (remove batch for function API)
            ic, oc, meta = orbits.canonicalize_pair(input_grid[0], target_grid[0])
            input_grid = ic.unsqueeze(0).to(device)
            target_grid = oc.unsqueeze(0).to(device)
            canon_meta = meta  # Store for optional inversion at eval-time

            # Track success
            topas_model._orbit_canon_success += 1
            topas_model._orbit_canon_total += 1

            # Log success periodically (use print for visibility even in speed mode)
            if global_step % 100 == 0:
                success_rate = topas_model._orbit_canon_success / max(1, topas_model._orbit_canon_total)
                print(f"[Orbit] Canonicalization active - rot={meta.get('rot', 0)}, reflect={meta.get('reflect', False)}, success_rate={success_rate:.2%}")

        except Exception as e:
            canon_meta = None
            topas_model._orbit_canon_total += 1
            # Log failures at WARNING level for visibility
            if global_step % 100 == 0:
                success_rate = topas_model._orbit_canon_success / max(1, topas_model._orbit_canon_total)
                logging.warning(f"[Orbit] Canonicalization FAILED: {e} | success_rate={success_rate:.2%}")

        # Best-effort HRM latents with real task ID for puzzle-specific learning
        hrm_latents = None
        try:
            if hasattr(hrm_model, "encode"):
                # Pass task_id for proper puzzle identification
                hrm_latents = hrm_model.encode(input_grid, task_id=task_id)
            else:
                hrm_out = hrm_model(input_grid)
                hrm_latents = hrm_out

            # CRITICAL: Fix dtype mismatch - HRM uses bfloat16, encoder/TOPAS uses float32
            # Cast HRM outputs to float32 to prevent "mat1 and mat2 must have the same dtype" errors
            if hrm_latents is not None and hrm_latents.dtype == torch.bfloat16:
                hrm_latents = hrm_latents.float()

            # === HARD FAIL SAFETY NET (GPU-FIRST enforcement) ===
            if hrm_latents is not None:
                assert torch.isfinite(hrm_latents).all(), "[Train] HRM latents contain NaN/Inf"
                assert hrm_latents.device.type == "cuda", f"[Train] HRM latents on wrong device: {hrm_latents.device}"

        except Exception:
            hrm_latents = None

        # === Adaptive Î» schedule for replay (stagnation-aware) ===
        if not hasattr(topas_model, "_acc_ema"):
            topas_model._acc_ema = 0.5  # Initialize to moderate value
            topas_model._stagnation = 0

        # Simple adaptive policy: track whether we're stuck
        # (Full implementation would use epoch_metrics from outer loop)
        topas_model._stagnation = getattr(topas_model, "_stagnation", 0)

        # Î» policy: high when potentially stuck, low when likely improving
        if topas_model._stagnation >= 8:
            lam = 0.7  # Crank replay when stuck
        elif global_step < 3000:
            lam = 0.15  # Start gentle (first ~3 epochs)
        else:
            lam = 0.3  # Default mid-strength

        # Set on model for dopamine block to use
        topas_model._lambda_replay = float(max(0.05, min(0.8, lam)))

        with torch.amp.autocast(device_type, enabled=(device.type=='cuda')):
            # Pass target shape, demos, and global_step for complete DSL+EBR integration
            target_shape = target_grid.shape[-2:]  # (H, W)
            try:
                outputs = topas_model.forward_pretraining(
                    input_grid,
                    hrm_latents=hrm_latents,
                    target_shape=target_shape,
                    demos=demos,
                    global_step=global_step
                )
            except TypeError:
                # Fallback for older signature
                outputs = topas_model.forward_pretraining(input_grid)

            # Expect outputs to be dict-like and contain 'logits'
            if isinstance(outputs, dict) and 'logits' in outputs and outputs['logits'] is not None:
                logits = outputs['logits']  # Should already be [B, H*W, C]
                
                # Check for None return (model detected issues)
                if logits is None:
                    logging.warning("Model returned None logits, skipping batch")
                    return None
                
                # Ensure target is properly shaped
                B = logits.size(0)
                H, W = target_grid.shape[-2:]
                # Robust flattening: handle non-contiguous tensors (rotated/cropped)
                target_flat = target_grid.reshape(B, -1).long()
                
                # Auto-align shapes if needed (model should handle this now)
                if logits.size(1) != target_flat.size(1):
                    logging.warning(f"Shape mismatch: logits {logits.shape} vs target {target_flat.shape}, skipping")
                    return None
                
                # Sanity check targets
                assert (target_flat >= 0).all() and (target_flat < 10).all(), f"Invalid target values: {target_flat.unique()}"
                
                # Cross-entropy with label smoothing
                label_smoothing = 0.05
                ce_loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target_flat.reshape(-1),
                    label_smoothing=label_smoothing
                )
                ce_loss = ce_loss.float()  # Ensure float32 for AMP/GradScaler
                
                # Batch debug probe for CE spikes
                if global_step % 100 == 0 and ce_loss > 2.0:  # Log when CE loss is high
                    batch_shapes = [tuple(input_grid.shape), tuple(target_grid.shape)]
                    logger.info(f"[BATCH DEBUG] CE_spike={ce_loss.item():.3f} batch_shapes={batch_shapes}")
                
                # Dream health check - log cached tokens if available
                if global_step % 100 == 0 and getattr(topas_model, "_dream_tokens", None) is not None:
                    tokens_shape = getattr(topas_model, "_dream_tokens").shape
                    logging.info(f"[Dream] _dream_tokens shape: {tokens_shape}")
                
                # Track loss for RelMem success approximation
                topas_model._last_loss_value = float(ce_loss.item())

                # Add DSL losses (weight already annealed in model)
                total_loss = ce_loss
                if 'losses' in outputs and outputs['losses']:
                    for loss_name, loss_value in outputs['losses'].items():
                        # GPU-FIRST + GRADIENT validation
                        if not isinstance(loss_value, torch.Tensor):
                            logging.warning(f"❌ {loss_name} not tensor: {type(loss_value)}")
                            continue
                        if not loss_value.requires_grad:
                            # mdl_prior is computed in no_grad (uses argmax - non-differentiable), silently skip
                            if loss_name != 'mdl_prior':
                                logging.warning(f"❌ {loss_name} no gradient!")
                            continue
                        if not loss_value.is_cuda:
                            logging.warning(f"❌ GPU VIOLATION: {loss_name} on {loss_value.device}")
                            continue

                        total_loss = total_loss + loss_value  # Weight already applied in model

                        # Special logging for bridge_dsl to verify gradient flow fix
                        if loss_name == 'bridge_dsl' and global_step % 20 == 0:
                            logging.info(f"[BRIDGE] ✅ Step {global_step}: bridge_dsl={loss_value.item():.6f} (requires_grad={loss_value.requires_grad})")
                        elif global_step % 100 == 0:  # Log occasionally for other losses
                            logging.info(f"[AUX] Step {global_step}: {loss_name}={loss_value.item():.4f} (ce={ce_loss.item():.3f})")

                # === ANDROMEDA CORTEX LOSSES ===
                cortex_losses = outputs.get('extras', {}).get('cortex', {}).get('losses', {})
                if cortex_losses:
                    # Get loss weights from config
                    w_recon = getattr(topas_model.config, 'lambda_cortex_recon', 1.0)
                    w_entropy = getattr(topas_model.config, 'lambda_cortex_entropy', 1.0)
                    w_sparsity = getattr(topas_model.config, 'lambda_cortex_sparsity', 0.5)

                    cortex_total = 0.0
                    if 'pc_recon' in cortex_losses and cortex_losses['pc_recon'].requires_grad:
                        total_loss = total_loss + w_recon * cortex_losses['pc_recon']
                        cortex_total += w_recon * cortex_losses['pc_recon'].item()
                    if 'pc_entropy' in cortex_losses and cortex_losses['pc_entropy'].requires_grad:
                        total_loss = total_loss + w_entropy * cortex_losses['pc_entropy']
                        cortex_total += w_entropy * cortex_losses['pc_entropy'].item()
                    if 'kl_sparsity' in cortex_losses and cortex_losses['kl_sparsity'].requires_grad:
                        total_loss = total_loss + w_sparsity * cortex_losses['kl_sparsity']
                        cortex_total += w_sparsity * cortex_losses['kl_sparsity'].item()

                    if global_step % 100 == 0 and cortex_total > 0:
                        cortex_ratio = cortex_total / (ce_loss.item() + cortex_total) * 100
                        logging.info(f"[Cortex] Step {global_step}: total={cortex_total:.4f} ({cortex_ratio:.1f}% of loss)")

                # === CRITIC HEAD LOSS (EM Likelihood Prediction) ===
                # Train critic to predict exact match likelihood (aligns with HRM value_logit)
                if 'critic_logit' in outputs and outputs['critic_logit'] is not None:
                    critic_logit = outputs['critic_logit']  # [B]
                    # Compute EM target: 1.0 if prediction matches target, 0.0 otherwise
                    pred_grid = outputs.get('pred_grid')  # [B, H, W]
                    if pred_grid is not None:
                        # Ensure target_grid matches pred_grid shape
                        B, H, W = pred_grid.shape
                        target_resized = target_grid
                        if target_resized.shape[-2:] != (H, W):
                            target_resized = torch.nn.functional.interpolate(
                                target_grid.unsqueeze(1).float(), size=(H, W), mode='nearest'
                            ).squeeze(1).long()
                        # Compute exact match per sample
                        em_target = (pred_grid == target_resized).all(dim=(1, 2)).float()  # [B]
                        # BCE loss
                        critic_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            critic_logit, em_target
                        )
                        # Add to total loss with weight 0.2 (same as MultiHeadPretrainer)
                        critic_weight = 0.2
                        total_loss = total_loss + critic_weight * critic_loss
                        if global_step % 100 == 0:
                            logging.info(f"[Critic] Step {global_step}: loss={critic_loss.item():.4f}, "
                                       f"EM_rate={em_target.mean().item():.2%}")

                # === ORBIT-CONTRASTIVE LOSS (OrthoNCE) ===
                # Enforces feature invariance under Dâ‚„ orbit
                # Initialize counter
                if not hasattr(topas_model, "_orbit_loss_success"):
                    topas_model._orbit_loss_success = 0
                    topas_model._orbit_loss_total = 0

                try:
                    from trainers import orbits
                    # Get lambda from config or CLI (more robust lookup)
                    lambda_orbit = 0.03  # default
                    if hasattr(topas_model.config, 'orbit_loss_weight'):
                        lambda_orbit = float(topas_model.config.orbit_loss_weight)
                    elif 'args' in locals() and hasattr(args, 'orbit_loss_weight'):
                        lambda_orbit = float(args.orbit_loss_weight)

                    if lambda_orbit > 0:
                        o_loss = orbits.orbit_invariance_loss(topas_model, input_grid[0], k=2, tau=0.2)
                        if torch.isfinite(o_loss):
                            total_loss = total_loss + lambda_orbit * o_loss
                            topas_model._orbit_loss_success += 1
                            topas_model._orbit_loss_total += 1

                            if global_step % 100 == 0:
                                success_rate = topas_model._orbit_loss_success / max(1, topas_model._orbit_loss_total)
                                print(f"[Orbit] loss={o_loss:.4f} λ={lambda_orbit:.3f} success_rate={success_rate:.2%}")
                        else:
                            topas_model._orbit_loss_total += 1
                            if global_step % 100 == 0:
                                logging.warning(f"[Orbit] Non-finite loss detected!")
                except Exception as e:
                    topas_model._orbit_loss_total += 1
                    if global_step % 100 == 0:
                        success_rate = topas_model._orbit_loss_success / max(1, topas_model._orbit_loss_total)
                        logging.warning(f"[Orbit] Loss computation FAILED: {e} | success_rate={success_rate:.2%}")

                # === Joint HRM Training with ACT Loss ===
                if hasattr(topas_model, 'planner_loss_head') and global_step > 1000:  # After initial warmup
                    try:
                        # Ensure HRM planner_loss_head is on correct device
                        if hasattr(topas_model.planner_loss_head, 'to'):
                            topas_model.planner_loss_head = topas_model.planner_loss_head.to(device)

                        # HRM supervision targets - need padded tokens to match NeuroPlanner interface
                        if hasattr(hrm_model, 'get_padded_tokens'):
                            padded_tokens, original_seq_len = hrm_model.get_padded_tokens(input_grid)
                            tokens = padded_tokens
                            # CRITICAL: Ensure tokens are on correct device immediately
                            tokens = tokens.to(device)
                        else:
                            # Fallback: create padded tokens manually
                            unpadded_tokens = topas_model.grid_to_tokens(input_grid)
                            original_seq_len = unpadded_tokens.shape[1]

                            # 🔧 ADAPTIVE SEQ_LEN: Use actual grid size instead of fixed 900
                            # ARC-II grids: 1x1 to 30x30 (1-900 pixels), but most are 5x5 to 20x20 (25-400)
                            # Padding to 900 wastes 4-36x compute on small grids
                            H, W = input_grid.shape[-2:]
                            actual_pixels = H * W
                            # Adaptive padding: at least 64 (for very small grids), at most 900 (max ARC size)
                            expected_seq_len = min(900, max(actual_pixels, 64))

                            if original_seq_len < expected_seq_len:
                                padding = expected_seq_len - original_seq_len
                                tokens = F.pad(unpadded_tokens, (0, padding), mode='constant', value=0)
                            else:
                                tokens = unpadded_tokens[:, :expected_seq_len]
                                original_seq_len = expected_seq_len

                            tokens = tokens.to(device)

                        # task_id was already unpacked above from the dataloader tuple
                        raw_pid = task_id
                        if isinstance(raw_pid, torch.Tensor):
                            pid = int(raw_pid.item())
                        elif isinstance(raw_pid, (int, np.integer)):
                            pid = int(raw_pid)
                        elif isinstance(raw_pid, str):
                            # map string â†’ stable integer in [0, 999]
                            pid = int(hashlib.sha1(raw_pid.encode()).hexdigest(), 16) % 1000
                        else:
                            pid = int(hashlib.sha1(str(raw_pid).encode()).hexdigest(), 16) % 1000
                        puzzle_ids = torch.tensor([pid], device=device, dtype=torch.long)
                        
                        # Ensure all tensors are on the same device
                        batch_dict = {
                            "inputs": tokens.to(device),
                            "labels": tokens.to(device),  # Identity reconstruction for early training
                            "puzzle_identifiers": puzzle_ids.to(device)
                        }
                        carry = topas_model.planner_loss_head.initial_carry(batch_dict)

                        # Ensure carry is on correct device
                        if isinstance(carry, dict):
                            carry = {k: v.to(device) if torch.is_tensor(v) else v for k, v in carry.items()}
                        elif torch.is_tensor(carry):
                            carry = carry.to(device)

                        new_carry, hrm_loss, hrm_metrics, _, _ = topas_model.planner_loss_head(
                            return_keys=[], carry=carry, batch=batch_dict
                        )

                        # Ensure hrm_loss is on correct device
                        if isinstance(hrm_loss, torch.Tensor):
                            hrm_loss = hrm_loss.to(device)

                        # Only add HRM loss if it has gradients
                        if isinstance(hrm_loss, torch.Tensor) and hrm_loss.requires_grad:
                            total_loss = total_loss + 0.2 * hrm_loss  # Start with modest weight
                        
                        # Log HRM metrics
                        if global_step % 100 == 0:
                            lm_loss = hrm_metrics.get("lm_loss", 0.0)
                            q_halt_loss = hrm_metrics.get("q_halt_loss", 0.0)
                            logging.info(f"[HRM-Joint] hrm_loss={hrm_loss:.3f}, lm_loss={lm_loss:.3f}, q_halt_loss={q_halt_loss:.3f}")
                    except Exception as e:
                        if global_step % 100 == 0:
                            logging.warning(f"[HRM-Joint] Supervision failed: {e}")
                
                # ---- RelMem auxiliary loss every N steps (with warm-up) ----
                # Delay RelMem until after 5 epochs (~2000 steps)
                relmem_warmup_epochs = 5
                current_epoch = global_step // 400  # assumes ~400 steps/epoch
                relmem_loss_interval = getattr(args, 'relmem_loss_interval', 25) if 'args' in locals() else 25
                if (hasattr(topas_model, "relmem") and topas_model.relmem is not None and 
                    current_epoch >= relmem_warmup_epochs and (global_step % relmem_loss_interval == 0)):
                    try:
                        reg_alpha = getattr(args, 'relmem_reg_alpha', 1e-4) if 'args' in locals() else 1e-4
                        reg_beta  = getattr(args, 'relmem_reg_beta', 1e-4)  if 'args' in locals() else 1e-4

                        # Safe aggregation: only add terms that are tensors with â‰¥1D.
                        relmem_aux = torch.tensor(0.0, device=device)
                        if hasattr(topas_model.relmem, 'inverse_loss_safe'):
                            inv_loss = topas_model.relmem.inverse_loss_safe()
                            if torch.is_tensor(inv_loss) and inv_loss.dim() >= 1:
                                relmem_aux = relmem_aux + reg_alpha * inv_loss.sum()
                        elif hasattr(topas_model.relmem, 'inverse_loss'):
                            inv_loss = topas_model.relmem.inverse_loss()
                            if torch.is_tensor(inv_loss) and inv_loss.dim() >= 1:
                                relmem_aux = relmem_aux + reg_alpha * inv_loss.sum()

                        # Only call inheritance_pass if a safe variant exists; otherwise skip to avoid 0D@0D matmul
                        if hasattr(topas_model.relmem, 'inheritance_pass_safe'):
                            inh = topas_model.relmem.inheritance_pass_safe()
                            if torch.is_tensor(inh) and inh.dim() >= 1:
                                relmem_aux = relmem_aux + reg_beta * inh.sum()
                        else:
                            # No safe variant; keep silent and skip to avoid console spam
                            pass
                        
                        if torch.is_tensor(relmem_aux) and relmem_aux.item() > 0:
                            total_loss = total_loss + relmem_aux
                            if global_step % 100 == 0:
                                logging.info(f"Step {global_step}: RelMem aux loss={relmem_aux.item():.6f}")
                    except Exception as e:
                        # Downgrade to debug to avoid spam when RelMem lacks safe hooks
                        logging.debug(f"RelMem auxiliary loss skipped: {e}")
                
                # RelMem binding on success (check if we have metrics to evaluate)
                if return_metrics:
                    try:
                        # Quick metrics for binding decision
                        preds = logits.argmax(dim=-1)  # [B, H*W]
                        targets_flat = target_grid.reshape(B, -1)  # [B, H*W]
                        exact_match = (preds == targets_flat).all(dim=1).float().mean().item()
                        
                        # Compute IoU for binding threshold
                        ious = []
                        for c in range(10):  # NUM_COLORS
                            pred_c = (preds == c)
                            target_c = (targets_flat == c)
                            intersection = (pred_c & target_c).sum().float()
                            union = (pred_c | target_c).sum().float()
                            if union > 0:
                                ious.append((intersection / union).item())
                        mean_iou = sum(ious) / len(ious) if ious else 0.0
                        
                        # Enhanced RelMem concept binding on success
                        if hasattr(topas_model, 'relmem') and topas_model.relmem is not None:
                            # Extract brain latent from outputs
                            brain_latent = None
                            if hasattr(outputs, 'brain') and outputs['brain'] is not None:
                                brain_latent = outputs['brain']
                            elif 'brain' in outputs and outputs['brain'] is not None:
                                brain_latent = outputs['brain']
                            elif 'latent' in outputs and outputs['latent'] is not None:
                                brain_latent = outputs['latent']

                            if brain_latent is not None:
                                # Lowered gates: allow wormhole mining with partial success (EM/IoU >= thresholds)
                                min_em_th = getattr(cli_args, "wormhole_min_em", 0.30) if 'cli_args' in globals() and cli_args else 0.30
                                min_iou_th = getattr(cli_args, "wormhole_min_iou", 0.30) if 'cli_args' in globals() and cli_args else 0.30
                                em_success = exact_match > 0
                                cond_partial = (isinstance(mean_iou, (int, float)) and mean_iou >= min_iou_th) or \
                                               (isinstance(exact_match, (int, float)) and exact_match >= min_em_th) or \
                                               (isinstance(em_success, bool) and em_success)

                                if cond_partial:
                                    # 🔧 WORMHOLE FIX: Extract DSL ops from RelMem and populate extras
                                    # This enables wormhole to mine templates even during pretrain
                                    dsl_ops_inferred = []
                                    try:
                                        from models.dsl_registry import DSL_OPS
                                        op_bias = topas_model.relmem.get_op_bias(dsl_ops=DSL_OPS)
                                        if op_bias and isinstance(op_bias, dict):
                                            # Top-5 operations by bias score
                                            top_ops = sorted(op_bias.items(), key=lambda x: x[1], reverse=True)[:5]
                                            dsl_ops_inferred = [op for op, score in top_ops if score > 0.1]
                                    except Exception:
                                        pass

                                    # Store in extras for wormhole mining
                                    if dsl_ops_inferred:
                                        if 'extras' not in outputs:
                                            outputs['extras'] = {}
                                        outputs['extras']['ops_used'] = dsl_ops_inferred

                                    # Perform concept binding
                                    try:
                                        if hasattr(topas_model, '_relmem_try_bind'):
                                            binding_stats = topas_model._relmem_try_bind(
                                                brain=brain_latent,
                                                ops_used=outputs.get('extras', {}).get('ops_used', []),
                                                iou=mean_iou,
                                                em=float(exact_match) if isinstance(exact_match, (int, float)) else (1.0 if em_success else 0.0)
                                            )
                                        else:
                                            # Fallback binding method - use proper API
                                            binding_stats = {'relmem_bound': False}
                                            if hasattr(topas_model.relmem, 'bind_concept_by_vector'):
                                                # Use helper that handles cid allocation internally
                                                topas_model.relmem.bind_concept_by_vector(
                                                    brain_latent,
                                                    op_name="success_pattern",
                                                    meta={"iou": float(mean_iou), "em": float(em_success)},
                                                    alpha=0.3
                                                )
                                                binding_stats['relmem_bound'] = True
                                                binding_stats['relmem_concept_id'] = 'auto'
                                            elif hasattr(topas_model.relmem, 'add_concept'):
                                                # Manual allocation
                                                cid = topas_model.relmem.add_concept(
                                                    brain_latent,
                                                    meta={"iou": float(mean_iou), "em": float(em_success)},
                                                    alpha=0.3
                                                )
                                                topas_model.relmem.bind_concept(cid, brain_latent, alpha=0.3)
                                                binding_stats['relmem_bound'] = True
                                                binding_stats['relmem_concept_id'] = cid
                                        
                                        # --- Wormhole integration ---
                                        try:
                                            if hasattr(topas_model, "dream") and hasattr(topas_model.dream, "wormhole"):
                                                rule_info = outputs.get('extras', {}).get('rule_info')
                                                ops = outputs.get('extras', {}).get('ops_used', [])
                                                programs = rule_info.get("programs", [ops]) if isinstance(rule_info, dict) else [ops]
                                                try:
                                                    mined = topas_model.dream.wormhole.mine_from_programs(programs, top_k=5)
                                                except Exception as e:
                                                    logging.getLogger(__name__).warning(f"[Wormhole] mining raised (diagnose upstream): {e}")
                                                    mined = []
                                                if mined:
                                                    logging.getLogger(__name__).info(f"[Wormhole] mined {len(mined)} templates")
                                                    # Bind mined templates into RelMem
                                                    for tpl in mined:
                                                        try:
                                                            tpl_sig = str(tpl)[:64]  # Extended from 32 to 64 chars
                                                            if hasattr(topas_model.relmem, "bind_concept_by_vector"):
                                                                topas_model.relmem.bind_concept_by_vector(
                                                                    brain_latent.squeeze(0), op_name="wormhole",
                                                                    meta={"template": tpl_sig, "source": "lowered_gate"}, alpha=0.5
                                                                )
                                                        except Exception as e:
                                                            logging.getLogger(__name__).warning(f"[RelMem] bind wormhole template failed: {e}")
                                        except Exception as e:
                                            logging.getLogger(__name__).warning(f"[Wormhole] integration failed: {e}")
                                        
                                        if binding_stats.get('relmem_bound') and global_step % 100 == 0:
                                            concept_id = binding_stats.get('relmem_concept_id', 'unknown')
                                            logging.info(f"RelMem bound concept {concept_id} on success (EM={exact_match:.3f}, IoU={mean_iou:.3f})")
                                    except Exception as e:
                                        if global_step % 200 == 0:
                                            logging.warning(f"RelMem concept binding failed: {e}")
                    except Exception as e:
                        if global_step % 100 == 0:
                            logging.warning(f"RelMem binding failed: {e}")
                
                # RelMem regularization
                relmem_reg_loss = torch.tensor(0.0, device=device)
                if hasattr(topas_model, 'relmem') and topas_model.relmem is not None:
                    try:
                        # Compute RelMem regularization
                        reg_alpha = getattr(args, 'relmem_reg_alpha', 0.01) if 'args' in locals() else 0.01
                        reg_beta = getattr(args, 'relmem_reg_beta', 0.02) if 'args' in locals() else 0.02
                        
                        if hasattr(topas_model.relmem, 'compute_regularization'):
                            relmem_reg_loss = topas_model.relmem.compute_regularization(alpha=reg_alpha, beta=reg_beta)
                        elif hasattr(topas_model.relmem, 'regularization_loss'):
                            relmem_reg_loss = topas_model.relmem.regularization_loss() * reg_alpha
                            
                        if torch.is_tensor(relmem_reg_loss) and relmem_reg_loss.item() > 0:
                            total_loss = total_loss + relmem_reg_loss
                            
                    except Exception as e:
                        if global_step % 100 == 0:
                            logging.warning(f"RelMem regularization failed: {e}")
                
                # Safety: detect and prevent loss explosions
                max_reasonable_loss = ce_loss * 10.0
                if total_loss > max_reasonable_loss or total_loss < -1.0:
                    aux_count = len(outputs.get('losses', {}))
                    logging.error(f"[LOSS] EXPLOSION: total={total_loss.item():.2f} ce={ce_loss.item():.2f} aux_count={aux_count}")
                    logging.error(f"[LOSS] Emergency: reverting to CE only")
                    total_loss = ce_loss

                loss = total_loss

                # TRANSPARENCY: Log total loss breakdown (verbose only) - AFTER safety check
                if global_step % 100 == 0:
                    aux_count = len(outputs.get('losses', {}))
                    aux_total = float((loss - ce_loss).item()) if loss != ce_loss else 0.0
                    logging.info(f"[LOSS] Step {global_step}: total={loss.item():.4f} ce={ce_loss.item():.4f} aux={aux_total:.4f} (n={aux_count})")
            else:
                logging.error("train_step: outputs missing 'logits'; keys=%s", (list(outputs.keys()) if isinstance(outputs, dict) else type(outputs)))
                return None

        if loss is None or (isinstance(loss, torch.Tensor) and not torch.isfinite(loss).all()):
            logging.error("Invalid loss (NaN/Inf) at global step %d, skipping", global_step)
            return None

        scaler.scale(loss).backward()

        # CRITICAL: Sanitize concept_proto gradients immediately after backward
        if hasattr(topas_model, 'relmem') and hasattr(topas_model.relmem, 'concept_proto'):
            if topas_model.relmem.concept_proto.grad is not None:
                if not torch.isfinite(topas_model.relmem.concept_proto.grad).all():
                    logger.error(f"[GRAD SANITIZE] concept_proto.grad has NaN/Inf! Zeroing to prevent corruption.")
                    topas_model.relmem.concept_proto.grad.zero_()

        # Gradient probe: verify dopamine replay produces real gradients
        if "dopamine_replay" in outputs.get("losses", {}):
            try:
                if hasattr(topas_model, 'pixel_fallback') and hasattr(topas_model.pixel_fallback, 'weight'):
                    grad = topas_model.pixel_fallback.weight.grad
                    if grad is not None:
                        gn = grad.norm().item()
                        logger.info(f"[GradProbe] pixel_fallback grad_norm={gn:.6e} (dopamine active)")
                    else:
                        logger.warning(f"[GradProbe] pixel_fallback.grad is None despite dopamine loss!")
            except Exception as probe_err:
                logger.debug(f"[GradProbe] Failed: {probe_err}")

        # Cache batch and programs for dream seeding (every 50 steps to reduce overhead)
        if global_step % 50 == 0:
            try:
                # Extract programs from outputs
                programs = outputs.get("extras", {}).get("programs", [])
                
                # Store in a global that main() can access
                # GPU-FIRST: Keep grids on GPU (only .cpu() when saving checkpoints)
                globals()['last_batch_for_dream'] = {
                    'test_grid': input_grid.detach(),
                    'task_id': task_id,
                    'programs': programs
                }
            except Exception:
                pass
        
        # Store outputs for enhanced dopamine scoring (when metrics computed)
        if return_metrics:
            try:
                # GPU-FIRST: Keep all tensors on GPU for dopamine computation
                globals()['last_outputs_for_dopamine'] = {
                    'outputs': outputs,
                    'input_grid': input_grid.detach(),
                    'target_grid': target_grid.detach(),
                    'global_step': global_step
                }
            except Exception:
                pass
        
        # === GRADIENT FLOW VERIFICATION (every 200 steps) ===
        if global_step % 200 == 0:
            scaler.unscale_(optimizer)  # Unscale to get true grad norms

            # Log encoder gradients to verify Dream losses reach it
            encoder_grad_norms = {}
            for name, param in topas_model.named_parameters():
                if param.grad is not None and 'encoder' in name:
                    encoder_grad_norms[name] = param.grad.norm().item()

            if encoder_grad_norms:
                total_enc_grad = sum(encoder_grad_norms.values())
                logger.info(f"[GRAD] step={global_step} encoder_total_norm={total_enc_grad:.6f} "
                           f"params_with_grad={len(encoder_grad_norms)}")

            # === HIERARCHICAL ABSTRACTOR GRADIENT CHECK ===
            if hasattr(topas_model, 'hierarchical_abstractor') and topas_model.hierarchical_abstractor is not None:
                hierarchical_grad_norms = {}
                hierarchical_grad_count = 0
                hierarchical_no_grad_params = []

                for name, param in topas_model.hierarchical_abstractor.named_parameters():
                    if param.grad is not None:
                        hierarchical_grad_norms[name] = param.grad.norm().item()
                        hierarchical_grad_count += 1
                    else:
                        hierarchical_no_grad_params.append(name)

                if hierarchical_grad_norms:
                    total_hierarchical_grad = sum(hierarchical_grad_norms.values())
                    logger.info(f"[GRAD-Hierarchical] step={global_step} total_norm={total_hierarchical_grad:.6f} "
                               f"params_with_grad={hierarchical_grad_count}/{len(list(topas_model.hierarchical_abstractor.parameters()))}")

                if hierarchical_no_grad_params:
                    logger.warning(f"[GRAD-Hierarchical] Parameters WITHOUT gradients: {hierarchical_no_grad_params[:5]}")

            # Check hierarchical gate network gradients (learnable gating)
            if hasattr(topas_model, 'hierarchical_gate') and topas_model.hierarchical_gate is not None:
                gate_grad_norm = 0.0
                for param in topas_model.hierarchical_gate.parameters():
                    if param.grad is not None:
                        gate_grad_norm += param.grad.norm().item()

                if gate_grad_norm > 0:
                    logger.info(f"[GRAD-HierarchicalGate] step={global_step} total_norm={gate_grad_norm:.4f}")

            # Check all hierarchical projection layers (multi-scale fusion)
            hierarchical_proj_layers = ['grid_proj', 'object_proj', 'abstract_proj', 'meta_proj']
            hierarchical_proj_grads = {}

            for proj_name in hierarchical_proj_layers:
                if hasattr(topas_model, proj_name):
                    proj_layer = getattr(topas_model, proj_name)
                    if proj_layer is not None and hasattr(proj_layer, 'weight') and proj_layer.weight.grad is not None:
                        grad_norm = proj_layer.weight.grad.norm().item()
                        hierarchical_proj_grads[proj_name] = grad_norm

            if hierarchical_proj_grads:
                grad_summary = ', '.join([f"{name}={norm:.2f}" for name, norm in hierarchical_proj_grads.items()])
                logger.info(f"[GRAD-HierarchicalProj] step={global_step} {grad_summary}")
            elif hasattr(topas_model, 'hierarchical_abstractor') and topas_model.hierarchical_abstractor is not None:
                logger.warning(f"[GRAD-HierarchicalProj] step={global_step} NO GRADIENTS on projection layers")

            # === HIERARCHICAL PATTERNS EXTRACTION VERIFICATION ===
            if 'hierarchical_patterns' in outputs.get('extras', {}):
                from trainers.hierarchical_abstraction import AbstractionLevel
                patterns = outputs['extras']['hierarchical_patterns']
                pattern_info = []
                for level in AbstractionLevel:
                    if level in patterns:
                        tensor = patterns[level]
                        pattern_info.append(f"{level.name}={list(tensor.shape)}")
                    elif isinstance(level, str) and level in patterns:  # Handle string keys
                        tensor = patterns[level]
                        pattern_info.append(f"{level}={list(tensor.shape)}")

                logger.info(f"[HierarchicalPatterns] step={global_step} extracted: {', '.join(pattern_info)}")

                # Check multi-scale contributions AND gate values
                contribution_levels = ['grid_contribution', 'object_contribution', 'abstract_contribution', 'meta_contribution']
                gate_levels = ['grid_gate', 'object_gate', 'abstract_gate', 'meta_gate']

                found_contributions = {}
                found_gates = {}

                for contrib_name, gate_name in zip(contribution_levels, gate_levels):
                    if contrib_name in outputs.get('extras', {}):
                        contrib = outputs['extras'][contrib_name]
                        contrib_norm = contrib.norm(dim=-1).mean().item()
                        level_name = contrib_name.replace('_contribution', '')
                        found_contributions[level_name] = contrib_norm

                    if gate_name in outputs.get('extras', {}):
                        gate_value = outputs['extras'][gate_name]
                        level_name = gate_name.replace('_gate', '')
                        found_gates[level_name] = gate_value

                if found_contributions:
                    contrib_summary = ', '.join([f"{name}={norm:.4f}" for name, norm in found_contributions.items()])
                    logger.info(f"[HierarchicalPatterns] contributions: {contrib_summary}")

                if found_gates:
                    gate_summary = ', '.join([f"{name}={val:.4f}" for name, val in found_gates.items()])
                    logger.info(f"[HierarchicalPatterns] gates (learned): {gate_summary}")

                # Check total contribution
                if 'hierarchical_total_contribution' in outputs.get('extras', {}):
                    total_contrib = outputs['extras']['hierarchical_total_contribution']
                    total_norm = total_contrib.norm(dim=-1).mean().item()
                    logger.info(f"[HierarchicalPatterns] total_contribution_norm={total_norm:.4f} (gated sum of {len(found_contributions)} levels)")

                if not found_contributions and not found_gates:
                    logger.warning(f"[HierarchicalPatterns] NO contributions/gates found in extras")
            else:
                if hasattr(topas_model, 'hierarchical_abstractor') and topas_model.hierarchical_abstractor is not None:
                    logger.warning(f"[HierarchicalPatterns] step={global_step} NOT EXTRACTED (check forward pass)")

            # Log per-loss gradient connectivity
            if hasattr(outputs, 'get') and 'losses' in outputs:
                for loss_name, loss_tensor in outputs["losses"].items():
                    if torch.is_tensor(loss_tensor) and loss_tensor.requires_grad:
                        has_grad_fn = loss_tensor.grad_fn is not None
                        logger.info(f"[GRAD] {loss_name}: requires_grad={loss_tensor.requires_grad} "
                                   f"has_grad_fn={has_grad_fn}")

            # Now apply gradient clipping
            torch.nn.utils.clip_grad_norm_(topas_model.parameters(), max_norm=0.5)
        else:
            # Normal path: just unscale and clip
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(topas_model.parameters(), max_norm=0.5)

        scaler.step(optimizer)
        scaler.update()
        
        # Apply post-optimizer hooks for RelMem
        if hasattr(topas_model, 'relmem') and topas_model.relmem is not None:
            try:
                if hasattr(topas_model.relmem, 'apply_post_optimizer_hooks'):
                    topas_model.relmem.apply_post_optimizer_hooks()
                elif hasattr(topas_model.relmem, 'post_optimizer_step'):
                    topas_model.relmem.post_optimizer_step()
            except Exception as e:
                if global_step % 500 == 0:
                    logging.warning(f"RelMem post-optimizer hooks failed: {e}")

        # === BRAIN-GRAPH LEARNING (Decoupled from metrics - runs on EVERY successful batch) ===
        if hasattr(topas_model, 'brain_graph') and topas_model.brain_graph is not None:
            try:
                # Quick lightweight EM check without full metrics
                preds = logits.argmax(dim=-1)  # [B, H*W]
                B = logits.size(0)
                targets_flat = target_grid.reshape(B, -1)  # [B, H*W]
                quick_em = (preds == targets_flat).all(dim=1).float().mean().item()

                if quick_em >= 0.5:  # Learn from highly successful examples
                    # Extract puzzle attributes (lightweight version)
                    puzzle_attrs = _extract_puzzle_attributes(input_grid[0] if input_grid.dim() == 4 else input_grid,
                                                            target_grid[0] if target_grid.dim() == 4 else target_grid)

                    # Extract DSL ops from RelMem op_bias
                    dsl_ops_used = []
                    if hasattr(topas_model, 'relmem') and topas_model.relmem is not None:
                        try:
                            from models.dsl_registry import DSL_OPS
                            op_bias = topas_model.relmem.get_op_bias(dsl_ops=DSL_OPS)
                            if op_bias and isinstance(op_bias, dict):
                                # Top-3 operations by bias score
                                top_ops = sorted(op_bias.items(), key=lambda x: x[1], reverse=True)[:3]
                                dsl_ops_used = [op for op, score in top_ops if score > 0.1]
                        except Exception:
                            pass

                    # Fallback if no ops available
                    if not dsl_ops_used:
                        dsl_ops_used = ["neural_pattern"]  # Placeholder for pure neural success

                    # Track node count before learning
                    nodes_before = len(topas_model.brain_graph.nodes)

                    # Learn from this successful pattern
                    topas_model.brain_graph.observe_success(
                        puzzle_attrs=puzzle_attrs,
                        dsl_ops_used=dsl_ops_used,
                        success=True,
                        task_id=str(task_id) if 'task_id' in locals() else None
                    )

                    # === BRAIN-GRAPH → RELMEM FUSION (Seed RelMem from learned brain patterns) ===
                    nodes_after = len(topas_model.brain_graph.nodes)
                    if nodes_after > nodes_before and hasattr(topas_model, 'relmem') and topas_model.relmem is not None:
                        try:
                            # Get the newly created concept's attributes
                            attr_signature = tuple(sorted(puzzle_attrs))
                            concept_id = f"learned_pattern_{hash(attr_signature) % 100000}"

                            if concept_id in topas_model.brain_graph.nodes:
                                # Synthesize puzzle embedding from brain-graph attributes
                                recalled_attrs = topas_model.brain_graph.attribute_recall(concept_id)
                                puzzle_emb = topas_model.brain_graph.to_puzzle_embedding(recalled_attrs)

                                if puzzle_emb is not None:
                                    # Seed RelMem concept with brain-graph embedding
                                    # This provides warm-start instead of random initialization
                                    success_rate = topas_model.brain_graph.nodes[concept_id].features.get('success_rate', 1.0)

                                    # Proper API: add_concept returns cid, then bind
                                    cid = topas_model.relmem.add_concept(
                                        puzzle_emb.unsqueeze(0),  # [1, D]
                                        meta={
                                            'brain_seeded': True,
                                            'success_rate': float(success_rate),
                                            'brain_concept_id': concept_id
                                        },
                                        alpha=0.3
                                    )
                                    topas_model.relmem.bind_concept(cid, puzzle_emb.unsqueeze(0), alpha=0.3)

                                    if global_step % 100 == 0:
                                        logging.info(f"[Brain→RelMem] Seeded concept cid={cid} from {concept_id[:12]} (success_rate={success_rate:.2f})")
                        except Exception as e:
                            if global_step % 500 == 0:
                                logging.warning(f"[Brain→RelMem] Fusion failed: {e}")

                    # Log brain-graph growth every 100 steps
                    if global_step % 100 == 0:
                        node_count = len(topas_model.brain_graph.nodes)
                        concept_count = sum(1 for n in topas_model.brain_graph.nodes.values() if n.kind == "concept")
                        logging.info(f"[BrainGraph] Learned from EM={quick_em:.0%} | Nodes: {node_count} (concepts={concept_count})")
            except Exception as e:
                if global_step % 500 == 0:
                    logging.warning(f"[BrainGraph] Learning failed: {e}")

        if return_metrics:
            # Fast neural-only metrics for batch monitoring
            try:
                metrics = compute_metrics(topas_model, input_grid, target_grid, hrm_latents=hrm_latents)
                if metrics is None or not isinstance(metrics, dict):
                    if global_step % 100 == 0:
                        logging.warning(f"[Metrics] compute_metrics returned invalid type: {type(metrics)}")
                    return loss.item() if isinstance(loss, torch.Tensor) else None
                return loss.item(), metrics
            except Exception as e:
                if global_step % 100 == 0:
                    logging.warning(f"[Metrics] Computation failed at step {global_step}: {e}")
                    logging.debug(f"[Metrics] Full traceback:", exc_info=True)
                # Track failure rate
                if not hasattr(topas_model, '_metrics_fail_count'):
                    topas_model._metrics_fail_count = 0
                    topas_model._metrics_total_count = 0
                topas_model._metrics_fail_count += 1
                topas_model._metrics_total_count += 1
                if topas_model._metrics_total_count % 100 == 0:
                    fail_rate = topas_model._metrics_fail_count / topas_model._metrics_total_count
                    logging.info(f"[Metrics] Failure rate: {fail_rate:.1%} ({topas_model._metrics_fail_count}/{topas_model._metrics_total_count})")
                return loss.item() if isinstance(loss, torch.Tensor) else None
        else:
            return loss.item() if isinstance(loss, torch.Tensor) else None

    except Exception:
        logging.exception("Exception in train_step")
        return None


# === Lightweight curriculum heuristics ===
def _heuristic_difficulty(demo_in: torch.Tensor, demo_out: torch.Tensor) -> str:
    """Cheap proxy: size, palette, and rough difference."""
    try:
        H, W = demo_in.shape[-2], demo_in.shape[-1]
        size_score = int(H*W >= 16) + int(H*W >= 64) + int(H*W >= 196)
        palette = int(torch.unique(demo_in).numel())
        color_score = int(palette >= 3) + int(palette >= 5)
        diff = (demo_in.view(-1) != demo_out.view(-1)).float().mean().item()
        diff_score = int(diff > 0.4) + int(diff > 0.7)
        s = size_score + color_score + diff_score
        return "easy" if s <= 2 else ("medium" if s <= 4 else "hard")
    except Exception:
        return "medium"


def main():
    # Create output directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Temporary logger for early initialization (will be reconfigured after config load)
    logger = None
    print("ðŸš€ Starting Simplified HRM-TOPAS Training")
    print("=" * 60)

    # Force GPU-only mode - no CPU fallback for maximum performance
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training. CPU fallback disabled for performance.")
    device = torch.device("cuda")
    print(f"Device: {device}")
    
    # Apply CLI dream + dataset settings
    try:
        cli_args = parse_args()

        # Load YAML config if provided, with CLI overrides
        config_dict = None
        config_dict = None
        if hasattr(cli_args, 'config') and cli_args.config:
            config_dict = load_config_with_overrides(cli_args.config, cli_args)
            # Convert dict back to Namespace for compatibility
            from argparse import Namespace
            cli_args = Namespace(**config_dict)
            print(f"ðŸ“ Using merged config (YAML + CLI overrides)")
    except Exception as e:
        print(f"âš ï¸  Config loading failed: {e}, using defaults")
        cli_args = None
        config_dict = None
        config_dict = None


    # Set up logging with verbose mode from config
    verbose_mode = config_dict.get("logging_verbose", True) if config_dict else True
    logger = setup_logging(verbose=verbose_mode)

    # Set up logging with verbose mode from config
    verbose_mode = config_dict.get('logging_verbose', True) if config_dict else True
    logger = setup_logging(verbose=verbose_mode)
    # === DETERMINISTIC SEEDING FOR REPRODUCIBILITY ===
    def _seed_everything(seed: int):
        import os, random, numpy as np
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    seed = getattr(cli_args, "seed", 1337)
    _seed_everything(seed)
    logger.info(f"[Determinism] Seeded everything with seed={seed}")

    topas_model, hrm_model = create_models(device, cli_args, config_dict)

    # CLI args are already applied in create_models, just log the status
    if cli_args and getattr(cli_args, "enable_dream", False):
        dream_status = "enabled" if (hasattr(topas_model, 'dream') and topas_model.dream is not None) else "failed"
        print(f"âœ… DreamEngine {dream_status} (micro_ticks={getattr(cli_args, 'dream_micro_ticks', 1)})")

    # store cli_args in trainer scope for later reference
    trainer_cli_args = cli_args
    
    # Optional UKS load
    if cli_args and cli_args.uks_load_path:
        try:
            if hasattr(topas_model, "relmem") and topas_model.relmem is not None:
                topas_model.relmem.load_uks(cli_args.uks_load_path)
                print(f"[UKS] Loaded RelMem state from {cli_args.uks_load_path}")
        except Exception as e:
            print(f"[UKS] Could not load RelMem state: {e}")

    # Always use ARC-2 dataset (only dataset available)
    if ARC2Dataset is not None:
        print("📦 Using ARC-II dataset (arc_2_dataset directory)")
        dataset = ARC2Dataset(
            "arc_2_dataset/training",
            None,  # No solution file for directory structure
            device=str(device)
        )
    else:
        print("❌ ARC2Dataset not available")
        raise ImportError("ARC2Dataset is required but not available. Check arc2_dataset_loader.py import.")
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

    # === Self-play & critique stack ===
    self_play_buffer = SelfPlayBuffer(maxlen=cli_args.selfplay_buffer_size if cli_args else 200)  # used by dopamine rewards

    # Wire dopamine buffer to model for replay loss
    topas_model._dopamine_buffer = self_play_buffer
    logger.info("[Dopamine] Buffer wired to model for replay loss")

    counterexample_gen = CounterexampleGenerator(device=str(device))  # for nightmare queue
    star_bootstrapper = STaRBootstrapper(topas_model, device=str(device))  # trace generator/validator
    consistency_enforcer = ConsistencyEnforcer(device=str(device))         # optional consistency step
    
    # === Policy-guided search integration ===
    try:
        op_policy = OpPolicyNet().to(device)
        logger.info("[Policy] OpPolicyNet initialized for guided search")
    except Exception as e:
        logger.warning(f"[Policy] OpPolicyNet initialization failed: {e}")
        op_policy = None

    # === Euphoria Gate - Adaptive heavy compute controller ===
    gate = EuphoriaGate(
        base_interval=50,  # Heavy features every 50 steps when calm
        window=50,  # Stay active for 50 steps after trigger
        acc_uptick=0.03,  # 3% ACC jump triggers gate
        euphoria_thresh=3.0,  # R>3 triggers gate
        puct_min=40,  # Min PUCT budget when pressured
        puct_max=120,  # Max PUCT budget when active
        mem_cap_mb=7000  # VRAM threshold for throttling
    )
    logger.info("[EuphoriaGate] Adaptive compute controller initialized")

    # === Alpha-ARC X Neural-Guided Search 2.0 initialization ===
    replay_buffer = None
    if cli_args and PrioritizedReplay is not None:
        try:
            replay_buffer = PrioritizedReplay(capacity=cli_args.replay_cap)
            logger.info(f"[Alpha-ARC X] PrioritizedReplay initialized with capacity {cli_args.replay_cap}")
        except Exception as e:
            logger.warning(f"[Alpha-ARC X] PrioritizedReplay initialization failed: {e}")

    # Set search mode on model
    if cli_args and hasattr(topas_model, 'config'):
        try:
            topas_model.config.search_alg = cli_args.search_alg
            logger.info(f"[Alpha-ARC X] Search algorithm set to: {cli_args.search_alg}")
        except Exception as e:
            logger.warning(f"[Alpha-ARC X] Failed to set search algorithm: {e}")
    
    # Run Dream pretrain if requested
    if cli_args and cli_args.dream_pretrain_epochs > 0:
        dream_pretrain_loop(topas_model, dataset, cli_args, device, logger)
        # Load pretrained Dream/ETS
        if os.path.exists('checkpoints/dream_pretrain.pth') and hasattr(topas_model, 'dream') and topas_model.dream:
            try:
                topas_model.dream.load_state('checkpoints/dream_pretrain.pth')
                logger.info("[Main] Loaded pretrained Dream/ETS")
            except Exception as e:
                logger.warning(f"[Main] Failed to load dream pretrain: {e}")
    
    # Self-play buffer initialized above in critique stack

    # Unified optimizer (TOPAS includes policy/value if attached as submodules)
    # topas_model.parameters() already recursively includes policy_net and value_net
    all_params = list(topas_model.parameters())
    print(f"[Optimizer] total parameters: {sum(p.numel() for p in all_params):,}")
    optimizer = optim.AdamW(all_params, lr=2e-5, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda")  # Fixed FutureWarning

    # === DISTILLATION OPTIMIZER (SEPARATE) ===
    # Create separate optimizer for policy/value heads to avoid double-step issues
    distill_params = []
    if hasattr(topas_model, 'policy_net') and topas_model.policy_net is not None:
        distill_params.extend(topas_model.policy_net.parameters())
    if hasattr(topas_model, 'value_net') and topas_model.value_net is not None:
        distill_params.extend(topas_model.value_net.parameters())

    if distill_params:
        # Higher LR for distillation (teacher signals are strong, can learn faster)
        distill_optimizer = optim.AdamW(distill_params, lr=3e-4, weight_decay=1e-5)
        distill_scaler = torch.amp.GradScaler("cuda")
        param_count = sum(p.numel() for p in distill_params)
        logger.info(f"[Distill] ✅ Separate optimizer created for {param_count:,} params (LR=3e-4)")
    else:
        distill_optimizer = None
        distill_scaler = None
        logger.warning("[Distill] No policy/value nets found - distillation disabled")

    num_epochs = 150  # Extended run with stable hyperparams
    total_steps = len(dataset) * num_epochs
    print(f"Training: {num_epochs} epochs, {total_steps} total steps")
    
    # Time estimation
    estimated_time_hours = total_steps / (10 * 3600)  # Assuming ~10 it/s from smoke test
    print(f"â±ï¸  Estimated training time: {estimated_time_hours:.1f} hours ({estimated_time_hours*60:.0f} minutes)")

    print("\nðŸŽ¯ Starting training loop...")
    print(">>> TRAIN STARTED")
    global_step = 0
    best_em = 0.0
    best_acc = 0.0
    stable_breakthrough_steps = 0  # counts metric windows at/above threshold for curriculum trigger

    # === Euphoric Dopamine Tracking ===
    prev_acc = None  # For velocity calculation
    _acc_ema = _EMA(beta=0.9, init=0.5)  # Smooth accuracy for expectation baseline

    # === Curriculum state ===
    active_bucket = "easy"
    bucket_unlock_patience = 3   # consecutive high-EM ticks to unlock next
    bucket_streak = 0
    
    for epoch in range(num_epochs):
        # === RelMem bias scheduling ===
        if hasattr(topas_model.config, 'relmem_op_bias_w'):
            if epoch < 5:
                # keep minimal bias during warmup
                topas_model.config.relmem_op_bias_w = 0.2
            elif 5 <= epoch < 30:
                # ramp up toward stronger bias
                progress = (epoch - 5) / 25.0
                topas_model.config.relmem_op_bias_w = 0.2 + progress * (0.5 - 0.2)
            else:
                # decay bias slightly after stabilization
                topas_model.config.relmem_op_bias_w = 0.3
            print(f"[RelMem] Epoch {epoch}: bias_w={topas_model.config.relmem_op_bias_w:.3f}")
        
        print(f"\n📈 Epoch {epoch + 1}/{num_epochs}")
        epoch_losses = []
        epoch_metrics = {'exact_match': [], 'accuracy': [], 'mean_iou': [], 'exact_match_refined': [], 'batch_sizes': []}
        last_metrics = None  # Store metrics for RelMem feedback
        from tqdm import tqdm
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for batch_idx, batch in enumerate(progress):
            import time
            t0_step = time.time()  # Track step time for gate

            # --- Curriculum gating by heuristic difficulty ---
            try:
                demos, test_inputs, test_outputs, task_id = batch
                if demos and len(demos) > 0:
                    d_in, d_out = demos[0][0], demos[0][1]
                    difficulty = _heuristic_difficulty(d_in, d_out)
                    if active_bucket == "easy" and difficulty != "easy":
                        continue
                    if active_bucket == "medium" and difficulty == "hard":
                        continue
            except Exception:
                pass

            # === STAGNATION-GATED REPLAY ===
            # Track EMA of accuracy and enable replay only during plateau
            if not hasattr(topas_model, '_acc_ema_stag'):
                topas_model._acc_ema_stag = 0.0
                topas_model._last_best_acc = 0.0
                topas_model._last_best_step = 0

            # Update stagnation detection every step
            if hasattr(topas_model, '_current_metrics') and topas_model._current_metrics:
                current_acc = topas_model._current_metrics.get('accuracy', 0.0)
                topas_model._acc_ema_stag = 0.95 * topas_model._acc_ema_stag + 0.05 * current_acc

                if topas_model._acc_ema_stag > topas_model._last_best_acc + 0.002:
                    topas_model._last_best_acc = topas_model._acc_ema_stag
                    topas_model._last_best_step = global_step

            # Enable replay only when plateaued (no improvement for 200 steps)
            plateau = (global_step - topas_model._last_best_step) > 200
            topas_model._allow_replay = plateau

            # Compute metrics every 30 steps (balanced: enough for brain-graph, not too slow)
            compute_metrics_now = (global_step % 30 == 0)

            result = train_step(
                topas_model, hrm_model, batch,
                optimizer, scaler, device,
                return_metrics=compute_metrics_now,
                global_step=global_step
            )

            # === ALPHA-ARC: DISTILLATION TRAINING ===
            # Distill policy/value heads from TOPAS teacher signals
            if compute_metrics_now and isinstance(result, tuple):
                try:
                    policy_head = getattr(topas_model, "policy_net", None)
                    value_head = getattr(topas_model, "value_net", None)

                    if policy_head is not None or value_head is not None:
                        loss_scalar, metrics = result

                        # Retrieve teacher signals from last forward pass
                        cached_outputs = globals().get('last_outputs_for_dopamine', {})
                        outputs = cached_outputs.get('outputs', {})
                        input_grid = cached_outputs.get('input_grid', None)
                        target_grid = cached_outputs.get('target_grid', None)

                        if isinstance(outputs, dict) and input_grid is not None and target_grid is not None:
                            with torch.no_grad():
                                extras = outputs.get('extras', {}) or {}
                                teacher_logits = extras.get('dsl_op_logits', None)
                                # Extract WGO program logits for additional teacher signal
                                wgo_op_bias = extras.get('wgo_op_bias', {})
                                # Value target from accuracy
                                acc = float(metrics.get("accuracy", 0.0))

                            distill_loss = torch.tensor(0.0, device=device)

                            # Policy distillation: KL divergence on operation distribution
                            if policy_head is not None and teacher_logits is not None:
                                try:
                                    policy_head.train()

                                    # Forward through policy head (need minimal features)
                                    B, H, W = input_grid.shape
                                    rel_feat = torch.zeros(B, 64, device=device)
                                    size_oracle = torch.tensor([[H, W, H, W]], device=device, dtype=torch.float32)
                                    theme_priors = torch.zeros(B, 10, device=device)

                                    pred = policy_head(input_grid, rel_feat, size_oracle, theme_priors, program_ops=[])

                                    # === ENSEMBLE TEACHER: Bridge + WGO ===
                                    # Soft targets from bridge teacher (temperature scaling)
                                    T = 1.0
                                    p_bridge = torch.softmax(teacher_logits.detach() / T, dim=-1)

                                    # Add WGO program head as additional teacher if available
                                    if wgo_op_bias and len(wgo_op_bias) > 0:
                                        # Convert WGO op_bias dict to logit tensor (aligned with DSL_OPS)
                                        from models.dsl_registry import DSL_OPS
                                        wgo_logits = torch.zeros_like(teacher_logits)
                                        for op_name, bias_value in wgo_op_bias.items():
                                            if op_name in DSL_OPS:
                                                op_idx = DSL_OPS.index(op_name)
                                                if op_idx < wgo_logits.shape[-1]:
                                                    # Convert probability to logit (inverse sigmoid approximation)
                                                    wgo_logits[0, op_idx] = np.log(max(bias_value, 1e-6) / (1 - min(bias_value, 1-1e-6)))

                                        p_wgo = torch.softmax(wgo_logits.detach() / T, dim=-1)

                                        # Ensemble: 70% bridge + 30% WGO (bridge is main teacher, WGO adds structure)
                                        p_teacher = 0.7 * p_bridge + 0.3 * p_wgo
                                    else:
                                        p_teacher = p_bridge

                                    p_student = torch.log_softmax(pred.op_logits / T, dim=-1)

                                    # KL divergence
                                    policy_distill = torch.sum(-p_teacher * p_student, dim=-1).mean()
                                    distill_loss = distill_loss + policy_distill

                                except Exception as e:
                                    if global_step % 100 == 0:
                                        logger.debug(f"[Distill] Policy skip: {e}")

                            # Value distillation: Regress to pixel accuracy
                            if value_head is not None:
                                try:
                                    value_head.train()

                                    # Forward through value head
                                    B, H, W = input_grid.shape
                                    rel_feat = torch.zeros(B, 64, device=device)
                                    size_oracle = torch.tensor([[H, W, H, W]], device=device, dtype=torch.float32)
                                    theme_priors = torch.zeros(B, 10, device=device)

                                    v_pred = value_head(input_grid, rel_feat, size_oracle, theme_priors, program_ops=[], program_len=0)

                                    # Target value from accuracy
                                    target_v = torch.tensor([acc], device=device, dtype=v_pred.solvability.dtype)
                                    value_distill = torch.nn.functional.mse_loss(v_pred.solvability, target_v)
                                    distill_loss = distill_loss + value_distill

                                except Exception as e:
                                    if global_step % 100 == 0:
                                        logger.debug(f"[Distill] Value skip: {e}")

                            # Backward pass for distillation using SEPARATE optimizer
                            if distill_loss.item() > 0 and distill_loss.requires_grad and distill_optimizer is not None:
                                # Clear any stale gradients from distillation params
                                distill_optimizer.zero_grad()

                                # Backward with distill scaler
                                distill_scaler.scale(distill_loss).backward()  # No 0.1 weight - KL loss already small

                                # Clip gradients for stability
                                if policy_head is not None:
                                    torch.nn.utils.clip_grad_norm_(policy_head.parameters(), 1.0)
                                if value_head is not None:
                                    torch.nn.utils.clip_grad_norm_(value_head.parameters(), 1.0)

                                # Step separate distillation optimizer
                                distill_scaler.step(distill_optimizer)
                                distill_scaler.update()

                                if global_step % 100 == 0:
                                    logger.info(f"[Distill] ✅ step={global_step} loss={distill_loss.item():.4f} (separate optimizer)")

                except Exception as e:
                    if global_step % 500 == 0:
                        logger.warning(f"[Distill] Exception: {e}")

            # === ALPHA-ARC: SELF-PLAY MICRO-BATCH INJECTION ===
            # Periodically inject self-play samples as mini-batches for continuous learning
            if (global_step % 30 == 0) and getattr(topas_model, "_allow_replay", False):
                try:
                    sp_samples = self_play_buffer.sample_batch(n=2)  # Small micro-batch
                    if sp_samples:
                        for sp_item in sp_samples:
                            try:
                                # Decode from buffer (format: (inp_encoded, out_encoded, score))
                                if len(sp_item) >= 2:
                                    sp_in_enc, sp_out_enc = sp_item[0], sp_item[1]
                                    sp_in = _decode_grid(sp_in_enc).to(device).unsqueeze(0)  # [1, H, W]
                                    sp_out = _decode_grid(sp_out_enc).to(device).unsqueeze(0)  # [1, H, W]

                                    optimizer.zero_grad(set_to_none=True)

                                    # Forward pass with self-play sample
                                    with torch.amp.autocast('cuda', enabled=True):
                                        sp_outputs = topas_model.forward_pretraining(
                                            sp_in,
                                            target_shape=sp_out.shape[-2:],
                                            demos=None,
                                            replay_mode=True
                                        )

                                    # Compute CE loss
                                    if 'logits' in sp_outputs and sp_outputs['logits'] is not None:
                                        sp_logits = sp_outputs['logits']
                                        sp_target = sp_out.reshape(sp_out.size(0), -1).long()

                                        if sp_logits.size(1) == sp_target.size(1):
                                            sp_ce = torch.nn.functional.cross_entropy(
                                                sp_logits.reshape(-1, sp_logits.size(-1)),
                                                sp_target.reshape(-1)
                                            )

                                            # Small weight for micro-batch
                                            scaler.scale(0.2 * sp_ce).backward()
                                            torch.nn.utils.clip_grad_norm_(topas_model.parameters(), 1.0)
                                            scaler.step(optimizer)
                                            scaler.update()

                            except Exception as e:
                                if global_step % 100 == 0:
                                    logger.debug(f"[SelfPlay-Micro] Sample skip: {e}")

                        if global_step % 100 == 0:
                            logger.info(f"[SelfPlay-Micro] step={global_step} injected={len(sp_samples)} samples")

                except Exception as e:
                    if global_step % 500 == 0:
                        logger.warning(f"[SelfPlay-Micro] Exception: {e}")

            # === (III) REPLAY MICROSTEP with hard-mining and stagnation gating ===
            try:
                # Only fire when training plateaus (stagnation gate)
                if not hasattr(topas_model, '_replay_last_best_acc'):
                    topas_model._replay_last_best_acc = 0.0
                    topas_model._replay_last_best_step = 0

                # Check if we're in a plateau (no improvement for 200 steps)
                current_acc = topas_model._acc_ema_stag if hasattr(topas_model, '_acc_ema_stag') else 0.0
                if current_acc > topas_model._replay_last_best_acc + 0.002:
                    topas_model._replay_last_best_acc = current_acc
                    topas_model._replay_last_best_step = global_step

                plateau = (global_step - topas_model._replay_last_best_step) > 200

                # Fire every 20 steps when plateaued
                if plateau and (global_step % 20 == 0) and (self_play_buffer is not None) and len(self_play_buffer.buffer) >= 4:
                    # Hard-mining: sample highest-loss example from recent buffer
                    # (Prioritized replay already does this via sample_prioritized with novelty=True)
                    from train_parent import _encode_grid_tensor, _decode_grid

                    buffer_data = getattr(self_play_buffer, 'buffer', [])
                    if len(buffer_data) >= 1:
                        # Score by reward (higher = harder/more important)
                        scored = [(i, reward) for i, (inp_enc, out_enc, reward) in enumerate(buffer_data)
                                  if isinstance(reward, (int, float))]
                        if scored:
                            scored.sort(key=lambda x: x[1], reverse=True)  # Hardest first
                            hard_idx = scored[0][0]
                            inp_enc, out_enc, _ = buffer_data[hard_idx]

                            replay_inp = _decode_grid(inp_enc).to(device).unsqueeze(0)  # [1, H, W]
                            replay_tgt = _decode_grid(out_enc).to(device).unsqueeze(0)  # [1, H, W]

                            optimizer.zero_grad(set_to_none=True)

                            # Forward with replay_mode
                            with torch.amp.autocast('cuda', enabled=True):
                                micro = topas_model.forward_pretraining(
                                    replay_inp, hrm_latents=None,
                                    target_shape=replay_tgt.shape[-2:], demos=None,
                                    global_step=global_step, replay_mode=True
                                )

                            micro_loss = micro.get("losses", {}).get("dopamine_replay", None)
                            if micro_loss is not None and micro_loss.requires_grad:
                                # Smaller LR "nudge" (0.5x)
                                scaler.scale(0.5 * micro_loss).backward()
                                torch.nn.utils.clip_grad_norm_(topas_model.parameters(), 1.0)
                                scaler.step(optimizer)
                                scaler.update()
                                logger.info(f"[MicroReplay] step={global_step} hard_loss={float(micro_loss.detach().cpu()):.4f} plateau={plateau}")
            except Exception as e:
                logger.warning(f"[MicroReplay] skipped: {e}")

            # === Dopamine capture & planner priors ===
            if compute_metrics_now and isinstance(result, tuple):
                try:
                    loss, metrics = result
                    em_val = float(metrics.get("exact_match", 0.0))
                    rolling_em.append(em_val)

                    # Store metrics for RelMem feedback on next step
                    last_metrics = {
                        'exact_match': float(metrics.get("exact_match", 0.0)),
                        'accuracy': float(metrics.get("accuracy", 0.0)),
                        'mean_iou': float(metrics.get("mean_iou", 0.0))
                    }

                    # Store metrics as model attributes for RelMem access
                    topas_model._current_metrics = last_metrics

                    # === BRAIN-GRAPH LEARNING (Ethical: learn from successful examples only) ===
                    if (hasattr(topas_model, 'brain_graph') and
                        topas_model.brain_graph is not None and
                        em_val >= 0.5):  # Only learn from highly successful examples

                        try:
                            # Extract puzzle attributes (observable properties only)
                            puzzle_attrs = _extract_puzzle_attributes(input_grid, target_grid)

                            # Extract DSL ops from RelMem op_bias (what operations scored highest)
                            dsl_ops_used = []
                            if hasattr(topas_model, 'relmem') and topas_model.relmem is not None:
                                from models.dsl_registry import DSL_OPS
                                op_bias = topas_model.relmem.get_op_bias(dsl_ops=DSL_OPS)
                                if op_bias:
                                    # Top-3 operations by bias score
                                    top_ops = sorted(op_bias.items(), key=lambda x: x[1], reverse=True)[:3]
                                    dsl_ops_used = [op for op, score in top_ops if score > 0.1]

                            # Fallback if no ops available
                            if not dsl_ops_used:
                                dsl_ops_used = ["neural_pattern"]  # Placeholder for pure neural success

                            # Learn from this successful pattern (Hebbian: "fire together, wire together")
                            topas_model.brain_graph.observe_success(
                                puzzle_attrs=puzzle_attrs,
                                dsl_ops_used=dsl_ops_used,
                                success=True,
                                task_id=str(task_id) if task_id is not None else None
                            )

                            # Log brain-graph growth every 100 steps
                            if global_step % 100 == 0:
                                node_count = len(topas_model.brain_graph.nodes)
                                concept_count = sum(1 for n in topas_model.brain_graph.nodes.values() if n.kind == "concept")
                                print(f"[BrainGraph] Learned from EM={em_val:.0%} | Nodes: {node_count} (concepts={concept_count})")

                        except Exception as e:
                            if global_step % 500 == 0:
                                logger.warning(f"[BrainGraph] Learning failed: {e}")

                    # === UPDATE EUPHORIC DOPAMINE TRACKING ===
                    current_acc = float(metrics.get("accuracy", 0.0))
                    ema_acc_value = _acc_ema.update(current_acc)
                    # Update model's _ema_acc for adaptive lambda in replay
                    topas_model._ema_acc = torch.tensor(ema_acc_value, device=device)
                    # prev_acc gets updated at end of metrics block

                    # === CURIOSITY-DRIVEN DREAM TRIGGERING ===
                    # Trigger extra dream cycle when model is confused/uncertain
                    if hasattr(topas_model, 'dream') and topas_model.dream is not None:
                        try:
                            # Calculate uncertainty from low accuracy + low exact match
                            accuracy = float(metrics.get("accuracy", 0.0))
                            uncertainty = 1.0 - (accuracy + em_val) / 2.0  # High when both are low

                            # Calculate novelty from task features if available
                            novelty_score = 0.0
                            if hasattr(topas_model, 'gccrf') and topas_model.gccrf is not None:
                                try:
                                    # Use GCCRF to estimate novelty
                                    demos_tensor = demos[0][0].to(device) if demos else None
                                    if demos_tensor is not None:
                                        novelty_score = float(topas_model.gccrf.compute_novelty(demos_tensor))
                                except Exception:
                                    pass

                            # Check if dream should be triggered
                            should_dream = topas_model.dream.should_trigger_dream(
                                novelty_score=novelty_score,
                                uncertainty=uncertainty
                            )

                            if should_dream and global_step % 50 == 0:  # Rate limit to every 50 steps
                                logger.info(f"[CuriosityDream] Triggering dream cycle (novelty={novelty_score:.3f}, uncertainty={uncertainty:.3f})")
                                try:
                                    # Pass brain tokens (768-dim) instead of concept_proto (256-dim)
                                    # Brain tokens come from model's last forward pass
                                    brain_tokens = None
                                    if isinstance(result, tuple) and len(result) >= 2:
                                        _, result_metrics = result
                                        if isinstance(result_metrics, dict):
                                            # Try to get brain/latent from outputs
                                            brain_tokens = result_metrics.get('latent') or result_metrics.get('brain')

                                    if brain_tokens is not None and torch.is_tensor(brain_tokens):
                                        # Ensure correct shape for dream.train_step()
                                        if brain_tokens.dim() == 2:  # [B, D]
                                            # GRADIENT FIX: Remove detach() to allow dream to influence brain representations
                                            dream_loss = topas_model.dream.train_step(brain_tokens)
                                            logger.info(f"[CuriosityDream] Dream loss: {dream_loss:.4f}")

                                            # CRITICAL FIX: Actually backpropagate dream loss (was orphaned before)
                                            if dream_loss.requires_grad and torch.isfinite(dream_loss):
                                                optimizer.zero_grad(set_to_none=True)
                                                scaler.scale(0.1 * dream_loss).backward()  # Small weight (0.1)
                                                torch.nn.utils.clip_grad_norm_(topas_model.parameters(), max_norm=1.0)
                                                scaler.step(optimizer)
                                                scaler.update()
                                                logger.info(f"[CuriosityDream] Backpropagated dream loss (λ=0.1)")
                                        else:
                                            logger.debug(f"[CuriosityDream] Brain tokens wrong shape: {brain_tokens.shape}")
                                    else:
                                        logger.debug(f"[CuriosityDream] No brain tokens available, skipping")
                                except Exception as e:
                                    logger.warning(f"[CuriosityDream] Dream step failed: {e}")
                        except Exception as e:
                            logger.debug(f"[CuriosityDream] Curiosity check failed: {e}")

                    # Curriculum unlock: promote buckets when stable high EM
                    try:
                        if em_val >= 0.90:
                            bucket_streak += 1
                        else:
                            bucket_streak = 0
                        if bucket_streak >= bucket_unlock_patience:
                            if active_bucket == "easy":
                                active_bucket = "medium"
                                logger.info("[Curriculum] Unlocked MEDIUM tasks")
                            elif active_bucket == "medium":
                                active_bucket = "hard"
                                logger.info("[Curriculum] Unlocked HARD tasks")
                            bucket_streak = 0
                    except Exception:
                        pass
                    
                    # === EUPHORIC DOPAMINE CALCULATION (ALWAYS, not just on breakthrough) ===
                    # Reconstruct current task from batch
                    demos, test_inputs, test_outputs, task_id = batch
                    if demos and len(demos) > 0:
                        grid_in = demos[0][0].to(device)
                        grid_out = demos[0][1].to(device)
                        if not torch.is_tensor(grid_in):
                            grid_in = torch.tensor(grid_in, device=device)
                        if not torch.is_tensor(grid_out):
                            grid_out = torch.tensor(grid_out, device=device)
                        task = Task(input=grid_in, output=grid_out, constraints={}, metadata={})

                        # Compute euphoric dopamine for EVERY metrics step
                        iou_val = float(metrics.get("mean_iou", 0.0))
                        R_euphoric, comps_euphoric = _dopamine_score_euphoric(
                            em=em_val,
                            acc=current_acc,
                            iou=iou_val,
                            prev_acc=prev_acc,
                            ema_acc=ema_acc_value
                        )

                        # ALWAYS store dopamine value (even if we don't capture)
                        topas_model._last_dopamine_value = float(R_euphoric)

                        # Update gate with current metrics and step time
                        step_dt = time.time() - t0_step
                        gate.update(global_step, current_acc, R_euphoric, step_dt)

                        # Pass gate state to model so forward_pretraining can use adaptive top-K
                        topas_model._gate_active = gate.active(global_step)

                        # Log gate status periodically
                        if global_step % 100 == 0:
                            gate_status = "ACTIVE" if gate.active(global_step) else "calm"
                            vram_mb = torch.cuda.memory_allocated()/(1024**2) if torch.cuda.is_available() else 0
                            logger.info(f"[EuphoriaGate] {gate_status}, VRAM={vram_mb:.0f}MB, step_time={step_dt:.3f}s")

                        # Log emotional state periodically
                        if global_step % 100 == 0:
                            if R_euphoric > 10.0:
                                logger.info(f"[Euphoria] ðŸŽ† HIGH! R={R_euphoric:.2f}, acc={current_acc:.1%}, em={em_val:.1%}")
                            elif R_euphoric < -1.0:
                                logger.info(f"[Pain] ðŸ˜° LOW! R={R_euphoric:.2f}, acc={current_acc:.1%}, regret={comps_euphoric.get('regret',0):.2f}")

                        # === NEW MULTI-THRESHOLD TRIGGER LOGIC ===
                        # Trigger dopamine capture on: high ACC, high IoU, ANY EM, or strong advantage
                        high_acc_trigger = current_acc >= 0.85
                        high_iou_trigger = iou_val >= 0.8
                        any_em_trigger = em_val >= 0.20
                        euphoria_trigger = R_euphoric > 3.0  # Direct euphoria threshold

                        should_capture = (high_acc_trigger or high_iou_trigger or any_em_trigger or euphoria_trigger) and task is not None

                    else:
                        task = None
                        should_capture = False
                        R_euphoric = 0.0
                        topas_model._last_dopamine_value = 0.0

                    # Dopamine capture re-enabled with simplified version (no STaR/PUCT loops)
                    # should_capture determined by triggers above

                    if should_capture:
                        logger.info(f"[Dopamine] R={R_euphoric:.2f} at step={global_step}")

                        # Initialize variables
                        good_traces = []
                        programs = []

                        # Generate programs from available sources
                        # (A) Try PUCT stepwise (cheap & bounded)
                        try:
                            if puct_search is not None and getattr(cli_args, "enable_puct_in_dopamine", True):
                                ops = _puct_plan_stepwise(topas_model, demos, grid_in, grid_out, cli_args, device)
                                if ops:
                                    programs.append(_as_program(ops))
                        except Exception:
                            pass

                        # (B) Try to harvest from STaR if euphoria gate is hot
                        try:
                            if star_bootstrapper is not None and gate.active(global_step):
                                star_task = _as_star_task(task)
                                # Low budget to avoid loops
                                tr = _sc_run_star(star_bootstrapper, star_task, {}, n=2)
                                good = [t for t in tr if getattr(t, "is_valid", False) or getattr(t, "operations", None)]
                                for t in good:
                                    programs.append(_as_program(getattr(t, "operations", [])))
                        except Exception:
                            pass

                        # (C) If TOPAS policy produced a program in this step, take it
                        try:
                            extras = outputs.get("extras", {}) if isinstance(outputs, dict) else {}
                            ri = extras.get("rule_info", {})
                            if isinstance(ri, dict) and ri.get("program"):
                                programs.append(_as_program(ri["program"]))
                        except Exception:
                            pass

                        star_task = _as_star_task(task)

                        # Store dopamine value for tagging
                        topas_model._last_dopamine_value = float(R_euphoric)

                        # Just store the experience in buffer
                        if self_play_buffer is not None:
                            try:
                                dopamine_reward(star_task, self_play_buffer, logger, global_step,
                                              score=float(R_euphoric), components=comps_euphoric)
                            except Exception as e:
                                logger.warning(f"[Dopamine] Storage failed: {e}")

                        # END OF ULTRA-MINIMAL DOPAMINE CAPTURE

                    # Near-miss repair - RE-ENABLED with correct API and safeguards
                    if should_capture and near_miss_repair is not None:
                        # Alpha-ARC X: Near-miss repair
                        if cli_args:
                            try:
                                pred_grid = outputs.get("grid", [None])[0] if isinstance(outputs, dict) else None
                                if pred_grid is not None and torch.is_tensor(pred_grid):
                                    # SAFEGUARD 1: Only repair if VERY close (< 5% error)
                                    max_error_pct = 0.05
                                    hamming_dist = _hamming(pred_grid, grid_out)
                                    error_rate = hamming_dist / max(1, pred_grid.numel())

                                    if error_rate <= max_error_pct:
                                        # SAFEGUARD 2: Max 3 repair attempts
                                        # SAFEGUARD 3: Timeout protection (100ms)
                                        import time
                                        start_time = time.time()
                                        timeout_s = 0.1

                                        # Prepare DSL shim + ops for near-miss (correct API)
                                        dsl_shim = getattr(topas_model, "dsl", None)
                                        if dsl_shim is None:
                                            try:
                                                from models.topas_arc_60M import _DSLShim
                                                dsl_shim = _DSLShim(topas_model)
                                            except Exception:
                                                dsl_shim = None

                                        dsl_ops = getattr(dsl_shim, "ops", None) if dsl_shim else None
                                        if dsl_ops is None:
                                            try:
                                                from models.dsl_registry import DSL_OPS
                                                dsl_ops = list(DSL_OPS)
                                            except Exception:
                                                dsl_ops = None

                                        if dsl_shim and dsl_ops:
                                            try:
                                                # Run bounded near-miss repair (correct signature)
                                                repaired_grid, repair_ops, improvement, _ = near_miss_repair(
                                                    pred_grid, grid_out, dsl_ops, dsl_shim, max_repairs=3
                                                )
                                                elapsed = time.time() - start_time

                                                if elapsed > timeout_s:
                                                    logger.warning(f"[Near-Miss] Timeout ({elapsed:.3f}s > {timeout_s}s), discarding")
                                                    repair_ops = None

                                                if repair_ops:
                                                    # Treat the repair as a program
                                                    programs.append(_as_program(repair_ops))

                                                    # Feed to wormhole as weak template
                                                    try:
                                                        if hasattr(topas_model, "dream") and hasattr(topas_model.dream, "wormhole"):
                                                            topas_model.dream.wormhole.consume_repair_ops(repair_ops, score=0.6)
                                                    except Exception:
                                                        pass

                                                    # Add to replay buffer
                                                    try:
                                                        from trainers.replay import ProgramTrace
                                                        repair_trace = ProgramTrace(
                                                            task_id=f"near_miss_{task_id}",
                                                            program=repair_ops,
                                                            params=[{} for _ in repair_ops],
                                                            score=0.95,
                                                            novelty=0.8,
                                                            depth=len(repair_ops),
                                                            source='near_miss_repair',
                                                            input_grid=grid_in.detach().cpu(),
                                                            output_grid=pred_grid.detach().cpu(),
                                                            target_grid=grid_out.detach().cpu()
                                                        )
                                                        if replay_buffer is not None:
                                                            replay_buffer.add(repair_trace)
                                                    except Exception:
                                                        pass

                                                    if global_step % 100 == 0:
                                                        logger.info(f"[Near-Miss] Repaired with {len(repair_ops)} ops (improvement={improvement:.1%}, error={error_rate:.1%})")
                                            except Exception as repair_err:
                                                logger.warning(f"[Near-Miss] Repair raised: {repair_err}")
                            except Exception as e:
                                if global_step % 200 == 0:
                                    logger.warning(f"[Near-Miss] Failed: {e}")
                        dream_stats = None
                        try:
                            if hasattr(topas_model, "dream") and topas_model.dream is not None:
                                dream_stats = getattr(topas_model.dream, "last_stats", None)
                        except Exception:
                            pass
                        ent_red = _extract_entropy_reduction(dream_stats)
                        mdl_gain = 0.0
                        try:
                            if hasattr(topas_model, "dream") and hasattr(topas_model.dream, "wormhole") and programs:
                                mined = topas_model.dream.wormhole.mine_from_programs(programs, top_k=3)
                                mdl_gain = _extract_mdl_gain(mined)
                                # Refresh TTL for successful programs
                                topas_model.dream.wormhole.refresh_ttl_from_programs(programs, bonus_ttl=3)
                        except Exception:
                            pass
                        # Use the R_euphoric already calculated above
                        R = R_euphoric
                        comps = comps_euphoric

                        ema = _dopamine_ema.update(R)
                        advantage = R - ema
                        global _last_dopamine_step
                        refractory = (global_step - _last_dopamine_step) < 10

                        # Accept is already determined by should_capture above
                        accept = True  # We're inside the should_capture block

                        # Use absolute reward for perfect solves, advantage for improvements
                        if em_val >= 0.999:
                            reward_score = float(R)  # Absolute reward for perfection
                            logger.info(f"[Dopamine] ðŸŽ† NUT BUST! ðŸŽ† R={R:.3f}, ema={ema:.3f}, advantage={advantage:.3f}, "
                                      f"surprise={comps.get('surprise',1.0):.2f}x, velocity={comps.get('velocity',1.0):.2f}x")
                        elif R > 5.0:
                            reward_score = float(R)  # High euphoria uses absolute
                            logger.info(f"[Dopamine] ðŸ˜Š EUPHORIA! R={R:.3f}, ema={ema:.3f}, base={comps.get('base',0):.2f}, "
                                      f"surprise={comps.get('surprise',1.0):.2f}x, em_factor={comps.get('em_factor',1.0):.1f}x")
                        elif R < 0:
                            reward_score = float(R)  # Negative = pain
                            logger.info(f"[Dopamine] ðŸ˜° PAIN! R={R:.3f}, ema={ema:.3f}, regret={comps.get('regret',0):.2f}")
                        else:
                            reward_score = float(advantage)  # Relative advantage for normal performance

                        # Store dopamine value on model for pattern binding
                        topas_model._last_dopamine_value = float(R)

                        if accept:
                            dopamine_reward(star_task, self_play_buffer, logger, global_step,
                                            score=reward_score, components=comps)

                            # DOPAMINE BOOST: Reinforce RelMem concepts for successful pattern
                            if hasattr(topas_model, 'relmem') and reward_score > 0:
                                try:
                                    # Get most recent concept (just bound from this success)
                                    active_cids = topas_model.relmem.concept_used.nonzero().flatten()
                                    if active_cids.numel() > 0:
                                        # Boost the most recently activated concept
                                        recent_cid = active_cids[-1].item()
                                        # Strengthen concept by increasing its activation count
                                        if hasattr(topas_model.relmem, 'concept_activation_count'):
                                            if not hasattr(topas_model.relmem, 'concept_activation_count'):
                                                topas_model.relmem.concept_activation_count = torch.zeros_like(topas_model.relmem.concept_used, dtype=torch.float32)
                                            topas_model.relmem.concept_activation_count[recent_cid] += reward_score * 10.0
                                        logger.info(f"[Dopamine] Boosted concept {recent_cid} by {reward_score*10.0:.1f}")
                                except Exception as e:
                                    logger.debug(f"[Dopamine] Concept reinforcement failed: {e}")

                            # Alpha-ARC X: Add successful programs to replay buffer
                            if replay_buffer is not None:
                                try:
                                    for t in good_traces:
                                        if hasattr(t, "operations") and t.operations:
                                            safe_ops = _stringify_ops(t.operations)
                                            if safe_ops:
                                                priority = float(advantage) + 0.1  # Ensure positive priority
                                                replay_buffer.add(safe_ops, priority)
                                    if programs:
                                        for prog in programs:
                                            if prog:
                                                safe_prog = _stringify_ops(prog)
                                                if safe_prog:
                                                    priority = float(advantage) + 0.1
                                                    replay_buffer.add(safe_prog, priority)
                                    logger.info(f"[Alpha-ARC X] Added {len(good_traces) + len(programs)} programs to replay buffer")
                                except Exception as e:
                                    logger.warning(f"[Alpha-ARC X] Replay buffer addition failed: {e}")

                            try:
                                if hasattr(topas_model, "dream") and hasattr(topas_model.dream, "wormhole") and programs:
                                    topas_model.dream.wormhole.refresh_ttl_from_programs(programs, bonus_ttl=3)
                            except Exception:
                                pass
                            try:
                                for t in good_traces:
                                    if hasattr(t, "operations") and t.operations:
                                        # Use _stringify_ops to ensure all operations are hashable strings
                                        safe_ops = _stringify_ops(t.operations)
                                        if safe_ops:
                                            for sop in safe_ops:
                                                op_success_count.update({sop: max(1, int(10 * max(0.0, advantage)))})
                            except Exception as op_err:
                                logger.warning(f"[Dopamine] advantage-based op_count update failed: {op_err}")
                            _last_dopamine_step = global_step
                        else:
                            if global_step % 100 == 0:  # Log rejections periodically
                                logger.info(f"[Dopamine] Skipped (R={R:+.3f}, adv={advantage:+.3f}, ema={ema:+.3f}, refractory={refractory}, threshold={threshold})")
                        # 5) Counterexamples â†’ nightmare queue
                        cex = counterexample_gen.generate_from_failure(task, topas_model, n_counterexamples=5)
                        if cex:
                            recent_failures.extend(cex)
                        # 6) Optional: consistency enforcement across valid traces
                        if len(good_traces) > 1:
                            try:
                                consistency_enforcer.enforce_consistency(good_traces, task)
                            except Exception:
                                pass
                        # 7) Stable-breakthrough counter (for curriculum)
                        stable_breakthrough_steps += 1
                    else:
                        stable_breakthrough_steps = max(0, stable_breakthrough_steps - 1)
                except Exception as e:
                    logger.warning(f"[Dopamine] capture pipeline skipped: {e}")

            # --- Internal monologue (mind-voice) control-plane ---
            try:
                if cli_args and cli_args.monologue_interval > 0 and (global_step % cli_args.monologue_interval == 0):
                    if 'task' not in locals() or task is None:
                        # Build a minimal Task from current batch if not present
                        demos, test_inputs, test_outputs, task_id = batch
                        if demos and len(demos) > 0:
                            grid_in = demos[0][0].to(device)
                            grid_out = demos[0][1].to(device)
                            if not torch.is_tensor(grid_in): 
                                grid_in = torch.tensor(grid_in, device=device)
                            if not torch.is_tensor(grid_out):
                                grid_out = torch.tensor(grid_out, device=device)
                            task = Task(input=grid_in, output=grid_out, constraints={}, metadata={})
                    if task is not None:
                        # Half of traces guided by policy-enhanced op_bias
                        planner_bias = build_policy_guided_bias(grid_in, grid_out, op_policy, device, temp=0.7)
                        star_task = _as_star_task(task)
                        traces = star_bootstrapper.generate_diverse_traces(star_task, n_traces=max(6, cli_args.monologue_min_traces), planner_op_bias=planner_bias)
                        vals = star_bootstrapper.verify_traces(traces, star_task)
                        valid_traces = [t for t, v in zip(traces, vals) if v.is_valid or v.similarity_score >= 0.90]
                        if len(valid_traces) >= 2:
                            c_res = consistency_enforcer.enforce_consistency(valid_traces, star_task)
                            monolog_score = c_res['metrics'].overall_consistency if c_res.get('metrics') else 0.0
                            # Use monologue score to steer schedule:
                            target = float(getattr(cli_args, "monologue_consistency_target", 0.85))
                            if monolog_score >= target:
                                # Confidence â†‘ : gently ramp RelMem op-bias and reward self-play
                                if hasattr(topas_model.config, 'relmem_op_bias_w'):
                                    topas_model.config.relmem_op_bias_w = min(
                                        getattr(topas_model.config, 'relmem_op_bias_w', 0.2) + 0.02,
                                        getattr(topas_model.config, '_bias_max', 0.5)
                                    )
                                # Nudge self-play weight slightly
                                if hasattr(cli_args, "selfplay_weight"):
                                    cli_args.selfplay_weight = float(cli_args.selfplay_weight + cli_args.monologue_selfplay_bonus)
                                
                                # Enhanced: Dopamine reward for strong consistency scores
                                if monolog_score >= 0.90:
                                    try:
                                        # Calculate consistency-based dopamine score
                                        consistency_reward = min(1.0, monolog_score * 1.2)  # Scale and cap at 1.0
                                        good_traces = [t for t, v in zip(traces, vals) if v.is_valid]
                                        prog_len = _extract_program_len([getattr(t, "operations", []) for t in good_traces])
                                        
                                        # Use enhanced scoring with consistency bonus
                                        R, comps = _dopamine_score(
                                            em=consistency_reward,
                                            acc=consistency_reward,
                                            iou=0.8,  # Reasonable default for consistency
                                            program_len=prog_len,
                                            entropy_red=0.1,
                                            mdl_gain=0.1,
                                            novelty=0.5,
                                            Lmax=12
                                        )
                                        
                                        ema = _dopamine_ema.update(R)
                                        advantage = R - ema
                                        
                                        # More lenient threshold for consistency rewards
                                        if advantage >= 0.10:
                                            logger.info(f"[Monologue] Consistency dopamine: score={monolog_score:.3f}, R={R:.3f}, adv={advantage:.3f}")
                                            dopamine_reward(star_task, self_play_buffer, logger, global_step,
                                                            score=float(advantage), components=comps)
                                            
                                            # Update operation success counts from consistent traces
                                            for t in good_traces:
                                                if hasattr(t, "operations") and t.operations:
                                                    # Use _stringify_ops to ensure all operations are hashable strings
                                                    safe_ops = _stringify_ops(t.operations)
                                                    if safe_ops:
                                                        for sop in safe_ops:
                                                            op_success_count.update({sop: max(1, int(5 * max(0.0, advantage)))})
                                    except Exception as e:
                                        logger.debug(f"[Monologue] Consistency dopamine failed: {e}")
                            else:
                                # Reasoning shaky â†’ increase nightmare pressure & shorten interval
                                recent = getattr(cli_args, "nightmare_min_interval", 200)
                                cli_args.nightmare_min_interval = max(100, int(0.75 * recent))
                                # Queue a few counterexamples immediately
                                try:
                                    cex = counterexample_gen.generate_from_failure(task, topas_model, n_counterexamples=4)
                                    if cex:
                                        recent_failures.extend(cex)
                                except Exception:
                                    pass
                            logger.info(f"[Monologue] consistency={monolog_score:.3f}, relmem_bias_w={getattr(topas_model.config,'relmem_op_bias_w',None)}")
            except Exception as e:
                logger.debug(f"[Monologue] skipped: {e}")

            # === Curriculum: escalate difficulty when breakthroughs persist ===
            if stable_breakthrough_steps >= 100:
                try:
                    # Use the last seen task to mine deep programs
                    demos, test_inputs, test_outputs, task_id = batch
                    if demos and len(demos) > 0:
                        grid_in = demos[0][0]
                        grid_out = demos[0][1]
                        deep = mine_deep_programs({"test": {"input": grid_in, "output": grid_out}}, max_depth=10)
                        # Only keep exact matches to avoid incorrect targets
                        exact_deep = [dp for dp in deep if dp.get("exact_match")]
                        for _dp in exact_deep:
                            # Store canonical encoded samples
                            enc_inp = _encode_grid_tensor(grid_in)
                            enc_out = _encode_grid_tensor(grid_out)
                            self_play_buffer.buffer.append((enc_inp, enc_out))
                        logger.info(f"[Curriculum] Injected {len(exact_deep)} deep-program exemplars")
                except Exception as e:
                    logger.warning(f"[Curriculum] deep mining failed: {e}")
                finally:
                    stable_breakthrough_steps = 0

            # === Adaptive Nightmare cycle ===
            if len(rolling_em) >= 50 and cli_args:
                fail_rate = 1.0 - (sum(rolling_em) / len(rolling_em))  # higher â†’ worse
                min_iv = int(getattr(cli_args, "nightmare_min_interval", 200))
                max_iv = int(getattr(cli_args, "nightmare_max_interval", 1000))
                # Map fail_rateâˆˆ[0,1] â†’ interval [max_iv, min_iv]
                interval = int(max_iv - (max_iv - min_iv) * max(0.0, min(1.0, fail_rate)))
                if interval < min_iv: interval = min_iv
                if global_step % interval == 0 and recent_failures:
                    nightmare_prune(topas_model, recent_failures, optimizer, scaler, device, logger,
                                    global_step, alpha=float(getattr(cli_args, "nightmare_alpha", 0.08)))
            
            # === Self-play training integration (existing) ===
            sp_loss_contribution = 0.0
            if cli_args and cli_args.selfplay_enable and self_play_buffer and global_step % cli_args.selfplay_interval == 0:
                try:
                    # Generate new puzzles from current batch
                    demos, test_inputs, test_outputs, task_id = batch
                    current_demos = [(test_inputs, test_outputs)] if test_inputs is not None and test_outputs is not None else []
                    
                    if current_demos:
                        new_puzzles = self_play_buffer.generate_batch(
                            current_demos, 
                            getattr(topas_model, 'wormhole', None), 
                            top_k=cli_args.selfplay_topk
                        )
                        if new_puzzles:
                            print(f"[SelfPlay] Generated {len(new_puzzles)} puzzles at step={global_step}")
                        else:
                            print(f"[SelfPlay] No puzzles generated at step={global_step} â†’ trying Dream motifs")
                            if hasattr(topas_model, "dream") and hasattr(topas_model, "painter"):
                                try:
                                    # Get last Dream features
                                    dream_feat = getattr(topas_model.dream, "last_features", None)
                                    if dream_feat is not None:
                                        grid, logits, size = topas_model.painter(dream_feat)
                                        enc_in = _encode_grid_tensor(grid)
                                        # Cheap target: identity reconstruction
                                        enc_out = _encode_grid_tensor(grid.clone())
                                        self_play_buffer.buffer.append((enc_in, enc_out, 0.1))
                                        print(f"[Dreamâ†’SelfPlay] Injected 1 painter-motif puzzle (buffer={len(self_play_buffer.buffer)})")
                                except Exception as e:
                                    print(f"[Dreamâ†’SelfPlay] Painter fallback failed: {e}")
                except Exception as e:
                    import traceback
                    logging.getLogger(__name__).exception("[SelfPlay] failure: %s", e)
                
                # Sample and compute self-play loss with importance replay
                def extract_score(sample):
                    """
                    Extract score from sample tuple, handling both (input, output) and (input, output, score) formats.
                    Dreamâ†’Painter samples carry a default score of 0.1, but we give them a small novelty bonus so they don't get drowned out.
                    """
                    score = 0.0
                    if len(sample) >= 3:
                        raw = sample[2]
                        score = float(raw) if hasattr(raw, "__float__") else 0.0
                    # Detect Dream-derived motifs (tiny 0.1 baseline score) and up-weight them
                    if abs(score - 0.1) < 1e-6:
                        score += 0.2  # novelty bonus
                    return score
                
                # Sample larger batch for stochastic importance weighting
                sp_candidates = self_play_buffer.sample_batch(32)
                if sp_candidates:
                    # Extract scores and compute stochastic importance weights
                    sp_candidates_with_scores = [(s, extract_score(s)) for s in sp_candidates]
                    scores = np.array([score for _, score in sp_candidates_with_scores])
                    
                    # Use stochastic importance sampling with softmax weighting
                    if np.sum(scores) > 0:
                        # Apply softmax with numerical stability
                        exp_scores = np.exp(scores - np.max(scores))
                        probabilities = exp_scores / np.sum(exp_scores)
                        # Sample up to 4 puzzles using importance weights (boundary check)
                        sample_size = min(4, len(sp_candidates))
                        selected_indices = np.random.choice(len(sp_candidates), size=sample_size, replace=False, p=probabilities)
                    else:
                        # Fallback to uniform random sampling if no scores available (boundary check)
                        sample_size = min(4, len(sp_candidates))
                        selected_indices = np.random.choice(len(sp_candidates), size=sample_size, replace=False)
                    
                    sp_samples = [sp_candidates_with_scores[i][0] for i in selected_indices]
                    
                    print(f"[SelfPlay] Training on {sample_size} importance-weighted puzzles from {len(sp_candidates)} candidates")
                    for sp_sample in sp_samples:
                        # Handle both tuple formats: (input, output) and (input, output, score)
                        sp_input = sp_sample[0]
                        sp_target = sp_sample[1]
                        try:
                            # Decode if samples are encoded tuples
                            if not torch.is_tensor(sp_input):
                                sp_input = _decode_grid(sp_input)
                            if not torch.is_tensor(sp_target):
                                sp_target = _decode_grid(sp_target)
                            sp_output = topas_model.forward_pretraining(sp_input.unsqueeze(0),
                                                                        target_shape=sp_target.shape[-2:])
                            if 'logits' in sp_output:
                                sp_loss = F.cross_entropy(sp_output['logits'].view(-1, 10), sp_target.view(-1).long().clamp(0, 9))
                                sp_loss_contribution += sp_loss * cli_args.selfplay_weight
                        except Exception:
                            continue
                    if sp_loss_contribution > 0:
                        print(f"[SelfPlay] applied sp_loss={float(sp_loss_contribution.item()):.4f}")
                        # Simple reinforcement to RelMem
                        try:
                            if hasattr(topas_model, "relmem") and topas_model.relmem is not None:
                                if float(sp_loss_contribution.item()) < 1.0:
                                    topas_model.relmem.queue_hebbian_update("pattern", sid=0, oid=0, eta=0.05)
                                else:
                                    topas_model.relmem.add_exception(0, "has_attr", 0)
                        except Exception:
                            pass

                    # Alpha-ARC X: Sample from replay buffer for additional learning
                    if replay_buffer is not None and len(replay_buffer) > 10:
                        try:
                            replay_samples = replay_buffer.sample(4)  # Sample 4 diverse traces
                            replay_loss_total = torch.tensor(0.0, device=device)
                            replay_count = 0

                            for trace in replay_samples:
                                if not hasattr(trace, 'input_grid') or not hasattr(trace, 'target_grid'):
                                    continue

                                # Execute replayed program for training
                                try:
                                    # Forward pass on replayed input
                                    replay_input = trace.input_grid.to(device) if torch.is_tensor(trace.input_grid) else torch.as_tensor(trace.input_grid, device=device)
                                    if replay_input.dim() == 2:
                                        replay_input = replay_input.unsqueeze(0)

                                    replay_target = trace.target_grid.to(device) if torch.is_tensor(trace.target_grid) else torch.as_tensor(trace.target_grid, device=device)
                                    if replay_target.dim() == 2:
                                        replay_target = replay_target.unsqueeze(0)

                                    # Forward through model
                                    replay_outputs = topas_model.forward_pretraining(
                                        replay_input,
                                        target_shape=replay_target.shape[-2:],
                                        replay_mode=True
                                    )

                                    replay_logits = replay_outputs.get('logits')
                                    if replay_logits is not None:
                                        # Compute CE loss on replayed trace
                                        B, H, W = replay_target.shape
                                        replay_logits_flat = replay_logits.reshape(-1, replay_logits.size(-1))
                                        replay_target_flat = replay_target.reshape(-1).long()

                                        # Weight by trace priority and score
                                        trace_priority = getattr(trace, 'score', 0.5) * 0.3
                                        replay_loss_sample = F.cross_entropy(replay_logits_flat, replay_target_flat)
                                        replay_loss_total = replay_loss_total + trace_priority * replay_loss_sample
                                        replay_count += 1

                                        # Update op_success_count
                                        if hasattr(trace, 'program'):
                                            for op in trace.program:
                                                op_success_count.update({op: max(1, int(trace.score * 5))})

                                except Exception as e:
                                    if global_step % 100 == 0:
                                        logger.warning(f"[Replay] Trace execution failed: {e}")
                                    continue

                            # Backprop on accumulated replay loss
                            if replay_count > 0:
                                replay_loss_avg = replay_loss_total / replay_count
                                scaler.scale(replay_loss_avg).backward()
                                scaler.step(optimizer)
                                scaler.update()
                                optimizer.zero_grad()

                                if global_step % 100 == 0:
                                    logger.info(f"[Replay Training] Trained on {replay_count} traces, loss={float(replay_loss_avg):.4f}")
                        except Exception as e:
                            logger.warning(f"[Replay Training] Failed: {e}")
                else:
                    print(f"[SelfPlay] No samples available for training")
            
            if result is not None:
                if compute_metrics_now and isinstance(result, tuple):
                    loss, metrics = result
                    # Add self-play contribution to loss
                    if sp_loss_contribution > 0:
                        loss = loss + sp_loss_contribution
                    epoch_losses.append(loss)
                    for k, v in metrics.items():
                        if k == 'batch_size':
                            epoch_metrics['batch_sizes'].append(v)  # batch_size → batch_sizes (plural)
                        elif k in epoch_metrics:
                            epoch_metrics[k].append(v)
                else:
                    loss = result
                    # Add self-play contribution to loss
                    if sp_loss_contribution > 0:
                        loss = loss + sp_loss_contribution
                    if loss is not None:
                        epoch_losses.append(loss)

                    # === Complete Quick Metrics (EM + Acc + IoU) for ALL batches ===
                    # Fills in metrics for 29/30 steps when return_metrics=False (performance optimization)
                    # Ensures epoch_metrics lists have equal length for accurate averaging
                    try:
                        if outputs is not None and 'logits' in outputs and target_grid is not None:
                            logits = outputs['logits']
                            preds = logits.argmax(dim=-1)  # [B, H*W]
                            B = preds.size(0)
                            targets_flat = target_grid.reshape(B, -1)  # [B, H*W]

                            # All three metrics (maintain list length parity)
                            quick_em = (preds == targets_flat).all(dim=1).float().mean().item()
                            quick_acc = (preds == targets_flat).float().mean().item()  # Pixel-wise accuracy

                            # Compute real IoU (fast version)
                            quick_iou = 0.0
                            for c in range(10):
                                pred_c = (preds == c)
                                targ_c = (targets_flat == c)
                                intersection = (pred_c & targ_c).sum().float()
                                union = (pred_c | targ_c).sum().float()
                                if union > 0:
                                    quick_iou += (intersection / union).item()
                            quick_iou /= 10.0  # Average across colors

                            epoch_metrics['exact_match'].append(quick_em)
                            epoch_metrics['accuracy'].append(quick_acc)
                            epoch_metrics['mean_iou'].append(quick_iou)
                            epoch_metrics['batch_sizes'].append(1)
                    except Exception as e:
                        if global_step % 500 == 0:
                            logging.warning(f"Quick metrics computation failed: {e}")
            
            # Enhanced RelMem stats logging
            relmem_log_interval = getattr(trainer_cli_args, 'relmem_log_interval', 200) if 'trainer_cli_args' in globals() else 200
            if global_step % relmem_log_interval == 0:
                try:
                    relmem_stats = {}
                    if hasattr(topas_model, 'relmem') and topas_model.relmem is not None:
                        if hasattr(topas_model.relmem, 'get_stats'):
                            relmem_stats = topas_model.relmem.get_stats()
                        elif hasattr(topas_model.relmem, 'stats'):
                            relmem_stats = topas_model.relmem.stats()
                        else:
                            # Enhanced basic stats collection
                            relmem_stats = {
                                'concepts_count': getattr(topas_model.relmem, 'concepts_count', 0),
                                'relations_count': len(getattr(topas_model.relmem, 'relations', [])),
                                'last_binding_success': getattr(topas_model.relmem, 'last_binding_success', False),
                                'regularization_strength': getattr(topas_model.relmem, 'regularization_strength', 0.0),
                                'active_concepts': getattr(topas_model.relmem, 'active_concepts', 0)
                            }
                    
                    if relmem_stats:
                        stats_str = ', '.join([f"{k}={v}" for k, v in relmem_stats.items()])
                        logging.info(f"[Step {global_step}] RelMem: {stats_str}")
                        
                except Exception as e:
                    if global_step % 1000 == 0:  # Less frequent warning
                        logging.warning(f"RelMem stats logging failed: {e}")

            # Update prev_acc for next iteration's velocity calculation
            if compute_metrics_now and isinstance(result, tuple):
                prev_acc = current_acc

            global_step += 1

            # Update progress bar
            if len(epoch_losses) > 0:
                postfix = {"loss": f"{sum(epoch_losses[-10:]) / min(10, len(epoch_losses)):.4f}", "step": global_step}
                if len(epoch_metrics['exact_match']) > 0:
                    postfix['EM'] = f"{sum(epoch_metrics['exact_match'][-5:]) / min(5, len(epoch_metrics['exact_match'])):.2%}"
                if len(epoch_metrics['accuracy']) > 0:
                    # ARC-II: Guard against empty accuracy list (ZeroDivisionError fix)
                    if len(epoch_metrics['accuracy']) > 0:
                        postfix['acc'] = f"{sum(epoch_metrics['accuracy'][-5:]) / min(5, len(epoch_metrics['accuracy'])):.2%}"
                    else:
                        postfix['acc'] = "0.00%"
                progress.set_postfix(postfix)

            # Save checkpoint every 2 epochs (more frequent for monitoring)
            if global_step % (len(dataset) * 2) == 0 and global_step > 0:
                # === ALPHA-ARC: Bundle format with policy/value nets ===
                checkpoint = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'topas': topas_model.state_dict(),  # Main model
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': sum(epoch_losses[-100:]) / min(100, len(epoch_losses)) if epoch_losses else 0,
                    'best_em': best_em,
                    'best_acc': best_acc
                }

                # Add policy/value nets if they exist
                if hasattr(topas_model, 'policy_net') and topas_model.policy_net is not None:
                    checkpoint['policy_head'] = topas_model.policy_net.state_dict()
                if hasattr(topas_model, 'value_net') and topas_model.value_net is not None:
                    checkpoint['value_head'] = topas_model.value_net.state_dict()

                # Add distillation optimizer state (separate from main optimizer)
                if 'distill_optimizer' in locals() and distill_optimizer is not None:
                    checkpoint['distill_optimizer'] = distill_optimizer.state_dict()
                if 'distill_scaler' in locals() and distill_scaler is not None:
                    checkpoint['distill_scaler'] = distill_scaler.state_dict()

                torch.save(checkpoint, f'checkpoints/checkpoint_step_{global_step}.pt')
                print(f"ðŸ'¾ Saved Alpha-ARC bundle at step {global_step}")

                # Also save as alpha_arc_bundle.pt (latest)
                torch.save(checkpoint, 'checkpoints/alpha_arc_bundle.pt')
                print(f"ðŸ'¾ Updated checkpoints/alpha_arc_bundle.pt")

        # ---- Epoch end: RelMem capacity monitoring + refinement ----
        try:
            if hasattr(topas_model, "relmem") and topas_model.relmem is not None:
                # Capacity monitoring (check before refinement)
                max_concepts = getattr(topas_model.relmem, 'N', 3072)
                active_concepts = int(topas_model.relmem.concept_used.sum().item()) if hasattr(topas_model.relmem, 'concept_used') else 0
                capacity_pct = (active_concepts / max(1, max_concepts)) * 100

                logger.info(f"[Epoch {epoch}] RelMem Capacity: {active_concepts}/{max_concepts} ({capacity_pct:.1f}% full)")

                # Warn if approaching capacity
                if capacity_pct > 80:
                    logger.warning(f"[RelMem] ⚠️ Capacity at {capacity_pct:.1f}% - approaching limit!")

                    # GPU-first pruning: trigger early to keep hot concepts on GPU
                    logger.info(f"[RelMem] 🔧 GPU-first pruning triggered at {capacity_pct:.1f}% capacity")
                    try:
                        if hasattr(topas_model.relmem, 'prune_if_needed'):
                            # Prune to 70% capacity, keep 60% hottest on GPU
                            topas_model.relmem.prune_if_needed(threshold=0.8)
                            # Recount after pruning
                            active_after = int(topas_model.relmem.concept_used.sum().item())
                            pruned_count = active_concepts - active_after
                            logger.info(f"[RelMem] Pruned {pruned_count} concepts: {active_concepts} → {active_after}")
                        elif hasattr(topas_model.relmem, 'prune_compact'):
                            # Fallback to old method
                            topas_model.relmem.prune_compact(max_concepts=int(max_concepts * 0.8), merge_cos=0.985)
                            active_after = int(topas_model.relmem.concept_used.sum().item())
                            pruned_count = active_concepts - active_after
                            logger.info(f"[RelMem] Pruned {pruned_count} concepts: {active_concepts} → {active_after}")
                    except Exception as prune_err:
                        logger.error(f"[RelMem] Auto-prune failed: {prune_err}")

                # Regular refinement step
                topas_model.relmem.refinement_step()
        except Exception as e:
            pass
        
        if len(epoch_losses) > 0:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            summary = f"Epoch {epoch+1} complete: avg_loss={avg_loss:.4f}"
            
            # ---- Logging: include RelMem stats ----
            try:
                if hasattr(topas_model, "relmem") and topas_model.relmem is not None:
                    relmem_info = topas_model.relmem.stats()
                    if relmem_info:
                        summary += f" | RelMem active={int(relmem_info.get('relmem_active', 0))} "
                        summary += f"depth={relmem_info.get('relmem_depth', 0):.2f} "
                        summary += f"exceptions={int(relmem_info.get('relmem_exceptions', 0))}"
            except Exception:
                pass

            # ---- Logging: include BrainGraph stats ----
            try:
                if hasattr(topas_model, "brain_graph") and topas_model.brain_graph is not None:
                    node_count = len(topas_model.brain_graph.nodes)
                    concept_count = sum(1 for n in topas_model.brain_graph.nodes.values() if n.kind == "concept")
                    if node_count > 0:
                        summary += f" | BrainGraph nodes={node_count} concepts={concept_count}"
            except Exception:
                pass

            # --- Wormhole consolidation ---
            try:
                if hasattr(topas_model, "dream") and hasattr(topas_model.dream, "wormhole"):
                    consolidator = getattr(topas_model.dream.wormhole, "consolidator", None)
                    if consolidator is not None:
                        # Consolidate programs collected during the epoch
                        # Assume extras logged some programs; you can adapt source
                        all_programs = []
                        if hasattr(topas_model, "task_history"):
                            for tid, perf in topas_model.task_history.items():
                                if "programs" in perf:
                                    all_programs.extend(perf["programs"])
                        if all_programs:
                            new_templates = consolidator.consolidate(all_programs, top_k=20)
                            logging.getLogger(__name__).info(
                                f"[Wormhole] Consolidated {len(new_templates)} templates at epoch {epoch+1}"
                            )
            except Exception as e:
                logging.getLogger(__name__).warning(f"[Wormhole] consolidation failed: {e}")
            
            if len(epoch_metrics['exact_match']) > 0:
                # FIXED: Weighted averaging by batch size (correct metric aggregation)
                batch_sizes = epoch_metrics.get('batch_sizes', [])

                # DEBUG: Log batch_sizes tracking status
                print(f"[DEBUG] epoch_metrics keys: {list(epoch_metrics.keys())}")
                print(f"[DEBUG] exact_match count: {len(epoch_metrics['exact_match'])}, batch_sizes count: {len(batch_sizes)}")

                if len(batch_sizes) == len(epoch_metrics['exact_match']) and sum(batch_sizes) > 0:
                    # Weighted average
                    total_samples = sum(batch_sizes)
                    avg_em = sum(em * bs for em, bs in zip(epoch_metrics['exact_match'], batch_sizes)) / total_samples
                    avg_acc = sum(acc * bs for acc, bs in zip(epoch_metrics['accuracy'], batch_sizes)) / total_samples if len(epoch_metrics['accuracy']) > 0 else 0.0
                    avg_iou = sum(iou * bs for iou, bs in zip(epoch_metrics['mean_iou'], batch_sizes)) / total_samples if len(epoch_metrics['mean_iou']) > 0 else 0.0
                    print(f"[DEBUG] Using WEIGHTED average: total_samples={total_samples}, avg_em={avg_em:.4f}")
                else:
                    # Fallback to simple average if batch_sizes not tracked
                    avg_em = sum(epoch_metrics['exact_match']) / len(epoch_metrics['exact_match']) if len(epoch_metrics['exact_match']) > 0 else 0.0
                    avg_acc = sum(epoch_metrics['accuracy']) / len(epoch_metrics['accuracy']) if len(epoch_metrics['accuracy']) > 0 else 0.0
                    avg_iou = sum(epoch_metrics['mean_iou']) / len(epoch_metrics['mean_iou']) if len(epoch_metrics['mean_iou']) > 0 else 0.0
                    print(f"[DEBUG] Using SIMPLE average (fallback): avg_em={avg_em:.4f}")
                summary += f", EM={avg_em:.2%}, acc={avg_acc:.2%}, IoU={avg_iou:.3f}"
                
                # EBR refined metrics removed - always identical to base EM
                
                # Track best metrics and save checkpoints
                if avg_em > best_em:
                    best_em = avg_em
                    print(f"ðŸŽ¯ New best EM: {best_em:.2%}")
                    
                    # Save best EM checkpoint with metric in filename
                    best_em_filename = f"checkpoints/best_em_{best_em*100:.1f}.pt"
                    torch.save(topas_model.state_dict(), best_em_filename)
                    print(f"ðŸ’¾ Saved best EM checkpoint: {best_em_filename}")
                    
                if avg_acc > best_acc:
                    best_acc = avg_acc
                    print(f"ðŸŽ¯ New best accuracy: {best_acc:.2%}")
                    
                    # Save best accuracy checkpoint with metric in filename
                    best_acc_filename = f"checkpoints/best_acc_{best_acc*100:.1f}.pt"
                    torch.save(topas_model.state_dict(), best_acc_filename)
                    print(f"ðŸ’¾ Saved best accuracy checkpoint: {best_acc_filename}")
                
                # Periodic evaluation checkpoints (every eval_interval epochs)
                if cli_args and hasattr(cli_args, 'eval_interval') and (epoch + 1) % cli_args.eval_interval == 0:
                    eval_filename = f"checkpoints/eval_epoch_{epoch+1}_em_{avg_em*100:.1f}.pt"
                    torch.save(topas_model.state_dict(), eval_filename)
                    print(f"ðŸ“Š Saved evaluation checkpoint: {eval_filename}")
            
            print(summary)

        # === DREAM: full offline consolidation on schedule ===
        try:
            # Debug: Check if conditions are met (use print for visibility in speed mode)
            if (epoch + 1) % 8 == 0:  # Only log on potential trigger epochs
                has_cli_args = trainer_cli_args is not None
                print(f"[Dream-Check] epoch={epoch+1}, has_cli_args={has_cli_args}")

            if trainer_cli_args:
                full_every = getattr(trainer_cli_args, "dream_full_every", 0)

                if (epoch + 1) % 8 == 0:  # Only log on potential trigger epochs
                    should_trigger = full_every and ((epoch + 1) % int(full_every) == 0)
                    print(f"[Dream-Check] full_every={full_every}, should_trigger={should_trigger}, epoch+1={epoch+1}")

                if full_every and ((epoch + 1) % int(full_every) == 0):
                    # Check we enabled dream in config
                    try:
                        enabled = bool(getattr(topas_model.config, "enable_dream", False))
                    except Exception:
                        enabled = False

                    print(f"[Dream-Check] enable_dream={enabled} on topas_model.config")

                    if enabled:
                        timeout = int(getattr(trainer_cli_args, "dream_full_timeout", 600))
                        bg = bool(getattr(trainer_cli_args, "dream_background", False))
                        force_cpu = bool(getattr(trainer_cli_args, "dream_force_cpu", False))
                        logging.info("[Dream-Trainer] Triggering full dream cycle (epoch %d)", epoch+1)
                        # epoch boundary: build canonical 1152-dim dream tokens and run dream cycle
                        stats = None
                        try:
                            cached_batch = globals().get('last_batch_for_dream', None)
                            outputs = topas_model.forward_pretraining(
                                cached_batch['test_grid'].to(device),
                                target_shape=cached_batch['test_grid'].shape[-2:]
                            ) if cached_batch is not None else None

                            hemi_stats = {}
                            if outputs is not None and "brain" in outputs and "slot_vecs" in outputs:
                                brain = outputs["brain"]         # [B, 1152] (symbolic/global)
                                slot_vecs = outputs["slot_vecs"] # [B, T, 512] (perceptual/objects)
                                B, T, _ = slot_vecs.shape
                                # Left: symbolic (brain expanded to slots)
                                left_tokens  = brain.unsqueeze(1).expand(B, T, -1)
                                # Right: perceptual (use slot vectors; if Dream expects ctrl_dim, pad w/ zeros)
                                if left_tokens.size(-1) != slot_vecs.size(-1):
                                    pad = torch.zeros(B, T, left_tokens.size(-1) - slot_vecs.size(-1), device=slot_vecs.device)
                                    right_tokens = torch.cat([slot_vecs, pad], dim=-1)
                                else:
                                    right_tokens = slot_vecs
                                # Run both cycles
                                hemi_stats['left']  = topas_model.run_dream_cycle(tokens=left_tokens,  demos_programs=outputs.get("extras", {}).get("programs") if outputs else None)
                                hemi_stats['right'] = topas_model.run_dream_cycle(tokens=right_tokens, demos_programs=outputs.get("extras", {}).get("programs") if outputs else None)
                                # Choose a winner (prefer refined EM if present)
                                def score(s): 
                                    if not isinstance(s, dict): return 0.0
                                    return float(s.get("exact_match") or s.get("em") or 0.0)  # Use base EM since EBR is identical
                                left_score, right_score = score(hemi_stats['left']), score(hemi_stats['right'])
                                winner = 'left' if left_score >= right_score else 'right'
                                logging.info(f"[Dream-UniHemi] left={left_score:.3f} right={right_score:.3f} â†’ winner={winner}")
                                # Small, bounded nudge to priors based on winner
                                if hasattr(topas_model.config, 'relmem_op_bias_w'):
                                    delta = 0.02 if winner == 'left' else -0.01
                                    topas_model.config.relmem_op_bias_w = float(
                                        max(0.15, min(getattr(topas_model.config, 'relmem_op_bias_w', 0.2) + delta,
                                                      getattr(topas_model.config, '_bias_max', 0.5)))
                                    )
                                # Optionally skew planner op_bias success counts slightly
                                if winner == 'left':
                                    op_success_count.update(["planner_align_bonus"])
                                else:
                                    op_success_count.update(["percept_align_bonus"])
                                stats = hemi_stats[winner]
                            else:
                                # fallback: single-hemisphere like before
                                dream_tokens = getattr(topas_model, "_dream_tokens", None)
                                stats = topas_model.run_dream_cycle(tokens=dream_tokens, demos_programs=outputs.get("extras", {}).get("programs") if outputs else None)
                        except Exception as e:
                            logging.exception("[Dream] Full cycle failed: %s", e)
                        # If stats contains EM or other metrics, push into epoch_metrics for visibility
                        if isinstance(stats, dict):
                            # EBR metrics collection removed - no additional value over base EM
                            # log other stats
                            logging.info("[Dream-Trainer] Full dream stats keys: %s", list(stats.keys()))
                        
                        # Generate self-play puzzles after dream cycle
                        if self_play_buffer and cli_args and cli_args.selfplay_enable:
                            try:
                                # Sample recent training examples for transformation
                                recent_samples = []
                                sample_count = 0
                                for batch_idx, batch in enumerate(dataloader):
                                    if sample_count >= cli_args.selfplay_topk:
                                        break
                                    try:
                                        demos, test_inputs, test_outputs, task_id = batch
                                        if test_inputs is not None and test_outputs is not None:
                                            recent_samples.append((test_inputs, test_outputs))
                                            sample_count += 1
                                    except Exception as e:
                                        logger.debug(f"[SelfPlay] Sample collection failed: {e}")
                                        continue
                                
                                if recent_samples and hasattr(topas_model, 'wormhole'):
                                    new_puzzles = self_play_buffer.generate_from_wormhole(
                                        recent_samples, 
                                        topas_model.wormhole,
                                        themes=getattr(topas_model.dream, 'theme', None) if hasattr(topas_model, 'dream') else None,
                                        top_k=cli_args.selfplay_topk
                                    )
                                    if new_puzzles:
                                        print(f"ðŸŽ® Generated {len(new_puzzles)} self-play puzzles (buffer: {len(self_play_buffer.buffer)})")
                                        
                            except Exception as e:
                                print(f"âš ï¸  Self-play generation failed: {e}")
                    else:
                        logging.info("[Dream-Trainer] Dream disabled in model config; skipping full cycle.")
        except Exception as e:
            import traceback
            logging.warning("[Dream-Trainer] Dream scheduling failed: %s", traceback.format_exc())

    # Save final checkpoint
    # === ALPHA-ARC: Bundle format with policy/value nets ===
    final_checkpoint = {
        'epoch': num_epochs,
        'global_step': global_step,
        'topas': topas_model.state_dict(),  # Main model
        'optimizer_state_dict': optimizer.state_dict(),
        'best_em': best_em,
        'best_acc': best_acc
    }

    # Add policy/value nets if they exist
    if hasattr(topas_model, 'policy_net') and topas_model.policy_net is not None:
        final_checkpoint['policy_head'] = topas_model.policy_net.state_dict()
    if hasattr(topas_model, 'value_net') and topas_model.value_net is not None:
        final_checkpoint['value_head'] = topas_model.value_net.state_dict()

    # Add distillation optimizer state (separate from main optimizer)
    if 'distill_optimizer' in locals() and distill_optimizer is not None:
        final_checkpoint['distill_optimizer'] = distill_optimizer.state_dict()
    if 'distill_scaler' in locals() and distill_scaler is not None:
        final_checkpoint['distill_scaler'] = distill_scaler.state_dict()

    torch.save(final_checkpoint, 'checkpoints/checkpoint_final.pt')
    torch.save(final_checkpoint, 'checkpoints/alpha_arc_bundle.pt')  # Also save as bundle
    print(f"ðŸ'¾ Saved final Alpha-ARC bundle: best_em={best_em:.2%}, best_acc={best_acc:.2%}")

    # Log signal purity metrics if HRM model has metrics
    if hrm_model is not None and hasattr(hrm_model, 'get_signal_purity_metrics'):
        purity_metrics = hrm_model.get_signal_purity_metrics()
        print("\nðŸ“Š Signal Purity Metrics:")
        print(f"   Total encode calls: {purity_metrics['total_calls']}")
        print(f"   Successful: {purity_metrics['successful_calls']} ({purity_metrics['success_rate']:.1%})")
        print(f"   Fallback: {purity_metrics['fallback_calls']} ({purity_metrics['fallback_rate']:.1%})")
        print(f"   Missing task_id: {purity_metrics['missing_task_id']} ({purity_metrics['missing_task_id_rate']:.1%})")
        print(f"   Unique tasks: {purity_metrics['unique_tasks']}")

    print("\nðŸŽ‰ Training completed successfully!")
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\nâœ… Simplified HRM-TOPAS training WORKS!")
    except Exception as e:
        print(f"\nâŒ Training failed: {e}")
        import traceback; traceback.print_exc()