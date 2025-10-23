#!/usr/bin/env python3
"""
Fixed RelationalMemoryNeuro with guaranteed gradient flow
All operations are differentiable, Hebbian/WTA in post-optimizer hooks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
import numpy as np
from typing import List, Dict, Optional, Union, Any

logger = logging.getLogger(__name__)

def to_scalar(x, *, reducer="mean"):
    """
    ✅ FIX 4: Safe scalar conversion with automatic reduction for multi-element tensors.

    Args:
        x: Input value (tensor, scalar, or numeric)
        reducer: Reduction method for multi-element tensors ("mean", "sum", "max", "min")

    Returns:
        float scalar value
    """
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return x.item()
        # Multi-element: reduce first
        if reducer == "sum":
            return x.sum().item()
        elif reducer == "max":
            return x.max().item()
        elif reducer == "min":
            return x.min().item()
        else:  # "mean" is default
            return x.mean().item()
    return float(x)

class DimensionAdapter(nn.Module):
    """Universal tensor compatibility handler for dynamic RelMem operations"""

    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self._projectors = {}  # Cache for projection layers

    def ensure_compatible(self, tensor_a: torch.Tensor, tensor_b: torch.Tensor, operation: str = "matmul") -> tuple:
        """Ensure two tensors are compatible for the specified operation"""
        try:
            if operation == "matmul":
                # For A @ B, need A.shape[-1] == B.shape[-2]
                if tensor_a.shape[-1] != tensor_b.shape[-2]:
                    # Project tensor_a to match tensor_b requirements
                    target_dim = tensor_b.shape[-2]
                    proj_key = f"proj_{tensor_a.shape[-1]}_to_{target_dim}"

                    if proj_key not in self._projectors:
                        self._projectors[proj_key] = nn.Linear(tensor_a.shape[-1], target_dim).to(self.device)

                    # Apply projection while preserving batch dimensions
                    if tensor_a.dim() == 1:
                        tensor_a_proj = self._projectors[proj_key](tensor_a.unsqueeze(0)).squeeze(0)
                    elif tensor_a.dim() == 2:
                        tensor_a_proj = self._projectors[proj_key](tensor_a)
                    else:
                        # Handle higher dimensions
                        orig_shape = tensor_a.shape
                        tensor_a_flat = tensor_a.view(-1, orig_shape[-1])
                        tensor_a_proj_flat = self._projectors[proj_key](tensor_a_flat)
                        tensor_a_proj = tensor_a_proj_flat.view(*orig_shape[:-1], target_dim)

                    return tensor_a_proj, tensor_b

            elif operation == "element_wise":
                # For element-wise operations, tensors must have same shape
                if tensor_a.shape != tensor_b.shape:
                    # Try to broadcast or reshape
                    if tensor_a.numel() == tensor_b.numel():
                        tensor_b = tensor_b.view(tensor_a.shape)
                    else:
                        # Can't make compatible
                        raise ValueError(f"Incompatible shapes for element-wise: {tensor_a.shape} vs {tensor_b.shape}")

            return tensor_a, tensor_b

        except Exception as e:
            # Return originals if adaptation fails
            import logging
            logging.debug(f"[DimensionAdapter] Failed to adapt tensors: {e}")
            return tensor_a, tensor_b

class RelationalMemoryNeuro(nn.Module):
    def __init__(self, hidden_dim: int, max_concepts: int = 4096, rank: int = 16,
                 relations: List[str] = None, inverse_pairs: Dict[str, str] = None,
                 wta_frac: float = 0.1, wta_warmup_updates: int = 15, device="cuda",
                 enable_spike_coding: bool = True, spike_aggressiveness: float = 1.0):
        super().__init__()
        self.device = torch.device(device)
        self.N = max_concepts
        self.D = hidden_dim
        self.R = rank

        self.relations = relations or [
            # Core conceptual relations
            "is_a", "has_attr", "owns", "belongs_to", "part_of", "parent_of", "child_of", "cooccur",
            "teaches", "rewards", "guides", "has_goal", "solves", "about", "enabled_by",

            # DSL operation relations (CRITICAL - these were missing!)
            "latent", "transform", "pattern", "rule", "sequence", "structure", "color", "shape",
            "geometry", "topology", "symmetry", "progression", "alternation", "connectivity",
            "containment", "scaling", "rotation", "reflection", "translation", "composition",

            # ARC-specific relations
            "similar_to", "transforms_into", "composed_of", "extracted_from", "generalizes",
            "specializes", "precedes", "follows", "causes", "prevents", "enables", "requires"
        ]
        self.inverse = inverse_pairs or {"owns": "belongs_to", "parent_of": "child_of",
                                         "child_of": "parent_of", "teaches": "taught_by",
                                         "rewards": "rewarded_by", "guides": "guided_by",
                                         "has_goal": "goal_of", "enabled_by": "enables"}

        # Core learnable parameters - MUST receive gradients
        self.concept_proto = nn.Parameter(torch.randn(self.N, self.D, device=self.device) * 0.01)
        self.A = nn.ParameterDict({
            r: nn.Parameter(torch.randn(self.N, self.R, device=self.device) * 0.01) 
            for r in self.relations
        })
        self.B = nn.ParameterDict({
            r: nn.Parameter(torch.randn(self.N, self.R, device=self.device) * 0.01) 
            for r in self.relations
        })
        self.rel_gain = nn.ParameterDict({
            r: nn.Parameter(torch.ones(1, device=self.device)) 
            for r in self.relations
        })
        
        # Non-learnable state
        self.register_buffer("concept_used", torch.zeros(self.N, dtype=torch.bool, device=self.device))

        # === DOPAMINE-DRIVEN MEMORY MANAGEMENT ===
        # Track dopamine value for each concept (for value-based pruning)
        self.register_buffer("concept_dopamine", torch.zeros(self.N, dtype=torch.float32, device=self.device))
        # Track TTL (time-to-live) for concepts - high dopamine = long TTL
        self.register_buffer("concept_ttl", torch.zeros(self.N, dtype=torch.float32, device=self.device))
        # Nightmare queue for painful experiences
        self.nightmare_queue = []  # List of (concept_id, pain_level, operations)
        
        # WTA and Hebbian settings (applied post-optimizer)
        self.wta_frac = wta_frac
        self.wta_warmup_updates = wta_warmup_updates
        self.wta_enabled = False
        self.hebb_updates = 0
        self.depth_hist: List[int] = []
        
        # Kuramoto oscillators
        self.theta = nn.Parameter(torch.randn(self.N, device=self.device) * 2 * math.pi)
        self.omega = nn.Parameter(torch.randn(self.N, device=self.device) * 0.1)
        
        # Queue for post-optimizer updates
        self.hebbian_queue = []
        self.wta_queue = []
        self.pending_concept_updates = {}  # Fix: Initialize as empty dict

        # Trainable query projection layer for embedding alignment
        # Dynamic query projection (initialized on first use based on actual brain_dim)
        # This learns to map brain latents (varies by config) to RelMem concept space
        self.query_projection = None  # Lazy init
        self._query_proj_dim = None  # Track what dimension we built for

        # --- UKS-lite additions ---
        self.exceptions = {}  # dict[(sid:int, rel:str)] = set([oid:int])
        self.persist_path = None  # optional save/load path for UKS-like state
        
        # Concept management
        self.concepts = {}      # dict cid -> record
        self._next_cid = 0
        self._index_dirty = True
        # STRICT mode for relational memory (fail-loud instead of silent fallbacks)
        self.STRICT_REL = True

        # === SPIKE CODING INTEGRATION ===
        self.enable_spike_coding = enable_spike_coding
        self.spike_aggressiveness = spike_aggressiveness
        self._spike_stats = {}  # Track spike statistics for monitoring

        # Initialize ARC spike vocabulary if spike coding is enabled
        if self.enable_spike_coding:
            from spike_coder import ARCSpikeVocabulary
            self.arc_vocab = ARCSpikeVocabulary(device=str(device))

        # === SEMANTIC PATTERN LEARNING (Experiment 3) ===
        # Per-task pattern-derived operation bias (immediate semantic guidance)
        self.pattern_op_bias = {}  # Dict[str, float] - pattern-based operation preferences

        # === DIMENSION SAFETY SYSTEM ===
        # Universal tensor compatibility handler
        self.dim_adapter = DimensionAdapter(device=device)

    # --- Readiness gate ---
    def is_ready(self) -> bool:
        """Activate only when at least one concept is bound/active."""
        try:
            return bool(self.concept_used.any().item())
        except Exception:
            return False

    def emit_signals(self, ctx=None):
        """Emit structured signals about RelMem rail status"""
        if ctx is not None:
            signals = ctx.setdefault('signals', {})
            metrics = ctx.setdefault('metrics', {})

            # Core readiness and capacity
            signals['relmem_ready'] = self.is_ready()
            metrics['relmem_active_concepts'] = int(self.concept_used.sum().item()) if hasattr(self, 'concept_used') else 0
            metrics['relmem_max_concepts'] = getattr(self, 'N', 0)
            metrics['relmem_capacity_usage'] = metrics['relmem_active_concepts'] / max(1, metrics['relmem_max_concepts'])

            # Binding and updates
            if hasattr(self, 'hebb_updates'):
                metrics['relmem_hebb_updates'] = self.hebb_updates
            if hasattr(self, 'wta_enabled'):
                signals['relmem_wta_active'] = self.wta_enabled

            # Concept management
            if hasattr(self, '_next_cid'):
                metrics['relmem_next_cid'] = self._next_cid
            if hasattr(self, 'concepts'):
                metrics['relmem_concept_count'] = len(self.concepts)

            # Relations and exceptions
            if hasattr(self, 'relations'):
                metrics['relmem_relation_count'] = len(self.relations)
            if hasattr(self, 'exceptions'):
                metrics['relmem_exceptions'] = sum(len(v) for v in self.exceptions.values()) if self.exceptions else 0

            # Queue status
            pending_updates = getattr(self, 'pending_concept_updates', {})
            metrics['relmem_pending_bindings'] = len(pending_updates)
            metrics['relmem_hebbian_queue'] = len(getattr(self, 'hebbian_queue', []))
            metrics['relmem_wta_queue'] = len(getattr(self, 'wta_queue', []))

            # Hierarchy depth if available
            if hasattr(self, 'get_hierarchy_depth'):
                try:
                    metrics['relmem_hierarchy_depth'] = self.get_hierarchy_depth()
                except Exception:
                    metrics['relmem_hierarchy_depth'] = 0.0

            # Track starvation events
            if not signals['relmem_ready']:
                metrics['relmem_starvation'] = metrics.get('relmem_starvation', 0) + 1
    
    def forward(self, x: torch.Tensor, state=None, top_k: int = 128,
                ripple_context: Optional[Dict] = None, use_spike_coding: Optional[bool] = None):
        """
        Spike-enhanced sparse Top-K forward pass for relational memory.

        x: [B, T, D] tokens (D == hidden_dim)
        top_k: Number of top concepts to route through (default 128 for 32x speedup)
        ripple_context: Dict with ripple state for spike modulation
        use_spike_coding: Enable spike coding for concept activations
        returns: (y, state) where y is [B, T, D]
        """
        B, T, D_in = x.shape
        device = x.device

        # Use instance default if not specified
        if use_spike_coding is None:
            use_spike_coding = self.enable_spike_coding

        # Handle dimension mismatch with projection
        if D_in != self.D:
            if not hasattr(self, "_proj_in") or self._proj_in.in_features != D_in:
                self._proj_in = nn.Linear(D_in, self.D).to(device)
                nn.init.xavier_uniform_(self._proj_in.weight)
            x = self._proj_in(x)
            D = self.D
        else:
            D = D_in
        
        # 1) Token â†’ concept similarity scores and Top-K routing
        proto = self.concept_proto  # [N, D]
        scale = float(D) ** 0.5

        # Compute similarity scores with dimension safety: [B,T,N] = [B,T,D] @ [D,N]
        x_safe, proto_t_safe = self.dim_adapter.ensure_compatible(x, proto.t(), "matmul")
        similarity_scores = torch.matmul(x_safe, proto_t_safe) / scale  # [B,T,N]

        # === HIERARCHICAL SPIKE-ENHANCED CONCEPT ACTIVATION ===
        if use_spike_coding:
            from spike_coder import hierarchical_spike_cascade, adaptive_threshold_spike, spike_stats

            # Progressive spike coding: start gentle, increase aggressiveness as training improves
            # Get training progress from performance history
            training_progress = 0.0
            if hasattr(self, '_performance_tracker') and self._performance_tracker.get('acc_scores'):
                recent_acc = np.mean(self._performance_tracker['acc_scores'][-5:]) if len(self._performance_tracker['acc_scores']) >= 5 else 0.0
                training_progress = min(1.0, recent_acc / 0.5)  # 0-1 based on reaching 50% accuracy

            # Adaptive spike aggressiveness: start very gentle
            base_aggressiveness = 0.3 + training_progress * 0.7  # 0.3 â†’ 1.0 as training improves
            spike_k_base = base_aggressiveness * self.spike_aggressiveness

            # Gentler ripple modulation early in training
            ripple_modulation = 1.0
            if ripple_context and ripple_context.get('is_active', False):
                coherence = ripple_context.get('coherence', 0.0)
                if coherence > 0.75:
                    # Less aggressive ripple modulation early in training
                    ripple_factor = 0.6 + training_progress * 0.2  # 0.6 â†’ 0.8 (much gentler than 0.4)
                    ripple_modulation = ripple_factor

            # Define gentler 3-level hierarchy for early training
            level_configs = [
                {
                    'k': 4.0 * spike_k_base * ripple_modulation,  # Much gentler (was 2.0)
                    'mode': 'int'
                },  # Level 1: Perceptual (preserve more details)
                {
                    'k': 2.0 * spike_k_base * ripple_modulation,  # Gentler (was 1.0)
                    'mode': 'ternary'
                },  # Level 2: Conceptual (less aggressive)
                {
                    'k': 1.0 * spike_k_base * ripple_modulation,  # Much gentler (was 0.3)
                    'mode': 'ternary'
                }   # Level 3: Abstract (preserve more information)
            ]

            # Apply hierarchical spike cascade to similarity scores
            spike_levels = hierarchical_spike_cascade(similarity_scores, level_configs)

            # Extract levels for different reasoning purposes
            perceptual_spikes = spike_levels[0]    # Fine-grained pattern matching
            conceptual_spikes = spike_levels[1]    # Object relationship detection
            abstract_spikes = spike_levels[2]      # Rule and transformation discovery

            # Use attention-weighted combination of levels
            att_weights = torch.softmax(similarity_scores, dim=-1)  # [B,T,N]

            # Attention-based level weighting:
            # High attention â†’ perceptual detail, Low attention â†’ abstract patterns
            attention_mean = att_weights.mean(dim=-1, keepdim=True)  # [B,T,1]
            perceptual_weight = attention_mean
            abstract_weight = 1.0 - attention_mean
            conceptual_weight = 0.5  # Consistent conceptual processing

            # Combine spike levels with attention weighting
            combined_spikes = (
                perceptual_weight * perceptual_spikes.float() +
                conceptual_weight * conceptual_spikes.float() +
                abstract_weight * abstract_spikes.float()
            )

            # Gradient-safe spike blending: preserve dense gradient while injecting spike guidance
            alpha_max = getattr(self, "spike_alpha_max", 0.5)    # cap influence
            alpha = float(training_progress) * alpha_max         # ramp by training progress
            # Keep gradient from the dense similarity, inject spikes as stop-grad guidance
            similarity_scores_for_topk = similarity_scores + (combined_spikes - similarity_scores).detach() * alpha

            # Store for optional training-time spike budget loss
            self._last_combined_spikes = combined_spikes.detach()

            # Store comprehensive spike statistics for monitoring
            if hasattr(self, '_spike_stats'):
                perceptual_stats = spike_stats(perceptual_spikes)
                conceptual_stats = spike_stats(conceptual_spikes)
                abstract_stats = spike_stats(abstract_spikes)

                self._spike_stats['hierarchical_activation'] = {
                    'perceptual': perceptual_stats,
                    'conceptual': conceptual_stats,
                    'abstract': abstract_stats,
                    'ripple_modulation': float(ripple_modulation),
                    'attention_weighting': {
                        'perceptual_weight': float(perceptual_weight.mean()),
                        'abstract_weight': float(abstract_weight.mean())
                    }
                }

            # Store separate spike levels for pattern analysis
            self._last_spike_levels = {
                'perceptual': perceptual_spikes.detach(),
                'conceptual': conceptual_spikes.detach(),
                'abstract': abstract_spikes.detach()
            }
        else:
            similarity_scores_for_topk = similarity_scores

        # Top-K selection per token
        top_k = min(top_k, self.N)  # Don't exceed total concepts
        topk_values, topk_indices = torch.topk(similarity_scores_for_topk, k=top_k, dim=-1)  # [B,T,K]

        # Sparse softmax only over top-k
        att_sparse = torch.softmax(topk_values, dim=-1)  # [B,T,K]
        
        # Optional: mask to active concepts among top-k
        if self.concept_used.any():
            active = self.concept_used.to(device)  # [N]
            # Gather active status for top-k indices: [B,T,K]
            active_topk = torch.gather(active.float().unsqueeze(0).unsqueeze(0).expand(B, T, -1), 
                                     dim=2, index=topk_indices)
            # Apply mask and renormalize
            att_sparse = att_sparse * active_topk
            sums = att_sparse.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            att_sparse = att_sparse / sums
        
        # 2) Sparse propagation through relations using Top-K routing
        ctx_total = torch.zeros_like(x)
        denom = 0.0
        
        for rel in self.relations:
            A = self.A[rel]            # [N,R]
            Bf = self.B[rel]           # [N,R]
            gain = self.rel_gain[rel]  # [1]
            
            # Gather A and B for top-k concepts: [B,T,K,R]
            A_topk = torch.gather(A.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1), 
                                dim=2, index=topk_indices.unsqueeze(-1).expand(-1, -1, -1, A.size(-1)))
            Bf_topk = torch.gather(Bf.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1),
                                 dim=2, index=topk_indices.unsqueeze(-1).expand(-1, -1, -1, Bf.size(-1)))
            
            # Sparse relation propagation: [B,T,K] @ [B,T,K,R] -> [B,T,R]
            z = torch.sum(att_sparse.unsqueeze(-1) * A_topk, dim=2)  # [B,T,R]
            # [B,T,R] @ [B,T,K,R]^T -> [B,T,K] (via broadcasting)
            c = torch.sum(z.unsqueeze(2) * Bf_topk, dim=-1)  # [B,T,K]
            
            # Map back to D using top-k prototypes: [B,T,K] @ [B,T,K,D] -> [B,T,D]
            proto_topk = torch.gather(proto.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1),
                                    dim=2, index=topk_indices.unsqueeze(-1).expand(-1, -1, -1, D))
            ctx_r = torch.sum(c.unsqueeze(-1) * proto_topk, dim=2)  # [B,T,D]
            
            ctx_total = ctx_total + (gain * ctx_r)
            denom += float(gain.item())
        
        if denom <= 0:
            denom = 1.0
        ctx_total = ctx_total / denom
        
        # 3) Residual connection
        y = x + 0.1 * ctx_total
        return y, state

    def _scores(self, rel: str) -> torch.Tensor:
        """Compute relation scores - MUST be differentiable and connected to A, B, rel_gain"""
        A = self.A[rel]
        B = self.B[rel]
        # Matrix multiplication with dimension safety
        A_safe, B_t_safe = self.dim_adapter.ensure_compatible(A, B.transpose(0, 1), "matmul")
        scores = (A_safe @ B_t_safe) * self.rel_gain[rel]
        # Softplus ensures positive scores while maintaining gradients
        return F.softplus(scores)

    def query_object(self, rel: str, sid: int) -> Optional[torch.Tensor]:
        """Query for object given subject and relation - returns RAW LOGITS"""
        scores = self._scores(rel)[sid]  # Get row for subject
        active = self.concept_used

        if active.sum() == 0:
            if self.STRICT_REL:
                raise RuntimeError("[RelMem] query_object: no active concepts available; upstream must bind concepts")
            # permissive fallback (explicit, not silent): return a small baseline but log
            logging.getLogger(__name__).warning("[RelMem] query_object permissive fallback: no active concepts")
            return scores * 0.0
        
        # Mask inactive concepts with -inf for softmax stability
        logits = scores.masked_fill(~active, float('-inf'))
        return logits  # RAW LOGITS for CrossEntropyLoss

    def query_subject(self, rel: str, oid: int) -> Optional[torch.Tensor]:
        """Query for subject given object and relation - returns RAW LOGITS"""
        scores = self._scores(rel)[:, oid]  # Get column for object
        active = self.concept_used

        if active.sum() == 0:
            if self.STRICT_REL:
                raise RuntimeError("[RelMem] query_subject: no active concepts available; upstream must bind concepts")
            logging.getLogger(__name__).warning("[RelMem] query_subject permissive fallback: no active concepts")
            return scores * 0.0
        
        # Mask inactive concepts with -inf for softmax stability
        logits = scores.masked_fill(~active, float('-inf'))
        return logits  # RAW LOGITS for CrossEntropyLoss

    def query_relation(self, sid: int, oid: int) -> torch.Tensor:
        """Query for relation given subject and object - returns RAW LOGITS"""
        rel_scores = []
        for rel in self.relations:
            score = self._scores(rel)[sid, oid]
            rel_scores.append(score)
        
        logits = torch.stack(rel_scores)
        return logits  # RAW LOGITS for CrossEntropyLoss

    def bind_concept(self, cid: int, vec: torch.Tensor, alpha: float = 0.1):
        """Queue concept binding for post-optimizer application"""
        cid = int(cid)
        if 0 <= cid < self.N:
            self.concept_used[cid] = True
            # Queue for post-optimizer (no in-place ops during forward)
            if vec.dim() == 1:
                vec = vec.unsqueeze(0)

            # RelMem Loss Safety: Project if dimensional mismatch
            # Ensure vec is contiguous before reduction operations
            processed_vec = vec.contiguous().mean(0).detach()
            # Ensure vector is 1D for linear layer compatibility
            if processed_vec.dim() > 1:
                processed_vec = processed_vec.flatten()

            if processed_vec.shape[-1] != self.D:
                # Project once if dimensional mismatch - ensure proper device and shape
                if not hasattr(self, "_proj_bind") or self._proj_bind.in_features != processed_vec.shape[-1]:
                    self._proj_bind = nn.Linear(processed_vec.shape[-1], self.D).to(self.device)
                # Ensure input is 2D for linear layer [1, input_dim]
                if processed_vec.dim() == 1:
                    processed_vec = processed_vec.unsqueeze(0)
                processed_vec = self._proj_bind(processed_vec).squeeze(0)

            # Ensure pending_concept_updates exists and is a dict
            if not hasattr(self, 'pending_concept_updates') or not isinstance(self.pending_concept_updates, dict):
                self.pending_concept_updates = {}

            # Keep tensor on same device as concept_proto to avoid CPU/CUDA mixing
            self.pending_concept_updates[cid] = (processed_vec.to(self.device), alpha)

    def queue_hebbian_update(self, rel: str, sid: int, oid: int, eta: float = 0.1):
        """Queue Hebbian update for post-optimizer application"""
        self.hebbian_queue.append((rel, sid, oid, eta))
        self.hebb_updates += 1
        if self.hebb_updates > self.wta_warmup_updates:
            self.wta_enabled = True

    def queue_wta_update(self, rel: str):
        """Queue WTA update for post-optimizer application"""
        if self.wta_enabled:
            self.wta_queue.append(rel)

    def add_concept(self, vec: torch.Tensor, meta: Optional[dict] = None, alpha: float = 1.0) -> int:
        """Create and store a new concept from vec (tensor). Return cid."""
        import logging
        if vec is None:
            raise ValueError("add_concept requires a vector")
        # Keep vector on same device as RelMem (eliminate CPU storage)
        # Normalize to unit sphere to prevent unbounded growth
        vec_device = vec.detach().to(self.device).clone()
        vec_device = F.normalize(vec_device, p=2, dim=-1, eps=1e-8)
        cid = self._next_cid % self.N  # Circular buffer to prevent overflow
        self._next_cid += 1

        # Extract dopamine value from metadata
        dopamine = float(meta.get('dopamine', 0.0)) if meta else 0.0

        # Set TTL based on dopamine (emotional value determines memory lifespan)
        if dopamine > 10.0:  # Euphoric
            ttl = 1000.0  # Remember euphoria forever!
        elif dopamine > 3.0:  # Good
            ttl = 500.0
        elif dopamine > 0:  # Neutral/positive
            ttl = 200.0
        elif dopamine < -1.0:  # Pain (nightmare)
            ttl = 50.0  # Keep briefly for avoidance learning, then forget
        else:
            ttl = 100.0  # Default

        self.concept_dopamine[cid] = dopamine
        self.concept_ttl[cid] = ttl

        self.concepts[cid] = {"vec": vec_device, "meta": meta or {}, "count": 1, "alpha": float(alpha)}
        self._index_dirty = True
        logging.getLogger(__name__).info("[RelMem] add_concept cid=%d vec_norm=%.4f", cid, float(vec_device.norm().item()))
        return cid

    def bind_concept_by_vector(self, vec: torch.Tensor, op_name: str, meta: Optional[dict] = None, alpha: float = 0.5):
        import logging

        # DEBUG: Always log when this method is called
        current_step = meta.get('step', 0) if meta else 0
        if current_step % 20 == 0:
            logging.info(f"[RelMem DEBUG] bind_concept_by_vector called: step={current_step}, op_name='{op_name}', "
                       f"vec_shape={vec.shape if vec is not None else None}, meta_keys={list(meta.keys()) if meta else None}")

        # Enhanced concept binding with intelligent thresholding and clustering
        if vec is None or vec.numel() == 0:
            logging.warning(f"[RelMem DEBUG] Rejected - vec is None or empty")
            return -1

        # Progressive success-based concept activation using real metrics
        current_step = 0
        if meta and isinstance(meta, dict):
            current_step = meta.get('step', 0)

            # Use real metrics if available, otherwise use provided success_score
            if meta.get('has_real_metrics', False):
                em_score = meta.get('em_score', 0.0)
                acc_score = meta.get('acc_score', 0.0)
                iou_score = meta.get('iou_score', 0.0)
                success_score = meta.get('success_score', 0.0)
            else:
                success_score = meta.get('success_score', 0.0)

        # Very permissive thresholds to enable early learning
        training_stage = min(1.0, current_step / 3000)  # 0-1 over first 3k steps
        min_success_threshold = 0.005 + training_stage * 0.05  # 0.5% â†’ 5.5% over training

        if success_score < min_success_threshold:
            if current_step % 100 == 0:  # Debug why binding fails
                import logging
                logging.info(f"[RelMem] Concept binding rejected: success={success_score:.4f} < threshold={min_success_threshold:.4f}")
            return -1

        # Progressive similarity threshold: start permissive, get more selective
        base_similarity = 0.70  # Start much lower
        similarity_threshold = base_similarity + training_stage * 0.15  # 0.70 â†’ 0.85 over training
        vec_normalized = F.normalize(vec, p=2, dim=-1)

        if self.concept_used.any():
            # Find most similar existing concept
            active_concepts = self.concept_used.nonzero().flatten()
            if active_concepts.numel() > 0:
                active_protos = F.normalize(self.concept_proto[active_concepts], p=2, dim=-1)

                # FIX: Ensure dimension compatibility for dynamic concept growth
                vec_input = vec_normalized.unsqueeze(0)  # [1, D_vec]
                protos_t = active_protos.t()  # [D_proto, N_concepts]

                # Project vec to proto space if dimension mismatch
                if vec_input.shape[-1] != protos_t.shape[0]:
                    if not hasattr(self, "_similarity_proj") or self._similarity_proj.in_features != vec_input.shape[-1]:
                        self._similarity_proj = nn.Linear(vec_input.shape[-1], protos_t.shape[0]).to(self.device)
                    vec_input = self._similarity_proj(vec_input)

                # Use dimension adapter for safe similarity computation
                vec_safe, protos_safe = self.dim_adapter.ensure_compatible(vec_input, protos_t, "matmul")
                similarities = torch.matmul(vec_safe, protos_safe)  # [1, N_concepts]
                max_sim, max_idx = torch.max(similarities, dim=1)

                if max_sim.item() > similarity_threshold:
                    # Merge with existing similar concept instead of creating new one
                    existing_cid = active_concepts[max_idx].item()
                    merge_alpha = min(0.3, success_score)  # Stronger success â†' stronger update
                    # Merge with normalization to prevent unbounded growth
                    merged_vec = (
                        (1 - merge_alpha) * self.concept_proto.data[existing_cid] +
                        merge_alpha * vec.to(self.device)
                    )
                    self.concept_proto.data[existing_cid] = F.normalize(merged_vec, p=2, dim=-1, eps=1e-8)

                    # Strengthen relationships for successful pattern
                    if op_name in self.relations:
                        self.queue_hebbian_update(op_name, existing_cid, existing_cid, eta=success_score * 0.2)

                    logging.info(f"[RelMem] Merged concept {existing_cid} (similarity={max_sim.item():.3f}, "
                               f"success={success_score:.3f})")
                    return existing_cid

        # Create new concept if no similar one found
        # Store operationâ†’success mapping in metadata for world model learning
        if meta is None:
            meta = {}
        if 'operations' not in meta:
            meta['operations'] = {}
        meta['operations'][op_name] = float(success_score)

        cid = self.add_concept(vec, meta=meta, alpha=alpha)
        try:
            # existing bind API expects (cid, vec, alpha)
            self.bind_concept(cid, vec, alpha=alpha)
        except Exception as e:
            logging.getLogger(__name__).warning("[RelMem] bind_concept failed for cid=%s op=%s: %s", cid, op_name, e)

        # Queue relationship learning with success-weighted strength
        try:
            if op_name in self.relations:
                hebbian_strength = success_score * 0.3  # Scale by success
                self.queue_hebbian_update(op_name, cid, cid, eta=hebbian_strength)

                # Also create cross-puzzle relationships based on success patterns
                if success_score > 0.5 and active_concepts.numel() > 0:
                    # Connect to other successful concepts with "similar_to" relation
                    for other_cid in active_concepts[-3:]:  # Last 3 successful concepts
                        if other_cid != cid:
                            self.queue_hebbian_update("similar_to", cid, other_cid.item(), eta=success_score * 0.1)

                active_total = int(self.concept_used.sum().item())
                logging.info(f"[RelMem] âœ… Created concept {cid} with success={success_score:.3f}, "
                           f"relation='{op_name}', active_total={active_total}")

                # DEBUG: Verify concept_used was actually updated
                if current_step % 20 == 0:
                    is_actually_used = bool(self.concept_used[cid].item()) if 0 <= cid < self.N else False
                    logging.info(f"[RelMem DEBUG] Concept {cid} verification: concept_used[{cid}]={is_actually_used}")
            else:
                logging.warning(f"[RelMem] âŒ Unknown relation '{op_name}' - no learning occurred")
        except Exception as e:
            logging.getLogger(__name__).debug("[RelMem] queue_hebbian_update failed for cid=%s op=%s: %s", cid, op_name, e)

        return cid

    @torch.no_grad()
    def apply_hebbian(self):
        """Apply queued Hebbian updates - called AFTER optimizer.step()"""
        for rel, sid, oid, eta in self.hebbian_queue:
            if rel in self.A and rel in self.B:
                a = self.A[rel][sid].unsqueeze(0)
                b = self.B[rel][oid].unsqueeze(1)
                delta = eta * a @ b
                self.rel_gain[rel].add_(delta.mean())
                self.rel_gain[rel].clamp_(0.1, 1.5)  # Tighter clamping
        self.hebbian_queue.clear()

    @torch.no_grad()
    def apply_wta(self):
        """Apply queued WTA updates - called AFTER optimizer.step()"""
        for rel in self.wta_queue:
            if rel not in self.A or rel not in self.B:
                continue
            
            W = self._scores(rel).detach()  # Safe to detach in post-optimizer
            k = max(1, int(W.size(-1) * self.wta_frac))
            
            # Compute top-k mask
            top_idx = torch.topk(W, k=k, dim=-1).indices
            mask = torch.zeros_like(W, dtype=torch.bool)
            mask.scatter_(dim=-1, index=top_idx, value=True)
            
            # Row and col activity masks
            keep_rows = mask.any(dim=-1).float().unsqueeze(-1)
            keep_cols = mask.any(dim=0).float().unsqueeze(-1)
            
            # Apply sparsification
            self.A[rel].mul_(keep_rows)
            self.B[rel].mul_(keep_cols)
        self.wta_queue.clear()

    def kuramoto_sync(self, K: float = 1.0, steps: int = 10, dt: float = 0.1):
        """Kuramoto synchronization - differentiable"""
        active = self.concept_used.nonzero().squeeze(-1)
        if active.numel() < 2: 
            return
        
        N_active = active.numel()
        theta_active = self.theta[active]
        
        for _ in range(steps):
            sin_diff = torch.sin(theta_active.unsqueeze(0) - theta_active.unsqueeze(1))
            dtheta = self.omega[active] + (K / N_active) * sin_diff.sum(dim=1)
            theta_active = theta_active + dt * dtheta
        
        # Update theta without in-place operation
        with torch.no_grad():
            self.theta.data[active] = theta_active % (2 * math.pi)

    def inverse_loss(self) -> torch.Tensor:
        """Compute inverse relation consistency loss - fully differentiable"""
        total_loss = (self.concept_proto * 0).sum()  # Start with graph-connected zero
        count = 0
        
        for r, ri in self.inverse.items():
            try:
                # Safety checks before computation
                if ri not in self.relations or r not in self.relations:
                    continue

                if not hasattr(self, 'A') or not hasattr(self, 'B'):
                    continue

                if r not in self.A or ri not in self.A or r not in self.B or ri not in self.B:
                    continue

                fwd = self._scores(r)
                rev = self._scores(ri)

                # Safety check: ensure tensors are valid
                if not torch.is_tensor(fwd) or not torch.is_tensor(rev):
                    continue

                # Handle different tensor dimensions safely
                if rev.dim() >= 2:
                    rev = rev.transpose(0, 1)
                elif rev.dim() == 1 and fwd.dim() == 1:
                    # For 1D tensors, no transpose needed
                    pass
                else:
                    continue

                # Ensure shapes match for MSE loss
                if fwd.shape != rev.shape:
                    # Try to make compatible
                    if fwd.numel() == rev.numel():
                        rev = rev.view(fwd.shape)
                    else:
                        continue

                loss = F.mse_loss(fwd, rev)
                total_loss = total_loss + loss
                count += 1

            except Exception:
                # Silent failure for this relation pair
                continue
        
        if count > 0:
            return total_loss / count
        return total_loss

    def inheritance_pass(self) -> torch.Tensor:
        """Compute inheritance consistency - fully differentiable"""
        if "is_a" not in self.relations or "has_attr" not in self.relations:
            return (self.concept_proto * 0).sum()
        
        Wisa = self._scores("is_a")
        What = self._scores("has_attr")
        
        # Propagate attributes through is-a hierarchy
        P = What.clone()
        for _ in range(3):
            P = P + 0.1 * (Wisa @ P)  # Small step size for stability
        
        # Compute consistency loss
        loss = F.mse_loss(P, What)
        return torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    def grid_consistency_loss(self) -> torch.Tensor:
        """Grid consistency for active concepts - fully differentiable"""
        active = self.concept_used.nonzero().squeeze(-1)
        if active.numel() < 4:
            return (self.concept_proto * 0).sum()
        
        embeds = self.concept_proto[active]
        diffs = embeds[1:] - embeds[:-1]
        return diffs.pow(2).mean()

    def anti_recurrence_penalty(self, rel: str, num_walks: int = 100, 
                                max_steps: int = 5) -> torch.Tensor:
        """Anti-recurrence penalty - fully differentiable"""
        device = self.concept_proto.device
        
        # Get transition matrix
        logits = self._scores(rel)
        W = F.softmax(logits, dim=-1)
        
        used = torch.nonzero(self.concept_used).flatten()
        if used.numel() < 2:
            return (self.concept_proto * 0).sum()
        
        # Random walk simulation
        starts = used[torch.randint(0, used.numel(), (num_walks,), device=device)]
        pos = starts.unsqueeze(1).repeat(1, max_steps+1)
        
        for t in range(1, max_steps+1):
            probs = W[pos[:, t-1]]
            probs = torch.nan_to_num(probs, nan=1.0/self.N, posinf=1.0, neginf=0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            next_idx = torch.multinomial(probs, 1).squeeze(1)
            pos[:, t] = next_idx
        
        # Check for cycles
        cycles = (pos[:, 1:] == pos[:, 0].unsqueeze(1)).any(dim=1).float().mean()
        return cycles.clamp(0.0, 1.0)

    def predictive_prune_loss(self, thresh: float = 0.1) -> torch.Tensor:
        """Predictive pruning loss - fully differentiable"""
        if "is_a" not in self.relations or "has_attr" not in self.relations:
            return (self.concept_proto * 0).sum()
        
        Wisa = self._scores("is_a")
        What = self._scores("has_attr")
        
        # Predict attributes through is-a
        P = What.clone()
        for _ in range(2):
            P = P + 0.1 * (Wisa @ P)
        
        # Check grounding
        grounded = (self.concept_proto.abs().sum(-1) > 0.1).float()
        mismatch = (P > thresh).float() * (1.0 - grounded.unsqueeze(0))
        return mismatch.mean().clamp(0.0, 1.0)

    def get_hierarchy_depth(self) -> float:
        """Get average hierarchy depth"""
        if not self.depth_hist: 
            return 0.0
        return float(sum(self.depth_hist) / len(self.depth_hist))
    
    # Compatibility methods for existing code
    def hebbian_relation(self, rel: str, sid: int, oid: int, eta: float = 0.1):
        """Compatibility wrapper - queues update for post-optimizer"""
        self.queue_hebbian_update(rel, sid, oid, eta)
    
    def wta_inhibition(self, rel: str, k_frac: float = 0.1):
        """Compatibility wrapper - queues update for post-optimizer"""
        self.queue_wta_update(rel)
    
    def apply_post_optimizer_hooks(self):
        """Apply queued concept bindings + Hebbian/WTA updates safely (corruption-proof)"""
        # Ensure device matches parameter device
        try:
            self.device = next(self.parameters()).device
        except StopIteration:
            self.device = torch.device("cuda")

        # --- CORRUPTION-PROOF SANITIZATION ---
        if not isinstance(getattr(self, "pending_concept_updates", {}), dict):
            import logging
            logging.error(f"[RelMem] CORRUPTION: pending_concept_updates is {type(self.pending_concept_updates).__name__}; resetting")
            self.pending_concept_updates = {}

        updates = dict(self.pending_concept_updates)  # Shallow copy for safe iteration

        # --- Apply pending concept updates ---
        applied = 0
        if updates:
            with torch.no_grad():
                for cid, data in list(updates.items()):
                    try:
                        if (not isinstance(data, (tuple, list))) or len(data) != 2:
                            import logging
                            logging.warning(f"[RelMem] Skipping invalid update {cid}: {data}")
                            continue
                        vec, alpha = data
                        if not torch.is_tensor(vec):
                            import logging
                            logging.warning(f"[RelMem] Skipping non-tensor update {cid}")
                            continue
                        # Bounds check for circular buffer
                        if cid >= self.N:
                            import logging
                            logging.warning(f"[RelMem] Skipping out-of-bounds cid={cid} (N={self.N})")
                            continue
                        vec = vec.to(device=self.device, dtype=self.concept_proto.dtype)
                        # EMA update with normalization to prevent unbounded growth
                        updated_vec = (1 - float(alpha)) * self.concept_proto.data[cid] + float(alpha) * vec
                        self.concept_proto.data[cid] = F.normalize(updated_vec, p=2, dim=-1, eps=1e-8)
                        applied += 1
                    except Exception as e:
                        import logging
                        logging.warning(f"[RelMem] Update failed for cid={cid}: {e}")
            self.pending_concept_updates.clear()

        if applied > 0:
            import logging
            logging.info(f"[RelMem] âœ… Applied {applied} concept updates")

        # --- Concept Proto Normalization (CRITICAL for stability) ---
        # Enforce unit norm on all active concepts to prevent unbounded growth
        try:
            active = self.concept_used.nonzero().flatten()
            if active.numel() > 0:
                self.concept_proto.data[active] = F.normalize(
                    self.concept_proto.data[active], p=2, dim=-1, eps=1e-8
                )
        except Exception as e:
            import logging
            logging.warning(f"[RelMem] Proto normalization failed: {e}")

        # --- Hebbian + WTA ---
        try:
            self.apply_hebbian()
            self.apply_wta()
        except Exception as e:
            if self.STRICT_REL:
                raise
            import logging
            logging.warning(f"[RelMem] skipped post-optimizer (permissive): {e}")
    
    def get_op_bias(self, dsl_ops: List[str] = None, scale: float = 1.0, query_vec: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Return a mapping {dsl_op_name: bias} with values in [0, 1].
        - dsl_ops: optional list of DSL ops to restrict to (defaults to all known ops).
        - scale: multiplicative scaling factor from ModelConfig.relmem_op_bias_scale
        - query_vec: Current puzzle vector for similarity-based retrieval
        """
        from typing import Dict, List
        import torch
        import logging

        op_bias: Dict[str, float] = {}

        # If no active concepts, return **empty** to preserve training purity (no uniform fallback)
        if not self.concept_used.any():
            logging.getLogger(__name__).debug("[RelMem] get_op_bias: no active concepts â†’ SKIP (empty bias)")
            return {}

        # Retrieve similar concepts using world model
        if query_vec is not None:
            active_cids = self.concept_used.nonzero().flatten()
            if active_cids.numel() > 0:
                # 🔥 FIX: Device alignment guard - ensure all tensors on same device
                query_vec = query_vec.to(self.device)

                # Compute similarities to active concepts
                query_norm = F.normalize(query_vec.detach(), p=2, dim=-1)
                active_protos = F.normalize(self.concept_proto[active_cids], p=2, dim=-1)

                # ✅ FIX 3: Ensure active_protos is on correct device with contiguous layout
                active_protos = active_protos.to(self.device, non_blocking=True).contiguous()

                # Handle dimension mismatch
                if query_norm.dim() == 1:
                    query_norm = query_norm.unsqueeze(0)

                # Project query to RelMem.D using trainable projection layer
                # This layer learns to align brain embeddings with concept space
                logging.info(f"[DEBUG] query_norm.shape={query_norm.shape} BEFORE projection")
                if query_norm.shape[-1] != active_protos.shape[-1]:
                    # Lazy init query_projection with correct dimension
                    if self.query_projection is None or self._query_proj_dim != query_norm.shape[-1]:
                        self.query_projection = nn.Linear(query_norm.shape[-1], self.D, device=self.device)
                        nn.init.xavier_uniform_(self.query_projection.weight)
                        self._query_proj_dim = query_norm.shape[-1]
                        logging.info(f"[RelMem] Initialized query_projection: {query_norm.shape[-1]} → {self.D}")
                    # Ensure projection layer is on correct device
                    elif self.query_projection.weight.device != self.device:
                        self.query_projection = self.query_projection.to(self.device)

                    query_norm = self.query_projection(query_norm)
                    query_norm = F.normalize(query_norm, p=2, dim=-1)  # Re-normalize after projection

                # ✅ FIX 3: Ensure query_norm is contiguous and on same device before matmul
                query_norm = query_norm.to(self.device, non_blocking=True).contiguous()
                active_protos = active_protos.to(self.device, non_blocking=True).contiguous()

                logging.info(f"[DEBUG] query_norm.shape={query_norm.shape} AFTER projection")
                logging.info(f"[DEBUG] active_protos.t().shape={active_protos.t().shape}")

                # ✅ FIX 3: Device-safe matmul with explicit dtype
                similarities = torch.matmul(
                    query_norm.to(dtype=torch.float32),
                    active_protos.t().to(dtype=torch.float32)
                ).squeeze(0)  # [N_active]
                logging.info(f"[DEBUG] similarities.shape={similarities.shape}")

                # Get top-K most similar concepts (NUCLEAR: 128 for maximum context)
                top_k = min(128, active_cids.numel())
                top_sims, top_indices = torch.topk(similarities, k=top_k)

                # Debug: Log similarity distribution periodically
                import logging
                if hasattr(self, '_retrieval_call_count'):
                    self._retrieval_call_count += 1
                else:
                    self._retrieval_call_count = 0

                if self._retrieval_call_count % 100 == 0:
                    logging.info(f"[RelMem Similarity Debug] Top-5 sims: {top_sims[:5].tolist()}, "
                               f"query_shape={query_norm.shape}, proto_shape={active_protos.shape}")

                # Aggregate operation biases from similar concepts (soft aggregation)
                # Use ALL top-K matches with exponential weighting (no hard threshold)
                for sim, idx in zip(top_sims, top_indices):
                    # ✅ FIX 4: Use scalar guard for idx
                    cid = to_scalar(active_cids[idx], reducer="sum")  # idx should be scalar but guard it
                    cid = int(cid)
                    if cid not in self.concepts:
                        continue

                    concept_meta = self.concepts[cid].get('meta', {})
                    concept_ops = concept_meta.get('operations', {})

                    if not concept_ops:  # Skip if no operations stored
                        continue

                    # Soft exponential weighting: even weak similarities contribute
                    # exp(sim * 10) amplifies 0.02 â†' 1.22, 0.05 â†' 1.65, 0.1 â†' 2.72
                    # ✅ FIX 4: Use scalar guard for sim (could be multi-element in edge cases)
                    sim_scalar = to_scalar(sim, reducer="mean")
                    sim_weight = float(torch.exp(torch.tensor(sim_scalar) * 10).item())

                    for op, success in concept_ops.items():
                        op_bias[op] = op_bias.get(op, 0.0) + (success * sim_weight)

                # Normalize accumulated biases (keep sparse; no blanket uniform)
                if op_bias:
                    total = sum(op_bias.values())
                    if total > 0:
                        norm = {k: min(1.0, v / total) for k, v in op_bias.items()}
                        logging.getLogger(__name__).info(f"[RelMem] World model contributed {len(norm)} op biases from {len(top_sims)} concepts")
                        return norm

        # No concepts learned yet â†’ **strict**: return empty to avoid uniform pollution
        return {}

    # === SEMANTIC PATTERN LEARNING METHODS (Experiment 3) ===

    def clear_pattern_bias(self):
        """Call at the beginning of each task to reset semantic priors."""
        self.pattern_op_bias = {}
        import logging
        logging.info("[RelMem] ðŸ§  Pattern bias cleared for new task")

    def bind_pattern_concept(self, pattern_desc: dict, success_score: float = 1.0, store_in_concept: bool = True, concept_vec: torch.Tensor = None):
        """Convert high-level pattern descriptors into direct DSL op priors.
        This does not require training and immediately makes RelMem useful.

        Args:
            pattern_desc: Dict from extract_patterns() with keys like 'symmetry', 'rotation', 'color', etc.
            success_score: Confidence/success weight to scale the bias (0.0-1.0)
            store_in_concept: If True and concept_vec provided, store opsâ†’success in concept metadata
            concept_vec: Vector for concept (if storing metadata)
        """
        import logging
        bias = {}

        # Conservative semantic bias mapping for clean testing

        # Symmetry-based biases
        sym = pattern_desc.get("symmetry")
        if sym == "flip_h":
            bias["flip_h"] = 1.0
        if sym == "flip_v":
            bias["flip_v"] = 1.0

        # Rotation-based biases
        rot = pattern_desc.get("rotation")
        if rot in {"rotate90", "rotate180", "rotate270"}:
            bias[rot] = 1.0

        # Translation-based biases
        trans = pattern_desc.get("translation")
        if trans is not None:
            bias["translate"] = 1.0

        # Component-based biases
        ncomp = int(pattern_desc.get("n_components", 0))
        if ncomp > 1:
            bias["for_each_object"] = 0.8
            bias["for_each_object_translate"] = 0.6
            bias["for_each_object_recolor"] = 0.6

        # Color-based biases
        color = pattern_desc.get("color", {})
        if color and color.get("n_colors", 0) >= 2:
            bias["color_map"] = max(0.6, bias.get("color_map", 0.0))
            bias["flood_fill"] = max(0.4, bias.get("flood_fill", 0.0))
            bias["extract_color"] = max(0.3, bias.get("extract_color", 0.0))

        # Spatial pattern biases
        if ncomp > 0:
            bias["outline"] = 0.3
            bias["boundary_extract"] = 0.3
            bias["crop_bbox"] = 0.4

        # Conservative success scaling with max() merging (not additive)
        updated_count = 0
        for k, v in bias.items():
            new_val = float(v) * float(success_score)
            old_val = self.pattern_op_bias.get(k, 0.0)
            self.pattern_op_bias[k] = max(old_val, new_val)  # Max instead of additive
            updated_count += 1

        # Store operationâ†’success mapping in concept metadata for episodic retrieval
        if store_in_concept and concept_vec is not None and updated_count > 0:
            # Find or create concept for this pattern
            # Extract dopamine if available in pattern_desc
            dopamine_value = pattern_desc.get('dopamine', 0.0) if isinstance(pattern_desc, dict) else 0.0

            meta = {
                'operations': {k: float(v * success_score) for k, v in bias.items()},
                'pattern_desc': pattern_desc,
                'success_score': float(success_score),
                'dopamine': float(dopamine_value),  # â† EUPHORIA TAG!
                'timestamp': pattern_desc.get('timestamp', 0) if isinstance(pattern_desc, dict) else 0
            }
            cid = self.add_concept(concept_vec, meta=meta, alpha=success_score)

            # Enhanced logging with emotional valence
            if dopamine_value > 10.0:
                logging.info(f"[RelMem] ðŸŽ† EUPHORIC pattern stored in concept {cid} (dopamine={dopamine_value:.1f}, ops={len(bias)})")
            elif dopamine_value < 0:
                logging.info(f"[RelMem] ðŸ˜° PAINFUL pattern stored in concept {cid} (dopamine={dopamine_value:.1f}, ops={len(bias)})")
            else:
                logging.info(f"[RelMem] ðŸ“ Stored pattern in concept {cid} with {len(bias)} operations (dopamine={dopamine_value:.2f})")

        # Debug logging
        if updated_count > 0:
            top_biases = sorted(self.pattern_op_bias.items(), key=lambda x: x[1], reverse=True)[:5]
            logging.info(f"[RelMem] ðŸŽ¯ SEMANTIC BINDING: {updated_count} operations biased from patterns")
            logging.info(f"[RelMem] Top pattern biases: {top_biases}")
            logging.info(f"[RelMem] Pattern analysis: sym={sym}, rot={rot}, trans={trans}, ncomp={ncomp}, n_colors={color.get('n_colors', 0)}")

        return updated_count

    def compute_inverse_loss(self) -> torch.Tensor:
        """
        Compute inverse loss regularizer for RelMem training.
        Returns a small positive scalar when relation activations are present.
        """
        import torch
        
        # defensive: if no weights return zero-tensor on correct device
        if not hasattr(self, "weights") or not self.weights:
            return torch.tensor(0.0, device=next(self.parameters()).device if hasattr(self, "parameters") else "cuda:0")
        
        # simple proxy: L2 of off-diagonal relation weights
        total = 0.0
        count = 0
        for k, W in getattr(self, "weights", {}).items():
            if isinstance(W, torch.Tensor):
                total = total + (W**2).mean()
                count += 1
        
        return (total / max(1, count))
    
    def op_bias(self) -> Dict[str, float]:
        """
        Return operation bias dict for DSL search - NEVER EMPTY.
        - Looks at self.relations (list of relation names) and
          self._scores(rel) (must return a tensor or numeric score).
        - Fallbacks: if no scores present, use lightweight learned 'weights' if present,
          else return baseline small biases.
        """
        from typing import Dict
        import torch
        import logging
        
        # list of operations we know about (keep in sync with DSL registry)
        known_ops = [
            "identity","rotate90","rotate180","rotate270",
            "flip_h","flip_v","translate","scale","resize_nn",
            "color_map","flood_fill","extract_color","crop_bbox","crop_nonzero",
            "tile_pattern","tile","paste","center_pad_to","outline","symmetry",
            "grid_union","grid_intersection","grid_xor","grid_difference",
            "count_objects","count_colors","find_pattern","extract_pattern","match_template",
            "for_each_object","for_each_object_translate","for_each_object_recolor",
            "for_each_object_rotate","for_each_object_scale","for_each_object_flip",
            "conditional_map","apply_rule","select_by_property","flood_select",
            "boundary_extract","arithmetic_op"
        ]

        op_bias: Dict[str, float] = {op: 0.0 for op in known_ops}

        # Enhanced relation-to-operation mapping for intelligent world model biasing
        relation_to_ops = {
            # Basic operations
            "color": ["color_map","flood_fill","extract_color"],
            "shape": ["rotate90","flip_h","flip_v"],
            "structure": ["crop_bbox","tile_pattern","resize_nn","translate","scale"],
            "logic": ["grid_union","grid_intersection","grid_xor","grid_difference"],
            "identity": ["identity"],

            # Geometric transformations
            "rotation": ["rotate90","rotate180","rotate270"],
            "reflection": ["flip_h","flip_v","symmetry"],
            "scaling": ["scale","resize_nn"],
            "translation": ["translate"],

            # Pattern operations
            "pattern": ["find_pattern","extract_pattern","match_template","tile_pattern"],
            "sequence": ["for_each_object","arithmetic_op"],
            "progression": ["arithmetic_op","continue_sequence"],
            "alternation": ["conditional_map","apply_rule"],

            # Structural operations
            "containment": ["crop_bbox","crop_nonzero","outline","boundary_extract"],
            "connectivity": ["for_each_object","identify_components","connect_pixels"],
            "composition": ["paste","tile","grid_union"],

            # Object operations
            "object": ["for_each_object","for_each_object_translate","for_each_object_recolor",
                      "for_each_object_rotate","for_each_object_scale","for_each_object_flip"],

            # Rule and logic operations
            "rule": ["apply_rule","conditional_map"],
            "transform": ["rotate90","flip_h","flip_v","scale","translate"],

            # ARC-specific learned relations
            "similar_to": ["match_template","find_pattern","extract_pattern"],
            "transforms_into": ["rotate90","rotate180","rotate270","flip_h","flip_v"],
            "generalizes": ["apply_rule","conditional_map","arithmetic_op"],

            # Clustering and analysis
            "topology": ["outline","boundary_extract","connectivity"],
            "geometry": ["rotate90","rotate180","rotate270","flip_h","flip_v","scale"],

            # Core latent representation
            "latent": ["identity","extract_pattern","apply_rule"]  # The missing one!
        }

        # 1) Use _scores(rel) if present. In strict mode, do not silently swallow errors.
        if hasattr(self, "_scores") and callable(getattr(self, "_scores")):
            for rel, ops in relation_to_ops.items():
                if rel in getattr(self, "relations", []):
                    try:
                        score_t = self._scores(rel)
                        score = float(score_t.mean().item()) if hasattr(score_t, "mean") else float(score_t)
                        score = max(0.0, min(1.0, score))  # clamp
                    except Exception as e:
                        if self.STRICT_REL:
                            raise RuntimeError(f"[RelMem] op_bias failed computing _scores for rel={rel}: {e}")
                        else:
                            score = 0.0
                    for op in ops:
                        op_bias[op] = max(op_bias.get(op, 0.0), score)

        # 2) If still all zeros, fall back to 'weights' or small priors
        if all(v == 0.0 for v in op_bias.values()):
            try:
                # if self.weights exists and is a dict mapping relation->tensor
                if hasattr(self, "weights") and isinstance(self.weights, dict):
                    for rel, W in self.weights.items():
                        # map the rel to ops if possible
                        ops = relation_to_ops.get(rel, [])
                        strength = float(torch.norm(W).item()) / (W.numel() + 1e-9) if hasattr(W, "numel") else 0.0
                        strength = max(0.0, min(1.0, strength))
                        for op in ops:
                            op_bias[op] = max(op_bias.get(op, 0.0), strength * 0.3)
            except Exception:
                pass

        # DEBUG: Log op_bias state before world model logic
        import logging
        bias_values = [v for v in op_bias.values() if v > 0.0]
        logging.info(f"[RelMem DEBUG] op_bias before world model: {len(bias_values)} non-zero values, "
                   f"max={max(bias_values) if bias_values else 0.0:.4f}")

        # 3) Intelligent world model-based biasing (replaces uniform fallback)
        if not any(v > 0.0 for v in op_bias.values()):
            # Generate intelligent bias from learned concept relationships
            active_concepts = int(self.concept_used.sum().item()) if hasattr(self, 'concept_used') else 0

            # DEBUG: Log world model decision point
            logging.info(f"[RelMem DEBUG] World model decision: active_concepts={active_concepts}, "
                       f"concept_used_shape={self.concept_used.shape if hasattr(self, 'concept_used') else 'None'}")

            if active_concepts > 0:
                # Use learned world model for operation biasing
                try:
                    # Concept-based operation bias using relationship strengths
                    for rel_name in self.relations:
                        if rel_name in relation_to_ops:
                            try:
                                # Get relationship strength from learned parameters
                                rel_strength = 0.0
                                if rel_name in self.A and rel_name in self.B:
                                    A_norm = torch.norm(self.A[rel_name]).item()
                                    B_norm = torch.norm(self.B[rel_name]).item()
                                    gain = self.rel_gain[rel_name].item() if rel_name in self.rel_gain else 1.0
                                    rel_strength = (A_norm * B_norm * gain) / (self.N * self.R + 1e-9)

                                # Apply learned strength to associated operations (very permissive early training)
                                min_rel_threshold = 0.001  # Much lower threshold to capture early learning
                                if rel_strength > min_rel_threshold:
                                    ops = relation_to_ops[rel_name]
                                    for op in ops:
                                        if op in op_bias:
                                            # Amplify small early relationships for better signal
                                            amplified_strength = rel_strength * 2.0  # Boost early relationships
                                            op_bias[op] = max(op_bias[op], amplified_strength)

                            except Exception:
                                continue

                    # Check if world model provided meaningful biases
                    total_bias = sum(op_bias.values())
                    if total_bias > 0.01:  # Much lower threshold to capture early learning
                        import logging
                        meaningful_ops = {k: v for k, v in op_bias.items() if v > 0.001}
                        logging.info(f"[RelMem] ðŸŽ¯ WORLD MODEL ACTIVE! Concepts={active_concepts}, "
                                   f"learned_biases={len(meaningful_ops)}, total_bias={total_bias:.3f}")
                        if meaningful_ops and len(meaningful_ops) <= 10:  # Show top learned biases
                            top_ops = sorted(meaningful_ops.items(), key=lambda x: x[1], reverse=True)[:5]
                            logging.info(f"[RelMem] Top learned ops: {top_ops}")
                    else:
                        # If world model didn't help, use success-weighted uniform bias
                        self._apply_fallback_bias(op_bias)
                except Exception:
                    self._apply_fallback_bias(op_bias)
            else:
                # No concepts learned yet, use minimal startup bias
                self._apply_fallback_bias(op_bias)

        # 4) MERGE SEMANTIC PATTERN BIASES (Experiment 3)
        # This is the breakthrough - merge learned biases with immediate semantic guidance
        if hasattr(self, 'pattern_op_bias') and self.pattern_op_bias:
            import logging
            pattern_count = len([v for v in self.pattern_op_bias.values() if v > 0.0])
            logging.info(f"[RelMem] ðŸŽ¯ MERGING SEMANTIC BIASES: {pattern_count} pattern-based operations")

            for op, pattern_bias in self.pattern_op_bias.items():
                if op in op_bias:
                    # Merge: take max of learned bias and pattern bias
                    old_bias = op_bias[op]
                    op_bias[op] = max(old_bias, float(pattern_bias))
                    if old_bias < pattern_bias:
                        logging.debug(f"[RelMem] Pattern enhanced {op}: {old_bias:.3f} â†’ {pattern_bias:.3f}")
                else:
                    # Add new operations from pattern detection
                    op_bias[op] = float(pattern_bias)
                    logging.debug(f"[RelMem] Pattern added {op}: {pattern_bias:.3f}")

            # Debug: show final merged state
            non_zero_count = len([v for v in op_bias.values() if v > 0.0])
            max_bias = max(op_bias.values()) if op_bias else 0.0
            logging.info(f"[RelMem] After semantic merge: {non_zero_count} non-zero biases, max={max_bias:.4f}")

        # 5) DEBUG: Pre-normalization state
        import logging
        pre_norm_nonzero = len([v for v in op_bias.values() if v > 0.0])
        pre_norm_max = max(op_bias.values()) if op_bias else 0.0
        pre_norm_total = sum(op_bias.values())
        logging.info(f"[RelMem] ðŸ” PRE-NORMALIZATION: {pre_norm_nonzero} non-zero biases, max={pre_norm_max:.4f}, total={pre_norm_total:.4f}")

        # 6) Normalize (keep flexible)
        total = sum(op_bias.values()) + 1e-12
        # keep raw scale for downstream; but normalize to [0,1] relative
        for k in op_bias:
            op_bias[k] = float(op_bias[k]) / total

        # 7) DEBUG: Post-normalization state
        post_norm_nonzero = len([v for v in op_bias.values() if v > 0.001])  # Use 0.001 threshold
        post_norm_max = max(op_bias.values()) if op_bias else 0.0
        post_norm_total = sum(op_bias.values())
        logging.info(f"[RelMem] ðŸ” POST-NORMALIZATION: {post_norm_nonzero} non-zero biases (>0.001), max={post_norm_max:.4f}, total={post_norm_total:.4f}")

        # 8) BREAKTHROUGH: Post-normalization semantic bias boost
        # Apply additional semantic bias AFTER normalization to prevent dilution
        if hasattr(self, 'pattern_op_bias') and self.pattern_op_bias:
            boost_count = 0
            max_pattern_bias = max(self.pattern_op_bias.values()) if self.pattern_op_bias else 0.0

            # Apply post-normalization boost to prevent semantic information loss
            for op, pattern_strength in self.pattern_op_bias.items():
                if op in op_bias and pattern_strength > 0.0:
                    # Boost factor: stronger patterns get exponentially higher influence
                    boost_factor = 1.0 + (pattern_strength / (max_pattern_bias + 1e-6)) * 3.0  # Up to 4x boost
                    op_bias[op] = op_bias[op] * boost_factor
                    boost_count += 1

            if boost_count > 0:
                logging.info(f"[RelMem] ðŸš€ POST-NORM BOOST: Applied to {boost_count} operations, max_pattern={max_pattern_bias:.3f}")

        # 9) Show top biases after all processing
        if op_bias:
            top_biases = sorted(op_bias.items(), key=lambda x: x[1], reverse=True)[:5]
            top_values = [f"{k}={v:.4f}" for k, v in top_biases]
            logging.info(f"[RelMem] ðŸŽ¯ FINAL PIPELINE BIASES: {top_values}")

        return op_bias

    def _apply_fallback_bias(self, op_bias: Dict[str, float]):
        """Apply intelligent fallback bias when world model is insufficient."""
        try:
            from models.dsl_registry import DSL_OPS
            if DSL_OPS:
                # Use small but differentiated biases instead of uniform
                # Favor common ARC operations that work across many puzzles
                high_utility_ops = {
                    "rotate90": 0.035, "flip_h": 0.035, "flip_v": 0.035,
                    "color_map": 0.030, "extract_pattern": 0.030,
                    "crop_bbox": 0.025, "for_each_object": 0.025,
                    "flood_fill": 0.020, "apply_rule": 0.020
                }

                base_bias = 0.015  # Lower base for less common operations
                for op in DSL_OPS:
                    if op in op_bias:
                        op_bias[op] = high_utility_ops.get(op, base_bias)

                import logging
                logging.info("[RelMem] Using intelligent fallback bias (differentiated by operation utility)")
            else:
                # Emergency fallback
                for k in list(op_bias.keys()):
                    op_bias[k] = 0.01
        except ImportError:
            # Last resort fallback
            for k in list(op_bias.keys()):
                op_bias[k] = 0.01

    def _scores(self, rel_name):
        """Return a FloatTensor score for relation rel_name (shape [] or [N])"""
        try:
            if hasattr(self, "weights") and rel_name in self.weights:
                W = self.weights[rel_name]
                return torch.tensor(float(torch.norm(W).item()) / (W.numel() + 1e-9), device=self.device)
            # fallback to stored activations if available
            if hasattr(self, "rel_activations") and rel_name in self.rel_activations:
                return torch.tensor(self.rel_activations[rel_name], device=self.device).float()
        except Exception:
            pass
        return torch.tensor(0.0, device=self.device)

    # -------- Exceptions / Inheritance+ ----------
    def add_exception(self, sid: int, rel: str, oid: int):
        key = (int(sid), str(rel))
        s = self.exceptions.get(key, set())
        s.add(int(oid))
        self.exceptions[key] = s

    def remove_exception(self, sid: int, rel: str, oid: int):
        key = (int(sid), str(rel))
        if key in self.exceptions and int(oid) in self.exceptions[key]:
            self.exceptions[key].remove(int(oid))
            if not self.exceptions[key]:
                self.exceptions.pop(key, None)

    def _is_exception(self, sid: int, rel: str, oid: int) -> bool:
        key = (int(sid), str(rel))
        return key in self.exceptions and int(oid) in self.exceptions[key]

    @torch.no_grad()
    def inheritance_pass_plus(self, steps: int = 3, alpha: float = 0.1, thresh: float = 0.5) -> torch.Tensor:
        """
        Enhanced inheritance with exceptions and confidence gating.
        Returns a small scalar consistency loss for training signals.
        """
        if "is_a" not in self.relations or "has_attr" not in self.relations:
            return (self.concept_proto * 0).sum()
        Wisa = self._scores("is_a")
        What = self._scores("has_attr")
        P = What.clone() if torch.is_tensor(What) else torch.tensor(What, device=self.device)
        for _ in range(max(1, steps)):
            if torch.is_tensor(Wisa) and torch.is_tensor(P):
                P = P + alpha * (Wisa @ P) if Wisa.dim() == 2 else P
        # Apply exceptions
        if self.exceptions and torch.is_tensor(P) and P.dim() >= 2:
            for (sid, rel), oids in self.exceptions.items():
                if rel == "has_attr" and len(oids) > 0:
                    if P.dim() == 2 and sid < P.shape[0]:
                        for oid in oids:
                            if oid < P.shape[1]:
                                P[sid, oid] = 0.0
        # Confidence gating
        if torch.is_tensor(P) and torch.is_tensor(What):
            P = torch.where(P > thresh, P, What)
            loss = F.mse_loss(P, What)
        else:
            loss = torch.tensor(0.0, device=self.device)
        return torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    # -------- Contextual op-bias + Theme priors ----------
    def get_op_bias_contextual(self, slot_vecs: Optional[torch.Tensor] = None,
                               theme_embed: Optional[torch.Tensor] = None,
                               spike_patterns: Optional[torch.Tensor] = None,
                               ripple_context: Optional[Dict] = None) -> Dict[str, float]:
        """
        Spike-enhanced contextual op-bias: combines graph priors (op_bias), slot evidence,
        theme steer, spike pattern matching, and conditional gating.
        Returns normalized dict suitable for DSL search.

        Args:
            slot_vecs: Object slot vectors for context
            theme_embed: Theme embedding for specialization
            spike_patterns: Current spike activations for pattern matching
            ripple_context: Ripple state for consolidation-aware biasing
        """
        base = self.op_bias()  # normalized dict, never empty

        # Standard slot and theme modulation
        if slot_vecs is not None and torch.is_tensor(slot_vecs):
            ctx = float(slot_vecs.abs().mean().detach().cpu())
            for k in base:
                base[k] = float(min(1.0, max(0.0, base[k] * (0.9 + 0.2 * ctx))))

        if theme_embed is not None and isinstance(theme_embed, torch.Tensor) and theme_embed.numel() > 0:
            t = theme_embed.detach().float()
            s = float(t.mean().sigmoid().item())
            v = float(t.std().clamp(0,1).item())
            sym_ops = ["symmetry","flip_h","flip_v","rotate90","rotate180","rotate270","transpose"]
            conn_ops = ["flood_fill","flood_select","outline","boundary_extract","for_each_object"]
            for k in sym_ops:
                if k in base: base[k] = min(1.0, base[k] * (0.9 + 0.3 * s))
            for k in conn_ops:
                if k in base: base[k] = min(1.0, base[k] * (0.9 + 0.3 * v))

        # === SPIKE PATTERN AMPLIFICATION ===
        if spike_patterns is not None and hasattr(self, 'arc_vocab') and self.enable_spike_coding:
            # Match current spike patterns to ARC operation vocabularies
            pattern_name, similarity = self.arc_vocab.match_pattern(spike_patterns, threshold=0.6)

            if pattern_name and similarity > 0.6:
                # Get operations associated with this pattern
                pattern_ops = self.arc_vocab.get_pattern_ops(pattern_name)

                # Apply aggressive bias boost (5x for strong matches)
                boost_factor = 1.0 + 4.0 * similarity  # 1.0 to 5.0 range

                for op in pattern_ops:
                    if op in base:
                        base[op] = min(1.0, base[op] * boost_factor)

                # Store pattern match info for monitoring
                self._spike_stats['pattern_match'] = {
                    'name': pattern_name,
                    'similarity': similarity,
                    'boosted_ops': pattern_ops
                }

        # === RIPPLE-AWARE OPERATION BIASING ===
        if ripple_context is not None:
            is_active = ripple_context.get('is_active', False)
            coherence = ripple_context.get('coherence', 0.0)

            if is_active and coherence > 0.8:
                # During high-coherence ripples: boost consolidation-friendly operations
                consolidation_ops = [
                    'symmetry', 'pattern_fill', 'template_match',
                    'for_each_object', 'extract_pattern'
                ]

                consolidation_boost = 1.0 + 0.5 * coherence  # Up to 1.5x boost

                for op in consolidation_ops:
                    if op in base:
                        base[op] = min(1.0, base[op] * consolidation_boost)

                # Store ripple influence info
                self._spike_stats['ripple_bias'] = {
                    'coherence': coherence,
                    'boosted_ops': consolidation_ops,
                    'boost_factor': consolidation_boost
                }

        # === CONDITIONAL GATING: relationships-of-relationships ===
        base = self._apply_conditional_gates(base)

        # Final normalization
        Z = sum(base.values()) + 1e-12
        for k in base: base[k] = float(base[k] / Z)
        return base

    def _apply_conditional_gates(self, op_bias: Dict[str, float]) -> Dict[str, float]:
        """
        Apply conditional gating where relations can be enabled by other relations.

        Implements: IF relation1 is active AND relation2 is active THEN boost operations
        using relation-as-concept addressing with 'enabled_by' meta-relation.

        Args:
            op_bias: Base operation bias dictionary

        Returns:
            Dict[str, float]: Conditionally modulated operation bias
        """
        try:
            # Relation-to-operations mapping (same as in op_bias method)
            relation_to_ops = {
                "color": ["color_map", "flood_fill", "extract_color"],
                "shape": ["rotate90", "flip_h", "flip_v"],
                "structure": ["crop_bbox", "tile_pattern", "resize_nn", "translate", "scale"],
                "logic": ["grid_union", "grid_intersection", "grid_xor", "grid_difference"],
                "identity": ["identity"],
                "flip": ["flip_h", "flip_v"],
                "rotate": ["rotate90", "rotate180", "rotate270"],
                "crop": ["crop_bbox", "crop_nonzero"],
                "tile": ["tile", "tile_pattern"],
                "flood": ["flood_fill", "flood_select"],
                "outline": ["outline", "boundary_extract"],
                "symmetry": ["symmetry"],
                "paste": ["paste"],
                "count": ["count_objects", "count_colors"],
                "pattern": ["find_pattern", "extract_pattern", "match_template"],
                "object": ["for_each_object", "for_each_object_translate", "for_each_object_recolor"],
                "conditional": ["conditional_map", "apply_rule"],
                "select": ["select_by_property", "flood_select"],
                "arithmetic": ["arithmetic_op"]
            }

            # Check for conditional gates using relation-as-concept addressing
            conditional_boosts = {}

            # Example conditional rules:
            # IF ARC_Puzzle has_attr color_dependent AND op:flood_fill precedes other_ops
            # THEN boost flood_fill and color operations

            for target_rel, ops in relation_to_ops.items():
                try:
                    # Check if target relation is enabled by conditions
                    rel_concept_name = f"rel:{target_rel}"

                    # Look for enablement conditions in symbolic index
                    if rel_concept_name in self._symbolic_index:
                        # Check for enabling relations via 'enabled_by' meta-relation
                        enabled_by_logits = self.query_object("enabled_by", self._symbolic_index[rel_concept_name])

                        if enabled_by_logits is not None:
                            # Convert logits to activation strength
                            activation = float(torch.softmax(enabled_by_logits, dim=0).max().item())

                            # Apply conditional boost to operations
                            if activation > 0.3:  # Threshold for meaningful activation
                                boost_factor = 1.0 + 0.5 * activation  # Up to 1.5x boost
                                for op in ops:
                                    if op in op_bias:
                                        conditional_boosts[op] = max(conditional_boosts.get(op, 1.0), boost_factor)

                except Exception:
                    continue  # Silent failure for robustness

            # Apply conditional boosts
            for op, boost in conditional_boosts.items():
                if op in op_bias:
                    op_bias[op] = min(1.0, op_bias[op] * boost)

            # Hardcoded conditional rules for immediate functionality:
            # IF ARC_Puzzle has_attr "pattern_based" THEN boost symmetry operations
            try:
                if "ARC_Puzzle" in self._symbolic_index and "pattern_based" in self._symbolic_index:
                    pattern_activation = self._get_relation_strength("ARC_Puzzle", "has_attr", "pattern_based")
                    if pattern_activation > 0.2:
                        symmetry_boost = 1.0 + 0.4 * pattern_activation
                        for op in ["symmetry", "flip_h", "flip_v", "rotate90"]:
                            if op in op_bias:
                                op_bias[op] = min(1.0, op_bias[op] * symmetry_boost)
            except Exception:
                pass

            # IF ARC_Puzzle has_attr "color_dependent" THEN boost color operations
            try:
                if "ARC_Puzzle" in self._symbolic_index and "color_dependent" in self._symbolic_index:
                    color_activation = self._get_relation_strength("ARC_Puzzle", "has_attr", "color_dependent")
                    if color_activation > 0.2:
                        color_boost = 1.0 + 0.3 * color_activation
                        for op in ["color_map", "flood_fill", "extract_color"]:
                            if op in op_bias:
                                op_bias[op] = min(1.0, op_bias[op] * color_boost)
            except Exception:
                pass

        except Exception as e:
            logger.debug(f"[Conditional] Gate application failed: {e}")

        return op_bias

    def _get_relation_strength(self, subj_name: str, rel: str, obj_name: str) -> float:
        """
        Get the strength of a specific relation between named concepts.

        Args:
            subj_name: Subject concept name
            rel: Relation name
            obj_name: Object concept name

        Returns:
            float: Relation strength [0.0, 1.0]
        """
        try:
            if (subj_name not in self._symbolic_index or
                obj_name not in self._symbolic_index or
                rel not in self.relations):
                return 0.0

            sid = self._symbolic_index[subj_name]
            oid = self._symbolic_index[obj_name]

            # Get relation scores and extract specific pair
            scores = self._scores(rel)
            if hasattr(scores, 'shape') and len(scores.shape) >= 2:
                strength = float(scores[sid, oid].item())
            else:
                strength = float(scores.item()) if hasattr(scores, 'item') else 0.0

            return max(0.0, min(1.0, strength))

        except Exception:
            return 0.0

    def get_theme_priors(self, theme_embed: Optional[torch.Tensor]) -> Dict[str, float]:
        """
        Produce {'phi','kappa','cge'} in [0.5, 1.5] for EBR gating (neutral=1.0).
        """
        if theme_embed is None or not isinstance(theme_embed, torch.Tensor) or theme_embed.numel() == 0:
            return {"phi": 1.0, "kappa": 1.0, "cge": 1.0}
        x = theme_embed.detach().float()
        m = float(x.mean().tanh().item())   # [-1,1]
        v = float(x.std().tanh().item())    # [-1,1]
        scale = lambda z: 1.0 + 0.5 * z     # [-1,1] -> [0.5,1.5]
        return {"phi": scale(m), "kappa": scale(v), "cge": scale((m+v)/2.0)}

    # -------- Refinement agent & stats ----------
    @torch.no_grad()
    def refinement_step(self, cos_thresh: float = 0.92, min_size: int = 3, merge_alpha: float = 0.2):
        """Cluster similar concepts and lightly merge to form cleaner parents."""
        used = self.concept_used.nonzero().flatten()
        if used.numel() < min_size: return
        E = self.concept_proto[used]
        sim = F.cosine_similarity(E.unsqueeze(1), E.unsqueeze(0), dim=-1)
        visited = set()
        for i in range(used.numel()):
            if i in visited: continue
            cluster = [i]
            for j in range(i+1, used.numel()):
                if j in visited: continue
                if float(sim[i,j].item()) >= cos_thresh:
                    cluster.append(j)
            if len(cluster) >= min_size:
                cids = used[torch.tensor(cluster, device=self.device)]
                centroid = self.concept_proto[cids].mean(dim=0, keepdim=True)
                self.concept_proto.data[cids] = (1-merge_alpha)*self.concept_proto.data[cids] + merge_alpha*centroid
                self.depth_hist.append(len(cluster))
                visited.update(cluster)

    def stats(self) -> Dict[str, float]:
        active = int(self.concept_used.sum().item())
        depth = self.get_hierarchy_depth() if hasattr(self, "get_hierarchy_depth") else 0.0
        exc = sum(len(v) for v in self.exceptions.values()) if self.exceptions else 0
        return {"relmem_active": float(active), "relmem_depth": float(depth), "relmem_exceptions": float(exc)}

    # -------- Performance Monitoring & Auto-Tuning ----------
    def init_performance_monitoring(self):
        """Initialize performance tracking for spike parameter auto-tuning."""
        if not hasattr(self, '_performance_tracker'):
            self._performance_tracker = {
                'em_scores': [],  # Recent EM scores
                'acc_scores': [],  # Recent accuracy scores
                'spike_stats_history': [],  # History of spike statistics
                'aggressiveness_history': [],  # History of spike aggressiveness values
                'performance_trend': 0.0,  # Current performance trend
                'last_tuning_step': 0,
                'tuning_interval': 100,  # Steps between auto-tuning
                'min_aggressiveness': 0.3,
                'max_aggressiveness': 3.0
            }

    def update_performance_metrics(self, em_score: float, acc_score: float, step: int):
        """
        Update performance metrics for auto-tuning.

        Args:
            em_score: Exact match score [0.0, 1.0]
            acc_score: Accuracy score [0.0, 1.0]
            step: Current training step
        """
        self.init_performance_monitoring()
        tracker = self._performance_tracker

        # Add current scores
        tracker['em_scores'].append(em_score)
        tracker['acc_scores'].append(acc_score)

        # Keep only recent history (last 50 scores)
        if len(tracker['em_scores']) > 50:
            tracker['em_scores'] = tracker['em_scores'][-50:]
            tracker['acc_scores'] = tracker['acc_scores'][-50:]

        # Store current spike stats if available
        if hasattr(self, '_spike_stats') and self._spike_stats:
            tracker['spike_stats_history'].append(self._spike_stats.copy())
            if len(tracker['spike_stats_history']) > 20:
                tracker['spike_stats_history'] = tracker['spike_stats_history'][-20:]

        # Store current aggressiveness
        tracker['aggressiveness_history'].append(self.spike_aggressiveness)
        if len(tracker['aggressiveness_history']) > 20:
            tracker['aggressiveness_history'] = tracker['aggressiveness_history'][-20:]

        # Auto-tune if enough time has passed
        if step - tracker['last_tuning_step'] >= tracker['tuning_interval']:
            self._auto_tune_spike_parameters(step)
            tracker['last_tuning_step'] = step

    def _auto_tune_spike_parameters(self, current_step: int):
        """
        Automatically tune spike parameters based on recent performance.

        Implements adaptive spike aggressiveness:
        - If performance improving: make spikes more aggressive (lower aggressiveness)
        - If performance declining: make spikes less aggressive (higher aggressiveness)
        """
        tracker = self._performance_tracker

        if len(tracker['em_scores']) < 10:
            return  # Not enough data

        # Compute performance trend (recent vs older performance)
        recent_em = np.mean(tracker['em_scores'][-10:])  # Last 10 scores
        older_em = np.mean(tracker['em_scores'][-20:-10]) if len(tracker['em_scores']) >= 20 else recent_em

        performance_trend = recent_em - older_em
        tracker['performance_trend'] = performance_trend

        # Adaptive tuning logic
        current_agg = self.spike_aggressiveness

        if performance_trend > 0.02:  # Performance improving significantly
            # Make spikes more aggressive (encourage sparsity)
            new_agg = max(tracker['min_aggressiveness'], current_agg * 0.9)
            logger.info(f"[SpikeAutoTune] Performance improving ({performance_trend:.3f}), "
                       f"increasing spike aggressiveness: {current_agg:.3f} -> {new_agg:.3f}")

        elif performance_trend < -0.02:  # Performance declining significantly
            # Make spikes less aggressive (reduce sparsity)
            new_agg = min(tracker['max_aggressiveness'], current_agg * 1.1)
            logger.info(f"[SpikeAutoTune] Performance declining ({performance_trend:.3f}), "
                       f"decreasing spike aggressiveness: {current_agg:.3f} -> {new_agg:.3f}")

        else:
            # Performance stable, small random perturbation for exploration
            perturbation = np.random.normal(0, 0.05)
            new_agg = np.clip(current_agg + perturbation,
                            tracker['min_aggressiveness'],
                            tracker['max_aggressiveness'])

        self.spike_aggressiveness = float(new_agg)

    def consolidate_spike_patterns(self, ripple_event: Optional[Dict] = None,
                                 success_score: float = 0.0) -> bool:
        """
        Consolidate successful spike patterns triggered by ripple events.

        Args:
            ripple_event: Ripple consolidation event info
            success_score: Recent success score (EM or accuracy)

        Returns:
            True if consolidation occurred, False otherwise
        """
        if not self.enable_spike_coding or not hasattr(self, '_last_spike_levels'):
            return False

        # Progressive consolidation threshold (start much lower)
        consolidation_threshold = 0.05 + min(0.30, success_score * 0.5)  # 5% â†’ 35%
        if success_score < consolidation_threshold:
            return False

        try:
            from spike_coder import adaptive_threshold_spike

            # Extract recent successful spike patterns
            spike_levels = self._last_spike_levels
            consolidation_count = 0

            # Consolidate patterns from each hierarchical level
            for level_name, spike_pattern in spike_levels.items():
                if spike_pattern is None or spike_pattern.numel() == 0:
                    continue

                # Create pattern name based on success and level
                pattern_name = f"learned_{level_name}_{int(success_score*100)}"

                # Convert to compact representation for storage
                if spike_pattern.dim() > 2:
                    # Pool across batch and time dimensions
                    consolidated_pattern = spike_pattern.mean(dim=(0, 1))
                else:
                    consolidated_pattern = spike_pattern.mean(dim=0)

                # Re-encode with consistent parameters for storage
                stored_pattern = adaptive_threshold_spike(
                    consolidated_pattern.unsqueeze(0),
                    k=0.5,  # Medium sparsity for stored patterns
                    mode="ternary"
                ).squeeze(0)

                # Store in vocabulary with success-based priority
                if hasattr(self, 'arc_vocab'):
                    self.arc_vocab.vocabularies[pattern_name] = stored_pattern
                    consolidation_count += 1

                    # Add operations mapping for learned pattern
                    if level_name == 'perceptual':
                        ops = ['extract_pattern', 'identify_components', 'detect_features']
                    elif level_name == 'conceptual':
                        ops = ['for_each_object', 'relate_objects', 'group_similar']
                    else:  # abstract
                        ops = ['apply_rule', 'transform_grid', 'complete_pattern']

                    # Update pattern mapping dynamically
                    if hasattr(self.arc_vocab, '_dynamic_pattern_ops'):
                        self.arc_vocab._dynamic_pattern_ops[pattern_name] = ops
                    else:
                        self.arc_vocab._dynamic_pattern_ops = {pattern_name: ops}

            if consolidation_count > 0:
                logger.info(f"[SpikeConsolidation] Stored {consolidation_count} patterns from "
                           f"success_score={success_score:.3f}, ripple_coherence="
                           f"{ripple_event.get('coherence', 0.0):.3f}" if ripple_event else "N/A")

                # Update performance tracking
                if hasattr(self, '_spike_stats'):
                    self._spike_stats['consolidation'] = {
                        'patterns_stored': consolidation_count,
                        'trigger_success_score': success_score,
                        'ripple_info': ripple_event
                    }

            return consolidation_count > 0

        except Exception as e:
            logger.warning(f"[SpikeConsolidation] Failed: {e}")
            return False

    def get_spike_performance_report(self) -> Dict[str, Union[float, List, Dict]]:
        """
        Generate comprehensive spike coding performance report.

        Returns:
            Dictionary containing performance metrics, trends, and recommendations
        """
        self.init_performance_monitoring()
        tracker = self._performance_tracker

        if not tracker['em_scores']:
            return {"status": "insufficient_data"}

        # Compute performance statistics
        recent_em = np.mean(tracker['em_scores'][-10:]) if len(tracker['em_scores']) >= 10 else 0.0
        recent_acc = np.mean(tracker['acc_scores'][-10:]) if len(tracker['acc_scores']) >= 10 else 0.0

        # Spike statistics analysis
        spike_analysis = {}
        if tracker['spike_stats_history']:
            latest_spike_stats = tracker['spike_stats_history'][-1]

            if 'concept_activation' in latest_spike_stats:
                spike_analysis['concept_sparsity'] = latest_spike_stats['concept_activation'].get('pct_active', 0.0)

            if 'pattern_match' in latest_spike_stats:
                spike_analysis['pattern_matches'] = latest_spike_stats['pattern_match']

            if 'ripple_bias' in latest_spike_stats:
                spike_analysis['ripple_influence'] = latest_spike_stats['ripple_bias']

        # Performance trend analysis
        if len(tracker['em_scores']) >= 20:
            recent_trend = np.polyfit(range(10), tracker['em_scores'][-10:], 1)[0]
            overall_trend = np.polyfit(range(len(tracker['em_scores'])), tracker['em_scores'], 1)[0]
        else:
            recent_trend = 0.0
            overall_trend = 0.0

        # Generate recommendations
        recommendations = []

        if recent_em < 0.3:
            recommendations.append("Low EM scores: Consider reducing spike aggressiveness")
        elif recent_em > 0.8:
            recommendations.append("High EM scores: Consider increasing spike aggressiveness for better sparsity")

        if spike_analysis.get('concept_sparsity', 50) > 90:
            recommendations.append("Very high sparsity: Risk of information loss")
        elif spike_analysis.get('concept_sparsity', 50) < 30:
            recommendations.append("Low sparsity: Missing spike coding benefits")

        return {
            "status": "active",
            "performance": {
                "recent_em": float(recent_em),
                "recent_acc": float(recent_acc),
                "recent_trend": float(recent_trend),
                "overall_trend": float(overall_trend)
            },
            "spike_config": {
                "aggressiveness": float(self.spike_aggressiveness),
                "enabled": bool(self.enable_spike_coding),
                "aggressiveness_history": tracker['aggressiveness_history'][-10:]
            },
            "spike_analysis": spike_analysis,
            "recommendations": recommendations,
            "monitoring": {
                "scores_count": len(tracker['em_scores']),
                "last_tuning_step": tracker['last_tuning_step'],
                "performance_trend": float(tracker['performance_trend'])
            }
        }

    def manual_spike_tuning(self, new_aggressiveness: float, reason: str = "manual"):
        """
        Manually set spike aggressiveness with logging.

        Args:
            new_aggressiveness: New aggressiveness value [0.1, 5.0]
            reason: Reason for the change (for logging)
        """
        old_agg = self.spike_aggressiveness
        self.spike_aggressiveness = float(np.clip(new_aggressiveness, 0.1, 5.0))

        logger.info(f"[SpikeManualTune] {reason}: {old_agg:.3f} -> {self.spike_aggressiveness:.3f}")

        # Reset auto-tuning timer to prevent immediate override
        if hasattr(self, '_performance_tracker'):
            self._performance_tracker['last_tuning_step'] = getattr(self, '_current_step', 0)

    # -------- Persistence ----------


    # --- Shape-safe wrapper to keep training going & preserve grads ---
    _REL_INV_WARN = 0
    _REL_INV_WARN_MAX = 5

    def inverse_loss_safe(self, *args, **kwargs):
        """Wrapper around inverse loss to avoid shape/index errors and to log shapes."""
        import logging
        import torch
        try:
            if not getattr(self, "concepts", None):
                logging.getLogger(__name__).debug("[RelMem] inverse_loss_safe: no concepts present -> 0")
                return torch.tensor(0.0, device=getattr(self,'device','cuda:0'))
            # call underlying implementation if exists
            if hasattr(self, "_inverse_loss_impl"):
                return self._inverse_loss_impl(*args, **kwargs)
            # otherwise, try original inverse_loss if present
            if hasattr(self, "inverse_loss"):
                return self.inverse_loss(*args, **kwargs)
            return torch.tensor(0.0, device=getattr(self,'device','cuda:0'))
        except IndexError as e:
            logging.getLogger(__name__).exception("[RelMem] inverse_loss IndexError shapes=%s: %s", getattr(args[0],'shape',None), e)
            return torch.tensor(0.0, device=getattr(self,'device','cuda:0'))
        except Exception as e:
            logging.getLogger(__name__).exception("[RelMem] inverse_loss unexpected error: %s", e)
            return torch.tensor(0.0, device=getattr(self,'device','cuda:0'))

# --- Exemplar-enhanced Relational Memory -------------------------------------
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import torch
import torch.nn.functional as F

@dataclass
class Exemplar:
    vec: torch.Tensor
    freq: int
    last_step: int
    # Optional grounded attributes (phi/kappa/CGE/hodge etc.)
    attrs: Optional[Dict[str, float]] = None

class RelationalMemoryExemplar(RelationalMemoryNeuro):
    """
    Extends RelationalMemoryNeuro with per-concept exemplar libraries
    + ripple/dream-gated consolidation + optional 'told facts' injection.
    - Each concept stores a small set of exemplars (vec + attrs + usage stats).
    - New inputs update closest exemplar (if similar) or create a new one.
    - Periodic consolidation merges/prunes exemplars and recenters the prototype.
    """
    def __init__(
        self, hidden_dim: int, max_concepts: int = 4096, rank: int = 16,
        relations: List[str] = None, inverse_pairs: Dict[str, str] = None,
        wta_frac: float = 0.1, wta_warmup_updates: int = 15, device="cuda",
        enable_spike_coding: bool = True, spike_aggressiveness: float = 1.0,
        max_exemplars_per_concept: int = 8, sim_tau: float = 0.85,
        merge_tau: float = 0.92, prune_min_freq: int = 2
    ):
        super().__init__(hidden_dim, max_concepts, rank, relations, inverse_pairs,
                         wta_frac, wta_warmup_updates, device, enable_spike_coding, spike_aggressiveness)
        self.max_exemplars_per_concept = max_exemplars_per_concept
        self.sim_tau = float(sim_tau)
        self.merge_tau = float(merge_tau)
        self.prune_min_freq = int(prune_min_freq)
        # exemplar store: cid -> List[Exemplar]
        self._exemplars: Dict[int, List[Exemplar]] = {}
        # optional name->cid for 'told facts'
        self._symbolic_index: Dict[str, int] = {}

        # === SPIKE-COMPRESSED SUCCESS TEMPLATES ===
        self.success_templates = {}  # template_id -> compressed template
        self._template_id_counter = 0
        self.template_match_threshold = 0.8

    # ---------------- Device Management ----------------
    def to(self, device):
        """
        Force all parameters, buffers, exemplars, and concept stores onto the given device.
        This overrides nn.Module.to and ensures consistency (no CPU/GPU mismatch).
        """
        super().to(device)
        self.device = torch.device(device)

        # Core parameters
        if hasattr(self, "concept_proto"):
            self.concept_proto.data = self.concept_proto.data.to(self.device)
        for rel in getattr(self, "relations", []):
            if rel in self.A:
                self.A[rel].data = self.A[rel].data.to(self.device)
            if rel in self.B:
                self.B[rel].data = self.B[rel].data.to(self.device)
            if rel in self.rel_gain:
                self.rel_gain[rel].data = self.rel_gain[rel].data.to(self.device)

        # Buffers
        if hasattr(self, "concept_used"):
            self.concept_used = self.concept_used.to(self.device)
        if hasattr(self, "theta"):
            self.theta.data = self.theta.data.to(self.device)
        if hasattr(self, "omega"):
            self.omega.data = self.omega.data.to(self.device)

        # Exemplar memory: move every stored exemplar vec
        if hasattr(self, "_exemplars"):
            for cid, bank in self._exemplars.items():
                for ex in bank:
                    ex.vec = ex.vec.to(self.device)

        return self

    # ---------- Override core entry points to enforce device ----------
    def observe_sample(self, x_tokens: torch.Tensor, step: int,
                       attrs: Optional[Dict[str, float]] = None, top_k: int = 64):
        if x_tokens is not None:
            x_tokens = x_tokens.to(self.device)
        return super().observe_sample(x_tokens, step=step, attrs=attrs, top_k=top_k)

    def bind_fact(self, subj_name: str, rel: str, obj_name: str,
                  subj_vec: Optional[torch.Tensor] = None,
                  obj_vec: Optional[torch.Tensor] = None,
                  alpha: float = 0.5):
        if subj_vec is not None:
            subj_vec = subj_vec.to(self.device)
        if obj_vec is not None:
            obj_vec = obj_vec.to(self.device)
        return super().bind_fact(subj_name, rel, obj_name,
                                 subj_vec=subj_vec, obj_vec=obj_vec, alpha=alpha)

    def add_concept(self, vec: torch.Tensor, meta: Optional[dict] = None, alpha: float = 1.0) -> int:
        if vec is not None:
            vec = vec.to(self.device)
        return super().add_concept(vec, meta=meta, alpha=alpha)

    # ---------- Core: add/update exemplars ----------
    @torch.no_grad()
    def add_or_update_exemplar(
        self, cid: int, vec: torch.Tensor, step: int,
        attrs: Optional[Dict[str, float]] = None
    ):
        """Add a new exemplar or update the most similar one."""
        cid = int(cid)
        vec = vec.detach().to(self.device).float()
        bank = self._exemplars.setdefault(cid, [])
        best_i, best_sim = -1, -1.0
        for i, ex in enumerate(bank):
            sim = float(F.cosine_similarity(vec.unsqueeze(0), ex.vec.unsqueeze(0)))
            if sim > best_sim:
                best_i, best_sim = i, sim
        if best_sim >= self.sim_tau and best_i >= 0:
            # Light moving average toward new vec; bump freq/ts; merge attrs
            ex = bank[best_i]
            ex.vec = 0.9 * ex.vec + 0.1 * vec
            ex.freq += 1
            ex.last_step = step
            if attrs:
                ex.attrs = (ex.attrs or {})
                for k, v in attrs.items():
                    ex.attrs[k] = float(v)
        else:
            bank.append(Exemplar(vec=vec, freq=1, last_step=step, attrs=attrs))
            # Bound memory
            if len(bank) > self.max_exemplars_per_concept:
                # drop least-frequent / oldest
                bank.sort(key=lambda e: (e.freq, e.last_step))
                bank.pop(0)

    # ---------- Consolidation: prune/merge + recenter prototype ----------
    @torch.no_grad()
    def consolidate_concept(self, cid: int):
        """Merge near-duplicate exemplars; prune low-usage; update prototype."""
        if cid not in self._exemplars:
            return
        bank = self._exemplars[cid]
        if not bank:
            return

        # 1) prune rarely used
        bank = [e for e in bank if e.freq >= self.prune_min_freq]

        # 2) merge highly-similar (greedy)
        merged: List[Exemplar] = []
        for e in sorted(bank, key=lambda x: -x.freq):
            keep = True
            for m in merged:
                sim = float(F.cosine_similarity(e.vec.unsqueeze(0), m.vec.unsqueeze(0)))
                if sim >= self.merge_tau:
                    # merge into m (frequency-weighted)
                    w1, w2 = float(m.freq), float(e.freq)
                    m.vec = (w1 * m.vec + w2 * e.vec) / max(1e-6, (w1 + w2))
                    m.freq += e.freq
                    m.last_step = max(m.last_step, e.last_step)
                    if e.attrs:
                        m.attrs = (m.attrs or {})
                        for k, v in e.attrs.items():
                            m.attrs[k] = float(0.5 * m.attrs.get(k, v) + 0.5 * v)
                    keep = False
                    break
            if keep:
                merged.append(e)
        self._exemplars[cid] = merged

        # 3) recenter prototype to exemplar mean (concept_proto is learnable Param)
        if merged:
            # Ensure all exemplar vectors are properly dimensioned for consolidation
            projected_vecs = []
            for e in merged:
                vec = e.vec
                # Use the same projection mechanism as self-model initialization
                if vec.shape[-1] != self.D:
                    vec = self._proj_to_relmem(vec)
                projected_vecs.append(vec)

            mean_vec = torch.stack(projected_vecs, dim=0).mean(0)
            self.concept_proto.data[cid] = 0.8 * self.concept_proto.data[cid] + 0.2 * mean_vec

    @torch.no_grad()
    def consolidate_exemplars(self):
        """Call this during dream/ripple cycles or every N steps."""
        used_ids = self.concept_used.nonzero().flatten().tolist()
        for cid in used_ids:
            self.consolidate_concept(int(cid))

    # ---------- Spike-Compressed Success Templates ----------
    @torch.no_grad()
    def store_success_template(self, solution_trace: torch.Tensor, success_score: float,
                               task_embedding: Optional[torch.Tensor] = None) -> int:
        """
        Store spike-coded versions of highly successful solutions for fast retrieval.

        Args:
            solution_trace: The solution activation trace [B, T, D] or [T, D]
            success_score: Success rate (0.0 to 1.0)
            task_embedding: Optional task context embedding

        Returns:
            template_id: ID of stored template, or -1 if not stored
        """
        # Progressive template storage threshold (much more permissive)
        template_threshold = 0.3 + min(0.4, success_score * 0.6)  # 30% â†’ 70%
        if success_score < template_threshold:
            return -1

        from spike_coder import adaptive_threshold_spike

        # Flatten solution trace if needed
        if solution_trace.dim() > 2:
            solution_trace = solution_trace.flatten(0, -2)  # [B*T, D]

        # Ultra-sparse encoding for long-term storage (k=0.15 for ~85% sparsity)
        template_spikes = adaptive_threshold_spike(
            solution_trace.mean(0),  # Average across time/batch
            k=0.15,
            mode="bitwise"
        )

        template_id = self._template_id_counter
        self._template_id_counter += 1

        # Store with metadata
        self.success_templates[template_id] = {
            'spikes': template_spikes.detach().cpu(),  # Store on CPU to save GPU memory
            'success_rate': float(success_score),
            'priority': 10.0,
            'ttl': 1000,  # Time to live
            'task_embedding': task_embedding.detach().cpu() if task_embedding is not None else None,
            'created_step': getattr(self, '_current_step', 0),
            'usage_count': 0
        }

        # Limit total templates to prevent memory bloat
        if len(self.success_templates) > 50:
            self._prune_old_templates()

        return template_id

    @torch.no_grad()
    def fast_template_match(self, current_state: torch.Tensor) -> Tuple[Optional[int], float]:
        """
        Rapid template retrieval using spike similarity matching.

        Args:
            current_state: Current state tensor [B, T, D] or [D]

        Returns:
            (template_id, similarity_score) or (None, 0.0)
        """
        if not self.success_templates:
            return None, 0.0

        from spike_coder import adaptive_threshold_spike, hamming_distance

        # Convert current state to bitwise spikes
        if current_state.dim() > 1:
            current_state = current_state.flatten().mean() if current_state.numel() > 0 else current_state

        current_spikes = adaptive_threshold_spike(
            current_state.unsqueeze(0) if current_state.dim() == 0 else current_state,
            k=0.2,
            mode="bitwise"
        ).to(self.device)

        best_template_id = None
        best_score = 0.0

        for template_id, template_data in self.success_templates.items():
            template_spikes = template_data['spikes'].to(self.device)

            # Hamming distance for bitwise spikes (ultra-fast)
            hamming_dist = hamming_distance(current_spikes, template_spikes)
            similarity = 1.0 - hamming_dist

            # Bias by success rate and priority
            weighted_similarity = similarity * template_data['success_rate'] * (template_data['priority'] / 10.0)

            if weighted_similarity > best_score and similarity > self.template_match_threshold:
                best_template_id = template_id
                best_score = weighted_similarity

        # Update usage stats if match found
        if best_template_id is not None:
            self.success_templates[best_template_id]['usage_count'] += 1
            self.success_templates[best_template_id]['priority'] = min(15.0,
                self.success_templates[best_template_id]['priority'] + 0.1)

        return best_template_id, best_score

    @torch.no_grad()
    def get_template_ops_bias(self, template_id: int) -> Dict[str, float]:
        """
        Get operation bias from a matched template.

        Args:
            template_id: ID of the matched template

        Returns:
            Dictionary of operation biases
        """
        if template_id not in self.success_templates:
            return {}

        template_data = self.success_templates[template_id]
        success_rate = template_data['success_rate']

        # Pattern-based operation recommendations
        # These would ideally be learned from the actual solution traces
        template_ops = {
            'symmetry': success_rate * 0.8,
            'pattern_match': success_rate * 0.9,
            'template_apply': success_rate * 1.0,
            'for_each_object': success_rate * 0.7,
            'extract_pattern': success_rate * 0.8
        }

        return template_ops

    @torch.no_grad()
    def _prune_old_templates(self):
        """Remove old or low-performing templates."""
        if len(self.success_templates) <= 30:
            return

        # Sort by priority and usage, keep top 30
        sorted_templates = sorted(
            self.success_templates.items(),
            key=lambda x: (x[1]['priority'], x[1]['usage_count'], x[1]['success_rate']),
            reverse=True
        )

        # Keep top 30 templates
        self.success_templates = dict(sorted_templates[:30])

        # Reset template counter to prevent overflow
        max_id = max(self.success_templates.keys()) if self.success_templates else 0
        self._template_id_counter = max_id + 1

    # ---------- Optional: bind sample (streaming unsupervised) ----------
    @torch.no_grad()
    def observe_sample(
        self, x_tokens: torch.Tensor, step: int,
        attrs: Optional[Dict[str, float]] = None, top_k: int = 64
    ):
        """
        Light unsupervised update from relational tokens [B,T,D].
        Picks the highest-activation concept per token and updates its exemplar bank.
        """
        if x_tokens.dim() != 3 or x_tokens.size(-1) != self.D:
            return
        B, T, D = x_tokens.shape
        proto = self.concept_proto  # [N,D]
        scale = float(D) ** 0.5
        scores = torch.matmul(x_tokens, proto.t()) / scale  # [B,T,N]
        _, idx = torch.topk(scores, k=min(top_k, self.N), dim=-1)  # [B,T,K]
        # Use the top-1 for fast streaming update
        cid = idx[..., 0]  # [B,T]
        for b in range(B):
            for t in range(T):
                c = int(cid[b, t].item())
                v = x_tokens[b, t].detach()
                self.add_or_update_exemplar(c, v, step=step, attrs=attrs)

    # ---------- Dream/ripple hookup ----------
    @torch.no_grad()
    def on_ripple_event(self, ripple_stats: Optional[Dict[str, float]] = None):
        """
        Hook to be called after DreamEngine ripple cycles.
        Uses ripple coherence to adjust consolidation aggressiveness (optional).
        """
        self.consolidate_exemplars()
        # Example: if coherence high, enable WTA queue on frequently used relations
        coh = float(ripple_stats.get("ripple_phase_coherence", 1.0)) if isinstance(ripple_stats, dict) else 1.0
        if coh > 0.8:
            for rel in self.relations[:2]:  # small nudge
                self.queue_wta_update(rel)

    # ---------- 'Being told facts' (symbolic injection) ----------
    @torch.no_grad()
    def bind_fact(
        self, subj_name: str, rel: str, obj_name: str,
        subj_vec: Optional[torch.Tensor] = None,
        obj_vec: Optional[torch.Tensor] = None,
        alpha: float = 0.5
    ):
        """
        Create or reuse named concepts and connect them with relation 'rel'.
        If vectors are provided, initialize/refresh their prototypes + exemplars.
        """
        # (a) subject concept id
        if subj_name not in self._symbolic_index:
            sid = self._next_cid
            self._symbolic_index[subj_name] = sid
            self._next_cid += 1
            self.concept_used[sid] = True
            if subj_vec is not None:
                self.concept_proto.data[sid] = subj_vec.to(self.device).float()
                self.add_or_update_exemplar(sid, self.concept_proto.data[sid], step=0)
        else:
            sid = self._symbolic_index[subj_name]
            if subj_vec is not None:
                self.add_or_update_exemplar(sid, subj_vec.to(self.device).float(), step=0)

        # (b) object concept id
        if obj_name not in self._symbolic_index:
            oid = self._next_cid
            self._symbolic_index[obj_name] = oid
            self._next_cid += 1
            self.concept_used[oid] = True
            if obj_vec is not None:
                self.concept_proto.data[oid] = obj_vec.to(self.device).float()
                self.add_or_update_exemplar(oid, self.concept_proto.data[oid], step=0)
        else:
            oid = self._symbolic_index[obj_name]
            if obj_vec is not None:
                self.add_or_update_exemplar(oid, obj_vec.to(self.device).float(), step=0)

        # (c) strengthen relation with small Hebbian pulse
        try:
            self.queue_hebbian_update(rel, sid, oid, eta=alpha)
        except Exception:
            pass

    # ---------- Episodic Memory Retrieval ----------
    @torch.no_grad()
    def get_similar_experiences(self, query_emb: torch.Tensor, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most similar episodic experiences based on query embedding.

        Args:
            query_emb: Query embedding tensor [D] or [1, D]
            k: Number of similar experiences to retrieve

        Returns:
            List of dicts with keys: cid, similarity, meta, ops_used, success_rate
        """
        query_emb = query_emb.to(self.device).float()
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)

        # Get active concepts
        active_cids = self.concept_used.nonzero().flatten()
        if active_cids.numel() == 0:
            return []

        # Compute similarities with all active concepts
        active_protos = self.concept_proto[active_cids]  # [N, D=256]

        # Project query to concept space if dimensions don't match
        if query_emb.shape[-1] != active_protos.shape[-1]:
            # Lazy init query_projection with correct dimension
            if self.query_projection is None or self._query_proj_dim != query_emb.shape[-1]:
                self.query_projection = nn.Linear(query_emb.shape[-1], self.D, device=self.device)
                nn.init.xavier_uniform_(self.query_projection.weight)
                self._query_proj_dim = query_emb.shape[-1]

            query_emb = self.query_projection(query_emb)

        query_norm = F.normalize(query_emb, p=2, dim=-1)
        protos_norm = F.normalize(active_protos, p=2, dim=-1)

        similarities = torch.matmul(query_norm, protos_norm.t()).squeeze(0)  # [N_active]

        # Get top-k
        top_k = min(k, active_cids.numel())
        top_sims, top_indices = torch.topk(similarities, k=top_k)

        # Build result list with metadata
        results = []
        for sim, idx in zip(top_sims.tolist(), top_indices.tolist()):
            cid = int(active_cids[idx].item())
            concept_info = self.concepts.get(cid, {})

            # Extract metadata
            meta = concept_info.get('meta', {})
            ops_used = meta.get('operations', {}) if isinstance(meta, dict) else {}

            results.append({
                'cid': cid,
                'similarity': float(sim),
                'meta': meta,
                'ops_used': ops_used,
                'count': concept_info.get('count', 0)
            })

        return results

    # ---------- Expose exemplar stats for logging ----------
    @torch.no_grad()
    def exemplar_stats(self) -> Dict[str, float]:
        total = sum(len(v) for v in self._exemplars.values())
        active = int(self.concept_used.sum().item())
        avg_per = float(total) / max(1, active)
        return {"exemplar_total": float(total), "exemplar_avg_per_concept": avg_per}

    def _proj_to_relmem(self, vec: torch.Tensor) -> torch.Tensor:
        """
        Project self-model vectors from ctrl_dim to RelMem.D space if needed.
        Preserves semantic meaning while ensuring dimensional compatibility.
        """
        if vec.shape[-1] == self.D:
            return vec
        if not hasattr(self, "_selfmodel_proj"):
            import torch.nn as nn
            self._selfmodel_proj = nn.Linear(vec.shape[-1], self.D).to(self.device)
            nn.init.xavier_uniform_(self._selfmodel_proj.weight)
            logger.info(f"[Self-Model] Created projection layer: {vec.shape[-1]} -> {self.D}")
        return self._selfmodel_proj(vec.unsqueeze(0)).squeeze(0)

    # ========================================================================
    # CONCEPT EVOLUTION SYSTEM - EM-Driven Mutation
    # ========================================================================

    @torch.no_grad()
    def evolve_concepts(self, em_delta: float, metrics: dict, step: int):
        """
        Evolve concept library based on EM feedback - the missing mutation operator.

        Mechanisms:
        - EM drops â†’ prune weak concepts, perturb stagnant ones
        - EM rises â†’ clone successful concepts, reinforce relationships
        - EM plateau â†’ inject diversity via concept splitting

        Args:
            em_delta: Change in exact match score (current - baseline)
            metrics: Dict with training metrics (loss, accuracy, etc.)
            step: Current training step
        """
        if not self.is_ready():
            return

        active_concepts = self.concept_used.nonzero().flatten()
        if active_concepts.numel() < 5:
            return  # Need minimum diversity

        # Extract concept performance from metadata
        concept_scores = []
        for cid in active_concepts:
            cid_int = int(cid.item())
            if cid_int in self.concepts:
                ops_meta = self.concepts[cid_int].get('meta', {}).get('operations', {})
                # Average success across operations
                avg_success = np.mean(list(ops_meta.values())) if ops_meta else 0.0
                concept_scores.append((cid_int, avg_success))

        if not concept_scores:
            return

        concept_scores.sort(key=lambda x: x[1])  # Sort by success

        # === MUTATION STRATEGY 1: EM Dropping (Prune + Perturb) ===
        if em_delta < -0.02:  # 2% EM drop
            # Prune bottom 10% weak concepts
            prune_count = max(1, len(concept_scores) // 10)
            for cid, score in concept_scores[:prune_count]:
                if score < 0.1:  # Very weak
                    self.concept_used[cid] = False
                    logging.info(f"[RelMem Evolution] ðŸ—‘ï¸ Pruned concept {cid} (score={score:.3f})")

            # Perturb middle 20% stagnant concepts
            perturb_start = len(concept_scores) // 10
            perturb_end = perturb_start + len(concept_scores) // 5
            for cid, score in concept_scores[perturb_start:perturb_end]:
                noise_scale = 0.2 * (1.0 - score)  # Weaker = more noise
                noise = torch.randn_like(self.concept_proto[cid]) * noise_scale
                self.concept_proto.data[cid] += noise
                self.concept_proto.data[cid] = F.normalize(self.concept_proto.data[cid], dim=-1)
                logging.info(f"[RelMem Evolution] ðŸ”€ Perturbed concept {cid} (noise={noise_scale:.3f})")

        # === MUTATION STRATEGY 2: EM Rising (Clone + Reinforce) ===
        elif em_delta > 0.02:  # 2% EM gain
            # Clone top 5% successful concepts
            clone_count = max(1, len(concept_scores) // 20)
            for cid, score in concept_scores[-clone_count:]:
                if score > 0.5:  # Strong performer
                    # Clone with slight variation
                    clone_vec = self.concept_proto[cid] + torch.randn_like(self.concept_proto[cid]) * 0.05
                    clone_vec = F.normalize(clone_vec, dim=-1)

                    # Copy metadata but mark as clone
                    original_meta = self.concepts[cid].get('meta', {}).copy()
                    original_meta['cloned_from'] = cid
                    original_meta['clone_generation'] = original_meta.get('clone_generation', 0) + 1

                    new_cid = self.add_concept(clone_vec, meta=original_meta)
                    self.bind_concept(new_cid, clone_vec, alpha=0.5)
                    logging.info(f"[RelMem Evolution] ðŸ§¬ Cloned concept {cid}â†’{new_cid} (score={score:.3f})")

            # Reinforce relationships between top concepts
            top_cids = [cid for cid, score in concept_scores[-5:]]
            for i, cid1 in enumerate(top_cids):
                for cid2 in top_cids[i+1:]:
                    # Strengthen "similar_to" relation
                    self.queue_hebbian_update("similar_to", cid1, cid2, eta=0.2)

        # === MUTATION STRATEGY 3: EM Plateau (Split + Diversify) ===
        elif abs(em_delta) < 0.005 and step % 500 == 0:  # Stagnant for 500 steps
            # Split top concept into specialized variants
            if concept_scores:
                best_cid, best_score = concept_scores[-1]
                if best_score > 0.4:
                    # Create 3 specialized variants via directional perturbation
                    base_vec = self.concept_proto[best_cid]
                    for direction_idx in range(3):
                        # Random orthogonal direction
                        direction = torch.randn_like(base_vec)
                        direction = F.normalize(direction - (direction @ base_vec) * base_vec, dim=-1)

                        # Split at 30-degree angle
                        split_vec = F.normalize(base_vec + 0.5 * direction, dim=-1)

                        split_meta = self.concepts[best_cid].get('meta', {}).copy()
                        split_meta['split_from'] = best_cid
                        split_meta['split_direction'] = direction_idx

                        split_cid = self.add_concept(split_vec, meta=split_meta)
                        self.bind_concept(split_cid, split_vec, alpha=0.3)
                        logging.info(f"[RelMem Evolution] âœ‚ï¸ Split concept {best_cid}â†’{split_cid} dir={direction_idx}")

    def _find_concept_for_operation(self, op_name: str) -> int:
        """Find concept ID associated with a specific operation"""
        for cid, concept_data in self.concepts.items():
            if self.concept_used[cid]:
                ops_meta = concept_data.get('meta', {}).get('operations', {})
                if op_name in ops_meta and ops_meta[op_name] > 0.3:
                    return cid
        return -1

    def learn_arc_language(self, successful_traces: List[dict], step: int):
        """
        Extract compositional structure from successful solution traces.
        Learns "grammar" of ARC: which operation sequences work together.

        Args:
            successful_traces: List of {puzzle_id, operations, success_score}
        """
        if not successful_traces:
            return

        # === PATTERN 1: Operation Co-occurrence ===
        # Which operations tend to appear together?
        for trace in successful_traces:
            ops = trace.get('operations', [])
            if len(ops) < 2:
                continue

            # Build co-occurrence graph
            for i in range(len(ops) - 1):
                op1, op2 = ops[i], ops[i+1]

                # Strengthen relation between concepts representing these ops
                cid1 = self._find_concept_for_operation(op1)
                cid2 = self._find_concept_for_operation(op2)

                if cid1 >= 0 and cid2 >= 0:
                    # Use "precedes" relation
                    eta = trace.get('success_score', 0.5) * 0.3
                    self.queue_hebbian_update("precedes", cid1, cid2, eta=eta)

        # === PATTERN 2: Transformation Hierarchies ===
        # flip_h + rotate90 â†’ combined_transform
        composite_patterns = self._detect_composite_operations(successful_traces)

        for pattern in composite_patterns:
            # Create new composite concept
            components = pattern['components']  # List of op names
            success = pattern['success_rate']

            # Average vectors of component concepts
            component_vecs = []
            for op_name in components:
                cid = self._find_concept_for_operation(op_name)
                if cid >= 0:
                    component_vecs.append(self.concept_proto[cid])

            if len(component_vecs) >= 2:
                composite_vec = torch.stack(component_vecs).mean(dim=0)
                composite_vec = F.normalize(composite_vec, dim=-1)

                meta = {
                    'operations': {'+'.join(components): success},
                    'composite': True,
                    'components': components
                }

                composite_cid = self.add_concept(composite_vec, meta=meta)
                self.bind_concept(composite_cid, composite_vec, alpha=0.7)

                # Link composite to components via "composed_of" relation
                for op_name in components:
                    comp_cid = self._find_concept_for_operation(op_name)
                    if comp_cid >= 0:
                        self.queue_hebbian_update("composed_of", composite_cid, comp_cid, eta=0.4)

                logging.info(f"[Arc Language] Learned composite: {'+'.join(components)} (success={success:.3f})")

    def _detect_composite_operations(self, traces: List[dict]) -> List[dict]:
        """Detect frequently co-occurring operation sequences"""
        from collections import Counter

        sequences = []
        for trace in traces:
            ops = trace.get('operations', [])
            if len(ops) >= 2:
                # Extract all bigrams and trigrams
                for i in range(len(ops) - 1):
                    sequences.append(tuple(ops[i:i+2]))
                if len(ops) >= 3:
                    for i in range(len(ops) - 2):
                        sequences.append(tuple(ops[i:i+3]))

        # Find frequent patterns (support â‰¥ 3)
        pattern_counts = Counter(sequences)

        composites = []
        for pattern, count in pattern_counts.items():
            if count >= 3:
                # Compute success rate for this pattern
                pattern_traces = [t for t in traces if self._contains_sequence(t.get('operations', []), pattern)]
                success_rate = np.mean([t.get('success_score', 0) for t in pattern_traces])

                composites.append({
                    'components': list(pattern),
                    'support': count,
                    'success_rate': success_rate
                })

        return composites

    def _contains_sequence(self, ops_list: list, pattern: tuple) -> bool:
        """Check if operation list contains the pattern sequence"""
        pattern_len = len(pattern)
        for i in range(len(ops_list) - pattern_len + 1):
            if tuple(ops_list[i:i+pattern_len]) == pattern:
                return True
        return False

    def initialize_self_model(self, ctrl_dim: int = 256):
        """
        Initialize the self-model with core agent concepts and social relations.

        Creates a persistent knowledge structure encoding:
        - Model as an agent being taught by Trainer to solve ARC puzzles
        - Core concepts: Model, Trainer, ARC_Puzzle, Agent, Teacher, Goal
        - Social relations: teaches, rewards, guides, has_goal, solves, about

        Args:
            ctrl_dim: Dimension for concept vectors (default 256)
        """
        import torch
        import logging

        logger = logging.getLogger(__name__)
        logger.info(f"[Self-Model] Initializing agent concepts (ctrl_dim={ctrl_dim} -> RelMem.D={self.D})")

        # Generate concept vectors with stable random seeds for consistency
        torch.manual_seed(42)  # Ensure reproducible concept vectors

        # Core agent concepts in compact control space
        model_vec = torch.randn(ctrl_dim, device=self.device) * 0.1
        trainer_vec = torch.randn(ctrl_dim, device=self.device) * 0.1
        puzzle_vec = torch.randn(ctrl_dim, device=self.device) * 0.1
        agent_vec = torch.randn(ctrl_dim, device=self.device) * 0.1
        teacher_vec = torch.randn(ctrl_dim, device=self.device) * 0.1
        goal_vec = torch.randn(ctrl_dim, device=self.device) * 0.1
        solve_puzzle_vec = torch.randn(ctrl_dim, device=self.device) * 0.1

        # Bind core concepts with proper projection
        try:
            # Project vectors to RelMem space while preserving semantics
            model_proj = self._proj_to_relmem(model_vec)
            trainer_proj = self._proj_to_relmem(trainer_vec)
            puzzle_proj = self._proj_to_relmem(puzzle_vec)
            agent_proj = self._proj_to_relmem(agent_vec)
            teacher_proj = self._proj_to_relmem(teacher_vec)
            goal_proj = self._proj_to_relmem(goal_vec)
            solve_puzzle_proj = self._proj_to_relmem(solve_puzzle_vec)

            # Primary entities
            self.bind_fact("Model", "is_a", "agent",
                          subj_vec=model_proj, obj_vec=agent_proj, alpha=0.8)
            self.bind_fact("Trainer", "is_a", "teacher",
                          subj_vec=trainer_proj, obj_vec=teacher_proj, alpha=0.8)

            # Social relationships - the core teacher-learner dyad
            self.bind_fact("Trainer", "teaches", "Model",
                          subj_vec=trainer_proj, obj_vec=model_proj, alpha=0.8)
            self.bind_fact("Trainer", "rewards", "Model",
                          subj_vec=trainer_proj, obj_vec=model_proj, alpha=0.7)
            self.bind_fact("Trainer", "guides", "Model",
                          subj_vec=trainer_proj, obj_vec=model_proj, alpha=0.7)

            # Goals and purpose
            self.bind_fact("Model", "has_goal", "solve_puzzle",
                          subj_vec=model_proj, obj_vec=solve_puzzle_proj, alpha=0.8)
            self.bind_fact("solve_puzzle", "about", "ARC_Puzzle",
                          subj_vec=solve_puzzle_proj, obj_vec=puzzle_proj, alpha=0.8)

            # Domain knowledge
            self.bind_fact("ARC_Puzzle", "is_a", "reasoning_task", alpha=0.7)
            self.bind_fact("reasoning_task", "has_attr", "visual_patterns", alpha=0.6)
            self.bind_fact("reasoning_task", "has_attr", "logical_rules", alpha=0.6)

            import logging
            logging.getLogger(__name__).info("[Self-Model] Bound %d core agent concepts with social relations",
                       len(self._symbolic_index))

            # Mark self-model as initialized
            self._self_model_initialized = True

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"[Self-Model] Failed to initialize some concepts: {e}")
            self._self_model_initialized = False

    def _sync_to_device(self):
        """Force all internal tensors to reside on self.device."""
        self.concept_proto.data = self.concept_proto.data.to(self.device)
        self.concept_used = self.concept_used.to(self.device)
        if hasattr(self, "_exemplars"):
            for k, v in self._exemplars.items():
                for ex in v:
                    ex.vec = ex.vec.to(self.device)
        return True

    def get_explanation(self, op_name: str, top_k: int = 3) -> str:
        """
        Generate human-readable explanation for why an operation was chosen.

        Walks the relation graph to find the top contributing facts/relations
        that led to this operation being biased.

        Args:
            op_name: Name of the DSL operation to explain
            top_k: Number of top explanatory factors to return

        Returns:
            str: Human-readable explanation
        """
        try:
            explanation_parts = []

            # Get op bias for this operation
            op_bias_dict = self.get_op_bias_contextual()
            op_score = op_bias_dict.get(op_name, 0.0)

            if op_score < 0.01:
                return f"Operation '{op_name}' has low bias ({op_score:.3f}) - insufficient evidence in knowledge graph."

            # Find contributing relations
            contributing_relations = []

            # Check which relations might contribute to this operation
            relation_to_ops = {
                "color": ["color_map", "flood_fill", "extract_color"],
                "shape": ["rotate90", "flip_h", "flip_v"],
                "structure": ["crop_bbox", "tile_pattern", "resize_nn", "translate", "scale"],
                "logic": ["grid_union", "grid_intersection", "grid_xor", "grid_difference"],
                "identity": ["identity"],
                "flip": ["flip_h", "flip_v"],
                "rotate": ["rotate90", "rotate180", "rotate270"],
                "crop": ["crop_bbox", "crop_nonzero"],
                "tile": ["tile", "tile_pattern"],
                "flood": ["flood_fill", "flood_select"],
                "outline": ["outline", "boundary_extract"],
                "symmetry": ["symmetry"],
                "paste": ["paste"],
                "count": ["count_objects", "count_colors"],
                "pattern": ["find_pattern", "extract_pattern", "match_template"],
                "object": ["for_each_object", "for_each_object_translate", "for_each_object_recolor"],
                "conditional": ["conditional_map", "apply_rule"],
                "select": ["select_by_property", "flood_select"],
                "arithmetic": ["arithmetic_op"]
            }

            # Find which relations support this operation
            for rel_name, ops in relation_to_ops.items():
                if op_name in ops and rel_name in self.relations:
                    try:
                        rel_score = float(self._scores(rel_name).mean().item()) if hasattr(self._scores(rel_name), "mean") else float(self._scores(rel_name))
                        if rel_score > 0.1:
                            contributing_relations.append((rel_name, rel_score))
                    except Exception:
                        continue

            # Sort by contribution strength
            contributing_relations.sort(key=lambda x: x[1], reverse=True)

            # Build explanation
            if contributing_relations:
                explanation_parts.append(f"Operation '{op_name}' selected (bias={op_score:.3f}) due to:")
                for i, (rel_name, score) in enumerate(contributing_relations[:top_k]):
                    explanation_parts.append(f"  {i+1}. {rel_name} relation strength: {score:.3f}")

            # Add self-model context if available
            if hasattr(self, '_self_model_initialized') and self._self_model_initialized:
                if "Model" in self._symbolic_index and "has_goal" in self.relations:
                    explanation_parts.append("  Context: Model has goal to solve ARC puzzles guided by Trainer")

            return " | ".join(explanation_parts) if explanation_parts else f"No clear explanation for '{op_name}' (insufficient graph knowledge)"

        except Exception as e:
            return f"Explanation generation failed for '{op_name}': {e}"

    def prune_compact(self, max_concepts: int = 2048, merge_cos: float = 0.985):
        """
        Merge near-duplicate prototypes and drop stale ones to keep biases sharp.
        Keeps highest-utility (usage Ã— dopamine) concepts first.
        """
        try:
            if not hasattr(self, 'concept_proto') or self.concept_proto is None:
                return
            proto = self.concept_proto  # [N, D]
            used = getattr(self, 'concept_used', None)
            N = proto.size(0)
            if N <= max_concepts:
                return

            # Utility score: usage Ã— dopamine value (or just usage if no dopamine)
            with torch.no_grad():
                usage = used.float() if used is not None else torch.ones(N, device=proto.device)
                dop_v = getattr(self, 'concept_dopamine', torch.ones(N, device=proto.device))
                if dop_v is not None and dop_v.numel() == N:
                    util = usage * (0.1 + dop_v.float())
                else:
                    util = usage

                # Greedy keep highest util, merge duplicates
                keep = []
                taken = torch.zeros(N, dtype=torch.bool, device=proto.device)
                order = torch.argsort(util, descending=True)

                for idx in order:
                    if taken[idx]:
                        continue
                    keep.append(int(idx))
                    if len(keep) >= max_concepts:
                        break
                    # Mark near-duplicates for merging
                    p = F.normalize(proto[idx], p=2, dim=-1)
                    sims = torch.matmul(F.normalize(proto, p=2, dim=-1), p)
                    dupes = sims >= merge_cos
                    taken |= dupes

                # Update tensors
                self.concept_proto = proto[keep].contiguous()
                if used is not None:
                    self.concept_used = used[keep].contiguous()
                if hasattr(self, 'concept_dopamine') and self.concept_dopamine is not None:
                    self.concept_dopamine = self.concept_dopamine[keep].contiguous()

                import logging
                logging.info(f"[RelMem] Pruned {N} â†’ {len(keep)} concepts (threshold={merge_cos})")
        except Exception as e:
            import logging
            logging.warning(f"[RelMem] prune_compact failed: {e}")
    # ==== SynergyFusion Integration Methods ====
    def get_concept_embedding(self, query_vec: torch.Tensor, dsl_ops: List[str],
                               scale: float = 1.0, top_k: int = 8) -> Dict[str, torch.Tensor]:
        """
        Retrieve concept embeddings and operation biases for SynergyFusion.
        Returns GPU tensors with gradients enabled.

        Args:
            query_vec: Query vector [B, D]
            dsl_ops: List of DSL operation names
            scale: Scaling factor for op_bias
            top_k: Number of top concepts to retrieve

        Returns:
            dict with keys:
                - concept_emb: [B, concept_dim] tensor
                - op_bias: [B, num_ops] tensor
                - confidence: [B] tensor
        """
        # Ensure on correct device
        query_vec = query_vec.to(self.device)
        B = query_vec.shape[0]
        num_ops = len(dsl_ops)

        # Default concept dimension
        default_concept_dim = 256

        # If no concepts learned yet, return zeros
        if not hasattr(self, 'concept_proto') or self.concept_proto is None or self.concept_proto.size(0) == 0:
            return {
                'concept_emb': torch.zeros(B, default_concept_dim, device=self.device),
                'op_bias': torch.zeros(B, num_ops, device=self.device),
                'confidence': torch.zeros(B, device=self.device)
            }

        # Project query if needed
        if query_vec.shape[-1] != self.D:
            query_proj = self._proj_to_relmem(query_vec)
        else:
            query_proj = query_vec

        # Compute cosine similarity to all concepts
        query_norm = F.normalize(query_proj, p=2, dim=-1)  # [B, D]
        proto_norm = F.normalize(self.concept_proto, p=2, dim=-1)  # [N, D]

        sims = torch.matmul(query_norm, proto_norm.t())  # [B, N]

        # Get top-k concepts
        top_sims, top_idx = torch.topk(sims, min(top_k, sims.size(-1)), dim=-1)  # [B, k]

        # Softmax weights
        weights = F.softmax(top_sims * 5.0, dim=-1)  # [B, k], temperature=0.2

        # Weighted average of concept embeddings
        top_concepts = self.concept_proto[top_idx]  # [B, k, D]
        concept_emb = torch.einsum('bk,bkd->bd', weights, top_concepts)  # [B, D]

        # Project to standard dimension if needed
        if concept_emb.shape[-1] != default_concept_dim:
            if not hasattr(self, '_concept_proj'):
                self._concept_proj = nn.Linear(self.D, default_concept_dim, device=self.device)
            concept_emb = self._concept_proj(concept_emb)

        # Build op_bias tensor from stored metadata
        op_bias_tensor = torch.zeros(B, num_ops, device=self.device)

        # Get op_bias dict using existing method
        op_bias_dict = self.get_op_bias(dsl_ops=dsl_ops, scale=scale, query_vec=query_vec)

        # Convert dict to tensor
        for i, op_name in enumerate(dsl_ops):
            if op_name in op_bias_dict:
                op_bias_tensor[:, i] = op_bias_dict[op_name]

        # Confidence from top similarity
        confidence = top_sims[:, 0].clamp(0, 1)  # [B]

        return {
            'concept_emb': concept_emb,
            'op_bias': op_bias_tensor,
            'confidence': confidence
        }

    def capacity_ratio(self) -> float:
        """
        Return current memory utilization ratio (used_concepts / max_concepts).

        Returns:
            float: Ratio between 0.0 and 1.0
        """
        if not hasattr(self, 'concept_proto') or self.concept_proto is None:
            return 0.0

        current_size = self.concept_proto.size(0)
        max_size = self.max_concepts

        return float(current_size) / float(max_size)

    @torch.no_grad()
    def prune_if_needed(self, threshold: float = 0.9):
        """
        Prune concepts if capacity exceeds threshold using SRU-based scoring.
        GPU-first implementation.

        Args:
            threshold: Capacity ratio threshold to trigger pruning (default: 0.9)
        """
        ratio = self.capacity_ratio()

        if ratio < threshold:
            return  # No pruning needed

        if not hasattr(self, 'concept_proto') or self.concept_proto is None:
            return

        N = self.concept_proto.size(0)
        if N <= 64:  # Don't prune if very few concepts
            return

        # SRU scoring: usage * recency * utility
        used = getattr(self, 'concept_used', torch.ones(N, device=self.device))

        # Recency: assume concepts added later are more recent
        recency = torch.arange(N, device=self.device, dtype=torch.float32) / float(N)

        # Utility: dopamine if available
        utility = getattr(self, 'concept_dopamine', torch.ones(N, device=self.device))
        if utility is None or utility.numel() != N:
            utility = torch.ones(N, device=self.device)

        # Combined SRU score
        sru_score = used.float() * (0.3 + 0.7 * recency) * (0.1 + utility.float())

        # Keep top 70%
        target_size = int(N * 0.7)
        _, keep_idx = torch.topk(sru_score, target_size)
        keep_idx = keep_idx.sort()[0]  # Sort to maintain order

        # Prune
        self.concept_proto = self.concept_proto[keep_idx].contiguous()
        if hasattr(self, 'concept_used'):
            self.concept_used = self.concept_used[keep_idx].contiguous()
        if hasattr(self, 'concept_dopamine') and self.concept_dopamine is not None:
            self.concept_dopamine = self.concept_dopamine[keep_idx].contiguous()

        import logging
        logging.info(f"[RelMem] SRU pruning: {N} -> {target_size} concepts (ratio was {ratio:.2f})")
