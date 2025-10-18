#!/usr/bin/env python3
"""
High-Fidelity Training Signal Injection from Concept Library
Extracts learned patterns from RelMem and Dream to guide training
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


def query_concept_library(
    relmem,
    brain_latent: torch.Tensor,
    dsl_ops: List[str],
    device: torch.device,
    top_k: int = 32
) -> dict:
    """
    Query RelMem + Dream for similar solved puzzles.
    Returns high-fidelity training signals extracted from concept library.

    Args:
        relmem: RelationalMemoryNeuro instance
        brain_latent: Brain encoding [B, ctrl_dim]
        dsl_ops: List of DSL operation names
        device: Torch device
        top_k: Number of similar concepts to retrieve

    Returns:
        Dictionary with:
        - op_logits: [B, num_ops] operation distribution from similar concepts
        - confidence: float, average similarity of retrieved concepts
        - op_sequence: [B, 3] next-op predictions for multi-step reasoning
        - transformation_bias: [B, 8] transformation type biases
    """
    if relmem is None or not relmem.is_ready():
        return {}

    # Project brain latent to concept space
    query_vec = brain_latent.detach()  # [B, ctrl_dim]
    B = query_vec.shape[0]

    if hasattr(relmem, 'query_projection'):
        query_vec_proj = relmem.query_projection(query_vec)  # [B, D]
    else:
        return {}

    signals = {}

    # === RETRIEVE TOP-K SIMILAR CONCEPTS ===
    K = min(top_k, relmem.N)
    similarities = F.cosine_similarity(
        query_vec_proj.unsqueeze(1),  # [B, 1, D]
        relmem.concept_proto.unsqueeze(0),  # [1, N, D]
        dim=-1
    )  # [B, N]

    # Weight by concept success
    concept_weights = torch.zeros(relmem.N, device=device)
    for cid, concept_data in relmem.concepts.items():
        if relmem.concept_used[cid]:
            ops_meta = concept_data.get('meta', {}).get('operations', {})
            avg_success = np.mean(list(ops_meta.values())) if ops_meta else 0.0
            concept_weights[cid] = avg_success

    # Weighted similarity: sim * success_rate
    weighted_sims = similarities * concept_weights.unsqueeze(0)  # [B, N]

    top_k_sims, top_k_indices = torch.topk(weighted_sims, k=K, dim=-1)  # [B, K]

    # Compute confidence (avg similarity of top-K)
    confidence = top_k_sims.mean(dim=-1)  # [B]
    signals['confidence'] = confidence.mean().item()

    # === EXTRACT OPERATION DISTRIBUTION ===
    op_name_to_idx = {op: idx for idx, op in enumerate(dsl_ops)}
    op_counts = torch.zeros(B, len(dsl_ops), device=device)

    for b in range(B):
        for k_idx in range(K):
            cid = int(top_k_indices[b, k_idx].item())
            if cid in relmem.concepts:
                ops_meta = relmem.concepts[cid].get('meta', {}).get('operations', {})

                # Map operation names to indices
                for op_name, success_score in ops_meta.items():
                    op_idx = op_name_to_idx.get(op_name, -1)
                    if op_idx >= 0:
                        # Weight by similarity and success
                        weight = top_k_sims[b, k_idx].item() * success_score
                        op_counts[b, op_idx] += weight

    # Normalize to logits
    signals['op_logits'] = op_counts / (op_counts.sum(dim=-1, keepdim=True) + 1e-6)

    # === EXTRACT TRANSFORMATION PATTERNS ===
    # Infer transformation types from operation clusters
    transform_patterns = torch.zeros(B, 8, device=device)
    # (flip_h, flip_v, rotate90, rotate180, rotate270, transpose, mirror, identity)

    transform_ops = {
        'flip_h': 0, 'flip_v': 1, 'rotate90': 2, 'rotate180': 3,
        'rotate270': 4, 'transpose': 5, 'mirror': 6, 'identity': 7
    }

    for b in range(B):
        for transform_name, transform_idx in transform_ops.items():
            op_idx = op_name_to_idx.get(transform_name, -1)
            if op_idx >= 0:
                transform_patterns[b, transform_idx] = op_counts[b, op_idx]

    signals['transformation_bias'] = transform_patterns

    # === EXTRACT OPERATION SEQUENCES ===
    # Look for multi-step patterns in concept metadata
    sequences = []
    for b in range(B):
        seq = [-1, -1, -1]  # Max 3-step sequence
        for k_idx in range(min(3, K)):
            cid = int(top_k_indices[b, k_idx].item())
            if cid in relmem.concepts:
                ops_meta = relmem.concepts[cid].get('meta', {}).get('operations', {})
                if ops_meta:
                    # Take highest success operation as next step
                    best_op = max(ops_meta.items(), key=lambda x: x[1])[0]
                    best_op_idx = op_name_to_idx.get(best_op, -1)
                    if best_op_idx >= 0:
                        seq[k_idx] = best_op_idx
        sequences.append(seq)

    signals['op_sequence'] = torch.tensor(sequences, device=device)  # [B, 3]

    # Log retrieval stats periodically
    if signals['confidence'] > 0.1:
        logger.debug(f"[Concept Library] Retrieved K={K} concepts, confidence={signals['confidence']:.3f}")

    return signals


def inject_auxiliary_losses(
    model,
    brain_latent: torch.Tensor,
    concept_signals: dict,
    global_step: int,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Inject high-fidelity training signals as auxiliary losses.

    Args:
        model: TOPAS model instance
        brain_latent: Brain encoding [B, ctrl_dim]
        concept_signals: Output from query_concept_library()
        global_step: Current training step
        device: Torch device

    Returns:
        Dictionary of auxiliary losses
    """
    aux_losses = {}

    if not concept_signals or concept_signals.get('confidence', 0) < 0.05:
        return aux_losses  # Insufficient confidence

    # === 1. OPERATION PREDICTION LOSS ===
    if hasattr(model, 'operation_head') and 'op_logits' in concept_signals:
        predicted_ops = model.operation_head(brain_latent.mean(dim=1))  # [B, num_ops]
        target_ops = concept_signals['op_logits']  # [B, num_ops]

        # KL divergence: push predicted ops toward successful patterns
        op_loss = F.kl_div(
            F.log_softmax(predicted_ops, dim=-1),
            F.softmax(target_ops, dim=-1),
            reduction='batchmean'
        )

        # Adaptive weighting based on concept confidence
        op_weight = concept_signals.get('confidence', 0.5) * 1.5  # λ_max = 1.5
        aux_losses['concept_operation'] = op_weight * op_loss

        if global_step % 100 == 0:
            logger.info(f"[Concept Signal] Op loss={op_loss:.4f}, weight={op_weight:.3f}")

    # === 2. TRANSFORMATION PATTERN LOSS ===
    if hasattr(model, 'transformation_head') and 'transformation_bias' in concept_signals:
        transform_logits = model.transformation_head(brain_latent.mean(dim=1))  # [B, 8]
        transform_targets = concept_signals['transformation_bias']  # [B, 8]

        transform_loss = F.kl_div(
            F.log_softmax(transform_logits, dim=-1),
            F.softmax(transform_targets, dim=-1),
            reduction='batchmean'
        )
        aux_losses['concept_transformation'] = 0.8 * transform_loss

    # === 3. OPERATION SEQUENCE PREDICTION ===
    if hasattr(model, 'sequence_head') and 'op_sequence' in concept_signals:
        seq_hidden = brain_latent.mean(dim=1)  # [B, ctrl_dim]
        next_op_logits = model.sequence_head(seq_hidden)  # [B, num_ops]

        # Target: next operation in successful sequence
        next_op_target = concept_signals['op_sequence'][:, 1]  # [B] - second step

        # Filter out invalid targets (-1)
        valid_mask = next_op_target >= 0
        if valid_mask.any():
            seq_loss = F.cross_entropy(
                next_op_logits[valid_mask],
                next_op_target[valid_mask],
                ignore_index=-1
            )
            aux_losses['concept_sequence'] = 1.0 * seq_loss

    return aux_losses


def create_auxiliary_heads(model, num_ops: int = 41, device: str = "cuda"):
    """
    Create auxiliary prediction heads for concept-driven training.

    Args:
        model: TOPAS model instance
        num_ops: Number of DSL operations
        device: Torch device
    """
    import torch.nn as nn

    if not hasattr(model, 'operation_head'):
        model.operation_head = nn.Sequential(
            nn.Linear(model.ctrl_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_ops)
        ).to(device)
        logger.info(f"[Concept Heads] Created operation_head: {model.ctrl_dim} → {num_ops}")

    if not hasattr(model, 'transformation_head'):
        model.transformation_head = nn.Sequential(
            nn.Linear(model.ctrl_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 8)  # 8 transformation types
        ).to(device)
        logger.info(f"[Concept Heads] Created transformation_head")

    if not hasattr(model, 'sequence_head'):
        model.sequence_head = nn.Sequential(
            nn.Linear(model.ctrl_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_ops)
        ).to(device)
        logger.info(f"[Concept Heads] Created sequence_head")


def post_batch_evolution(model, batch_metrics: dict, step: int, ctx: dict):
    """
    Called after each training batch to evolve concept ecosystem.
    This is where EM feedback drives mutation and learning.

    Args:
        model: TOPAS model instance
        batch_metrics: Dict with training metrics (loss, em, accuracy, etc.)
        step: Current training step
        ctx: Context dict for tracking state across batches
    """
    # Extract EM delta
    em_history = ctx.setdefault('em_history', [])
    current_em = batch_metrics.get('exact_match', 0.0)
    em_history.append(current_em)

    # Compute EM delta (compare to 10-step moving average)
    if len(em_history) >= 10:
        em_baseline = np.mean(em_history[-10:])
        em_delta = current_em - em_baseline
    else:
        em_delta = 0.0

    # === STEP 1: Evolve RelMem Concepts ===
    if hasattr(model, 'relmem') and model.relmem is not None:
        try:
            model.relmem.evolve_concepts(em_delta, batch_metrics, step)
        except Exception as e:
            logger.warning(f"[Evolution] RelMem evolution failed: {e}")

    # === STEP 2: Synchronize Dream ↔ RelMem ===
    if hasattr(model, 'dream') and model.dream is not None:
        try:
            model.dream.sync_concept_ecosystem(model.relmem, em_delta, step)
        except Exception as e:
            logger.warning(f"[Evolution] Dream sync failed: {e}")

    # === STEP 3: Learn from Successful Traces ===
    if 'successful_traces' in ctx and step % 50 == 0:
        # Batch up successful traces and extract patterns
        traces = ctx.get('successful_traces', [])
        if traces and hasattr(model, 'relmem'):
            try:
                model.relmem.learn_arc_language(traces, step)
            except Exception as e:
                logger.warning(f"[Evolution] ARC language learning failed: {e}")

        # Clear trace buffer
        ctx['successful_traces'] = []

    # === STEP 4: Inject Diversity if Stuck ===
    if abs(em_delta) < 0.003 and len(em_history) >= 100:
        # Check for prolonged plateau (last 100 steps)
        recent_em = em_history[-100:]
        if np.std(recent_em) < 0.01:  # Very flat
            logger.warning(f"[Evolution] EM plateau detected (std={np.std(recent_em):.4f}), injecting diversity")

            # Force concept splitting
            if hasattr(model, 'relmem'):
                try:
                    model.relmem.evolve_concepts(0.0, batch_metrics, step)  # Triggers split logic
                except Exception as e:
                    logger.warning(f"[Evolution] Forced concept split failed: {e}")

            # Increase dream mutation rate
            if hasattr(model, 'dream'):
                model.dream.cfg.fsho_levy_scale *= 1.5  # More Levy jumps

    # Store updated context
    ctx['em_delta'] = em_delta
    ctx['em_current'] = current_em

    # Log evolution stats
    if step % 100 == 0:
        logger.info(f"[Evolution] step={step}, EM={current_em:.3f}, delta={em_delta:+.3f}, "
                   f"history_len={len(em_history)}")
