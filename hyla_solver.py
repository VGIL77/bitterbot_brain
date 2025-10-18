"""
HyLa: Hypothesis Lattice with Certified Sketches for ARC-AGI-II

One-glance solver using abductive rule induction + typed sketch DSL + certificate-guided pruning.
No heavy search or TTT - just fast abduction with proof obligations.

References:
- ARC Prize team: efficient program synthesis with learned guidance (not brute force)
- DreamCoder: library learning + sketches for few-shot induction
- Meta-Interpretive Learning (MIL): abduction + deduction loop for rule discovery
"""

import torch
from typing import List, Tuple, Dict, Any, Optional
import logging
from dataclasses import dataclass, field
import math
from collections import Counter

# Reuse existing certificate infrastructure
from trainers.certificates import (
    mine_certificates,
    extend_certificates,
    certificate_penalty,
    should_prune_hard,
    certificate_value_adjustment
)
from models.dsl_search import DSLProgram, apply_program, generate_op_parameters, CORE_OPS
from trainers.demo_utils import unpack_demo_batch

# Build banner for import-path verification
BUILD_ID = "2025-10-04_unified-unpack_v3"
logger = logging.getLogger(__name__)
logger.info(f"[HyLa] build={BUILD_ID} file={__file__}")


# [PATCH][HYLA] helpers --------------------------------------------------------
def _modal_target_shape(demos: List[Tuple[torch.Tensor, torch.Tensor]]) -> Optional[Tuple[int,int]]:
    """Return the modal (H,W) of outputs across demos (None if unavailable)."""
    try:
        shapes = [(int(o.shape[-2]), int(o.shape[-1])) for _, o in demos if hasattr(o, 'shape')]
        if not shapes:
            return None
        return Counter(shapes).most_common(1)[0][0]
    except Exception:
        return None

def _fallback_sketches_for_shape(demos) -> List[List[Tuple[str, Dict[str, Any]]]]:
    """Produce minimal, safe sketches when HyLa struggles."""
    HtWt = _modal_target_shape(demos) or None
    sketches: List[List[Tuple[str, Dict[str, Any]]]] = []
    sketches.append([("crop_nonzero", {})])
    sketches.append([("flip_v", {})])
    sketches.append([("flip_h", {})])
    sketches.append([("rotate90", {})])
    if HtWt:
        Ht, Wt = HtWt
        sketches.append([("resize_nn", {"H": int(Ht), "W": int(Wt)})])
        sketches.append([("crop_nonzero", {}), ("resize_nn", {"H": int(Ht), "W": int(Wt)})])
    return sketches[:16]


@dataclass
class Hypothesis:
    """Typed sketch with certificate guards"""
    sketch: DSLProgram  # Program with holes
    certs: Dict[str, callable]  # Certificate constraints
    meta: Dict[str, Any] = field(default_factory=dict)  # ECS score, invariants, etc.

    def __post_init__(self):
        if 'ecs_score' not in self.meta:
            self.meta['ecs_score'] = 0.0

    @property
    def id(self) -> str:
        """Generate unique ID from program operations"""
        return "sketch__" + "__".join(self.sketch.ops)

    @property
    def program(self) -> DSLProgram:
        """Alias for sketch (for compatibility)"""
        return self.sketch

    @property
    def certificates(self) -> Dict[str, callable]:
        """Alias for certs (for compatibility)"""
        return self.certs


def auto_formalize(demos: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, callable]:
    """
    Extract abductive clues (invariants) from demonstrations.

    This is your Auto-Formalizer: it computes compact explanatory invariants
    that act as typed constraints for sketch hole-filling.

    Args:
        demos: List of (input, output) demonstration pairs

    Returns:
        Dict of certificate functions (name -> penalty_fn)
    """
    # Defensive: normalize demos first (handles dicts/3-tuples) - use ensure_demo_pairs
    from trainers.demo_utils import ensure_demo_pairs
    demo_pairs = ensure_demo_pairs(demos)

    # Mine basic certificates
    certs = mine_certificates(demo_pairs)

    # Extend with advanced invariants (D4, color perm, connectivity, etc.)
    certs = extend_certificates(certs, demo_pairs)

    logger.debug(f"[HyLa] Auto-formalized {len(certs)} certificates from {len(demos)} demos")

    return certs


def explanatory_compression_score(
    program: DSLProgram,
    certs: Dict[str, callable],
    demos: List[Tuple[torch.Tensor, torch.Tensor]]
) -> float:
    """
    Compute ECS (Explanatory Compression Score):

    ECS = -MDL(program) + λ1·Σcert_satisfied - λ2·Σviolations + λ3·predictive_margin

    Where:
    - MDL ~ program length + param entropy (shorter is better)
    - cert_satisfied = number of certificates that hold
    - violations = weighted certificate penalties
    - predictive_margin = how well it fits all demos

    Args:
        program: DSLProgram to score
        certs: Certificate dict
        demos: Demonstration pairs

    Returns:
        ECS score (higher is better)
    """
    # MDL: program length + parameter complexity
    mdl = len(program.ops)
    for p in program.params:
        mdl += len(str(p)) * 0.01  # Small penalty for complex params

    # Certificate satisfaction
    cert_satisfied = 0
    cert_violations = 0.0

    for pair in demos:  # demos already normalized to strict 2-tuples
        inp, out = pair[0], pair[1]
        try:
            pred = apply_program(inp, program)

            # Count satisfied certificates
            for cert_fn in certs.values():
                penalty = cert_fn(pred)
                if penalty < 0.1:  # Satisfied (tolerance for soft constraints)
                    cert_satisfied += 1
                cert_violations += penalty

        except Exception:
            cert_violations += 10.0  # Execution failure penalty

    # Predictive margin: how well does it predict?
    predictive_margin = 0.0
    for pair in demos:  # demos already normalized to strict 2-tuples
        inp, out = pair[0], pair[1]
        try:
            pred = apply_program(inp, program)
            if pred.shape == out.shape:
                accuracy = (pred == out).float().mean().item()
                predictive_margin += accuracy
            else:
                predictive_margin -= 0.5  # Shape mismatch penalty
        except Exception:
            predictive_margin -= 1.0

    # Weights (tunable)
    λ1, λ2, λ3 = 0.5, 2.0, 5.0

    ecs = -mdl + λ1 * cert_satisfied - λ2 * cert_violations + λ3 * predictive_margin

    return ecs


def propose_certified_sketches(
    demos: List[Tuple[torch.Tensor, torch.Tensor]],
    max_hyp: int = 32
) -> List[Hypothesis]:
    """
    Build hypothesis lattice: start with certificate-constrained atomic sketches,
    then expand to 2-3 op compositions.

    This is your combinatorics layer - but certificate-typed, so the space is small.

    Args:
        demos: Demonstration pairs
        max_hyp: Max hypotheses to return

    Returns:
        List of Hypothesis objects, ranked by initial plausibility
    """
    certs = auto_formalize(demos)

    # Seed frontier with single-op sketches consistent with certificates
    frontier: List[Hypothesis] = []

    # Generate 1-op sketches (OPTIMIZED: validate on first demo only for speed)
    for op in CORE_OPS:
        param_options = generate_op_parameters(op, None)

        for params in param_options[:3]:  # Limit to 3 params per op
            program = DSLProgram(ops=[op], params=[params])

            # FAST: Check if program violates constraints on FIRST demo only
            # (Full validation happens later in hyla_one_glance)
            violates_hard = False
            if demos:  # Quick check on first demo
                inp, out = demos[0][0], demos[0][1]
                try:
                    pred = apply_program(inp, program)
                    if should_prune_hard(certs, pred, threshold=1.0):
                        violates_hard = True
                except Exception:
                    violates_hard = True

            if not violates_hard:
                hyp = Hypothesis(
                    sketch=program,
                    certs=certs,
                    meta={'depth': 1}
                )
                frontier.append(hyp)

            # Early stop if we have enough 1-op candidates
            if len(frontier) >= 30:
                break
        if len(frontier) >= 30:
            break

    # Generate 2-op sketches (compositions) - OPTIMIZED
    two_op_frontier = []
    for hyp in frontier[:10]:  # Only expand top 10 1-op sketches
        for op2 in CORE_OPS[:8]:  # Limit second op choices
            param_options = generate_op_parameters(op2, None)

            for params2 in param_options[:2]:
                program = DSLProgram(
                    ops=hyp.sketch.ops + [op2],
                    params=hyp.sketch.params + [params2]
                )

                # FAST: Check first demo only
                violates_hard = False
                if demos:
                    inp, out = demos[0][0], demos[0][1]
                    try:
                        pred = apply_program(inp, program)
                        if should_prune_hard(certs, pred, threshold=1.0):
                            violates_hard = True
                    except Exception:
                        violates_hard = True

                if not violates_hard:
                    hyp2 = Hypothesis(
                        sketch=program,
                        certs=certs,
                        meta={'depth': 2}
                    )
                    two_op_frontier.append(hyp2)

                # Early stop if we have enough 2-op candidates
                if len(two_op_frontier) >= 20:
                    break
            if len(two_op_frontier) >= 20:
                break
        if len(two_op_frontier) >= 20:
            break

    # Combine 1-op and 2-op sketches
    all_hyps = frontier + two_op_frontier

    # Score all hypotheses by ECS
    for hyp in all_hyps:
        hyp.meta['ecs_score'] = explanatory_compression_score(hyp.sketch, hyp.certs, demos)

    # Rank by ECS
    all_hyps.sort(key=lambda h: h.meta['ecs_score'], reverse=True)

    # Propagate certificates into hypotheses (refresh with latest mines)
    from trainers.certificates import mine_certificates
    certs = mine_certificates(demos)
    if certs:
        for h in all_hyps:
            h.certs = certs

    result_hyps = all_hyps[:max_hyp]

    if result_hyps:
        logger.info(f"[HyLa] Produced {len(result_hyps)} sketches (ECS max={result_hyps[0].meta['ecs_score']:.2f})")
    else:
        logger.info(f"[HyLa] No valid sketches produced")

    return result_hyps


def hyla_one_glance(
    demos: List[Tuple[torch.Tensor, torch.Tensor]],
    max_depth: int = 6,
    beam_width: int = 32,
    strict: bool = False,
    verbose: bool = False,
    test_input: Optional[torch.Tensor] = None,
    max_hyp: Optional[int] = None
) -> List[Hypothesis]:
    """
    One-glance solver: propose certified sketches ranked by ECS score.

    This is your fast path - no search, just abductive reasoning + deduction.
    Returns hypothesis list for Market integration (not executed predictions).

    Args:
        demos: Demonstration pairs
        max_depth: Maximum program depth (passed as max_hyp for backward compat)
        beam_width: Beam width (passed as max_hyp for backward compat)
        strict: If True, only return hypotheses that validate on all demos
        verbose: Enable debug logging
        test_input: (deprecated) Test input - not used in lattice mode
        max_hyp: Override for max hypotheses (defaults to beam_width)

    Returns:
        List[Hypothesis] ranked by ECS score, or empty list on failure
    """
    # Backward compat: use beam_width as max_hyp if not specified
    if max_hyp is None:
        max_hyp = min(beam_width, 32)

    # Propose certified sketches
    hypotheses = propose_certified_sketches(demos, max_hyp=max_hyp)

    if not hypotheses:
        if verbose:
            logger.debug("[HyLa] No valid hypotheses generated - returning fallback sketches")
        # Return fallback sketches for robustness
        fallback_sketches = _fallback_sketches_for_shape(demos)
        certs = auto_formalize(demos)
        fallback_hyps = [
            Hypothesis(sketch=DSLProgram(ops=[op for op, _ in sk], params=[p for _, p in sk]),
                      certs=certs,
                      meta={'ecs_score': -10.0, 'depth': len(sk), 'fallback': True})
            for sk in fallback_sketches
        ]
        if verbose:
            logger.info(f"[HyLa] fallback frontier: {len(fallback_hyps)} sketches")
        return fallback_hyps

    # Strict mode: filter to only hypotheses that validate on ALL demos
    if strict:
        validated_hyps = []
        for hyp in hypotheses:
            all_demos_satisfied = True
            for pair in demos:
                inp, out = pair[0], pair[1]
                try:
                    pred = apply_program(inp, hyp.sketch)
                    if not torch.equal(pred, out):
                        all_demos_satisfied = False
                        break
                except Exception:
                    all_demos_satisfied = False
                    break

            if all_demos_satisfied:
                validated_hyps.append(hyp)

        if verbose:
            logger.info(f"[HyLa] Strict validation: {len(validated_hyps)}/{len(hypotheses)} passed")

        hypotheses = validated_hyps if validated_hyps else hypotheses[:max(1, len(hypotheses)//4)]

    if verbose:
        logger.info(f"[HyLa] Generated {len(hypotheses)} certified sketches (top ECS: {hypotheses[0].meta['ecs_score']:.2f})")

    return hypotheses


def get_warm_start_for_puct(
    demos: List[Tuple[torch.Tensor, torch.Tensor]],
    max_candidates: int = 8
) -> Dict[str, float]:
    """
    Get top sketch operations as warm-start op_bias for PUCT.

    If HyLa doesn't find a solution, we pass its top operations as priors to PUCT
    for a warm-started search.

    Args:
        demos: Demonstration pairs
        max_candidates: Max sketches to consider

    Returns:
        op_bias dict mapping operation names to bias weights
    """
    hypotheses = propose_certified_sketches(demos, max_hyp=max_candidates)

    if not hypotheses:
        return {}

    # Count operation frequencies weighted by ECS
    op_weights = {}
    total_weight = 0.0

    for hyp in hypotheses:
        weight = max(0.1, hyp.meta['ecs_score'])  # Ensure positive
        for op in hyp.sketch.ops:
            op_weights[op] = op_weights.get(op, 0.0) + weight
            total_weight += weight

    # Normalize to [0, 1] range
    if total_weight > 0:
        op_bias = {op: w / total_weight for op, w in op_weights.items()}
    else:
        op_bias = {}

    logger.debug(f"[HyLa] Warm-start op_bias: {op_bias}")

    return op_bias
