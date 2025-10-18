"""
Bridge between HyLa hypothesis lattice and LMSR prediction market

Synchronizes HyLa's certified sketches with the market's hypothesis universe
and feeds HyLa's ECS scores as market evidence.
"""

import logging
from typing import Dict, List, Any
from hyla_solver import Hypothesis

logger = logging.getLogger(__name__)

# [PATCH] Default universe when HyLa returns empty
DEFAULT_UNIVERSE = [
    {"ops": [("crop_nonzero", {})], "ecs": 0.0},
    {"ops": [("resize_nn", {"H": 9, "W": 9})], "ecs": 0.0},
    {"ops": [("flip_v", {})], "ecs": 0.0},
    {"ops": [("rotate90", {})], "ecs": 0.0}
]


def hypotheses_to_sketch_ids(hypotheses: List[Hypothesis]) -> List[str]:
    """
    Convert HyLa hypotheses to market sketch IDs.

    Format: sketch:op1|op2|...#{param_keys}

    Args:
        hypotheses: List of HyLa Hypothesis objects

    Returns:
        List of sketch ID strings for market universe
    """
    sketch_ids = []

    for hyp in hypotheses:
        ops_str = "|".join(hyp.sketch.ops)

        # Encode param structure (not values)
        param_sigs = []
        for p in hyp.sketch.params:
            if isinstance(p, dict):
                keys = ",".join(sorted(p.keys()))
                param_sigs.append(f"{{{keys}}}")
            else:
                param_sigs.append("*")

        params_str = "|".join(param_sigs)
        sketch_id = f"sketch:{ops_str}#{params_str}"
        sketch_ids.append(sketch_id)

    return sketch_ids


def ecs_to_market_context(hypotheses: List[Hypothesis]) -> Dict[str, Any]:
    """
    Convert HyLa's ECS scores to market context.
    [PATCH] Never returns empty instruments.
    """
    hyps = hypotheses or []
    if not hyps:
        logger.debug("[MARKET] empty HyLa hyps; seeding minimal universe")
        # Convert DEFAULT_UNIVERSE to Hypothesis format
        from hyla_solver import Hypothesis
        from models.dsl_search import DSLProgram
        hyps = []
        for d in DEFAULT_UNIVERSE:
            ops = [op for op, _ in d["ops"]]
            params = [p for _, p in d["ops"]]
            prog = DSLProgram(ops=ops, params=params)
            hyps.append(Hypothesis(sketch=prog, certs={}, meta={'ecs_score': d["ecs"]}))

    sketch_ids = hypotheses_to_sketch_ids(hyps)

    # Map ECS scores to MDL costs
    mdl_costs = {}
    for sketch_id, hyp in zip(sketch_ids, hyps):
        ecs = hyp.meta.get('ecs_score', 0.0)
        mdl = 10.0 - ecs
        mdl_costs[sketch_id] = max(0.0, mdl)

    # Map certificate satisfaction
    invariant_compliance = {}
    for sketch_id, hyp in zip(sketch_ids, hyps):
        invariant_compliance[sketch_id] = (hyp.meta.get('ecs_score', 0.0) > 0)

    # Map sketch_id to ops
    hypothesis_relations = {}
    for sketch_id, hyp in zip(sketch_ids, hyps):
        hypothesis_relations[sketch_id] = list(hyp.sketch.ops)

    context = {
        'mdl_costs': mdl_costs,
        'invariant_compliance': invariant_compliance,
        'hypothesis_relations': hypothesis_relations
    }

    logger.debug(f"[HyLa→Market] Generated context for {len(sketch_ids)} hypotheses")

    return context


def sync_market_universe(market, hypotheses: List[Hypothesis]):
    """
    Synchronize market universe with HyLa's hypothesis lattice.

    Args:
        market: HypothesisMarket instance
        hypotheses: List of HyLa hypotheses
    """
    if market is None:
        return False

    sketch_ids = hypotheses_to_sketch_ids(hypotheses)

    # Add existing market hypotheses to preserve state
    existing_ids = set(market.market.ids)
    combined_ids = list(existing_ids | set(sketch_ids))

    # Reinitialize with combined universe
    market.initialize(combined_ids)

    logger.info(f"[HyLa→Market] Synced universe: {len(existing_ids)} existing + {len(sketch_ids)} HyLa = {len(combined_ids)} total")

    return True  # Success


def report_success_to_market(market, winning_hypothesis: Hypothesis, success: bool = True):
    """
    Feed back HyLa solution success to market for learning.

    Args:
        market: HypothesisMarket instance
        winning_hypothesis: Hypothesis that solved the task
        success: Whether it actually worked
    """
    if market is None:
        return

    sketch_id = hypotheses_to_sketch_ids([winning_hypothesis])[0]

    # Strong buy signal for winning hypothesis
    context = {
        'probe_results': {sketch_id: success},
        'invariant_compliance': {sketch_id: success},
        'mdl_costs': {sketch_id: 1.0 if success else 20.0}
    }

    market.update(context)

    logger.info(f"[HyLa→Market] Reported {'success' if success else 'failure'} for {sketch_id}")
