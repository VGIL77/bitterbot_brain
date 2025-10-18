"""
Logarithmic Market Scoring Rule (LMSR) for Hypothesis Aggregation

Implements a prediction market over program hypotheses where:
- Each hypothesis (DSL sketch + params) is a market outcome
- Trader-agents buy/sell shares based on invariants, MDL, probes, and RelMem priors
- Market prices = calibrated posterior probabilities p(h | evidence)
- Used as PUCT priors to guide program search

References:
- Hanson (2003): Logarithmic market scoring rules for modular combinatorial information aggregation
- LMSR properties: closed-form prices, compositional, bounded loss
- Applied to ARC program synthesis for one-shot generalization

Integration:
- Prices fed into PUCT search as child priors
- Trader PnL tracked for credit assignment
- CEGIS counterexamples drive market updates
"""

import math
import collections
import logging
from typing import Dict, List, Callable, Any, Tuple, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# LMSR Market Maker
# ═══════════════════════════════════════════════════════════

class LMSRMarket:
    """
    Logarithmic Market Scoring Rule market maker.

    Maintains probability distribution over hypotheses via share prices.
    Traders buy/sell shares; prices reflect aggregate belief.
    """

    def __init__(self, liquidity: float = 20.0):
        """
        Args:
            liquidity: Market depth parameter 'b'. Higher = slower price movement.
                      Typical range: 10-50 for ~100 hypotheses.
        """
        self.b = float(liquidity)
        self.q = collections.defaultdict(float)  # shares outstanding by hypothesis id
        self.ids = []  # current hypothesis universe
        self._trade_history = []  # for diagnostics

    def set_universe(self, ids: List[str]):
        """Set the hypothesis universe for this puzzle."""
        self.ids = list(ids)
        # Initialize shares to zero (uniform prior)
        for hid in self.ids:
            if hid not in self.q:
                self.q[hid] = 0.0
        logger.debug(f"[Market] Universe set: {len(self.ids)} hypotheses")

    def prices(self) -> Dict[str, float]:
        """
        Compute current market prices: p_i = exp(q_i/b) / Σ_j exp(q_j/b)

        Returns:
            Dict mapping hypothesis ID -> probability in (0,1), summing to 1.0
        """
        if not self.ids:
            return {}

        # Numerical stability: subtract max before exp
        max_q = max(self.q[i] for i in self.ids)
        exps = [math.exp((self.q[i] - max_q) / self.b) for i in self.ids]
        Z = sum(exps)

        if Z == 0.0 or not math.isfinite(Z):
            # Fallback to uniform if numerical issues
            logger.warning("[Market] Numerical instability, falling back to uniform")
            uniform = 1.0 / len(self.ids)
            return {i: uniform for i in self.ids}

        prices = {i: (exps[k] / Z) for k, i in enumerate(self.ids)}
        return prices

    def get_op_prior(self, operation: str) -> float:
        """
        Get operation prior based on hypothesis probabilities.
        Returns weight multiplier for operation based on how often it appears in high-probability hypotheses.

        Args:
            operation: DSL operation name

        Returns:
            Prior weight (0.0-1.0+) based on market beliefs
        """
        prices_dict = self.prices()
        if not prices_dict:
            return 0.0

        op_prob = 0.0
        for hyp_id, prob in prices_dict.items():
            # hypothesis ID format: "sketch__op1__op2" or single operation
            if operation in hyp_id:
                op_prob += prob

        return op_prob  # Higher prob = operation appears in successful hypotheses

    def trade(self, deltas: Dict[str, float]) -> float:
        """
        Execute trades (buy/sell shares) and compute cost.

        Args:
            deltas: Dict[hypothesis_id] -> Δshares
                   Positive = buy (bullish), Negative = sell (bearish)

        Returns:
            Cost paid for this trade (>=0). Use as learning signal for traders.
        """
        # Cost function: C(q) = b * log Σ_j exp(q_j/b)
        def cost_fn():
            max_q = max(self.q[i] for i in self.ids) if self.ids else 0.0
            exps = [math.exp((self.q[i] - max_q) / self.b) for i in self.ids]
            total = sum(exps)
            return max_q + self.b * math.log(total) if total > 0 else 0.0

        cost_before = cost_fn()

        # Update shares
        for hid, delta in deltas.items():
            if hid in self.ids:
                self.q[hid] += float(delta)

        cost_after = cost_fn()
        cost_paid = cost_after - cost_before

        self._trade_history.append((deltas.copy(), cost_paid))

        logger.debug(f"[Market] Trade executed: {len(deltas)} hypotheses, cost={cost_paid:.4f}")
        return cost_paid

    def reset(self):
        """Reset market state (for new puzzle)."""
        self.q.clear()
        self.ids.clear()
        self._trade_history.clear()
        logger.debug("[Market] Reset")


# ═══════════════════════════════════════════════════════════
# Trader Agents
# ═══════════════════════════════════════════════════════════

@dataclass
class TraderAgent:
    """
    Base class for specialized trader agents.
    Each agent implements a strategy for buying/selling hypothesis shares.
    """
    name: str
    pnl: float = 0.0  # Profit & loss for credit assignment
    trades: int = 0

    def compute_trades(
        self,
        hypotheses: List[str],
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compute desired share changes based on agent's strategy.

        Args:
            hypotheses: List of hypothesis IDs in current universe
            context: Dict with keys like 'witness', 'mdl_costs', 'probe_results', etc.

        Returns:
            Dict[hypothesis_id] -> Δshares (positive=buy, negative=sell)
        """
        raise NotImplementedError


class InvariantTrader(TraderAgent):
    """Trader that buys hypotheses satisfying witness invariants."""

    def compute_trades(
        self,
        hypotheses: List[str],
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Buy hypotheses that satisfy witness constraints; short violators.

        context keys:
            'invariant_compliance': Dict[hypothesis_id] -> bool (satisfies invariants)
        """
        compliance = context.get('invariant_compliance', {})
        trade_size = 2.0  # shares per trade

        deltas = {}
        for hid in hypotheses:
            if hid in compliance:
                deltas[hid] = trade_size if compliance[hid] else -trade_size

        self.trades += 1
        return deltas


class MDLTrader(TraderAgent):
    """Trader that buys shorter (simpler) programs."""

    def compute_trades(
        self,
        hypotheses: List[str],
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Buy hypotheses with lower MDL cost; short complex ones.

        context keys:
            'mdl_costs': Dict[hypothesis_id] -> float (bits)
        """
        mdl_costs = context.get('mdl_costs', {})
        if not mdl_costs:
            return {}

        # Normalize costs to [0,1], invert to get preference
        costs = [mdl_costs[hid] for hid in hypotheses if hid in mdl_costs]
        if not costs:
            return {}

        min_cost, max_cost = min(costs), max(costs)
        range_cost = max_cost - min_cost if max_cost > min_cost else 1.0

        deltas = {}
        for hid in hypotheses:
            if hid in mdl_costs:
                norm_cost = (mdl_costs[hid] - min_cost) / range_cost
                # Low cost → high preference → buy more
                preference = 1.0 - norm_cost
                deltas[hid] = preference * 3.0 - 1.5  # range [-1.5, +1.5]

        self.trades += 1
        return deltas


class ProbeTrader(TraderAgent):
    """Trader that buys hypotheses passing quick execution probes."""

    def compute_trades(
        self,
        hypotheses: List[str],
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Buy passing hypotheses; short failing ones.

        context keys:
            'probe_results': Dict[hypothesis_id] -> bool (passed probe)
        """
        probe_results = context.get('probe_results', {})
        trade_size = 2.5  # strong signal

        deltas = {}
        for hid in hypotheses:
            if hid in probe_results:
                deltas[hid] = trade_size if probe_results[hid] else -trade_size

        self.trades += 1
        return deltas


class RelMemTrader(TraderAgent):
    """Trader that buys hypotheses matching RelMem relation strengths."""

    def compute_trades(
        self,
        hypotheses: List[str],
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Buy hypotheses whose operations align with fired RelMem relations.

        context keys:
            'relmem_relations': Dict[relation_name] -> strength in [0,1]
            'hypothesis_relations': Dict[hypothesis_id] -> List[relation_name]
        """
        rel_strengths = context.get('relmem_relations', {})
        hyp_relations = context.get('hypothesis_relations', {})

        if not rel_strengths or not hyp_relations:
            return {}

        deltas = {}
        for hid in hypotheses:
            if hid not in hyp_relations:
                continue

            # Score = sum of relation strengths for operations in this hypothesis
            score = sum(rel_strengths.get(rel, 0.0) for rel in hyp_relations[hid])
            score /= max(1, len(hyp_relations[hid]))  # average

            # Map score to shares: [0,1] -> [-1, +2]
            deltas[hid] = score * 3.0 - 1.0

        self.trades += 1
        return deltas


# ═══════════════════════════════════════════════════════════
# Market Orchestrator
# ═══════════════════════════════════════════════════════════

class HypothesisMarket:
    """
    High-level market orchestrator combining LMSR + trader agents.

    Usage:
        market = HypothesisMarket(liquidity=20.0)
        market.initialize(hypotheses=['h1', 'h2', ...])
        market.update(context={'invariant_compliance': {...}, ...})
        priors = market.get_priors()  # feed to PUCT
    """

    def __init__(
        self,
        liquidity: float = 20.0,
        traders: Optional[List[TraderAgent]] = None
    ):
        self.market = LMSRMarket(liquidity=liquidity)

        # Default trader portfolio
        if traders is None:
            traders = [
                InvariantTrader(name="invariant_trader"),
                MDLTrader(name="mdl_trader"),
                ProbeTrader(name="probe_trader"),
                RelMemTrader(name="relmem_trader"),
            ]

        self.traders = traders
        logger.info(f"[HypothesisMarket] Initialized with {len(self.traders)} traders")

    def initialize(self, hypotheses: List[str]):
        """Set hypothesis universe for current puzzle."""
        self.market.set_universe(hypotheses)
        logger.debug(f"[HypothesisMarket] Initialized {len(hypotheses)} hypotheses")

    def update(self, context: Dict[str, Any]):
        """
        Run trading round: each agent computes trades, execute batch update.

        Args:
            context: Evidence dict (invariants, MDL, probes, RelMem, etc.)
        """
        if not self.market.ids:
            logger.warning("[HypothesisMarket] No hypotheses in universe, skipping update")
            return

        # Aggregate trades from all agents
        all_deltas = collections.defaultdict(float)

        for trader in self.traders:
            try:
                deltas = trader.compute_trades(self.market.ids, context)
                for hid, delta in deltas.items():
                    all_deltas[hid] += delta
            except Exception as e:
                logger.warning(f"[HypothesisMarket] Trader {trader.name} failed: {e}")

        # Execute aggregated trade
        if all_deltas:
            cost = self.market.trade(dict(all_deltas))

            # Update trader PnLs (simplified: split cost equally)
            cost_per_trader = cost / len(self.traders)
            for trader in self.traders:
                trader.pnl -= cost_per_trader

        logger.debug(f"[HypothesisMarket] Updated with {len(all_deltas)} trades")

    def get_priors(self) -> Dict[str, float]:
        """Get current market prices as PUCT priors."""
        return self.market.prices()

    def reset(self):
        """Reset for new puzzle."""
        self.market.reset()
        for trader in self.traders:
            trader.pnl = 0.0
            trader.trades = 0
        logger.debug("[HypothesisMarket] Reset for new puzzle")

    def diagnostics(self) -> Dict[str, Any]:
        """Return market diagnostics for logging/debugging."""
        prices = self.market.prices()
        return {
            'num_hypotheses': len(self.market.ids),
            'top_5': sorted(prices.items(), key=lambda x: x[1], reverse=True)[:5],
            'entropy': -sum(p * math.log2(p + 1e-12) for p in prices.values()),
            'trader_stats': {t.name: {'pnl': t.pnl, 'trades': t.trades} for t in self.traders},
        }
