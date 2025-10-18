
# alpha_evolve.py
# GPU-first Alpha-Evolve orchestrator for ARC-II

import math
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F

def _try_import(name):
    try:
        mod = __import__(name, fromlist=['*'])
        return mod
    except Exception:
        return None

puct_search = _try_import('puct_search')
dsl_search = _try_import('dsl_search')
wormhole_offline = _try_import('wormhole_offline')
alpha_dsl = _try_import('alpha_dsl')
hypothesis_market = _try_import('hypothesis_market')
hyla_solver = _try_import('hyla_solver')
arc_constraints = _try_import('arc_constraints')

def _torch_no_grad_inference():
    return torch.inference_mode()

def softmax_dict(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return {}
    vals = torch.tensor(list(d.values()), dtype=torch.float32)
    probs = torch.softmax(vals, dim=0).tolist()
    return {k: p for k, p in zip(d.keys(), probs)}

def blend_priors(*priors: Dict[str, float], weights: Optional[List[float]] = None) -> Dict[str, float]:
    sources = [p for p in priors if p]
    if not sources:
        return {}
    keys = set().union(*[p.keys() for p in sources])
    if weights is None:
        weights = [1.0] * len(sources)
    import math
    out = {}
    wsum = sum(weights) + 1e-8
    for k in keys:
        acc = 0.0
        for wi, src in zip(weights, sources):
            p = max(1e-6, float(src.get(k, 0.0)))
            acc += wi * math.log(p)
        out[k] = math.exp(acc / wsum)
    s = sum(out.values()) + 1e-12
    return {k: v / s for k, v in out.items()}

def mdl_penalty(program_ops: List[Any], per_op: float = 0.02) -> float:
    return per_op * float(len(program_ops or []))

def grid_energy_proxy(grid: torch.Tensor) -> float:
    if not isinstance(grid, torch.Tensor):
        return 0.0
    if grid.dtype != torch.float32:
        grid = grid.float()
    if grid.dim() == 3 and grid.size(-1) == 10:
        probs = torch.softmax(grid, dim=-1)
        ent = -(probs * torch.log(probs.clamp_min(1e-9))).sum(-1).mean().item()
        return float(ent)
    return 1.0

def select_top2(candidates: List[Dict[str, Any]]) -> List[List[List[int]]]:
    if not candidates:
        return []
    ranked = sorted(candidates, key=lambda c: c.get('score', -1e9), reverse=True)
    outs = []
    for c in ranked[:2]:
        grid = c.get('grid')
        if isinstance(grid, torch.Tensor):
            g = grid.detach().cpu().to(dtype=torch.long)
            if g.dim() == 2:
                outs.append(g.tolist())
            elif g.dim() == 3 and g.size(-1) == 10:
                outs.append(torch.argmax(g, dim=-1).tolist())
            elif g.dim() == 3 and g.size(0) == 1:
                outs.append(g[0].tolist())
        elif isinstance(grid, list):
            outs.append(grid)
    if len(outs) == 1:
        outs.append(outs[0])
    return outs[:2]

@dataclass
class AlphaEvolveConfig:
    monologue_k: int = 8
    monologue_temp: float = 0.7
    use_inference_monologue: bool = True
    use_orbit_canon: bool = True
    certificates: str = "hard"
    use_ebr: bool = True
    ebr_iters: int = 5
    puct_depth: int = 6
    puct_sims: int = 2000
    puct_c: float = 1.5
    puct_beam: int = 24
    enable_market: bool = True
    market_liquidity: float = 20.0
    hyla_max_depth: int = 4
    hyla_beam_width: int = 50
    self_play_enable: bool = True
    n_self_play_games: int = 4
    alpha_dsl_enable: bool = True
    alpha_dsl_sims: int = 400
    alpha_dsl_max_depth: int = 10
    meta_adapt_enable: bool = True
    meta_inner_steps: int = 2
    meta_inner_lr: float = 1e-3
    replay_capacity: int = 20000
    wormhole_enable: bool = True
    w_prior: float = 1.0
    w_cert: float = 1.0
    w_mdl: float = -1.0
    w_energy: float = -1.0
    w_relmem: float = 0.5

class AlphaEvolver:
    def __init__(self, model, dsl_ops: List[str], device: Optional[torch.device] = None, cfg: Optional[AlphaEvolveConfig] = None):
        self.model = model
        self.cfg = cfg or AlphaEvolveConfig()
        self.dsl_ops = dsl_ops
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self._init_runtime_buffers()

    def _init_runtime_buffers(self):
        self.replay = []
        self.replay_capacity = self.cfg.replay_capacity
        self.wormhole = getattr(getattr(self.model, "dream", None), "wormhole", None)

    def solve_task(self, demos: List[List[List[int]]], test_grid: List[List[int]]) -> List[List[List[int]]]:
        with _torch_no_grad_inference():
            cand = []
            demos_canon, test_canon, (rot, ref) = self._maybe_orbit_canon(demos, test_grid)
            priors, aux = self._collect_priors(demos_canon, test_canon)
            if self.cfg.use_inference_monologue and self.cfg.monologue_k > 1:
                cand += self._run_monologue(demos_canon, test_canon, priors, aux)
            else:
                cand += self._run_puct(demos_canon, test_canon, priors, aux)
            if self.cfg.alpha_dsl_enable and alpha_dsl is not None:
                cand += self._run_alpha_dsl(demos_canon, test_canon, priors)
            if self.cfg.self_play_enable:
                cand += self._run_self_play(demos_canon, test_canon)
            attempts_canon = select_top2(cand)
            attempts = [self._d4_apply(g, *self._d4_inverse(rot, ref)) for g in attempts_canon]
            if self.cfg.meta_adapt_enable:
                self._meta_adapt_on_demos(demos)
            return attempts

    def _maybe_orbit_canon(self, demos, test_grid):
        if not self.cfg.use_orbit_canon:
            return demos, test_grid, (0, False)
        try:
            from trainers.orbits import apply_orbit_canon, d4_inverse
        except Exception:
            return demos, test_grid, (0, False)
        demos_canon, test_canon, (rot, ref) = apply_orbit_canon(demos, test_grid, self.dsl_ops)
        self._d4_inverse = d4_inverse
        return demos_canon, test_canon, (rot, ref)

    def _collect_priors(self, demos, test_grid):
        relmem_bias = {}
        if hasattr(self.model, "relmem") and hasattr(self.model.relmem, "get_op_bias"):
            try:
                relmem_bias = self.model.relmem.get_op_bias(dsl_ops=self.dsl_ops, query_vec=None, scale=1.0)
            except Exception:
                relmem_bias = {}
        hrm_bias = {}
        if hasattr(self.model, "get_hmr_op_priors"):
            try:
                hrm_bias = self.model.get_hmr_op_priors(demos, test_grid)
            except Exception:
                hrm_bias = {}
        market_priors = {}
        hyla_priors = {}
        if self.cfg.enable_market and hypothesis_market is not None and hyla_solver is not None:
            try:
                hyla = hyla_solver.HyLaSolver(self.dsl_ops)
                hyps = hyla.generate_lattice(demos, max_depth=self.cfg.hyla_max_depth, beam_width=self.cfg.hyla_beam_width)
                market = hypothesis_market.HypothesisMarket(liquidity=self.cfg.market_liquidity)
                market.set_universe([h.id for h in hyps])
                market.ingest_hypotheses(hyps, demos=demos, test=test_grid)
                market_priors = market.get_op_priors()
                hyla_priors = market_priors
            except Exception:
                market_priors = {}
                hyla_priors = {}
        priors = blend_priors(relmem_bias, hrm_bias, market_priors, hyla_priors, weights=[1.0, 0.8, 0.6, 0.6])
        aux = {'relmem': relmem_bias, 'hrm': hrm_bias, 'market': market_priors}
        return priors, aux

    def _run_puct(self, demos, test_grid, priors, aux):
        c = self.cfg
        candidates = []
        try:
            program, info = puct_search.puct_search(
                demos=demos,
                test=test_grid,
                dsl_ops=self.dsl_ops,
                sims=c.puct_sims,
                depth=c.puct_depth,
                c_puct=c.puct_c,
                beam=c.puct_beam,
                op_priors=priors,
                relmem_bias=aux.get('relmem', {}),
                return_info=True
            )
            grid_logits = self._execute_program(test_grid, program)
            grid_refined, e = self._maybe_ebr(test_grid, grid_logits)
            score = self._score_candidate(program, priors, e, aux)
            candidates.append({'grid': grid_refined, 'program': program, 'score': score})
        except Exception:
            pass
        return candidates

    def _run_monologue(self, demos, test_grid, priors, aux):
        """
        Run k diversified PUCT traces with slight noise/temperature and
        score candidates with prior mass + MDL + energy + (optional) HRM value.
        """
        k = int(self.cfg.monologue_k)
        T = float(self.cfg.monologue_temp)
        candidates = []
        for i in range(k):
            # jitter: soften priors to diversify search
            pri = {k2: max(1e-6, float(v2))**(1.0/T) for k2, v2 in (priors or {}).items()}
            candidates += self._run_puct(demos, test_grid, pri, aux)
        # re-score with HRM value if available
        scored = []
        for c in candidates:
            score = float(c.get("score", 0.0))
            try:
                if hasattr(self.model, "hrm_bridge") and hasattr(self.model, "planner") and self.model.hrm_bridge:
                    # Use current feature map via model.painter or encoder cache if available
                    # Fall back to a neutral prior if we cannot fetch states
                    v = 0.0
                    if hasattr(self.model, "_last_hrm_control"):
                        zH = self.model._last_hrm_control.get("z_H", None)
                        if zH is not None and isinstance(zH, torch.Tensor):
                            # Build a small dummy global (zeros) if feature not at hand
                            B = zH.size(0)
                            g = torch.zeros(B, getattr(self.model.config, "width", 512), device=zH.device, dtype=zH.dtype)
                            v = float(self.model.hrm_bridge.predict_value(g, zH).mean().item())
                    score += 0.5 * v  # light weight to keep MDL/energy in play
            except Exception:
                pass
            c["score"] = score
            scored.append(c)
        # sort + unique by grid checksum when possible
        ranked = sorted(scored, key=lambda x: x.get("score", -1e9), reverse=True)
        return ranked

    def _run_alpha_dsl(self, demos, test_grid, priors):
        c = self.cfg
        candidates = []
        try:
            searcher = alpha_dsl.AlphaDSL(policy=None, value=None, dsl_ops=self.dsl_ops)
            program = searcher.search(demos, num_simulations=c.alpha_dsl_sims, max_depth=c.alpha_dsl_max_depth, op_priors=priors)
            grid_logits = self._execute_program(test_grid, program)
            grid_refined, e = self._maybe_ebr(test_grid, grid_logits)
            score = self._score_candidate(program, priors, e, {})
            candidates.append({'grid': grid_refined, 'program': program, 'score': score})
        except Exception:
            pass
        return candidates

    def _run_self_play(self, demos, test_grid):
        candidates = []
        try:
            from self_play import generate_self_play_traces
            traces = generate_self_play_traces(demos, n_games=self.cfg.n_self_play_games, max_depth=self.cfg.puct_depth)
            for t in traces:
                program = t.get('program')
                if not program:
                    continue
                grid_logits = self._execute_program(test_grid, program)
                grid_refined, e = self._maybe_ebr(test_grid, grid_logits)
                score = self._score_candidate(program, {}, e, {})
                candidates.append({'grid': grid_refined, 'program': program, 'score': score})
            if self.cfg.wormhole_enable and self.wormhole is not None:
                try:
                    progs = [t['program'] for t in traces if 'program' in t]
                    self.wormhole.mine_from_programs(progs, top_k=5)
                except Exception:
                    pass
        except Exception:
            pass
        return candidates

    def _execute_program(self, test_grid, program):
        if hasattr(dsl_search, 'apply_program'):
            return dsl_search.apply_program(test_grid, program)
        if hasattr(alpha_dsl, 'apply_program'):
            return alpha_dsl.apply_program(test_grid, program)
        return torch.tensor(test_grid)

    def _maybe_ebr(self, test_grid, grid_logits):
        if not self.cfg.use_ebr:
            return grid_logits, grid_energy_proxy(grid_logits)
        try:
            from energy_refinement import refine_with_ebr
            refined, energy = refine_with_ebr(pred_logits=grid_logits, test_grid=test_grid, iters=self.cfg.ebr_iters)
            return refined, float(energy)
        except Exception:
            return grid_logits, grid_energy_proxy(grid_logits)

    def _score_candidate(self, program, priors, energy, aux):
        mdl = -mdl_penalty(program or [])
        prior_mass = 0.0
        if priors and program:
            for op in program:
                name = op[0] if isinstance(op, (list, tuple)) else str(op)
                prior_mass += float(priors.get(name, 1e-6))
            import math
            prior_mass = math.log(prior_mass + 1e-6)
        relmem_boost = 0.0
        relmem = aux.get('relmem', {})
        if relmem and program:
            relmem_boost = sum(float(relmem.get((op[0] if isinstance(op, (list,tuple)) else str(op)), 0.0)) for op in program) / (len(program) + 1e-9)
        cert_bonus = 0.0
        if arc_constraints is not None and hasattr(arc_constraints, 'quick_cert_proxy'):
            try:
                cert_bonus = float(arc_constraints.quick_cert_proxy(program))
            except Exception:
                cert_bonus = 0.0
        score = (self.cfg.w_prior * prior_mass +
                 self.cfg.w_mdl * (-mdl) +
                 self.cfg.w_energy * energy +
                 self.cfg.w_cert * cert_bonus +
                 self.cfg.w_relmem * relmem_boost)
        return score

    def _meta_adapt_on_demos(self, demos):
        try:
            from meta_learner import create_meta_learner
            meta = create_meta_learner(self.model, inner_lr=self.cfg.meta_inner_lr, device=self.device)
            meta.adapt_on_demos(demos, steps=self.cfg.meta_inner_steps)
        except Exception:
            pass

    def _d4_apply(self, grid, rot, ref):
        try:
            from trainers.orbits import d4_apply
            return d4_apply(grid, rot, ref)
        except Exception:
            return grid

    def _d4_inverse(self, rot, ref):
        inv_rot = (4 - (rot % 4)) % 4
        inv_ref = ref
        return inv_rot, inv_ref
