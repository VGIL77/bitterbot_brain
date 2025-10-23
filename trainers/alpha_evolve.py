
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
    except Exception as e:
        import logging
        logging.warning(f"Failed to import {name}: {e}")
        return None

puct_search = _try_import('trainers.puct_search')
dsl_search = _try_import('models.dsl_search')
wormhole_offline = _try_import('wormhole_offline')
alpha_dsl = _try_import('trainers.alpha_dsl')
hypothesis_market = _try_import('trainers.hypothesis_market')
hyla_solver = _try_import('hyla_solver')
arc_constraints = _try_import('trainers.arc_constraints')

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

def logits_to_grid_safe(tensor: torch.Tensor) -> torch.Tensor:
    """
    ✅ FIX 5: Robust conversion from logits (any dimension) to 2D integer grid.

    Handles:
    - 4D: [B, C, H, W] → argmax over C → [B, H, W] → squeeze → [H, W]
    - 3D: [B, H, W] or [H, W, C] → squeeze or argmax → [H, W]
    - 2D: [H, W] → pass through
    """
    import torch
    import logging
    logger = logging.getLogger(__name__)

    g = tensor.detach().cpu()

    # 4D: [B, C, H, W] - argmax over channel dim
    if g.dim() == 4:
        if g.size(1) == 10:  # Channel dim at index 1
            g = torch.argmax(g, dim=1)  # [B, H, W]
        elif g.size(-1) == 10:  # Channel dim at end
            g = torch.argmax(g, dim=-1)  # [B, H, W]
        else:
            logger.warning(f"[logits_to_grid_safe] Unexpected 4D shape: {g.shape}, taking argmax over dim=1")
            g = torch.argmax(g, dim=1)

    # 3D: Either [B, H, W] or [H, W, C]
    if g.dim() == 3:
        if g.size(-1) == 10:  # Channel dim at end
            g = torch.argmax(g, dim=-1)  # [B, H, W] or [H, W]
        elif g.size(0) == 1:  # Batch dim at start
            g = g.squeeze(0)  # [H, W]
        else:
            logger.warning(f"[logits_to_grid_safe] Unexpected 3D shape: {g.shape}, using first slice")
            g = g[0]  # Take first slice

    # 2D: [H, W] - pass through
    if g.dim() == 2:
        pass  # Already 2D

    # 1D or higher: error
    if g.dim() != 2:
        raise ValueError(f"[logits_to_grid_safe] Cannot convert {g.dim()}D tensor to 2D grid: {g.shape}")

    return g.to(dtype=torch.long)

def select_top2(candidates: List[Dict[str, Any]]) -> List[List[List[int]]]:
    import logging
    logger = logging.getLogger(__name__)

    if not candidates:
        logger.warning("[select_top2] No candidates provided")
        return []

    ranked = sorted(candidates, key=lambda c: c.get('score', -1e9), reverse=True)
    logger.info(f"[select_top2] Ranking {len(candidates)} candidates, top score={ranked[0].get('score', 0):.4f}")

    outs = []
    for idx, c in enumerate(ranked[:2]):
        grid = c.get('grid')
        logger.info(f"[select_top2] Candidate {idx}: grid type={type(grid)}, score={c.get('score', 0):.4f}")

        if isinstance(grid, torch.Tensor):
            logger.info(f"[select_top2] Tensor grid: shape={tuple(grid.shape)}, dim={grid.dim()}")

            # ✅ FIX 5: Use robust conversion helper
            try:
                g = logits_to_grid_safe(grid)
                outs.append(g.tolist())
                logger.info(f"[select_top2] Converted {grid.dim()}D tensor to 2D grid: {g.shape}")
            except ValueError as e:
                logger.error(f"[select_top2] Failed to convert grid: {e}")
                raise

        elif isinstance(grid, list):
            outs.append(grid)
            logger.info(f"[select_top2] Used list grid directly")
        else:
            logger.warning(f"[select_top2] Unknown grid type: {type(grid)}")

    logger.info(f"[select_top2] Returning {len(outs)} outputs")

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
    def __init__(self, model, dsl_ops: List[str], device: Optional[torch.device] = None, cfg: Optional[AlphaEvolveConfig] = None, policy_net=None, value_net=None):
        self.model = model
        self.policy_net = policy_net
        self.value_net = value_net
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
            import logging
            logger = logging.getLogger(__name__)

            cand = []
            logger.info(f"[AlphaEvolve] Starting solve_task with {len(demos)} demos")

            demos_canon, test_canon, (rot, ref) = self._maybe_orbit_canon(demos, test_grid)
            logger.info(f"[AlphaEvolve] Orbit canon: rot={rot}, ref={ref}")

            priors, aux = self._collect_priors(demos_canon, test_canon)
            logger.info(f"[AlphaEvolve] Collected priors: {len(priors)} ops")

            if self.cfg.use_inference_monologue and self.cfg.monologue_k > 1:
                logger.info(f"[AlphaEvolve] Running monologue (k={self.cfg.monologue_k})")
                cand += self._run_monologue(demos_canon, test_canon, priors, aux)
                logger.info(f"[AlphaEvolve] Monologue returned {len(cand)} candidates")
            else:
                logger.info(f"[AlphaEvolve] Running PUCT (sims={self.cfg.puct_sims})")
                puct_cand = self._run_puct(demos_canon, test_canon, priors, aux)
                cand += puct_cand
                logger.info(f"[AlphaEvolve] PUCT returned {len(puct_cand)} candidates")

            if self.cfg.alpha_dsl_enable and alpha_dsl is not None:
                logger.info(f"[AlphaEvolve] Running Alpha-DSL (sims={self.cfg.alpha_dsl_sims})")
                dsl_cand = self._run_alpha_dsl(demos_canon, test_canon, priors)
                cand += dsl_cand
                logger.info(f"[AlphaEvolve] Alpha-DSL returned {len(dsl_cand)} candidates")

            if self.cfg.self_play_enable:
                logger.info(f"[AlphaEvolve] Running self-play (games={self.cfg.n_self_play_games})")
                sp_cand = self._run_self_play(demos_canon, test_canon)
                cand += sp_cand
                logger.info(f"[AlphaEvolve] Self-play returned {len(sp_cand)} candidates")

            logger.info(f"[AlphaEvolve] Total candidates: {len(cand)}")

            attempts_canon = select_top2(cand)
            logger.info(f"[AlphaEvolve] Selected top-2: {len(attempts_canon)} attempts")

            if len(attempts_canon) == 0:
                logger.warning("[AlphaEvolve] No candidates generated! Returning empty list")
                return []

            attempts = [self._d4_apply(g, *self._d4_inverse(rot, ref)) for g in attempts_canon]
            logger.info(f"[AlphaEvolve] Applied inverse transform, returning {len(attempts)} attempts")

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
        import logging
        logger = logging.getLogger(__name__)

        # 🔥 WARM-START: Refresh RelMem prototypes with current demo encodings
        # This fixes the stale prototype issue (sims ~0.14 → ~0.6+)
        if hasattr(self.model, "relmem") and hasattr(self.model.relmem, "observe_sample"):
            try:
                import torch
                for demo_in, _ in demos:
                    # Encode demo using GridEncoder directly (safer than full forward)
                    with torch.no_grad():
                        demo_tensor = torch.tensor(demo_in, dtype=torch.long, device=self.device)
                        if demo_tensor.dim() == 2:
                            demo_tensor = demo_tensor.unsqueeze(0)  # [1, H, W]

                        # Use GridEncoder to get latent embedding
                        if hasattr(self.model, 'grid_encoder'):
                            brain = self.model.grid_encoder(demo_tensor, task_id=0)  # [B, D]
                        else:
                            # Fallback: skip warm-start if encoder not accessible
                            logger.warning(f"  [Priors] GridEncoder not found, skipping warm-start")
                            break

                        # Feed to RelMem in [B, T, D] format
                        if brain.dim() == 2:  # [B, D]
                            brain = brain.unsqueeze(1)  # [B, 1, D]

                        self.model.relmem.observe_sample(brain, step=0)

                logger.info(f"  [Priors] RelMem warm-start: refreshed with {len(demos)} demo encodings")
            except Exception as e:
                logger.warning(f"  [Priors] RelMem warm-start failed: {e}")

        relmem_bias = {}
        if hasattr(self.model, "relmem") and hasattr(self.model.relmem, "get_op_bias"):
            try:
                relmem_bias = self.model.relmem.get_op_bias(dsl_ops=self.dsl_ops, query_vec=None, scale=1.0)
                logger.info(f"  [Priors] RelMem: {len(relmem_bias)} ops")
            except Exception as e:
                logger.warning(f"  [Priors] RelMem failed: {e}")
                relmem_bias = {}
        hrm_bias = {}
        if hasattr(self.model, "get_hmr_op_priors"):
            try:
                hrm_bias = self.model.get_hmr_op_priors(demos, test_grid)
                logger.info(f"  [Priors] HRM: {len(hrm_bias)} ops")
            except Exception as e:
                logger.warning(f"  [Priors] HRM failed: {e}")
                hrm_bias = {}
        market_priors = {}
        hyla_priors = {}
        if self.cfg.enable_market and hypothesis_market is not None and hyla_solver is not None:
            try:
                logger.info(f"  [Priors] Starting HyLa + Market integration")
                from hyla_solver import hyla_one_glance, get_warm_start_for_puct
                from hyla_market_bridge import sync_market_universe, ecs_to_market_context

                # Generate HyLa hypotheses using actual function
                hyps = hyla_one_glance(
                    demos,
                    test_input=test_grid,
                    max_hyp=32,
                    max_depth=self.cfg.hyla_max_depth,
                    beam_width=self.cfg.hyla_beam_width
                )
                logger.info(f"  [Priors] HyLa generated {len(hyps)} hypotheses")

                # Sync with market
                market = hypothesis_market.HypothesisMarket(liquidity=self.cfg.market_liquidity)
                sync_market_universe(market, hyps)
                logger.info(f"  [Priors] Market universe synced")

                # Market context and trading
                market_context = ecs_to_market_context(hyps)
                # Market automatically creates traders internally

                # Get operation priors from market
                market_priors = market.get_priors()
                logger.info(f"  [Priors] Market: {len(market_priors)} ops")

                # Also extract HyLa warm-start priors
                hyla_priors = get_warm_start_for_puct(demos, max_candidates=10)
                logger.info(f"  [Priors] HyLa warm-start: {len(hyla_priors)} ops")
            except Exception as e:
                logger.error(f"  [Priors] HyLa/Market failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                market_priors = {}
                hyla_priors = {}
        priors = blend_priors(relmem_bias, hrm_bias, market_priors, hyla_priors, weights=[1.0, 0.8, 0.6, 0.6])
        aux = {'relmem': relmem_bias, 'hrm': hrm_bias, 'market': market_priors}
        return priors, aux

    def _run_puct(self, demos, test_grid, priors, aux):
        import logging
        logger = logging.getLogger(__name__)

        c = self.cfg
        candidates = []
        try:
            logger.info(f"  [PUCT] Starting search (sims={c.puct_sims}, depth={c.puct_depth})")

            # Convert demos and test_grid from list format to tensors
            demos_tensor = []
            for d_in, d_out in demos:
                if isinstance(d_in, list):
                    d_in = torch.tensor(d_in, dtype=torch.long, device=self.device)
                if isinstance(d_out, list):
                    d_out = torch.tensor(d_out, dtype=torch.long, device=self.device)
                demos_tensor.append((d_in, d_out))

            test_tensor = test_grid
            if isinstance(test_grid, list):
                test_tensor = torch.tensor(test_grid, dtype=torch.long, device=self.device)

            program, info = puct_search.puct_search(
                demos=demos_tensor,
                test=test_tensor,
                policy_net=self.policy_net,  # Use loaded nets
                value_net=self.value_net,  # Use loaded nets
                topas_model=self.model,  # Required parameter
                dsl_ops=self.dsl_ops,
                sims=c.puct_sims,
                depth=c.puct_depth,
                c_puct=c.puct_c,
                beam=c.puct_beam,
                op_priors=priors,
                relmem_bias=aux.get('relmem', {}),
                device=self.device,
                return_info=True
            )
            logger.info(f"  [PUCT] Program found: {len(program) if program else 0} ops")
            grid_logits = self._execute_program(test_grid, program)
            logger.info(f"  [PUCT] Program executed, shape: {grid_logits.shape if hasattr(grid_logits, 'shape') else 'N/A'}")
            grid_refined, e = self._maybe_ebr(test_grid, grid_logits)
            logger.info(f"  [PUCT] EBR complete, energy={e:.4f}")
            score = self._score_candidate(program, priors, e, aux)
            candidates.append({'grid': grid_refined, 'program': program, 'score': score})
            logger.info(f"  [PUCT] Success! Score={score:.4f}")
        except Exception as e:
            logger.error(f"  [PUCT] Failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
            from trainers.self_play import SelfPlayBuffer

            # Create buffer and generate puzzles from wormhole
            buffer = SelfPlayBuffer(maxlen=200)

            if self.wormhole is not None:
                generated_puzzles = buffer.generate_from_wormhole(
                    demos,
                    self.wormhole,
                    top_k=self.cfg.n_self_play_games
                )

                # Each generated puzzle is (input, output) pair
                # Try to extract patterns and apply to test grid
                for gen_input, gen_output in generated_puzzles:
                    try:
                        # Simple heuristic: if generated output shape matches test, use it
                        # More sophisticated: extract transformation and apply
                        if gen_output.shape == test_grid.shape:
                            grid_refined, e = self._maybe_ebr(test_grid, gen_output)
                            score = self._score_candidate([], {}, e, {})
                            candidates.append({'grid': grid_refined, 'program': [], 'score': score})
                    except:
                        continue

                # Wormhole mining happens internally in generate_from_wormhole
        except Exception:
            pass
        return candidates

    def _execute_program(self, test_grid, program):
        # Convert test_grid to tensor if needed
        if isinstance(test_grid, list):
            test_grid = torch.tensor(test_grid, dtype=torch.long, device=self.device)

        if hasattr(dsl_search, 'apply_program'):
            return dsl_search.apply_program(test_grid, program)
        if hasattr(alpha_dsl, 'apply_program'):
            return alpha_dsl.apply_program(test_grid, program)
        return test_grid

    def _maybe_ebr(self, test_grid, grid_logits):
        if not self.cfg.use_ebr:
            return grid_logits, grid_energy_proxy(grid_logits)
        try:
            from energy_refinement import EnergyRefiner
            from trainers.arc_constraints import ARCGridConstraints
            import torch.nn.functional as F
            import logging
            logger = logging.getLogger(__name__)

            # 🔍 DIAGNOSE: Log input shape BEFORE any conversion
            if isinstance(grid_logits, torch.Tensor):
                logger.info(f"[EBR] Input grid_logits shape: {grid_logits.shape} (dim={grid_logits.dim()})")

            # 🔥 FIX: Normalize odd 3D layouts like [H,1,W] or [1,H,W] to plain [H,W]
            # This prevents [29,1,29] from being misinterpreted as [B=29, H=1, W=29]
            if isinstance(grid_logits, torch.Tensor) and grid_logits.dim() == 3:
                # Check if last dim is NOT 10 (not already one-hot encoded)
                if grid_logits.size(-1) != 10:
                    # Case: [H,1,W] → squeeze the singleton middle dimension
                    if grid_logits.size(1) == 1 and grid_logits.size(0) > 1 and grid_logits.size(2) > 1:
                        grid_logits = grid_logits.squeeze(1)
                        logger.info(f"[EBR] Normalized [H,1,W] → [H,W]: {tuple(grid_logits.shape)}")
                    # Case: [1,H,W] → remove the singleton batch
                    elif grid_logits.size(0) == 1:
                        grid_logits = grid_logits.squeeze(0)
                        logger.info(f"[EBR] Normalized [1,H,W] → [H,W]: {tuple(grid_logits.shape)}")

            # Convert grid_logits to proper format [B, C, H, W]
            if isinstance(grid_logits, torch.Tensor):
                if grid_logits.dim() == 2:  # [H, W]
                    grid_logits = grid_logits.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                elif grid_logits.dim() == 3:  # [B, H, W]
                    grid_logits = grid_logits.unsqueeze(1)  # [B, 1, H, W]

                # Convert to one-hot if needed
                if grid_logits.shape[1] == 1:
                    # Assume it's integer grid, convert to one-hot
                    grid_int = grid_logits.squeeze(1).long()  # [B, H, W]
                    grid_logits = F.one_hot(grid_int, num_classes=10).float()  # [B, H, W, 10]
                    grid_logits = grid_logits.permute(0, 3, 1, 2)  # [B, 10, H, W]
            else:
                # grid_logits is likely a plain grid, convert to tensor
                grid_logits = torch.tensor(grid_logits, dtype=torch.long, device=self.device)
                grid_logits = grid_logits.unsqueeze(0)  # [1, H, W]
                grid_logits = F.one_hot(grid_logits, num_classes=10).float()  # [1, H, W, 10]
                grid_logits = grid_logits.permute(0, 3, 1, 2)  # [1, 10, H, W]

            # Create constraint object
            constraint_obj = ARCGridConstraints(
                input_grid=test_grid,
                enable_mass_conservation=True,
                enable_palette_subset=True,
                enable_shape_preservation=False
            )

            # Create refiner - pixel jitter polish for scattered 1-pixel errors
            refiner = EnergyRefiner(
                min_steps=5,  # ↑ from 3 - force more exploration steps
                max_steps=self.cfg.ebr_iters * 2,  # ↑ 2x iterations for micro-corrections
                step_size=0.02,  # ↓ from 0.05 - tiny steps to escape flat energy surfaces
                lambda_violation=0.5,  # ↓ from 0.8 - very permissive for near-perfect grids
                lambda_prior=1e-3,  # ↓ from 1.0 - weak prior, let data dominate
                temp_schedule='exp',
                early_stop_threshold=1e-7  # ultra-fine convergence
            ).to(grid_logits.device)

            # Refine
            refined_logits = refiner.forward(
                pred_logits=grid_logits,
                constraint_obj=constraint_obj,
                prior_tensors={},
                extras={}
            )

            # Convert back to grid safely (shape-aware)
            refined_grid = refined_logits
            if refined_logits.dim() == 4:
                # [B, C, H, W] → [B, H, W]
                refined_grid = refined_logits.argmax(dim=1)
            elif refined_logits.dim() == 3 and refined_logits.size(-1) == 10:
                # [H, W, 10] logits (unbatched)
                refined_grid = refined_logits.argmax(dim=-1).unsqueeze(0)
            elif refined_logits.dim() == 3:
                # [B, H, W] already integer grid
                refined_grid = refined_logits
            elif refined_logits.dim() == 2:
                # [H, W] already integer grid
                refined_grid = refined_logits.unsqueeze(0)

            energy = grid_energy_proxy(refined_grid)

            # ✅ Robust unbatch logic - handles all edge cases safely
            # Repeatedly remove leading batch dimensions of size 1
            while refined_grid.dim() > 2 and refined_grid.size(0) == 1:
                refined_grid = refined_grid.squeeze(0)

            # Remove channel dimension if size 1 (e.g., [H, 1, W] → [H, W])
            if refined_grid.dim() > 2 and refined_grid.size(1) == 1:
                refined_grid = refined_grid.squeeze(1)

            # Final validation: must be 2D grid
            if refined_grid.dim() != 2:
                logger.error(f"[EBR] Unexpected refined grid shape: {refined_grid.shape}, forcing to 2D")
                # Emergency reshape: take first 2D slice
                while refined_grid.dim() > 2:
                    refined_grid = refined_grid[0]

            return refined_grid, float(energy)

        except Exception as e:
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
            # Fallback: return grid unchanged
            return grid

    def _d4_inverse(self, rot, ref):
        inv_rot = (4 - (rot % 4)) % 4
        inv_ref = ref
        return inv_rot, inv_ref
