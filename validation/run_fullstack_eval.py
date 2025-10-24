#!/usr/bin/env python3
"""
Full-Throttle ARC-II Inference Orchestrator
- Forces the single happy-path: HyLa → Market → TOPAS (Alpha‑Evolve) + TTT + Orbit Canon + PUCT + Alpha‑DSL + Self-Play + EBR
- Produces a Kaggle-ready submission.json
"""

import argparse
import os
import sys
from types import SimpleNamespace
from datetime import datetime
from pathlib import Path

# Make sure we can import the repo modules even if run from a subfolder
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import the repo's evaluation driver (already builds submission dict)
from eval_with_hyla_market import run_evaluation

def build_fullstack_args(args: argparse.Namespace) -> SimpleNamespace:
    """
    Build a 'packed' args object that matches eval_with_hyla_market's expectations.
    We turn on every rail and push strong-but-stable search budgets.
    """
    # Defaults chosen to prioritize EM over speed, while remaining Kaggle-offline friendly.
    return SimpleNamespace(
        # Search / PUCT / DSL
        refine_depth=args.refine_depth,
        refine_simulations=args.refine_simulations,
        refine_c_puct=args.refine_c_puct,
        beam=args.beam,
        # Canon + certificates
        use_orbit_canon=True,
        certificates='hard',
        # HyLa + Market
        hyla_max_depth=args.hyla_max_depth,
        hyla_beam_width=args.hyla_beam_width,
        enable_market=True,
        market_liquidity=args.market_liquidity,
        # Alpha-DSL + Self-Play
        alpha_dsl_enable=True,
        alpha_dsl_sims=args.alpha_dsl_sims,
        alpha_dsl_max_depth=args.alpha_dsl_max_depth,
        self_play_enable=True,
        self_play_games=args.self_play_games,
        # EBR
        use_ebr=True,
        ebr_iters=args.ebr_iters,
        # TTT (LoRA)
        ttt_enable=True,
        ttt_steps=args.ttt_steps,
        ttt_lr=args.ttt_lr,
        # Orchestrator
        alpha_evolve=True,
        # Task filtering
        task_indices=args.task_indices,
        # Experimental features
        enable_near_miss=args.enable_near_miss,
        near_miss_threshold=args.near_miss_threshold,
    )

def main():
    parser = argparse.ArgumentParser(description="Full‑Throttle ARC‑II Inference → Kaggle submission")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best training checkpoint (bundle or state_dict)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"], help="Inference device")
    parser.add_argument("--output", type=str, default="submission.json", help="Output submission filename")
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit evaluation to N tasks (for testing)")
    parser.add_argument("--task-indices", type=str, default=None, help="Comma-separated task indices to run (e.g., '1,5,10' for high-ACC tasks)")
    # Search budgets (tuned for EM)
    parser.add_argument("--refine-depth", type=int, default=10)
    parser.add_argument("--refine-simulations", type=int, default=2500)
    parser.add_argument("--refine-c-puct", type=float, default=1.5)
    parser.add_argument("--beam", type=int, default=48)
    # HyLa + Market
    parser.add_argument("--hyla-max-depth", type=int, default=6)
    parser.add_argument("--hyla-beam-width", type=int, default=64)
    parser.add_argument("--market-liquidity", type=float, default=25.0)
    # Alpha-DSL + Self-Play
    parser.add_argument("--alpha-dsl-sims", type=int, default=600)
    parser.add_argument("--alpha-dsl-max-depth", type=int, default=12)
    parser.add_argument("--self-play-games", type=int, default=4)
    # EBR
    parser.add_argument("--ebr-iters", type=int, default=7)
    # TTT
    parser.add_argument("--ttt-steps", type=int, default=12)
    parser.add_argument("--ttt-lr", type=float, default=2e-3)
    # Experimental features (FAIL-FAST when enabled)
    parser.add_argument("--enable-near-miss", action="store_true", default=False,
                       help="Enable near-miss repair (FAILS LOUD if broken)")
    parser.add_argument("--near-miss-threshold", type=float, default=0.70,
                       help="Minimum ACC to attempt repair (default: 70%%)")
    args = parser.parse_args()

    # Pretty print
    print("="*78)
    print(" FULL‑THROTTLE GOD‑TIER INFERENCE (single happy path) ".center(78, "="))
    print("="*78)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device:     {args.device}")
    print(f"Output:     {args.output}")
    print("-"*78)
    print("Rails: OrbitCanon + TTT + HyLa + Market + PUCT + Alpha‑DSL + Self‑Play + EBR")
    print("Policy: Single orchestrated path (no optional branches)")
    print("-"*78)

    # Ensure a logs directory exists (eval_with_hyla_market writes a detailed log there)
    Path("logs").mkdir(exist_ok=True, parents=True)

    # Build packed args for the lower-level evaluator
    packed = build_fullstack_args(args)

    # Run evaluation and emit the Kaggle file
    submission = run_evaluation(args.checkpoint, args.output, args.device, args=packed, max_tasks=args.max_tasks)

    # Quick sanity check: two attempts per test entry
    n_tasks = len(submission)
    sample_tid = next(iter(submission)) if n_tasks else None
    if sample_tid:
        print(f"\nSample task: {sample_tid} → #preds: {len(submission[sample_tid])}")

    # Summary
    print(f"\n{'='*78}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*78}")
    print(f"Tasks processed: {n_tasks}")
    print(f"Output file: {args.output}")
    print(f"Detailed log: logs/eval_detailed_*.log (latest)")
    print(f"{'='*78}")
    print("\n✅ Ready for Kaggle submission or further analysis")

if __name__ == "__main__":
    main()
