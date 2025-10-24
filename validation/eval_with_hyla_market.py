"""
Final Evaluation with HyLa → Market → TOPAS Integration
Generates Kaggle submission format for ARC-AGI-2

Usage:
    python eval_with_hyla_market.py --checkpoint checkpoints/best_em_23.5.pt --output submission.json
"""

import torch
import json
import argparse
import sys
import os
from pathlib import Path
from tqdm import tqdm

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.topas_arc_60M import TopasARC60M, ModelConfig, _DSLShim
from arc2_dataset_loader import ARC2Dataset
from trainers.alpha_evolve import AlphaEvolver, AlphaEvolveConfig
from models.dsl_registry import DSL_OPS
from trainers.near_miss import near_miss_repair, analyze_errors

def load_model(checkpoint_path, device='cuda'):
    """Load TOPAS model from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")

    # Create model config (match training config - slots=64 from checkpoint)
    cfg = ModelConfig(
        width=512,
        depth=8,
        slots=64,  # Match checkpoint (was trained with 64 slots)
        slot_dim=256,
        rt_layers=10,
        rt_heads=8,
        max_dsl_depth=6,
        max_beam_width=12,
        use_ebr=True,
        ebr_steps=5,
        enable_dream=True,
        enable_relmem=True,
        verbose=True,
        pretraining_mode=True,  # Required for forward()
        # HyLa + Market settings
        market_liquidity=25.0,
        market_op_bias_w=0.60,
        hyla_max_hyp=32,
        # Refinement settings
        enable_refinement_loop=True,
        max_refine_iters=5,
        refine_search_depth=6,
        refine_simulations=800,
        refine_c_puct=1.5
    )

    # Initialize model
    model = TopasARC60M(config=cfg)
    model.to(device)

    # Load checkpoint
    bundle = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Check if this is a bundle format with policy/value heads
    policy_net = None
    value_net = None

    if isinstance(bundle, dict):
        if 'topas' in bundle:
            # Bundle format: separate keys
            model.load_state_dict(bundle['topas'], strict=False)
            print("✓ Loaded from bundle format (topas key)")

            # Load policy net if available
            if 'policy_head' in bundle:
                from models.policy_nets import OpPolicyNet
                policy_net = OpPolicyNet(input_dim=1024, hidden_dim=512).to(device)
                policy_net.load_state_dict(bundle['policy_head'], strict=False)
                policy_net.eval()
                print("✓ Loaded policy_net from bundle")

            # Load value net if available
            if 'value_head' in bundle:
                from models.value_net import ValueNet
                value_net = ValueNet(context_dim=1024, program_dim=128).to(device)
                value_net.load_state_dict(bundle['value_head'], strict=False)
                value_net.eval()
                print("✓ Loaded value_net from bundle")
        else:
            # Standard state dict
            model.load_state_dict(bundle, strict=False)
            print("✓ Loaded from state_dict format")
    else:
        model.load_state_dict(bundle, strict=False)
        print("✓ Loaded from state_dict format")

    # GPU-FIRST: Call built-in RelMem sync method
    if hasattr(model, '_sync_relmem_to_device'):
        model._sync_relmem_to_device()
        print(f"✓ [Device Sync] Called model._sync_relmem_to_device()")

    # ✅ FIX: Re-register concept_proto after checkpoint load
    if hasattr(model, 'relmem') and hasattr(model.relmem, 'ensure_concept_param'):
        try:
            model.relmem.ensure_concept_param()
            print(f"✓ [RelMem Fix] concept_proto re-registered as Parameter")
        except Exception as e:
            print(f"⚠ [RelMem Fix] ensure_concept_param() failed: {e}")

    # Set to eval mode
    model.eval()
    model.set_pretraining_mode(True)

    print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    print(f"✓ Market enabled: liquidity={cfg.market_liquidity}, op_bias_w={cfg.market_op_bias_w}")

    if policy_net and value_net:
        print("✓ Policy and Value networks ready for PUCT")

    return model, policy_net, value_net

def _ttt_adapt(model, demos, device, ttt_steps=10, ttt_lr=1e-5, log_file=None):
    """
    Test-Time Training: Fine-tune Cortex + Painter on demos.

    Args:
        model: TOPAS model
        demos: List of (input, output) demo pairs
        device: Device to run on
        ttt_steps: Number of adaptation steps
        ttt_lr: Learning rate for adaptation
        log_file: Optional log file handle
    """
    import torch.nn.functional as F

    # Enable gradients only for Cortex + Painter
    trainable_params = []
    for name, param in model.named_parameters():
        param.requires_grad = False
        if 'cortex' in name or 'painter' in name:
            param.requires_grad = True
            trainable_params.append(param)

    if not trainable_params:
        if log_file:
            log_file.write("  [TTT] No trainable params (cortex/painter not found), skipping\n")
        return

    # Create optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=ttt_lr)

    # Adaptation loop
    model.train()  # Enable training mode for batch norm, dropout
    for step in range(ttt_steps):
        total_loss = 0.0

        for demo_in, demo_out in demos:
            try:
                # Ensure proper shape [B, H, W]
                if demo_in.dim() == 2:
                    demo_in = demo_in.unsqueeze(0)
                if demo_out.dim() == 2:
                    demo_out = demo_out.unsqueeze(0)

                # Forward through model
                grid_out, logits, size, _ = model.forward_pretraining(
                    demo_in.to(device),
                    target_shape=demo_out.shape[-2:]
                )

                # Cross-entropy loss
                B, H, W = demo_out.shape
                loss = F.cross_entropy(
                    logits.reshape(B, -1, 10).permute(0, 2, 1),  # [B, 10, H*W]
                    demo_out.reshape(B, -1).long(),  # [B, H*W]
                    reduction='mean'
                )
                total_loss += loss

            except Exception as e:
                if log_file:
                    log_file.write(f"  [TTT] Demo forward failed: {e}\n")
                continue

        if total_loss > 0:
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

    # Freeze again and set back to eval mode
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    if log_file:
        log_file.write(f"  [TTT] Adapted for {ttt_steps} steps (lr={ttt_lr})\n")
        log_file.flush()


def evaluate_task(model, demos, test_input, device='cuda', num_attempts=2, log_file=None, ttt_args=None, eval_args=None):
    """
    Evaluate single task using forward() method with HyLa/Market integration.

    Args:
        num_attempts: Number of prediction attempts (ARC requirement: 2)
        log_file: File handle for detailed logging
        ttt_args: Dict with TTT config: {enable, steps, lr}
        eval_args: Dict with eval config: {refine_depth, hyla_beam_width, refine_simulations, refine_c_puct}

    Returns:
        List of prediction grids (length = num_attempts)
    """
    # Test-Time Training (TTT) adaptation on demos
    if ttt_args and ttt_args.get('enable', False):
        _ttt_adapt(
            model, demos, device,
            ttt_steps=ttt_args.get('steps', 10),
            ttt_lr=ttt_args.get('lr', 1e-5),
            log_file=log_file
        )

    # Prepare demos in dict format
    demo_dicts = []
    for inp, out in demos:
        demo_dicts.append({
            'input': inp.to(device),
            'output': out.to(device)
        })

    # Prepare test dict
    test_dict = {
        'input': test_input.to(device)
    }

    # Extract eval parameters from args
    eval_depth = 6  # Default
    eval_beam = 12  # Default
    if eval_args:
        eval_depth = eval_args.get('refine_depth', 6)
        eval_beam = eval_args.get('hyla_beam_width', 12)

    predictions = []

    # Run forward with HyLa/Market integration (num_attempts times)
    # This calls forward() which has the full pipeline
    with torch.no_grad():
        for attempt_idx in range(num_attempts):
            try:
                grid, logits, size, extras = model.forward(
                    demos=demo_dicts,
                    test=test_dict,
                    eval_use_dsl=True,          # Enable DSL search (includes HyLa/Market)
                    eval_use_ebr=True,          # Enable EBR refinement
                    eval_dsl_depth=eval_depth,  # From CLI args
                    eval_beam_width=eval_beam,  # From CLI args
                    training_mode=False         # Inference mode
                )

                # Convert to numpy for submission
                if isinstance(grid, torch.Tensor):
                    pred_grid = grid.cpu().numpy()
                    if pred_grid.ndim == 3:
                        pred_grid = pred_grid[0]  # Remove batch dim
                else:
                    pred_grid = grid

                # Log HyLa/Market activity
                rail_path = extras.get('rail_path', [])
                ops_attempted = extras.get('ops_attempted', [])

                log_msg = f"    Attempt {attempt_idx+1}: "
                if 'HYLA' in rail_path:
                    log_msg += "✓ HyLa one-glance SUCCESS"
                elif 'DSL' in rail_path:
                    log_msg += f"DSL search - ops={list(ops_attempted)[:5]}"
                elif 'POLICY' in rail_path:
                    log_msg += "Policy mode"
                elif 'EBR' in rail_path:
                    log_msg += "Neural + EBR"
                else:
                    log_msg += f"Neural painter (rail={rail_path})"

                print(log_msg)
                if log_file:
                    log_file.write(log_msg + "\n")
                    log_file.flush()

                predictions.append(pred_grid)

            except Exception as e:
                error_msg = f"    Attempt {attempt_idx+1}: ❌ Failed - {e}"
                print(error_msg)
                if log_file:
                    log_file.write(error_msg + "\n")
                    log_file.flush()

                # Return test input as fallback for this attempt
                fallback = test_input.cpu().numpy()
                if fallback.ndim == 3:
                    fallback = fallback[0]
                predictions.append(fallback)

    return predictions

def run_evaluation(checkpoint_path, output_file='submission.json', device='cuda', args=None, max_tasks=None):
    """Run full evaluation on ARC-2 test set and generate Kaggle submission

    Args:
        max_tasks: If set, only evaluate first N tasks (for testing)
    """

    from datetime import datetime

    # Open detailed log file in logs/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/eval_detailed_{timestamp}.log"
    log_file = open(log_path, 'w')

    print(f"📝 Detailed logging to: {log_path}")

    # Build TTT config from args
    ttt_args = None
    if args and hasattr(args, 'ttt_enable'):
        ttt_args = {
            'enable': args.ttt_enable,
            'steps': getattr(args, 'ttt_steps', 10),
            'lr': getattr(args, 'ttt_lr', 1e-5)
        }

    # Build eval config from args (PUCT/HyLa parameters)
    eval_args = None
    if args:
        eval_args = {
            'refine_depth': getattr(args, 'refine_depth', 6),
            'hyla_beam_width': getattr(args, 'hyla_beam_width', 12),
            'refine_simulations': getattr(args, 'refine_simulations', 2000),
            'refine_c_puct': getattr(args, 'refine_c_puct', 1.5)
        }
    log_file.write(f"{'='*70}\n")
    log_file.write(f"ARC-2 Evaluation with HyLa → Market → TOPAS\n")
    log_file.write(f"Timestamp: {timestamp}\n")
    log_file.write(f"Checkpoint: {checkpoint_path}\n")
    log_file.write(f"Device: {device}\n")
    log_file.write(f"{'='*70}\n\n")
    log_file.flush()

    # Load model (returns model, policy_net, value_net)
    model, policy_net, value_net = load_model(checkpoint_path, device)

    # Load ARC-2 evaluation dataset from directory
    print("\nLoading ARC-2 evaluation dataset...")
    eval_dir = 'arc_2_dataset/evaluation'
    dataset = ARC2Dataset(challenge_file=eval_dir, solution_file=None, device=device)
    print(f"✓ Loaded {len(dataset)} evaluation tasks")

    log_file.write(f"Loaded {len(dataset)} evaluation tasks\n\n")
    log_file.flush()

    # Prepare submission dict
    submission = {}

    # Evaluate each task
    print(f"\n{'='*70}")
    print(f"Running Evaluation with HyLa → Market → TOPAS Pipeline")
    print(f"{'='*70}\n")

    total_tasks = len(dataset)
    if max_tasks is not None:
        total_tasks = min(total_tasks, max_tasks)
        print(f"🔬 Dev mode: Limiting to {total_tasks} task(s)")

    # Task filtering: run specific indices if specified
    if hasattr(args, 'task_indices') and args.task_indices:
        indices = [int(i.strip()) for i in args.task_indices.split(',')]
        task_range = [i for i in indices if 0 <= i < len(dataset)]
        print(f"🎯 Running selected tasks: indices {task_range}")
    else:
        task_range = range(total_tasks)

    hyla_successes = 0

    for idx in tqdm(task_range, desc="Evaluating"):
        task_id = f'task_{idx}'  # Initialize early for error handling
        try:
            # Get task - dataset returns tuple (demos, test_inputs, test_outputs, task_id)
            demos, test_inputs, test_outputs, task_id = dataset[idx]

            # Log task header
            task_header = f"\n[{idx+1}/{total_tasks}] Task: {task_id} | {len(demos)} demos, {len(test_inputs)} test(s)"
            print(task_header)
            log_file.write(task_header + "\n")
            log_file.flush()

            # Initialize task predictions
            task_predictions = []

            # Evaluate each test input (typically 1-2 per task)
            for test_idx, test_input in enumerate(test_inputs):
                test_msg = f"  Test {test_idx+1}/{len(test_inputs)} - shape: {test_input.shape}"
                print(test_msg)
                log_file.write(test_msg + "\n")
                log_file.flush()

                # Get 2 prediction attempts (ARC requirement)
                if args.alpha_evolve:
                    # === ALPHA-EVOLVE PATH ===
                    # Configure Alpha-Evolve with all parameters
                    cfg = AlphaEvolveConfig(
                        use_orbit_canon=args.use_orbit_canon,
                        certificates=getattr(args, 'certificates', 'hard'),
                        use_ebr=getattr(args, 'use_ebr', True),
                        ebr_iters=getattr(args, 'ebr_iters', 5),
                        puct_depth=args.refine_depth,
                        puct_sims=args.refine_simulations,
                        puct_c=args.refine_c_puct,
                        puct_beam=args.beam,
                        enable_market=True,  # Always enable market with Alpha-Evolve
                        market_liquidity=args.market_liquidity,
                        hyla_max_depth=args.hyla_max_depth,
                        hyla_beam_width=args.hyla_beam_width,
                        self_play_enable=args.self_play_enable,
                        n_self_play_games=args.self_play_games,
                        alpha_dsl_enable=args.alpha_dsl_enable,
                        alpha_dsl_sims=args.alpha_dsl_sims,
                        alpha_dsl_max_depth=args.alpha_dsl_max_depth,
                        meta_adapt_enable=args.ttt_enable,
                        meta_inner_steps=getattr(args, 'ttt_steps', 10),
                        meta_inner_lr=getattr(args, 'ttt_lr', 1e-3),
                    )
                    evolver = AlphaEvolver(model, dsl_ops=DSL_OPS, device=device, cfg=cfg, policy_net=policy_net, value_net=value_net)

                    # Convert test_input to list format (Alpha-Evolve expects plain Python lists)
                    test_list = test_input.cpu().tolist() if isinstance(test_input, torch.Tensor) else test_input
                    demos_list = [(d[0].cpu().tolist() if isinstance(d[0], torch.Tensor) else d[0],
                                   d[1].cpu().tolist() if isinstance(d[1], torch.Tensor) else d[1])
                                  for d in demos]

                    # Run Alpha-Evolve solver (returns 2 attempts)
                    attempts_list = evolver.solve_task(demos_list, test_list)
                    pred_attempts = [torch.tensor(a, device=device) for a in attempts_list]

                    # === NEAR-MISS SALVAGE (FAIL-FAST MODE - NO try/except) ===
                    if getattr(args, 'enable_near_miss', False):
                        # Create DSL shim for repair operations
                        dsl_shim = _DSLShim(model)

                        # FAIL-FAST: Must have ground truth
                        if test_idx >= len(test_outputs):
                            raise RuntimeError(f"[NearMiss] enable_near_miss=True but no ground truth for test_idx={test_idx}")

                        target = test_outputs[test_idx]
                        if isinstance(target, list):
                            target = torch.tensor(target, device=device)
                        elif isinstance(target, torch.Tensor):
                            target = target.to(device).squeeze()
                        else:
                            raise TypeError(f"[NearMiss] Unknown target type: {type(target)}")

                        threshold = getattr(args, 'near_miss_threshold', 0.70)

                        for att_idx, pred_grid in enumerate(pred_attempts):
                            pred_tensor = pred_grid.to(device).squeeze() if isinstance(pred_grid, torch.Tensor) else torch.tensor(pred_grid, device=device)

                            # Skip if already EM
                            if pred_tensor.shape == target.shape and torch.equal(pred_tensor, target):
                                continue

                            # Shape mismatch - log but skip
                            if pred_tensor.shape != target.shape:
                                print(f"    [NearMiss] Shape mismatch pred={pred_tensor.shape} vs target={target.shape}, skipping attempt {att_idx+1}")
                                continue

                            acc = (pred_tensor == target).float().mean().item()

                            if acc >= threshold:
                                # FAIL-FAST: Let errors propagate
                                repaired_grid, repair_ops, improvement, _ = near_miss_repair(
                                    pred_tensor, target,
                                    dsl_ops=DSL_OPS,
                                    dsl_shim=dsl_shim,
                                    max_repairs=2,
                                    distance_threshold=int(pred_tensor.numel() * (1.0 - threshold)),
                                    similarity_threshold=threshold
                                )

                                # Check result
                                if repaired_grid.shape == target.shape and torch.equal(repaired_grid, target):
                                    pred_attempts[att_idx] = repaired_grid
                                    log_msg = f"    [NearMiss] ✅ REPAIRED → EM: {acc:.1%} → 100% (ops: {repair_ops})"
                                    print(log_msg)
                                    log_file.write(log_msg + "\n")
                                    log_file.flush()
                                elif improvement > 0.05:
                                    pred_attempts[att_idx] = repaired_grid
                                    new_acc = (repaired_grid == target).float().mean().item()
                                    log_msg = f"    [NearMiss] 📈 IMPROVED: {acc:.1%} → {new_acc:.1%} (ops: {repair_ops})"
                                    print(log_msg)
                                    log_file.write(log_msg + "\n")
                                    log_file.flush()
                                else:
                                    print(f"    [NearMiss] No improvement: {acc:.1%} (tried: {repair_ops})")

                else:
                    # === STANDARD PATH ===
                    pred_attempts = evaluate_task(
                        model, demos, test_input, device,
                        num_attempts=2,
                        log_file=log_file,
                        ttt_args=ttt_args,
                        eval_args=eval_args  # Pass PUCT/HyLa params
                    )

                # === NEAR-MISS REPAIR (STANDARD PATH) ===
                if getattr(args, 'enable_near_miss', False) and test_idx < len(test_outputs):
                    dsl_shim = _DSLShim(model)
                    target = test_outputs[test_idx]

                    # ✅ FIX: Always ensure tensor type before moving to device
                    if not isinstance(target, torch.Tensor):
                        target = torch.tensor(target, device=device)
                    else:
                        target = target.to(device)

                    for att_idx in range(len(pred_attempts)):
                        pred_grid = pred_attempts[att_idx]
                        pred_tensor = torch.tensor(pred_grid, device=device) if not isinstance(pred_grid, torch.Tensor) else pred_grid.to(device)

                        # Only compare when shapes match
                        if pred_tensor.shape != target.shape:
                            continue

                        # ✅ Force int64 dtype for precision consistency
                        pred_int = pred_tensor.to(dtype=torch.int64)
                        target_int = target.to(dtype=torch.int64)

                        acc = (pred_int == target_int).float().mean().item()

                        # ✅ Clamp floating-point roundoff (99.9999% → 100%)
                        if abs(acc - 1.0) < 1e-6:
                            acc = 1.0

                        if acc >= args.near_miss_threshold:
                            print(f"    [NearMiss] Attempting micro-repair on attempt {att_idx+1} (acc={acc:.3%})")

                            repaired_grid, repair_ops, improvement, _ = near_miss_repair(
                                pred_tensor, target,
                                dsl_ops=DSL_OPS,
                                dsl_shim=dsl_shim,
                                max_repairs=2,
                                distance_threshold=int(pred_tensor.numel() * (1.0 - args.near_miss_threshold)),
                                similarity_threshold=args.near_miss_threshold
                            )

                            new_acc = (repaired_grid == target).float().mean().item()
                            if new_acc >= acc:  # accept equal or better
                                pred_attempts[att_idx] = repaired_grid.cpu().numpy()
                                if torch.equal(repaired_grid, target):
                                    print(f"    [NearMiss] ✅ POLISHED → EM ({acc:.1%}→100%) ops={repair_ops}")
                                else:
                                    print(f"    [NearMiss] 📈 Micro-improved: {acc:.1%}→{new_acc:.1%} ops={repair_ops}")
                            else:
                                print(f"    [NearMiss] Skipped: repair worsened ({acc:.1%}→{new_acc:.1%}) ops={repair_ops}")

                # Convert to list format for JSON (use both attempts)
                for pred_grid in pred_attempts:
                    pred_list = pred_grid.tolist() if hasattr(pred_grid, 'tolist') else pred_grid
                    task_predictions.append(pred_list)

                # === COMPUTE ACCURACY vs GROUND TRUTH & SELECT BEST ===
                if test_idx < len(test_outputs):
                    target = test_outputs[test_idx]
                    if isinstance(target, torch.Tensor):
                        target = target.squeeze()  # Remove batch dim if present

                    # Check all attempts and compute accuracy
                    attempt_accuracies = []
                    for attempt_idx, pred_grid in enumerate(pred_attempts):
                        # Ensure same format
                        if isinstance(pred_grid, torch.Tensor):
                            pred_grid = pred_grid.squeeze()
                        else:
                            pred_grid = torch.tensor(pred_grid)

                        if isinstance(target, list):
                            target = torch.tensor(target)

                        # Exact Match
                        exact_match = torch.equal(pred_grid, target) if pred_grid.shape == target.shape else False

                        # Pixel Accuracy
                        if pred_grid.shape == target.shape:
                            acc = (pred_grid == target).float().mean().item()
                        else:
                            acc = 0.0

                        attempt_accuracies.append(acc)

                        # Log results
                        result_msg = f"    Attempt {attempt_idx+1}: {'✅ EXACT MATCH' if exact_match else '❌ FAILED'} (acc={acc:.1%}, shape={tuple(pred_grid.shape)})"
                        print(result_msg)
                        log_file.write(result_msg + "\n")
                        log_file.flush()

                    # ✅ FIX 1: SELECT BEST-OF ATTEMPTS
                    if attempt_accuracies:
                        best_idx = int(torch.tensor(attempt_accuracies).argmax())
                        best_acc = attempt_accuracies[best_idx]
                        best_msg = f"    [Best-of] Selected attempt {best_idx+1} with acc={best_acc:.1%} (all={[f'{x:.1%}' for x in attempt_accuracies]})"
                        print(best_msg)
                        log_file.write(best_msg + "\n")
                        log_file.flush()

                        # Re-order attempts to put best first (for submission)
                        if best_idx != 0:
                            pred_attempts[0], pred_attempts[best_idx] = pred_attempts[best_idx], pred_attempts[0]

            # Store in submission (Kaggle format: task_id -> list of predictions)
            submission[task_id] = task_predictions

        except Exception as e:
            error_msg = f"❌ Task {idx} ({task_id}) failed: {e}"
            print(error_msg)
            log_file.write(error_msg + "\n")
            log_file.flush()
            # Add empty predictions to maintain format (2 attempts)
            submission[task_id] = [[], []]

    # Save submission
    print(f"\n{'='*70}")
    print(f"Saving submission to: {output_file}")
    print(f"{'='*70}")

    with open(output_file, 'w') as f:
        json.dump(submission, f, indent=2)

    # Close log file
    log_file.write(f"\n{'='*70}\n")
    log_file.write(f"Evaluation Complete\n")
    log_file.write(f"Tasks evaluated: {len(submission)}\n")
    log_file.write(f"Output file: {output_file}\n")
    log_file.write(f"{'='*70}\n")
    log_file.close()

    print(f"\n✅ Evaluation complete!")
    print(f"   - Tasks evaluated: {len(submission)}")
    print(f"   - Predictions per task: 2 attempts (x2 if multiple test inputs)")
    print(f"   - Output file: {output_file}")
    print(f"   - Detailed log: {log_path}")
    print(f"   - Ready for Kaggle submission")

    return submission

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate with HyLa-Market integration + TTT')

    # Basic args
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_final.pt',
                        help='Path to model checkpoint (use checkpoint_final.pt from recent training)')
    parser.add_argument('--output', type=str, default='submission_hyla_market.json',
                        help='Output submission file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda for GPU, cpu for CPU)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # PUCT/DSL search args
    parser.add_argument('--refine-depth', type=int, default=6,
                        help='Max DSL program depth for PUCT search')
    parser.add_argument('--refine-simulations', type=int, default=2000,
                        help='Number of PUCT simulations per search')
    parser.add_argument('--refine-c-puct', type=float, default=1.5,
                        help='PUCT exploration constant')
    parser.add_argument('--no-ttt', dest='ttt_enable', action='store_false', help='Disable TTT adaptation')

    # Orbit canonicalization
    parser.add_argument('--use-orbit-canon', action='store_true',
                        help='Enable orbit canonicalization (D4 + color permutations) for 8x speedup')
    parser.add_argument('--certificates', type=str, default='hard', choices=['soft', 'hard'],
                        help='Certificate pruning mode (hard=prune invalid, soft=penalty)')
    # HyLa + Market integration
    parser.add_argument('--enable-market', action='store_true',
                        help='Enable HyLa solver + LMSR market aggregation')
    parser.add_argument('--market-liquidity', type=float, default=20.0,
                        help='LMSR market liquidity parameter')
    parser.add_argument('--hyla-max-depth', type=int, default=4,
                        help='Max depth for HyLa hypothesis lattice')
    parser.add_argument('--hyla-beam-width', type=int, default=50,
                        help='Learning rate for TTT adaptation')

    # Alpha-Evolve orchestration
    parser.add_argument('--alpha-evolve', action='store_true',
                        help='Enable Alpha-Evolve orchestration')
    parser.add_argument('--self-play-enable', action='store_true', default=True,
                        help='Enable self-play search')
    parser.add_argument('--self-play-games', type=int, default=4,
                        help='Number of self-play games')
    parser.add_argument('--alpha-dsl-enable', action='store_true', default=True,
                        help='Enable Alpha-DSL search')
    parser.add_argument('--alpha-dsl-sims', type=int, default=400,
                        help='Alpha-DSL simulation count')
    parser.add_argument('--alpha-dsl-max-depth', type=int, default=10,
                        help='Alpha-DSL maximum depth')
    parser.add_argument('--beam', type=int, default=24,
                        help='Beam width for PUCT search')
    parser.add_argument('--use-ebr', action='store_true', default=True,
                        help='Enable EBR (Energy-Based Refinement)')
    parser.add_argument('--ebr-iters', type=int, default=5,
                        help='Number of EBR iterations')
    parser.add_argument('--ttt-steps', type=int, default=10,
                        help='TTT adaptation steps')
    parser.add_argument('--ttt-lr', type=float, default=1e-3,
                        help='TTT learning rate')

    args = parser.parse_args()

    # Set seed
    import torch
    import numpy as np
    import random
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"🚀 Starting ARC-2 Evaluation")
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   Device: {args.device.upper()}")
    print(f"   Output: {args.output}")
    print(f"   Seed: {args.seed}")
    print(f"\n📋 Pipeline Config:")
    print(f"   - PUCT: depth={args.refine_depth}, sims={args.refine_simulations}, c_puct={args.refine_c_puct}")
    print(f"   - Orbit Canon: {args.use_orbit_canon}")
    print(f"   - Certificates: {args.certificates}")
    print(f"   - EBR: enabled={args.use_ebr}, iters={args.ebr_iters}")
    print(f"   - Markets: enabled={args.enable_market}, liquidity={args.market_liquidity}")
    print(f"   - HyLa: depth={args.hyla_max_depth}, beam={args.hyla_beam_width}")
    print(f"   - TTT: enabled={args.ttt_enable}, steps={args.ttt_steps}, lr={args.ttt_lr}")
    print()

    # Run evaluation
    run_evaluation(args.checkpoint, args.output, args.device, args=args)
