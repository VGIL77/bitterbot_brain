#!/usr/bin/env python3
"""
Evaluate TTT (Test-Time Training) on ARC tasks

Compares performance with and without TTT adaptation on demos.
"""

import torch
import argparse
import logging
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from models.topas_arc_60M import TopasARC60M, ModelConfig
from arc2_dataset_loader import ARC2Dataset
# Use the exact same metric path as training
from train_parent import compute_metrics
from train_parent import NeuroPlannerWrapper

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_model(checkpoint_path, device, ttt_config):
    """Load model from checkpoint with TTT config"""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    # Create model config
    config = ModelConfig()
    config.width = 512
    config.depth = 8
    config.slots = 64
    config.slot_dim = 256
    config.pretraining_mode = True

    # TTT configuration
    config.ttt_enable = ttt_config['enable']
    config.ttt_r = ttt_config['r']
    config.ttt_alpha = ttt_config['alpha']
    config.ttt_steps = ttt_config['steps']
    config.ttt_lr = ttt_config['lr']
    config.ttt_lr_ratio = ttt_config['lr_ratio']

    # Verify config before model creation
    logger.info(f"Creating model with config: width={config.width}, depth={config.depth}, slots={config.slots}, slot_dim={config.slot_dim}")

    # Create TOPAS model
    model = TopasARC60M(config).to(device)

    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        logger.info("✅ TOPAS model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load checkpoint: {e}")
        raise

    # Build HRM (NeuroPlanner) just like training
    hrm_cfg = {
        "batch_size": 1, "seq_len": 900, "vocab_size": 10,
        "num_puzzle_identifiers": 1000, "puzzle_emb_ndim": 0,
        "H_cycles": 3, "L_cycles": 4, "H_layers": 4, "L_layers": 4,
        "hidden_size": 512, "expansion": 3.0, "num_heads": 8,
        "pos_encodings": "rope", "halt_max_steps": 6,
        "halt_exploration_prob": 0.1, "forward_dtype": "bfloat16",
    }
    try:
        hrm_model = NeuroPlannerWrapper(hrm_cfg).to(device)
        hrm_model.eval()
        logger.info("✅ HRM model created")
    except Exception as e:
        logger.warning(f"[HRM] init failed -> continuing without HRM: {e}")
        hrm_model = None

    model.eval()
    return model, hrm_model, config

def evaluate_puzzle(model, hrm_model, task_id, demos, test_inputs, test_outputs, device,
                    enable_ttt=False, use_search=True):
    """
    Evaluate a complete puzzle (may have multiple test cases).
    Puzzle is solved only if ALL test cases are solved.
    Returns 2 attempts per test case (duplicated for now).
    """
    with torch.no_grad():
        # Reset loop counter for each puzzle
        if hasattr(model, '_forward_call_count'):
            model._forward_call_count = {}
        if hasattr(model, '_forward_last_step'):
            model._forward_last_step = -1

        # Clean demos (strip extraneous batch dim)
        demos_clean = []
        for d_in, d_out in demos:
            if d_in.dim() == 3 and d_in.shape[0] == 1:  d_in = d_in.squeeze(0)
            if d_out.dim() == 3 and d_out.shape[0] == 1: d_out = d_out.squeeze(0)
            demos_clean.append((d_in, d_out))

        # Setup model for evaluation
        model.config.ttt_enable = bool(enable_ttt)
        model._last_demos = demos_clean

        # Track puzzle-level results
        num_test_cases = len(test_inputs)
        test_cases_solved = 0
        total_accuracy = 0.0

        # Evaluate each test case in the puzzle
        for test_idx in range(num_test_cases):
            test_input = test_inputs[test_idx].squeeze(0) if test_inputs[test_idx].dim() == 3 else test_inputs[test_idx]
            test_output = test_outputs[test_idx].squeeze(0) if test_outputs[test_idx].dim() == 3 else test_outputs[test_idx]

            # Add batch dim
            ti = test_input.unsqueeze(0) if test_input.dim() == 2 else test_input
            to = test_output.unsqueeze(0) if test_output.dim() == 2 else test_output

            # HRM latents
            hrm_latents = None
            if hrm_model is not None:
                try:
                    hrm_latents = hrm_model.encode(ti, task_id=task_id)
                except Exception as e:
                    logging.warning(f"[HRM] encode failed for {task_id}: {e}")

            # Call compute_metrics (same path as training)
            try:
                metrics = compute_metrics(
                    model,
                    input_grid=ti,
                    target_grid=to,
                    hrm_latents=hrm_latents,
                    demos=demos_clean,
                    enable_ttt=enable_ttt
                )

                # Check if test case solved
                em_ref = float(metrics.get("exact_match_refined", 0.0) or 0.0)
                em_base = float(metrics.get("exact_match", 0.0) or 0.0)
                em = max(em_ref, em_base)
                acc = float(metrics.get("accuracy", 0.0) or 0.0)

                if em >= 0.9999:
                    test_cases_solved += 1
                total_accuracy += acc

            except Exception as e:
                logging.error(f"Test case {test_idx} failed: {e}")
                # Count as unsolved
                pass

        # Puzzle is solved only if ALL test cases are solved
        puzzle_solved = (test_cases_solved == num_test_cases)
        avg_accuracy = total_accuracy / max(1, num_test_cases)

        return {
            'puzzle_solved': puzzle_solved,
            'test_cases_solved': test_cases_solved,
            'total_test_cases': num_test_cases,
            'accuracy': avg_accuracy
        }

def main():
    parser = argparse.ArgumentParser(description='Evaluate TTT on ARC tasks')

    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')

    # TTT configuration
    parser.add_argument('--ttt-enable', action='store_true',
                       help='Enable TTT adaptation')
    parser.add_argument('--ttt-r', type=int, default=8,
                       help='LoRA rank (default: 8)')
    parser.add_argument('--ttt-alpha', type=float, default=16.0,
                       help='LoRA alpha (default: 16.0)')
    parser.add_argument('--ttt-steps', type=int, default=10,
                       help='TTT adaptation steps (default: 10)')
    parser.add_argument('--ttt-lr', type=float, default=1e-3,
                       help='TTT learning rate (default: 1e-3)')
    parser.add_argument('--ttt-lr-ratio', type=float, default=1.0,
                       help='LoRA+ LR ratio (default: 1.0)')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='arc2',
                        choices=['arc1', 'arc2'],
                        help='Dataset family (default: arc2)')
    parser.add_argument('--split', type=str, default='evaluation',
                        choices=['train', 'evaluation'],
                        help='Data split to evaluate (default: evaluation)')
    parser.add_argument('--mode', type=str, default='search+ebr',
                        choices=['search+ebr','ebr_only'],
                        help='Eval mode: full search+EBR (training parity) or EBR only')
    parser.add_argument('--num-tasks', type=int, default=None,
                       help='Number of tasks to evaluate (default: all)')

    # Development mode
    parser.add_argument('--dev-mode', action='store_true',
                       help='Development mode: run on subset for quick testing')
    parser.add_argument('--dev-tasks', type=int, default=20,
                       help='Number of tasks in dev mode (default: 20)')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')

    args = parser.parse_args()

    logger = setup_logging()
    device = torch.device(args.device)

    logger.info("=" * 60)
    logger.info("TTT Evaluation on ARC Tasks")
    logger.info("=" * 60)

    # TTT config
    ttt_config = {
        'enable': args.ttt_enable,
        'r': args.ttt_r,
        'alpha': args.ttt_alpha,
        'steps': args.ttt_steps,
        'lr': args.ttt_lr,
        'lr_ratio': args.ttt_lr_ratio
    }

    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"TTT Enabled: {args.ttt_enable}")
    if args.ttt_enable:
        logger.info(f"  - Rank: {args.ttt_r}")
        logger.info(f"  - Alpha: {args.ttt_alpha}")
        logger.info(f"  - Steps: {args.ttt_steps}")
        logger.info(f"  - LR: {args.ttt_lr}")
    logger.info("")

    # Load model and HRM
    model, hrm_model, config = load_model(args.checkpoint, device, ttt_config)

    # Safety: Disable DreamEngine & RelMem during LoRA adaptation to prevent recursion
    if args.ttt_enable:
        logger.info("Freezing DreamEngine and RelMem for TTT safety...")
        if hasattr(model, "dream_engine") and model.dream_engine is not None:
            for p in model.dream_engine.parameters():
                p.requires_grad_(False)
        if hasattr(model, "dream") and model.dream is not None:
            for p in model.dream.parameters():
                p.requires_grad_(False)
        if hasattr(model, "relmem") and model.relmem is not None:
            for p in model.relmem.parameters():
                p.requires_grad_(False)
        logger.info("✅ Heavy modules frozen for TTT\n")

    # Load dataset (mirror training semantics)
    logger.info(f"Loading {args.dataset}/{args.split} ...")
    if args.dataset == 'arc2':
        if args.split == 'train':
            challenge_file = "arc_2_dataset/training"   # dir with individual JSONs (has solutions embedded)
            solution_file  = None  # Solutions are in the same JSON files
        else:
            # Evaluation split: no public solutions; we will demo-holdout
            challenge_file = "arc_2_dataset/evaluation"   # dir with JSONs
            solution_file  = None
        dataset = ARC2Dataset(challenge_file=challenge_file,
                              solution_file=solution_file,
                              device=args.device)
    else:
        from trainers.arc_dataset_loader import ARCDataset
        dataset = ARCDataset(split=args.split)

    # Determine number of tasks to evaluate
    if args.dev_mode:
        num_eval = args.dev_tasks
        logger.info(f"DEV MODE: Evaluating {num_eval} of {len(dataset)} tasks")
    elif args.num_tasks:
        num_eval = min(args.num_tasks, len(dataset))
        logger.info(f"Evaluating {num_eval} of {len(dataset)} tasks")
    else:
        num_eval = len(dataset)
        logger.info(f"Evaluating ALL {num_eval} tasks")
    logger.info("")

    # Evaluate
    logger.info("Starting evaluation...")
    logger.info("-" * 60)

    results = {
        'puzzles_solved': 0,
        'total_puzzles': 0,
        'total_test_cases': 0,
        'test_cases_solved': 0,
        'accuracies': []
    }

    for idx in range(num_eval):
        try:
            demos, test_inputs, test_outputs, task_id = dataset[idx]

            # Check if we have test outputs
            if not test_outputs or len(test_outputs) == 0:
                # Evaluation split with no public solutions
                # Use demo-holdout: last train pair as pseudo-test
                if args.split == 'evaluation':
                    if len(demos) < 2:
                        logger.info(f"[{idx+1}/{num_eval}] Task {task_id}: not enough demos for holdout, skipping")
                        continue
                    # Hold out last demo as pseudo-test
                    *demos_kept, (pseudo_in, pseudo_out) = demos
                    demos = demos_kept
                    test_inputs = [pseudo_in]
                    test_outputs = [pseudo_out]
                    pseudo_mode = True
                else:
                    logger.info(f"[{idx+1}/{num_eval}] Task {task_id}: no solutions, skipping")
                    continue
            else:
                pseudo_mode = False

        except StopIteration:
            continue
        except Exception as e:
            logger.error(f"[{idx+1}/{num_eval}] Failed to load task: {e}")
            continue

        logger.info(f"[{idx+1}/{num_eval}] Puzzle: {task_id}")
        logger.info(f"  Demos: {len(demos)}, Test cases: {len(test_inputs)}")

        # Evaluate puzzle (handles multiple test cases)
        use_search = (args.mode == 'search+ebr')
        puzzle_results = evaluate_puzzle(
            model, hrm_model, task_id, demos, test_inputs, test_outputs,
            device, enable_ttt=args.ttt_enable, use_search=use_search
        )

        if puzzle_results:
            results['puzzles_solved'] += int(puzzle_results['puzzle_solved'])
            results['total_puzzles'] += 1
            results['total_test_cases'] += puzzle_results['total_test_cases']
            results['test_cases_solved'] += puzzle_results['test_cases_solved']
            results['accuracies'].append(puzzle_results['accuracy'])

            status = "✅ SOLVED" if puzzle_results['puzzle_solved'] else "❌ FAILED"
            logger.info(f"  {status} ({puzzle_results['test_cases_solved']}/{puzzle_results['total_test_cases']} tests, acc: {puzzle_results['accuracy']:.2%})")
            if pseudo_mode:
                logger.info("  (demo-holdout mode)")
        else:
            logger.info(f"  ⚠️  Evaluation failed")

        logger.info("")

    # Summary
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)

    puzzle_em = results['puzzles_solved'] / max(1, results['total_puzzles'])
    test_case_em = results['test_cases_solved'] / max(1, results['total_test_cases'])
    avg_acc = sum(results['accuracies']) / max(1, len(results['accuracies']))

    logger.info(f"Puzzles Solved: {results['puzzles_solved']}/{results['total_puzzles']} ({puzzle_em:.2%})")
    logger.info(f"Test Cases Solved: {results['test_cases_solved']}/{results['total_test_cases']} ({test_case_em:.2%})")
    logger.info(f"Avg Pixel Accuracy: {avg_acc:.2%}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"TTT Enabled: {args.ttt_enable}")

    if args.ttt_enable:
        logger.info(f"TTT Config: r={args.ttt_r}, steps={args.ttt_steps}, lr={args.ttt_lr}")

    logger.info("=" * 60)

if __name__ == "__main__":
    main()
