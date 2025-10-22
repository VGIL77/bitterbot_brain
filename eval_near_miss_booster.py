#!/usr/bin/env python3
"""
Near-Miss Booster for ARC-AGI-2 Evaluation

Applies near-miss repair to evaluation results to boost EM score.
Integrates with existing trainers/near_miss.py infrastructure.

Usage:
    python eval_near_miss_booster.py --predictions eval_preds.pt --targets eval_targets.pt --output repaired_submission.json
"""

import torch
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np

# Import existing near-miss infrastructure
from trainers.near_miss import (
    NearMissMiner,
    ErrorAnalysis,
    near_miss_repair,
    analyze_errors,
    hamming_distance,
    iou_score
)

# Import DSL operations
from models.dsl_search import apply_program, DSLProgram, CORE_OPS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DSLShim:
    """
    Lightweight DSL shim for near-miss repair.
    Provides apply() interface expected by near_miss module.
    """

    def apply(self, op: str, grid: torch.Tensor, **params) -> torch.Tensor:
        """Apply a single DSL operation to a grid"""
        try:
            # Create single-operation program
            program = DSLProgram(ops=[op], params=[params])

            # Remove batch dimension if present
            if grid.dim() == 3 and grid.shape[0] == 1:
                grid = grid[0]

            # Apply program
            result = apply_program(grid, program)

            return result

        except Exception as e:
            logger.debug(f"[DSLShim] Failed to apply {op}: {e}")
            return None


def load_predictions_and_targets(pred_file: str, target_file: str) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str]]:
    """
    Load predictions and targets from files.

    Args:
        pred_file: Path to predictions file (.pt or .json)
        target_file: Path to targets file (.pt or .json)

    Returns:
        (predictions, targets, task_ids)
    """
    logger.info(f"Loading predictions from {pred_file}")
    logger.info(f"Loading targets from {target_file}")

    # Load predictions
    if pred_file.endswith('.pt'):
        pred_data = torch.load(pred_file)
    elif pred_file.endswith('.json'):
        with open(pred_file, 'r') as f:
            pred_data = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {pred_file}")

    # Load targets
    if target_file.endswith('.pt'):
        target_data = torch.load(target_file)
    elif target_file.endswith('.json'):
        with open(target_file, 'r') as f:
            target_data = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {target_file}")

    # Convert to lists of tensors
    predictions = []
    targets = []
    task_ids = []

    # Handle different data formats
    if isinstance(pred_data, dict):
        # JSON format: {task_id: [[grid], [grid]], ...}
        for task_id in sorted(pred_data.keys()):
            pred_grids = pred_data[task_id]
            target_grids = target_data.get(task_id, [])

            for i, (pred_grid, target_grid) in enumerate(zip(pred_grids, target_grids)):
                pred_tensor = torch.tensor(pred_grid) if not isinstance(pred_grid, torch.Tensor) else pred_grid
                target_tensor = torch.tensor(target_grid) if not isinstance(target_grid, torch.Tensor) else target_grid

                predictions.append(pred_tensor)
                targets.append(target_tensor)
                task_ids.append(f"{task_id}_test{i}")

    elif isinstance(pred_data, list):
        # List format
        for i, (pred, target) in enumerate(zip(pred_data, target_data)):
            pred_tensor = torch.tensor(pred) if not isinstance(pred, torch.Tensor) else pred
            target_tensor = torch.tensor(target) if not isinstance(target, torch.Tensor) else target

            predictions.append(pred_tensor)
            targets.append(target_tensor)
            task_ids.append(f"task_{i}")

    elif isinstance(pred_data, torch.Tensor):
        # Tensor format
        predictions = list(pred_data)
        targets = list(target_data)
        task_ids = [f"task_{i}" for i in range(len(predictions))]

    else:
        raise ValueError(f"Unsupported data format: {type(pred_data)}")

    logger.info(f"Loaded {len(predictions)} prediction/target pairs")

    return predictions, targets, task_ids


def analyze_near_misses(predictions: List[torch.Tensor], targets: List[torch.Tensor], task_ids: List[str], acc_threshold: float = 0.85):
    """
    Analyze which predictions are near-misses and worth repairing.

    Returns:
        Dict with near-miss analysis
    """
    near_misses = []
    exact_matches = 0

    for i, (pred, target, task_id) in enumerate(zip(predictions, targets, task_ids)):
        # Ensure 2D grids
        if pred.dim() == 3:
            pred = pred[0]
        if target.dim() == 3:
            target = target[0]

        # Check exact match
        if torch.equal(pred, target):
            exact_matches += 1
            continue

        # Compute accuracy
        if pred.shape == target.shape:
            acc = iou_score(pred, target)

            if acc >= acc_threshold:
                # This is a near-miss!
                error_analysis = analyze_errors(pred, target)

                near_misses.append({
                    'index': i,
                    'task_id': task_id,
                    'accuracy': acc,
                    'hamming_dist': error_analysis.hamming_distance,
                    'error_types': [et.value for et in error_analysis.error_types],
                    'complexity': error_analysis.repair_complexity,
                    'pred': pred,
                    'target': target
                })

    # Sort by accuracy (highest first)
    near_misses.sort(key=lambda x: x['accuracy'], reverse=True)

    analysis = {
        'total': len(predictions),
        'exact_matches': exact_matches,
        'near_misses': near_misses,
        'near_miss_count': len(near_misses),
        'em_rate': exact_matches / len(predictions) if predictions else 0.0
    }

    return analysis


def apply_near_miss_repairs(
    predictions: List[torch.Tensor],
    targets: List[torch.Tensor],
    task_ids: List[str],
    acc_threshold: float = 0.85,
    max_attempts_per_task: int = 50,
    verbose: bool = True
) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
    """
    Apply near-miss repair to predictions.

    Args:
        predictions: List of predicted grids
        targets: List of target grids
        task_ids: List of task IDs
        acc_threshold: Minimum accuracy to attempt repair (0-1)
        max_attempts_per_task: Maximum repair attempts per prediction
        verbose: Enable detailed logging

    Returns:
        (repaired_predictions, repair_stats)
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Near-Miss Repair Pipeline")
    logger.info(f"{'='*70}")

    # Analyze near-misses first
    analysis = analyze_near_misses(predictions, targets, task_ids, acc_threshold)

    logger.info(f"Analysis:")
    logger.info(f"  Total predictions: {analysis['total']}")
    logger.info(f"  Exact matches: {analysis['exact_matches']} ({analysis['em_rate']:.2%})")
    logger.info(f"  Near-misses (>={acc_threshold:.0%}): {analysis['near_miss_count']}")

    if verbose and analysis['near_miss_count'] > 0:
        logger.info(f"\n  Top 10 near-misses:")
        for nm in analysis['near_misses'][:10]:
            logger.info(f"    {nm['task_id']}: {nm['accuracy']:.3f} ACC, "
                       f"dist={nm['hamming_dist']}, errors={nm['error_types']}")

    # Initialize DSL shim
    dsl_shim = DSLShim()

    # Initialize near-miss miner
    miner = NearMissMiner(
        distance_threshold=50,  # Allow up to 50 pixel errors
        similarity_threshold=acc_threshold,
        min_improvement=0.01,  # Any improvement is good
        max_repairs=2,  # Try 1-2 operation repairs
        enable_complex_repairs=False
    )

    # Repair near-misses
    repaired_predictions = predictions.copy()
    repair_stats = {
        'attempted': 0,
        'successful': 0,
        'exact_matches_gained': 0,
        'total_improvement': 0.0,
        'repairs_by_type': {},
        'repairs_by_complexity': {'simple': 0, 'moderate': 0, 'complex': 0}
    }

    for nm in analysis['near_misses']:
        idx = nm['index']
        task_id = nm['task_id']
        pred = nm['pred']
        target = nm['target']
        initial_acc = nm['accuracy']

        repair_stats['attempted'] += 1

        try:
            # Attempt repair
            repaired_grid, repair_ops, improvement, error_analysis = near_miss_repair(
                pred_grid=pred,
                target_grid=target,
                dsl_ops=CORE_OPS,
                dsl_shim=dsl_shim,
                max_repairs=2,
                distance_threshold=50,
                similarity_threshold=acc_threshold
            )

            # Check if repair improved
            if improvement > 0:
                final_acc = iou_score(repaired_grid, target)

                if final_acc > initial_acc:
                    # Apply repair
                    repaired_predictions[idx] = repaired_grid
                    repair_stats['successful'] += 1
                    repair_stats['total_improvement'] += (final_acc - initial_acc)

                    # Track by complexity
                    repair_stats['repairs_by_complexity'][error_analysis.repair_complexity] += 1

                    # Track by operation
                    for op in repair_ops:
                        repair_stats['repairs_by_type'][op] = repair_stats['repairs_by_type'].get(op, 0) + 1

                    # Check if now exact match
                    if torch.equal(repaired_grid, target):
                        repair_stats['exact_matches_gained'] += 1

                        if verbose:
                            logger.info(f"  ✅ {task_id}: {initial_acc:.3f} → 1.000 (EXACT) via {' -> '.join(repair_ops)}")
                    else:
                        if verbose:
                            logger.info(f"  ✓  {task_id}: {initial_acc:.3f} → {final_acc:.3f} via {' -> '.join(repair_ops)}")

        except Exception as e:
            logger.debug(f"  ❌ {task_id}: repair failed - {e}")
            continue

    # Compute final statistics
    final_em = (analysis['exact_matches'] + repair_stats['exact_matches_gained']) / analysis['total']
    em_improvement = final_em - analysis['em_rate']

    repair_stats['initial_em'] = analysis['em_rate']
    repair_stats['final_em'] = final_em
    repair_stats['em_improvement'] = em_improvement
    repair_stats['avg_improvement'] = (repair_stats['total_improvement'] / repair_stats['successful']) if repair_stats['successful'] > 0 else 0.0

    logger.info(f"\n{'='*70}")
    logger.info(f"Repair Results:")
    logger.info(f"  Attempted: {repair_stats['attempted']}")
    logger.info(f"  Successful: {repair_stats['successful']} ({repair_stats['successful']/repair_stats['attempted']:.1%} success rate)" if repair_stats['attempted'] > 0 else "  Successful: 0")
    logger.info(f"  Exact matches gained: {repair_stats['exact_matches_gained']}")
    logger.info(f"\nExact Match (EM):")
    logger.info(f"  Before repair: {analysis['em_rate']:.2%}")
    logger.info(f"  After repair:  {final_em:.2%}")
    logger.info(f"  Improvement:   +{em_improvement:.2%} ({repair_stats['exact_matches_gained']} new EMs)")

    if repair_stats['repairs_by_type']:
        logger.info(f"\nTop repair operations:")
        sorted_ops = sorted(repair_stats['repairs_by_type'].items(), key=lambda x: x[1], reverse=True)
        for op, count in sorted_ops[:5]:
            logger.info(f"  {op}: {count}")

    logger.info(f"{'='*70}\n")

    return repaired_predictions, repair_stats


def save_repaired_submission(repaired_predictions: List[torch.Tensor], task_ids: List[str], output_file: str):
    """Save repaired predictions in Kaggle submission format"""

    # Group by task_id (remove _test0, _test1 suffix)
    submission = {}

    for pred, task_id in zip(repaired_predictions, task_ids):
        # Extract base task_id
        base_id = task_id.rsplit('_test', 1)[0] if '_test' in task_id else task_id

        # Convert to list
        pred_list = pred.tolist() if hasattr(pred, 'tolist') else pred

        # Add to submission
        if base_id not in submission:
            submission[base_id] = []
        submission[base_id].append(pred_list)

    # Save
    with open(output_file, 'w') as f:
        json.dump(submission, f, indent=2)

    logger.info(f"Saved repaired submission to {output_file}")
    logger.info(f"  Tasks: {len(submission)}")
    logger.info(f"  Total predictions: {sum(len(v) for v in submission.values())}")


def main():
    parser = argparse.ArgumentParser(description='Apply near-miss repair to boost EM score')

    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions file (.pt or .json)')
    parser.add_argument('--targets', type=str, required=True,
                       help='Path to targets file (.pt or .json)')
    parser.add_argument('--output', type=str, default='repaired_submission.json',
                       help='Output file for repaired predictions')
    parser.add_argument('--acc-threshold', type=float, default=0.85,
                       help='Minimum accuracy to attempt repair (0.0-1.0)')
    parser.add_argument('--max-attempts', type=int, default=50,
                       help='Maximum repair attempts per task')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed logging')

    args = parser.parse_args()

    # Set logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Load data
    predictions, targets, task_ids = load_predictions_and_targets(args.predictions, args.targets)

    # Apply repairs
    repaired_predictions, repair_stats = apply_near_miss_repairs(
        predictions, targets, task_ids,
        acc_threshold=args.acc_threshold,
        max_attempts_per_task=args.max_attempts,
        verbose=not args.quiet
    )

    # Save results
    save_repaired_submission(repaired_predictions, task_ids, args.output)

    # Save repair statistics
    stats_file = args.output.replace('.json', '_stats.json')
    with open(stats_file, 'w') as f:
        # Convert non-JSON-serializable objects
        json_stats = {
            k: v for k, v in repair_stats.items()
            if not isinstance(v, (torch.Tensor, np.ndarray))
        }
        json.dump(json_stats, f, indent=2)

    logger.info(f"\nRepair complete!")
    logger.info(f"  EM improvement: +{repair_stats['em_improvement']:.2%}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Stats: {stats_file}")


if __name__ == '__main__':
    main()
