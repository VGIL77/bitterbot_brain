#!/usr/bin/env python3
"""
Kaggle ARC-II Submission Pipeline
Master orchestrator for training → checkpoint → inference → submission

Usage:
    python kaggle_submit.py
    # Or with custom config:
    python kaggle_submit.py --config configs/kaggle_submission.yaml
"""

import subprocess
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from checkpoint_utils import CheckpointManager


def setup_logging():
    """Configure logging for orchestrator"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/kaggle_submit_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__), log_file


def run_training(config_path='configs/atlas_andromeda.yaml', training_script='run_andromeda_phase4.sh'):
    """
    Run training for 150 epochs with checkpoint management.

    Args:
        config_path: Path to training config YAML
        training_script: Shell script to launch training

    Returns:
        Exit code (0 = success)
    """
    logger = logging.getLogger(__name__)

    logger.info("="*70)
    logger.info("[1/3] STARTING TRAINING (150 epochs)")
    logger.info("="*70)
    logger.info(f"Config: {config_path}")
    logger.info(f"Script: {training_script}")
    logger.info("")

    # Check if training script exists
    training_path = Path(training_script)
    if not training_path.exists():
        logger.error(f"Training script not found: {training_script}")
        return 1

    # Launch training
    try:
        result = subprocess.run(
            ['bash', str(training_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        logger.info("✅ Training completed successfully")
        return 0

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Training failed with exit code {e.returncode}")
        logger.error(f"Output: {e.output}")
        return e.returncode


def find_best_checkpoint():
    """
    Find best checkpoint from training run.

    Returns:
        (best_checkpoint_path, best_em) or (None, 0.0) if not found
    """
    logger = logging.getLogger(__name__)

    logger.info("")
    logger.info("="*70)
    logger.info("[2/3] FINDING BEST CHECKPOINT")
    logger.info("="*70)

    # Create checkpoint manager
    ckpt_mgr = CheckpointManager(
        save_dir='checkpoints',
        start_epoch=100,
        save_every=4
    )

    # Find best
    best_ckpt, best_em = ckpt_mgr.get_best_checkpoint()

    if best_ckpt is None:
        logger.error("❌ No checkpoints found!")
        return None, 0.0

    logger.info(f"✅ Best checkpoint: {best_ckpt.name}")
    logger.info(f"   Validation EM: {best_em:.2%}")
    logger.info("")

    return best_ckpt, best_em


def run_inference(checkpoint_path, output_file='submission.json', enable_market=False, enable_ttt=False):
    """
    Run inference with full neurosymbolic pipeline.

    Args:
        checkpoint_path: Path to best checkpoint
        output_file: Output submission JSON path
        enable_market: Enable HyLa + LMSR market aggregation
        enable_ttt: Enable test-time training

    Returns:
        Exit code (0 = success)
    """
    logger = logging.getLogger(__name__)

    logger.info("="*70)
    logger.info("[3/3] RUNNING INFERENCE (121 tasks)")
    logger.info("="*70)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Output: {output_file}")
    logger.info(f"Markets: {'ENABLED' if enable_market else 'disabled'}")
    logger.info(f"TTT: {'ENABLED' if enable_ttt else 'disabled'}")
    logger.info("")

    # Build command
    cmd = [
        'python', 'eval_with_hyla_market.py',
        '--checkpoint', str(checkpoint_path),
        '--output', output_file,
        '--refine-depth', '6',
        '--refine-simulations', '2000',
        '--refine-c-puct', '1.5',
        '--use-orbit-canon',
        '--certificates', 'hard',
        '--use-ebr',
        '--ebr-iters', '5',
        '--seed', '42'
    ]

    # Add optional flags
    if enable_market:
        cmd.extend([
            '--enable-market',
            '--market-liquidity', '20.0',
            '--hyla-max-depth', '4',
            '--hyla-beam-width', '50'
        ])

    if enable_ttt:
        cmd.extend([
            '--ttt-enable',
            '--ttt-steps', '10',
            '--ttt-lr', '1e-5'
        ])

    # Log full command
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info("")

    # Run inference
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        logger.info("✅ Inference completed successfully")
        logger.info(f"   Submission saved: {output_file}")
        return 0

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Inference failed with exit code {e.returncode}")
        logger.error(f"Output: {e.output}")
        return e.returncode


def validate_submission(submission_path):
    """
    Validate submission.json format.

    Args:
        submission_path: Path to submission JSON

    Returns:
        True if valid, False otherwise
    """
    import json

    logger = logging.getLogger(__name__)

    try:
        with open(submission_path, 'r') as f:
            submission = json.load(f)

        # Basic checks
        assert isinstance(submission, dict), "Submission must be dict"

        # Check format
        for task_id, attempts in submission.items():
            assert isinstance(attempts, list), f"Task {task_id}: attempts must be list"
            assert len(attempts) >= 2, f"Task {task_id}: need at least 2 attempts"

            for attempt in attempts:
                assert isinstance(attempt, list), f"Task {task_id}: attempt must be 2D list"
                for row in attempt:
                    assert isinstance(row, list), f"Task {task_id}: row must be list"
                    for cell in row:
                        assert isinstance(cell, int), f"Task {task_id}: cell must be int"
                        assert 0 <= cell <= 9, f"Task {task_id}: cell must be 0-9"

        logger.info(f"✅ Submission valid: {len(submission)} tasks")
        return True

    except Exception as e:
        logger.error(f"❌ Submission validation failed: {e}")
        return False


def main():
    """Main orchestration pipeline"""

    parser = argparse.ArgumentParser(
        description='Kaggle ARC-II Submission Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline (train + infer)
    python kaggle_submit.py

    # Skip training, use existing checkpoint
    python kaggle_submit.py --skip-training --checkpoint checkpoints/checkpoint_epoch140.pt

    # Enable all bells and whistles
    python kaggle_submit.py --enable-market --enable-ttt
        """
    )

    parser.add_argument('--config', type=str, default='configs/atlas_andromeda.yaml',
                        help='Training config path')
    parser.add_argument('--training-script', type=str, default='run_andromeda_phase4.sh',
                        help='Training launch script')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training, use existing checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Specific checkpoint to use (if skipping training)')
    parser.add_argument('--output', type=str, default='submission.json',
                        help='Output submission file')
    parser.add_argument('--enable-market', action='store_true',
                        help='Enable HyLa + LMSR market aggregation')
    parser.add_argument('--enable-ttt', action='store_true',
                        help='Enable test-time training')

    args = parser.parse_args()

    # Setup logging
    logger, log_file = setup_logging()

    logger.info("="*70)
    logger.info("🚀 KAGGLE ARC-II SUBMISSION PIPELINE")
    logger.info("="*70)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Markets: {args.enable_market}")
    logger.info(f"TTT: {args.enable_ttt}")
    logger.info("")

    # Phase 1: Training (or skip)
    if args.skip_training:
        logger.info("[1/3] SKIPPING TRAINING (using existing checkpoint)")
        if args.checkpoint:
            best_ckpt = Path(args.checkpoint)
            best_em = 0.0  # Unknown
        else:
            logger.error("❌ --skip-training requires --checkpoint")
            return 1
    else:
        exit_code = run_training(args.config, args.training_script)
        if exit_code != 0:
            logger.error("❌ Training failed, aborting")
            return exit_code

        # Phase 2: Find best checkpoint
        best_ckpt, best_em = find_best_checkpoint()
        if best_ckpt is None:
            logger.error("❌ No checkpoint found, aborting")
            return 1

    # Phase 3: Inference
    exit_code = run_inference(
        best_ckpt,
        args.output,
        enable_market=args.enable_market,
        enable_ttt=args.enable_ttt
    )
    if exit_code != 0:
        logger.error("❌ Inference failed")
        return exit_code

    # Validate submission
    logger.info("")
    logger.info("="*70)
    logger.info("VALIDATING SUBMISSION")
    logger.info("="*70)

    valid = validate_submission(args.output)
    if not valid:
        logger.error("❌ Submission validation failed")
        return 1

    # Success!
    logger.info("")
    logger.info("="*70)
    logger.info("✅ PIPELINE COMPLETE!")
    logger.info("="*70)
    logger.info(f"Submission ready: {args.output}")
    logger.info(f"Log file: {log_file}")

    if not args.skip_training:
        logger.info(f"Best checkpoint: {best_ckpt.name} (EM={best_em:.2%})")

    logger.info("")
    logger.info("Next steps:")
    logger.info(f"  1. Review {args.output}")
    logger.info(f"  2. Upload to Kaggle")
    logger.info(f"  3. Check logs in {log_file}")

    return 0


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logging.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)
