#!/usr/bin/env python3
"""
Checkpoint Management for Kaggle Submission
Tracks best validation EM and manages checkpoint lifecycle
"""

import torch
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages checkpoint saving and best model tracking.

    Strategy:
    - Save every N epochs starting at start_epoch
    - Track best validation EM across all saved checkpoints
    - Provide best checkpoint selection for inference
    """

    def __init__(
        self,
        save_dir: str = 'checkpoints',
        start_epoch: int = 100,
        save_every: int = 4,
        metric_name: str = 'val_em'
    ):
        """
        Args:
            save_dir: Directory to save checkpoints
            start_epoch: Epoch to start saving (default: 100)
            save_every: Save frequency in epochs (default: 4)
            metric_name: Metric to track for best model (default: 'val_em')
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)

        self.start_epoch = start_epoch
        self.save_every = save_every
        self.metric_name = metric_name

        # Track best across session
        self.best_metric = 0.0
        self.best_epoch = None
        self.best_path = None

        logger.info(
            f"[CheckpointManager] Initialized: "
            f"save_dir={save_dir}, start_epoch={start_epoch}, "
            f"save_every={save_every}, metric={metric_name}"
        )

    def should_save(self, epoch: int) -> bool:
        """Check if checkpoint should be saved at this epoch"""
        return epoch >= self.start_epoch and epoch % self.save_every == 0

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        additional_state: Optional[Dict] = None
    ) -> Path:
        """
        Save checkpoint with metadata.

        Args:
            model: Model to save
            optimizer: Optimizer state to save
            epoch: Current epoch
            metrics: Dict of metrics (must contain self.metric_name)
            additional_state: Optional additional state to save

        Returns:
            Path to saved checkpoint
        """
        if not self.should_save(epoch):
            return None

        # Build checkpoint dict
        checkpoint = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
        }

        # Add additional state if provided
        if additional_state:
            checkpoint.update(additional_state)

        # Save checkpoint
        checkpoint_path = self.save_dir / f'checkpoint_epoch{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)

        logger.info(
            f"[Checkpoint] Saved epoch {epoch}: "
            f"{self.metric_name}={metrics.get(self.metric_name, 0.0):.4f} "
            f"→ {checkpoint_path}"
        )

        # Check if this is new best
        current_metric = metrics.get(self.metric_name, 0.0)
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.best_epoch = epoch
            self.best_path = checkpoint_path

            logger.info(
                f"🏆 [Checkpoint] NEW BEST: "
                f"{self.metric_name}={current_metric:.4f} @ epoch {epoch}"
            )

        return checkpoint_path

    def get_best_checkpoint(self) -> Tuple[Optional[Path], float]:
        """
        Find best checkpoint by scanning all saved checkpoints.

        Returns:
            (best_checkpoint_path, best_metric_value)
        """
        checkpoints = list(self.save_dir.glob('checkpoint_epoch*.pt'))

        if not checkpoints:
            logger.warning("[Checkpoint] No checkpoints found")
            return None, 0.0

        best_ckpt = None
        best_metric = 0.0

        for ckpt_path in checkpoints:
            try:
                # Load metadata only (map_location='cpu' for speed)
                meta = torch.load(ckpt_path, map_location='cpu')

                # Extract metric
                metrics = meta.get('metrics', {})
                current_metric = metrics.get(self.metric_name, 0.0)

                # Update best
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_ckpt = ckpt_path

            except Exception as e:
                logger.warning(f"[Checkpoint] Failed to load {ckpt_path}: {e}")
                continue

        if best_ckpt:
            logger.info(
                f"✅ [Checkpoint] Best found: "
                f"{best_ckpt.name} ({self.metric_name}={best_metric:.4f})"
            )

        return best_ckpt, best_metric

    def load_best(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cuda'
    ) -> Dict:
        """
        Load best checkpoint into model and optimizer.

        Args:
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            device: Device to load tensors to

        Returns:
            Full checkpoint dict
        """
        best_ckpt, best_metric = self.get_best_checkpoint()

        if best_ckpt is None:
            raise FileNotFoundError("No best checkpoint found")

        # Load checkpoint
        checkpoint = torch.load(best_ckpt, map_location=device)

        # Load model state
        model.load_state_dict(checkpoint['model_state'])
        logger.info(f"[Checkpoint] Loaded model from {best_ckpt.name}")

        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            logger.info(f"[Checkpoint] Loaded optimizer state")

        return checkpoint

    def cleanup_old_checkpoints(self, keep_n: int = 5):
        """
        Remove old checkpoints, keeping only the N most recent.

        Args:
            keep_n: Number of recent checkpoints to keep
        """
        checkpoints = sorted(
            self.save_dir.glob('checkpoint_epoch*.pt'),
            key=lambda p: int(p.stem.split('epoch')[-1])
        )

        if len(checkpoints) <= keep_n:
            return

        # Remove oldest
        for ckpt in checkpoints[:-keep_n]:
            ckpt.unlink()
            logger.info(f"[Checkpoint] Removed old checkpoint: {ckpt.name}")


def integrate_checkpoint_manager(trainer_args):
    """
    Helper to integrate CheckpointManager into training script.

    Usage in train_parent.py:
        ckpt_mgr = integrate_checkpoint_manager(args)

        for epoch in range(args.epochs):
            ... train ...

            metrics = {'val_em': val_em, 'val_acc': val_acc}
            ckpt_mgr.save(model, optimizer, epoch, metrics)
    """
    return CheckpointManager(
        save_dir=getattr(trainer_args, 'checkpoint_dir', 'checkpoints'),
        start_epoch=getattr(trainer_args, 'checkpoint_start_epoch', 100),
        save_every=getattr(trainer_args, 'checkpoint_every', 4),
        metric_name=getattr(trainer_args, 'checkpoint_metric', 'val_em')
    )


if __name__ == '__main__':
    # Test checkpoint manager
    import torch.nn as nn

    logging.basicConfig(level=logging.INFO)

    # Create dummy model
    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters())

    # Create manager
    mgr = CheckpointManager(
        save_dir='test_checkpoints',
        start_epoch=100,
        save_every=4
    )

    # Simulate training
    for epoch in range(98, 115):
        metrics = {
            'val_em': 0.5 + (epoch - 100) * 0.02,  # Increasing EM
            'val_acc': 0.8
        }

        if mgr.should_save(epoch):
            mgr.save(model, optimizer, epoch, metrics)

    # Get best
    best_ckpt, best_em = mgr.get_best_checkpoint()
    print(f"\nBest checkpoint: {best_ckpt} (EM={best_em:.4f})")

    # Cleanup test dir
    import shutil
    shutil.rmtree('test_checkpoints')
