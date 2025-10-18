"""
Phased Training Module for WGO Oracle Pretraining

Implements 3-phase training strategy:
  Phase 1: TOPAS encoder warmup (foundation)
  Phase 2: WGO supervised pretrain on synthetics (oracle learning)
  Phase 3: Joint training with learned WGO oracle (full cooperation)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from enum import IntEnum
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TrainingPhase(IntEnum):
    """Training phase enumeration"""
    TOPAS_WARMUP = 1
    WGO_SUPERVISED = 2
    JOINT_TRAINING = 3


def parse_phases(phase_str: str) -> List[int]:
    """Parse --training-phases CLI argument"""
    return [int(p.strip()) for p in phase_str.split(',')]


def apply_quick_test_mode(cli_args):
    """Override phase settings for quick test"""
    cli_args.phase1_epochs = 2
    cli_args.phase2_epochs = 1
    cli_args.phase2_synthetic_count = 1000
    cli_args.phase3_epochs = 5
    logger.info("[QUICK-TEST] Phase lengths: 2+1+5 epochs")


def apply_full_train_mode(cli_args):
    """Override phase settings for full training"""
    cli_args.phase1_epochs = 10
    cli_args.phase2_epochs = 7
    cli_args.phase2_synthetic_count = 20000
    cli_args.phase3_epochs = 35
    logger.info("[FULL-TRAIN] Phase lengths: 10+7+35 epochs")


# ============ PHASE 1: TOPAS FOUNDATION (ENCODER WARMUP) ============

def run_phase1(topas_model, hrm_model, dataset, cli_args, device, scaler):
    """
    Phase 1: TOPAS encoder warmup to learn meaningful feature extraction.

    Goal: Train encoder/slots/painter to extract good features for WGO to learn from.
    Duration: 5-12 epochs (default 8)
    Trains: TOPAS encoder, slots, painter, RelMem
    Skips: WGO (not needed yet), Bridge (no oracle yet)
    """
    logger.info("\n" + "="*70)
    logger.info("PHASE 1: TOPAS FOUNDATION (ENCODER WARMUP)")
    logger.info("="*70)
    logger.info(f"Epochs: {cli_args.phase1_epochs}, LR: {cli_args.phase1_lr}")
    logger.info(f"Goal: Train encoder to extract meaningful features")
    logger.info("="*70 + "\n")

    # Mark phase for conditional logic in forward pass
    topas_model._current_phase = TrainingPhase.TOPAS_WARMUP
    topas_model._skip_wgo_forward = cli_args.phase1_skip_wgo

    # Phase 1 optimizer (standard TOPAS params)
    phase1_params = list(topas_model.parameters()) + list(hrm_model.parameters())
    optimizer = optim.AdamW(phase1_params, lr=cli_args.phase1_lr, weight_decay=1e-5)

    logger.info(f"[PHASE-1] Optimizer: {sum(p.numel() for p in phase1_params):,} params")
    logger.info(f"[PHASE-1] WGO computation: {'SKIPPED' if cli_args.phase1_skip_wgo else 'ACTIVE'}")

    # Simplified training loop - just encoder warmup, no complexity
    global_step = 0
    for epoch in range(cli_args.phase1_epochs):
        logger.info(f"\n[PHASE-1] Epoch {epoch+1}/{cli_args.phase1_epochs}")
        epoch_loss = []

        dataloader = DataLoader(dataset, batch_size=None, num_workers=0)
        progress = tqdm(dataloader, desc=f"Phase1-Epoch{epoch+1}")

        for batch_idx, batch in enumerate(progress):
            # Unpack batch
            demos, test_inputs, test_outputs, task_id = batch[:4]

            # Get first test input/output
            if isinstance(test_inputs, (list, tuple)) and len(test_inputs) > 0:
                test_grid = test_inputs[0].to(device)
            else:
                test_grid = test_inputs.to(device) if torch.is_tensor(test_inputs) else None

            if isinstance(test_outputs, (list, tuple)) and len(test_outputs) > 0:
                target_grid = test_outputs[0].to(device)
            else:
                target_grid = test_outputs.to(device) if torch.is_tensor(test_outputs) else None

            if test_grid is None or target_grid is None:
                continue

            # Ensure batch dimension
            if test_grid.dim() == 2:
                test_grid = test_grid.unsqueeze(0)
            if target_grid.dim() == 2:
                target_grid = target_grid.unsqueeze(0)

            # SIMPLE FORWARD (no auxiliary losses, no metrics)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=True):
                outputs = topas_model.forward_simple_warmup(test_grid, target_grid)

            loss = outputs['loss']

            # Backward (just CE loss)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(topas_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss.append(loss.item())
            global_step += 1

            # Early stop for quick test
            if hasattr(cli_args, 'quick_test') and cli_args.quick_test and batch_idx >= 50:
                break

        avg_loss = sum(epoch_loss) / max(len(epoch_loss), 1)
        logger.info(f"[PHASE-1] Epoch {epoch+1} complete: avg_loss={avg_loss:.4f} (simple CE only)")

    # Save Phase 1 checkpoint
    torch.save({
        'topas_model': topas_model.state_dict(),
        'hrm_model': hrm_model.state_dict(),
        'epoch': cli_args.phase1_epochs,
        'phase': 1
    }, cli_args.phase1_checkpoint)

    logger.info(f"[PHASE-1] Complete - Saved to {cli_args.phase1_checkpoint}")
    logger.info(f"[PHASE-1] Encoder trained for {cli_args.phase1_epochs} epochs")

    return global_step


# ============ PHASE 2: WGO SUPERVISED PRETRAIN ============

def run_phase2(topas_model, cli_args, device):
    """
    Phase 2: WGO supervised learning on synthetic tasks.

    Goal: Train WGO heads to predict ops/params/size/symmetry from encoder features.
    Duration: 3-8 epochs (default 5)
    Trains: WGO heads (program, size, symmetry, histogram, critic)
    Freezes: TOPAS encoder/slots (preserve Phase 1 learning)
    Data: Synthetic tasks with ground-truth labels
    """
    logger.info("\n" + "="*70)
    logger.info("PHASE 2: WGO SUPERVISED PRETRAIN")
    logger.info("="*70)
    logger.info(f"Epochs: {cli_args.phase2_epochs}, LR: {cli_args.phase2_lr}")
    logger.info(f"Synthetic tasks: {cli_args.phase2_synthetic_count}")
    logger.info(f"Goal: Train WGO to predict ops/params/size/symmetry")
    logger.info("="*70 + "\n")

    # Freeze TOPAS components (preserve Phase 1 learning)
    logger.info("[PHASE-2] Freezing TOPAS encoder/slots to preserve learned features...")
    topas_model.encoder.requires_grad_(False)
    topas_model.slots.requires_grad_(False)
    topas_model.painter.requires_grad_(False)

    # Unfreeze WGO heads for training
    logger.info("[PHASE-2] Unfreezing WGO heads for supervised learning...")
    topas_model.pretrainer.program_head.requires_grad_(True)
    topas_model.pretrainer.size_class_head.requires_grad_(True)
    topas_model.pretrainer.symmetry_class_head.requires_grad_(True)
    topas_model.pretrainer.color_histogram_head.requires_grad_(True)
    topas_model.pretrainer.critic_head.requires_grad_(True)

    # Generate synthetic dataset
    logger.info(f"[PHASE-2] Generating {cli_args.phase2_synthetic_count} synthetic tasks...")
    from synthetic_curriculum import make_supervised_task

    synthetic_data = []
    for i in range(cli_args.phase2_synthetic_count):
        try:
            task = make_supervised_task(
                max_depth=cli_args.phase2_synthetic_depth,
                pretrainer_vocab=topas_model.pretrainer.op_vocab
            )
            synthetic_data.append(task)
        except Exception as e:
            if i < 10:  # Log first few errors
                logger.warning(f"[PHASE-2] Synthetic generation failed (task {i}): {e}")

    logger.info(f"[PHASE-2] Generated {len(synthetic_data)} synthetic tasks")

    # Split train/validation
    split_idx = int(len(synthetic_data) * (1 - cli_args.phase2_validation_split))
    train_data = synthetic_data[:split_idx]
    val_data = synthetic_data[split_idx:]
    logger.info(f"[PHASE-2] Train: {len(train_data)}, Val: {len(val_data)}")

    # WGO optimizer (only WGO head params)
    wgo_params = []
    wgo_params.extend(topas_model.pretrainer.program_head.parameters())
    wgo_params.extend(topas_model.pretrainer.size_class_head.parameters())
    wgo_params.extend(topas_model.pretrainer.symmetry_class_head.parameters())
    wgo_params.extend(topas_model.pretrainer.color_histogram_head.parameters())
    wgo_params.extend(topas_model.pretrainer.critic_head.parameters())

    optimizer = optim.Adam(wgo_params, lr=cli_args.phase2_lr)
    logger.info(f"[PHASE-2] WGO optimizer: {sum(p.numel() for p in wgo_params):,} params")

    # Training loop
    for epoch in range(cli_args.phase2_epochs):
        topas_model.pretrainer.train()  # Training mode for WGO
        epoch_loss = 0.0
        epoch_components = {'program': 0, 'size': 0, 'symmetry': 0, 'histogram': 0, 'critic': 0}
        batch_count = 0

        # Batch the data
        for batch_start in tqdm(range(0, len(train_data), cli_args.phase2_batch_size),
                               desc=f"Phase2-Epoch{epoch+1}"):
            batch_tasks = train_data[batch_start:batch_start + cli_args.phase2_batch_size]

            # Get features from frozen encoder
            inputs = torch.stack([t['input'].to(device) for t in batch_tasks])

            with torch.no_grad():
                # Forward through frozen encoder/slots to get brain
                enc_in = inputs.unsqueeze(1).float() / 9.0  # [B, 1, H, W]
                feat, glob = topas_model.encoder(enc_in)
                slot_vecs, _, _ = topas_model.slots(feat)
                slots_rel = topas_model.reln(slot_vecs)
                pooled = slots_rel.mean(dim=1)
                brain = torch.cat([glob, pooled], dim=-1)  # [B, ctrl_dim]

            # WGO predictions (WITH gradients - no torch.no_grad here!)
            # Note: Pass full batch, not just inputs[0]
            wgo_preds = topas_model.pretrainer(demos=[], test={'input': inputs})

            # Prepare targets
            targets = {
                'program_tokens': torch.stack([t['program_tokens'].to(device) for t in batch_tasks]),
                'size_class': torch.tensor([t['size_class'] for t in batch_tasks], device=device),
                'symmetry_class': torch.tensor([t['symmetry_class'] for t in batch_tasks], device=device),
                'color_histogram': torch.stack([t['color_histogram'].to(device) for t in batch_tasks]),
                'critic': torch.tensor([t['critic'] for t in batch_tasks], device=device)
            }

            # Supervised loss
            loss, components = topas_model.pretrainer.compute_loss(wgo_preds, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wgo_params, 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            for key in components:
                if key in epoch_components and torch.is_tensor(components[key]):
                    epoch_components[key] += components[key].item()
            batch_count += 1

        # Validation
        val_loss, val_acc = validate_wgo_phase2(topas_model, val_data, device)

        avg_loss = epoch_loss / max(batch_count, 1)
        avg_components = {k: v / max(batch_count, 1) for k, v in epoch_components.items()}

        logger.info(f"[PHASE-2] Epoch {epoch+1}/{cli_args.phase2_epochs}: "
                   f"train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.2%}")
        logger.info(f"[PHASE-2]   Components: program={avg_components.get('program', 0):.4f}, "
                   f"size={avg_components.get('size', 0):.4f}, "
                   f"symmetry={avg_components.get('symmetry', 0):.4f}")

    # Save learned WGO
    torch.save(topas_model.pretrainer.state_dict(), cli_args.phase2_checkpoint)
    logger.info(f"[PHASE-2] Complete - Saved learned WGO to {cli_args.phase2_checkpoint}")
    logger.info(f"[PHASE-2] WGO now provides LEARNED priors (not random!)")


def validate_wgo_phase2(topas_model, val_data, device):
    """Validate WGO on held-out synthetic tasks"""
    topas_model.pretrainer.eval()

    total_loss = 0.0
    correct_predictions = {'program': 0, 'size': 0, 'symmetry': 0}
    total_count = 0

    with torch.no_grad():
        for task in val_data[:min(len(val_data), 200)]:  # Sample for speed
            # Get brain features
            inp = task['input'].unsqueeze(0).to(device)
            enc_in = inp.unsqueeze(1).float() / 9.0
            feat, glob = topas_model.encoder(enc_in)
            slot_vecs, _, _ = topas_model.slots(feat)
            slots_rel = topas_model.reln(slot_vecs)
            pooled = slots_rel.mean(dim=1)
            brain = torch.cat([glob, pooled], dim=-1)

            # WGO predictions
            wgo_preds = topas_model.pretrainer(demos=[], test={'input': inp[0]})

            # Targets (single sample)
            targets = {
                'program_tokens': task['program_tokens'].unsqueeze(0).to(device),
                'size_class': torch.tensor([task['size_class']], device=device),
                'symmetry_class': torch.tensor([task['symmetry_class']], device=device),
                'color_histogram': task['color_histogram'].unsqueeze(0).to(device),
                'critic': torch.tensor([task['critic']], device=device)
            }

            # Loss
            loss, _ = topas_model.pretrainer.compute_loss(wgo_preds, targets)
            total_loss += loss.item()

            # Accuracy (first token after START)
            if 'program_tokens' in wgo_preds:
                pred_tokens = wgo_preds['program_tokens'].argmax(dim=-1)[0, 1]  # First op
                true_tokens = targets['program_tokens'][0, 1]
                if pred_tokens == true_tokens:
                    correct_predictions['program'] += 1

            if 'size_class' in wgo_preds:
                pred_size = wgo_preds['size_class'].argmax(dim=-1)[0]
                if pred_size == targets['size_class'][0]:
                    correct_predictions['size'] += 1

            if 'symmetry_class' in wgo_preds:
                pred_sym = wgo_preds['symmetry_class'].argmax(dim=-1)[0]
                if pred_sym == targets['symmetry_class'][0]:
                    correct_predictions['symmetry'] += 1

            total_count += 1

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = sum(correct_predictions.values()) / (3 * max(total_count, 1))

    return avg_loss, avg_acc


# ============ PHASE 3: JOINT TRAINING WITH LEARNED ORACLE ============

def run_phase3(topas_model, hrm_model, dataset, cli_args, device, scaler, global_step_start=0):
    """
    Phase 3: Joint training with learned WGO oracle providing intelligent priors.

    Goal: All components cooperate, WGO provides learned (not random) priors.
    Duration: 20-40 epochs (default 30)
    Trains: TOPAS, HRM, Bridge, RelMem, Policy/Value
    Freezes: WGO heads (learned teacher)
    Uses: WGO priors in search, bridge learning, distillation
    """
    logger.info("\n" + "="*70)
    logger.info("PHASE 3: JOINT TRAINING WITH LEARNED WGO ORACLE")
    logger.info("="*70)
    logger.info(f"Epochs: {cli_args.phase3_epochs}, LR: {cli_args.phase3_lr}")
    logger.info(f"Goal: Full three-brain cooperation with LEARNED WGO priors")
    logger.info("="*70 + "\n")

    # Unfreeze TOPAS for continued training
    logger.info("[PHASE-3] Unfreezing TOPAS encoder/slots for continued learning...")
    topas_model.encoder.requires_grad_(True)
    topas_model.slots.requires_grad_(True)
    topas_model.painter.requires_grad_(True)

    # Freeze WGO as learned teacher
    if cli_args.phase3_freeze_wgo:
        logger.info("[PHASE-3] Freezing WGO heads as learned teacher...")
        topas_model.pretrainer.requires_grad_(False)
        topas_model.pretrainer.eval()
    else:
        logger.info("[PHASE-3] WGO heads continue training (not frozen)")

    # Mark phase
    topas_model._current_phase = TrainingPhase.JOINT_TRAINING
    topas_model._skip_wgo_forward = False  # Enable WGO oracle

    # NOTE: Rest of Phase 3 uses the standard main() training loop
    # This function just sets up the state, then returns to main() to continue
    logger.info("[PHASE-3] Setup complete - Continuing with standard training loop")
    logger.info(f"[PHASE-3] WGO will provide LEARNED priors (trained in Phase 2)")

    return global_step_start


# ============ HELPER FUNCTIONS ============

def freeze_module_by_name(model, module_names: List[str], freeze: bool = True):
    """Freeze specific modules by name"""
    for name, module in model.named_modules():
        if any(target in name for target in module_names):
            module.requires_grad_(not freeze)


def get_trainable_param_count(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_phase_transition(from_phase: int, to_phase: int):
    """Log phase transition with clear visual separator"""
    logger.info("\n" + "🔄" * 35)
    logger.info(f"PHASE TRANSITION: Phase {from_phase} → Phase {to_phase}")
    logger.info("🔄" * 35 + "\n")
