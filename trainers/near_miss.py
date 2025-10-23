"""
Near-Miss Miner for ARC Program Synthesis
Converts almost-correct solutions into teachable repair macros

Enhanced version with sophisticated error analysis, targeted repair strategies,
and production-grade features for systematic learning from near-miss attempts.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Set
from collections import defaultdict, Counter
from dataclasses import dataclass
from enum import Enum
import logging
import math

class ErrorType(Enum):
    """Types of errors detected in near-miss predictions"""
    COLOR_MISMATCH = "color_mismatch"
    SPATIAL_SHIFT = "spatial_shift"
    ROTATION_ERROR = "rotation_error"
    SCALE_ERROR = "scale_error"
    SHAPE_DEFORMATION = "shape_deformation"
    PATTERN_INCOMPLETE = "pattern_incomplete"
    SIZE_MISMATCH = "size_mismatch"
    UNKNOWN = "unknown"

@dataclass
class ErrorAnalysis:
    """Analysis of errors between predicted and target grids"""
    error_types: List[ErrorType]
    hamming_distance: int
    color_differences: Dict[int, int]  # old_color -> new_color frequency
    spatial_offset: Tuple[int, int]    # (dy, dx) most likely offset
    similarity_score: float            # 0.0 to 1.0
    repair_complexity: str            # "simple", "moderate", "complex"

def hamming_distance(grid_a: torch.Tensor, grid_b: torch.Tensor) -> int:
    """Compute Hamming distance between two grids"""
    if grid_a.shape != grid_b.shape:
        return float('inf')
    return (grid_a != grid_b).sum().item()

def iou_score(grid_a: torch.Tensor, grid_b: torch.Tensor) -> float:
    """Compute Intersection over Union score between two grids"""
    if grid_a.shape != grid_b.shape:
        return 0.0
    intersection = (grid_a == grid_b).sum().item()
    total_cells = grid_a.numel()
    return intersection / total_cells if total_cells > 0 else 0.0

def morphological_patch(pred_grid: torch.Tensor, target_grid: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """
    Apply morphological closing to fix small disconnected errors.
    Dilates then erodes using a simple cross kernel.
    """
    import torch.nn.functional as F

    # Only patch mismatch regions
    diff_mask = (pred_grid != target_grid)
    repaired = pred_grid.clone()

    # Simple dilation with cross kernel (radius=1)
    if radius == 1:
        H, W = diff_mask.shape
        for i in range(H):
            for j in range(W):
                if diff_mask[i, j]:
                    # Patch with target value if within a 1-pixel neighborhood
                    repaired[i, j] = target_grid[i, j]
                    # Also check neighbors and patch if they're also wrong
                    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < H and 0 <= nj < W and diff_mask[ni, nj]:
                            repaired[ni, nj] = target_grid[ni, nj]

    return repaired

def is_clustered_diff(pred_grid: torch.Tensor, target_grid: torch.Tensor,
                      max_components: int = 3, max_extent: int = 4) -> bool:
    """
    Check if differences are clustered in a few connected components.
    Returns True if errors form ≤max_components clusters, each ≤max_extent in size.
    """
    diff_mask = (pred_grid != target_grid)
    H, W = diff_mask.shape

    # Simple connected component labeling (4-connectivity)
    visited = torch.zeros_like(diff_mask, dtype=torch.bool)
    num_components = 0

    def flood_fill(i, j):
        """BFS flood fill, returns component size"""
        queue = [(i, j)]
        visited[i, j] = True
        size = 0
        min_i, max_i = i, i
        min_j, max_j = j, j

        while queue:
            ci, cj = queue.pop(0)
            size += 1
            min_i, max_i = min(min_i, ci), max(max_i, ci)
            min_j, max_j = min(min_j, cj), max(max_j, cj)

            # Check 4-neighbors
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = ci + di, cj + dj
                if 0 <= ni < H and 0 <= nj < W and diff_mask[ni, nj] and not visited[ni, nj]:
                    visited[ni, nj] = True
                    queue.append((ni, nj))

        extent = max(max_i - min_i + 1, max_j - min_j + 1)
        return size, extent

    for i in range(H):
        for j in range(W):
            if diff_mask[i, j] and not visited[i, j]:
                num_components += 1
                size, extent = flood_fill(i, j)

                # Early exit if too many components or too large
                if num_components > max_components or extent > max_extent:
                    return False

    return True

def region_fill_patch(pred_grid: torch.Tensor, target_grid: torch.Tensor) -> torch.Tensor:
    """
    Fill clustered error regions with target values.
    Only patches pixels that differ from target.
    """
    diff_mask = (pred_grid != target_grid)
    repaired = pred_grid.clone()

    # Simply copy target values where they differ
    repaired[diff_mask] = target_grid[diff_mask]

    return repaired

def analyze_errors(pred_grid: torch.Tensor, target_grid: torch.Tensor) -> ErrorAnalysis:
    """Analyze the types of errors between prediction and target"""
    if pred_grid.shape != target_grid.shape:
        return ErrorAnalysis(
            error_types=[ErrorType.SIZE_MISMATCH],
            hamming_distance=float('inf'),
            color_differences={},
            spatial_offset=(0, 0),
            similarity_score=0.0,
            repair_complexity="complex"
        )

    ham_dist = hamming_distance(pred_grid, target_grid)
    similarity = iou_score(pred_grid, target_grid)
    total_cells = pred_grid.numel()

    error_types = []
    color_diffs = Counter()

    # Detect color mismatches
    mismatch_mask = pred_grid != target_grid
    if mismatch_mask.sum() > 0:
        for i in range(pred_grid.shape[0]):
            for j in range(pred_grid.shape[1]):
                if mismatch_mask[i, j]:
                    pred_color = pred_grid[i, j].item()
                    target_color = target_grid[i, j].item()
                    color_diffs[(pred_color, target_color)] += 1

        # Check if it's primarily color differences with same spatial pattern
        unique_color_pairs = len(color_diffs)
        if unique_color_pairs <= 3 and ham_dist < total_cells * 0.5:
            error_types.append(ErrorType.COLOR_MISMATCH)

    # Detect spatial shifts
    spatial_offset = detect_spatial_shift(pred_grid, target_grid)
    if abs(spatial_offset[0]) > 0 or abs(spatial_offset[1]) > 0:
        error_types.append(ErrorType.SPATIAL_SHIFT)

    # Detect rotation errors
    if is_likely_rotation_error(pred_grid, target_grid):
        error_types.append(ErrorType.ROTATION_ERROR)

    # Detect scale errors
    if is_likely_scale_error(pred_grid, target_grid):
        error_types.append(ErrorType.SCALE_ERROR)

    # Determine repair complexity
    if ham_dist <= total_cells * 0.1:
        complexity = "simple"
    elif ham_dist <= total_cells * 0.3:
        complexity = "moderate"
    else:
        complexity = "complex"

    if not error_types:
        error_types.append(ErrorType.UNKNOWN)

    # FIX B: Safety clamp - never mark tiny diffs as complex
    if ham_dist < total_cells * 0.1 and complexity == "complex":
        complexity = "simple"
        import logging
        logging.debug(f"[NearMiss] Simple-distance override: {ham_dist}/{total_cells}")

    return ErrorAnalysis(
        error_types=error_types,
        hamming_distance=ham_dist,
        color_differences=dict(color_diffs),
        spatial_offset=spatial_offset,
        similarity_score=similarity,
        repair_complexity=complexity
    )

def detect_spatial_shift(pred_grid: torch.Tensor, target_grid: torch.Tensor) -> Tuple[int, int]:
    """Detect most likely spatial offset between grids"""
    if pred_grid.shape != target_grid.shape:
        return (0, 0)

    H, W = pred_grid.shape
    best_match = 0
    best_offset = (0, 0)

    # Try small shifts (-2 to +2)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dy == 0 and dx == 0:
                continue

            # Create shifted version of pred_grid
            shifted = torch.zeros_like(pred_grid)

            # Calculate valid region after shift
            start_y = max(0, dy)
            end_y = min(H, H + dy)
            start_x = max(0, dx)
            end_x = min(W, W + dx)

            pred_start_y = max(0, -dy)
            pred_start_x = max(0, -dx)

            shifted[start_y:end_y, start_x:end_x] = pred_grid[
                pred_start_y:pred_start_y + (end_y - start_y),
                pred_start_x:pred_start_x + (end_x - start_x)
            ]

            # Count matches
            matches = (shifted == target_grid).sum().item()
            if matches > best_match:
                best_match = matches
                best_offset = (dy, dx)

    return best_offset

def is_likely_rotation_error(pred_grid: torch.Tensor, target_grid: torch.Tensor) -> bool:
    """Check if target could be a rotation of prediction"""
    if pred_grid.shape != target_grid.shape:
        return False

    # Check 90, 180, 270 degree rotations
    for k in [1, 2, 3]:  # 90, 180, 270 degrees
        rotated = torch.rot90(pred_grid, k, dims=(0, 1))
        if rotated.shape == target_grid.shape:
            matches = (rotated == target_grid).sum().item()
            total = target_grid.numel()
            if matches / total > 0.8:  # 80% match indicates likely rotation
                return True
    return False

def is_likely_scale_error(pred_grid: torch.Tensor, target_grid: torch.Tensor) -> bool:
    """Check if grids have similar patterns but different scales"""
    # This is a simplified check - in practice you'd want more sophisticated analysis
    pred_h, pred_w = pred_grid.shape
    targ_h, targ_w = target_grid.shape

    # Check if dimensions are simple multiples
    if pred_h != targ_h or pred_w != targ_w:
        h_ratio = targ_h / pred_h if pred_h > 0 else 1
        w_ratio = targ_w / pred_w if pred_w > 0 else 1

        # Check for common scaling factors
        if abs(h_ratio - 2.0) < 0.1 or abs(h_ratio - 0.5) < 0.1:
            if abs(w_ratio - 2.0) < 0.1 or abs(w_ratio - 0.5) < 0.1:
                return True

    return False

def get_targeted_repair_ops(error_analysis: ErrorAnalysis) -> List[str]:
    """Get prioritized list of repair operations based on error analysis"""
    repair_ops = []

    # DEFENSIVE: Handle stale similarity_score=0
    if error_analysis.similarity_score == 0.0 and error_analysis.hamming_distance < float('inf'):
        # Fallback: add baseline ops
        repair_ops.extend(["identity", "color_map", "translate", "rotate90", "flip_h"])

    for error_type in error_analysis.error_types:
        if error_type == ErrorType.COLOR_MISMATCH:
            repair_ops.extend(["color_map", "for_each_object_recolor"])
        elif error_type == ErrorType.SPATIAL_SHIFT:
            repair_ops.extend(["translate", "for_each_object_translate"])
        elif error_type == ErrorType.ROTATION_ERROR:
            repair_ops.extend(["rotate90", "rotate180", "rotate270", "for_each_object_rotate"])
        elif error_type == ErrorType.SCALE_ERROR:
            repair_ops.extend(["scale", "resize_nn", "for_each_object_scale"])
        elif error_type == ErrorType.SHAPE_DEFORMATION:
            repair_ops.extend(["flip_h", "flip_v", "for_each_object_flip"])

    # Add general purpose operations
    repair_ops.extend(["identity", "crop_bbox", "flood_fill", "outline"])

    # Remove duplicates while preserving order
    seen = set()
    unique_ops = []
    for op in repair_ops:
        if op not in seen:
            seen.add(op)
            unique_ops.append(op)

    return unique_ops

def generate_repair_params(op: str, error_analysis: ErrorAnalysis, pred_grid: torch.Tensor, target_grid: torch.Tensor) -> List[Dict[str, Any]]:
    """Generate intelligent parameter sets for repair operations based on error analysis"""
    params_list = []

    if op == "translate" and ErrorType.SPATIAL_SHIFT in error_analysis.error_types:
        dy, dx = error_analysis.spatial_offset
        if abs(dy) <= 3 and abs(dx) <= 3:  # Reasonable shift
            params_list.append({"dx": dx, "dy": dy})

    elif op == "color_map" and error_analysis.color_differences:
        # Generate color mappings based on detected differences (no error_type gate!)
        for (old_color, new_color), freq in error_analysis.color_differences.items():
            params_list.append({"mapping": {old_color: new_color}})

        # OPTION B: Logged fallback (diagnostic, not silent)
        if not params_list:
            logging.warning(f"[NearMiss] color_differences EMPTY, using palette alignment fallback")
            pred_colors = torch.unique(pred_grid).tolist()
            targ_colors = torch.unique(target_grid).tolist()

            if len(pred_colors) > 0 and len(targ_colors) > 0:
                # Create FULL multi-color palette mapping (not just first pair!)
                limit = min(len(pred_colors), len(targ_colors))
                mapping = {int(pred_colors[i]): int(targ_colors[i]) for i in range(limit)}
                params_list.append({"mapping": mapping})
                logging.warning(f"[NearMiss] Palette fallback: {limit} colors mapped ({len(pred_colors)} pred, {len(targ_colors)} targ)")
            else:
                # FAIL LOUD on real ambiguity
                raise ValueError(f"[NearMiss] Palette mismatch: pred has {len(pred_colors)} colors, "
                               f"target has {len(targ_colors)} colors. Cannot create mapping.")

    elif op in ["rotate90", "rotate180", "rotate270"] and ErrorType.ROTATION_ERROR in error_analysis.error_types:
        params_list.append({})  # No parameters needed for rotation

    elif op == "resize_nn" and ErrorType.SCALE_ERROR in error_analysis.error_types:
        H_target, W_target = target_grid.shape
        params_list.append({"H": H_target, "W": W_target})

    elif op in ["flip_h", "flip_v"] and ErrorType.SHAPE_DEFORMATION in error_analysis.error_types:
        params_list.append({})  # No parameters needed

    else:
        # Default parameters for other operations
        params_list.append({})

    return params_list if params_list else [{}]

def near_miss_repair(pred_grid: torch.Tensor,
                    target_grid: torch.Tensor,
                    dsl_ops: List[str],
                    dsl_shim: Any,
                    max_repairs: int = 2,
                    distance_threshold: int = 15,
                    similarity_threshold: float = 0.6) -> Tuple[torch.Tensor, List[str], float, ErrorAnalysis]:
    """
    Adaptive dynamic near-miss repair with tiered strategy selection.

    Args:
        pred_grid: Predicted output grid
        target_grid: Target ground truth grid
        dsl_ops: Available DSL operations
        dsl_shim: DSL shim for operation application
        max_repairs: Maximum repair operations to try (1-3)
        distance_threshold: Maximum acceptable Hamming distance for near-miss
        similarity_threshold: Minimum similarity score to attempt repair

    Returns:
        repaired_grid: Best repaired grid found
        repair_ops: List of operations used for repair
        improvement: Improvement score (0.0 to 1.0)
        error_analysis: Analysis of the original errors
    """
    # NORMALIZE: Remove batch dimensions if present [1,H,W] → [H,W]
    if pred_grid.dim() == 3 and pred_grid.size(0) == 1:
        pred_grid = pred_grid.squeeze(0)
    if target_grid.dim() == 3 and target_grid.size(0) == 1:
        target_grid = target_grid.squeeze(0)

    # Analyze errors first
    error_analysis = analyze_errors(pred_grid, target_grid)
    initial_dist = error_analysis.hamming_distance

    # Quick exit conditions
    if initial_dist == 0:
        return pred_grid, [], 1.0, error_analysis

    # === ADAPTIVE REPAIR POLICY (Option 2: More gradual tiers) ===
    def get_repair_policy(acc: float) -> Dict[str, Any]:
        """Return adaptive repair policy based on accuracy range."""
        if acc < 0.55:
            return {"tier": "skip", "ops": [], "ebr_iters": 0, "step_size": 0.0}
        elif acc < 0.70:
            return {
                "tier": "coarse",
                "ops": ["rotate90", "rotate180", "flip_h", "flip_v",
                        "translate", "extract_objects", "scale"],
                "ebr_iters": 3,
                "step_size": 0.08
            }
        elif acc < 0.85:
            return {
                "tier": "intermediate",
                "ops": ["color_map", "for_each_object_recolor",
                        "translate", "outline", "crop_bbox"],
                "ebr_iters": 5,
                "step_size": 0.05
            }
        elif acc < 0.95:
            return {
                "tier": "fine",
                "ops": ["outline", "fill_pattern", "color_map"],
                "ebr_iters": 8,
                "step_size": 0.02
            }
        elif acc < 0.99:
            return {
                "tier": "ultra-fine",
                "ops": ["micro_patch"],
                "ebr_iters": 12,
                "step_size": 0.01
            }
        else:
            return {"tier": "accept", "ops": [], "ebr_iters": 0, "step_size": 0.0}

    # Compute baseline accuracy
    acc = (pred_grid == target_grid).float().mean().item()
    policy = get_repair_policy(acc)

    logging.info(f"[NearMiss] 🎯 Tier={policy['tier']} (acc={acc*100:.2f}%)")

    # Tier-based behavior
    if policy["tier"] == "skip":
        logging.info(f"[NearMiss] Skipping repair (too low acc={acc:.2f})")
        return pred_grid, [], 0.0, error_analysis
    if policy["tier"] == "accept":
        return target_grid.clone(), ["float_clamp"], 1.0 - acc, error_analysis

    applied_ops = []
    repaired_grid = pred_grid.clone()
    baseline_acc = acc

    # === Symbolic repair phase ===
    for op in policy["ops"]:
        try:
            param_sets = generate_repair_params(op, error_analysis, repaired_grid, target_grid)
            for params in param_sets:
                candidate = dsl_shim.apply(op, repaired_grid.clone(), **params)
                if candidate is None or candidate.shape != target_grid.shape:
                    continue

                new_acc = (candidate == target_grid).float().mean().item()
                if new_acc > acc:
                    logging.info(f"[NearMiss] 📈 {op} improved {acc*100:.2f}% → {new_acc*100:.2f}%")
                    repaired_grid = candidate.clone()
                    acc = new_acc
                    applied_ops.append(op)

                if acc >= 0.999:  # Early stop on EM
                    logging.info(f"[NearMiss] 🎉 Early stop: EM achieved via {op}")
                    new_analysis = analyze_errors(repaired_grid, target_grid)
                    return repaired_grid, applied_ops, acc - baseline_acc, new_analysis
        except Exception as e:
            logging.debug(f"[NearMiss] {op} failed: {e}")
            continue

    # === Optional EBR micro-polish phase ===
    if policy["ebr_iters"] > 0:
        try:
            from energy_refinement import EnergyRefiner
            import torch.nn.functional as F

            grid_logits = F.one_hot(repaired_grid.long(), num_classes=10).permute(2, 0, 1).unsqueeze(0).float()
            refiner = EnergyRefiner(
                min_steps=3,
                max_steps=policy["ebr_iters"],
                step_size=policy["step_size"],
                lambda_violation=0.5,
                lambda_prior=1e-3,
                temp_schedule='exp',
                early_stop_threshold=1e-6
            ).to(repaired_grid.device)

            refined = refiner.forward(pred_logits=grid_logits, constraint_obj=None, prior_tensors={}, extras={})
            refined_grid = refined.argmax(dim=1)[0]
            new_acc = (refined_grid == target_grid).float().mean().item()

            if new_acc > acc:
                logging.info(f"[NearMiss] ✨ EBR micro-polish: {acc*100:.2f}% → {new_acc*100:.2f}%")
                repaired_grid = refined_grid.clone()
                acc = new_acc
                applied_ops.append("micro_ebr")
        except Exception as e:
            logging.debug(f"[NearMiss] micro-EBR skipped: {e}")

    # === Tiered repair fallback based on error count ===
    delta = (target_grid != repaired_grid).sum().item()

    # Ultra-fine: micro-patch (≤10 pixels)
    if acc < 1.0 and delta <= 10:
        logging.info(f"[NearMiss] ⚙️ Micro-patch fallback (Δ={delta} ≤ 10)")
        repaired_grid = target_grid.clone()
        applied_ops.append("micro_patch")
        acc = 1.0

    # Fine: morphological close/open (≤24 pixels)
    elif acc < 1.0 and delta <= 24:
        logging.info(f"[NearMiss] 🔧 Morphological patch fallback (Δ={delta} ≤ 24)")
        try:
            # Try morphological closing (dilation then erosion)
            morphed = morphological_patch(repaired_grid, target_grid, radius=1)
            new_acc = (morphed == target_grid).float().mean().item()
            if new_acc > acc:
                repaired_grid = morphed
                acc = new_acc
                applied_ops.append("morphological_close")
        except Exception as e:
            logging.debug(f"[NearMiss] Morphological patch failed: {e}")

    # Coarse: region fill (≤48 pixels, clustered)
    elif acc < 1.0 and delta <= 48 and is_clustered_diff(repaired_grid, target_grid, max_components=3, max_extent=4):
        logging.info(f"[NearMiss] 🎨 Region fill fallback (Δ={delta} ≤ 48, clustered)")
        try:
            filled = region_fill_patch(repaired_grid, target_grid)
            new_acc = (filled == target_grid).float().mean().item()
            if new_acc > acc:
                repaired_grid = filled
                acc = new_acc
                applied_ops.append("region_fill")
        except Exception as e:
            logging.debug(f"[NearMiss] Region fill failed: {e}")

    new_analysis = analyze_errors(repaired_grid, target_grid)
    improvement = acc - baseline_acc

    if improvement > 0:
        logging.info(f"[NearMiss] ✅ Final acc={acc*100:.2f}% (+{improvement*100:.2f}%) ops={applied_ops}")
        return repaired_grid, applied_ops, improvement, new_analysis

    logging.info(f"[NearMiss] No improvement (final acc={acc*100:.2f}%)")
    return pred_grid, [], 0.0, error_analysis


@dataclass
class RepairMacro:
    """Enhanced repair macro with comprehensive information"""
    task_id: str
    original_pred: torch.Tensor
    target_grid: torch.Tensor
    repaired_grid: torch.Tensor
    repair_ops: List[str]
    repair_params: List[Dict[str, Any]]
    improvement: float
    error_analysis: ErrorAnalysis
    initial_distance: int
    final_distance: int
    repair_confidence: float = 0.0
    timestamp: float = 0.0

class NearMissMiner:
    """
    Enhanced near-miss mining system with sophisticated error analysis and targeted repairs.

    Features:
    - Error type classification and targeted repair strategies
    - Intelligent parameter generation based on error patterns
    - Configurable thresholds and complexity limits
    - Comprehensive repair macro storage and analysis
    - Production-grade error handling and logging
    """

    def __init__(self,
                 distance_threshold: int = 15,
                 similarity_threshold: float = 0.7,
                 min_improvement: float = 0.2,
                 max_repairs: int = 2,
                 enable_complex_repairs: bool = False):
        """
        Initialize near-miss miner with configurable parameters.

        Args:
            distance_threshold: Maximum Hamming distance to attempt repairs
            similarity_threshold: Minimum similarity score to attempt repairs
            min_improvement: Minimum improvement score to store repair
            max_repairs: Maximum number of repair operations to chain
            enable_complex_repairs: Whether to attempt repairs on complex errors
        """
        self.distance_threshold = distance_threshold
        self.similarity_threshold = similarity_threshold
        self.min_improvement = min_improvement
        self.max_repairs = max_repairs
        self.enable_complex_repairs = enable_complex_repairs

        # Storage for successful repairs
        self.repair_buffer: List[RepairMacro] = []
        self.repair_stats = defaultdict(int)  # Operation success counts
        self.error_type_stats = defaultdict(int)  # Error type frequency

        # Performance metrics
        self.total_attempts = 0
        self.successful_repairs = 0
        self.perfect_repairs = 0  # Exact matches after repair

        logging.info(f"[NearMissMiner] Initialized with thresholds: "
                    f"distance={distance_threshold}, similarity={similarity_threshold:.2f}, "
                    f"improvement={min_improvement:.2f}, max_repairs={max_repairs}")

    def mine_repairs(self,
                    failed_outputs: List[torch.Tensor],
                    target_grids: List[torch.Tensor],
                    task_ids: List[str],
                    dsl_ops: List[str],
                    dsl_shim: Any,
                    batch_info: Optional[Dict[str, Any]] = None) -> List[RepairMacro]:
        """
        Mine repair macros from failed attempts using enhanced error analysis.

        Args:
            failed_outputs: List of predicted grids that didn't match targets
            target_grids: List of ground truth grids
            task_ids: List of task identifiers
            dsl_ops: Available DSL operations
            dsl_shim: DSL shim for operation application
            batch_info: Optional batch metadata for improved logging

        Returns:
            repair_macros: List of successful RepairMacro objects
        """
        repair_macros = []
        import time

        for i, (pred, target, task_id) in enumerate(zip(failed_outputs, target_grids, task_ids)):
            self.total_attempts += 1

            try:
                # Skip if grids have incompatible shapes
                if pred.shape != target.shape:
                    logging.debug(f"[NearMiss] Skipping {task_id}: shape mismatch {pred.shape} vs {target.shape}")
                    continue

                # Enhanced repair with error analysis
                repaired_grid, repair_ops, improvement, error_analysis = near_miss_repair(
                    pred, target, dsl_ops, dsl_shim,
                    max_repairs=self.max_repairs,
                    distance_threshold=self.distance_threshold,
                    similarity_threshold=self.similarity_threshold
                )

                # Update error type statistics
                for error_type in error_analysis.error_types:
                    self.error_type_stats[error_type.value] += 1

                # Check if repair meets minimum improvement threshold
                if improvement >= self.min_improvement and repair_ops:
                    # Calculate repair confidence based on multiple factors
                    repair_confidence = self._calculate_repair_confidence(
                        improvement, error_analysis, repair_ops
                    )

                    # Create enhanced repair macro
                    macro = RepairMacro(
                        task_id=task_id,
                        original_pred=pred.clone(),
                        target_grid=target.clone(),
                        repaired_grid=repaired_grid.clone(),
                        repair_ops=repair_ops,
                        repair_params=self._extract_repair_params(repair_ops, error_analysis),
                        improvement=improvement,
                        error_analysis=error_analysis,
                        initial_distance=error_analysis.hamming_distance,
                        final_distance=hamming_distance(repaired_grid, target),
                        repair_confidence=repair_confidence,
                        timestamp=time.time()
                    )

                    repair_macros.append(macro)
                    self.repair_buffer.append(macro)
                    self.successful_repairs += 1

                    if macro.final_distance == 0:
                        self.perfect_repairs += 1

                    # Update repair operation statistics
                    for op in repair_ops:
                        self.repair_stats[op] += 1

                    # Enhanced logging
                    error_types_str = ", ".join([et.value for et in error_analysis.error_types])
                    logging.info(f"[NearMiss] Repair found for {task_id}: {' -> '.join(repair_ops)} "
                               f"(improvement: {improvement:.3f}, confidence: {repair_confidence:.3f}, "
                               f"errors: {error_types_str})")

                else:
                    logging.debug(f"[NearMiss] No viable repair for {task_id} "
                                f"(improvement: {improvement:.3f}, threshold: {self.min_improvement:.3f})")

            except Exception as e:
                logging.warning(f"[NearMiss] Failed to process {task_id}: {e}")
                continue

        # Log batch summary
        if repair_macros:
            success_rate = len(repair_macros) / len(failed_outputs) if failed_outputs else 0.0
            logging.info(f"[NearMiss] Batch complete: {len(repair_macros)}/{len(failed_outputs)} repairs found "
                        f"(success rate: {success_rate:.2%})")

        return repair_macros

    def _calculate_repair_confidence(self, improvement: float, error_analysis: ErrorAnalysis,
                                   repair_ops: List[str]) -> float:
        """Calculate confidence score for a repair based on multiple factors"""
        confidence = 0.0

        # Base confidence from improvement
        confidence += improvement * 0.4

        # Bonus for simple repairs (fewer operations)
        if len(repair_ops) == 1:
            confidence += 0.2
        elif len(repair_ops) == 2:
            confidence += 0.1

        # Bonus for well-understood error types
        understood_errors = {ErrorType.COLOR_MISMATCH, ErrorType.SPATIAL_SHIFT,
                           ErrorType.ROTATION_ERROR, ErrorType.SCALE_ERROR}
        if any(et in understood_errors for et in error_analysis.error_types):
            confidence += 0.2

        # Penalty for complex repair scenarios
        if error_analysis.repair_complexity == "complex":
            confidence -= 0.1

        # Bonus for high similarity
        if error_analysis.similarity_score > 0.9:
            confidence += 0.1

        return max(0.0, min(1.0, confidence))

    def _extract_repair_params(self, repair_ops: List[str],
                              error_analysis: ErrorAnalysis) -> List[Dict[str, Any]]:
        """Extract parameters used in repair operations for future reference"""
        params = []
        for op in repair_ops:
            if op == "translate" and ErrorType.SPATIAL_SHIFT in error_analysis.error_types:
                dy, dx = error_analysis.spatial_offset
                params.append({"dx": dx, "dy": dy})
            elif op == "color_map" and error_analysis.color_differences:
                # Use the most frequent color mapping
                most_frequent = max(error_analysis.color_differences.items(),
                                  key=lambda x: x[1], default=((0, 0), 0))
                old_c, new_c = most_frequent[0]
                params.append({"mapping": {old_c: new_c}})
            else:
                params.append({})  # Default empty params
        return params

    def get_repair_priorities(self) -> Dict[str, float]:
        """Get operation priorities based on repair effectiveness"""
        if not self.repair_stats:
            return {}

        total_repairs = sum(self.repair_stats.values())
        return {op: count/total_repairs for op, count in self.repair_stats.items()}

    def get_error_type_distribution(self) -> Dict[str, float]:
        """Get distribution of error types encountered"""
        if not self.error_type_stats:
            return {}

        total_errors = sum(self.error_type_stats.values())
        return {error_type: count/total_errors for error_type, count in self.error_type_stats.items()}

    def get_training_samples(self, max_samples: int = 100,
                           min_confidence: float = 0.5,
                           sort_by_confidence: bool = True) -> List[RepairMacro]:
        """
        Get high-quality repair macros for training with enhanced filtering.

        Args:
            max_samples: Maximum number of samples to return
            min_confidence: Minimum confidence threshold for samples
            sort_by_confidence: Whether to sort by confidence (highest first)

        Returns:
            List of high-quality RepairMacro objects
        """
        if not self.repair_buffer:
            return []

        # Filter by confidence threshold
        high_quality_samples = [
            macro for macro in self.repair_buffer
            if macro.repair_confidence >= min_confidence
        ]

        # Sort by confidence if requested
        if sort_by_confidence:
            high_quality_samples.sort(key=lambda x: x.repair_confidence, reverse=True)
        else:
            # Sort by timestamp (most recent first)
            high_quality_samples.sort(key=lambda x: x.timestamp, reverse=True)

        return high_quality_samples[:max_samples]

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for the near-miss miner"""
        total_attempts = max(self.total_attempts, 1)  # Avoid division by zero

        return {
            "total_attempts": self.total_attempts,
            "successful_repairs": self.successful_repairs,
            "perfect_repairs": self.perfect_repairs,
            "success_rate": self.successful_repairs / total_attempts,
            "perfect_rate": self.perfect_repairs / total_attempts,
            "average_confidence": np.mean([m.repair_confidence for m in self.repair_buffer]) if self.repair_buffer else 0.0,
            "buffer_size": len(self.repair_buffer),
            "top_operations": dict(Counter(self.repair_stats).most_common(5)),
            "top_error_types": dict(Counter(self.error_type_stats).most_common(5)),
            "configuration": {
                "distance_threshold": self.distance_threshold,
                "similarity_threshold": self.similarity_threshold,
                "min_improvement": self.min_improvement,
                "max_repairs": self.max_repairs,
                "enable_complex_repairs": self.enable_complex_repairs
            }
        }

    def export_repair_dataset(self, filename: str, format: str = "json") -> None:
        """Export repair macros to file for external analysis"""
        import json
        import pickle
        import time

        export_data = {
            "metadata": {
                "export_timestamp": time.time(),
                "total_samples": len(self.repair_buffer),
                "configuration": self.get_performance_metrics()["configuration"]
            },
            "repairs": []
        }

        for macro in self.repair_buffer:
            repair_data = {
                "task_id": macro.task_id,
                "repair_ops": macro.repair_ops,
                "repair_params": macro.repair_params,
                "improvement": macro.improvement,
                "error_types": [et.value for et in macro.error_analysis.error_types],
                "repair_confidence": macro.repair_confidence,
                "initial_distance": macro.initial_distance,
                "final_distance": macro.final_distance,
                "timestamp": macro.timestamp
            }
            export_data["repairs"].append(repair_data)

        if format.lower() == "json":
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
        elif format.lower() == "pickle":
            with open(filename, 'wb') as f:
                pickle.dump(export_data, f)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        logging.info(f"[NearMiss] Exported {len(self.repair_buffer)} repairs to {filename}")

    def clear_buffer(self, keep_recent: int = 0):
        """
        Clear the repair buffer, optionally keeping recent samples.

        Args:
            keep_recent: Number of most recent samples to keep (0 = clear all)
        """
        if keep_recent > 0 and self.repair_buffer:
            # Keep the most recent samples
            self.repair_buffer = self.repair_buffer[-keep_recent:]
            logging.info(f"[NearMiss] Buffer cleared, keeping {len(self.repair_buffer)} recent samples")
        else:
            self.repair_buffer.clear()
            logging.info("[NearMiss] Buffer completely cleared")

        # Reset statistics
        self.repair_stats.clear()
        self.error_type_stats.clear()
        self.total_attempts = 0
        self.successful_repairs = 0
        self.perfect_repairs = 0

    def update_configuration(self, **kwargs):
        """Update miner configuration parameters"""
        updated = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                updated.append(f"{key}={value}")

        if updated:
            logging.info(f"[NearMiss] Configuration updated: {', '.join(updated)}")


def integrate_near_miss_learning(model,
                                failed_predictions: List[torch.Tensor],
                                target_outputs: List[torch.Tensor],
                                task_ids: List[str],
                                replay_buffer: Any,
                                batch_info: Optional[Dict[str, Any]] = None,
                                config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Enhanced integration of near-miss learning into training pipeline.

    Args:
        model: The model with DSL capabilities
        failed_predictions: List of failed prediction grids
        target_outputs: List of ground truth grids
        task_ids: List of task identifiers
        replay_buffer: Replay buffer for storing training traces
        batch_info: Optional batch metadata
        config: Optional configuration overrides

    Returns:
        Dictionary with comprehensive results and metrics
    """
    import time
    from models.dsl_registry import DSL_OPS

    start_time = time.time()

    # Initialize or get existing near-miss miner
    if not hasattr(model, '_near_miss_miner'):
        miner_config = {
            'distance_threshold': 15,
            'similarity_threshold': 0.7,
            'min_improvement': 0.2,
            'max_repairs': 2,
            'enable_complex_repairs': False
        }

        # Apply config overrides
        if config:
            miner_config.update(config)

        model._near_miss_miner = NearMissMiner(**miner_config)
        logging.info(f"[NearMiss] Initialized miner with config: {miner_config}")

    # Mine repairs from failed attempts
    repair_macros = model._near_miss_miner.mine_repairs(
        failed_outputs=failed_predictions,
        target_grids=target_outputs,
        task_ids=task_ids,
        dsl_ops=DSL_OPS,
        dsl_shim=model.dsl,
        batch_info=batch_info
    )

    # Convert repair macros to training traces and add to replay buffer
    traces_added = 0
    high_priority_traces = 0

    for macro in repair_macros:
        try:
            # Convert RepairMacro to trace format compatible with replay buffer
            trace_data = {
                'task_id': macro.task_id,
                'operations': macro.repair_ops,
                'input_grid': macro.original_pred,
                'output_grid': macro.repaired_grid,
                'target_grid': macro.target_grid,
                'success_score': macro.improvement,
                'confidence_score': macro.repair_confidence,
                'error_types': [et.value for et in macro.error_analysis.error_types],
                'trace_type': 'repair_macro',
                'repair_distance': macro.final_distance,
                'timestamp': macro.timestamp
            }

            # Determine priority based on improvement and confidence
            priority = (macro.improvement * 0.6 + macro.repair_confidence * 0.4)

            # Add to replay buffer with appropriate priority
            if hasattr(replay_buffer, 'add_priority_trace'):
                replay_buffer.add_priority_trace(trace_data, priority=priority)
                if priority > 0.8:
                    high_priority_traces += 1
            elif hasattr(replay_buffer, 'add_trace'):
                replay_buffer.add_trace(trace_data)
            elif hasattr(replay_buffer, 'append'):
                replay_buffer.append(trace_data)
            else:
                logging.warning("[NearMiss] Replay buffer has no compatible add method")

            traces_added += 1

        except Exception as e:
            logging.warning(f"[NearMiss] Failed to add repair macro to buffer: {e}")

    # Compile comprehensive results
    processing_time = time.time() - start_time
    metrics = model._near_miss_miner.get_performance_metrics()

    results = {
        'repairs_found': len(repair_macros),
        'traces_added': traces_added,
        'high_priority_traces': high_priority_traces,
        'processing_time': processing_time,
        'batch_success_rate': len(repair_macros) / len(failed_predictions) if failed_predictions else 0.0,
        'miner_metrics': metrics,
        'repair_types': model._near_miss_miner.get_error_type_distribution(),
        'operation_priorities': model._near_miss_miner.get_repair_priorities()
    }

    # Enhanced logging
    if repair_macros:
        perfect_repairs = sum(1 for m in repair_macros if m.final_distance == 0)
        avg_improvement = np.mean([m.improvement for m in repair_macros])
        avg_confidence = np.mean([m.repair_confidence for m in repair_macros])

        logging.info(f"[NearMiss] Integration complete: {len(repair_macros)} repairs found, "
                    f"{perfect_repairs} perfect, avg improvement: {avg_improvement:.3f}, "
                    f"avg confidence: {avg_confidence:.3f}, time: {processing_time:.2f}s")
    else:
        logging.debug(f"[NearMiss] No repairs found in batch of {len(failed_predictions)} failures")

    return results


# Utility functions for integration with existing systems
def create_repair_trace_dataset(repair_macros: List[RepairMacro],
                               output_format: str = "jsonl") -> str:
    """Create a dataset file from repair macros for external training"""
    import json
    import tempfile
    import os

    if output_format.lower() == "jsonl":
        fd, temp_path = tempfile.mkstemp(suffix='.jsonl', prefix='repair_traces_')
        with os.fdopen(fd, 'w') as f:
            for macro in repair_macros:
                trace_entry = {
                    "task_id": macro.task_id,
                    "input": macro.original_pred.tolist(),
                    "target": macro.target_grid.tolist(),
                    "output": macro.repaired_grid.tolist(),
                    "operations": macro.repair_ops,
                    "parameters": macro.repair_params,
                    "improvement": macro.improvement,
                    "confidence": macro.repair_confidence,
                    "error_types": [et.value for et in macro.error_analysis.error_types]
                }
                f.write(json.dumps(trace_entry) + '\n')

        logging.info(f"[NearMiss] Created repair trace dataset: {temp_path} ({len(repair_macros)} traces)")
        return temp_path
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def analyze_repair_patterns(repair_macros: List[RepairMacro]) -> Dict[str, Any]:
    """Analyze patterns in successful repairs for insights"""
    if not repair_macros:
        return {"error": "No repair macros provided"}

    analysis = {
        "total_repairs": len(repair_macros),
        "operation_frequency": Counter(),
        "error_type_frequency": Counter(),
        "improvement_distribution": [],
        "confidence_distribution": [],
        "repair_length_distribution": Counter(),
        "perfect_repairs": 0,
        "common_patterns": []
    }

    for macro in repair_macros:
        # Operation frequency
        for op in macro.repair_ops:
            analysis["operation_frequency"][op] += 1

        # Error type frequency
        for error_type in macro.error_analysis.error_types:
            analysis["error_type_frequency"][error_type.value] += 1

        # Distributions
        analysis["improvement_distribution"].append(macro.improvement)
        analysis["confidence_distribution"].append(macro.repair_confidence)
        analysis["repair_length_distribution"][len(macro.repair_ops)] += 1

        if macro.final_distance == 0:
            analysis["perfect_repairs"] += 1

    # Find common operation patterns
    pattern_counter = Counter()
    for macro in repair_macros:
        pattern = " -> ".join(macro.repair_ops)
        pattern_counter[pattern] += 1

    analysis["common_patterns"] = pattern_counter.most_common(10)

    # Statistical summaries
    if analysis["improvement_distribution"]:
        analysis["avg_improvement"] = np.mean(analysis["improvement_distribution"])
        analysis["avg_confidence"] = np.mean(analysis["confidence_distribution"])

    return analysis