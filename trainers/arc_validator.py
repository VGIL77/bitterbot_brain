#!/usr/bin/env python3
"""
ARC-II Data Validator
Ensures all grids conform to ARC-AGI-2 specification
"""
import torch
import logging

logger = logging.getLogger(__name__)

# ARC-II Specification Constants
ARC_MIN_DIM = 1
ARC_MAX_DIM = 30
ARC_MIN_COLOR = 0
ARC_MAX_COLOR = 9
ARC_NUM_COLORS = 10


def validate_arc_grid(grid: torch.Tensor, name: str = "grid", clamp: bool = True) -> torch.Tensor:
    """
    Validate and optionally clamp grid to ARC-II specification.

    Args:
        grid: Input grid tensor [H, W] or [B, H, W]
        name: Name for error messages
        clamp: If True, clamp values to valid range; if False, raise on violation

    Returns:
        Validated (and possibly clamped) grid on same device

    Raises:
        ValueError: If grid violates ARC-II spec and clamp=False
    """
    if grid is None:
        raise ValueError(f"{name} is None")

    # Check for NaN/Inf
    if not torch.isfinite(grid.float()).all():
        raise ValueError(f"{name} contains NaN or Inf values")

    # Get shape (handle batched and unbatched)
    if grid.dim() == 2:
        h, w = grid.shape
    elif grid.dim() == 3:
        h, w = grid.shape[-2:]
    else:
        raise ValueError(f"{name} has invalid dimensions: {grid.shape} (expected 2D or 3D)")

    # Check dimensions
    if not (ARC_MIN_DIM <= h <= ARC_MAX_DIM):
        raise ValueError(f"{name} height {h} out of range [{ARC_MIN_DIM}, {ARC_MAX_DIM}]")
    if not (ARC_MIN_DIM <= w <= ARC_MAX_DIM):
        raise ValueError(f"{name} width {w} out of range [{ARC_MIN_DIM}, {ARC_MAX_DIM}]")

    # Check/clamp color values
    if grid.dtype in [torch.float32, torch.float64]:
        # Floating point grid - validate range
        if grid.min() < ARC_MIN_COLOR or grid.max() > ARC_MAX_COLOR:
            if clamp:
                grid = torch.clamp(grid, ARC_MIN_COLOR, ARC_MAX_COLOR)
                logger.debug(f"{name} clamped to [{ARC_MIN_COLOR}, {ARC_MAX_COLOR}]")
            else:
                raise ValueError(f"{name} values out of range: min={grid.min()}, max={grid.max()}")
    else:
        # Integer grid
        if grid.min() < ARC_MIN_COLOR or grid.max() > ARC_MAX_COLOR:
            if clamp:
                grid = torch.clamp(grid, ARC_MIN_COLOR, ARC_MAX_COLOR)
                logger.debug(f"{name} clamped to [{ARC_MIN_COLOR}, {ARC_MAX_COLOR}]")
            else:
                raise ValueError(f"{name} values out of range: min={grid.min()}, max={grid.max()}")

    # Ensure integer type for ARC grids
    if grid.dtype not in [torch.long, torch.int32, torch.int64]:
        grid = grid.long()

    return grid


def validate_arc_demo(demo: tuple, demo_idx: int = 0) -> tuple:
    """
    Validate a single ARC demo pair.

    Args:
        demo: (input_grid, output_grid) tuple
        demo_idx: Demo index for error messages

    Returns:
        Validated (input_grid, output_grid) tuple
    """
    if not isinstance(demo, (tuple, list)) or len(demo) < 2:
        raise ValueError(f"Demo {demo_idx} invalid format: expected (input, output) tuple")

    input_grid, output_grid = demo[0], demo[1]

    # Validate both grids
    input_grid = validate_arc_grid(input_grid, f"Demo {demo_idx} input", clamp=True)
    output_grid = validate_arc_grid(output_grid, f"Demo {demo_idx} output", clamp=True)

    return (input_grid, output_grid)


def validate_arc_task(demos: list, test_input: torch.Tensor, test_output: torch.Tensor = None) -> dict:
    """
    Validate complete ARC task.

    Args:
        demos: List of (input, output) tuples
        test_input: Test input grid
        test_output: Optional test output grid

    Returns:
        Dict with validated components: {'demos': [...], 'test_input': tensor, 'test_output': tensor or None}
    """
    # Validate demos
    validated_demos = []
    for i, demo in enumerate(demos):
        try:
            validated_demo = validate_arc_demo(demo, i)
            validated_demos.append(validated_demo)
        except Exception as e:
            logger.warning(f"Demo {i} validation failed: {e}, skipping")
            continue

    # Validate test input
    test_input = validate_arc_grid(test_input, "Test input", clamp=True)

    # Validate test output if provided
    if test_output is not None:
        test_output = validate_arc_grid(test_output, "Test output", clamp=True)

    return {
        'demos': validated_demos,
        'test_input': test_input,
        'test_output': test_output
    }


def arc_safe_output(grid: torch.Tensor) -> torch.Tensor:
    """
    Ensure model output conforms to ARC-II spec before evaluation.
    Used as final safety check before computing metrics.

    Args:
        grid: Model output grid

    Returns:
        Clamped and validated grid
    """
    return validate_arc_grid(grid, "Model output", clamp=True)
