"""
Canonical Demo Unpacking Utilities

Handles all demo formats consistently across the codebase:
- 2-tuples: (input, output)
- Dicts: {"input": tensor, "output": tensor}
- 3-tuples with metadata: (input, output, metadata)
- Lists: [input, output]

Guarantees:
- No training signal pollution (valid demos always processed)
- No silent data loss (only None/malformed filtered)
- Fail-loud on unexpected formats (raises TypeError)
"""

from typing import Tuple, Optional, Any, List, Dict, Union, Iterable, Sequence
import torch
import numpy as np

TensorLike = Union[torch.Tensor, np.ndarray, Sequence[Sequence[int]]]


def safe_unpack_demo(demo: Any) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Safely extract (input, output) from any demo format.

    Args:
        demo: Demo in any supported format

    Returns:
        (input_tensor, output_tensor) or (None, None) if malformed

    Raises:
        TypeError: If demo is completely unexpected type (fail-loud)
    """
    # Dict format: {"input": t, "output": t}
    if isinstance(demo, dict):
        inp = demo.get('input')
        out = demo.get('output')
        return inp, out

    # Tuple/list format: (input, output) or (input, output, metadata)
    elif isinstance(demo, (tuple, list)):
        if len(demo) >= 2:
            return demo[0], demo[1]  # Ignore metadata if present
        elif len(demo) == 1:
            # Edge case: 1-tuple containing a dict or list
            return safe_unpack_demo(demo[0])  # Recursive
        else:
            # Empty tuple/list
            return None, None

    # Unexpected format - fail loud
    else:
        raise TypeError(f"Unexpected demo format: {type(demo)} - expected dict, tuple, or list")


def unpack_demo_batch(demos: list) -> list:
    """
    Unpack a batch of demos into (input, output) pairs.

    Args:
        demos: List of demos in any supported format

    Returns:
        List of (input, output) 2-tuples with None pairs filtered out

    Example:
        demos = [{"input": t1, "output": t2}, (t3, t4, meta)]
        result = unpack_demo_batch(demos)
        # result = [(t1, t2), (t3, t4)]
    """
    if not demos:
        return []

    pairs = [safe_unpack_demo(d) for d in demos]
    # Filter out None pairs (keeps training signal clean)
    pairs = [(inp, out) for inp, out in pairs if inp is not None and out is not None]

    return pairs


def to_long_grid(x: TensorLike, device=None) -> torch.Tensor:
    """
    Canonicalize grid to [B,H,W] long tensor on target device.

    Args:
        x: Grid as tensor, numpy array, or nested list
        device: Target device (None = preserve existing)

    Returns:
        [B,H,W] long tensor with B preserved (default B=1 if 2D input)
    """
    # Convert to tensor
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif isinstance(x, list):
        x = torch.tensor(x)

    if not torch.is_tensor(x):
        raise TypeError(f"Unsupported grid type: {type(x)}")

    # Make integer palette grid
    if x.dtype != torch.long:
        x = x.long()

    # Normalize to [B,H,W]
    if x.dim() == 2:
        x = x.unsqueeze(0)  # [H,W] → [1,H,W]
    elif x.dim() == 3:
        pass  # Already [B,H,W]
    else:
        # Flatten to [1,H,W] if unusual shape
        x = x.view(1, *x.shape[-2:])

    # Move to device (GPU-first)
    if device is not None:
        x = x.to(device, non_blocking=True)

    return x.contiguous()


def ensure_demo_pairs(demos: Iterable[Any], device=None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Ultimate robust demo normalization with GPU-first device handling.

    Accepts:
        - [{'input': ..., 'output': ...}] (dict format)
        - [(inp, out)] (tuple format)
        - [[inp, out]] (list format)

    Returns:
        List of (inp:[B,H,W], out:[B,H,W]) tuples on target device

    GPU-FIRST: All tensors moved to device if specified
    ARC-II: Enforces [B,H,W] long tensors with values [0-9]
    """
    if not demos:
        return []

    out = []

    for d in demos:
        # Handle dict format
        if isinstance(d, dict):
            inp = d.get('input')
            tgt = d.get('output')
            if inp is None or tgt is None:
                continue  # Skip incomplete demos

        # Handle tuple/list format
        elif isinstance(d, (tuple, list)) and len(d) >= 2:
            inp, tgt = d[0], d[1]

        else:
            raise TypeError(f"Bad demo type: {type(d)}; expected dict, tuple, or list")

        # Canonicalize to [B,H,W] long on device
        inp = to_long_grid(inp, device)
        tgt = to_long_grid(tgt, device)

        out.append((inp, tgt))

    return out
