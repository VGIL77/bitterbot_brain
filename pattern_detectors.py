
# pattern_detectors.py
# Lightweight, dependency-free pattern detectors for ARC grids (H x W integer tensors)
from typing import Dict, Any, Optional, Tuple
import torch

@torch.no_grad()
def _equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.all(a == b).item()

@torch.no_grad()
def detect_symmetry(grid: torch.Tensor) -> Optional[str]:
    """Return 'flip_h', 'flip_v', 'rot90', 'rot180', 'rot270', or None if no strong symmetry."""
    assert grid.dim() == 2, "grid must be HxW"
    # Horizontal / vertical mirror
    if _equal(grid, torch.flip(grid, dims=[1])):
        return "flip_v"  # vertical axis mirror (flip left-right)
    if _equal(grid, torch.flip(grid, dims=[0])):
        return "flip_h"  # horizontal axis mirror (flip up-down)
    # Rotations
    if _equal(grid, torch.rot90(grid, 1, [0,1])):
        return "rotate90"
    if _equal(grid, torch.rot90(grid, 2, [0,1])):
        return "rotate180"
    if _equal(grid, torch.rot90(grid, 3, [0,1])):
        return "rotate270"
    return None

@torch.no_grad()
def analyze_colors(grid: torch.Tensor) -> Dict[str, Any]:
    """Return color histogram and simple stats."""
    vals, counts = torch.unique(grid, return_counts=True)
    hist = {int(v.item()): int(c.item()) for v, c in zip(vals, counts)}
    if 0 in hist:  # treat 0 as background
        bg = hist.pop(0)
    total = max(1, sum(hist.values()))
    dominant = max(hist.items(), key=lambda kv: kv[1])[0] if hist else None
    return {
        "hist": hist,
        "dominant": dominant,
        "n_colors": len(hist),
    }

@torch.no_grad()
def connected_components(grid: torch.Tensor, color: Optional[int] = None) -> Tuple[int, torch.Tensor]:
    """Return (#components, component_ids tensor)."""
    H, W = grid.shape
    comp = torch.full((H, W), -1, dtype=torch.long, device=grid.device)
    visited = torch.zeros_like(grid, dtype=torch.bool)
    cid = 0
    colors = [color] if color is not None else [int(c) for c in torch.unique(grid).tolist() if int(c) != 0]
    for col in colors:
        mask = (grid == col)
        for i in range(H):
            for j in range(W):
                if mask[i, j] and not visited[i, j]:
                    # BFS flood fill
                    q = [(i, j)]
                    while q:
                        y, x = q.pop()
                        if y < 0 or y >= H or x < 0 or x >= W: 
                            continue
                        if visited[y, x] or not mask[y, x]: 
                            continue
                        visited[y, x] = True
                        comp[y, x] = cid
                        for dy, dx in [(1,0),(-1,0),(0,1),(0,-1)]:
                            q.append((y+dy, x+dx))
                    cid += 1
    return cid, comp

@torch.no_grad()
def detect_translation(a: torch.Tensor, b: torch.Tensor) -> Optional[Tuple[int,int]]:
    """If b is a pure translation of a (ignoring zeros), return (dx, dy); else None."""
    Ha, Wa = a.shape
    Hb, Wb = b.shape
    if (Ha, Wa) != (Hb, Wb):
        return None
    # Try a small window of plausible dx,dy
    max_shift = 6  # ARC grids are small
    for dy in range(-max_shift, max_shift+1):
        for dx in range(-max_shift, max_shift+1):
            shifted = torch.zeros_like(a)
            ys = slice(max(0, dy), min(Ha, Ha+dy))
            xs = slice(max(0, dx), min(Wa, Wa+dx))
            ysb = slice(max(0, -dy), min(Ha, Ha-dy))
            xsb = slice(max(0, -dx), min(Wa, Wa-dx))
            shifted[ys, xs] = a[ysb, xsb]
            if _equal(shifted, b):
                return (dx, dy)
    return None

@torch.no_grad()
def extract_patterns(input_grid: torch.Tensor, output_grid: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    """Compute a compact pattern descriptor to bias DSL search.

    Keys:

    - symmetry: Optional[str]

    - rotation: Optional[str]

    - translation: Optional[(dx,dy)] if input->output looks like pure translation

    - color: dict with histogram stats

    - n_components: int across all nonzero colors

    """
    desc: Dict[str, Any] = {}
    desc["symmetry"] = detect_symmetry(input_grid)
    desc["color"] = analyze_colors(input_grid)
    ncomp, _ = connected_components(input_grid)
    desc["n_components"] = int(ncomp)
    if output_grid is not None and input_grid.shape == output_grid.shape:
        # Try rotations
        for rot, k in [("rotate90",1),("rotate180",2),("rotate270",3)]:
            if _equal(torch.rot90(input_grid, k, [0,1]), output_grid):
                desc["rotation"] = rot
                break
        else:
            desc["rotation"] = None
        # Try translation
        desc["translation"] = detect_translation(input_grid, output_grid)
    else:
        desc["rotation"] = None
        desc["translation"] = None
    return desc
