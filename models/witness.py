"""
Witness Graph Extraction for ARC

Extracts objects, symmetries, connectivity, and invariants from grids
to seed hypothesis markets and guide program synthesis.

References:
- Object-centric priors for visual reasoning (Spelke, Kinzler 2007)
- Structural priors in program synthesis (Lake et al. 2015, ARC Prize)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
import torch

# ARC uses small integer palettes; background is 0
BACKGROUND = 0

# D4 dihedral group transformations (4 rotations × 2 reflections)
D4_TRANSFORMS = [
    lambda x: x,                          # identity
    lambda x: torch.rot90(x, 1, (-2, -1)),  # 90° CCW
    lambda x: torch.rot90(x, 2, (-2, -1)),  # 180°
    lambda x: torch.rot90(x, 3, (-2, -1)),  # 270° CCW
    lambda x: torch.flip(x, (-1, )),      # mirror horizontal
    lambda x: torch.flip(x, (-2, )),      # mirror vertical
    lambda x: torch.rot90(torch.flip(x, (-1,)), 1, (-2, -1)),  # mirror + rot90
    lambda x: torch.rot90(torch.flip(x, (-1,)), 3, (-2, -1)),  # mirror + rot270
]


@dataclass
class WitnessNode:
    """Object node in witness graph."""
    node_id: int
    color: int
    bbox: Tuple[int, int, int, int]  # (r0, c0, r1, c1) inclusive
    area: int
    centroid: Tuple[float, float]
    sym: Dict[str, bool]  # D4 symmetries
    feats: Dict[str, float] = field(default_factory=dict)


@dataclass
class WitnessEdge:
    """Relational edge in witness graph."""
    src: int
    dst: int
    rel: str  # "connectivity" | "same_color" | "symmetry"
    weight: float = 1.0


@dataclass
class WitnessSet:
    """Complete witness extracted from a grid."""
    nodes: List[WitnessNode] = field(default_factory=list)
    edges: List[WitnessEdge] = field(default_factory=list)
    invariants: Dict[str, Any] = field(default_factory=dict)

    def node_index_by_color(self) -> Dict[int, List[int]]:
        """Index nodes by color for fast lookup."""
        out: Dict[int, List[int]] = {}
        for n in self.nodes:
            out.setdefault(n.color, []).append(n.node_id)
        return out


# ══════════════════════════════════════════════════════════════════════
# Object Extraction and Features
# ══════════════════════════════════════════════════════════════════════

def _bbox_from_points(idx: torch.Tensor) -> Tuple[int, int, int, int]:
    """Compute bounding box from point coordinates."""
    r0, c0 = int(idx[:, 0].min()), int(idx[:, 1].min())
    r1, c1 = int(idx[:, 0].max()), int(idx[:, 1].max())
    return r0, c0, r1, c1


def _centroid(idx: torch.Tensor) -> Tuple[float, float]:
    """Compute centroid of point cloud."""
    return float(idx[:, 0].float().mean()), float(idx[:, 1].float().mean())


def _binary_components(mask: torch.Tensor) -> List[torch.Tensor]:
    """
    4-connected component labeling via iterative DFS.

    Args:
        mask: [H, W] or [1, H, W] bool tensor

    Returns:
        List of component coordinate tensors [N, 2]
    """
    # Handle batch dimension
    if mask.dim() == 3:
        mask = mask.squeeze(0)

    H, W = mask.shape
    vis = torch.zeros_like(mask, dtype=torch.bool)
    comps: List[torch.Tensor] = []

    for i in range(H):
        for j in range(W):
            if mask[i, j] and not vis[i, j]:
                stack = [(i, j)]
                acc = []
                while stack:
                    r, c = stack.pop()
                    if r < 0 or r >= H or c < 0 or c >= W:
                        continue
                    if vis[r, c] or not mask[r, c]:
                        continue
                    vis[r, c] = True
                    acc.append((r, c))
                    # 4-connectivity
                    stack.extend([(r+1, c), (r-1, c), (r, c+1), (r, c-1)])

                comps.append(torch.tensor(acc, dtype=torch.long, device=mask.device))

    return comps


def extract_objects(grid: torch.Tensor) -> List[Dict[str, Any]]:
    """
    Extract connected objects from grid.

    Args:
        grid: [H, W] or [1, H, W] integer tensor

    Returns:
        List of dicts with keys:
            'color': int
            'idx': LongTensor[N, 2] of (row, col) coordinates
            'bbox': (r0, c0, r1, c1)
    """
    if grid.dim() == 3:
        grid = grid.squeeze(0)

    objs: List[Dict[str, Any]] = []
    colors = torch.unique(grid)
    colors = colors[colors != BACKGROUND]

    for c in colors.tolist():
        mask = (grid == c)
        for idx in _binary_components(mask):
            if len(idx) == 0:
                continue
            r0, c0, r1, c1 = _bbox_from_points(idx)
            objs.append({
                'color': c,
                'idx': idx,
                'bbox': (r0, c0, r1, c1)
            })

    return objs


# ══════════════════════════════════════════════════════════════════════
# Symmetry (D4) Detection
# ══════════════════════════════════════════════════════════════════════

def _crop_tensor(grid: torch.Tensor, bbox: Tuple[int, int, int, int]) -> torch.Tensor:
    """Crop grid to bounding box."""
    r0, c0, r1, c1 = bbox
    return grid[r0:r1+1, c0:c1+1]


def detect_symmetries(grid: torch.Tensor, obj: Dict[str, Any]) -> Dict[str, bool]:
    """
    Detect D4 symmetries for an object.

    Args:
        grid: Full grid tensor
        obj: Object dict with 'bbox' and 'color'

    Returns:
        Dict mapping symmetry name -> bool
    """
    crop = _crop_tensor(grid, obj['bbox'])
    # Re-bin to binary mask to avoid palette confounds
    mask = (crop == obj['color']).to(grid.dtype)

    def eq(a, b):
        return (a.shape == b.shape) and torch.equal(a, b)

    syms = {
        'rot90': False,
        'rot180': False,
        'rot270': False,
        'flip_h': False,
        'flip_v': False,
        'flip_h_rot90': False,
        'flip_h_rot270': False
    }

    base = mask
    T = D4_TRANSFORMS

    syms['rot90'] = eq(T[1](base), base)
    syms['rot180'] = eq(T[2](base), base)
    syms['rot270'] = eq(T[3](base), base)
    syms['flip_h'] = eq(T[4](base), base)
    syms['flip_v'] = eq(T[5](base), base)
    syms['flip_h_rot90'] = eq(T[6](base), base)
    syms['flip_h_rot270'] = eq(T[7](base), base)

    return syms


# ══════════════════════════════════════════════════════════════════════
# Color Histogram and Graph Connectivity
# ══════════════════════════════════════════════════════════════════════

def color_hist(grid: torch.Tensor) -> Dict[int, int]:
    """Compute color histogram excluding background."""
    vals, counts = torch.unique(grid, return_counts=True)
    return {
        int(v.item()): int(c.item())
        for v, c in zip(vals, counts)
        if int(v.item()) != BACKGROUND
    }


def _bbox_adjacent(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> bool:
    """Check if two bounding boxes are adjacent (1-pixel dilated overlap)."""
    (r0, c0, r1, c1), (u0, v0, u1, v1) = b1, b2
    # Expand each bbox by 1 and test overlap
    r0e, c0e, r1e, c1e = r0-1, c0-1, r1+1, c1+1
    return not (r1e < u0 or u1 < r0e or c1e < v0 or v1 < c0e)


def connectivity(objs: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """
    Compute adjacency edges between objects.

    Args:
        objs: List of object dicts with 'bbox'

    Returns:
        List of (i, j) edge tuples
    """
    edges = []
    for i in range(len(objs)):
        for j in range(i+1, len(objs)):
            if _bbox_adjacent(objs[i]['bbox'], objs[j]['bbox']):
                edges.append((i, j))
    return edges


# ══════════════════════════════════════════════════════════════════════
# Invariants Across Demos
# ══════════════════════════════════════════════════════════════════════

def _components_count_all(grid: torch.Tensor) -> int:
    """Count total connected components in grid."""
    # Handle batch dimension (NeuroPlanner provides [1,H,W])
    if grid.dim() == 3:
        grid = grid.squeeze(0)

    mask = (grid != BACKGROUND)
    return len(_binary_components(mask))


def infer_invariants(train_pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, Any]:
    """
    Mine invariants that hold across all training pairs.

    Args:
        train_pairs: List of (input, output) demonstration pairs

    Returns:
        Dict of invariant name -> value
    """
    if not train_pairs:
        return {}

    # Normalize demo format (handles dicts/3-tuples) - use ensure_demo_pairs for robustness
    from trainers.demo_utils import ensure_demo_pairs
    train_pairs = ensure_demo_pairs(train_pairs)

    inv: Dict[str, Any] = {}

    # Palette subset/equality (use indexing to avoid tuple unpacking errors)
    pal_in = [set(torch.unique(pair[0]).tolist()) for pair in train_pairs]
    pal_out = [set(torch.unique(pair[1]).tolist()) for pair in train_pairs]

    if all(b.issubset(a) for a, b in zip(pal_in, pal_out)):
        inv['palette_subset'] = True

    # Component count constant?
    comp_in = [_components_count_all(pair[0]) for pair in train_pairs]
    comp_out = [_components_count_all(pair[1]) for pair in train_pairs]

    if comp_in and comp_out and all(ci == co for ci, co in zip(comp_in, comp_out)):
        inv['component_count'] = comp_out[0]

    # Shape preservation?
    shapes_in = [tuple(pair[0].shape[-2:]) for pair in train_pairs]
    shapes_out = [tuple(pair[1].shape[-2:]) for pair in train_pairs]

    if all(si == so for si, so in zip(shapes_in, shapes_out)):
        inv['shape_preserved'] = True

    # Mass conservation (non-zero pixel count)
    mass_in = [(pair[0] != 0).sum().item() for pair in train_pairs]
    mass_out = [(pair[1] != 0).sum().item() for pair in train_pairs]

    if all(mi == mo for mi, mo in zip(mass_in, mass_out)):
        inv['mass_conserved'] = mass_out[0] if mass_out else 0

    # Grid scaling (upscaling/downscaling)
    if shapes_in and shapes_out:
        scale_h = [so[0] / max(1, si[0]) for si, so in zip(shapes_in, shapes_out)]
        scale_w = [so[1] / max(1, si[1]) for si, so in zip(shapes_in, shapes_out)]

        if len(set(zip(scale_h, scale_w))) == 1:  # All same scale factor
            sh, sw = scale_h[0], scale_w[0]
            if sh > 1.0 or sw > 1.0:
                inv['upscaling'] = (sh, sw)
            elif sh < 1.0 or sw < 1.0:
                inv['downscaling'] = (sh, sw)

    # Background preservation (background pixel ratio)
    bg_in = [(pair[0] == 0).float().mean().item() for pair in train_pairs]
    bg_out = [(pair[1] == 0).float().mean().item() for pair in train_pairs]

    if all(abs(bi - bo) < 0.1 for bi, bo in zip(bg_in, bg_out)):
        inv['background_constant'] = sum(bg_out) / len(bg_out) if bg_out else 0.0

    # Color permutation (simple check - output uses different palette)
    palette_changed = not all(b == a for a, b in zip(pal_in, pal_out))
    all_output_same = len(set(frozenset(p) for p in pal_out)) == 1

    if palette_changed and all_output_same:
        inv['color_transform'] = list(pal_out[0]) if pal_out else []

    return inv


# ══════════════════════════════════════════════════════════════════════
# Main Entry: Build WitnessSet from Grid
# ══════════════════════════════════════════════════════════════════════

def build_witness(
    grid: torch.Tensor,
    demos: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
) -> WitnessSet:
    """
    Build complete witness graph from a grid.

    Args:
        grid: [H, W] or [1, H, W] grid tensor
        demos: Optional demonstration pairs for invariant mining

    Returns:
        WitnessSet with nodes, edges, and invariants
    """
    if grid.dim() == 3:
        grid = grid.squeeze(0)

    # Defensive normalization of demo formats (handles dicts/3-tuples)
    if demos is not None and len(demos) > 0:
        try:
            from trainers.demo_utils import ensure_demo_pairs
            demos = ensure_demo_pairs(demos)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"[Witness] Demo normalization failed: {e}")
            demos = []

    objs = extract_objects(grid)
    nodes: List[WitnessNode] = []

    for k, o in enumerate(objs):
        idx = o['idx']
        area = int(idx.shape[0])
        r0, c0, r1, c1 = o['bbox']
        cen = _centroid(idx)
        syms = detect_symmetries(grid, o)

        feats = {
            'bbox_h': float(r1 - r0 + 1),
            'bbox_w': float(c1 - c0 + 1),
            'area': float(area),
            'centroid_r': float(cen[0]),
            'centroid_c': float(cen[1]),
            'sym_rot90': float(syms['rot90']),
            'sym_rot180': float(syms['rot180']),
            'sym_rot270': float(syms['rot270']),
            'sym_flip_h': float(syms['flip_h']),
            'sym_flip_v': float(syms['flip_v'])
        }

        nodes.append(WitnessNode(
            node_id=k,
            color=o['color'],
            bbox=o['bbox'],
            area=area,
            centroid=cen,
            sym=syms,
            feats=feats
        ))

    # Build edges
    undirected = connectivity(objs)
    edges = [WitnessEdge(i, j, 'connectivity', 1.0) for (i, j) in undirected]

    # Add same-color edges (weaker signal)
    for i in range(len(objs)):
        for j in range(i+1, len(objs)):
            if objs[i]['color'] == objs[j]['color']:
                edges.append(WitnessEdge(i, j, 'color', 0.5))

    # Mine invariants if demos provided (with error handling)
    try:
        inv = infer_invariants(demos or [])
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"[Witness] Invariant inference failed: {e}")
        inv = {}

    return WitnessSet(nodes=nodes, edges=edges, invariants=inv)
